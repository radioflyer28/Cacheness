"""
BlobStore - Low-level Blob Storage Interface
============================================

A simpler, lower-level API for storing and retrieving binary blobs with metadata.
This class abstracts away the caching semantics (TTL, eviction) and provides
pure storage functionality.

The BlobStore is designed to be reusable for non-caching use cases:
- ML model versioning
- Artifact storage  
- Data pipeline checkpoints

Usage:
    from cacheness.storage import BlobStore
    
    # Create a blob store
    store = BlobStore(
        cache_dir="./blobs",
        backend="sqlite",
        compression="lz4"
    )
    
    # Store data
    blob_id = store.put(my_data, metadata={"type": "model", "version": "1.0"})
    
    # Retrieve data
    data = store.get(blob_id)
    
    # Get metadata only (without loading blob)
    metadata = store.get_metadata(blob_id)
    
    # List blobs
    blob_ids = store.list(prefix="model_")
    
    # Delete
    store.delete(blob_id)
"""

import hashlib
import logging
import uuid
from copy import deepcopy
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ..error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobManifestMalformedError,
    CacheBlobManifestUnauthenticatedError,
    CacheBlobManifestUnsupportedVersionError,
    CacheBlobPayloadMissingError,
    CacheBlobPayloadTamperedError,
    CacheManifestIntegrityError,
    CacheBlobPayloadUnsupportedVersionError,
    CacheManifestUnsupportedVersionError,
    CacheReason,
    CacheStorageError,
)
from .backends import MetadataBackend, JsonBackend
from .clear_recovery import ClearRecoveryCoordinator, LegacyClearEvidenceAdapter
from .coordination import InstanceAdmission, KeyCoordinatorRegistry, StoreAdmissionBarrier
from .guarded_handler_io import GuardedHandlerIO
from .handlers import HandlerRegistry
from .integrity import (
    ManifestKeyError,
    ManifestKeyProvider,
    ManifestSigningKeyProvider,
    sha256_and_size,
    sign_hmac_sha256,
    verify_hmac_sha256,
)
from .manifest import (
    BlobManifestV1,
    ManifestDecodeError,
    canonical_signing_bytes_from_record,
    decode_canonical_manifest_record,
)
from .manifest_repository import (
    ManifestCursor,
    ManifestExpectation,
    create_manifest_repository,
)
from .legacy_manifest import LegacyManifestIdentity, recognize_legacy_fixture_tree
from .lifecycle import AuthorityLifecycleEngine
from .lifecycle_authority import EntryExpectation, LifecycleAuthority
from .sqlite_lifecycle_authority import SqliteLifecycleAuthority
from .path_security import encode_physical_name, resolve_managed_locator
from .reconciliation import _Reconciler, ReconciliationReport
from ..metadata import InMemoryBackend, MetadataBackend as CoreMetadataBackend
from ..metadata import SqliteBackend

# Import CacheConfig for proper handler configuration
from ..config import CacheConfig, CompressionConfig

logger = logging.getLogger(__name__)


_IMMUTABLE_METADATA_PATCH_FIELDS = frozenset(
    {
        "schema_version",
        "key",
        "cache_key",
        "generation",
        "state",
        "locator",
        "actual_path",
        "handler_type",
        "data_type",
        "payload_format",
        "payload_format_version",
        "storage_format",
        "digest_algorithm",
        "digest",
        "byte_size",
        "file_size",
        "created_at",
        "handler_metadata",
        "user_metadata",
        "signature_algorithm",
        "signature",
    }
)


def _ordinary_admitted(method: Callable) -> Callable:
    """Admit ordinary BlobStore work outside a finite clear snapshot boundary."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        # Legacy fixtures must fail before an admission sidecar can mutate the
        # exact evidence tree an operator is being asked to migrate.
        self._require_canonical_store()
        with self._instance_admission.operation():
            if self._authority_mode:
                return method(self, *args, **kwargs)
            with self._admission_barrier.ordinary_admission():
                self._refresh_metadata_view_for_lifecycle()
                return method(self, *args, **kwargs)

    return wrapped


def _raise_translated_recovery_failure(
    coordinator: ClearRecoveryCoordinator, error: CacheStorageError
) -> None:
    """Map tagged recovery-admission failures into the direct BlobStore API."""
    if isinstance(error, (CacheBlobBackendError, CacheBlobLifecycleConflictError)):
        raise error

    if not ClearRecoveryCoordinator.is_lifecycle_conflict(error) and (
        "clear_recovery_failure" not in error.context
    ):
        raise error

    if ClearRecoveryCoordinator.is_lifecycle_conflict(error):
        translated_type = CacheBlobLifecycleConflictError
    else:
        translated_type = CacheBlobBackendError
    raise translated_type(
        str(error),
        context={
            **error.context,
            "backend": error.context.get("backend", coordinator.kind),
        },
    ) from error


class BlobStore:
    """
    Low-level blob storage with metadata support.
    
    Provides a simple put/get interface for storing arbitrary Python objects
    with associated metadata. Unlike the higher-level UnifiedCache, BlobStore
    does not implement caching semantics like TTL or eviction policies.
    
    Features:
    - Content-addressable storage option (hash-based keys)
    - Pluggable metadata backends (JSON, SQLite)
    - Type-aware serialization via handlers
    - Configurable compression
    - Rich, queryable metadata
    
    Attributes:
        cache_dir: Root directory for blob storage
        backend: Metadata backend instance
        handlers: Handler registry for type detection
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path] = ".blobstore",
        backend: Optional[Union[str, MetadataBackend]] = None,
        compression: str = "lz4",
        compression_level: int = 3,
        content_addressable: bool = False,
        *,
        config: CacheConfig | None = None,
        manifest_key_provider: ManifestSigningKeyProvider | None = None,
        lifecycle_authority: LifecycleAuthority | None = None,
    ):
        """
        Initialize a BlobStore.
        
        Args:
            cache_dir: Directory for storing blobs and metadata
            backend: Metadata backend - "json", "sqlite", or a MetadataBackend instance
            compression: Compression codec (lz4, zstd, gzip, blosclz, etc.)
            compression_level: Compression level (1-9)
            content_addressable: If True, use content hash as blob key
            config: Optional caller-owned configuration for lifecycle policy
        """
        configured_path = (
            config.storage.cache_dir
            if config is not None and cache_dir == ".blobstore"
            else cache_dir
        )
        self.cache_dir = Path(configured_path)
        self._authority_mode = True
        self._owns_lifecycle_authority = lifecycle_authority is None
        self.lifecycle_authority = lifecycle_authority
        self.guarded_handler_io = (
            GuardedHandlerIO(self.cache_dir) if self.cache_dir.is_dir() else None
        )
        self._owns_backend = False
        self._released_resources = {
            "authority": False,
            "manifest_repository": False,
            "guarded_handler_io": False,
            "backend": False,
            "admission_barrier": False,
        }
        try:
            self._initialize_after_guarded_io(
                backend,
                compression,
                compression_level,
            content_addressable,
            config,
            manifest_key_provider,
            lifecycle_authority,
            )
        except BaseException:
            # Cancellation must not retain a partially initialized owner and
            # its managed root descriptor or internally created backend.
            self._close_failed_initialization_resources()
            raise

        logger.debug(f"BlobStore initialized at {self.cache_dir}")

    def _initialize_after_guarded_io(
        self,
        backend: Optional[Union[str, MetadataBackend]],
        compression: str,
        compression_level: int,
        content_addressable: bool,
        config: CacheConfig | None,
        manifest_key_provider: ManifestSigningKeyProvider | None,
        lifecycle_authority: LifecycleAuthority | None,
    ) -> None:
        """Finish initialization after the managed-root descriptor is acquired."""
        self._legacy_identity: LegacyManifestIdentity | None = None
        # A compatibility fixture is recognized only when it presents the
        # fixed provenance filename.  This is not a directory scan and runs
        # before metadata/key initialization so inspection cannot create a
        # sidecar, rotate a key, or convert the historical store.
        if (self.cache_dir / "provenance.json").is_file():
            self._legacy_identity = recognize_legacy_fixture_tree(self.cache_dir)
        
        self.compression = compression
        self.compression_level = compression_level
        self.content_addressable = content_addressable
        
        # Keep an explicitly supplied config object intact. The flat path keeps
        # its historical handler-compression defaults and constructs one local
        # configuration only when no caller-owned configuration was supplied.
        self.config = (
            CacheConfig(
                cache_dir=self.cache_dir,
                compression=CompressionConfig(
                    pickle_compression_codec=compression,
                    pickle_compression_level=compression_level,
                    blosc2_array_clevel=compression_level,
                ),
            )
            if config is None
            else config
        )
        self.lifecycle_limits = self.config.lifecycle_limits
        self._instance_admission = InstanceAdmission(self.lifecycle_limits)
        self._immutable_metadata_patch_fields = _IMMUTABLE_METADATA_PATCH_FIELDS
        if lifecycle_authority is None:
            lifecycle_authority = SqliteLifecycleAuthority.for_root(
                self.cache_dir,
                lifecycle_limits=self.lifecycle_limits,
                lifecycle_topology=getattr(
                    self.config,
                    "lifecycle_topology",
                    None,
                ),
            )
        self.lifecycle_authority = lifecycle_authority
        if lifecycle_authority is not None:
            self._admission_barrier = None
            self._key_coordinator = KeyCoordinatorRegistry()
            if type(backend) in {InMemoryBackend, JsonBackend, SqliteBackend}:
                self.backend = backend
            elif backend is None or (
                isinstance(backend, str) and backend in {"json", "sqlite"}
            ):
                self.backend = InMemoryBackend()
                self._owns_backend = True
            else:
                raise CacheBlobBackendError(
                    "BlobStore metadata backend is unsupported by the authority lifecycle",
                    context={"backend_type": type(backend).__name__},
                )
            self.handlers = HandlerRegistry()
            self._manifest_key_provider = (
                ManifestKeyProvider(
                    self.cache_dir / "blob_manifest_hmac_key.bin",
                    lifecycle_limits=self.lifecycle_limits,
                )
                if manifest_key_provider is None
                else manifest_key_provider
            )
            self.manifest_repository = None
            self.lifecycle = AuthorityLifecycleEngine(self, lifecycle_authority)
            self._reconciler = None
            self._clear_recovery = None
            self._legacy_clear_evidence = None
            self._authority_lifecycle = self.lifecycle
            return
        self._authority_lifecycle = None
        # Exact legacy fixtures are read-only migration evidence. They reject
        # every public operation before admission, so creating a lifecycle lock
        # for them would itself mutate the fixture merely by opening it.
        self._admission_barrier = (
            None
            if self._legacy_identity is not None
            else StoreAdmissionBarrier.acquire(self.guarded_handler_io.root)
        )
        # This is intentionally per instance.  Same-process independent
        # stores still exercise the manifest repository's CAS authority.
        self._key_coordinator = KeyCoordinatorRegistry()
        
        # Initialize metadata backend
        if self._legacy_identity is not None:
            self.backend = InMemoryBackend()
            self._owns_backend = True
        elif backend is None or backend == "json":
            self.backend = JsonBackend(self.cache_dir / "cache_metadata.json")
            self._owns_backend = True
        elif backend == "sqlite":
            from .backends import SqliteBackend as LegacySqliteBackend
            self.backend = LegacySqliteBackend(self.cache_dir / "cache_metadata.db")
            self._owns_backend = True
        elif isinstance(backend, (MetadataBackend, CoreMetadataBackend)):
            self.backend = backend
        else:
            raise ValueError(f"Unknown backend type: {backend}")
        
        # Initialize handler registry
        self.handlers = HandlerRegistry()
        self._manifest_key_provider = (
            ManifestKeyProvider(
                self.cache_dir / "blob_manifest_hmac_key.bin",
                lifecycle_limits=self.lifecycle_limits,
            )
            if manifest_key_provider is None
            else manifest_key_provider
        )
        self.manifest_repository = create_manifest_repository(
            self.backend,
            lifecycle_limits=self.lifecycle_limits,
            file_ops=self.guarded_handler_io.file_ops,
            # The scheduler is control authority: it must be signed by the
            # same persistent store trust root as the canonical manifests,
            # rather than by a public or independently replaceable key.
            inventory_key_provider=self._manifest_key,
        )
        self.lifecycle = AuthorityLifecycleEngine(self, self.lifecycle_authority)
        self._reconciler = _Reconciler(self, lifecycle_limits=self.lifecycle_limits)

        # The predecessor coordinator remains an exact local compatibility
        # parser. It has no new-operation call site: fresh clear authority is
        # owned by LifecycleEngine and its signed operation records.
        self._clear_recovery = None
        self._legacy_clear_evidence = None
        if (
            self._legacy_identity is None
            and ClearRecoveryCoordinator.can_coordinate(self.backend)
        ):
            self._clear_recovery = ClearRecoveryCoordinator(
                self.guarded_handler_io.file_ops,
                self.backend,
            )
            self._legacy_clear_evidence = LegacyClearEvidenceAdapter(
                self.guarded_handler_io.file_ops,
                self.backend,
            )
            try:
                if self._legacy_clear_evidence.has_evidence():
                    with self._clear_recovery.admission():
                        self._legacy_clear_evidence.recover()
                        self._reconcile_sqlite_manifest_records_after_clear()
            except CacheStorageError as exc:
                _raise_translated_recovery_failure(self._clear_recovery, exc)

    def _close_failed_initialization_resources(self) -> None:
        """Release only resources this incomplete store has taken ownership of."""
        if self._owns_lifecycle_authority and self.lifecycle_authority is not None:
            try:
                self.lifecycle_authority.close()
                self._released_resources["authority"] = True
            except Exception:
                logger.exception("Failed to close internally created lifecycle authority")
        lifecycle = getattr(self, "lifecycle", None)
        operation_repository = getattr(lifecycle, "operation_repository", None)
        operation_close = getattr(operation_repository, "close", None)
        if callable(operation_close):
            try:
                operation_close()
            except Exception:
                logger.exception("Failed to close BlobStore lifecycle operation locks")
        repository_close = getattr(
            getattr(self, "manifest_repository", None), "close", None
        )
        if callable(repository_close):
            try:
                repository_close()
            except Exception:
                logger.exception("Failed to close BlobStore manifest repository lock root")
        if self._owns_backend and hasattr(self, "backend"):
            try:
                self.backend.close()
            except Exception:
                logger.exception("Failed to close internally created BlobStore backend")
        if self.guarded_handler_io is not None:
            try:
                self.guarded_handler_io.close()
            except Exception:
                logger.exception("Failed to close BlobStore managed-root descriptor")
        if getattr(self, "_admission_barrier", None) is not None:
            try:
                self._admission_barrier.release()
            except Exception:
                logger.exception("Failed to release BlobStore admission barrier")

    @property
    def legacy_identity(self) -> LegacyManifestIdentity | None:
        """Expose the attached compatibility identity without persisting it."""
        return self._legacy_identity

    @staticmethod
    def inspect_legacy_fixture_tree(root: str | Path) -> LegacyManifestIdentity:
        """Return an in-memory identity for exact read-only legacy evidence.

        Canonical storage never invokes this compatibility adapter as a
        fallback.  Callers may use it to inspect one known historical store
        before an explicit Phase 7 migration.
        """
        return recognize_legacy_fixture_tree(root)
    
    @_ordinary_admitted
    def put(
        self,
        data: Any,
        key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a blob with optional metadata.
        
        Args:
            data: The data to store (any Python object)
            key: Optional key for the blob. If None, generates a unique key.
                 If content_addressable=True, key is ignored and content hash is used.
            metadata: Optional dictionary of metadata to store with the blob
            
        Returns:
            The blob key (can be used to retrieve the blob)
        """
        self._require_canonical_store()
        # Generate key
        if self.content_addressable:
            # Use content hash as key
            blob_key = self._compute_content_hash(data)
        elif key is not None:
            blob_key = key
        else:
            blob_key = self._generate_unique_key()

        with self._key_coordinator.hold(self._storage_id_for_key(blob_key)):
            stored_key = (
                self._authority_lifecycle.put(data, key=blob_key, metadata=metadata)
                if self._authority_lifecycle is not None
                else self.lifecycle.put(data, key=blob_key, metadata=metadata)
            )
        logger.debug(f"Stored blob {stored_key} through the lifecycle engine")
        return stored_key
    
    @_ordinary_admitted
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a blob by key.
        
        Args:
            key: The blob key
            
        Returns:
            The stored data, or None if not found
        """
        self._require_canonical_store()
        if self._authority_lifecycle is not None:
            return self._authority_lifecycle.get(key)
        with self._key_coordinator.hold(self._storage_id_for_key(key)):
            for attempt in range(2):
                authenticated = self._load_authenticated_manifest(
                    key,
                    operation="get",
                    require_payload_contract=True,
                    require_locator=True,
                )
                if authenticated is None:
                    logger.debug(f"Blob not found: {key}")
                    return None
                manifest, handler, actual_path = authenticated
                assert handler is not None
                assert actual_path is not None
                handler_metadata = self._handler_metadata(manifest)

                # The resolved handler receives only this private snapshot. It
                # is reauthenticated before hashing or deserializing.  A newer
                # committed generation invalidates this snapshot, while an
                # unchanged generation preserves exact payload error classes.
                try:
                    with self.guarded_handler_io.open_snapshot(
                        actual_path, handler_metadata
                    ) as snapshot:
                        if not self._read_generation_is_stable(
                            key, manifest, attempt
                        ):
                            continue
                        digest, byte_size = sha256_and_size(snapshot.path)
                        if (
                            digest != manifest.digest
                            or byte_size != manifest.byte_size
                        ):
                            raise CacheBlobPayloadTamperedError(
                                "Canonical BlobStore payload integrity check failed"
                            )
                        return handler.get(snapshot.path, snapshot.metadata)
                except FileNotFoundError as exc:
                    if not self._read_generation_is_stable(key, manifest, attempt):
                        continue
                    raise CacheBlobPayloadMissingError(
                        "Canonical BlobStore payload is missing"
                    ) from exc
                except CacheBlobPayloadTamperedError:
                    raise
                except OSError as exc:
                    if not self._read_generation_is_stable(key, manifest, attempt):
                        continue
                    raise CacheBlobPayloadTamperedError(
                        "Canonical BlobStore payload could not be verified"
                    ) from exc

        raise CacheBlobLifecycleConflictError(
            "Canonical BlobStore read exhausted its bounded generation retry",
            context={"key": key, "operation": "get"},
        )
    
    @_ordinary_admitted
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get blob metadata without loading the blob content.
        
        Args:
            key: The blob key
            
        Returns:
            Metadata dictionary, or None if not found
        """
        self._require_canonical_store()
        if self._authority_lifecycle is not None:
            return self._authority_lifecycle.get_metadata(key)
        with self._key_coordinator.hold(self._storage_id_for_key(key)):
            authenticated = self._load_authenticated_manifest(
                key,
                operation="get_metadata",
                require_locator=True,
            )
            if authenticated is None:
                return None
            manifest, _handler, _locator = authenticated
            return self._manifest_entry_data(manifest)
    
    @_ordinary_admitted
    def update_metadata(self, key: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for an existing blob.
        
        Args:
            key: The blob key
            metadata: New metadata to merge with existing nested metadata
            
        Returns:
            True if successful, False if blob not found
        """
        self._require_canonical_store()
        if self._authority_lifecycle is not None:
            return self._authority_lifecycle.update_metadata(key, metadata)
        with self._key_coordinator.hold(self._storage_id_for_key(key)):
            if not isinstance(metadata, dict):
                raise CacheBlobManifestMalformedError(
                    "BlobStore metadata patches must be dictionaries"
                )
            immutable_fields = _IMMUTABLE_METADATA_PATCH_FIELDS.intersection(metadata)
            if immutable_fields:
                raise CacheBlobLifecycleConflictError(
                    "BlobStore metadata patches cannot change canonical structural fields",
                    context={"fields": sorted(immutable_fields)},
                )

            authenticated = self._load_authenticated_manifest_with_raw(
                key,
                operation="update_metadata",
                require_locator=True,
            )
            if authenticated is None:
                return False
            manifest, observed_record, _handler, _locator = authenticated
            expected = ManifestExpectation.from_authenticated_record(
                manifest.generation,
                observed_record,
            )
            user_metadata = {**dict(manifest.user_metadata), **metadata}
            updated_manifest = replace(manifest, user_metadata=user_metadata)
            signed_manifest = updated_manifest.with_signature(
                sign_hmac_sha256(
                    updated_manifest.signing_bytes(),
                    self._manifest_key(),
                )
            )
            self.manifest_repository.publish_if_expected(
                key,
                expected,
                signed_manifest.canonical_bytes(),
                entry_data=self._manifest_entry_data(signed_manifest),
            )
            return True
    
    @_ordinary_admitted
    def delete(self, key: str) -> bool:
        """
        Delete a blob and its metadata.
        
        Args:
            key: The blob key
            
        Returns:
            True if deleted, False if not found
        """
        self._require_canonical_store()
        with self._key_coordinator.hold(self._storage_id_for_key(key)):
            deleted = self._authority_lifecycle.delete(key=key)
        if deleted:
            logger.debug(f"Deleted blob {key} through the lifecycle engine")
        return deleted
    
    @_ordinary_admitted
    def exists(self, key: str) -> bool:
        """
        Check if a blob exists.
        
        Args:
            key: The blob key
            
        Returns:
            True if the blob exists
        """
        self._require_canonical_store()
        if self._authority_lifecycle is not None:
            return self._authority_lifecycle.exists(key)
        with self._key_coordinator.hold(self._storage_id_for_key(key)):
            for attempt in range(2):
                authenticated = self._load_authenticated_manifest(
                    key,
                    operation="exists",
                    require_payload_contract=True,
                    require_locator=True,
                )
                if authenticated is None:
                    return False
                manifest, _handler, actual_path = authenticated
                assert actual_path is not None
                try:
                    with self.guarded_handler_io.open_snapshot(
                        actual_path, {}
                    ) as snapshot:
                        if not self._read_generation_is_stable(
                            key, manifest, attempt, operation="exists"
                        ):
                            continue
                        digest, byte_size = sha256_and_size(snapshot.path)
                        if digest != manifest.digest or byte_size != manifest.byte_size:
                            raise CacheBlobPayloadTamperedError(
                                "Canonical BlobStore payload integrity check failed"
                            )
                        return True
                except FileNotFoundError as exc:
                    if not self._read_generation_is_stable(
                        key, manifest, attempt, operation="exists"
                    ):
                        continue
                    raise CacheBlobPayloadMissingError(
                        "Canonical BlobStore payload is missing"
                    ) from exc
                except CacheBlobPayloadTamperedError:
                    raise
                except OSError as exc:
                    if not self._read_generation_is_stable(
                        key, manifest, attempt, operation="exists"
                    ):
                        continue
                    raise CacheBlobPayloadTamperedError(
                        "Canonical BlobStore payload could not be verified"
                    ) from exc

        raise CacheBlobLifecycleConflictError(
            "Canonical BlobStore existence check exhausted its bounded generation retry",
            context={"key": key, "operation": "exists"},
        )
    
    @_ordinary_admitted
    def list(
        self,
        prefix: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        List blob keys with optional filtering.
        
        Args:
            prefix: Only return keys starting with this prefix
            metadata_filter: Filter by metadata field values (exact match)
            
        Returns:
            List of matching blob keys
        """
        self._require_canonical_store()
        if self._authority_lifecycle is not None:
            return self._authority_lifecycle.list(prefix, metadata_filter)
        entries = self._list_backend_entries(operation="list")
        keys = []

        cursor: ManifestCursor | None = None
        while True:
            page = self.manifest_repository.list_page(
                cursor, page_size=self.lifecycle_limits.manifest_page_size
            )
            for key, expected_raw in page.entries:
                if prefix and not key.startswith(prefix):
                    continue

                authenticated = self._load_authenticated_manifest(
                    key,
                    operation="list",
                    require_locator=True,
                )
                if authenticated is None:
                    raise CacheBlobLifecycleConflictError(
                        "Canonical BlobStore list authority disappeared during selection",
                        context={"key": key, "operation": "list"},
                    )
                manifest, _handler, _locator = authenticated
                observed_raw = self.manifest_repository.get_raw(key)
                if (
                    observed_raw != expected_raw
                    or manifest.canonical_bytes() != expected_raw
                ):
                    raise CacheBlobLifecycleConflictError(
                        "Canonical BlobStore list authority changed during selection",
                        context={"key": key, "operation": "list"},
                    )
                if metadata_filter:
                    metadata = self._manifest_entry_data(manifest)
                    searchable_metadata = {
                        **dict(manifest.user_metadata),
                        **dict(manifest.handler_metadata),
                        **metadata,
                    }
                    if any(
                        searchable_metadata.get(field) != value
                        for field, value in metadata_filter.items()
                    ):
                        continue
                keys.append(key)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor

        # Legacy projections are never returned or used as authoritative
        # metadata, but their locators remain a containment tripwire for this
        # long-standing all-or-error public operation.
        self._preflight_entries(entries, operation="list")
        return keys
    
    def clear(self) -> int:
        """
        Remove every managed payload with recoverable clear tombstones.

        Returns:
            Number of blobs removed
        """
        with self._instance_admission.operation():
            self._require_canonical_store()
            try:
                cleared = self._authority_lifecycle.clear()
            except (CacheBlobBackendError, CacheBlobLifecycleConflictError):
                raise
            except (CacheStorageError, OSError) as exc:
                raise CacheBlobBackendError(
                    "BlobStore clear lifecycle could not complete",
                    context={"operation": "clear"},
                ) from exc
        logger.debug("Cleared %s BlobStore targets through the lifecycle engine", cleared)
        return cleared

    def reconcile(
        self,
        *,
        apply: bool = False,
        resume_token: str | None = None,
        now: Any | None = None,
    ) -> ReconciliationReport:
        """Report bounded lifecycle debt without deserializing or scanning payloads.

        Reconciliation defaults to a non-mutating inspection.  The explicit
        apply path is intentionally implemented by the private reconciler so
        it can revalidate exact evidence immediately before every action.
        """
        with self._instance_admission.operation():
            self._require_canonical_store()
            if self._authority_lifecycle is not None:
                reclaimed = self._authority_lifecycle.reconcile() if apply else 0
                return ReconciliationReport(
                    findings=(),
                    resume_token=None,
                    applied=bool(reclaimed),
                    manifest_records_seen=0,
                    operation_records_seen=reclaimed,
                )
            return self._reconciler.reconcile(
                apply=apply,
                resume_token=resume_token,
                now=now,
            )

    def _reconcile_sqlite_manifest_records_after_clear(self) -> None:
        """Drop SQLite sidecar records orphaned by a completed clear recovery.

        The reopen-only predecessor adapter can leave stale SQLite canonical
        sidecars after it establishes legacy ``cache_entries`` authority.
        Remove them only after that adapter has reached a terminal outcome.
        """
        if type(self.backend) is not SqliteBackend:
            return
        active_keys = {
            entry["cache_key"]
            for entry in self.backend.list_entries()
            if isinstance(entry, dict) and isinstance(entry.get("cache_key"), str)
        }
        for key in self.manifest_repository.list_keys():
            if key not in active_keys:
                self.manifest_repository.remove(key)

    def _stage_clear_payloads(self, entries: List[Dict[str, Any]]) -> List[tuple[Path, Path]]:
        """Copy each live payload to a private tombstone before deleting it."""
        staged_payloads: List[tuple[Path, Path]] = []
        for entry in entries:
            logical_key = entry.get("cache_key", "")
            actual_path = self._entry_locator(
                entry,
                logical_key,
                operation="clear",
            )
            if not self.guarded_handler_io.file_ops.exists(actual_path):
                continue

            tombstone = self.guarded_handler_io.root / (
                f"clear-tombstone-{uuid.uuid4().hex}"
            )
            tombstone_registered = False
            try:
                with self.guarded_handler_io.file_ops.open_read(actual_path) as source:
                    self.guarded_handler_io.file_ops.write_stream_to_locator(
                        tombstone,
                        source,
                    )
                staged_payloads.append((actual_path, tombstone))
                tombstone_registered = True
                if not self.guarded_handler_io.file_ops.delete(actual_path):
                    raise FileNotFoundError(
                        "Blob payload disappeared while staging clear reconciliation"
                    )
            except Exception:
                cleanup_error = None
                try:
                    if (
                        not tombstone_registered
                        and self.guarded_handler_io.file_ops.exists(tombstone)
                    ):
                        self.guarded_handler_io.file_ops.delete(tombstone)
                except Exception as exc:
                    logger.exception("Failed to remove incomplete BlobStore clear tombstone")
                    cleanup_error = exc
                try:
                    self._restore_clear_tombstones(staged_payloads)
                except Exception as exc:
                    logger.exception("Failed to roll back staged BlobStore clear payloads")
                    cleanup_error = exc
                if cleanup_error is not None:
                    raise RuntimeError(
                        "BlobStore clear staging failed and recovery was incomplete"
                    ) from cleanup_error
                raise

        return staged_payloads

    def _restore_clear_tombstones(
        self, staged_payloads: List[tuple[Path, Path]]
    ) -> None:
        """Restore metadata-referenced payloads after an uncommitted clear fails."""
        recovery_error = None
        for actual_path, tombstone in reversed(staged_payloads):
            try:
                with self.guarded_handler_io.file_ops.open_read(tombstone) as source:
                    self.guarded_handler_io.file_ops.write_stream_to_locator(
                        actual_path,
                        source,
                    )
                self.guarded_handler_io.file_ops.delete(tombstone)
            except Exception as exc:
                logger.exception("Failed to restore a staged BlobStore clear payload")
                recovery_error = exc

        if recovery_error is not None:
            raise RuntimeError("BlobStore clear rollback could not restore every payload") from recovery_error

    def _snapshot_clear_metadata(
        self, entries: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Capture exact backend records before a clear can mutate them."""
        snapshot = {}
        for entry in entries:
            cache_key = entry.get("cache_key")
            stored_entry = self.backend.get_entry(cache_key)
            if stored_entry is not None:
                snapshot[cache_key] = deepcopy(stored_entry)
        return snapshot

    def _restore_clear_metadata(
        self, metadata_snapshot: Dict[str, Dict[str, Any]]
    ) -> None:
        """Restore missing or changed entries after a failed metadata clear."""
        for cache_key, entry in metadata_snapshot.items():
            if self.backend.get_entry(cache_key) != entry:
                self.backend.put_entry(cache_key, deepcopy(entry))
    
    def close(self) -> None:
        """Drain admitted work and release only resources owned by this store.

        Close changes admission to ``CLOSING`` before it waits.  A drain timeout
        deliberately preserves the live resources and closing state, making a
        later call the only route to complete release.  Neither this method nor
        its retry path performs lifecycle deletion or reconciliation.
        """
        should_release = self._instance_admission.begin_close()
        if not should_release:
            return

        closed = False
        try:
            self._release_owned_resources()
            closed = True
        except CacheStorageError:
            raise
        except CacheBlobLifecycleTimeoutError:
            # Bounded first-key admission is a lifecycle outcome, not an
            # unauthenticated-signing-key outcome.  Callers can retry it
            # without reading or retiring a live winner's key bytes.
            raise
        except Exception as exc:
            raise CacheBlobBackendError(
                "BlobStore close could not release an owned resource",
                context={"operation": "close"},
            ) from exc
        finally:
            self._instance_admission.finish_close(closed=closed)

    def _release_owned_resources(self) -> None:
        """Release successful owned resources once, leaving failed work retryable.

        The lifecycle operation repository retains bounded authority-lock
        descriptors.  Flush and close them before the shared managed-root
        descriptor is released.
        """
        if self._authority_mode:
            if self._owns_lifecycle_authority and not self._released_resources["authority"]:
                self.lifecycle_authority.close()
                self._released_resources["authority"] = True
            if (
                self.guarded_handler_io is not None
                and not self._released_resources["guarded_handler_io"]
            ):
                self.guarded_handler_io.close()
                self._released_resources["guarded_handler_io"] = True
            if self._owns_backend and not self._released_resources["backend"]:
                self.backend.close()
                self._released_resources["backend"] = True
            self._released_resources["admission_barrier"] = True
            return
        operation_repository = self.lifecycle.operation_repository
        flush = getattr(operation_repository, "flush", None)
        if callable(flush):
            flush()
        operation_close = getattr(operation_repository, "close", None)
        if callable(operation_close):
            operation_close()

        if not self._released_resources["manifest_repository"]:
            repository_close = getattr(self.manifest_repository, "close", None)
            if callable(repository_close):
                repository_close()
            self._released_resources["manifest_repository"] = True

        if not self._released_resources["guarded_handler_io"]:
            self.guarded_handler_io.close()
            self._released_resources["guarded_handler_io"] = True

        if self._owns_backend and not self._released_resources["backend"]:
            self.backend.close()
            self._released_resources["backend"] = True

        if not self._released_resources["admission_barrier"]:
            if self._admission_barrier is not None:
                self._admission_barrier.release()
            self._released_resources["admission_barrier"] = True
    
    def __enter__(self):
        self._instance_admission.require_open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    # Private helper methods

    def _materialize_authority_store(self) -> GuardedHandlerIO:
        """Open payload I/O only after authority has created or validated its root."""
        if self.guarded_handler_io is None:
            self.cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.guarded_handler_io = GuardedHandlerIO(self.cache_dir)
        return self.guarded_handler_io

    @staticmethod
    def _authority_absent_expectation() -> EntryExpectation:
        """Return the explicit absence token used by authority preparation."""
        return EntryExpectation.absent()

    def _authority_manifest_key(self, *, initialize_new_store: bool = False) -> bytes:
        """Initialize the persistent key only after authority intent commits."""
        key_context = {
            "operation": (
                "initialize_manifest_key"
                if initialize_new_store
                else "get_manifest_key"
            ),
            "provider": type(self._manifest_key_provider).__name__,
        }
        initialize_or_get = getattr(
            self._manifest_key_provider,
            "get_or_initialize_new_store",
            None,
        )
        try:
            if initialize_new_store and callable(initialize_or_get):
                key = initialize_or_get()
            else:
                try:
                    key = self._manifest_key_provider.get_key()
                except ManifestKeyError:
                    if not initialize_new_store:
                        raise
                    initializer = getattr(
                        self._manifest_key_provider,
                        "initialize_new_store",
                        None,
                    )
                    if not callable(initializer):
                        raise
                    key = initializer()
        except Exception as exc:
            raise CacheBlobManifestUnauthenticatedError(
                "Canonical BlobStore signing key is unavailable",
                context=key_context,
                reason=CacheReason.MANIFEST_SIGNING_KEY_INVALID,
            ) from exc
        if type(key) is not bytes or len(key) != 32:
            raise CacheBlobManifestUnauthenticatedError(
                "Canonical BlobStore signing key is invalid",
                context=key_context,
                reason=CacheReason.MANIFEST_SIGNING_KEY_INVALID,
            )
        return key

    def _authenticated_authority_manifest(
        self, raw: bytes, *, allow_tombstone: bool = False
    ) -> BlobManifestV1:
        """Authenticate authority-owned canonical bytes before trusting locators."""
        try:
            raw_record = decode_canonical_manifest_record(raw)
        except CacheManifestUnsupportedVersionError as exc:
            raise CacheBlobManifestUnsupportedVersionError(
                "Authority manifest schema version is unsupported"
            ) from exc
        except CacheManifestIntegrityError as exc:
            raise CacheBlobManifestMalformedError("Authority manifest is malformed") from exc
        if not verify_hmac_sha256(
            canonical_signing_bytes_from_record(raw_record),
            raw_record.get("signature"),
            self._authority_manifest_key(),
        ):
            raise CacheBlobManifestUnauthenticatedError(
                "Authority manifest cannot be authenticated"
            )
        try:
            manifest = BlobManifestV1.from_mapping(raw_record)
        except CacheManifestUnsupportedVersionError as exc:
            raise CacheBlobManifestUnsupportedVersionError(
                "Authority manifest schema version is unsupported"
            ) from exc
        except CacheManifestIntegrityError as exc:
            raise CacheBlobManifestMalformedError("Authority manifest is malformed") from exc
        if manifest.canonical_bytes() != raw:
            raise CacheBlobManifestMalformedError("Authority manifest is not canonical")
        return manifest

    def _authority_get(self, key: str) -> Optional[Any]:
        """Read one committed authority entry through the Phase 2 snapshot order."""
        entry = self.lifecycle_authority.read_entry(key)
        if entry is None:
            return None
        manifest = self._authenticated_authority_manifest(entry.manifest)
        handler = self.handlers.get_handler_by_type(manifest.handler_type)
        metadata = self._handler_metadata(manifest)
        with self._materialize_authority_store().open_snapshot(
            manifest.locator, metadata
        ) as snapshot:
            digest, byte_size = sha256_and_size(snapshot.path)
            if digest != manifest.digest or byte_size != manifest.byte_size:
                raise CacheBlobPayloadTamperedError(
                    "Canonical BlobStore payload integrity check failed"
                )
            return handler.get(snapshot.path, snapshot.metadata)

    def _compute_content_hash(self, data: Any) -> str:
        """Compute a content-based hash for the data."""
        import pickle
        try:
            serialized = pickle.dumps(data)
        except Exception:
            # Fall back to repr for non-pickleable objects
            serialized = repr(data).encode()
        return hashlib.sha256(serialized).hexdigest()[:16]

    def _require_canonical_store(self) -> None:
        """Reject every public operation on exact read-only legacy evidence."""
        if self._legacy_identity is not None:
            self._legacy_identity.require_explicit_migration()

    def _refresh_metadata_view_for_lifecycle(self) -> None:
        """Refresh a live JSON view through canonical descriptor authority.

        The Phase 1 clear coordinator used one exclusive admission lock around
        every ordinary operation.  Lifecycle publication now relies on exact
        manifest CAS instead, but a JSON writer must still fail closed rather
        than overwriting malformed or stale metadata from its in-memory view.
        """
        if type(self.backend) is not JsonBackend:
            return
        try:
            self.manifest_repository.refresh_authoritative_view()
        except CacheStorageError as exc:
            raise CacheBlobBackendError(
                "BlobStore lifecycle could not refresh metadata state",
                context={"operation": "put", "backend": "json"},
            ) from exc
    
    def _storage_id_for_key(self, key: str) -> str:
        """Map one public logical key to a backend-safe physical ID."""
        return encode_physical_name(key, namespace="blob-store")

    def _initialize_inventory_provenance_key(self) -> bytes:
        """Create/read the trust root only after inventory compatibility proof.

        ``FileOperationRecordRepository`` holds the store-level initialization
        lease while invoking this method.  It therefore cannot leave a fresh
        key behind merely because a raw or v1 lifecycle family made the store
        migration-required.  Existing v3 provenance calls this path only to
        verify against the original, caller-owned key.
        """
        try:
            initializer = getattr(
                self._manifest_key_provider, "get_or_initialize_new_store", None
            )
            key = (
                initializer()
                if callable(initializer)
                else self._manifest_key_provider.get_key()
            )
        except Exception as exc:
            raise CacheBlobManifestUnauthenticatedError(
                "Canonical BlobStore signing key is unavailable",
                context={
                    # Inventory provenance is initialized as part of the
                    # established fresh-store manifest-key boundary.  Keep
                    # the public operation context stable for callers that
                    # distinguish unavailable trust roots from migration
                    # evidence.
                    "operation": "initialize_manifest_key",
                    "provider": type(self._manifest_key_provider).__name__,
                },
                reason=CacheReason.MANIFEST_SIGNING_KEY_INVALID,
            ) from exc
        if type(key) is not bytes or len(key) != 32:
            raise CacheBlobManifestUnauthenticatedError(
                "Canonical BlobStore signing key is invalid",
                context={"operation": "initialize_manifest_key"},
                reason=CacheReason.MANIFEST_SIGNING_KEY_INVALID,
            )
        return key

    def _manifest_key(self, *, initialize_new_store: bool = False) -> bytes:
        """Return the strict persistent key without silently downgrading signing."""
        operation = "get_manifest_key"
        if initialize_new_store:
            # Repository availability is a storage boundary, not a key-provider
            # failure.  Preserve its typed cause instead of misreporting an
            # unavailable metadata projection as unauthenticated key material.
            try:
                has_manifest = bool(
                    self.manifest_repository.list_page(page_size=1).entries
                )
            except CacheStorageError:
                raise
        else:
            has_manifest = True
        if initialize_new_store and not has_manifest:
            # Establish current-v2 provenance only after one fail-closed
            # compatibility proof under the inventory initialization
            # transition. In particular, a missing key must never turn a raw
            # primary, sidecar, or pending record into a fresh store. Keep a
            # typed migration-required result distinct from unavailable key
            # material so callers can take the documented migration path.
            self.lifecycle.operation_repository.initialize_new_store()
        try:
            if initialize_new_store and not has_manifest:
                operation = "initialize_manifest_key"
                initializer = getattr(
                    self._manifest_key_provider, "get_or_initialize_new_store", None
                )
                if callable(initializer):
                    key = initializer()
                else:
                    # An injected provider owns its trust-root provisioning.
                    # It may expose only the narrow public ``get_key`` protocol.
                    key = self._manifest_key_provider.get_key()
            else:
                key = self._manifest_key_provider.get_key()
            if type(key) is not bytes or len(key) != 32:
                raise ManifestKeyError(
                    "Canonical manifest key provider returned invalid key material"
                )
        except Exception as exc:
            raise CacheBlobManifestUnauthenticatedError(
                "Canonical BlobStore signing key is unavailable",
                context={
                    "operation": operation,
                    "provider": type(self._manifest_key_provider).__name__,
                },
                reason=CacheReason.MANIFEST_SIGNING_KEY_INVALID,
            ) from exc
        return key

    def _delete_or_prove_absent(self, locator: Path) -> None:
        """Remove a contained payload only when deletion is conclusively known."""
        cleanup_error: Exception | None = None
        try:
            # Lifecycle cleanup is a post-authority transition.  It must use
            # the durable deletion primitive rather than the compatibility
            # ``delete`` helper so the Windows backend keeps deletion bound to
            # its reparse-safe disposition handle as well.
            if self.guarded_handler_io.file_ops.delete_durable(locator):
                return
        except Exception as exc:
            cleanup_error = exc

        try:
            if not self.guarded_handler_io.file_ops.exists(locator):
                return
        except Exception as exc:
            cleanup_error = exc

        context = {"operation": "payload_cleanup"}
        if cleanup_error is not None:
            context["cleanup_error"] = type(cleanup_error).__name__
        raise CacheStorageError("Could not prove managed payload cleanup", context=context)

    def _cleanup_uncommitted_candidate(
        self,
        candidate_locator: Path,
        triggering_error: BaseException,
    ) -> None:
        """Erase an uncommitted candidate or expose the unresolved residue."""
        try:
            self._delete_or_prove_absent(candidate_locator)
        except CacheStorageError as cleanup_error:
            raise CacheStorageError(
                "Candidate payload cleanup could not be confirmed",
                context={
                    "operation": "put",
                    "cleanup_error": cleanup_error.context.get("cleanup_error"),
                },
            ) from triggering_error

    def _cleanup_prior_payload(self, previous_locator: Path) -> None:
        """Erase the superseded payload without rolling back published metadata."""
        try:
            self._delete_or_prove_absent(previous_locator)
        except CacheStorageError as cleanup_error:
            raise CacheStorageError(
                "Prior payload cleanup failed after metadata publication",
                context={"operation": "put"},
            ) from cleanup_error

    def _list_backend_entries(self, *, operation: str) -> List[Dict[str, Any]]:
        """Read compatibility projections with the canonical backend taxonomy."""
        try:
            return self.manifest_repository.list_backend_entries()
        except CacheBlobBackendError:
            raise
        except (CacheStorageError, OSError, TypeError, ValueError) as exc:
            raise CacheBlobBackendError(
                "BlobStore compatibility metadata projection failed",
                context={"operation": operation},
            ) from exc

    def _entry_locator(
        self,
        entry: Dict[str, Any],
        logical_key: str,
        *,
        operation: str,
    ) -> Path:
        """Validate every persisted locator before consuming an entry.

        Older backends can surface ``actual_path`` at the entry top level or
        inside ``metadata``. Every supplied form is validated even though the
        first form keeps the historical precedence for the actual operation.
        """
        nested_metadata = entry.get("metadata", {})
        nested_path = (
            nested_metadata.get("actual_path")
            if isinstance(nested_metadata, dict)
            else None
        )
        locator_values = [entry.get("actual_path"), nested_path]
        validated = []
        for locator in locator_values:
            if locator is not None:
                validated.append(
                    resolve_managed_locator(
                        self.guarded_handler_io.root,
                        locator,
                        operation=operation,
                    )
                )
        if validated:
            return validated[0]
        return self.guarded_handler_io.file_ops.blob_locator(
            self._storage_id_for_key(logical_key), shard_chars=0
        )

    def _load_authenticated_manifest_with_raw(
        self,
        key: str,
        *,
        operation: str,
        require_payload_contract: bool = False,
        require_locator: bool = False,
        allowed_states: frozenset[str] | None = None,
    ) -> tuple[BlobManifestV1, bytes, Any | None, Path | None] | None:
        """Load one committed manifest and retain its exact authority bytes.

        A raw repository miss is the only absence outcome. Every other record
        is decoded, authenticated, and checked for key/state consistency before
        callers may resolve a handler, open a snapshot, expose metadata, or
        trust a physical locator.
        """
        raw_manifest = self.manifest_repository.get_raw(key)
        if raw_manifest is None:
            return None
        try:
            raw_record = decode_canonical_manifest_record(raw_manifest)
        except CacheManifestUnsupportedVersionError as exc:
            raise CacheBlobManifestUnsupportedVersionError(
                "Canonical BlobStore manifest schema is unsupported"
            ) from exc
        except ManifestDecodeError as exc:
            raise CacheBlobManifestMalformedError(
                "Canonical BlobStore manifest is malformed"
            ) from exc
        if not verify_hmac_sha256(
            canonical_signing_bytes_from_record(raw_record),
            raw_record.get("signature"),
            self._manifest_key(),
        ):
            raise CacheBlobManifestUnauthenticatedError(
                "Canonical BlobStore manifest signature is invalid"
            )
        try:
            manifest = BlobManifestV1.from_mapping(raw_record)
        except CacheManifestUnsupportedVersionError as exc:
            if "Payload format" in str(exc):
                raise CacheBlobPayloadUnsupportedVersionError(
                    "Canonical BlobStore payload format is unsupported"
                ) from exc
            raise CacheBlobManifestUnsupportedVersionError(
                "Canonical BlobStore manifest schema is unsupported"
            ) from exc
        except ManifestDecodeError as exc:
            raise CacheBlobManifestMalformedError(
                "Canonical BlobStore manifest is malformed"
            ) from exc
        # A valid signature covers the canonical semantic projection, not an
        # arbitrary JSON spelling.  The repository record itself is authority,
        # so accepting reordered or whitespace-padded bytes here would make
        # normal operations disagree with reconciliation and could turn a
        # later exact-CAS expectation into a representation rewrite.
        if manifest.canonical_bytes() != raw_manifest:
            raise CacheBlobManifestMalformedError(
                "Canonical BlobStore manifest bytes are not canonical"
            )
        if manifest.key != key:
            raise CacheBlobLifecycleConflictError(
                "Canonical BlobStore manifest key conflicts with lookup"
            )
        expected_states = (
            frozenset({"committed"}) if allowed_states is None else allowed_states
        )
        if manifest.state not in expected_states:
            raise CacheBlobLifecycleConflictError(
                "Canonical BlobStore manifest state conflicts with the operation",
                context={
                    "key": key,
                    "operation": operation,
                    "state": manifest.state,
                },
            )

        handler = None
        if require_payload_contract:
            handler = self._resolve_payload_handler(manifest)

        actual_path = None
        if require_locator:
            actual_path = resolve_managed_locator(
                self.guarded_handler_io.root,
                manifest.locator,
                operation=operation,
            )
        return manifest, raw_manifest, handler, actual_path

    def _load_authenticated_manifest(
        self,
        key: str,
        *,
        operation: str,
        require_payload_contract: bool = False,
        require_locator: bool = False,
        allowed_states: frozenset[str] | None = None,
    ) -> tuple[BlobManifestV1, Any | None, Path | None] | None:
        """Load a manifest without exposing its private exact-record snapshot."""
        authenticated = self._load_authenticated_manifest_with_raw(
            key,
            operation=operation,
            require_payload_contract=require_payload_contract,
            require_locator=require_locator,
            allowed_states=allowed_states,
        )
        if authenticated is None:
            return None
        manifest, _raw_manifest, handler, actual_path = authenticated
        return manifest, handler, actual_path

    def _read_generation_is_stable(
        self,
        key: str,
        first_manifest: BlobManifestV1,
        attempt: int,
        *,
        operation: str = "get",
    ) -> bool:
        """Reauthenticate M2 and allow only one retry for a newer generation."""
        authenticated = self._load_authenticated_manifest(
            key,
            operation=f"{operation}_reauthenticate",
            require_payload_contract=True,
            require_locator=True,
        )
        if authenticated is None:
            raise CacheBlobLifecycleConflictError(
                "Canonical BlobStore authority disappeared after a committed snapshot",
                context={"key": key, "operation": operation},
            )
        second_manifest, _handler, _locator = authenticated
        if second_manifest.generation == first_manifest.generation:
            return True
        if attempt == 0:
            return False
        raise CacheBlobLifecycleConflictError(
            "Canonical BlobStore authority changed during both read attempts",
            context={"key": key, "operation": operation},
        )

    def _resolve_payload_handler(self, manifest: BlobManifestV1) -> Any:
        """Resolve one signed handler contract without opening payload bytes."""
        resolver = getattr(self.handlers, "resolve_payload_contract", None)
        try:
            if callable(resolver):
                return resolver(
                    manifest.handler_type,
                    manifest.payload_format,
                    manifest.payload_format_version,
                )

            # Small compatibility registries used by existing direct callers
            # predate explicit payload contract lookup. They still need an
            # authenticated exact native-format agreement before a snapshot.
            handler = self.handlers.get_handler_by_type(manifest.handler_type)
            declared_format = manifest.handler_metadata.get("storage_format")
            if (
                declared_format != manifest.payload_format
                or manifest.payload_format_version != 1
            ):
                raise CacheManifestUnsupportedVersionError(
                    "Canonical manifest declares an unsupported native payload contract"
                )
            return handler
        except (CacheManifestUnsupportedVersionError, ValueError) as exc:
            raise CacheBlobPayloadUnsupportedVersionError(
                "Canonical BlobStore payload contract is unsupported"
            ) from exc

    @staticmethod
    def _handler_metadata(manifest: BlobManifestV1) -> Dict[str, Any]:
        """Build the compatible handler view solely from signed manifest data."""
        return {
            **dict(manifest.user_metadata),
            **dict(manifest.handler_metadata),
            "cache_key": manifest.key,
            "data_type": manifest.handler_type,
            "storage_format": manifest.payload_format,
            "file_size": manifest.byte_size,
            "created_at": manifest.created_at,
        }

    def _manifest_entry_data(self, manifest: BlobManifestV1) -> Dict[str, Any]:
        """Build the public metadata shape without trusting backend projections."""
        return {
            "cache_key": manifest.key,
            "data_type": manifest.handler_type,
            "file_size": manifest.byte_size,
            "created_at": manifest.created_at,
            "metadata": {
                **dict(manifest.user_metadata),
                **dict(manifest.handler_metadata),
                "actual_path": manifest.locator,
            },
        }

    def _preflight_entries(
        self, entries: List[Dict[str, Any]], *, operation: str
    ) -> None:
        """Fail before list/clear can expose or mutate any safe sibling."""
        for entry in entries:
            logical_key = entry.get("cache_key")
            self._entry_locator(entry, logical_key, operation=operation)

    def _preflight_clear_manifests(self) -> List[tuple[str, Path]]:
        """Authenticate every clear target before recovery may mutate anything."""
        manifest_keys = self.manifest_repository.list_keys()
        backend_entries = self._list_backend_entries(operation="clear")
        backend_keys = set()
        for entry in backend_entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("cache_key"), str):
                raise CacheBlobLifecycleConflictError(
                    "BlobStore clear found a metadata entry without a canonical key"
                )
            backend_keys.add(entry["cache_key"])
        if backend_keys != set(manifest_keys):
            raise CacheBlobLifecycleConflictError(
                "BlobStore clear requires a canonical manifest for every metadata entry"
            )

        mappings = []
        for key in manifest_keys:
            authenticated = self._load_authenticated_manifest(
                key,
                operation="clear",
                require_locator=True,
            )
            if authenticated is None:
                raise CacheBlobLifecycleConflictError(
                    "BlobStore manifest disappeared during clear preflight"
                )
            _manifest, _handler, actual_path = authenticated
            assert actual_path is not None
            mappings.append((key, actual_path))

        # Do not trust these backend-shaped paths for recovery mappings. They
        # are validated only after all canonical records authenticate so an
        # unsafe compatibility projection cannot bypass the containment guard.
        self._preflight_entries(backend_entries, operation="clear")
        return mappings
    
    def _generate_unique_key(self) -> str:
        """Generate a unique blob key."""
        return uuid.uuid4().hex[:16]
