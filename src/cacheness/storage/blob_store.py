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
    CacheBlobManifestMalformedError,
    CacheBlobManifestUnauthenticatedError,
    CacheBlobManifestUnsupportedVersionError,
    CacheBlobPayloadMissingError,
    CacheBlobPayloadTamperedError,
    CacheBlobPayloadUnsupportedVersionError,
    CacheManifestUnsupportedVersionError,
    CacheReason,
    CacheStorageError,
)
from .backends import MetadataBackend, JsonBackend
from .clear_recovery import ClearRecoveryCoordinator
from .guarded_handler_io import GuardedHandlerIO
from .handlers import HandlerRegistry
from .integrity import (
    ManifestKeyError,
    ManifestKeyProvider,
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
from .manifest_repository import ManifestExpectation, create_manifest_repository
from .legacy_manifest import LegacyManifestIdentity, recognize_legacy_fixture_tree
from .lifecycle import LifecycleEngine
from .path_security import encode_physical_name, resolve_managed_locator
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


def _clear_coordinated(method: Callable) -> Callable:
    """Serialize a local lifecycle operation with clear/recovery when available."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        coordinator = self._clear_recovery
        if coordinator is None:
            return method(self, *args, **kwargs)
        try:
            with coordinator.mutation_admission():
                return method(self, *args, **kwargs)
        except CacheStorageError as exc:
            _raise_translated_recovery_failure(coordinator, exc)

    return wrapped


def _clear_read_coordinated(method: Callable) -> Callable:
    """Exclude an active clear without forcing terminal cleanup during reads."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        coordinator = self._clear_recovery
        if coordinator is None:
            return method(self, *args, **kwargs)
        try:
            with coordinator.read_admission():
                return method(self, *args, **kwargs)
        except CacheStorageError as exc:
            _raise_translated_recovery_failure(coordinator, exc)

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
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.guarded_handler_io = GuardedHandlerIO(self.cache_dir)
        self._owns_backend = False
        try:
            self._initialize_after_guarded_io(
                backend,
                compression,
                compression_level,
                content_addressable,
                config,
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
        
        # Initialize metadata backend
        if self._legacy_identity is not None:
            self.backend = InMemoryBackend()
            self._owns_backend = True
        elif backend is None or backend == "json":
            self.backend = JsonBackend(self.cache_dir / "cache_metadata.json")
            self._owns_backend = True
        elif backend == "sqlite":
            from .backends import SqliteBackend
            self.backend = SqliteBackend(self.cache_dir / "cache_metadata.db")
            self._owns_backend = True
        elif isinstance(backend, (MetadataBackend, CoreMetadataBackend)):
            self.backend = backend
        else:
            raise ValueError(f"Unknown backend type: {backend}")
        
        # Initialize handler registry
        self.handlers = HandlerRegistry()
        self.manifest_repository = create_manifest_repository(self.backend)
        self._manifest_key_provider = ManifestKeyProvider(
            self.cache_dir / "blob_manifest_hmac_key.bin"
        )
        self.lifecycle = LifecycleEngine(self, lifecycle_limits=self.lifecycle_limits)

        # Clear recovery is deliberately confined to exact local backend
        # identities. Capability-shaped or wrapped backends never inherit a
        # crash boundary merely because they expose similarly named methods.
        self._clear_recovery = None
        if (
            self._legacy_identity is None
            and ClearRecoveryCoordinator.can_coordinate(self.backend)
        ):
            self._clear_recovery = ClearRecoveryCoordinator(
                self.guarded_handler_io.file_ops,
                self.backend,
            )
            try:
                with self._clear_recovery.admission():
                    self._clear_recovery.recover()
                    self._reconcile_sqlite_manifest_records_after_clear()
            except Exception:
                raise

    def _close_failed_initialization_resources(self) -> None:
        """Release only resources this incomplete store has taken ownership of."""
        if self._owns_backend and hasattr(self, "backend"):
            try:
                self.backend.close()
            except Exception:
                logger.exception("Failed to close internally created BlobStore backend")
        try:
            self.guarded_handler_io.close()
        except Exception:
            logger.exception("Failed to close BlobStore managed-root descriptor")

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
        self._refresh_metadata_view_for_lifecycle()
        # Generate key
        if self.content_addressable:
            # Use content hash as key
            blob_key = self._compute_content_hash(data)
        elif key is not None:
            blob_key = key
        else:
            blob_key = self._generate_unique_key()

        stored_key = self.lifecycle.put(data, key=blob_key, metadata=metadata)
        logger.debug(f"Stored blob {stored_key} through the lifecycle engine")
        return stored_key
    
    @_clear_read_coordinated
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a blob by key.
        
        Args:
            key: The blob key
            
        Returns:
            The stored data, or None if not found
        """
        self._require_canonical_store()
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

        # The resolved handler receives only this private snapshot. It is
        # hashed and size-checked inside the same live context before any
        # trusted-payload deserialization can begin.
        try:
            with self.guarded_handler_io.open_snapshot(
                actual_path, handler_metadata
            ) as snapshot:
                digest, byte_size = sha256_and_size(snapshot.path)
                if digest != manifest.digest or byte_size != manifest.byte_size:
                    raise CacheBlobPayloadTamperedError(
                        "Canonical BlobStore payload integrity check failed"
                    )
                data = handler.get(snapshot.path, snapshot.metadata)
        except FileNotFoundError as exc:
            raise CacheBlobPayloadMissingError(
                "Canonical BlobStore payload is missing"
            ) from exc
        except CacheBlobPayloadTamperedError:
            raise
        except OSError as exc:
            raise CacheBlobPayloadTamperedError(
                "Canonical BlobStore payload could not be verified"
            ) from exc
        
        return data
    
    @_clear_read_coordinated
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get blob metadata without loading the blob content.
        
        Args:
            key: The blob key
            
        Returns:
            Metadata dictionary, or None if not found
        """
        self._require_canonical_store()
        authenticated = self._load_authenticated_manifest(
            key,
            operation="get_metadata",
            require_locator=True,
        )
        if authenticated is None:
            return None
        manifest, _handler, _locator = authenticated
        return self._manifest_entry_data(manifest)
    
    @_clear_coordinated
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

        authenticated = self._load_authenticated_manifest(
            key,
            operation="update_metadata",
            require_locator=True,
        )
        if authenticated is None:
            return False
        manifest, _handler, _locator = authenticated
        observed_record = self.manifest_repository.get_raw(key)
        if observed_record is None or observed_record != manifest.canonical_bytes():
            raise CacheBlobLifecycleConflictError(
                "BlobStore metadata authority changed before conditional patch",
                context={"key": key, "operation": "update_metadata"},
            )
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
    
    def delete(self, key: str) -> bool:
        """
        Delete a blob and its metadata.
        
        Args:
            key: The blob key
            
        Returns:
            True if deleted, False if not found
        """
        self._require_canonical_store()
        deleted = self.lifecycle.delete(key=key)
        if deleted:
            logger.debug(f"Deleted blob {key} through the lifecycle engine")
        return deleted
    
    @_clear_read_coordinated
    def exists(self, key: str) -> bool:
        """
        Check if a blob exists.
        
        Args:
            key: The blob key
            
        Returns:
            True if the blob exists
        """
        self._require_canonical_store()
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
            with self.guarded_handler_io.open_snapshot(actual_path, {}) as snapshot:
                digest, byte_size = sha256_and_size(snapshot.path)
                if digest != manifest.digest or byte_size != manifest.byte_size:
                    raise CacheBlobPayloadTamperedError(
                        "Canonical BlobStore payload integrity check failed"
                    )
        except FileNotFoundError as exc:
            raise CacheBlobPayloadMissingError(
                "Canonical BlobStore payload is missing"
            ) from exc
        except CacheBlobPayloadTamperedError:
            raise
        except OSError as exc:
            raise CacheBlobPayloadTamperedError(
                "Canonical BlobStore payload could not be verified"
            ) from exc
        return True
    
    @_clear_read_coordinated
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
        entries = self._list_backend_entries(operation="list")
        keys = []

        for key in self.manifest_repository.list_keys():
            if prefix and not key.startswith(prefix):
                continue

            authenticated = self._load_authenticated_manifest(
                key,
                operation="list",
                require_locator=True,
            )
            assert authenticated is not None
            manifest, _handler, _locator = authenticated
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

        # Legacy projections are never returned or used as authoritative
        # metadata, but their locators remain a containment tripwire for this
        # long-standing all-or-error public operation.
        self._preflight_entries(entries, operation="list")
        return keys
    
    @_clear_coordinated
    def clear(self) -> int:
        """
        Remove every managed payload with recoverable clear tombstones.

        Returns:
            Number of blobs removed
        """
        self._require_canonical_store()
        if self._clear_recovery is None:
            error = ClearRecoveryCoordinator.unsupported_error(self.backend)
            raise CacheBlobBackendError(str(error), context=error.context) from error

        mappings = self._preflight_clear_manifests()
        try:
            cleared = self._clear_recovery.clear(mappings)
        except (CacheBlobBackendError, CacheBlobLifecycleConflictError):
            raise
        except CacheStorageError as exc:
            if ClearRecoveryCoordinator.is_lifecycle_conflict(exc):
                raise CacheBlobLifecycleConflictError(
                    str(exc), context=exc.context
                ) from exc
            raise CacheBlobBackendError(str(exc), context=exc.context) from exc
        except Exception as exc:
            raise CacheBlobBackendError(
                "BlobStore clear transaction failed",
                context={
                    "operation": "clear",
                    "backend": type(self.backend).__name__,
                },
            ) from exc
        if type(self.backend) is SqliteBackend:
            for cache_key, _ in mappings:
                self.manifest_repository.remove(cache_key)
        return cleared

    def _reconcile_sqlite_manifest_records_after_clear(self) -> None:
        """Drop SQLite sidecar records orphaned by a completed clear recovery.

        Clear recovery remains owned by the Phase 1 coordinator. This only
        removes dedicated canonical BLOB rows after that coordinator has
        reached a terminal state and established ``cache_entries`` authority.
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
    
    def close(self):
        """Close the blob store and release resources."""
        self.guarded_handler_io.close()
        self.backend.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    # Private helper methods
    
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
        """Refresh a live JSON view without reacquiring global clear admission.

        The Phase 1 clear coordinator used one exclusive admission lock around
        every ordinary operation.  Lifecycle publication now relies on exact
        manifest CAS instead, but a JSON writer must still fail closed rather
        than overwriting malformed or stale metadata from its in-memory view.
        """
        coordinator = self._clear_recovery
        if coordinator is None:
            return
        try:
            coordinator._refresh_backend_view()
        except CacheStorageError as exc:
            raise CacheBlobBackendError(
                "BlobStore lifecycle could not refresh metadata state",
                context={"operation": "put", "backend": coordinator.kind},
            ) from exc
    
    def _storage_id_for_key(self, key: str) -> str:
        """Map one public logical key to a backend-safe physical ID."""
        return encode_physical_name(key, namespace="blob-store")

    def _manifest_key(self, *, initialize_new_store: bool = False) -> bytes:
        """Return the strict persistent key without silently downgrading signing."""
        if initialize_new_store and not self.manifest_repository.list_keys():
            try:
                return self._manifest_key_provider.get_or_initialize_new_store()
            except ManifestKeyError as exc:
                raise CacheBlobManifestUnauthenticatedError(
                    "Canonical BlobStore signing key is unavailable",
                    reason=CacheReason.MANIFEST_SIGNING_KEY_INVALID,
                ) from exc
        try:
            return self._manifest_key_provider.get_key()
        except ManifestKeyError as exc:
            raise CacheBlobManifestUnauthenticatedError(
                "Canonical BlobStore signing key is unavailable",
                reason=CacheReason.MANIFEST_SIGNING_KEY_INVALID,
            ) from exc

    def _delete_or_prove_absent(self, locator: Path) -> None:
        """Remove a contained payload only when deletion is conclusively known."""
        cleanup_error: Exception | None = None
        try:
            if self.guarded_handler_io.file_ops.delete(locator):
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

    def _load_authenticated_manifest(
        self,
        key: str,
        *,
        operation: str,
        require_payload_contract: bool = False,
        require_locator: bool = False,
        allowed_states: frozenset[str] | None = None,
    ) -> tuple[BlobManifestV1, Any | None, Path | None] | None:
        """Load one committed manifest before any direct public operation.

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
        return manifest, handler, actual_path

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
