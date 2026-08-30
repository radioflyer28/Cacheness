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
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime, timezone

from ..error_handling import (
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
from .manifest import BlobManifestV1, ManifestDecodeError
from .manifest_repository import create_manifest_repository
from .path_security import encode_physical_name, resolve_managed_locator
from ..metadata import MetadataBackend as CoreMetadataBackend
from ..metadata import SqliteBackend

# Import CacheConfig for proper handler configuration
from ..config import CacheConfig, CompressionConfig

logger = logging.getLogger(__name__)


def _clear_coordinated(method: Callable) -> Callable:
    """Serialize a local lifecycle operation with clear/recovery when available."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        coordinator = self._clear_recovery
        if coordinator is None:
            return method(self, *args, **kwargs)
        with coordinator.mutation_admission():
            return method(self, *args, **kwargs)

    return wrapped


def _clear_read_coordinated(method: Callable) -> Callable:
    """Exclude an active clear without forcing terminal cleanup during reads."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        coordinator = self._clear_recovery
        if coordinator is None:
            return method(self, *args, **kwargs)
        with coordinator.read_admission():
            return method(self, *args, **kwargs)

    return wrapped


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
    ):
        """
        Initialize a BlobStore.
        
        Args:
            cache_dir: Directory for storing blobs and metadata
            backend: Metadata backend - "json", "sqlite", or a MetadataBackend instance
            compression: Compression codec (lz4, zstd, gzip, blosclz, etc.)
            compression_level: Compression level (1-9)
            content_addressable: If True, use content hash as blob key
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.guarded_handler_io = GuardedHandlerIO(self.cache_dir)
        
        self.compression = compression
        self.compression_level = compression_level
        self.content_addressable = content_addressable
        
        # Create config for handlers
        self.config = CacheConfig(
            cache_dir=self.cache_dir,
            compression=CompressionConfig(
                pickle_compression_codec=compression,
                pickle_compression_level=compression_level,
                blosc2_array_clevel=compression_level,
            ),
        )
        
        # Initialize metadata backend
        if backend is None or backend == "json":
            self.backend = JsonBackend(self.cache_dir / "cache_metadata.json")
        elif backend == "sqlite":
            from .backends import SqliteBackend
            self.backend = SqliteBackend(self.cache_dir / "cache_metadata.db")
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

        # Clear recovery is deliberately confined to exact local backend
        # identities. Capability-shaped or wrapped backends never inherit a
        # crash boundary merely because they expose similarly named methods.
        self._clear_recovery = None
        if ClearRecoveryCoordinator.can_coordinate(self.backend):
            self._clear_recovery = ClearRecoveryCoordinator(
                self.guarded_handler_io.file_ops,
                self.backend,
            )
            try:
                with self._clear_recovery.admission():
                    self._clear_recovery.recover()
                    self._reconcile_sqlite_manifest_records_after_clear()
            except Exception:
                self.guarded_handler_io.close()
                raise
        
        logger.debug(f"BlobStore initialized at {self.cache_dir}")
    
    @_clear_coordinated
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
        # Generate key
        if self.content_addressable:
            # Use content hash as key
            blob_key = self._compute_content_hash(data)
        elif key is not None:
            blob_key = key
        else:
            blob_key = self._generate_unique_key()

        storage_id = self._storage_id_for_key(blob_key)
        existing = self.backend.get_entry(blob_key)
        previous_locator = None
        if existing is not None:
            # Refuse to overwrite a record whose evidence points outside this
            # store before serializing or publishing a replacement payload.
            previous_locator = self._entry_locator(
                existing,
                blob_key,
                operation="overwrite",
            )
        
        # Get appropriate handler
        handler = self.handlers.get_handler(data)

        # A candidate is not authoritative until metadata publication has
        # returned successfully. Its physical base remains derived from the
        # stable logical-key ID, while the nonce prevents overwriting a prior
        # committed payload before that publication boundary.
        candidate_id = f"{storage_id}-candidate-{uuid.uuid4().hex}"
        candidate_locator: Path | None = None
        metadata_committed = False
        try:
            result = self.guarded_handler_io.put(
                handler,
                data,
                candidate_id,
                self.config,
            )
            candidate_locator = resolve_managed_locator(
                self.guarded_handler_io.root,
                result["actual_path"],
                operation="candidate_publish",
            )

            digest, byte_size = sha256_and_size(candidate_locator)
            handler_metadata = dict(result.get("metadata", {}) or {})
            storage_format = result.get("storage_format", "pickle")
            payload_format = result.get("payload_format", storage_format)
            payload_format_version = result.get("payload_format_version", 1)
            handler_metadata["storage_format"] = storage_format
            handler_metadata.setdefault("compression_codec", self.compression)
            created_at = datetime.now(timezone.utc).isoformat()
            manifest = BlobManifestV1(
                schema_version=1,
                key=blob_key,
                generation=uuid.uuid4().hex,
                state="committed",
                locator=str(candidate_locator),
                handler_type=handler.data_type,
                payload_format=payload_format,
                payload_format_version=payload_format_version,
                digest_algorithm="sha256",
                digest=digest,
                byte_size=byte_size,
                created_at=created_at,
                handler_metadata=handler_metadata,
                user_metadata=dict(metadata or {}),
            )
            signed_manifest = manifest.with_signature(
                sign_hmac_sha256(
                    manifest.signing_bytes(),
                    self._manifest_key(initialize_new_store=True),
                )
            )
            raw_manifest = signed_manifest.canonical_bytes()
            entry_data = {
                "cache_key": blob_key,
                "data_type": signed_manifest.handler_type,
                "file_size": signed_manifest.byte_size,
                "created_at": signed_manifest.created_at,
                "metadata": {
                    **dict(signed_manifest.user_metadata),
                    **dict(signed_manifest.handler_metadata),
                    "actual_path": signed_manifest.locator,
            "storage_format": storage_format,
                    "compression_codec": self.compression,
                },
            }
            self.manifest_repository.put_raw(
                blob_key, raw_manifest, entry_data=entry_data
            )
            metadata_committed = True
        except BaseException as exc:
            if candidate_locator is not None and not metadata_committed:
                self._cleanup_uncommitted_candidate(candidate_locator, exc)
            raise

        # The candidate is now the sole authoritative payload. A failure to
        # erase the superseded payload cannot roll metadata back to stale
        # evidence, but it must remain visible to callers for reconciliation.
        if previous_locator is not None and previous_locator != candidate_locator:
            self._cleanup_prior_payload(previous_locator)
        
        logger.debug(f"Stored blob {blob_key}: {handler.data_type}, {entry_data['file_size']} bytes")
        
        return blob_key
    
    @_clear_read_coordinated
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a blob by key.
        
        Args:
            key: The blob key
            
        Returns:
            The stored data, or None if not found
        """
        raw_manifest = self.manifest_repository.get_raw(key)
        if raw_manifest is None:
            logger.debug(f"Blob not found: {key}")
            return None
        try:
            manifest = BlobManifestV1.from_canonical_bytes(raw_manifest)
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
        if not verify_hmac_sha256(
            manifest.signing_bytes(),
            manifest.signature,
            self._manifest_key(),
        ):
            raise CacheBlobManifestUnauthenticatedError(
                "Canonical BlobStore manifest signature is invalid",
            )
        if manifest.key != key:
            raise CacheBlobLifecycleConflictError(
                "Canonical BlobStore manifest key conflicts with lookup"
            )
        if manifest.state != "committed":
            raise CacheBlobLifecycleConflictError(
                "Canonical BlobStore manifest is not committed"
            )

        resolver = getattr(self.handlers, "resolve_payload_contract", None)
        try:
            if callable(resolver):
                handler = resolver(
                    manifest.handler_type,
                    manifest.payload_format,
                    manifest.payload_format_version,
                )
            else:
                # Small compatibility registries used by existing direct
                # callers predate the explicit handler declaration protocol.
                # They still need an authenticated exact identity and a
                # signed native-format agreement before a snapshot can open.
                handler = self.handlers.get_handler_by_type(manifest.handler_type)
                declared_format = manifest.handler_metadata.get("storage_format")
                if (
                    declared_format != manifest.payload_format
                    or manifest.payload_format_version != 1
                ):
                    raise CacheManifestUnsupportedVersionError(
                        "Canonical manifest declares an unsupported native payload contract"
                    )
        except (CacheManifestUnsupportedVersionError, ValueError) as exc:
            raise CacheBlobPayloadUnsupportedVersionError(
                "Canonical BlobStore payload contract is unsupported"
            ) from exc

        actual_path = resolve_managed_locator(
            self.guarded_handler_io.root,
            manifest.locator,
            operation="get",
        )
        handler_metadata = {
            **dict(manifest.user_metadata),
            **dict(manifest.handler_metadata),
            "cache_key": manifest.key,
            "data_type": manifest.handler_type,
            "storage_format": manifest.payload_format,
            "file_size": manifest.byte_size,
            "created_at": manifest.created_at,
        }

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
        
        # Update access time
        self.backend.update_access_time(key)
        
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
        entry = self.backend.get_entry(key)
        if entry is None:
            return None
        self._entry_locator(entry, key, operation="get_metadata")
        return {**entry, "cache_key": entry.get("cache_key", key)}
    
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
        existing = self.backend.get_entry(key)
        if existing is None:
            return False
        
        # Validate both the existing and prospective record before a metadata
        # write. A metadata patch cannot introduce an unsafe locator.
        self._entry_locator(existing, key, operation="update_metadata")
        nested_meta = existing.get("metadata", {})
        if not isinstance(nested_meta, dict):
            nested_meta = {}
        else:
            nested_meta = dict(nested_meta)
        
        # Merge user metadata into nested dict
        nested_meta.update(metadata)
        
        # Update the entry
        updated = {**existing, "metadata": nested_meta}
        self._entry_locator(updated, key, operation="update_metadata")
        
        self.backend.put_entry(key, updated)
        return True
    
    @_clear_coordinated
    def delete(self, key: str) -> bool:
        """
        Delete a blob and its metadata.
        
        Args:
            key: The blob key
            
        Returns:
            True if deleted, False if not found
        """
        entry = self.backend.get_entry(key)
        if entry is None:
            return False
        
        actual_path = self._entry_locator(entry, key, operation="delete")
        self.guarded_handler_io.file_ops.delete(actual_path)
        
        # Remove metadata
        self.backend.remove_entry(key)
        
        logger.debug(f"Deleted blob: {key}")
        return True
    
    @_clear_read_coordinated
    def exists(self, key: str) -> bool:
        """
        Check if a blob exists.
        
        Args:
            key: The blob key
            
        Returns:
            True if the blob exists
        """
        entry = self.backend.get_entry(key)
        if entry is None:
            return False
        
        actual_path = self._entry_locator(entry, key, operation="exists")
        return self.guarded_handler_io.file_ops.exists(actual_path)
    
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
        entries = self.backend.list_entries()
        self._preflight_entries(entries, operation="list")
        keys = []
        
        for entry in entries:
            key = entry.get("cache_key", "")
            
            # Apply prefix filter
            if prefix and not key.startswith(prefix):
                continue
            
            # Apply metadata filter
            if metadata_filter:
                match = True
                for field, value in metadata_filter.items():
                    if entry.get(field) != value:
                        match = False
                        break
                if not match:
                    continue
            
            keys.append(key)
        
        return keys
    
    @_clear_coordinated
    def clear(self) -> int:
        """
        Remove every managed payload with recoverable clear tombstones.

        Returns:
            Number of blobs removed
        """
        if self._clear_recovery is None:
            raise ClearRecoveryCoordinator.unsupported_error(self.backend)

        entries = self.backend.list_entries()
        self._preflight_entries(entries, operation="clear")
        mappings = []
        for entry in entries:
            cache_key = entry.get("cache_key", "")
            actual_path = self._entry_locator(entry, cache_key, operation="clear")
            # Every metadata entry participates, even when its payload is
            # already absent, so snapshot and journal cardinalities agree.
            mappings.append((cache_key, actual_path))
        cleared = self._clear_recovery.clear(mappings)
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
    
    def _storage_id_for_key(self, key: str) -> str:
        """Map one public logical key to a backend-safe physical ID."""
        return encode_physical_name(key, namespace="blob-store")

    def _manifest_key(self, *, initialize_new_store: bool = False) -> bytes:
        """Return the strict persistent key without silently downgrading signing."""
        try:
            return self._manifest_key_provider.get_key()
        except ManifestKeyError as exc:
            if initialize_new_store and not self.manifest_repository.list_keys():
                try:
                    return self._manifest_key_provider.initialize_new_store()
                except ManifestKeyError as initialization_error:
                    raise CacheBlobManifestUnauthenticatedError(
                        "Canonical BlobStore signing key is unavailable",
                        reason=CacheReason.MANIFEST_SIGNING_KEY_INVALID,
                    ) from initialization_error
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

    def _preflight_entries(
        self, entries: List[Dict[str, Any]], *, operation: str
    ) -> None:
        """Fail before list/clear can expose or mutate any safe sibling."""
        for entry in entries:
            logical_key = entry.get("cache_key")
            self._entry_locator(entry, logical_key, operation=operation)
    
    def _generate_unique_key(self) -> str:
        """Generate a unique blob key."""
        return uuid.uuid4().hex[:16]
