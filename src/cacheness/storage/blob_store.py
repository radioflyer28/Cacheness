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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone

from .backends import MetadataBackend, JsonBackend
from .clear_recovery import ClearRecoveryCoordinator
from .guarded_handler_io import GuardedHandlerIO
from .handlers import HandlerRegistry
from .path_security import encode_physical_name, resolve_managed_locator
from ..metadata import MetadataBackend as CoreMetadataBackend

# Import CacheConfig for proper handler configuration
from ..config import CacheConfig, CompressionConfig

logger = logging.getLogger(__name__)


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
            except Exception:
                self.guarded_handler_io.close()
                raise
        
        logger.debug(f"BlobStore initialized at {self.cache_dir}")
    
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
        if existing is not None:
            # Refuse to overwrite a record whose evidence points outside this
            # store before serializing or publishing a replacement payload.
            self._entry_locator(existing, blob_key, operation="overwrite")
        
        # Get appropriate handler
        handler = self.handlers.get_handler(data)
        
        # Store through the private handler stage and guarded publication seam.
        result = self.guarded_handler_io.put(handler, data, storage_id, self.config)
        
        # Build entry metadata
        # Note: JsonBackend stores custom fields in nested 'metadata' dict
        custom_metadata = dict(metadata or {})
        custom_metadata["actual_path"] = result["actual_path"]
        custom_metadata["storage_format"] = result.get("storage_format", "pickle")
        custom_metadata["compression_codec"] = self.compression
        
        entry_data = {
            "cache_key": blob_key,
            "data_type": handler.data_type,
            "file_size": result.get("file_size", 0),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": custom_metadata,
        }
        
        # Store metadata
        self.backend.put_entry(blob_key, entry_data)
        
        logger.debug(f"Stored blob {blob_key}: {handler.data_type}, {entry_data['file_size']} bytes")
        
        return blob_key
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a blob by key.
        
        Args:
            key: The blob key
            
        Returns:
            The stored data, or None if not found
        """
        entry = self.backend.get_entry(key)
        if entry is None:
            logger.debug(f"Blob not found: {key}")
            return None
        
        actual_path = self._entry_locator(entry, key, operation="get")
        if not self.guarded_handler_io.file_ops.exists(actual_path):
            logger.warning(f"Blob file missing: {actual_path}")
            return None
        
        # Get the handler based on data type
        data_type = entry.get("data_type", "object")
        handler = self.handlers.get_handler_by_type(data_type)
        
        # Build handler metadata by merging entry with nested metadata
        nested_meta = entry.get("metadata", {})
        handler_metadata = {
            **entry,
            **nested_meta,  # Flatten nested metadata to top level
        }
        
        # The handler is intentionally called only on the still-live private
        # snapshot, never on a metadata-controlled managed path.
        with self.guarded_handler_io.open_snapshot(
            actual_path, handler_metadata
        ) as snapshot:
            data = handler.get(snapshot.path, snapshot.metadata)
        
        # Update access time
        self.backend.update_access_time(key)
        
        return data
    
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
    
    def clear(self) -> int:
        """
        Remove every managed payload with recoverable clear tombstones.

        Returns:
            Number of blobs removed
        """
        if self._clear_recovery is None:
            raise ClearRecoveryCoordinator.unsupported_error(self.backend)

        with self._clear_recovery.admission():
            entries = self.backend.list_entries()
            self._preflight_entries(entries, operation="clear")
            mappings = []
            for entry in entries:
                cache_key = entry.get("cache_key", "")
                actual_path = self._entry_locator(entry, cache_key, operation="clear")
                # Every metadata entry participates, even when its payload is
                # already absent, so snapshot and journal cardinalities agree.
                mappings.append((cache_key, actual_path))
            return self._clear_recovery.clear(mappings)

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
