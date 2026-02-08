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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone

from .backends import MetadataBackend, JsonBackend
from .handlers import HandlerRegistry
from .compression import read_file

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
        elif isinstance(backend, MetadataBackend):
            self.backend = backend
        else:
            raise ValueError(f"Unknown backend type: {backend}")

        # Initialize handler registry
        self.handlers = HandlerRegistry()

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
        elif key:
            blob_key = self._sanitize_key(key)
        else:
            blob_key = self._generate_unique_key()

        # Get appropriate handler
        handler = self.handlers.get_handler(data)

        # Determine file path
        base_path = self.cache_dir / blob_key

        # Store the data using the handler
        result = handler.put(data, base_path, self.config)

        # Build entry metadata
        # Note: JsonBackend stores custom fields in nested 'metadata' dict
        custom_metadata = metadata or {}
        custom_metadata["actual_path"] = str(result.get("actual_path", base_path))
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

        logger.debug(
            f"Stored blob {blob_key}: {handler.data_type}, {entry_data['file_size']} bytes"
        )

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

        # Get the file path - may be in top-level or nested metadata
        nested_meta = entry.get("metadata", {})
        actual_path_str = entry.get("actual_path") or nested_meta.get("actual_path")

        if actual_path_str:
            actual_path = Path(actual_path_str)
        else:
            # Fallback: try common extensions
            for ext in [".pkl", ".b2nd", ".parquet", ".npz", ""]:
                candidate = self.cache_dir / f"{key}{ext}"
                if candidate.exists():
                    actual_path = candidate
                    break
            else:
                actual_path = self.cache_dir / key

        if not actual_path.exists():
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

        if handler is None:
            # Fall back to generic read
            return read_file(actual_path)

        # Read using handler
        data = handler.get(actual_path, handler_metadata)

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
        return self.backend.get_entry(key)

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

        # Get or create nested metadata dict
        nested_meta = existing.get("metadata", {})
        if not isinstance(nested_meta, dict):
            nested_meta = {}

        # Merge user metadata into nested dict
        nested_meta.update(metadata)

        # Update the entry
        updated = {**existing, "metadata": nested_meta}

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

        # Delete the file
        actual_path = Path(entry.get("actual_path", self.cache_dir / key))
        if actual_path.exists():
            actual_path.unlink()

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

        # Also verify the file exists
        actual_path = Path(entry.get("actual_path", self.cache_dir / key))
        return actual_path.exists()

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
        Remove all blobs.

        Returns:
            Number of blobs removed
        """
        return self.backend.clear_all()

    def close(self):
        """Close the blob store and release resources."""
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

    def _sanitize_key(self, key: str) -> str:
        """Sanitize a user-provided key."""
        # Remove problematic characters
        safe_key = "".join(c for c in key if c.isalnum() or c in "-_.")
        return safe_key[:64] or self._generate_unique_key()

    def _generate_unique_key(self) -> str:
        """Generate a unique blob key."""
        import uuid

        return uuid.uuid4().hex[:16]
