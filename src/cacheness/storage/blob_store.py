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

Features:
- Content-addressable storage (xxhash or SHA-256 based keys)
- Pluggable metadata backends (JSON, SQLite)
- Type-aware serialization via handlers
- Configurable compression
- File integrity verification (xxhash-based file hashes)
- Cryptographic signing (HMAC-SHA256 entry signing)
- Integrity auditing (orphan/dangling/mismatch detection)
- Thread-safe operations

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

    # Verify integrity
    report = store.verify_integrity(repair=True)

    # Delete
    store.delete(blob_id)
"""

import glob
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone

import xxhash

from .backends import MetadataBackend, JsonBackend
from .backends.blob_backends import BlobBackend, FilesystemBlobBackend, get_blob_backend
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
    - Content-addressable storage option (xxhash or SHA-256 based keys)
    - Pluggable metadata backends (JSON, SQLite)
    - Type-aware serialization via config-aware handlers
    - Configurable compression
    - File integrity verification (xxhash-based file hashes)
    - Cryptographic signing (HMAC-SHA256 entry signing)
    - Integrity auditing (orphan/dangling/mismatch detection)
    - Thread-safe operations
    - Rich, queryable metadata

    Attributes:
        cache_dir: Root directory for blob storage
        backend: Metadata backend instance
        handlers: Handler registry for type detection
        signer: Optional CacheEntrySigner for metadata integrity
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = ".blobstore",
        backend: Optional[Union[str, MetadataBackend]] = None,
        compression: str = "lz4",
        compression_level: int = 3,
        content_addressable: bool = False,
        blob_backend: Optional[Union[str, BlobBackend]] = None,
        enable_signing: bool = False,
        signing_key_file: str = "cache_signing_key.bin",
        use_in_memory_key: bool = False,
        config: Optional[CacheConfig] = None,
    ):
        """
        Initialize a BlobStore.

        Args:
            cache_dir: Directory for storing blobs and metadata
            backend: Metadata backend - "json", "sqlite", or a MetadataBackend instance
            compression: Compression codec (lz4, zstd, gzip, blosclz, etc.)
            compression_level: Compression level (1-9)
            content_addressable: If True, use content hash as blob key
            blob_backend: Blob storage backend for file operations (delete, exists).
                Can be a string ('filesystem', 'memory') or a BlobBackend instance.
                Defaults to 'filesystem' if not provided.
            enable_signing: If True, enable HMAC-SHA256 entry signing
            signing_key_file: Name of the signing key file
            use_in_memory_key: If True, use ephemeral in-memory signing key
            config: Optional CacheConfig for handler configuration. If not provided,
                a default config is created from compression parameters.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.compression = compression
        self.compression_level = compression_level
        self.content_addressable = content_addressable

        # Thread safety
        self._lock = threading.RLock()

        # Create or use provided config for handlers
        if config is not None:
            self.config = config
        else:
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

        # Initialize handler registry with config for proper type detection
        self.handlers = HandlerRegistry(self.config)

        # Initialize blob backend for file operations (delete, exists)
        if blob_backend is None or blob_backend == "filesystem":
            self.blob_backend = FilesystemBlobBackend(self.cache_dir, shard_chars=0)
        elif isinstance(blob_backend, str):
            # Use registry to get backend by name
            if blob_backend == "memory":
                self.blob_backend = get_blob_backend(blob_backend)
            else:
                # Pass cache_dir for filesystem-like backends
                self.blob_backend = get_blob_backend(
                    blob_backend, base_dir=self.cache_dir, shard_chars=0
                )
        elif isinstance(blob_backend, BlobBackend):
            self.blob_backend = blob_backend
        else:
            raise ValueError(f"Unknown blob_backend type: {blob_backend}")

        # Initialize entry signer for metadata integrity protection
        self.signer = None
        if enable_signing:
            self._init_signer(signing_key_file, use_in_memory_key)

        logger.debug(f"BlobStore initialized at {self.cache_dir}")

    def _init_signer(
        self,
        signing_key_file: str,
        use_in_memory_key: bool,
    ) -> None:
        """Initialize the cache entry signer."""
        try:
            from ..security import create_cache_signer

            self.signer = create_cache_signer(
                cache_dir=self.cache_dir,
                key_file=signing_key_file,
                use_in_memory_key=use_in_memory_key,
            )
            info = self.signer.get_field_info()
            logger.info(
                f"Entry signing enabled (v{info['signature_version']}, "
                f"{len(info['signed_fields'])} fields)"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize entry signer: {e}")
            self.signer = None

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
        with self._lock:
            # Generate key
            if self.content_addressable:
                # Use xxhash content hash as key
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

            # Calculate file hash for integrity verification
            actual_path = Path(str(result.get("actual_path", base_path)))
            file_hash = self._calculate_file_hash(actual_path)

            # Build entry metadata
            # Note: JsonBackend stores custom fields in nested 'metadata' dict
            # We store file_hash and entry_signature in nested metadata too so
            # JsonBackend preserves them (it only keeps specific top-level fields).
            custom_metadata = metadata or {}
            custom_metadata["actual_path"] = str(actual_path)
            custom_metadata["storage_format"] = result.get("storage_format", "pickle")
            custom_metadata["compression_codec"] = self.compression
            if file_hash:
                custom_metadata["file_hash"] = file_hash

            entry_data = {
                "cache_key": blob_key,
                "data_type": handler.data_type,
                "file_size": result.get("file_size", 0),
                "file_hash": file_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": custom_metadata,
            }

            # Sign entry if signer is available
            if self.signer is not None:
                # Build signable data (flatten for signing)
                signable = {**entry_data, **custom_metadata}
                signature = self.signer.sign_entry(signable)
                entry_data["entry_signature"] = signature
                # Also store in nested metadata so JsonBackend preserves it
                custom_metadata["entry_signature"] = signature

            # Store metadata
            self.backend.put_entry(blob_key, entry_data)

            logger.debug(
                f"Stored blob {blob_key}: {handler.data_type}, "
                f"{entry_data['file_size']} bytes"
            )

            return blob_key

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a blob by key.

        Verifies entry signature if signing is enabled.

        Args:
            key: The blob key

        Returns:
            The stored data, or None if not found
        """
        with self._lock:
            entry = self.backend.get_entry(key)
            if entry is None:
                logger.debug(f"Blob not found: {key}")
                return None

            # Verify entry signature if signer is available
            if self.signer is not None:
                nested_meta = entry.get("metadata", {})
                stored_signature = entry.get("entry_signature") or nested_meta.get(
                    "entry_signature"
                )
                if stored_signature:
                    # Reconstruct signable data matching what was signed on put()
                    signable = {**entry, **nested_meta, "cache_key": key}
                    if not self.signer.verify_entry(signable, stored_signature):
                        logger.warning(f"Signature verification failed for blob {key}")
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

            if not self.blob_backend.exists(str(actual_path)):
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            entry = self.backend.get_entry(key)
            if entry is None:
                return False

            # Delete the file via blob backend
            nested_meta = entry.get("metadata", {})
            actual_path_str = entry.get("actual_path") or nested_meta.get("actual_path")
            actual_path = Path(
                actual_path_str if actual_path_str else self.cache_dir / key
            )
            self.blob_backend.delete_blob(str(actual_path))

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
        with self._lock:
            entry = self.backend.get_entry(key)
            if entry is None:
                return False

            # Also verify the file exists via blob backend
            nested_meta = entry.get("metadata", {})
            actual_path_str = entry.get("actual_path") or nested_meta.get("actual_path")
            actual_path = Path(
                actual_path_str if actual_path_str else self.cache_dir / key
            )
            return self.blob_backend.exists(str(actual_path))

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
        with self._lock:
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
        Remove all blobs and their files.

        Returns:
            Number of metadata entries removed
        """
        with self._lock:
            files_removed = self._clear_blob_files()
            count = self.backend.clear_all()
            logger.debug(
                f"Cleared {count} entries and removed {files_removed} blob files"
            )
            return count

    def close(self):
        """Close the blob store and release resources."""
        with self._lock:
            self.backend.close()
            self.blob_backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ── Integrity verification ────────────────────────────────────────

    def verify_integrity(
        self, repair: bool = False, verify_hashes: bool = False
    ) -> Dict[str, Any]:
        """
        Verify blob store integrity by cross-checking blob files and metadata.

        Detects:
        - Orphaned blobs: files in cache_dir with no metadata entry
        - Dangling metadata: entries pointing to missing blob files
        - Size mismatches: metadata file_size != actual file size on disk
        - Hash mismatches: metadata file_hash != actual file hash (if verify_hashes)

        Args:
            repair: If True, delete orphaned blobs and remove dangling entries.
            verify_hashes: If True, also verify file hashes (slower but catches
                corruption).

        Returns:
            Dict with keys: orphaned_blobs, dangling_entries, size_mismatches,
            hash_mismatches (if verify_hashes), repaired (if repair).
        """
        with self._lock:
            # 1. Inventory all blob files in cache_dir
            blob_extensions = ["pkl", "npz", "b2nd", "b2tr", "parquet"]
            pickle_codecs = ["lz4", "zstd", "gzip", "zst", "gz", "bz2", "xz"]

            blob_files: set[str] = set()
            patterns = [str(self.cache_dir / f"*.{ext}") for ext in blob_extensions]
            for codec in pickle_codecs:
                patterns.append(str(self.cache_dir / f"*.pkl.{codec}"))
            patterns.append(str(self.cache_dir / "*.pkl.*"))

            for pattern in patterns:
                for file_path in glob.glob(pattern):
                    blob_files.add(os.path.normpath(file_path))

            entry_paths: dict[str, dict] = {}
            for entry in self.backend.iter_entry_summaries():
                actual_path = entry.get("actual_path")
                if actual_path:
                    norm_path = os.path.normpath(actual_path)
                    entry_paths[norm_path] = {
                        "cache_key": entry.get("cache_key", ""),
                        "file_size": entry.get("file_size"),
                        "file_hash": entry.get("file_hash"),
                    }

            # 3. Find orphaned blobs (files with no metadata entry)
            known_paths = set(entry_paths.keys())
            orphaned_blobs = sorted(blob_files - known_paths)

            # 4. Find dangling metadata (entries pointing to missing files)
            dangling_entries = []
            for path, info in entry_paths.items():
                if not os.path.exists(path):
                    dangling_entries.append(
                        {
                            "cache_key": info["cache_key"],
                            "expected_path": path,
                        }
                    )

            # 5. Check size mismatches
            size_mismatches = []
            for path, info in entry_paths.items():
                if os.path.exists(path) and info["file_size"] is not None:
                    actual_size = os.path.getsize(path)
                    if actual_size != info["file_size"]:
                        size_mismatches.append(
                            {
                                "cache_key": info["cache_key"],
                                "path": path,
                                "expected_size": info["file_size"],
                                "actual_size": actual_size,
                            }
                        )

            # 6. Check hash mismatches (optional, expensive)
            hash_mismatches = []
            if verify_hashes:
                for path, info in entry_paths.items():
                    if os.path.exists(path) and info.get("file_hash"):
                        current_hash = self._calculate_file_hash(Path(path))
                        if current_hash != info["file_hash"]:
                            hash_mismatches.append(
                                {
                                    "cache_key": info["cache_key"],
                                    "path": path,
                                    "expected_hash": info["file_hash"],
                                    "actual_hash": current_hash,
                                }
                            )

            # 7. Repair if requested
            repaired = {"orphans_deleted": 0, "dangling_removed": 0}
            if repair:
                for blob_path in orphaned_blobs:
                    try:
                        os.remove(blob_path)
                        repaired["orphans_deleted"] += 1
                    except OSError as e:
                        logger.warning(
                            f"Failed to remove orphaned blob {blob_path}: {e}"
                        )

                for entry in dangling_entries:
                    try:
                        self.backend.remove_entry(entry["cache_key"])
                        repaired["dangling_removed"] += 1
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove dangling entry {entry['cache_key']}: {e}"
                        )

            report: Dict[str, Any] = {
                "orphaned_blobs": orphaned_blobs,
                "dangling_entries": dangling_entries,
                "size_mismatches": size_mismatches,
            }
            if verify_hashes:
                report["hash_mismatches"] = hash_mismatches
            if repair:
                report["repaired"] = repaired

            total_issues = (
                len(orphaned_blobs)
                + len(dangling_entries)
                + len(size_mismatches)
                + len(hash_mismatches)
            )
            if total_issues == 0:
                logger.info("Blob store integrity check passed — no issues found")
            else:
                logger.warning(
                    f"Blob store integrity check found {total_issues} issue(s): "
                    f"{len(orphaned_blobs)} orphaned, "
                    f"{len(dangling_entries)} dangling, "
                    f"{len(size_mismatches)} size mismatches"
                    + (
                        f", {len(hash_mismatches)} hash mismatches"
                        if verify_hashes
                        else ""
                    )
                )

            return report

    # ── Low-level composition API ─────────────────────────────────────
    # These methods are used by UnifiedCache to delegate storage operations
    # without going through the full BlobStore.put/get pipeline.

    def _write_blob(
        self,
        data: Any,
        base_path: Path,
        config: Optional["CacheConfig"] = None,
        compute_hash: bool = True,
    ) -> tuple:
        """
        Low-level: serialize data to disk via handler.

        Does NOT acquire the lock — caller is responsible for synchronization.
        Does NOT write metadata — caller handles metadata storage.

        Args:
            data: The data to serialize
            base_path: Base file path (handler adds extension)
            config: Optional CacheConfig override
            compute_hash: Whether to compute xxhash file hash

        Returns:
            Tuple of (handler, result_dict, file_hash_or_None)
        """
        handler = self.handlers.get_handler(data)
        result = handler.put(data, base_path, config or self.config)
        file_hash = None
        if compute_hash:
            actual_path = Path(str(result.get("actual_path", base_path)))
            file_hash = self._calculate_file_hash(actual_path)
        return handler, result, file_hash

    def _read_blob(
        self,
        path: Path,
        data_type: str,
        handler_metadata: Dict[str, Any],
    ) -> Any:
        """
        Low-level: deserialize data from disk via handler.

        Does NOT acquire the lock — caller is responsible for synchronization.
        Does NOT check metadata or verify signatures.

        Args:
            path: Path to the blob file
            data_type: Handler data type identifier (e.g. "dataframe", "array")
            handler_metadata: Metadata dict passed to handler.get()

        Returns:
            Deserialized data object
        """
        handler = self.handlers.get_handler_by_type(data_type)
        if handler is None:
            return read_file(path)
        return handler.get(path, handler_metadata)

    def _clear_blob_files(self) -> int:
        """
        Delete all blob files from the cache directory.

        Does NOT acquire the lock — caller is responsible for synchronization.

        Returns:
            Number of blob files deleted
        """
        blob_extensions = ["pkl", "npz", "b2nd", "b2tr", "parquet"]
        pickle_codecs = ["lz4", "zstd", "gzip", "zst", "gz", "bz2", "xz"]

        patterns = [str(self.cache_dir / f"*.{ext}") for ext in blob_extensions]
        for codec in pickle_codecs:
            patterns.append(str(self.cache_dir / f"*.pkl.{codec}"))
        patterns.append(str(self.cache_dir / "*.pkl.*"))

        processed: set[str] = set()
        for pattern in patterns:
            for file_path in glob.glob(pattern):
                if file_path not in processed:
                    try:
                        os.remove(file_path)
                        processed.add(file_path)
                    except OSError as e:
                        logger.warning(f"Failed to remove blob file {file_path}: {e}")

        return len(processed)

    # ── Private helper methods ────────────────────────────────────────

    def _calculate_file_hash(self, file_path: Path) -> Optional[str]:
        """
        Calculate XXH3_64 hash of a blob file for integrity verification.

        Args:
            file_path: Path to the blob file

        Returns:
            Hex string of the file hash, or None if file doesn't exist or error
        """
        try:
            if not file_path.exists():
                return None

            hasher = xxhash.xxh3_64()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate hash for {file_path}: {e}")
            return None

    def _compute_content_hash(self, data: Any) -> str:
        """Compute a content-based hash for the data using xxhash."""
        import pickle

        try:
            serialized = pickle.dumps(data)
        except Exception:
            # Fall back to repr for non-pickleable objects
            serialized = repr(data).encode()
        return xxhash.xxh3_64(serialized).hexdigest()[:16]

    def _sanitize_key(self, key: str) -> str:
        """Sanitize a user-provided key."""
        # Remove problematic characters
        safe_key = "".join(c for c in key if c.isalnum() or c in "-_.")
        return safe_key[:64] or self._generate_unique_key()

    def _generate_unique_key(self) -> str:
        """Generate a unique blob key."""
        import uuid

        return uuid.uuid4().hex[:16]
