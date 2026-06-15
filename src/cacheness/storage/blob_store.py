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

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast
from datetime import datetime, timezone

import xxhash

from .backends import MetadataBackend, JsonBackend
from .backends.blob_backends import BlobBackend, FilesystemBlobBackend, get_blob_backend
from .handlers import HandlerRegistry
from .compression import read_file
from .paths import resolve_actual_path, to_relative_path

# Import CacheConfig for proper handler configuration
from ..config import CacheConfig, CompressionConfig
from ..interfaces import BlobReadContext, WriteBlobResult, IntegrityReport
from ..signing_fields import (
    extract_legacy_blobstore_signable_fields,
    extract_signable_fields,
)

if TYPE_CHECKING:
    from ..interfaces import RotationResult

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
        key_fallback_policy: str = "warn",
        config: Optional[CacheConfig] = None,
        namespace: str = "default",
        use_hkdf_derivation: bool = True,
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
            key_fallback_policy: What to do when key file cannot be written.
                'raise' = raise CacheSecurityError
                'warn' = log WARNING + use in-memory key
                'fallback' = silently use in-memory key
            config: Optional CacheConfig for handler configuration. If not provided,
                a default config is created from compression parameters.
            namespace: Namespace for blob isolation. Non-default namespaces store
                blobs in a subdirectory. Defaults to "default".
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.compression = compression
        self.compression_level = compression_level
        self.content_addressable = content_addressable
        self._namespace = namespace

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
            self.blob_backend = FilesystemBlobBackend(
                self.cache_dir, shard_chars=0, namespace=self._namespace
            )
        elif isinstance(blob_backend, str):
            # Use registry to get backend by name
            if blob_backend == "memory":
                self.blob_backend = get_blob_backend(blob_backend)
            else:
                # Pass cache_dir for filesystem-like backends
                self.blob_backend = get_blob_backend(
                    blob_backend,
                    base_dir=self.cache_dir,
                    shard_chars=0,
                    namespace=self._namespace,
                )
        elif isinstance(blob_backend, BlobBackend):
            self.blob_backend = blob_backend
        else:
            raise ValueError(f"Unknown blob_backend type: {blob_backend}")

        # Initialize entry signer for metadata integrity protection
        self.signer = None
        if enable_signing:
            self._init_signer(
                signing_key_file,
                use_in_memory_key,
                key_fallback_policy,
                use_hkdf_derivation=use_hkdf_derivation,
                minimum_signature_version=(
                    self.config.security.minimum_signature_version
                ),
            )

        # Initialize encryption (requires config with SecurityConfig)
        self._encryption_key: Optional[bytes] = None
        if config is not None and config.security.enable_content_encryption:
            self._init_encryptor(config.security, namespace)

        logger.debug(f"BlobStore initialized at {self.cache_dir}")

    # ── Path normalization helpers ────────────────────────────────────

    def _to_relative_path(self, path_str: str) -> str:
        """Convert an absolute path to a cache-dir-relative, forward-slash path.

        Delegates to :func:`cacheness.storage.paths.to_relative_path`.
        """
        return to_relative_path(path_str, self.cache_dir)

    def _resolve_actual_path(self, actual_path_str: str) -> str:
        """Resolve a stored ``actual_path`` to a backend-usable path string.

        Returns a *string* suitable for ``blob_backend.exists()``,
        ``blob_backend.delete_blob()``, etc.  Delegates to
        :func:`cacheness.storage.paths.resolve_actual_path`.
        """
        return str(resolve_actual_path(actual_path_str, self.cache_dir))

    def _init_signer(
        self,
        signing_key_file: str,
        use_in_memory_key: bool,
        key_fallback_policy: str = "warn",
        use_hkdf_derivation: bool = True,
        minimum_signature_version: int = 1,
    ) -> None:
        """Initialize the cache entry signer."""
        try:
            from ..security import create_cache_signer

            self.signer = create_cache_signer(
                cache_dir=self.cache_dir,
                key_file=signing_key_file,
                use_in_memory_key=use_in_memory_key,
                key_fallback_policy=key_fallback_policy,
                namespace_id=self._namespace,
                use_hkdf_derivation=use_hkdf_derivation,
                minimum_signature_version=minimum_signature_version,
            )
            info = self.signer.get_field_info()
            logger.info(
                f"Entry signing enabled (v{info['signature_version']}, "
                f"{len(info['signed_fields'])} fields)"
            )
        except Exception as e:  # intentionally broad — signer init failure is non-fatal
            logger.warning(f"Failed to initialize entry signer: {e}")
            self.signer = None

    def _init_encryptor(
        self,
        security_config: "Any",
        namespace: str,
    ) -> None:
        """Initialize encryption key from security config.

        Reads the master key from the encryption key file (which defaults
        to the signing key file) and derives a per-namespace AES-256 key.
        """
        from ..encryption import derive_encryption_key

        key_path = self.cache_dir / security_config.encryption_key_file
        if not key_path.is_file():
            logger.warning(
                f"Encryption key file not found: {key_path} — encryption disabled"
            )
            return
        master_key = key_path.read_bytes()
        if len(master_key) != 32:
            logger.warning(
                f"Encryption key file has invalid length ({len(master_key)} bytes, "
                f"expected 32) — encryption disabled"
            )
            return
        self._encryption_key = derive_encryption_key(master_key, namespace)
        logger.info("Content encryption enabled (AES-256-GCM)")

    def rotate_key(self, new_key_file: "str | Path") -> "RotationResult":
        """Rotate the signing key and re-sign all blob entries.

        Loads a new 32-byte signing key, replaces the current key file,
        creates a fresh signer, then iterates every entry to re-sign.

        Best-effort: individual entry failures are recorded in
        :attr:`RotationResult.failures`, not raised.

        Args:
            new_key_file: Path to a file containing exactly 32 random bytes.

        Returns:
            :class:`RotationResult` with re-sign counts.

        Raises:
            CacheSecurityError: If signing is not enabled, the file does
                not exist, or the key is not 32 bytes.
        """
        from ..error_handling import CacheSecurityError
        from ..interfaces import RotationResult
        import copy

        from ..encryption import decrypt_blob, derive_encryption_key, encrypt_blob
        from ..security import create_cache_signer, write_staged_key_file

        if self.signer is None:
            raise CacheSecurityError("Entry signing is not enabled — cannot rotate key")

        new_key_path = Path(new_key_file)
        if not new_key_path.is_file():
            raise CacheSecurityError(f"Key file does not exist: {new_key_path}")
        new_key_bytes = new_key_path.read_bytes()
        if len(new_key_bytes) != 32:
            raise CacheSecurityError(
                f"Invalid key length ({len(new_key_bytes)} bytes) — expected 32"
            )

        with self._lock:
            dest = Path(self.signer.key_file_path)
            staged_key_path = write_staged_key_file(dest, new_key_bytes)
            new_signer = create_cache_signer(
                cache_dir=staged_key_path.parent,
                key_file=staged_key_path.name,
                use_in_memory_key=False,
                key_fallback_policy=self.signer.key_fallback_policy,
                namespace_id=self._namespace,
                use_hkdf_derivation=self.signer.use_hkdf_derivation,
                minimum_signature_version=self.signer.minimum_signature_version,
            )

            result = RotationResult()
            entries = self.backend.iter_entry_summaries()
            result.total = len(entries)
            old_entries: dict[str, dict[str, Any]] = {}
            old_blobs: dict[Path, bytes] = {}
            current_cache_key: str | None = None
            rotation_succeeded = False
            old_enc_key: bytes | None = None
            new_enc_key: bytes | None = None

            def rollback_rotation() -> None:
                for blob_path, ciphertext in old_blobs.items():
                    try:
                        blob_path.write_bytes(ciphertext)
                    except OSError as exc:
                        logger.warning(
                            f"Failed to restore blob {blob_path} after rotation failure: {exc}"
                        )
                for cache_key, old_entry in old_entries.items():
                    try:
                        self.backend.put_entry(cache_key, copy.deepcopy(old_entry))
                    except (
                        Exception
                    ) as exc:  # intentionally broad — best-effort rollback
                        logger.warning(
                            f"Failed to restore metadata for {cache_key} after rotation failure: {exc}"
                        )
                for rotating_path in dest.parent.rglob("*.rotating"):
                    try:
                        rotating_path.unlink()
                    except OSError:
                        pass
                try:
                    staged_key_path.unlink(missing_ok=True)
                except OSError:
                    pass

            try:
                if self._encryption_key is not None:
                    old_enc_key = self._encryption_key
                    new_enc_key = derive_encryption_key(new_key_bytes, self._namespace)

                for entry in entries:
                    current_cache_key = entry["cache_key"]
                    full_entry = self.backend.get_entry(current_cache_key)
                    if full_entry is None:
                        result.skipped += 1
                        continue

                    old_entries[current_cache_key] = copy.deepcopy(full_entry)
                    nested_meta = full_entry.get("metadata", {})
                    stored_signature = full_entry.get(
                        "entry_signature"
                    ) or nested_meta.get("entry_signature")
                    if stored_signature:
                        if not self._verify_entry_signature(
                            current_cache_key,
                            full_entry,
                            nested_meta,
                            stored_signature,
                        ):
                            raise CacheSecurityError(
                                f"Existing blob signature did not verify: {current_cache_key}"
                            )
                    elif not self.config.security.allow_unsigned_entries:
                        raise CacheSecurityError(
                            f"Unsigned blob entry cannot be rotated: {current_cache_key}"
                        )

                    if new_enc_key is not None and nested_meta.get(
                        "encryption_algorithm"
                    ):
                        assert old_enc_key is not None
                        actual_path_str = nested_meta.get("actual_path")
                        if not actual_path_str:
                            result.skipped += 1
                            continue
                        resolved = self._resolve_actual_path(actual_path_str)
                        blob_path = Path(resolved)
                        if not blob_path.is_file():
                            result.skipped += 1
                            continue

                        old_iv = bytes.fromhex(nested_meta["encryption_iv"])
                        ciphertext = blob_path.read_bytes()
                        old_blobs.setdefault(blob_path, ciphertext)
                        plaintext = decrypt_blob(ciphertext, old_enc_key, old_iv)
                        new_ciphertext, new_iv, _ = encrypt_blob(plaintext, new_enc_key)
                        rotating_path = blob_path.with_name(
                            f"{blob_path.name}.rotating"
                        )
                        rotating_path.write_bytes(new_ciphertext)
                        os.replace(rotating_path, blob_path)

                        nested_meta["encryption_iv"] = new_iv.hex()
                        file_hash = self._calculate_file_hash(blob_path)
                        if file_hash:
                            nested_meta["file_hash"] = file_hash
                            full_entry["file_hash"] = file_hash
                        full_entry["file_size"] = len(new_ciphertext)
                        result.re_encrypted += 1

                    signable = extract_signable_fields(
                        current_cache_key, full_entry, nested_meta
                    )
                    new_sig = new_signer.sign_entry(signable)
                    full_entry["entry_signature"] = new_sig
                    nested_meta["entry_signature"] = new_sig
                    full_entry["metadata"] = nested_meta
                    self.backend.put_entry(current_cache_key, full_entry)
                    result.re_signed += 1

                os.replace(staged_key_path, dest)
                new_signer.key_file_path = dest
                self.signer = new_signer
                if new_enc_key is not None:
                    self._encryption_key = new_enc_key
                rotation_succeeded = True
            except (
                Exception
            ) as e:  # intentionally broad — rollback keeps old key active
                result.failed += 1
                result.failures.append(
                    {"cache_key": current_cache_key or "<rotation>", "error": str(e)}
                )
                logger.warning(
                    "BlobStore key rotation failed before publishing new key; active key retained: %s",
                    e,
                )
                rollback_rotation()

            logger.info(
                f"BlobStore key rotation complete: {result.re_signed}/{result.total} "
                f"entries re-signed, {result.failed} failed, "
                f"{result.skipped} skipped"
            )

            if not rotation_succeeded:
                return result

        return result

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

            # Store the data using the handler (writes to local filesystem)
            result = handler.put(data, base_path, self.config)

            # Encrypt blob content if encryption is enabled
            # Encryption happens AFTER handler compression, BEFORE blob backend write
            encryption_meta: Dict[str, str] = {}
            handler_path = Path(result.actual_path)
            if self._encryption_key is not None:
                from ..encryption import encrypt_blob

                plaintext = handler_path.read_bytes()
                ciphertext, iv, algo = encrypt_blob(plaintext, self._encryption_key)
                handler_path.write_bytes(ciphertext)
                result.file_size = len(ciphertext)
                encryption_meta["encryption_algorithm"] = algo.decode()
                encryption_meta["encryption_iv"] = iv.hex()

            # Persist through blob_backend (may rename, upload, etc.)
            final_path = self.blob_backend.write_blob_from_path(
                str(handler_path), handler_path.name
            )
            # Capture any backend-specific write metadata (e.g. s3_etag)
            write_meta = self.blob_backend.get_write_metadata()
            if write_meta:
                result.extra.update(write_meta)

            actual_path = Path(final_path) if "://" not in final_path else None

            # Calculate file hash for integrity verification
            if actual_path is not None:
                file_hash = self._calculate_file_hash(actual_path)
            else:
                file_hash = self._calculate_blob_hash(final_path)

            # Build entry metadata
            # Note: JsonBackend stores custom fields in nested 'metadata' dict
            # We store file_hash and entry_signature in nested metadata too so
            # JsonBackend preserves them (it only keeps specific top-level fields).
            custom_metadata = metadata or {}
            custom_metadata["actual_path"] = self._to_relative_path(final_path)
            custom_metadata["storage_format"] = result.storage_format
            custom_metadata["compression_codec"] = self.compression
            if file_hash:
                custom_metadata["file_hash"] = file_hash
            # Add encryption metadata if blob was encrypted
            custom_metadata.update(encryption_meta)

            entry_data = {
                "cache_key": blob_key,
                "data_type": handler.data_type,
                "file_size": result.file_size,
                "file_hash": file_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": custom_metadata,
            }

            # Sign entry if signer is available
            if self.signer is not None:
                signable = extract_signable_fields(
                    blob_key, entry_data, custom_metadata
                )
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

    def _verify_entry_signature(
        self,
        cache_key: str,
        entry: Dict[str, Any],
        metadata: Dict[str, Any],
        stored_signature: str,
    ) -> bool:
        """Verify a BlobStore entry with canonical fields, then legacy fallback."""
        if self.signer is None:
            return True

        canonical = extract_signable_fields(cache_key, entry, metadata)
        if self.signer.verify_entry(canonical, stored_signature):
            return True

        legacy = extract_legacy_blobstore_signable_fields(cache_key, entry, metadata)
        if legacy != canonical and self.signer.verify_entry(legacy, stored_signature):
            logger.debug(
                "Verified legacy flattened BlobStore signature for %s", cache_key
            )
            return True

        return False

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
                    if not self._verify_entry_signature(
                        key, entry, nested_meta, stored_signature
                    ):
                        logger.warning(f"Signature verification failed for blob {key}")
                        return None

            # Get the file path - may be in top-level or nested metadata
            nested_meta = entry.get("metadata", {})
            actual_path_str = entry.get("actual_path") or nested_meta.get("actual_path")

            if actual_path_str:
                resolved = self._resolve_actual_path(actual_path_str)
                backend_path = str(resolved)
                if "://" in backend_path:
                    actual_path = self.cache_dir / Path(actual_path_str).name
                else:
                    actual_path = Path(resolved)
            else:
                # Fallback: try common extensions
                for ext in [".pkl", ".b2nd", ".parquet", ".npz", ""]:
                    candidate = self.cache_dir / f"{key}{ext}"
                    if candidate.exists():
                        actual_path = candidate
                        break
                else:
                    actual_path = self.cache_dir / key
                backend_path = str(actual_path)

            if not self.blob_backend.exists(backend_path):
                logger.warning(f"Blob file missing: {actual_path}")
                return None

            # Get the handler based on data type
            data_type = entry.get("data_type", "object")
            handler = self.handlers.get_handler_by_type(data_type)

            # Build handler metadata by merging entry with nested metadata
            nested_meta = entry.get("metadata", {})
            handler_metadata = cast(
                BlobReadContext,
                {
                    **entry,
                    **nested_meta,  # Flatten nested metadata to top level
                },
            )

            if nested_meta.get("encryption_algorithm"):
                data = self._read_encrypted_blob(
                    backend_path,
                    actual_path,
                    data_type,
                    handler_metadata,
                    missing_key_message=(
                        f"Encrypted entry {key} but no encryption key configured"
                    ),
                )
            elif handler is None:
                data = read_file(actual_path)
            else:
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

    def put_file(
        self,
        file_path: str | Path,
        *,
        key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        move: bool = False,
    ) -> str:
        """Store an arbitrary file as a blob.

        Reads the file into memory as raw bytes and delegates to
        :meth:`put`.  File metadata (original filename, MIME type,
        original size) is merged into the *metadata* dict automatically.

        Args:
            file_path: Path to the source file.
            key: Optional explicit blob key.
            metadata: Optional metadata dict (file metadata is merged in).
            move: If ``True``, delete the source file after a successful
                store (move-in semantics).  Defaults to ``False`` (copy-in).

        Returns:
            The blob key.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            IsADirectoryError: If *file_path* is a directory.
        """
        import mimetypes

        src = Path(file_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file does not exist: {src}")
        if src.is_dir():
            raise IsADirectoryError(f"Expected a file, got a directory: {src}")

        data = src.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(src))

        file_meta: Dict[str, Any] = {
            "original_filename": src.name,
            "mime_type": mime_type or "application/octet-stream",
            "original_size": len(data),
        }
        merged = {**file_meta, **(metadata or {})}  # user metadata wins
        blob_key = self.put(data, key=key, metadata=merged)

        if move:
            src.unlink()

        return blob_key

    def get_file(
        self,
        key: str,
        *,
        dest: str | Path | None = None,
        move: bool = False,
        overwrite: bool = True,
    ) -> Optional[bytes | Path]:
        """Retrieve a cached file blob, optionally writing it to disk.

        Args:
            key: The blob key.
            dest: Optional destination path.  Parent directories are
                created automatically.  If *dest* is a directory, the
                original filename from metadata is used (falls back to
                ``<key>.bin``).
            move: If ``True``, delete the cache entry after a successful
                write to *dest* (move-out semantics).  Requires *dest*
                to be set — raises ``ValueError`` otherwise.  Defaults
                to ``False`` (copy-out).
            overwrite: If ``False``, raise ``FileExistsError`` when
                *dest* already exists on disk.  Defaults to ``True``
                (silently overwrite).

        Returns:
            * ``bytes`` when *dest* is ``None`` and entry exists.
            * ``pathlib.Path`` when *dest* is given and entry exists.
            * ``None`` on miss.

        Raises:
            ValueError: If *move* is ``True`` but *dest* is ``None``.
            FileExistsError: If *overwrite* is ``False`` and *dest*
                already exists.
        """
        if move and dest is None:
            raise ValueError(
                "move=True requires dest to be set. "
                "Without a destination path, use get() + delete() instead."
            )

        data = self.get(key)
        if data is None:
            return None

        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"Expected bytes from blob store, got {type(data).__name__}. "
                "get_file() should only be used with blobs stored via put_file()."
            )
        raw = bytes(data) if not isinstance(data, bytes) else data

        if dest is None:
            return raw

        dest_path = Path(dest)
        if dest_path.is_dir():
            entry = self.backend.get_entry(key)
            name = None
            if entry:
                nested = entry.get("metadata", {})
                name = nested.get("original_filename")
            dest_path = dest_path / (name or f"{key}.bin")

        if not overwrite and dest_path.exists():
            raise FileExistsError(
                f"Destination already exists: {dest_path}. "
                "Pass overwrite=True to overwrite."
            )

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(raw)

        if move:
            self.delete(key)

        return dest_path

    def delete(self, key: str) -> bool:
        """
        Delete a blob and its metadata.

        Uses metadata-first ordering: removes the metadata entry before
        deleting the blob file.  This ensures a crash between the two steps
        leaves an orphaned blob (harmless) rather than a dangling metadata
        pointer (dangerous).

        Args:
            key: The blob key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            entry = self.backend.get_entry(key)
            if entry is None:
                return False

            # Resolve blob path BEFORE removing metadata (need entry data)
            nested_meta = entry.get("metadata", {})
            actual_path_str = entry.get("actual_path") or nested_meta.get("actual_path")
            resolved = (
                self._resolve_actual_path(actual_path_str)
                if actual_path_str
                else str(self.cache_dir / key)
            )

            # Remove metadata first — crash here leaves entry intact (safe)
            self.backend.remove_entry(key)

            # Delete blob second — crash here leaves orphaned blob (harmless,
            # cleaned by verify_integrity)
            try:
                self.blob_backend.delete_blob(resolved)
            except Exception as exc:  # intentionally broad — orphan cleanup
                logger.warning(
                    f"Failed to delete blob file for {key} at {resolved}: {exc}. "
                    f"Orphaned blob will be cleaned by verify_integrity."
                )

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
            resolved = (
                self._resolve_actual_path(actual_path_str)
                if actual_path_str
                else str(self.cache_dir / key)
            )
            return self.blob_backend.exists(resolved)

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
                    nested_metadata = entry.get("metadata", {})
                    metadata_dict = entry.get("metadata_dict")
                    if metadata_dict is None and isinstance(nested_metadata, dict):
                        metadata_dict = nested_metadata.get("metadata_dict")
                    for field, value in metadata_filter.items():
                        field_value = entry.get(field)
                        if field_value is None and isinstance(nested_metadata, dict):
                            field_value = nested_metadata.get(field)
                        if field_value is None and isinstance(metadata_dict, dict):
                            field_value = metadata_dict.get(field)
                        if field_value != value:
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
        self,
        repair: bool = False,
        verify_hashes: bool = True,
        verify_signatures: bool = False,
    ) -> IntegrityReport:
        """
        Verify blob store integrity by cross-checking blob files and metadata.

        Detects:
        - Orphaned blobs: blobs in storage with no metadata entry
        - Dangling metadata: entries pointing to missing blobs
        - Size mismatches: metadata file_size != actual size in storage
        - Hash mismatches: metadata file_hash != actual file hash (if verify_hashes)
        - Signature failures: entries with invalid or missing HMAC signatures
          (if verify_signatures)

        Works with any blob backend (filesystem, S3, in-memory, etc.).

        Args:
            repair: If True, delete orphaned blobs and remove dangling entries.
            verify_hashes: If True, also verify file hashes (slower but catches
                corruption).
            verify_signatures: If True, verify HMAC signatures on all entries
                that have them. Requires a signer to be configured.

        Returns:
            Dict with keys: orphaned_blobs, dangling_entries, size_mismatches,
            hash_mismatches (if verify_hashes), signature_failures
            (if verify_signatures), repaired (if repair).
        """
        with self._lock:
            # 1. Inventory all blobs in storage (delegates to blob_backend
            #    which handles sharding / recursive listing automatically)
            blob_files: set[str] = set(self.blob_backend.list_blobs())

            # 2. Collect metadata entry paths
            entry_paths: dict[str, dict] = {}
            for entry in self.backend.iter_entry_summaries():
                actual_path = entry.get("actual_path")
                if actual_path:
                    # Resolve to the same format that list_blobs() produces
                    # so set-comparison works correctly.
                    norm_path = self._resolve_actual_path(actual_path)
                    entry_paths[norm_path] = {
                        "cache_key": entry.get("cache_key", ""),
                        "file_size": entry.get("file_size"),
                        "file_hash": entry.get("file_hash"),
                        "s3_etag": entry.get("s3_etag"),
                    }

            # 3. Find orphaned blobs (blobs with no metadata entry)
            known_paths = set(entry_paths.keys())
            orphaned_blobs = sorted(blob_files - known_paths)

            # 4. Find dangling metadata (entries pointing to missing blobs)
            dangling_entries = []
            for path, info in entry_paths.items():
                if not self.blob_backend.exists(path):
                    dangling_entries.append(
                        {
                            "cache_key": info["cache_key"],
                            "expected_path": path,
                        }
                    )

            # 5. Check size mismatches
            size_mismatches = []
            for path, info in entry_paths.items():
                if info["file_size"] is None:
                    continue
                actual_size = self.blob_backend.get_size(path)
                if actual_size == -1:
                    continue
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
            #    For remote blobs (S3), use a cheap ETag HEAD check instead
            #    of downloading the entire blob for xxhash verification.
            #    A matching ETag confirms the blob hasn't changed since
            #    upload.  A mismatching ETag is a definitive integrity
            #    failure — the object was modified or replaced.
            #    Full download + xxhash is only used when:
            #      - No stored s3_etag (older entries, non-S3 backends)
            #      - Non-remote blob (local files always use xxhash)
            hash_mismatches = []
            if verify_hashes:
                for path, info in entry_paths.items():
                    if not info.get("file_hash"):
                        continue

                    # Remote blob with stored ETag → cheap HEAD check
                    if (
                        "://" in path
                        and info.get("s3_etag")
                        and hasattr(self.blob_backend, "verify_etag")
                    ):
                        if self.blob_backend.verify_etag(  # type: ignore[call-non-callable]
                            path, info["s3_etag"]
                        ):
                            # ETag matches — blob is unchanged, skip download
                            continue
                        # ETag mismatch — object was modified/replaced
                        actual_etag = None
                        if hasattr(self.blob_backend, "get_etag"):
                            actual_etag = self.blob_backend.get_etag(  # type: ignore[call-non-callable]
                                path
                            )
                        logger.warning(
                            f"S3 ETag mismatch for {path}: "
                            f"expected {info['s3_etag']}, "
                            f"got {actual_etag}"
                        )
                        hash_mismatches.append(
                            {
                                "cache_key": info["cache_key"],
                                "path": path,
                                "expected_hash": info["file_hash"],
                                "actual_hash": f"etag-mismatch:{actual_etag}",
                            }
                        )
                        continue

                    current_hash = self._calculate_blob_hash(path)
                    if current_hash and current_hash != info["file_hash"]:
                        hash_mismatches.append(
                            {
                                "cache_key": info["cache_key"],
                                "path": path,
                                "expected_hash": info["file_hash"],
                                "actual_hash": current_hash,
                            }
                        )

            # 7. Check HMAC signatures (optional)
            #    Signature verification is handled by the caller
            #    (_verification_mixin) which has access to
            #    _extract_signable_fields() for proper normalization.
            signature_failures: list[dict[str, Any]] = []

            # 8. Repair if requested
            repaired = {"orphans_deleted": 0, "dangling_removed": 0}
            if repair:
                for blob_path in orphaned_blobs:
                    try:
                        self.blob_backend.delete_blob(blob_path)
                        repaired["orphans_deleted"] += 1
                    except OSError as e:
                        logger.warning(
                            f"Failed to remove orphaned blob {blob_path}: {e}"
                        )

                for entry in dangling_entries:
                    try:
                        self.backend.remove_entry(entry["cache_key"])
                        repaired["dangling_removed"] += 1
                    except (
                        Exception
                    ) as e:  # intentionally broad — repair failure is non-fatal
                        logger.warning(
                            f"Failed to remove dangling entry {entry['cache_key']}: {e}"
                        )

            report = IntegrityReport(
                orphaned_blobs=orphaned_blobs,
                dangling_entries=dangling_entries,
                size_mismatches=size_mismatches,
                hash_mismatches=hash_mismatches if verify_hashes else None,
                signature_failures=signature_failures if verify_signatures else None,
                repaired=repaired if repair else None,
            )

            total_issues = (
                len(orphaned_blobs)
                + len(dangling_entries)
                + len(size_mismatches)
                + len(hash_mismatches)
                + len(signature_failures)
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
                    + (
                        f", {len(signature_failures)} signature failures"
                        if verify_signatures
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
    ) -> WriteBlobResult:
        """
        Low-level: serialize data to disk via handler, then persist
        through the blob backend.

        Does NOT acquire the lock — caller is responsible for synchronization.
        Does NOT write metadata — caller handles metadata storage.

        The handler writes to a local staging path (``base_path`` + extension).
        The blob backend then persists the file (filesystem rename, S3 upload,
        in-memory store, etc.).  Any backend-specific write metadata (e.g.
        ``s3_etag``) is injected into the result dict.

        Args:
            data: The data to serialize
            base_path: Base file path (handler adds extension)
            config: Optional CacheConfig override
            compute_hash: Whether to compute xxhash file hash

        Returns:
            WriteBlobResult with handler, result, and optional file hash.
        """
        handler = self.handlers.get_handler(data)
        result = handler.put(data, base_path, config or self.config)

        # Encrypt blob content if encryption is enabled
        handler_path = Path(result.actual_path)
        if self._encryption_key is not None:
            from ..encryption import encrypt_blob

            plaintext = handler_path.read_bytes()
            ciphertext, iv, algo = encrypt_blob(plaintext, self._encryption_key)
            handler_path.write_bytes(ciphertext)
            result.file_size = len(ciphertext)
            result.extra["encryption_algorithm"] = algo.decode()
            result.extra["encryption_iv"] = iv.hex()

        # Persist through blob_backend (rename, upload, etc.)
        final_path = self.blob_backend.write_blob_from_path(
            str(handler_path), handler_path.name
        )

        # Inject backend write metadata (e.g. s3_etag)
        write_meta = self.blob_backend.get_write_metadata()
        if write_meta:
            result.extra.update(write_meta)

        # Update actual_path to the final storage location (relative)
        result.actual_path = self._to_relative_path(final_path)

        # Compute file hash from final location
        file_hash = None
        if compute_hash:
            if "://" not in final_path:
                file_hash = self._calculate_file_hash(Path(final_path))
            else:
                file_hash = self._calculate_blob_hash(final_path)

        return WriteBlobResult(handler=handler, result=result, file_hash=file_hash)

    def _read_blob(
        self,
        path: Path,
        data_type: str,
        handler_metadata: BlobReadContext,
    ) -> Any:
        """
        Low-level: deserialize data from disk via handler.

        Does NOT acquire the lock — caller is responsible for synchronization.
        Does NOT check metadata or verify signatures.
        Transparently decrypts encrypted blobs when encryption key is available.

        Args:
            path: Path to the blob file
            data_type: Handler data type identifier (e.g. "dataframe", "array")
            handler_metadata: Metadata dict passed to handler.get()

        Returns:
            Deserialized data object
        """
        # Check if blob is encrypted
        enc_algo = handler_metadata.get("encryption_algorithm")
        if enc_algo:
            actual_path = handler_metadata.get("actual_path")
            backend_path = (
                str(self._resolve_actual_path(actual_path))
                if isinstance(actual_path, str)
                else str(path)
            )
            return self._read_encrypted_blob(
                backend_path,
                path,
                data_type,
                handler_metadata,
                missing_key_message=(
                    f"Encrypted blob at {path} but no encryption key configured"
                ),
            )

        handler = self.handlers.get_handler_by_type(data_type)
        if handler is None:
            return read_file(path)
        return handler.get(path, handler_metadata)

    def _read_encrypted_blob(
        self,
        backend_path: str,
        display_path: Path,
        data_type: str,
        handler_metadata: BlobReadContext,
        missing_key_message: str,
    ) -> Any:
        """Decrypt a blob read through the configured backend."""
        if self._encryption_key is None:
            logger.warning(missing_key_message)
            return None

        from ..encryption import decrypt_blob

        ciphertext = self.blob_backend.read_blob(backend_path)
        encryption_metadata = cast(dict[str, Any], handler_metadata)
        iv = bytes.fromhex(encryption_metadata["encryption_iv"])
        plaintext = decrypt_blob(ciphertext, self._encryption_key, iv)

        handler = self.handlers.get_handler_by_type(data_type)
        if handler is not None:
            try:
                return handler.get_bytes(plaintext, handler_metadata)
            except NotImplementedError:
                pass

        suffix = display_path.suffix
        fd = -1
        temp_path: Optional[Path] = None
        try:
            fd, temp_name = tempfile.mkstemp(dir=self.cache_dir, suffix=suffix)
            temp_path = Path(temp_name)
            if os.name != "nt":
                os.chmod(temp_path, 0o600)
            with os.fdopen(fd, "wb") as tmp_file:
                fd = -1
                tmp_file.write(plaintext)

            if handler is None:
                return read_file(temp_path)
            return handler.get(temp_path, handler_metadata)
        finally:
            if fd != -1:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _clear_blob_files(self) -> int:
        """
        Delete all blob files from the cache directory.

        Does NOT acquire the lock — caller is responsible for synchronization.

        Returns:
            Number of blob files deleted
        """
        reserved_metadata_names = {
            "cacheness_namespaces.json",
        }

        def is_reserved(path: Path) -> bool:
            if any(part == ".intents" for part in path.parts):
                return True
            name = path.name
            return (
                name in reserved_metadata_names
                or name.endswith(".db")
                or name.endswith(".db-wal")
                or name.endswith(".db-shm")
                or name.startswith("cache_metadata.json")
                or name.endswith("_metadata.json")
                or name.startswith("cache_signing_key")
                or name.startswith(".cache_signing_key")
            )

        blob_root = getattr(
            self.blob_backend, "base_dir", self.cache_dir / self._namespace
        )

        removed_count = 0
        for file_path in blob_root.rglob("*"):
            if not file_path.is_file() or is_reserved(file_path):
                continue
            try:
                file_path.unlink()
                removed_count += 1
            except OSError as e:
                logger.warning(f"Failed to remove blob file {file_path}: {e}")

        return removed_count

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
        except Exception as e:  # intentionally broad — hash failure returns None
            logger.warning(f"Failed to calculate hash for {file_path}: {e}")
            return None

    def _calculate_blob_hash(self, blob_path: str) -> Optional[str]:
        """
        Calculate XXH3_64 hash of a remote blob for integrity verification.

        Downloads the blob via the blob backend and hashes it in chunks.

        Args:
            blob_path: Blob path/URI understood by the blob backend

        Returns:
            Hex string of the blob hash, or None on error
        """
        try:
            stream = self.blob_backend.read_blob_stream(blob_path)
            hasher = xxhash.xxh3_64()
            for chunk in iter(lambda: stream.read(8192), b""):
                hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:  # intentionally broad — hash failure returns None
            logger.warning(f"Failed to calculate hash for {blob_path}: {e}")
            return None

    def _compute_content_hash(self, data: Any) -> str:
        """Compute a content-based hash for the data using xxhash."""
        import pickle

        try:
            serialized = pickle.dumps(data)
        except Exception:  # intentionally broad — pickle fallback for non-pickleable
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
