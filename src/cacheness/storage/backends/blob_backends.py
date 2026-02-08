"""
Blob Storage Backends
=====================

Abstract interface and registry for blob storage backends.

Blob backends handle the actual data storage, separate from metadata backends.
This separation allows combinations like:
- SQLite metadata + Filesystem blobs (default)
- PostgreSQL metadata + S3 blobs (distributed team cache)
- In-memory metadata + S3 blobs (serverless/testing)

Usage:
    from cacheness.storage.backends.blob_backends import (
        BlobBackend,
        FilesystemBlobBackend,
        register_blob_backend,
        get_blob_backend,
        list_blob_backends,
    )

    # Use built-in filesystem backend
    backend = get_blob_backend("filesystem", base_dir="./cache")
    blob_path = backend.write_blob("my_key", data_bytes)
    data = backend.read_blob(blob_path)

    # Register custom backend (e.g., S3)
    class S3BlobBackend(BlobBackend):
        def __init__(self, bucket: str, region: str = "us-east-1"):
            self.bucket = bucket
            self.region = region
            # ... initialize S3 client

        def write_blob(self, blob_id: str, data: bytes) -> str:
            # Upload to S3, return s3://bucket/key URL
            ...

    register_blob_backend("s3", S3BlobBackend)
    backend = get_blob_backend("s3", bucket="my-cache", region="us-west-2")
"""

import logging
import os
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Dict, List, Type, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Base Class
# =============================================================================


class BlobBackend(ABC):
    """
    Abstract base class for blob storage backends.

    Blob backends are responsible for storing and retrieving raw binary data.
    They work in conjunction with metadata backends, which store the index
    and metadata about the blobs.

    Implementations must provide both synchronous byte-based methods
    (write_blob/read_blob) and optionally streaming methods for large objects.

    All blob paths/URLs returned by write operations should be strings that
    can be passed back to read/delete operations.
    """

    @abstractmethod
    def write_blob(self, blob_id: str, data: bytes) -> str:
        """
        Write blob data to storage.

        Args:
            blob_id: Unique identifier for the blob (used to generate path)
            data: Raw bytes to store

        Returns:
            Storage path/URL where the blob was written.
            This path can be passed to read_blob() or delete_blob().
        """
        pass

    @abstractmethod
    def read_blob(self, blob_path: str) -> bytes:
        """
        Read blob data from storage.

        Args:
            blob_path: Storage path/URL returned by write_blob()

        Returns:
            Raw bytes of the blob

        Raises:
            FileNotFoundError: If blob doesn't exist
        """
        pass

    @abstractmethod
    def delete_blob(self, blob_path: str) -> bool:
        """
        Delete blob from storage.

        Args:
            blob_path: Storage path/URL returned by write_blob()

        Returns:
            True if blob was deleted, False if it didn't exist
        """
        pass

    @abstractmethod
    def exists(self, blob_path: str) -> bool:
        """
        Check if blob exists in storage.

        Args:
            blob_path: Storage path/URL to check

        Returns:
            True if blob exists, False otherwise
        """
        pass

    def write_blob_stream(self, blob_id: str, stream: BinaryIO) -> str:
        """
        Write blob from a stream (for large objects).

        Default implementation reads entire stream into memory.
        Override for backends that support true streaming uploads.

        Args:
            blob_id: Unique identifier for the blob
            stream: File-like object with read() method

        Returns:
            Storage path/URL where the blob was written
        """
        data = stream.read()
        return self.write_blob(blob_id, data)

    def read_blob_stream(self, blob_path: str) -> BinaryIO:
        """
        Read blob as a stream (for large objects).

        Default implementation reads entire blob into memory.
        Override for backends that support true streaming downloads.

        Args:
            blob_path: Storage path/URL returned by write_blob()

        Returns:
            File-like object with read() method
        """
        data = self.read_blob(blob_path)
        return BytesIO(data)

    def get_size(self, blob_path: str) -> int:
        """
        Get size of blob in bytes.

        Default implementation reads the blob to get size.
        Override for backends that can query size without reading.

        Args:
            blob_path: Storage path/URL

        Returns:
            Size in bytes, or -1 if unknown
        """
        try:
            data = self.read_blob(blob_path)
            return len(data)
        except Exception:
            return -1

    def close(self) -> None:
        """
        Close and clean up any resources.

        Default implementation does nothing. Override in backends that
        hold connections or other resources.
        """
        pass

    def __enter__(self):
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are cleaned up."""
        self.close()
        return False


# =============================================================================
# Filesystem Backend Implementation
# =============================================================================


class FilesystemBlobBackend(BlobBackend):
    """
    Filesystem-based blob storage backend.

    This is the default backend that stores blobs as files in a directory.
    Blob paths are absolute filesystem paths.

    Supports Git-style directory sharding to avoid overloading directories
    with too many files. With shard_chars=2 (default), blob IDs are stored as:
        "abc123..." -> "ab/abc123..."

    Attributes:
        base_dir: Root directory for blob storage
        shard_chars: Number of leading characters for directory sharding (0 to disable)
    """

    def __init__(self, base_dir: Union[str, Path], shard_chars: int = 2):
        """
        Initialize filesystem blob backend.

        Args:
            base_dir: Directory where blobs will be stored
            shard_chars: Number of leading chars for Git-style sharding (default: 2)
        """
        self.base_dir = Path(base_dir)
        self.shard_chars = shard_chars
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"FilesystemBlobBackend initialized at {self.base_dir} (shard_chars={shard_chars})"
        )

    def write_blob(self, blob_id: str, data: bytes) -> str:
        """Write blob to filesystem."""
        blob_path = self._get_blob_path(blob_id)
        blob_path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically using temp file
        temp_path = blob_path.with_suffix(blob_path.suffix + ".tmp")
        try:
            temp_path.write_bytes(data)
            temp_path.replace(blob_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        logger.debug(f"Wrote blob {blob_id} ({len(data)} bytes) to {blob_path}")
        return str(blob_path)

    def read_blob(self, blob_path: str) -> bytes:
        """Read blob from filesystem."""
        path = Path(blob_path)
        if not path.exists():
            raise FileNotFoundError(f"Blob not found: {blob_path}")
        return path.read_bytes()

    def delete_blob(self, blob_path: str) -> bool:
        """Delete blob from filesystem."""
        path = Path(blob_path)
        if path.exists():
            path.unlink()
            logger.debug(f"Deleted blob: {blob_path}")
            return True
        return False

    def exists(self, blob_path: str) -> bool:
        """Check if blob exists on filesystem."""
        return Path(blob_path).exists()

    def write_blob_stream(self, blob_id: str, stream: BinaryIO) -> str:
        """Write blob from stream to filesystem."""
        blob_path = self._get_blob_path(blob_id)
        blob_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = blob_path.with_suffix(blob_path.suffix + ".tmp")
        try:
            with open(temp_path, "wb") as f:
                # Read in chunks for memory efficiency
                while True:
                    chunk = stream.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
            temp_path.replace(blob_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        return str(blob_path)

    def read_blob_stream(self, blob_path: str) -> BinaryIO:
        """Read blob from filesystem as stream."""
        path = Path(blob_path)
        if not path.exists():
            raise FileNotFoundError(f"Blob not found: {blob_path}")
        return open(path, "rb")

    def get_size(self, blob_path: str) -> int:
        """Get blob size from filesystem."""
        path = Path(blob_path)
        if path.exists():
            return path.stat().st_size
        return -1

    def _get_blob_path(self, blob_id: str) -> Path:
        """
        Convert blob ID to filesystem path with Git-style directory sharding.

        With shard_chars=2 (default):
            "abc123def456" -> base_dir/ab/abc123def456

        With shard_chars=0 (disabled):
            "abc123def456" -> base_dir/abc123def456
        """
        # Sanitize blob_id to prevent path traversal
        safe_id = blob_id.replace("..", "__").replace("/", os.sep).replace("\\", os.sep)

        # Apply Git-style sharding if enabled
        if self.shard_chars > 0 and len(safe_id) >= self.shard_chars:
            shard_dir = safe_id[: self.shard_chars]
            return self.base_dir / shard_dir / safe_id

        return self.base_dir / safe_id


# =============================================================================
# In-Memory Backend Implementation (for testing)
# =============================================================================


class InMemoryBlobBackend(BlobBackend):
    """
    In-memory blob storage backend.

    Stores blobs in a dictionary. Useful for testing or ephemeral caches.
    Data is lost when the backend is closed or garbage collected.
    """

    def __init__(self):
        """Initialize in-memory blob backend."""
        self._storage: Dict[str, bytes] = {}
        logger.debug("InMemoryBlobBackend initialized")

    def write_blob(self, blob_id: str, data: bytes) -> str:
        """Write blob to memory."""
        # Use blob_id as the "path" - we prefix with memory:// for clarity
        blob_path = f"memory://{blob_id}"
        self._storage[blob_path] = data
        logger.debug(f"Wrote blob {blob_id} ({len(data)} bytes) to memory")
        return blob_path

    def read_blob(self, blob_path: str) -> bytes:
        """Read blob from memory."""
        if blob_path not in self._storage:
            raise FileNotFoundError(f"Blob not found: {blob_path}")
        return self._storage[blob_path]

    def delete_blob(self, blob_path: str) -> bool:
        """Delete blob from memory."""
        if blob_path in self._storage:
            del self._storage[blob_path]
            logger.debug(f"Deleted blob: {blob_path}")
            return True
        return False

    def exists(self, blob_path: str) -> bool:
        """Check if blob exists in memory."""
        return blob_path in self._storage

    def get_size(self, blob_path: str) -> int:
        """Get blob size from memory."""
        if blob_path in self._storage:
            return len(self._storage[blob_path])
        return -1

    def clear(self) -> int:
        """Clear all blobs from memory. Returns count of blobs cleared."""
        count = len(self._storage)
        self._storage.clear()
        return count


# =============================================================================
# Blob Backend Registry
# =============================================================================

# Registry storage: name -> (backend_class, is_builtin)
_blob_backend_registry: Dict[str, Type[BlobBackend]] = {}
_builtin_blob_backends = {"filesystem", "memory"}


def _initialize_builtin_blob_backends():
    """Initialize registry with built-in backends."""
    global _blob_backend_registry
    _blob_backend_registry["filesystem"] = FilesystemBlobBackend
    _blob_backend_registry["memory"] = InMemoryBlobBackend


# Initialize on module load
_initialize_builtin_blob_backends()


def register_blob_backend(
    name: str, backend_class: Type[BlobBackend], force: bool = False
) -> None:
    """
    Register a custom blob storage backend.

    Args:
        name: Unique name for the backend (e.g., "s3", "azure", "gcs")
        backend_class: Class that implements BlobBackend interface
        force: If True, overwrite existing registration

    Raises:
        ValueError: If name already registered and force=False
        ValueError: If backend_class doesn't inherit from BlobBackend

    Example:
        >>> class S3BlobBackend(BlobBackend):
        ...     def __init__(self, bucket: str, region: str = "us-east-1"):
        ...         self.bucket = bucket
        ...         self.region = region
        ...
        ...     def write_blob(self, blob_id: str, data: bytes) -> str:
        ...         # Upload to S3
        ...         return f"s3://{self.bucket}/{blob_id}"
        ...
        ...     # ... implement other methods
        >>>
        >>> register_blob_backend("s3", S3BlobBackend)
    """
    # Validate backend class
    if not isinstance(backend_class, type):
        raise ValueError(f"backend_class must be a class, got {type(backend_class)}")

    if not issubclass(backend_class, BlobBackend):
        raise ValueError(
            f"Backend class {backend_class.__name__} must inherit from BlobBackend"
        )

    # Check for duplicate registration
    if name in _blob_backend_registry and not force:
        raise ValueError(
            f"Blob backend '{name}' already registered. "
            f"Use force=True to overwrite or unregister_blob_backend() first."
        )

    _blob_backend_registry[name] = backend_class
    logger.info(f"Registered blob backend '{name}' ({backend_class.__name__})")


def unregister_blob_backend(name: str) -> bool:
    """
    Unregister a blob storage backend.

    Args:
        name: Name of the backend to unregister

    Returns:
        True if backend was unregistered, False if not found

    Note:
        Built-in backends (filesystem, memory) can be unregistered but
        will be re-registered on module reload.
    """
    if name in _blob_backend_registry:
        del _blob_backend_registry[name]
        logger.info(f"Unregistered blob backend '{name}'")
        return True

    logger.warning(f"Blob backend '{name}' not found for unregistration")
    return False


def get_blob_backend(name: str, **options) -> BlobBackend:
    """
    Get a blob backend instance by name.

    Args:
        name: Name of the registered backend
        **options: Backend-specific configuration options

    Returns:
        Configured BlobBackend instance

    Raises:
        ValueError: If backend name not registered

    Example:
        >>> # Get filesystem backend
        >>> backend = get_blob_backend("filesystem", base_dir="./cache")
        >>>
        >>> # Get custom S3 backend
        >>> backend = get_blob_backend("s3", bucket="my-cache", region="us-west-2")
    """
    if name not in _blob_backend_registry:
        available = list(_blob_backend_registry.keys())
        raise ValueError(
            f"Unknown blob backend: '{name}'. Available backends: {available}"
        )

    backend_class = _blob_backend_registry[name]

    try:
        return backend_class(**options)
    except TypeError as e:
        raise ValueError(
            f"Failed to create blob backend '{name}' with options {options}: {e}"
        )


def list_blob_backends() -> List[Dict[str, any]]:
    """
    List all registered blob backends.

    Returns:
        List of dictionaries with backend info:
        - name: Backend name
        - class: Backend class name
        - is_builtin: Whether it's a built-in backend

    Example:
        >>> backends = list_blob_backends()
        >>> for b in backends:
        ...     print(f"{b['name']}: {b['class']} (builtin={b['is_builtin']})")
        filesystem: FilesystemBlobBackend (builtin=True)
        memory: InMemoryBlobBackend (builtin=True)
    """
    result = []

    # Add builtin backends first
    for name in sorted(_builtin_blob_backends):
        if name in _blob_backend_registry:
            backend_class = _blob_backend_registry[name]
            result.append(
                {
                    "name": name,
                    "class": backend_class.__name__,
                    "is_builtin": True,
                }
            )

    # Add custom backends
    for name in sorted(_blob_backend_registry.keys()):
        if name not in _builtin_blob_backends:
            backend_class = _blob_backend_registry[name]
            result.append(
                {
                    "name": name,
                    "class": backend_class.__name__,
                    "is_builtin": False,
                }
            )

    return result


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Base class
    "BlobBackend",
    # Built-in implementations
    "FilesystemBlobBackend",
    "InMemoryBlobBackend",
    # Registry functions
    "register_blob_backend",
    "unregister_blob_backend",
    "get_blob_backend",
    "list_blob_backends",
]
