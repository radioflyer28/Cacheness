"""Payload backend interfaces and built-in implementations.

Application selection happens through a :class:`RoleRegistry` carried by
``StoreTopology``. This module deliberately contains no global selector: a
payload participant only supplies storage primitives to the composed store.
"""

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from io import BytesIO
import os
from pathlib import Path
import tempfile
from typing import BinaryIO, Dict, Union

from cacheness.interfaces import GuardedReadSnapshot
from cacheness.storage.guarded_handler_io import GuardedHandlerIO
from cacheness.storage.path_security import ManagedFileOps, resolve_storage_root

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

    topology_capabilities = {
        "durable": True,
        "process_scope": "host",
        "host_scope": "host",
        "immutable_generations": True,
        "streaming": True,
        "listing": True,
    }

    def __init__(self, base_dir: Union[str, Path], shard_chars: int = 2):
        """
        Initialize filesystem blob backend.
        
        Args:
            base_dir: Directory where blobs will be stored
            shard_chars: Number of leading chars for Git-style sharding (default: 2)
        """
        configured_root = Path(base_dir)
        configured_root.mkdir(parents=True, exist_ok=True)
        self.base_dir = resolve_storage_root(configured_root)
        self.shard_chars = shard_chars
        self._file_ops = ManagedFileOps(self.base_dir)
        logger.debug(f"FilesystemBlobBackend initialized at {self.base_dir} (shard_chars={shard_chars})")

    def write_blob(self, blob_id: str, data: bytes) -> str:
        """Write blob to filesystem."""
        blob_path = self._file_ops.write_bytes(
            blob_id, data, shard_chars=self.shard_chars
        )
        logger.debug(f"Wrote blob {blob_id} ({len(data)} bytes) to {blob_path}")
        return str(blob_path)

    def read_blob(self, blob_path: str) -> bytes:
        """Read blob from filesystem."""
        return self._file_ops.read_bytes(blob_path)

    def delete_blob(self, blob_path: str) -> bool:
        """Delete blob from filesystem."""
        if self._file_ops.delete(blob_path):
            logger.debug(f"Deleted blob: {blob_path}")
            return True
        return False

    def exists(self, blob_path: str) -> bool:
        """Check if blob exists on filesystem."""
        return self._file_ops.exists(blob_path)

    def write_blob_stream(self, blob_id: str, stream: BinaryIO) -> str:
        """Write blob from stream to filesystem."""
        return str(
            self._file_ops.write_stream(blob_id, stream, shard_chars=self.shard_chars)
        )

    def read_blob_stream(self, blob_path: str) -> BinaryIO:
        """Read blob from filesystem as stream."""
        return self._file_ops.open_read(blob_path)

    def get_size(self, blob_path: str) -> int:
        """Get blob size from filesystem."""
        return self._file_ops.get_size(blob_path)

    def _get_blob_path(self, blob_id: str) -> Path:
        """
        Convert blob ID to filesystem path with Git-style directory sharding.
        
        With shard_chars=2 (default):
            "abc123def456" -> base_dir/ab/abc123def456
        
        With shard_chars=0 (disabled):
            "abc123def456" -> base_dir/abc123def456
        """
        return self._file_ops.blob_locator(blob_id, self.shard_chars)

    def materialize_handler_io(self) -> GuardedHandlerIO:
        """Provide guarded generation I/O rooted at this participant's root."""
        return GuardedHandlerIO(self.base_dir)

    def close(self) -> None:
        """Release the managed root descriptor when this backend is closed."""
        self._file_ops.close()


# =============================================================================
# In-Memory Backend Implementation (for testing)
# =============================================================================

class InMemoryBlobBackend(BlobBackend):
    """
    In-memory blob storage backend.
    
    Stores blobs in a dictionary. Useful for testing or ephemeral caches.
    Data is lost when the backend is closed or garbage collected.
    """

    topology_capabilities = {
        "durable": False,
        "process_scope": "process",
        "host_scope": "process",
        "immutable_generations": True,
        "streaming": False,
        "listing": False,
    }

    def __init__(self):
        """Initialize in-memory blob backend."""
        self._storage: Dict[str, bytes] = {}
        self._closed = False
        logger.debug("InMemoryBlobBackend initialized")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("In-memory blob backend is closed")

    def write_blob(self, blob_id: str, data: bytes) -> str:
        """Write blob to memory."""
        self._require_open()
        # Use blob_id as the "path" - we prefix with memory:// for clarity
        blob_path = f"memory://{blob_id}"
        self._storage[blob_path] = data
        logger.debug(f"Wrote blob {blob_id} ({len(data)} bytes) to memory")
        return blob_path

    def read_blob(self, blob_path: str) -> bytes:
        """Read blob from memory."""
        self._require_open()
        if blob_path not in self._storage:
            raise FileNotFoundError(f"Blob not found: {blob_path}")
        return self._storage[blob_path]

    def delete_blob(self, blob_path: str) -> bool:
        """Delete blob from memory."""
        self._require_open()
        if blob_path in self._storage:
            del self._storage[blob_path]
            logger.debug(f"Deleted blob: {blob_path}")
            return True
        return False

    def exists(self, blob_path: str) -> bool:
        """Check if blob exists in memory."""
        self._require_open()
        return blob_path in self._storage

    def get_size(self, blob_path: str) -> int:
        """Get blob size from memory."""
        self._require_open()
        if blob_path in self._storage:
            return len(self._storage[blob_path])
        return -1

    def clear(self) -> int:
        """Clear all blobs from memory. Returns count of blobs cleared."""
        self._require_open()
        count = len(self._storage)
        self._storage.clear()
        return count

    def materialize_handler_io(self) -> "InMemoryHandlerIO":
        """Provide the process-local generation adapter for this participant."""
        return InMemoryHandlerIO(self)

    def close(self) -> None:
        """Discard the process-local payload generation set exactly once."""
        if not self._closed:
            self._storage.clear()
            self._closed = True


class InMemoryHandlerIO:
    """Adapt guarded handler staging to an ephemeral in-memory payload backend.

    Handler contracts use filesystem paths, so this adapter uses private
    temporary files only while serializing or reading.  The committed payload
    bytes themselves live exclusively in ``InMemoryBlobBackend``; lifecycle
    transitions remain delegated to the selected authority engine.
    """

    def __init__(self, backend: InMemoryBlobBackend) -> None:
        self.backend = backend
        self._temporary_root = tempfile.TemporaryDirectory(prefix="cacheness-memory-")
        self.root = Path(self._temporary_root.name)
        self._staging = GuardedHandlerIO(self.root)
        self._closed = False

    @contextmanager
    def stage(self, handler, data, config):
        """Reuse guarded private handler serialization before copying bytes to memory."""
        self._require_open()
        with self._staging.stage(handler, data, config) as staged:
            yield staged

    def publish_generation(self, staged, locator: Path | str) -> dict:
        """Publish one immutable generation into process-local backend storage."""
        self._require_open()
        locator_text = Path(locator).as_posix()
        with staged.open() as (source, file_size):
            memory_locator = f"memory://{locator_text}"
            if self.backend.exists(memory_locator):
                raise FileExistsError("In-memory payload generation already exists")
            self.backend.write_blob(locator_text, source.read())
        return staged.result_for(Path(locator_text), file_size)

    @contextmanager
    def open_snapshot(self, locator: Path | str, metadata: Dict[str, object]):
        """Materialize one private read snapshot from committed memory bytes."""
        self._require_open()
        locator_text = Path(locator).as_posix()
        suffix = "".join(Path(locator_text).suffixes)
        with tempfile.TemporaryDirectory(prefix="cacheness-memory-read-") as temporary:
            snapshot_path = Path(temporary) / f"snapshot{suffix}"
            with snapshot_path.open("xb") as snapshot:
                snapshot_path.chmod(0o600)
                snapshot.write(self.backend.read_blob(f"memory://{locator_text}"))
                snapshot.flush()
                os.fsync(snapshot.fileno())
            snapshot_metadata = dict(metadata)
            snapshot_metadata["actual_path"] = str(snapshot_path)
            yield GuardedReadSnapshot(snapshot_path, snapshot_metadata)

    def delete_or_prove_absent(self, locator: Path | str) -> None:
        """Remove one exact process-local generation or prove it already absent."""
        self._require_open()
        locator_text = Path(locator).as_posix()
        memory_locator = f"memory://{locator_text}"
        self.backend.delete_blob(memory_locator)
        if self.backend.exists(memory_locator):
            raise OSError("In-memory payload deletion did not reach absence")

    def close(self) -> None:
        """Release only temporary handler paths; caller/store ownership closes bytes."""
        if not self._closed:
            self._staging.close()
            self._temporary_root.cleanup()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("In-memory handler adapter is closed")


__all__ = [
    "BlobBackend",
    "FilesystemBlobBackend",
    "InMemoryBlobBackend",
    "InMemoryHandlerIO",
]
