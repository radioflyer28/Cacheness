"""
Storage Layer
=============

Low-level blob storage infrastructure providing:
- Pluggable metadata backends (JSON, SQLite)
- Type-aware serialization handlers (DataFrames, arrays, objects)
- Compression support (blosc2, lz4, zstd, gzip)
- Security features (HMAC signing, integrity verification)

This layer is designed to be reusable beyond caching use cases, such as:
- ML model versioning
- Artifact storage
- Data pipeline checkpoints

Usage:
    # Direct storage layer access
    from cacheness.storage import BlobStore

    store = BlobStore(backend="sqlite", compression="lz4")
    blob_id = store.put(data, metadata={"type": "model", "version": "1.0"})
    data = store.get(blob_id)

    # Access backends directly
    from cacheness.storage.backends import SqliteBackend, JsonBackend, MemoryBackend

    # Access handlers directly
    from cacheness.storage.handlers import HandlerRegistry, ArrayHandler, ObjectHandler
"""

# Import from backends subpackage
from .backends import (
    MetadataBackend,
    JsonBackend,
    create_metadata_backend,
)

# Import from handlers subpackage
from .handlers import (
    CacheHandler,
    HandlerRegistry,
    ArrayHandler,
    ObjectHandler,
)

# Import compression utilities
from .compression import (
    write_file as write_compressed,
    read_file as read_compressed,
    is_pickleable,
    BLOSC_AVAILABLE,
    DILL_AVAILABLE,
)

# Import security
from .security import CacheEntrySigner

# Import BlobStore
from .blob_store import BlobStore

# Conditionally import SqliteBackend
try:
    from .backends import SqliteBackend

    _HAS_SQLITE = True
except ImportError:
    _HAS_SQLITE = False

__all__ = [
    # Main API
    "BlobStore",
    # Backends
    "MetadataBackend",
    "JsonBackend",
    "create_metadata_backend",
    # Handlers
    "CacheHandler",
    "HandlerRegistry",
    "ArrayHandler",
    "ObjectHandler",
    # Compression
    "write_compressed",
    "read_compressed",
    "is_pickleable",
    "BLOSC_AVAILABLE",
    "DILL_AVAILABLE",
    # Security
    "CacheEntrySigner",
]

if _HAS_SQLITE:
    __all__.append("SqliteBackend")
