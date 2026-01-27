"""
Metadata Backends
=================

Pluggable backends for storing cache metadata.

Available backends:
- JsonBackend: Simple JSON file storage (no dependencies)
- SqliteBackend: SQLite database with SQLAlchemy ORM (requires sqlalchemy)
- MemoryBackend: In-memory storage for testing/ephemeral caches

Usage:
    from cacheness.storage.backends import JsonBackend, SqliteBackend, MemoryBackend
    
    # JSON backend
    backend = JsonBackend(Path("cache_dir"))
    
    # SQLite backend
    backend = SqliteBackend(Path("cache_dir"))
    
    # In-memory backend
    backend = MemoryBackend()
"""

from .base import MetadataBackend

# Import implementations - these re-export from the cacheness.metadata module
# to maintain backward compatibility during the migration
from cacheness.metadata import JsonBackend, create_metadata_backend

# Conditionally import SqliteBackend if SQLAlchemy is available
try:
    from cacheness.metadata import SqliteBackend
    _HAS_SQLITE = True
except ImportError:
    _HAS_SQLITE = False

__all__ = [
    "MetadataBackend",
    "JsonBackend",
    "create_metadata_backend",
]

if _HAS_SQLITE:
    __all__.append("SqliteBackend")
