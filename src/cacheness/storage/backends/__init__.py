"""
Storage Backends
================

Pluggable backends for storing cache metadata and blob data.

Metadata Backends (Phase 2.2):
- JsonBackend: Simple JSON file storage (no dependencies)
- SqliteBackend: SQLite database with SQLAlchemy ORM (requires sqlalchemy)
- PostgresBackend: PostgreSQL database for distributed caching (requires psycopg2/psycopg)

Blob Storage Backends (Phase 2.3):
- FilesystemBlobBackend: Local filesystem storage (default)
- InMemoryBlobBackend: In-memory storage for testing

Registry APIs:
- Metadata: register_metadata_backend(), get_metadata_backend(), list_metadata_backends()
- Blobs: register_blob_backend(), get_blob_backend(), list_blob_backends()

Usage:
    from cacheness.storage.backends import JsonBackend, SqliteBackend

    # JSON backend
    backend = JsonBackend(Path("cache_dir"))

    # SQLite backend
    backend = SqliteBackend(Path("cache_dir"))

    # PostgreSQL backend (requires psycopg2-binary or psycopg)
    from cacheness.storage.backends import PostgresBackend
    backend = PostgresBackend(connection_url="postgresql://localhost/cache")

    # Using the metadata registry
    from cacheness.storage.backends import (
        register_metadata_backend,
        get_metadata_backend,
        list_metadata_backends
    )

    # Register a custom backend
    register_metadata_backend("postgresql", MyPostgresBackend)

    # Get a backend instance
    backend = get_metadata_backend("postgresql", connection_url="...")

    # Using the blob registry
    from cacheness.storage.backends import (
        BlobBackend,
        register_blob_backend,
        get_blob_backend,
        list_blob_backends
    )

    # Get filesystem blob backend
    blob_backend = get_blob_backend("filesystem", base_dir="./cache")
"""

import logging
from typing import Type, Dict

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

# Conditionally import PostgresBackend
try:
    from .postgresql_backend import PostgresBackend

    _HAS_POSTGRES = True
except ImportError:
    _HAS_POSTGRES = False

logger = logging.getLogger(__name__)


# =============================================================================
# Metadata Backend Registry (Phase 2.2)
# =============================================================================

# Registry storage
_metadata_backend_registry: Dict[str, Type[MetadataBackend]] = {}


def _initialize_builtin_backends():
    """Initialize registry with built-in backends."""
    global _metadata_backend_registry

    # Always available
    _metadata_backend_registry["json"] = JsonBackend

    # SQLite (if SQLAlchemy available)
    if _HAS_SQLITE:
        _metadata_backend_registry["sqlite"] = SqliteBackend

    # PostgreSQL (if psycopg2/psycopg available)
    if _HAS_POSTGRES:
        _metadata_backend_registry["postgresql"] = PostgresBackend


# Initialize on module load
_initialize_builtin_backends()


def register_metadata_backend(
    name: str, backend_class: Type[MetadataBackend], force: bool = False
) -> None:
    """
    Register a custom metadata backend.

    Args:
        name: Unique name for the backend (e.g., "postgresql", "redis")
        backend_class: Class that implements MetadataBackend interface
        force: If True, overwrite existing registration

    Raises:
        ValueError: If name already registered and force=False
        ValueError: If backend_class doesn't inherit from MetadataBackend

    Example:
        >>> from cacheness.storage.backends import register_metadata_backend
        >>>
        >>> class PostgresBackend(MetadataBackend):
        ...     def __init__(self, connection_url: str):
        ...         self.url = connection_url
        ...     # ... implement required methods
        >>>
        >>> register_metadata_backend("postgresql", PostgresBackend)
    """
    # Validate backend class
    if not isinstance(backend_class, type):
        raise ValueError(f"backend_class must be a class, got {type(backend_class)}")

    if not issubclass(backend_class, MetadataBackend):
        raise ValueError(
            f"Backend class {backend_class.__name__} must inherit from MetadataBackend"
        )

    # Check for duplicate registration
    if name in _metadata_backend_registry and not force:
        raise ValueError(
            f"Metadata backend '{name}' already registered. "
            f"Use force=True to overwrite or unregister_metadata_backend() first."
        )

    _metadata_backend_registry[name] = backend_class
    logger.info(f"Registered metadata backend '{name}' ({backend_class.__name__})")


def unregister_metadata_backend(name: str) -> bool:
    """
    Unregister a metadata backend.

    Args:
        name: Name of the backend to unregister

    Returns:
        True if backend was unregistered, False if not found

    Note:
        Built-in backends (json, sqlite, memory) can be unregistered but
        will be re-registered on module reload.
    """
    if name in _metadata_backend_registry:
        del _metadata_backend_registry[name]
        logger.info(f"Unregistered metadata backend '{name}'")
        return True

    logger.warning(f"Metadata backend '{name}' not found for unregistration")
    return False


def get_metadata_backend(name: str, **options) -> MetadataBackend:
    """
    Get a metadata backend instance by name.

    Args:
        name: Name of the registered backend
        **options: Backend-specific configuration options

    Returns:
        Configured MetadataBackend instance

    Raises:
        ValueError: If backend name not registered

    Example:
        >>> # Get built-in backend
        >>> backend = get_metadata_backend("sqlite", db_file="cache.db")
        >>>
        >>> # Get custom registered backend
        >>> backend = get_metadata_backend(
        ...     "postgresql",
        ...     connection_url="postgresql://localhost/cache"
        ... )
    """
    if name not in _metadata_backend_registry:
        available = list(_metadata_backend_registry.keys())
        raise ValueError(
            f"Unknown metadata backend: '{name}'. Available backends: {available}"
        )

    backend_class = _metadata_backend_registry[name]

    try:
        return backend_class(**options)
    except TypeError as e:
        raise ValueError(
            f"Failed to create backend '{name}' with options {options}: {e}"
        )


def list_metadata_backends() -> list:
    """
    List all registered metadata backends.

    Returns:
        List of dictionaries with backend information:
        - name: Backend name
        - class: Backend class name
        - is_builtin: Whether it's a built-in backend

    Example:
        >>> for info in list_metadata_backends():
        ...     print(f"{info['name']}: {info['class']} (builtin={info['is_builtin']})")
        json: JsonBackend (builtin=True)
        sqlite: SqliteBackend (builtin=True)
        postgresql: PostgresBackend (builtin=True)
    """
    builtin_names = {"json", "sqlite", "sqlite_memory", "postgresql"}

    result = []
    for name, backend_class in _metadata_backend_registry.items():
        result.append(
            {
                "name": name,
                "class": backend_class.__name__,
                "is_builtin": name in builtin_names,
            }
        )

    return result


# =============================================================================
# Blob Backend Registry (Phase 2.3)
# =============================================================================

# Import blob backend registry from blob_backends module
from .blob_backends import (  # noqa: F401, E402
    BlobBackend,
    FilesystemBlobBackend,
    InMemoryBlobBackend,
    register_blob_backend,
    unregister_blob_backend,
    get_blob_backend,
    list_blob_backends,
)

# Conditionally import S3BlobBackend
try:
    from .s3_backend import S3BlobBackend, BOTO3_AVAILABLE  # noqa: F401

    _HAS_S3 = BOTO3_AVAILABLE
except ImportError:
    _HAS_S3 = False


__all__ = [
    # Metadata Backend - Base interface
    "MetadataBackend",
    # Metadata Backend - Built-in backends
    "JsonBackend",
    # Metadata Backend - Factory function (legacy)
    "create_metadata_backend",
    # Metadata Backend - Registry API
    "register_metadata_backend",
    "unregister_metadata_backend",
    "get_metadata_backend",
    "list_metadata_backends",
    # Blob Backend - Base interface
    "BlobBackend",
    # Blob Backend - Built-in backends
    "FilesystemBlobBackend",
    "InMemoryBlobBackend",
    # Blob Backend - Registry API
    "register_blob_backend",
    "unregister_blob_backend",
    "get_blob_backend",
    "list_blob_backends",
]

if _HAS_SQLITE:
    __all__.append("SqliteBackend")

if _HAS_POSTGRES:
    __all__.append("PostgresBackend")

if _HAS_S3:
    __all__.append("S3BlobBackend")
