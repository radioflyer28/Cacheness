"""Metadata backend package for pluggable cache metadata storage.

Re-exports all public names for backward compatibility.
"""

import logging
from pathlib import Path

from ._compat import (
    SQLALCHEMY_AVAILABLE,
    CACHETOOLS_AVAILABLE,
    Base,
    CacheEntry,
    CacheEntryMixin,
    CacheStats,
    CacheStatsMixin,
    CacheNamespace,
    _get_namespace_models,
    _ns_model_cache,
    _CORE_TABLES,
    DEFAULT_NAMESPACE,
    NAMESPACE_ID_PATTERN,
    NamespaceInfo,
    Migration,
    validate_namespace_id,
)
from .base import MetadataBackend, CachedMetadataBackend, create_entry_cache
from .json_backend import JsonBackend

# Conditionally import SqliteBackend (requires SQLAlchemy)
try:
    from .sqlite_backend import SqliteBackend
except ImportError:
    pass

logger = logging.getLogger(__name__)


def create_metadata_backend(backend_type: str = "auto", **kwargs) -> MetadataBackend:
    """
    Factory function to create metadata backends with optional entry caching.

    Args:
        backend_type: "auto", "sqlite", "json", "sqlite_memory", or "postgresql"
                     "auto" prefers SQLite (file-based) if available, falls back to in-memory SQLite, then JSON
        **kwargs: Backend-specific configuration including optional 'config' for caching
                  and 'namespace' for multi-tenant namespace isolation

    Returns:
        MetadataBackend instance (potentially wrapped with caching)
    """
    # Extract cache config if provided
    cache_config = kwargs.pop("config", None)

    # Extract namespace (defaults to 'default' for backward compatibility)
    namespace = kwargs.pop("namespace", DEFAULT_NAMESPACE)

    # Create the base backend
    if backend_type == "json":
        metadata_file = kwargs.get("metadata_file", Path("cache_metadata.json"))
        backend = JsonBackend(metadata_file, namespace=namespace)
    elif backend_type == "sqlite":
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is required for SQLite backend but is not available. Install with: uv add sqlalchemy"
            )
        db_file = kwargs.get("db_file", "cache_metadata.db")
        echo = kwargs.get("echo", False)
        backend = SqliteBackend(db_file, echo, namespace=namespace)
    elif backend_type == "sqlite_memory":
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is required for in-memory SQLite backend but is not available. Install with: uv add sqlalchemy"
            )
        echo = kwargs.get("echo", False)
        backend = SqliteBackend(":memory:", echo, namespace=namespace)
    elif backend_type == "postgresql":
        # Import PostgresBackend from the storage backends
        try:
            from ..storage.backends.postgresql_backend import PostgresBackend
        except ImportError as e:
            raise ImportError(
                f"PostgreSQL backend is not available. Install with: pip install psycopg2-binary sqlalchemy. Error: {e}"
            )

        connection_url = kwargs.get("connection_url")
        if not connection_url:
            raise ValueError("PostgreSQL backend requires 'connection_url' parameter")

        backend = PostgresBackend(
            connection_url=connection_url,
            pool_size=kwargs.get("pool_size", 10),
            max_overflow=kwargs.get("max_overflow", 20),
            pool_pre_ping=kwargs.get("pool_pre_ping", True),
            pool_recycle=kwargs.get("pool_recycle", 3600),
            echo=kwargs.get("echo", False),
            table_prefix=kwargs.get("table_prefix", ""),
            namespace=namespace,
        )
    elif backend_type == "auto":
        # Auto mode: prefer file-based SQLite > in-memory SQLite > JSON
        if SQLALCHEMY_AVAILABLE:
            try:
                # Try file-based SQLite first
                db_file = kwargs.get("db_file", "cache_metadata.db")
                echo = kwargs.get("echo", False)
                backend = SqliteBackend(db_file, echo, namespace=namespace)
            except Exception:
                try:
                    # Fall back to in-memory SQLite if file-based fails
                    logger.warning("File-based SQLite failed, using in-memory SQLite")
                    echo = kwargs.get("echo", False)
                    backend = SqliteBackend(":memory:", echo, namespace=namespace)
                except Exception:
                    # Final fallback to JSON
                    logger.warning("SQLite backends failed, falling back to JSON")
                    metadata_file = kwargs.get(
                        "metadata_file", Path("cache_metadata.json")
                    )
                    backend = JsonBackend(metadata_file, namespace=namespace)
        else:
            # SQLAlchemy not available, use JSON
            metadata_file = kwargs.get("metadata_file", Path("cache_metadata.json"))
            backend = JsonBackend(metadata_file, namespace=namespace)
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. Supported: 'auto', 'json', 'sqlite', 'sqlite_memory', 'postgresql'"
        )

    # Apply memory cache layer wrapper for disk-persistent backends
    if (
        cache_config is not None
        and hasattr(cache_config, "enable_memory_cache")
        and cache_config.enable_memory_cache
    ):
        logger.debug(f"Wrapping {backend_type} backend with memory cache layer")
        backend = CachedMetadataBackend(backend, cache_config)

    return backend


__all__ = [
    "SQLALCHEMY_AVAILABLE",
    "CACHETOOLS_AVAILABLE",
    "Base",
    "CacheEntry",
    "CacheEntryMixin",
    "CacheStats",
    "CacheStatsMixin",
    "CacheNamespace",
    "_get_namespace_models",
    "_ns_model_cache",
    "_CORE_TABLES",
    "DEFAULT_NAMESPACE",
    "NAMESPACE_ID_PATTERN",
    "NamespaceInfo",
    "Migration",
    "validate_namespace_id",
    "MetadataBackend",
    "CachedMetadataBackend",
    "create_entry_cache",
    "JsonBackend",
    "SqliteBackend",
    "create_metadata_backend",
]
