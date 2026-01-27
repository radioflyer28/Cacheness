"""
cacheness - High-performance disk caching library with compression and multiple backend support.

This library provides a unified interface for caching Python objects to disk with
automatic compression, multiple storage formats, and flexible metadata backends.

Key Features:
- Automatic type detection and optimal storage format selection
- Compression support (blosc2, lz4) for space efficiency
- Multiple metadata backends (JSON, SQLite)
- Support for NumPy arrays, Pandas/Polars DataFrames, and arbitrary Python objects
- Thread-safe operations
- TTL (Time To Live) support
- Comprehensive caching statistics

Quick Start:
    >>> from cacheness import cacheness
    >>>
    >>> # Create a cache instance
    >>> cache = cacheness("/path/to/cache/dir")
    >>>
    >>> # Store some data
    >>> cache.put({"key": "value"}, description="My data")
    >>>
    >>> # Retrieve data
    >>> data = cache.get("some_key")
    >>>
    >>> # Get cache statistics
    >>> stats = cache.get_stats()
"""

from .core import CacheConfig, UnifiedCache as cacheness, get_cache
from .decorators import cached
from .handlers import ArrayHandler, HandlerRegistry, ObjectHandler
from .metadata import JsonBackend, create_metadata_backend
from .interfaces import CacheHandler  # Export interface for custom handlers

# Import config validation and file loading (Phase 2.4)
from .config import (
    CacheBlobConfig,
    CacheMetadataConfig,
    CacheStorageConfig,
    CompressionConfig,
    SerializationConfig,
    HandlerConfig,
    SecurityConfig,
    ConfigValidationError,
    validate_config,
    validate_config_strict,
    load_config_from_dict,
    load_config_from_json,
    save_config_to_json,
    create_cache_config,
)

# YAML loading requires PyYAML - make it optional
try:
    from .config import load_config_from_yaml, save_config_to_yaml
    _has_yaml_config = True
except ImportError:
    _has_yaml_config = False

# Import optional components if available
try:
    from .metadata import SqliteBackend
    # Backward compatibility aliases
    JsonMetadataBackend = JsonBackend
    SQLiteMetadataBackend = SqliteBackend
    
    _has_metadata_backends = True
except ImportError:
    _has_metadata_backends = False

# Import SQL cache if SQLAlchemy is available
try:
    from .sql_cache import SqlCache, SqlCacheAdapter
    _has_sql_cache = True
    # Backward compatibility aliases  
    SQLAlchemyPullThroughCache = SqlCache
    SQLAlchemySqlCacheAdapter = SqlCacheAdapter
except ImportError:
    _has_sql_cache = False

__version__ = "0.3.14"
__author__ = "radioflyer28"
__email__ = "akgithub.2drwc@aleeas.com"

# Import storage layer components for convenience
# These are also available via `from cacheness.storage import ...`
try:
    from .storage import BlobStore
    _has_blob_store = True
except ImportError:
    _has_blob_store = False


# =============================================================================
# Module-Level Handler Registration API (Phase 2.1)
# =============================================================================

# Global default registry for module-level registration
_default_registry: HandlerRegistry | None = None


def _get_default_registry() -> HandlerRegistry:
    """Get or create the default handler registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = HandlerRegistry()
    return _default_registry


def register_handler(
    handler: CacheHandler,
    priority: int | None = None,
    name: str | None = None
) -> None:
    """
    Register a custom handler with the default registry.
    
    This is a convenience function for registering handlers at module level.
    For more control, create a HandlerRegistry instance directly.
    
    Args:
        handler: Handler instance implementing CacheHandler interface
        priority: Position in handler list (0 = highest priority, None = append)
        name: Optional name for the handler (defaults to handler.data_type)
        
    Example:
        >>> from cacheness import register_handler, CacheHandler
        >>> 
        >>> class ParquetHandler(CacheHandler):
        ...     @property
        ...     def data_type(self): return "parquet"
        ...     # ... implement other methods
        >>> 
        >>> register_handler(ParquetHandler(), priority=0)
    """
    _get_default_registry().register_handler(handler, priority=priority, name=name)


def unregister_handler(handler_name: str) -> bool:
    """
    Unregister a handler from the default registry.
    
    Args:
        handler_name: The data_type of the handler to remove
        
    Returns:
        True if handler was removed, False if not found
    """
    return _get_default_registry().unregister_handler(handler_name)


def list_handlers() -> list:
    """
    List all handlers in the default registry.
    
    Returns:
        List of handler info dictionaries with keys:
        - name: Handler data_type
        - priority: Position in handler list
        - class: Handler class name
        - is_builtin: Whether it's a built-in handler
    """
    return _get_default_registry().list_handlers()


# =============================================================================
# Module-Level Metadata Backend Registration API (Phase 2.2)
# =============================================================================

# Import backend registry functions
try:
    from .storage.backends import (
        register_metadata_backend,
        unregister_metadata_backend,
        get_metadata_backend,
        list_metadata_backends,
        MetadataBackend,  # Base class for custom backends
    )
    _has_backend_registry = True
except ImportError:
    _has_backend_registry = False


# =============================================================================
# Module-Level Blob Backend Registration API (Phase 2.3)
# =============================================================================

# Import blob backend registry functions
try:
    from .storage.backends import (
        register_blob_backend,
        unregister_blob_backend,
        get_blob_backend,
        list_blob_backends,
        BlobBackend,  # Base class for custom blob backends
        FilesystemBlobBackend,
        InMemoryBlobBackend,
    )
    _has_blob_backend_registry = True
except ImportError:
    _has_blob_backend_registry = False


__all__ = [
    # Core classes
    "cacheness",
    "CacheConfig",
    "get_cache",
    # Sub-configuration classes (Phase 2.4)
    "CacheBlobConfig",
    "CacheMetadataConfig", 
    "CacheStorageConfig",
    "CompressionConfig",
    "SerializationConfig",
    "HandlerConfig",
    "SecurityConfig",
    # Configuration validation (Phase 2.4)
    "ConfigValidationError",
    "validate_config",
    "validate_config_strict",
    "load_config_from_dict",
    "load_config_from_json",
    "save_config_to_json",
    "create_cache_config",
    # Handlers
    "HandlerRegistry",
    "ObjectHandler",
    "ArrayHandler",
    "CacheHandler",  # Interface for custom handlers
    # Handler registration API
    "register_handler",
    "unregister_handler",
    "list_handlers",
    # Metadata backends
    "JsonBackend",
    "create_metadata_backend",
    # Decorators
    "cached",
    # Version info
    "__version__",
]

# Add YAML config support if PyYAML available (Phase 2.4)
if _has_yaml_config:
    __all__.extend([
        "load_config_from_yaml",
        "save_config_to_yaml",
    ])

# Add backend registry functions if available (Phase 2.2)
if _has_backend_registry:
    __all__.extend([
        "register_metadata_backend",
        "unregister_metadata_backend", 
        "get_metadata_backend",
        "list_metadata_backends",
        "MetadataBackend",  # Base class for custom backends
    ])

# Add blob backend registry functions if available (Phase 2.3)
if _has_blob_backend_registry:
    __all__.extend([
        "register_blob_backend",
        "unregister_blob_backend",
        "get_blob_backend",
        "list_blob_backends",
        "BlobBackend",  # Base class for custom blob backends
        "FilesystemBlobBackend",
        "InMemoryBlobBackend",
    ])

# Add BlobStore if available (new storage layer API)
if _has_blob_store:
    __all__.append("BlobStore")

# Add metadata backends to exports if available
if _has_metadata_backends:
    __all__.extend([
        "SqliteBackend",  # New simple name
        "JsonMetadataBackend", "SQLiteMetadataBackend"  # Backward compatibility
    ])

# Add SQL cache to exports if available
if _has_sql_cache:
    __all__.extend([
        "SqlCache", "SqlCacheAdapter",  # New simple names
        "SQLAlchemyPullThroughCache", "SQLAlchemyDataAdapter"  # Backward compatibility
    ])
