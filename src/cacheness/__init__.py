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

__version__ = "0.3.6"
__author__ = "radioflyer28"
__email__ = "akgithub.2drwc@aleeas.com"

__all__ = [
    # Core classes
    "cacheness",
    "CacheConfig",
    "get_cache",
    # Handlers
    "HandlerRegistry",
    "ObjectHandler",
    "ArrayHandler",
    # Metadata backends
    "JsonBackend",
    "create_metadata_backend",
    # Decorators
    "cached",
    # Version info
    "__version__",
]

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
