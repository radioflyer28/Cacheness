# Configuration Guide

Comprehensive guide to configuring cacheness for optimal performance and functionality.

## Configuration Basics

### Simple Configuration

```python
from cacheness import cacheness, CacheConfig

# Basic configuration
config = CacheConfig(
    cache_dir="./my_cache",
    metadata_backend="sqlite",     # "sqlite", "json", or "auto"  
    default_ttl_hours=48,          # Default TTL for entries
    max_cache_size_mb=5000,        # Maximum cache size
)

cache = cacheness(config)
```

### Sub-Configuration Approach (Recommended)

```python
from cacheness.config import (
    CacheStorageConfig, 
    CacheMetadataConfig, 
    CompressionConfig,
    SerializationConfig,
    HandlerConfig,
    SecurityConfig
)

# Advanced configuration with sub-configurations
config = CacheConfig(
    storage=CacheStorageConfig(
        cache_dir="./advanced_cache",
        max_cache_size_mb=10000,
        cleanup_on_init=True
    ),
    metadata=CacheMetadataConfig(
        backend="sqlite",
        verify_cache_integrity=True,
        store_cache_key_params=True
    ),
    compression=CompressionConfig(
        use_blosc2_arrays=True,
        pickle_compression_codec="zstd",
        pickle_compression_level=5
    ),
    security=SecurityConfig(
        enable_entry_signing=True,
        security_level="enhanced",
        delete_invalid_signatures=True
    ),
    default_ttl_hours=48
)

cache = cacheness(config)
```

## Configuration Options Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_dir` | str | `"./cache"` | Cache directory path |
| `default_ttl_hours` | int | `24` | Default time-to-live in hours |
| `max_cache_size_mb` | int | `2000` | Maximum cache size in MB |

### Storage Configuration (`CacheStorageConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_dir` | str | `"./cache"` | Cache directory path |
| `max_cache_size_mb` | int | `2000` | Maximum cache size in MB |
| `cleanup_on_init` | bool | `True` | Clean expired entries on initialization |

### Metadata Configuration (`CacheMetadataConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | str | `"auto"` | Backend: "auto", "sqlite", "json" |
| `database_url` | str | `None` | Custom SQLite database path |
| `verify_cache_integrity` | bool | `False` | Enable file hash verification |
| `store_cache_key_params` | bool | `True` | Store cache key parameters |

### Security Configuration (`SecurityConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_entry_signing` | bool | `True` | Enable HMAC-SHA256 cache entry signing |
| `security_level` | str | `"enhanced"` | Security level: "minimal", "enhanced", "paranoid" |
| `use_in_memory_key` | bool | `False` | Use in-memory signing key (no disk persistence) |
| `delete_invalid_signatures` | bool | `True` | Auto-delete entries with invalid signatures |
| `allow_unsigned_entries` | bool | `True` | Accept entries created before signing was enabled |
| `signing_key_file` | str | `"cache_signing_key.bin"` | Path to signing key file |
| `custom_signed_fields` | list | `None` | Custom fields to sign (overrides security_level) |

**See [Security Guide](SECURITY.md) for comprehensive security configuration.**

### Compression Configuration (`CompressionConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_blosc2_arrays` | bool | `True` | High-performance array compression |
| `npz_compression` | bool | `True` | Enable NPZ compression |
| `pickle_compression_codec` | str | `"zstd"` | Compression codec for objects |
| `pickle_compression_level` | int | `5` | Compression level (0-19) |
| `compression_threshold_bytes` | int | `1024` | Only compress objects larger than this |
| `enable_parallel_compression` | bool | `True` | Use multiple threads for compression |
| `parquet_compression` | str | `"lz4"` | Compression for DataFrames |
| `blosc2_array_codec` | str | `"lz4"` | Codec for array compression |
| `blosc2_array_clevel` | int | `5` | Array compression level (0-9) |
| `pickle_compression_level` | int | `5` | Compression level (1-9) |
| `blosc2_array_codec` | str | `"lz4"` | Array compression algorithm |
| `parquet_compression` | str | `"lz4"` | DataFrame compression |

### Serialization Configuration (`SerializationConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_collections` | bool | `True` | Recursive collection analysis |
| `enable_object_introspection` | bool | `True` | Deep object inspection |
| `max_collection_depth` | int | `10` | Nesting depth limit |
| `max_tuple_recursive_length` | int | `10` | Tuple recursion limit |

### Handler Configuration (`HandlerConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_pandas_dataframes` | bool | `True` | Enable pandas DataFrame caching |
| `enable_polars_dataframes` | bool | `True` | Enable polars DataFrame caching |
| `enable_pandas_series` | bool | `True` | Enable pandas Series caching |
| `enable_polars_series` | bool | `True` | Enable polars Series caching |
| `enable_numpy_arrays` | bool | `True` | Enable NumPy array caching |
| `enable_object_pickle` | bool | `True` | Enable general object caching |
| `enable_tensorflow_tensors` | bool | `False` | Enable TensorFlow tensor caching |
| `enable_dill_fallback` | bool | `True` | Use dill for complex objects (functions, lambdas) |
| `handler_priority` | list | See below | Handler priority order |

Default handler priority:
```python
["pandas_series", "polars_series", "pandas_dataframes", "polars_dataframes", "numpy_arrays", "tensorflow_tensors", "object_pickle"]
```

**Advanced Object Serialization:**
- `enable_dill_fallback=True`: Enables caching of functions, lambdas, closures, and other objects that standard pickle cannot handle
- `enable_tensorflow_tensors=False`: TensorFlow handler disabled by default due to import overhead and system compatibility issues

## Use Case Configurations

### Machine Learning Workflows

```python
# High-performance ML cache with security
ml_config = CacheConfig(
    storage=CacheStorageConfig(
        cache_dir="./ml_cache",
        max_cache_size_mb=10000,  # Large cache for datasets
        cleanup_on_init=True
    ),
    metadata=CacheMetadataConfig(
        backend="sqlite",  # Fast metadata operations
        store_cache_key_params=True  # Track experiment parameters
    ),
    compression=CompressionConfig(
        use_blosc2_arrays=True,      # Optimal for numeric data
        pickle_compression_codec="zstd",
        pickle_compression_level=6   # Higher compression for models
    ),
    security=SecurityConfig(
        enable_entry_signing=True,
        security_level="enhanced",   # Good security for ML models
        delete_invalid_signatures=True
    ),
    handlers=HandlerConfig(
        handler_priority=[
            "numpy_arrays",          # Prioritize arrays
            "pandas_dataframes",
            "object_pickle"
        ]
    ),
    default_ttl_hours=168  # 1 week
)
```

### API Response Caching

```python
# Fast API response cache
api_config = CacheConfig(
    storage=CacheStorageConfig(
        cache_dir="./api_cache",
        max_cache_size_mb=1000,
        cleanup_on_init=True
    ),
    metadata=CacheMetadataConfig(
        backend="sqlite",
        store_cache_key_params=False  # No sensitive API params
    ),
    compression=CompressionConfig(
        pickle_compression_codec="lz4",  # Fast compression
        pickle_compression_level=1
    ),
    serialization=SerializationConfig(
        enable_collections=False,    # Skip deep introspection
        max_tuple_recursive_length=3
    ),
    default_ttl_hours=6  # Short TTL for API data
)
```

### Data Processing Pipeline

```python
# Precision-focused configuration
data_config = CacheConfig(
    storage=CacheStorageConfig(
        cache_dir="./data_cache",
        max_cache_size_mb=15000,  # Large data cache
    ),
    metadata=CacheMetadataConfig(
        backend="sqlite",
        verify_cache_integrity=True,  # Ensure data integrity
        store_cache_key_params=True
    ),
    compression=CompressionConfig(
        use_blosc2_arrays=True,
        parquet_compression="snappy",  # Good for mixed data types
        pickle_compression_codec="zstd",
        pickle_compression_level=7     # High compression
    ),
    serialization=SerializationConfig(
        enable_collections=True,           # Full parameter tracking
        enable_object_introspection=True,
        max_collection_depth=15           # Deep nested structures
    ),
    handlers=HandlerConfig(
        handler_priority=[
            "pandas_dataframes",
            "polars_dataframes", 
            "numpy_arrays",
            "object_pickle"
        ]
    ),
    default_ttl_hours=72  # 3 days
)
```

## Backend Selection

### SQLite Backend (Recommended)

**When to use:**
- Large number of cache entries (>1000)
- Frequent metadata queries
- Production deployments
- Need advanced querying (custom metadata)

**Configuration:**
```python
config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="sqlite",
        database_url="./custom_cache.db",  # Optional custom path
        verify_cache_integrity=True        # Enable integrity checks
    )
)
```

### JSON Backend

**When to use:**
- Small cache (< 1000 entries)
- Simple deployment requirements
- Development/testing
- SQLAlchemy not available

**Configuration:**
```python
config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="json"
    )
)
```

### Auto Backend (Default)

Automatically selects the best available backend:
1. SQLite if SQLAlchemy is available
2. JSON as fallback

## Compression Strategies

### Object Compression

| Codec | Speed | Compression | Best For |
|-------|-------|-------------|----------|
| `lz4` | Fastest | Good | Real-time applications |
| `zstd` | Fast | Excellent | General purpose (default) |
| `lz4hc` | Medium | Better | Balanced performance |
| `zlib` | Slower | Good | Compatibility |

### Array Compression

| Codec | Speed | Compression | Best For |
|-------|-------|-------------|----------|
| `lz4` | Fastest | Good | Frequent access (default) |
| `zstd` | Fast | Better | Storage efficiency |
| `lz4hc` | Medium | Better | Balanced |

### DataFrame Compression

| Codec | Speed | Compression | Best For |
|-------|-------|-------------|----------|
| `lz4` | Fastest | Good | General purpose (default) |
| `snappy` | Fast | Good | Mixed data types |
| `gzip` | Slower | Better | Long-term storage |

## Performance Tuning

### High-Throughput Scenarios

```python
# Optimized for speed
fast_config = CacheConfig(
    compression=CompressionConfig(
        pickle_compression_codec="lz4",
        pickle_compression_level=1,
        blosc2_array_codec="lz4"
    ),
    serialization=SerializationConfig(
        enable_collections=False,        # Skip expensive introspection
        max_tuple_recursive_length=2
    ),
    handlers=HandlerConfig(
        handler_priority=["numpy_arrays", "object_pickle"]  # Minimal handlers
    )
)
```

### Storage-Optimized Scenarios

```python
# Optimized for compression
compact_config = CacheConfig(
    compression=CompressionConfig(
        pickle_compression_codec="zstd",
        pickle_compression_level=9,      # Maximum compression
        blosc2_array_codec="zstd",
        parquet_compression="gzip"
    ),
    storage=CacheStorageConfig(
        max_cache_size_mb=50000,         # Large cache with high compression
        cleanup_on_init=True
    )
)
```

## Security Considerations

### Cache Entry Signing

Enable cryptographic signing for cache integrity:

```python
# Secure production configuration
secure_config = CacheConfig(
    security=SecurityConfig(
        enable_entry_signing=True,
        security_level="enhanced",       # Default security level
        use_in_memory_key=True,          # No key persistence
        delete_invalid_signatures=True   # Auto-cleanup tampered entries
    )
)
```

For comprehensive security configuration, see the [Security Guide](SECURITY.md).

### Sensitive Parameters

```python
# Disable parameter storage for sensitive data
secure_config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="sqlite",
        store_cache_key_params=False     # Don't store sensitive params
    )
)

# Use with API keys, passwords, etc.
cache = cacheness(secure_config)
cache.put(data, api_key="secret", user_id="12345")  # api_key not stored in metadata
```

### File Integrity

```python
# Enable integrity checking
integrity_config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="sqlite",
        verify_cache_integrity=True      # Verify file hashes
    )
)
```

## Environment-Specific Configurations

### Development

```python
dev_config = CacheConfig(
    cache_dir="./dev_cache",
    default_ttl_hours=2,                 # Short TTL
    cleanup_on_init=True,                # Clean start
    metadata_backend="json",             # Simple backend
    max_cache_size_mb=500               # Small cache
)
```

### Production

```python
prod_config = CacheConfig(
    storage=CacheStorageConfig(
        cache_dir="/var/cache/app",
        max_cache_size_mb=20000,
        cleanup_on_init=False            # Preserve cache across restarts
    ),
    metadata=CacheMetadataConfig(
        backend="sqlite",
        verify_cache_integrity=True,
        database_url="/var/cache/app/metadata.db"
    ),
    compression=CompressionConfig(
        pickle_compression_level=6,      # Balanced compression
        use_blosc2_arrays=True
    ),
    security=SecurityConfig(
        enable_entry_signing=True,
        security_level="enhanced",       # Good security/performance balance
        use_in_memory_key=True,          # No key persistence in production
        delete_invalid_signatures=True   # Auto-cleanup
    ),
    default_ttl_hours=168               # 1 week
)
```

### Testing

```python
test_config = CacheConfig(
    cache_dir="./test_cache",
    default_ttl_hours=None,             # No expiration during tests
    cleanup_on_init=True,               # Fresh cache for each test
    metadata_backend="json",            # Simple, no external dependencies
    max_cache_size_mb=100              # Small test cache
)
```

## Advanced Object Serialization

### Dill Serialization for Complex Objects

Cacheness supports advanced Python objects using the **dill** library for enhanced serialization capabilities:

```python
# Enable/disable dill fallback (enabled by default)
config = CacheConfig(
    handlers=HandlerConfig(
        enable_dill_fallback=True       # Use dill for complex objects
    )
)

# Examples of objects that require dill:
cache = cacheness(config)

# Functions and closures
def create_multiplier(factor):
    return lambda x: x * factor

multiplier = create_multiplier(2.5)
cache.put(multiplier, operation="multiply", factor=2.5)

# Partial functions
from functools import partial
import operator
multiply_by_10 = partial(operator.mul, 10)
cache.put(multiply_by_10, operation="partial_multiply")

# Complex nested functions
@cached(ttl_hours=24)
def create_complex_processor():
    import numpy as np
    base_value = np.random.rand()
    return lambda data: data + base_value * np.random.normal(0, 0.1)
```

**Configuration options:**
```python
# Strict mode: only standard pickle-compatible objects
strict_config = CacheConfig(
    handlers=HandlerConfig(
        enable_dill_fallback=False      # Disable dill, fail on complex objects
    )
)

# When dill is disabled, functions/lambdas will raise ValueError
```

### TensorFlow Tensor Support

Native TensorFlow tensor caching with optimized storage:

```python
# Enable TensorFlow handler (disabled by default)
tf_config = CacheConfig(
    handlers=HandlerConfig(
        enable_tensorflow_tensors=True  # Enable native tensor caching
    )
)

import tensorflow as tf
cache = cacheness(tf_config)

# Cache TensorFlow tensors with preserved metadata
tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
cache.put(tensor, model="cnn", layer="conv1", weights="initial")

# Retrieve maintains all tensor properties
cached_tensor = cache.get(model="cnn", layer="conv1", weights="initial")
assert cached_tensor.dtype == tf.float32
assert cached_tensor.shape == (2, 2)
```

**Storage details:**
- Uses `.b2tr` file extension (TensorFlow tensor format)
- Blosc2 compression for efficient storage
- Preserves dtype, shape, and device information
- GPU memory efficient loading

**Why disabled by default:**
- TensorFlow import adds ~2-3 second startup overhead
- Potential system compatibility issues with mutex locks
- Large memory footprint for simple caching needs

**Enabling for production:**
```python
# Only enable if you regularly cache TensorFlow tensors
production_config = CacheConfig(
    handlers=HandlerConfig(
        enable_tensorflow_tensors=True,
        handler_priority=[
            "tensorflow_tensors",       # Prioritize tensor handling
            "numpy_arrays",
            "pandas_dataframes",
            "object_pickle"
        ]
    ),
    compression=CompressionConfig(
        blosc2_array_codec="zstd",      # Better compression for tensors
        blosc2_array_clevel=7           # Higher compression level
    )
)
```

## Troubleshooting Configuration

### Common Issues

**Cache misses with expected data:**
```python
# Enable parameter storage for debugging
debug_config = CacheConfig(
    metadata=CacheMetadataConfig(
        store_cache_key_params=True      # See what parameters were used
    )
)

# Check entries
entries = cache.list_entries()
for entry in entries:
    print(f"Entry: {entry['cache_key_params']}")
```

**Performance issues:**
```python
# Monitor cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Backend: {stats['backend_type']}")

# Use SQLite for large caches
if stats['total_entries'] > 1000:
    # Switch to SQLite backend
    pass
```

**Storage issues:**
```python
# Monitor cache size
stats = cache.get_stats()
if stats['total_size_mb'] > config.max_cache_size_mb * 0.9:
    print("Cache nearly full - consider cleanup or size increase")
```

### Debug Configuration

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

debug_config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="sqlite",
        store_cache_key_params=True,
        verify_cache_integrity=True
    )
)

cache = cacheness(debug_config)
# Cache operations will now log detailed information
```
