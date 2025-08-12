# Performance Guide

Comprehensive guide to optimizing cacheness performance for different use cases and workloads.

## Performance Overview

Cacheness is designed for high-performance caching with several automatic optimizations:

- **Intelligent Compression**: Automatic codec selection based on data type
- **Parallel Processing**: Multi-threaded operations for large directories
- **Backend Optimization**: SQLite for metadata-heavy workloads, JSON for simplicity
- **Storage Format Selection**: Optimal formats for different data types

## Benchmarks

### Metadata Backend Performance

| Operation | SQLite (small) | SQLite (large) | JSON (small) | JSON (large) |
|-----------|----------------|----------------|--------------|--------------|
| `list_entries()` | 0.5ms | **2.3ms** | 0.3ms | 1.2s |
| `get_stats()` | 0.8ms | **4.1ms** | 0.5ms | 850ms |
| `cleanup_expired()` | 1.2ms | **12ms** | 0.8ms | 1.5s |

*Small: <1k entries, Large: 10k+ entries*

**Key Insight**: SQLite provides 10-500x performance improvement for large caches.

### Storage Format Performance

| Data Type | Format | Write Speed | Read Speed | Compression |
|-----------|--------|-------------|------------|-------------|
| NumPy Arrays | NPZ + Blosc2 | **4x faster** | **4x faster** | 60-80% reduction |
| DataFrames | Parquet + LZ4 | **2x faster** | **3x faster** | 40-60% reduction |
| Objects | Pickle + ZSTD | **1.5x faster** | **1.5x faster** | 30-50% reduction |

### Serialization Performance

| Object Type | Avg Time (μs) | Cache Key Quality | Use Case |
|-------------|---------------|-------------------|----------|
| Basic types | 0.10-0.14 | ⭐⭐⭐⭐⭐ | Primitives, fast path |
| Collections | 0.59-0.89 | ⭐⭐⭐⭐⭐ | Lists, dicts, sets |
| Objects w/ `__dict__` | 0.81 | ⭐⭐⭐⭐⭐ | Custom classes |
| Large tuples | 0.34 | ⭐⭐⭐ | Performance fallback |
| NumPy arrays | 1.67 | ⭐⭐⭐⭐⭐ | Scientific data |

## Optimization Strategies

### 1. Choose the Right Backend

**SQLite Backend** (Recommended for production):
```python
from cacheness import CacheConfig
from cacheness.config import CacheMetadataConfig

# High-performance configuration
config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="sqlite",
        database_url="./fast_cache.db"
    ),
    max_cache_size_mb=10000
)
```

**When to use SQLite:**
- Large number of cache entries (>1000)
- Frequent metadata operations
- Production deployments
- Custom metadata queries

**JSON Backend**:
```python
config = CacheConfig(
    metadata=CacheMetadataConfig(backend="json")
)
```

**When to use JSON:**
- Small caches (<1000 entries)
- Development/testing
- Simple deployments

### 2. Optimize Compression Settings

**Speed-Optimized Configuration**:
```python
from cacheness.config import CompressionConfig

speed_config = CacheConfig(
    compression=CompressionConfig(
        pickle_compression_codec="lz4",    # Fastest compression
        pickle_compression_level=1,        # Minimal compression overhead
        blosc2_array_codec="lz4",         # Fast array compression
        parquet_compression="lz4"         # Fast DataFrame compression
    )
)
```

**Compression-Optimized Configuration**:
```python
compression_config = CacheConfig(
    compression=CompressionConfig(
        pickle_compression_codec="zstd",   # Excellent compression
        pickle_compression_level=7,        # High compression
        blosc2_array_codec="zstd",        # Better array compression
        parquet_compression="gzip"        # Better DataFrame compression
    )
)
```

**Balanced Configuration** (Default):
```python
balanced_config = CacheConfig(
    compression=CompressionConfig(
        pickle_compression_codec="zstd",   # Good compression
        pickle_compression_level=5,        # Balanced level
        blosc2_array_codec="lz4",         # Fast array access
        parquet_compression="lz4"         # Fast DataFrame access
    )
)
```

### 3. Tune Serialization Performance

**Performance-Optimized Serialization**:
```python
from cacheness.config import SerializationConfig

perf_config = CacheConfig(
    serialization=SerializationConfig(
        enable_collections=False,           # Skip expensive collection analysis
        enable_object_introspection=False,  # Skip __dict__ inspection
        max_tuple_recursive_length=2,       # Limit tuple recursion
        max_collection_depth=3              # Limit nesting depth
    )
)
```

**Quality-Optimized Serialization**:
```python
quality_config = CacheConfig(
    serialization=SerializationConfig(
        enable_collections=True,            # Full collection analysis
        enable_object_introspection=True,   # Deep object inspection
        max_tuple_recursive_length=50,      # Allow deep recursion
        max_collection_depth=20             # Allow deep nesting
    )
)
```

### 4. Handler Priority Optimization

**Array-Heavy Workloads**:
```python
from cacheness.config import HandlerConfig

array_config = CacheConfig(
    handlers=HandlerConfig(
        handler_priority=[
            "numpy_arrays",        # Process arrays first
            "object_pickle"        # Minimal fallback
        ],
        enable_pandas_dataframes=False,  # Disable unused handlers
        enable_polars_dataframes=False
    )
)
```

**DataFrame-Heavy Workloads**:
```python
df_config = CacheConfig(
    handlers=HandlerConfig(
        handler_priority=[
            "pandas_dataframes",   # Process DataFrames first
            "polars_dataframes",
            "pandas_series",
            "numpy_arrays",
            "object_pickle"
        ]
    )
)
```

## Use Case Optimizations

### High-Frequency API Caching

```python
api_config = CacheConfig(
    storage=CacheStorageConfig(
        cache_dir="./api_cache",
        max_cache_size_mb=2000,
        cleanup_on_init=True
    ),
    metadata=CacheMetadataConfig(
        backend="sqlite",              # Fast metadata ops
        store_cache_key_params=False   # Skip parameter storage for speed
    ),
    compression=CompressionConfig(
        pickle_compression_codec="lz4",  # Fastest compression
        pickle_compression_level=1
    ),
    serialization=SerializationConfig(
        enable_collections=False,        # Skip deep analysis
        max_tuple_recursive_length=3
    ),
    default_ttl_hours=6
)

# Use with decorators for maximum performance
from cacheness import cached

@cached(cache_instance=cacheness(api_config), ttl_hours=1)
def fetch_api_data(endpoint, params):
    # Fast caching with minimal overhead
    return api_call(endpoint, params)
```

### ML Model Training Pipeline

```python
ml_config = CacheConfig(
    storage=CacheStorageConfig(
        cache_dir="./ml_cache",
        max_cache_size_mb=20000,       # Large cache for datasets
        cleanup_on_init=False          # Preserve models across sessions
    ),
    metadata=CacheMetadataConfig(
        backend="sqlite",              # Handle many experiments
        store_cache_key_params=True,   # Track experiment parameters
        verify_cache_integrity=True    # Ensure model integrity
    ),
    compression=CompressionConfig(
        use_blosc2_arrays=True,        # Optimal for numeric data
        pickle_compression_codec="zstd",
        pickle_compression_level=6,     # Good compression for models
        blosc2_array_codec="lz4"       # Fast array access
    ),
    handlers=HandlerConfig(
        handler_priority=[
            "numpy_arrays",            # Prioritize arrays
            "pandas_dataframes",       # Then DataFrames
            "object_pickle"            # Models and misc objects
        ]
    ),
    default_ttl_hours=168             # 1 week
)
```

### Large Dataset Processing

```python
data_config = CacheConfig(
    storage=CacheStorageConfig(
        cache_dir="/fast_ssd/data_cache",  # Use fast storage
        max_cache_size_mb=50000,           # Very large cache
        cleanup_on_init=False
    ),
    metadata=CacheMetadataConfig(
        backend="sqlite",
        database_url="/fast_ssd/metadata.db"  # Fast storage for metadata
    ),
    compression=CompressionConfig(
        use_blosc2_arrays=True,
        blosc2_array_codec="lz4",          # Fast decompression
        parquet_compression="snappy",       # Good for mixed data
        pickle_compression_codec="zstd",
        pickle_compression_level=3          # Balanced for large objects
    ),
    default_ttl_hours=72
)
```

## Automatic Optimizations

### Path Content Hashing

The library automatically optimizes path hashing based on directory size:

| Directory Size | File Count | Processing Method | Performance |
|---------------|------------|-------------------|-------------|
| < 4GB | < 80 files | Sequential | 0.001-0.2s |
| ≥ 4GB OR ≥ 80 files | Any | Parallel | 1.2x-1.6x speedup |

```python
from pathlib import Path

# Automatically uses optimal processing
large_dataset = Path("./datasets/images/")  # 500+ files → parallel
small_config = Path("./config/")            # 10 files → sequential

cache.put(data, dataset_path=large_dataset)  # Automatic parallel hashing
cache.put(config, config_path=small_config)  # Automatic sequential hashing
```

### Intelligent Compression

The library automatically adjusts compression based on data characteristics:

- **Small data** (<1MB): Lower compression levels, fewer threads
- **Medium data** (1-10MB): Moderate compression and threading
- **Large data** (>10MB): Higher compression levels, more threads

### Fallback Mechanisms

Performance optimizations with graceful fallbacks:

1. **orjson** → built-in json (1.5-5x performance degradation)
2. **Polars** → Pandas → Pickle (graceful feature degradation)
3. **SQLite** → JSON (10-500x performance degradation for large caches)
4. **Blosc2** → NPZ → Pickle (compression efficiency degradation)

## Performance Monitoring

### Built-in Statistics

```python
# Get comprehensive performance metrics
stats = cache.get_stats()

print(f"Total entries: {stats['total_entries']}")
print(f"Total size: {stats['total_size_mb']:.2f} MB")
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Backend: {stats['backend_type']}")
print(f"Average entry size: {stats['avg_entry_size_mb']:.3f} MB")

# Check for performance issues
if stats['hit_rate'] < 0.5:
    print("Low hit rate - check cache TTL and key consistency")

if stats['total_size_mb'] > config.max_cache_size_mb * 0.9:
    print("Cache nearly full - consider cleanup or size increase")
```

### Debug Logging

```python
import logging

# Enable performance logging
logging.basicConfig(level=logging.DEBUG)

# Cache operations will log timing information
cache.put(large_array, dataset="performance_test")
# DEBUG: Stored array (15.2MB) in 0.234s using numpy_arrays handler
# DEBUG: Compression ratio: 3.2x (ZSTD level 5)

cache.get(dataset="performance_test")
# DEBUG: Retrieved array (15.2MB) in 0.089s using numpy_arrays handler
```

### Custom Performance Tracking

```python
import time
from cacheness import cached

class PerformanceTracker:
    def __init__(self):
        self.metrics = []
    
    def time_operation(self, name, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        self.metrics.append({
            'operation': name,
            'elapsed': elapsed,
            'timestamp': time.time()
        })
        
        return result

tracker = PerformanceTracker()

# Measure cache performance
@cached(ttl_hours=24)
def expensive_computation(data):
    time.sleep(1)  # Simulate work
    return len(data)

# First call (cache miss)
result1 = tracker.time_operation(
    "cache_miss", 
    expensive_computation, 
    list(range(1000))
)

# Second call (cache hit)
result2 = tracker.time_operation(
    "cache_hit", 
    expensive_computation, 
    list(range(1000))
)

print(f"Cache miss: {tracker.metrics[0]['elapsed']:.3f}s")
print(f"Cache hit: {tracker.metrics[1]['elapsed']:.3f}s")
print(f"Speedup: {tracker.metrics[0]['elapsed'] / tracker.metrics[1]['elapsed']:.1f}x")
```

## Performance Best Practices

### 1. Cache Key Design

```python
# Good: Hierarchical, consistent parameters
cache.put(model, 
          project="customer_analysis",
          model_type="xgboost", 
          version="v2.1",
          features="demographic")

# Avoid: Inconsistent parameter names
cache.put(model, proj="customer", model="xgb", ver="2.1")
```

### 2. TTL Strategy

```python
# Different TTL for different data types
@cached(ttl_hours=1)      # Short: real-time data
def get_stock_price(symbol): pass

@cached(ttl_hours=24)     # Medium: daily data
def get_weather_forecast(city): pass

@cached(ttl_hours=168)    # Long: stable data
def train_ml_model(data): pass

@cached(ttl_hours=None)   # Permanent: reference data
def load_country_codes(): pass
```

### 3. Cache Size Management

```python
# Monitor and manage cache size
def manage_cache_size(cache, target_usage=0.8):
    stats = cache.get_stats()
    current_usage = stats['total_size_mb'] / cache.config.max_cache_size_mb
    
    if current_usage > target_usage:
        # Clean expired entries first
        cache.cleanup_expired()
        
        # If still over limit, remove oldest entries
        if current_usage > target_usage:
            entries = cache.list_entries()
            oldest_entries = sorted(entries, key=lambda x: x['created_at'])
            
            for entry in oldest_entries[:len(oldest_entries)//4]:  # Remove 25%
                cache.invalidate_by_cache_key(entry['cache_key'])

# Run periodically
manage_cache_size(cache)
```

### 4. Bulk Operations

```python
# Efficient bulk caching
def cache_multiple_results(datasets, processor):
    for dataset_name, data in datasets.items():
        cache.put(
            processor(data),
            dataset=dataset_name,
            processor_version="v1.0",
            batch=True  # Group related entries
        )

# Efficient bulk retrieval
def get_multiple_results(dataset_names):
    results = {}
    for name in dataset_names:
        result = cache.get(dataset=name, processor_version="v1.0")
        if result is not None:
            results[name] = result
    return results
```

## Troubleshooting Performance Issues

### Common Performance Problems

**1. Low Cache Hit Rate**
```python
stats = cache.get_stats()
if stats['hit_rate'] < 0.5:
    # Check parameter consistency
    entries = cache.list_entries()
    for entry in entries[-10:]:  # Check recent entries
        print(f"Parameters: {entry.get('cache_key_params', {})}")
```

**2. Slow Metadata Operations**
```python
# Switch to SQLite backend
if stats['total_entries'] > 1000 and stats['backend_type'] == 'json':
    print("Consider switching to SQLite backend for better performance")
    
    new_config = CacheConfig(
        metadata=CacheMetadataConfig(backend="sqlite")
    )
```

**3. High Memory Usage**
```python
# Use streaming for large objects
def process_large_dataset(data_path):
    # Instead of loading everything into memory
    # large_data = pd.read_csv(data_path)  # Memory intensive
    
    # Use chunked processing
    chunks = pd.read_csv(data_path, chunksize=10000)
    results = []
    
    for i, chunk in enumerate(chunks):
        chunk_result = cache.get(data_path=data_path, chunk_id=i)
        if chunk_result is None:
            chunk_result = process_chunk(chunk)
            cache.put(chunk_result, data_path=data_path, chunk_id=i)
        results.append(chunk_result)
    
    return combine_results(results)
```

**4. Storage Performance Issues**
```python
# Use fast storage for cache
fast_config = CacheConfig(
    storage=CacheStorageConfig(
        cache_dir="/fast_ssd/cache",     # SSD storage
        max_cache_size_mb=20000
    ),
    metadata=CacheMetadataConfig(
        database_url="/fast_ssd/metadata.db"  # Fast storage for metadata
    )
)
```
