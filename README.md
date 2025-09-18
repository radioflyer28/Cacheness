# Cacheness

Fast Python disk cache with key-value store hashing and a "cachetools-like" decorator. Caches NumPy/Pandas/Polars natively; other objects use pickle. Uses Blosc2 for fast compression.

**Key Features:**
- **Function decorators** for automatic caching with `@cached`
- **Multi-format storage** with automatic type detection and optimal format selection
- **SQL pull-through cache** for intelligent API and database caching with automatic gap detection
- **Key-based caching** using xxhash (XXH3_64) for fast, deterministic cache keys
- **Advanced compression** using Blosc2 and LZ4 for fast compression
- **Multiple backends** with SQLite and JSON metadata support
- **Performance-optimized** parallel processing for hashing large directories

## Quick Start

### Installation

```bash
# Basic installation
pip install cacheness

# Recommended (includes SQLAlchemy + performance optimizations)
pip install cacheness[recommended]

# Full installation with DataFrame support
pip install cacheness[recommended,dataframes]
```

### Basic Usage

```python
from cacheness import cacheness

# Create a cache instance
cache = cacheness()

# Store data using keyword arguments as cache keys
cache.put({"results": [1, 2, 3]}, 
          model="xgboost", 
          dataset="customer_data", 
          version="1.0")

# Retrieve data using the same key parameters
data = cache.get(model="xgboost", dataset="customer_data", version="1.0")
print(data)  # {"results": [1, 2, 3]}
```

### Function Decorators

```python
from cacheness import cached

@cached(ttl_hours=24)
def expensive_computation(n):
    """This function will be automatically cached."""
    import time
    time.sleep(2)  # Simulate expensive work
    return n ** 2

# First call takes 2 seconds
result1 = expensive_computation(5)  # Computed and cached

# Second call returns instantly from cache
result2 = expensive_computation(5)  # Retrieved from cache
```

### SQL Pull-Through Cache

For intelligent API caching with automatic gap detection and database backend selection:

```python
from cacheness.sql_cache import SqlCache
from sqlalchemy import Float, Integer

# Simple function-based approach (no inheritance required!)
def fetch_stock_data(symbol, start_date, end_date):
    # Your API logic here - return DataFrame
    return api_client.get_historical_data(symbol, start_date, end_date)

# Create cache with builder pattern - chooses optimal database backend
cache = SqlCache.for_timeseries(
    "stocks.db",  # Uses DuckDB for analytical workloads
    data_fetcher=fetch_stock_data,
    price=Float,
    volume=Integer
)

# Automatic gap detection and caching
data = cache.get_data(symbol="AAPL", start_date="2024-01-01", end_date="2024-01-31")
```

# SQLite for transactional/row-wise operations (ACID compliance, moderate concurrency)
cache = SqlCache.with_sqlite("stocks.db", stock_table, StockAdapter())

# PostgreSQL for production environments (high concurrency, advanced features)
cache = SqlCache.with_postgresql("postgresql://...", stock_table, StockAdapter())

# Get data - automatically fetches missing data gaps
data = cache.get_data(symbol="AAPL", start_date="2024-01-01")
```

## Intelligent Storage & Access Patterns

Cacheness automatically optimizes storage and backend selection based on your data and access patterns.

### UnifiedCache - Intelligent Function Caching
```python
from cacheness import cached

# Automatically chooses optimal storage format:
# • DataFrames → Parquet format
# • NumPy arrays → Blosc compression  
# • Custom objects → Pickle serialization
# • API responses → LZ4 compression

@cached()  # Works with any Python object
def process_data(df):
    return df.groupby('category').sum()

@cached.for_api()  # Optimized for API responses
def fetch_user_data(user_id):
    return requests.get(f"/api/users/{user_id}").json()
```

### SqlCache - Access-Pattern-Optimized Caching
```python
from cacheness.sql_cache import SqlCache
from sqlalchemy import Float, Integer

# Individual record lookups → SQLite (row-wise optimization)
user_cache = SqlCache.for_lookup_table(
    "users.db", 
    data_fetcher=fetch_user_profile,
    user_id=Integer,
    name=String(100)
)

# Bulk analytics → DuckDB (columnar optimization)  
analytics_cache = SqlCache.for_analytics_table(
    "analytics.db",
    data_fetcher=fetch_sales_data,
    department=String(50),
    revenue=Float
)

# Real-time data → SQLite (fast updates)
realtime_cache = SqlCache.for_realtime_timeseries(
    "prices.db",
    data_fetcher=fetch_live_prices,
    price=Float,
    volume=Integer
)

# Historical analysis → DuckDB (analytical queries)
historical_cache = SqlCache.for_timeseries(
    "history.db", 
    data_fetcher=fetch_historical_data,
    price=Float,
    volume=Integer
)
```

### Backend Selection Guide

| **Access Pattern** | **Method** | **Database** | **Optimized For** |
|-------------------|------------|--------------|-------------------|
| Individual lookups | `for_lookup_table()` | SQLite | Row-wise access, transactions |
| Bulk analytics | `for_analytics_table()` | DuckDB | Columnar queries, aggregations |
| Real-time data | `for_realtime_timeseries()` | SQLite | Fast updates, recent data |
| Historical analysis | `for_timeseries()` | DuckDB | Time-series analytics |

## Core Concepts

### Key-Based Caching System

The library uses **xxhash (XXH3_64)** to generate deterministic cache keys from your parameters:

```python
# These are equivalent - parameter order doesn't matter
cache.put(data, model="xgboost", dataset="train", version="1.0")
cache.put(data, version="1.0", model="xgboost", dataset="train")  # Same cache key

# Different parameters = different cache entries
cache.put(data1, model="xgboost", dataset="train")  # Key: abc123...
cache.put(data2, model="lightgbm", dataset="train") # Key: def456...
cache.put(data3, model="xgboost", dataset="test")   # Key: ghi789...
```

### Supported Data Types

| Data Type | Storage Format | Compression | Benefits |
|-----------|---------------|-------------|----------|
| NumPy arrays | NPZ or Blosc2 | LZ4/ZSTD | 60-80% size reduction, 4x faster I/O |
| DataFrames & Series | Parquet | LZ4 | 40-60% size reduction, columnar efficiency |
| TensorFlow tensors* | Blosc2 | LZ4/ZSTD | Native tensor format, GPU memory efficient |
| Python objects | Pickle + Blosc | LZ4 | 30-50% size reduction, universal compatibility |
| Complex objects** | Dill + Blosc | LZ4 | Functions, lambdas, advanced serialization |

*TensorFlow handler disabled by default due to import overhead  
**Includes functions, lambdas, closures, and other objects that pickle cannot handle

### Configuration

```python
from cacheness import cacheness, CacheConfig

# Simple configuration
config = CacheConfig(
    cache_dir="./my_cache",
    metadata_backend="sqlite",     # "sqlite" (production), "json" (dev), or "auto"  
    default_ttl_hours=48,          # Default TTL for entries
    max_cache_size_mb=5000,        # Maximum cache size
)

cache = cacheness(config)

# See docs/BACKEND_SELECTION.md for choosing between JSON and SQLite backends
```

## Quick Examples

### Basic Usage

```python
import numpy as np
import pandas as pd

# Cache NumPy arrays (automatically compressed with Blosc2)
features = np.random.rand(1000, 50)
cache.put(features, dataset="training", preprocessing="standard_scaled")

# Cache DataFrames (automatically stored as Parquet)
df = pd.DataFrame({"user_id": range(1000), "amount": np.random.exponential(50, 1000)})
cache.put(df, source="transactions", date_range="2024_q1")

# Retrieve cached data
cached_features = cache.get(dataset="training", preprocessing="standard_scaled")
cached_df = cache.get(source="transactions", date_range="2024_q1")
```

### Decorator Usage

```python
# Cache function results with TTL
@cached(ttl_hours=6)
def fetch_weather_data(city, units="metric"):
    return api_call(f"weather/{city}", units=units)

# Custom cache instance for specific use cases
ml_cache = cacheness(CacheConfig(cache_dir="./ml_cache", default_ttl_hours=168))

@cached(cache_instance=ml_cache)
def train_model(data, hyperparams):
    return expensive_model_training(data, hyperparams)
```

### Advanced Object Serialization

Cacheness supports advanced Python objects using **dill** for enhanced serialization:

> ⚠️ **IMPORTANT WARNINGS**: Dill serialization can execute arbitrary code and may introduce bugs if class definitions change between cache storage and retrieval. See [Security Considerations](#dill-security-considerations) below.

```python
# Cache complex classes with large datasets (common ML use case)
from dataclasses import dataclass
import numpy as np

@dataclass
class ModelState:
    weights: np.ndarray
    metadata: dict
    preprocessing_func: callable
    
    def __post_init__(self):
        # Complex initialization with closures
        scale_factor = self.metadata.get('scale_factor', 1.0)
        self.preprocessing_func = lambda x: x * scale_factor + np.random.normal(0, 0.01)

# Create model state with large data
model_state = ModelState(
    weights=np.random.randn(1000, 500),  # Large weight matrix
    metadata={'epochs': 100, 'scale_factor': 2.5, 'accuracy': 0.94},
    preprocessing_func=None  # Will be set in __post_init__
)

# Cache the entire initialized object (not possible with standard pickle)
cache.put(model_state, model="resnet", checkpoint="epoch_100")

# Retrieve and use - all data and functions preserved
cached_model = cache.get(model="resnet", checkpoint="epoch_100")
processed_data = cached_model.preprocessing_func(test_input)
```

#### Dill Security Considerations

**🚨 Critical Security Risks:**
- **Code Execution**: Dill can execute arbitrary code during deserialization
- **Cache Tampering**: Malicious modification of cache files can compromise your application
- **Version Conflicts**: Class definition changes can cause crashes or silent bugs

**🛡️ Safe Usage Patterns:**

```python
# ✅ SAFE: Validate cached objects after retrieval
def safe_cache_get(cache, **kwargs):
    try:
        cached_obj = cache.get(**kwargs)
        if cached_obj is None:
            return None
            
        # Validate object type and critical attributes
        if not isinstance(cached_obj, ModelState):
            raise ValueError("Cached object has unexpected type")
        
        if not hasattr(cached_obj, 'weights') or not hasattr(cached_obj, 'preprocessing_func'):
            raise ValueError("Cached object missing required attributes")
            
        # Test critical functionality
        test_input = np.array([[1.0, 2.0]])
        _ = cached_obj.preprocessing_func(test_input)
        
        return cached_obj
    except Exception as e:
        print(f"⚠️ Cache validation failed: {e}")
        return None  # Force recreation

# ✅ SAFE: Version your cached classes
@dataclass 
class ModelState:
    _version: str = "1.0"  # Track class version
    weights: np.ndarray = None
    # ... other fields
    
    def __post_init__(self):
        if self._version != "1.0":
            raise ValueError(f"Incompatible class version: {self._version}")

# ❌ UNSAFE: Never cache objects from untrusted sources
# ❌ UNSAFE: Don't cache in production without validation
# ❌ UNSAFE: Don't ignore cache retrieval errors
```

**Configuration:**
```python
# Enable/disable dill fallback (enabled by default)
config = CacheConfig(enable_dill_fallback=True)
cache = cacheness(config)

# For production: disable dill for security (recommended)
config_secure = CacheConfig(enable_dill_fallback=False)

# When disabled, only standard pickle-compatible objects work
config_strict = CacheConfig(enable_dill_fallback=False)
```

> 🔒 **Production Recommendation**: Disable dill in production environments (`enable_dill_fallback=False`) to eliminate code execution risks. Use dill only in trusted development environments with proper validation.

### TensorFlow Tensor Support

Native TensorFlow tensor caching with optimized storage:

```python
import tensorflow as tf

# Enable TensorFlow handler (disabled by default due to import overhead)
config = CacheConfig(enable_tensorflow_tensors=True)
cache = cacheness(config)

# Cache TensorFlow tensors directly
tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
cache.put(tensor, model="cnn", layer="conv1", weights="initial")

# Retrieve maintains tensor properties
cached_tensor = cache.get(model="cnn", layer="conv1", weights="initial")
print(cached_tensor.dtype)  # <dtype: 'float32'>
print(cached_tensor.shape)  # (2, 2)
```

**Storage Benefits:**
- Native tensor format with Blosc2 compression
- Preserves dtype, shape, and tensor metadata
- GPU memory efficient loading
- File extension: `.b2tr` (TensorFlow tensor format)

## Advanced Features

- **Cache Entry Signing**: HMAC-SHA256 signatures for metadata integrity protection
- **Custom Metadata**: Rich metadata tracking with SQLAlchemy ORM for experiment tracking and data lineage
- **Path Content Hashing**: Automatic content-based hashing for cache key, using key values that contain file & directory paths
- **Multi-format Storage**: Optimized formats for different data types (NPZ, Parquet, compressed pickle)
- **Intelligent Compression**: Automatic codec selection and parallel processing for large datasets

## Security and Integrity

### Cache Entry Signing

Protect cache metadata from tampering with HMAC-SHA256 signatures:

```python
from cacheness import cacheness, CacheConfig, SecurityConfig

# Default: Signing enabled with enhanced security level
cache = cacheness()  # Entry signing active by default

# Custom security configuration
config = CacheConfig(
    security=SecurityConfig(
        enable_entry_signing=True,        # Enable/disable signing
        delete_invalid_signatures=True,   # Auto-cleanup tampered entries
        use_in_memory_key=False,          # Persistent vs in-memory keys
        allow_unsigned_entries=True,      # Backward compatibility
        custom_signed_fields=None         # Custom fields to sign (uses default if None)
    )
)
cache = cacheness(config)
```

**Default Signed Fields:**
Signs 6 fields for enhanced security: `cache_key`, `file_hash`, `data_type`, `file_size`, `created_at`, `prefix`

**Custom Field Selection:**
```python
# Sign only critical fields for minimal overhead
config = CacheConfig(
    security=SecurityConfig(
        custom_signed_fields=["cache_key", "file_hash", "data_type", "file_size"]
    )
)

# Sign all available fields for maximum security
config = CacheConfig(
    security=SecurityConfig(
        custom_signed_fields=["cache_key", "file_hash", "data_type", "file_size", 
                             "created_at", "prefix", "description", "actual_path"]
    )
)

### In-Memory Signing Keys

For enhanced security, use in-memory-only signing keys:

```python
# High-security configuration
config = CacheConfig(
    security=SecurityConfig(
        use_in_memory_key=True,           # No key files on disk
        delete_invalid_signatures=True    # Clean up on restart
    )
)
cache = cacheness(config)

# Benefits:
# ✅ No cryptographic material persisted to disk
# ✅ Cache entries invalidated on process restart
# ✅ Ideal for containerized/high-security environments
# ✅ Perfect for temporary/session-based caching
```

**Key Management:**
- **Persistent Keys** (default): Key stored in `cache_signing_key.bin`, cache survives restarts
- **In-Memory Keys**: Generated per process, cache invalidated on restart, enhanced security

### Automatic Signature Verification

All cache retrievals automatically verify signatures:

```python
# Signatures verified on every cache hit
data = cache.get(model="xgboost", dataset="training")

# Invalid signatures are handled based on configuration:
# delete_invalid_signatures=True  → Entry deleted, cache miss returned
# delete_invalid_signatures=False → Warning logged, data still returned
```

**Use Cases:**
- **Development**: `delete_invalid_signatures=False` for debugging
- **Production**: `delete_invalid_signatures=True` for automatic cleanup
- **High Security**: `use_in_memory_key=True` + `delete_invalid_signatures=True`

## Real-World Examples

Comprehensive examples are available in the [`examples/`](examples/) directory:

- **[API Request Caching](examples/api_request_caching.py)** - Intelligent API caching with TTL strategies
- **[Stock Cache Example](examples/stock_cache_example.py)** - SQL pull-through cache with Yahoo Finance integration
- **[ML Pipeline Caching](examples/ml_pipeline_caching.py)** - Multi-stage ML training pipeline caching
- **[S3 Caching](examples/s3_caching.py)** - Caching of S3 file downloads with ETag (remote file hash) validation
- **[Custom Metadata Demo](examples/custom_metadata_demo.py)** - Advanced metadata tracking workflows
- **[Configurable Serialization](examples/configurable_serialization_demo.py)** - Custom serialization examples

## Cache Management

```python
# Get cache statistics
stats = cache.get_stats()
print(f"Total entries: {stats['total_entries']}")
print(f"Total size: {stats['total_size_mb']:.2f} MB")
print(f"Hit rate: {stats.get('hit_rate', 0):.1%}")

# List and manage entries
entries = cache.list_entries()
for entry in entries:
    print(f"{entry['cache_key']}: {entry.get('description', 'No description')} ({entry['size_mb']:.2f}MB)")

# Cleanup operations
cache.cleanup_expired()                     # Remove expired entries
cache.invalidate(model="old", version="1")  # Remove specific entry
cache.clear()                               # Clear all entries
```

## Performance

### Backend Performance (10k+ entries)

| Backend | `list_entries()` | `get_stats()` | `cleanup_expired()` |
|---------|------------------|---------------|-------------------|
| **SQLite** | **2.3ms** | **4.1ms** | **12ms** |
| **JSON** | 1.2s | 850ms | 1.5s |

*SQLite provides 10-500x performance improvement for large caches*

### Storage Optimizations

- **Intelligent Compression**: Automatic codec selection (LZ4 for dataframe/series, Blosc2 for numpy arrays, ZSTD for objects)
- **Format Selection**: NPZ for arrays, Parquet for DataFrames, compressed pickle for objects
- **Graceful Fallbacks**: orjson → json, SQLite → JSON, Blosc2 → NPZ → pickle
- **Parallel Processing**: Automatic when hashing content of large directories (≥4GB or ≥80 files)

For detailed performance tuning, see the **[Performance Guide](docs/PERFORMANCE.md)**.

## Configuration

### Simple Configuration

```python
from cacheness import cacheness, CacheConfig

# Simple configuration for most use cases
config = CacheConfig(
    cache_dir="./my_cache",
    metadata_backend="sqlite",    # "sqlite" (recommended) or "json"
    default_ttl_hours=48,         # Default TTL for entries
    max_cache_size_mb=5000,       # Maximum cache size
    cleanup_on_init=True          # Clean expired entries on startup
)

cache = cacheness(config)
```

### Handler Configuration

Control which data type handlers are enabled:

```python
# Configure data type handlers
config = CacheConfig(
    # Core object handlers
    enable_object_pickle=True,          # General Python objects (default: True)
    enable_dill_fallback=True,          # Classes, Functions, lambdas, closures (default: True)
    
    # Array and tensor handlers  
    enable_numpy_arrays=True,           # NumPy arrays (default: True)
    enable_tensorflow_tensors=False,    # TensorFlow tensors (default: False)
    
    # DataFrame handlers
    enable_pandas_dataframes=True,      # Pandas DataFrames (default: True)
    enable_polars_dataframes=True,      # Polars DataFrames (default: True)
    enable_pandas_series=True,          # Pandas Series (default: True)
    enable_polars_series=True,          # Polars Series (default: True)
    
    # Performance options
    compression_threshold_bytes=1024,   # Only compress objects larger than this (default: 1024)
    enable_parallel_compression=True,   # Use multiple threads for compression (default: True)
)

cache = cacheness(config)
```

### Advanced Configuration

For complex scenarios, use detailed configuration objects:

```python
from cacheness.config import CacheStorageConfig, CacheMetadataConfig, CompressionConfig

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
        pickle_compression_codec="zstd",
        pickle_compression_level=5,
        use_blosc2_arrays=True,
        blosc2_array_codec="lz4"
    ),
    default_ttl_hours=48
)
```

For comprehensive configuration options, see the **[Configuration Guide](docs/CONFIGURATION.md)**.

## Path Content Hashing

Automatic content-based caching for files and directories:

```python
from pathlib import Path

# Files with same content but different paths share cache entries
dataset_v1 = Path("data/train.csv")
dataset_v2 = Path("backup/data/train.csv")  # Same content, different name

cache.put(features, dataset=dataset_v1)
result = cache.get(dataset=dataset_v2)  # ✓ Cache hit - same content

# Automatic parallel processing for large directories (≥4GB or ≥80 files)
large_dataset = Path("./datasets/images/")  # 500+ files → parallel hashing
small_config = Path("./config/")            # 10 files → sequential hashing
```

## Custom Metadata

For experiment tracking, data lineage, and advanced workflows:

```python
from cacheness.custom_metadata import custom_metadata_model, CustomMetadataBase
from cacheness.metadata import Base
from sqlalchemy import Column, String, Float

@custom_metadata_model("ml_experiments")
class MLExperimentMetadata(Base, CustomMetadataBase):
    experiment_id = Column(String(100), nullable=False, unique=True, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    accuracy = Column(Float, nullable=False, index=True)

# Store with structured metadata
experiment_metadata = MLExperimentMetadata(
    experiment_id="exp_001", model_type="xgboost", accuracy=0.94
)

cache.put(model, experiment="exp_001", custom_metadata={"ml_experiments": experiment_metadata})

# Query with SQL
query = cache.query_custom("ml_experiments")
high_accuracy_models = query.filter(MLExperimentMetadata.accuracy >= 0.9).all()
```

For detailed metadata workflows, see the **[Custom Metadata Guide](docs/CUSTOM_METADATA.md)**.


## Requirements

### Core Dependencies
- **Python**: ≥ 3.11
- **xxhash**: Fast hashing for cache keys

### Optional Dependencies (Graceful Fallbacks)
- **blosc2**: High-performance compression (fallback: standard compression)
- **numpy**: Optimized array handling and storage
- **sqlalchemy**: SQLite backend (10-500x faster for large caches)
- **orjson**: High-performance JSON (1.5-5x faster than built-in)
- **pandas/polars**: DataFrame and Series parquet storage optimization 
- **pyarrow**: Pandas parquet file support

Missing optional dependencies are handled gracefully with automatic fallbacks.

## Documentation

- **[Security Guide](docs/SECURITY.md)** - Cache entry signing, integrity protection, and security best practices
- **[Configuration Guide](docs/CONFIGURATION.md)** - Detailed configuration options and use cases
- **[SQL Cache Guide](docs/SQL_CACHE.md)** - Pull-through cache for APIs and time-series data
- **[Custom Metadata Guide](docs/CUSTOM_METADATA.md)** - Advanced metadata workflows with SQLAlchemy
- **[Performance Guide](docs/PERFORMANCE.md)** - Optimization strategies and benchmarks
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation

## License

GPLv3 License - see LICENSE file for details.
