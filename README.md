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
```python
from cacheness import SqlCache, SqlCacheAdapter
from sqlalchemy import Table, Column, String, Date, Float, MetaData

# Define table schema
metadata = MetaData()
stock_table = Table(
    'stock_prices', metadata,
    Column('symbol', String(10), primary_key=True),
    Column('date', Date, primary_key=True),
    Column('close', Float)
)

# Create data adapter
class StockAdapter(SqlCacheAdapter):
    def get_table_definition(self):
        return stock_table
    
    def fetch_data(self, **kwargs):
        # Your API logic here
        return fetch_stock_data(kwargs['symbol'], kwargs['start_date'])
    
    def parse_query_params(self, **kwargs):
        return {'symbol': kwargs['symbol'], 'date': {'start': kwargs['start_date']}}

# Choose backend based on workload:
# DuckDB for analytical/columnar workloads (time-series analysis, aggregations)
cache = SqlCache.with_duckdb("stocks.db", stock_table, StockAdapter())
```

# SQLite for transactional/row-wise operations (ACID compliance, moderate concurrency)
cache = SqlCache.with_sqlite("stocks.db", stock_table, StockAdapter())

# PostgreSQL for production environments (high concurrency, advanced features)
cache = SqlCache.with_postgresql("postgresql://...", stock_table, StockAdapter())

# Get data - automatically fetches missing data gaps
data = cache.get_data(symbol="AAPL", start_date="2024-01-01")
```

## Database Backend Selection

The SQL pull-through cache supports multiple database backends optimized for different workload characteristics:

### DuckDB Backend - Analytical Workloads
```python
# Optimized for columnar efficiency: time-series analysis, aggregations, data science
cache = SqlCache.with_duckdb("analytics.db", table, adapter)
```
**Best for:**
- Time-series data analysis and aggregations
- Large dataset processing with analytical queries
- Data science workflows requiring columnar operations
- OLAP-style workloads

### SQLite Backend - Transactional Workloads  
```python
# Optimized for row-wise transactional efficiency: ACID compliance, moderate concurrency
cache = SqlCache.with_sqlite("cache.db", table, adapter)
```
**Best for:**
- Row-wise operations and transactional workloads
- ACID compliance requirements
- Moderate concurrent access patterns
- Simple deployment and maintenance

### PostgreSQL Backend - Production Environments
```python
# Full-featured production database: high concurrency, advanced SQL features
cache = SqlCache.with_postgresql("postgresql://user:pass@host/db", table, adapter)
```
**Best for:**
- Production environments with high concurrency
- Advanced SQL features and complex queries
- Horizontal scaling requirements
- Enterprise-grade reliability

### Backend Comparison

| Backend | Best Use Case | Concurrency | Deployment | Query Performance |
|---------|---------------|-------------|------------|------------------|
| **DuckDB** | Analytics & Data Science | Low-Medium | Simple | Excellent (Analytical) |
| **SQLite** | Transactional Apps | Medium | Very Simple | Good (Row-wise) |
| **PostgreSQL** | Production Systems | High | Complex | Excellent (All types) |

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
| NumPy arrays | NPZ + Blosc2 | LZ4/ZSTD | 60-80% size reduction, 4x faster I/O |
| DataFrames & Series | Parquet | LZ4 | 40-60% size reduction, columnar efficiency |
| Python objects | Pickle + Blosc | LZ4 | 30-50% size reduction, universal compatibility |

### Configuration

```python
from cacheness import cacheness, CacheConfig

# Simple configuration
config = CacheConfig(
    cache_dir="./my_cache",
    metadata_backend="sqlite",     # "sqlite", "json", or "auto"  
    default_ttl_hours=48,          # Default TTL for entries
    max_cache_size_mb=5000,        # Maximum cache size
)

cache = cacheness(config)
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

## Advanced Features

- **Custom Metadata**: Rich metadata tracking with SQLAlchemy ORM for experiment tracking and data lineage
- **Path Content Hashing**: Automatic content-based hashing for cache key, using key values that contain file & directory paths
- **Multi-format Storage**: Optimized formats for different data types (NPZ, Parquet, compressed pickle)
- **Intelligent Compression**: Automatic codec selection and parallel processing for large datasets

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
query = cache.query_custom_metadata("ml_experiments")
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

- **[Configuration Guide](docs/CONFIGURATION.md)** - Detailed configuration options and use cases
- **[SQL Cache Guide](docs/SQL_CACHE.md)** - Pull-through cache for APIs and time-series data
- **[Custom Metadata Guide](docs/CUSTOM_METADATA.md)** - Advanced metadata workflows with SQLAlchemy
- **[Performance Guide](docs/PERFORMANCE.md)** - Optimization strategies and benchmarks
- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation

## License

GPLv3 License - see LICENSE file for details.
