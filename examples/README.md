# Cacheness Examples

This directory contains examples organized by complexity level to help you learn cacheness step by step.

## 🚀 Getting Started (New Users Start Here!)

### Basic Function Caching
- **`simple_function_caching.py`** - Basic function caching with `@cached` decorator
- **`simple_api_caching.py`** - Cache API responses with minimal setup
- **`simple_object_caching.py`** - Cache complex objects and data structures

### SQL Pull-Through Caching  
- **`beginner_sql_cache.py`** - Basic SQL caching with new builder pattern
- **`simple_stock_cache.py`** - Real-world stock data caching example
- **`simple_backend_demo.py`** - When to use SQLite vs DuckDB

### Configuration & Features
- **`simple_config_demo.py`** - Basic configuration options and TTL strategies
- **`intelligent_storage_demo.py`** - How UnifiedCache auto-optimizes storage formats

## 🎯 Intermediate Examples

### Specialized Use Cases
- **`simple_ml_pipeline.py`** - Machine learning workflow caching
- **`custom_metadata_demo.py`** - Custom metadata for cache organization

### Advanced API Usage
- **`api_request_caching.py`** - Complete API caching with TTL strategies
- **`s3_caching.py`** - S3 file caching with ETag validation

## 🏗️ Advanced Examples (Complex Use Cases)

### Production-Ready Patterns
- **`stock_cache_example.py`** - Full-featured stock cache with gap detection
- **`ml_pipeline_caching.py`** - Multi-stage ML pipeline with custom config
- **`database_backend_comparison.py`** - Detailed backend performance comparison

### Advanced Configuration
- **`configurable_serialization_demo.py`** - Advanced serialization options
- **`dill_class_caching_demo.py`** - Complex object caching with security

## 📖 Learning Path

### 👶 New to Caching?
1. Start with `simple_function_caching.py`
2. Try `simple_api_caching.py` 
3. See `intelligent_storage_demo.py` to understand auto-optimization
4. Move to `simple_config_demo.py`

### 🗄️ Need SQL Caching?
1. Begin with `beginner_sql_cache.py`
2. See real example in `simple_stock_cache.py`
3. Learn backend selection in `simple_backend_demo.py`

### 🤖 Machine Learning?
1. Start with `simple_ml_pipeline.py`
2. For complex workflows, see `ml_pipeline_caching.py`

### 🔧 Custom Requirements?
1. Check `custom_metadata_demo.py` for organization
2. See advanced examples for production patterns

## 🆕 New API Features

The examples showcase the **intelligent caching** system:

### UnifiedCache - Intelligent Function Caching:
```python
# Automatically optimizes storage based on data type
@cached()                # Any Python object (auto-detects best format)
@cached.for_api()        # API calls (6h TTL, fast compression, error handling)

# Storage optimization examples:
# • DataFrames → Parquet format
# • NumPy arrays → Blosc compression  
# • Objects → Pickle/Dill serialization
# • JSON responses → LZ4 compression
```

### SqlCache - Pull-Through Database Caching:
```python
# Access-pattern-optimized SQL caching
SqlCache.for_lookup_table()      # Individual records (SQLite)
SqlCache.for_analytics_table()   # Bulk analysis (DuckDB)
SqlCache.for_realtime_timeseries()  # Real-time data (SQLite)
SqlCache.for_timeseries()        # Historical analysis (DuckDB)
```

### 🎯 Key Architecture Principles:

#### **UnifiedCache** (Function Caching):
- **Intelligent storage**: Automatically detects optimal format for your data
- **Universal compatibility**: Works with any Python object
- **Performance optimized**: DataFrames use Parquet, arrays use Blosc, etc.
- **Simple API**: Just use `@cached()` and it handles the rest

#### **SqlCache** (Pull-Through Caching):
- **Database-aware**: Chooses SQLite vs DuckDB based on access patterns
- **Gap detection**: Automatically fetches missing data ranges
- **Schema-driven**: Uses SQLAlchemy table definitions
- **Pull-through pattern**: Cache sits between your app and data source

## 💡 Tips

- **Start simple**: Use the basic examples to understand concepts
- **Access patterns matter**: Choose the right method for your use case
- **TTL strategy**: Short TTL for changing data, long TTL for stable data
- **Backend selection**: SQLite for lookups, DuckDB for analytics
- **No inheritance needed**: New builders use simple functions instead of complex adapters

Happy caching! 🚀