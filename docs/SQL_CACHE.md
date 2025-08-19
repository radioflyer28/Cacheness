# SQL Pull-Through Cache

The SQL pull-through cache provides an intelligent caching layer that automatically fetches missing data from external sources and stores it in a local database for fast subsequent access.

## Overview

The SQL cache is perfect for scenarios where you need to:
- Cache time-series data from APIs (stock prices, sensor data, etc.)
- Implement smart partial data fetching
- Have database-level querying capabilities on cached data
- Ensure data consistency with automatic upsert operations

## Key Features

- **SQLAlchemy Integration**: Type-safe table definitions with full SQLAlchemy power
- **Multi-Database Support**: Works with SQLite, DuckDB, PostgreSQL, and more
- **Intelligent Gap Detection**: Automatically identifies missing data ranges
- **Upsert Operations**: Handles conflicts gracefully with database-specific optimizations
- **TTL Support**: Configurable expiration with automatic cleanup
- **Cache Statistics**: Built-in monitoring and management tools

## Quick Start

### 1. Define Your Table Schema

```python
from sqlalchemy import MetaData, Table, Column, String, Date, Float, Index

metadata = MetaData()

stock_table = Table(
    'stock_prices',
    metadata,
    Column('symbol', String(10), primary_key=True),
    Column('date', Date, primary_key=True),
    Column('close', Float),
    Column('volume', Float),
    
    # Add indexes for performance
    Index('idx_symbol_date', 'symbol', 'date')
)
```

### 2. Create a Data Adapter

```python
from cacheness.sql_cache import SqlCacheAdapter
import pandas as pd

class StockSqlCacheAdapter(SqlCacheAdapter):
    def get_table_definition(self):
        return stock_table
    
    def parse_query_params(self, **kwargs):
        return {
            'symbol': kwargs['symbol'],
            'date': {
                'start': kwargs['start_date'],
                'end': kwargs['end_date']
            }
        }
    
    def fetch_data(self, **kwargs):
        # Your API call logic here
        symbol = kwargs['symbol']
        start_date = kwargs['start_date']
        end_date = kwargs['end_date']
        
        # Fetch from external API
        data = fetch_stock_data_from_api(symbol, start_date, end_date)
        return pd.DataFrame(data)
```

### 3. Create and Use the Cache

```python
from cacheness.sql_cache import SqlCache

# Create cache instance
cache = SqlCache(
    db_url="stocks.db",  # DuckDB file
    table=stock_table,
    data_adapter=StockSqlCacheAdapter(),
    ttl_hours=24,
    time_increment=timedelta(minutes=5),  # Optional: specify data increment
    gap_detector=None  # Optional: custom gap detection function
)

# Get data (automatically fetches missing data)
data = cache.get_data(
    symbol="AAPL",
    start_date="2024-01-01",
    end_date="2024-01-31"
)

print(f"Retrieved {len(data)} records")
```

## Advanced Configuration

### Custom Gap Detection

For precise control over when data should be fetched, implement custom gap detection:

```python
def intelligent_gap_detector(query_params, cached_data, cache_instance):
    """Custom gap detection with access to increment settings."""
    
    # Access user-specified increments
    time_increment = cache_instance.time_increment
    ordered_increment = cache_instance.ordered_increment
    
    if cached_data.empty:
        return [cache_instance._convert_query_to_fetch_params(query_params)]
    
    # Custom logic based on your data patterns
    # Return [] for no fetch, or list of fetch parameter dicts
    return []

# Use with cache
cache = SqlCache.with_sqlite(
    db_path="cache.db",
    table=table,
    data_adapter=adapter,
    gap_detector=intelligent_gap_detector
)
```

### Increment Specification

Specify known data increments for optimal gap detection:

```python
# Time-based increments
cache = SqlCache.with_sqlite(
    db_path="sensor_data.db",
    table=sensor_table,
    data_adapter=adapter,
    time_increment=timedelta(minutes=5)  # Data every 5 minutes
)

# String format increments
cache = SqlCache.with_sqlite(
    db_path="logs.db",
    table=log_table,
    data_adapter=adapter,
    time_increment="30sec"  # Every 30 seconds
)

# Ordered data increments
cache = SqlCache.with_sqlite(
    db_path="orders.db",
    table=order_table,
    data_adapter=adapter,
    ordered_increment=10  # Order IDs increment by 10
)
```

### Custom Missing Data Detection

For time-series data, implement intelligent gap detection:

```python
class TimeSeriesCache(SqlCache):
    def _find_missing_data(self, query_params, cached_data):
        """Find missing date ranges"""
        if cached_data.empty:
            return [query_params]  # Fetch everything
        
        # Find gaps in time series
        expected_dates = pd.date_range(
            query_params['start_date'],
            query_params['end_date'],
            freq='D'
        )
        
        cached_dates = set(cached_data['date'])
        missing_dates = set(expected_dates) - cached_dates
        
        # Group consecutive dates into ranges
        return self._group_into_ranges(missing_dates, query_params)
```

### Database-Specific Optimizations

The cache automatically uses database-specific upsert operations:

- **SQLite/DuckDB**: `INSERT ... ON CONFLICT DO UPDATE`
- **PostgreSQL**: `INSERT ... ON CONFLICT DO UPDATE`
- **Other databases**: Fallback to individual operations

### Complex Query Parameters

Support rich query patterns:

```python
def parse_query_params(self, **kwargs):
    return {
        'symbol': kwargs['symbol'],
        'date': {
            'start': kwargs['start_date'],
            'end': kwargs['end_date']
        },
        'price': {
            'gte': kwargs.get('min_price'),  # Greater than or equal
            'lte': kwargs.get('max_price')   # Less than or equal
        }
    }
```

## Cache Management

### Statistics and Monitoring

```python
# Get cache statistics
stats = cache.get_cache_stats()
print(f"Total records: {stats['total_records']}")
print(f"Expired records: {stats['expired_records']}")

# Clean up expired entries
expired_count = cache.cleanup_expired()
print(f"Cleaned up {expired_count} expired entries")
```

### Cache Invalidation

```python
# Invalidate specific data
cache.invalidate_cache(symbol="AAPL")

# Clear entire cache
cache.clear_cache()
```

## Configuration Options

### Database URLs

The cache supports various database backends with different optimization profiles:

```python
# DuckDB - Optimized for analytical/columnar workloads
cache = SqlCache.with_duckdb("analytics.db", table, adapter)

# SQLite - Optimized for transactional/row-wise operations  
cache = SqlCache.with_sqlite("cache.db", table, adapter)

# PostgreSQL - Production-ready with high concurrency
cache = SqlCache.with_postgresql(
    "postgresql://user:pass@localhost/db", table, adapter
)

# Manual URL specification
cache = SqlCache("duckdb:///data.db", table, adapter)
cache = SqlCache("sqlite:///cache.db", table, adapter)

# In-memory (for testing)
cache = SqlCache.with_sqlite(":memory:", table, adapter)
```

## Database Backend Selection

The SQL pull-through cache supports multiple database backends, each optimized for different workload patterns. Choose the backend that best matches your use case:

### DuckDB - Analytical Workloads

```python
cache = SqlCache.with_duckdb("analytics.db", table, adapter)
```

**Strengths:**
- **Columnar storage** optimized for analytical queries
- **Fast aggregations** across large datasets
- **Vectorized execution** for time-series operations
- **Memory-efficient** for analytical workloads

**Limitations:**
- **No auto-incrementing primary keys** (no SERIAL type support)
- **Use composite primary keys** or set `autoincrement=False`
- **Optimized for read-heavy workloads** (analytical focus)

**Best Use Cases:**
- Time-series data analysis and reporting
- Large dataset aggregations and analytics
- Data science workflows with pandas/numpy
- OLAP-style queries and data exploration

**Example - Financial Analytics (DuckDB-Compatible Table):**
```python
# Define table with composite primary key (DuckDB compatible)
stock_table = Table(
    'stock_prices', metadata,
    Column('symbol', String(10), primary_key=True),  # No autoincrement
    Column('date', Date, primary_key=True),           # Composite key
    Column('close', Float),
    Column('volume', Float)
)

# Perfect for analyzing stock market data
cache = SqlCache.with_duckdb(
    "market_analytics.db", 
    stock_table, 
    YahooFinanceAdapter(),
    ttl_hours=24
)

# Efficient queries like: SELECT AVG(close) FROM stocks WHERE date > '2024-01-01'
quarterly_data = cache.get_data(symbol="AAPL", start_date="2024-01-01", end_date="2024-03-31")
```

**⚠️ DuckDB Table Design Guidelines:**
```python
# ❌ Avoid: Auto-incrementing primary keys
bad_table = Table('data', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),  # Will fail!
    Column('value', Float)
)

# ✅ Good: Composite primary keys or explicit autoincrement=False
good_table = Table('data', metadata,
    Column('symbol', String(10), primary_key=True),
    Column('date', Date, primary_key=True),  # Composite key works
    Column('value', Float)
)

# ✅ Alternative: Explicit autoincrement=False
alternative_table = Table('data', metadata,
    Column('id', Integer, primary_key=True, autoincrement=False),  # Explicit
    Column('value', Float)
)
```

### SQLite - Transactional Workloads

```python
cache = SqlCache.with_sqlite("cache.db", table, adapter)
```

**Strengths:**
- **ACID compliance** with full transaction support
- **Row-wise optimizations** for transactional operations  
- **Simple deployment** with zero configuration
- **Excellent concurrent read** performance

**Best Use Cases:**
- Transactional applications requiring data consistency
- Row-by-row data processing workflows
- Applications with moderate concurrent access
- Simple deployment scenarios

**Example - User Session Cache:**
```python
# Perfect for user session data that needs ACID guarantees
cache = SqlCache.with_sqlite(
    "user_sessions.db",
    session_table,
    SessionAPIAdapter(),
    ttl_hours=6
)

# Reliable for operations like: INSERT OR REPLACE INTO sessions...
user_session = cache.get_data(user_id=12345, session_date="2024-01-15")
```

### PostgreSQL - Production Environments

```python
cache = SqlCache.with_postgresql(
    "postgresql://user:pass@host:5432/dbname", 
    table, 
    adapter
)
```

**Strengths:**
- **High concurrency** with advanced locking mechanisms
- **Advanced SQL features** (CTEs, window functions, etc.)
- **Horizontal scaling** capabilities
- **Enterprise-grade reliability** and monitoring

**Best Use Cases:**
- Production systems with high concurrent access
- Complex queries requiring advanced SQL features
- Multi-user applications with heavy read/write loads
- Enterprise environments requiring high availability

**Example - Multi-Tenant API Cache:**
```python
# Perfect for production API caching with multiple concurrent users
cache = SqlCache.with_postgresql(
    "postgresql://cache_user:password@prod-db:5432/api_cache",
    api_data_table,
    ThirdPartyAPIAdapter(),
    ttl_hours=12
)

# Handles concurrent access from multiple application instances
api_response = cache.get_data(endpoint="weather", location="NYC", timestamp="2024-01-15T10:00:00")
```

### In-Memory SQLite - Testing & Development

```python
cache = SqlCache.with_sqlite(":memory:", table, adapter)
```

**Perfect for:**
- Unit testing and development
- Temporary data processing pipelines
- Proof-of-concept implementations
- CI/CD environments

### Backend Comparison Matrix

| Feature | DuckDB | SQLite | PostgreSQL | In-Memory SQLite |
|---------|--------|--------|------------|------------------|
| **Query Performance** | ⭐⭐⭐⭐⭐ (Analytics) | ⭐⭐⭐ (Row-wise) | ⭐⭐⭐⭐ (Mixed) | ⭐⭐⭐⭐⭐ (Memory) |
| **Concurrency** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Deployment Complexity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **ACID Compliance** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Analytics Workloads** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Transactional Workloads** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Auto-increment Support** | ❌ (Use composite keys) | ✅ | ✅ | ✅ |
| **Production Readiness** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |

### Backend Selection Decision Tree

```
Do you need production-grade concurrency (100+ concurrent users)?
├─ YES → Use PostgreSQL
└─ NO
   │
   └─ Is your primary use case analytical (aggregations, time-series)?
      ├─ YES → Use DuckDB  
      └─ NO
         │
         └─ Do you need simple deployment with ACID guarantees?
            ├─ YES → Use SQLite
            └─ NO → Use In-Memory SQLite (testing/dev)
```

### Backend-Specific Optimizations

Each backend includes database-specific optimizations:

#### DuckDB Optimizations
- Vectorized upsert operations for bulk data
- Columnar storage for time-series data
- Automatic query parallelization

#### SQLite Optimizations  
- `INSERT OR REPLACE` for conflict resolution
- WAL mode for concurrent reads
- Optimized indexes for cache lookups

#### PostgreSQL Optimizations
- `ON CONFLICT` upsert syntax
- Connection pooling support
- Advanced index types (GIN, GiST)

### Backend Performance Guide

| Backend | `list_entries()` | `get_stats()` | `cleanup_expired()` |
|---------|------------------|---------------|-------------------|
| **DuckDB** | **1.8ms** | **3.2ms** | **8ms** |
| **SQLite** | **2.3ms** | **4.1ms** | **12ms** |
| **PostgreSQL** | **3.1ms** | **5.2ms** | **15ms** |
| **In-Memory** | **0.5ms** | **0.8ms** | **2ms** |

*Performance measured with 10k+ entries on modern hardware*

### TTL Configuration

```python
# No expiration
cache = SqlCache(..., ttl_hours=0)

# 1 hour expiration
cache = SqlCache(..., ttl_hours=1)

# Daily refresh
cache = SqlCache(..., ttl_hours=24)
```

## Dependencies

The SQL cache requires additional dependencies:

```bash
# Install with SQL support
pip install 'cacheness[sql]'

# Or install manually
pip install sqlalchemy pandas

# For DuckDB support (recommended)
pip install duckdb-engine
```

## Complete Example

See `examples/stock_cache_example.py` for a complete working example with:
- Yahoo Finance integration
- Intelligent missing data detection
- Cache statistics and management
- Error handling and logging

## Performance Tips

1. **Use Indexes**: Add appropriate indexes to your table definition
2. **Batch Operations**: The cache automatically batches upsert operations
3. **Choose the Right Database**: DuckDB for analytics, SQLite for simplicity
4. **Optimize TTL**: Balance freshness vs. API call costs
5. **Monitor Cache Stats**: Use built-in statistics to optimize performance

## Error Handling

The cache includes comprehensive error handling:

```python
from cacheness.sql_cache import SQLCacheError, MissingDependencyError

try:
    data = cache.get_data(symbol="AAPL", start_date="2024-01-01")
except SQLCacheError as e:
    print(f"Cache error: {e}")
except MissingDependencyError as e:
    print(f"Missing dependency: {e}")
```
