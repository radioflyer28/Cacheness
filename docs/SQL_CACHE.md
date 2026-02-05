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

# SQL Pull-Through Cache

The SQL pull-through cache provides an intelligent caching layer that automatically fetches missing data from external sources and stores it in a local database for fast subsequent access.

## Overview

The SQL cache is perfect for scenarios where you need to:
- Cache time-series data from APIs (stock prices, sensor data, etc.)
- Implement smart partial data fetching
- Have database-level querying capabilities on cached data
- Ensure data consistency with automatic upsert operations

## Key Features

- **Intelligent Backend Selection**: Automatically chooses SQLite vs DuckDB based on access patterns
- **Simple Builder Pattern**: No inheritance required - just provide a function
- **Multi-Database Support**: Works with SQLite, DuckDB, PostgreSQL, and more
- **Intelligent Gap Detection**: Automatically identifies missing data ranges
- **Upsert Operations**: Handles conflicts gracefully with database-specific optimizations
- **TTL Support**: Configurable expiration with automatic cleanup
- **Cache Statistics**: Built-in monitoring and management tools

## Quick Start

### Simple Builder Pattern (Recommended)

#### For Individual Record Lookups
```python
from cacheness.sql_cache import SqlCache
from sqlalchemy import Integer, String, Float

def fetch_user_data(user_id):
    """Fetch user data from API - just return a DataFrame."""
    api_data = requests.get(f"/api/users/{user_id}").json()
    return pd.DataFrame([api_data])

# Creates SQLite cache optimized for row-wise lookups
user_cache = SqlCache.for_lookup_table(
    "users.db",
    primary_keys=["user_id"],
    data_fetcher=fetch_user_data,
    ttl_seconds=43200,  # 12 hours
    user_id=Integer,
    name=String(100),
    email=String(255)
)

# Use it
user_data = user_cache.get_data(user_id=123)
```

#### For Time-Series Data
```python
def fetch_stock_prices(symbol, start_date, end_date):
    """Fetch stock data from API."""
    return yfinance_client.get_data(symbol, start_date, end_date)

# Creates DuckDB cache optimized for analytical queries  
stock_cache = SqlCache.for_timeseries(
    "stocks.db",
    data_fetcher=fetch_stock_prices,
    ttl_seconds=86400,  # 24 hours
    price=Float,
    volume=Integer,
    market_cap=Float
)

# Use it
stock_data = stock_cache.get_data(
    symbol="AAPL", 
    start_date="2024-01-01", 
    end_date="2024-01-31"
)
```

#### For Analytics Workloads
```python
def fetch_sales_data(department, quarter):
    """Fetch sales analytics from data warehouse."""
    return warehouse_client.get_sales(department, quarter)

# Creates DuckDB cache optimized for bulk analytical queries
sales_cache = SqlCache.for_analytics_table(
    "sales.db",
    primary_keys=["department", "quarter"],
    data_fetcher=fetch_sales_data,
    ttl_seconds=172800,  # 48 hours
    department=String(50),
    quarter=Integer,
    revenue=Float,
    profit_margin=Float
)

# Use it
q1_sales = sales_cache.get_data(department="Engineering", quarter=1)
```

### Builder Method Reference

| **Method** | **Database** | **Best For** | **TTL Default** |
|------------|--------------|--------------|-----------------|
| `for_lookup_table()` | SQLite | Individual record access | 12 hours |
| `for_analytics_table()` | DuckDB | Bulk queries, reports | 12 hours |
| `for_timeseries()` | DuckDB | Historical time-series | 24 hours |
| `for_realtime_timeseries()` | SQLite | Real-time data updates | 1 hour |

## Advanced Configuration

### Custom Table Schemas (Advanced)

For more control over table structure, you can define schemas manually using SQLAlchemy:

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

def fetch_stock_data(symbol, start_date, end_date):
    """Your data fetching logic here."""
    return fetch_from_api(symbol, start_date, end_date)

# Use the custom table with builder pattern
cache = SqlCache.for_timeseries(
    "custom_stocks.db",
    data_fetcher=fetch_stock_data,
    table=stock_table  # Use custom table
)
```

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
cache = SqlCache.for_timeseries(
    "sensor_data.db",
    data_fetcher=fetch_sensor_data,
    gap_detector=intelligent_gap_detector,
    time_increment=timedelta(minutes=5)  # Data every 5 minutes
)
```

### Increment Specification

Specify known data increments for optimal gap detection:

```python
from datetime import timedelta

# Time-based increments
cache = SqlCache.for_timeseries(
    "sensor_data.db",
    data_fetcher=fetch_sensor_data,
    time_increment=timedelta(minutes=5)  # Data every 5 minutes
)

# String format increments  
cache = SqlCache.for_realtime_timeseries(
    "logs.db",
    data_fetcher=fetch_log_data,
    time_increment="30sec"  # Every 30 seconds
)

# Ordered data increments
cache = SqlCache.for_lookup_table(
    "orders.db", 
    primary_keys=["order_id"],
    data_fetcher=fetch_order_data,
    ordered_increment=10,  # Order IDs increment by 10
    order_id=Integer,
    customer_id=Integer,
    amount=Float
)
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

### Database Backend Selection

The builder methods automatically select the optimal database backend:

```python
# Automatically uses SQLite (optimized for individual lookups)
user_cache = SqlCache.for_lookup_table(
    "users.db",
    primary_keys=["user_id"],
    data_fetcher=fetch_user_data,
    user_id=Integer,
    name=String(100)
)

# Automatically uses DuckDB (optimized for analytics)
sales_cache = SqlCache.for_analytics_table(
    "sales.db",
    primary_keys=["department", "quarter"],
    data_fetcher=fetch_sales_data,
    department=String(50),
    revenue=Float
)

# Automatically uses DuckDB (optimized for time-series)
stock_cache = SqlCache.for_timeseries(
    "stocks.db",
    data_fetcher=fetch_stock_data,
    price=Float,
    volume=Integer
)

# Automatically uses SQLite (optimized for real-time updates)
sensor_cache = SqlCache.for_realtime_timeseries(
    "sensors.db",
    data_fetcher=fetch_sensor_data,
    temperature=Float,
    humidity=Float
)
```

### Manual Database Specification

For advanced use cases, you can specify the database type explicitly:

```python
# Force DuckDB for a lookup table (unusual but possible)
cache = SqlCache.for_lookup_table(
    "duckdb:///users.db",  # Explicit DuckDB URL
    primary_keys=["user_id"],
    data_fetcher=fetch_user_data,
    user_id=Integer,
    name=String(100)
)

# PostgreSQL for production deployment
cache = SqlCache.for_timeseries(
    "postgresql://user:pass@localhost/production_db",
    data_fetcher=fetch_stock_data,
    price=Float,
    volume=Integer
)

# In-memory for testing
cache = SqlCache.for_lookup_table(
    "sqlite:///:memory:",
    primary_keys=["id"],
    data_fetcher=fetch_test_data,
    id=Integer,
    value=String(50)
)
```

## Database Backend Details

### DuckDB - Analytical Workloads

**Automatically Selected For:**
- `SqlCache.for_analytics_table()`
- `SqlCache.for_timeseries()`

**Strengths:**
- **Columnar storage** optimized for analytical queries
- **Fast aggregations** across large datasets  
- **Vectorized execution** for time-series operations
- **Memory-efficient** for analytical workloads

**Best Use Cases:**
- Time-series data analysis and reporting
- Large dataset aggregations and analytics
- Data science workflows with pandas/numpy
- OLAP-style queries and data exploration

**Example:**
```python
def fetch_stock_data(symbol, start_date, end_date):
    return yfinance_client.get_data(symbol, start_date, end_date)

# Uses DuckDB automatically
stock_cache = SqlCache.for_timeseries(
    "market_analytics.db",
    data_fetcher=fetch_stock_data,
    ttl_seconds=86400,  # 24 hours
    symbol=String(10),
    date=Date,
    close=Float,
    volume=Float
)

# Efficient for analytical queries
quarterly_data = stock_cache.get_data(
    symbol="AAPL", 
    start_date="2024-01-01", 
    end_date="2024-03-31"
)
```

### SQLite - Transactional Workloads

**Automatically Selected For:**
- `SqlCache.for_lookup_table()`
- `SqlCache.for_realtime_timeseries()`

**Strengths:**
- **Row-wise storage** optimized for OLTP operations
- **Fast individual record access** and updates
- **ACID compliance** with strong consistency
- **Auto-incrementing primary keys** support

**Best Use Cases:**
- Individual record lookups and updates
- Real-time data ingestion with frequent writes
- Configuration and metadata caching
- Small to medium-sized datasets

**Example:**
```python
def fetch_user_data(user_id):
    return api_client.get_user(user_id)

# Uses SQLite automatically  
user_cache = SqlCache.for_lookup_table(
    "users.db",
    primary_keys=["user_id"],
    data_fetcher=fetch_user_data,
    ttl_seconds=43200,  # 12 hours
    user_id=Integer,
    name=String(100),
    email=String(255),
    created_at=DateTime
)

# Efficient for individual lookups
user = user_cache.get_data(user_id=123)
```

## Performance Guidelines

### When to Use Each Builder Method

| **Data Pattern** | **Builder Method** | **Database** | **Why** |
|------------------|-------------------|--------------|---------|
| User profiles, config data | `for_lookup_table()` | SQLite | Fast individual record access |
| Real-time metrics, IoT data | `for_realtime_timeseries()` | SQLite | Optimized for frequent writes |
| Historical analysis, reporting | `for_timeseries()` | DuckDB | Columnar analytics performance |
| Aggregations, data science | `for_analytics_table()` | DuckDB | Vectorized analytical queries |

### Performance Tips

1. **Use appropriate primary keys** for your access patterns
2. **Set reasonable TTL values** to balance freshness and performance  
3. **Consider data volume** when choosing between SQLite and DuckDB
4. **Use composite primary keys** for multi-dimensional data
5. **Implement efficient data fetchers** to minimize external API calls

## Legacy API Reference

For backward compatibility, the original adapter-based API is still supported:

```python
from cacheness.sql_cache import SqlCacheAdapter, SqlCache
from sqlalchemy import Table, Column, Integer, String, Float

# Legacy adapter pattern (still supported)
class MyAdapter(SqlCacheAdapter):
    def get_table_definition(self):
        return Table('data', metadata, 
                    Column('id', Integer, primary_key=True),
                    Column('value', Float))
    
    def fetch_data(self, **kwargs):
        return fetch_from_external_source(**kwargs)

# Legacy cache creation  
cache = SqlCache("sqlite:///cache.db", adapter=MyAdapter())
```

**Note:** The new builder pattern is recommended for all new code as it's simpler and more maintainable.
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

## Database Optimizations

The cache automatically applies database-specific optimizations:

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

### Performance Comparison

| Backend | Individual Lookups | Bulk Analytics | Real-time Updates |
|---------|-------------------|----------------|-------------------|
| **SQLite** | **Excellent** | Good | **Excellent** |
| **DuckDB** | Good | **Excellent** | Good |
| **PostgreSQL** | **Excellent** | **Excellent** | **Excellent** |
| **In-Memory** | **Excellent** | **Excellent** | **Excellent** |

*Performance depends on data volume and query patterns*

### TTL Configuration

All builder methods support TTL configuration:

```python
# Short TTL for real-time data
cache = SqlCache.for_realtime_timeseries(
    "sensors.db", 
    data_fetcher=fetch_sensor_data,
    ttl_seconds=3600,  # Refresh every hour
    temperature=Float
)

# Long TTL for historical data
cache = SqlCache.for_analytics_table(
    "reports.db",
    primary_keys=["department", "quarter"], 
    data_fetcher=fetch_quarterly_data,
    ttl_seconds=604800,  # Refresh weekly
    department=String(50),
    revenue=Float
)

# No expiration for static data
cache = SqlCache.for_lookup_table(
    "config.db",
    primary_keys=["setting_name"],
    data_fetcher=fetch_config_data,
    ttl_seconds=0,  # Never expire
    setting_name=String(100),
    setting_value=String(500)
)
```

## Dependencies

The SQL cache requires additional dependencies:

```bash
# Install with SQL support
pip install 'cacheness[sql]'

# Or install manually
pip install sqlalchemy pandas

# For DuckDB support (recommended for analytics)
pip install duckdb-engine
```

## Complete Example

Here's a complete working example using the new builder pattern:

```python
import pandas as pd
from cacheness.sql_cache import SqlCache
from sqlalchemy import String, Float, Date
import yfinance as yf

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    # Reset index to make date a column
    data = data.reset_index()
    data['symbol'] = symbol
    return data[['symbol', 'Date', 'Close', 'Volume']]

# Create optimized time-series cache
stock_cache = SqlCache.for_timeseries(
    "stocks.db",
    data_fetcher=fetch_stock_data,
    ttl_seconds=86400,  # 24 hours
    symbol=String(10),
    Date=Date,
    Close=Float,
    Volume=Float
)

# Use the cache
apple_data = stock_cache.get_data(
    symbol="AAPL",
    start_date="2024-01-01", 
    end_date="2024-01-31"
)

print(f"Retrieved {len(apple_data)} records for AAPL")

# Get cache statistics
stats = stock_cache.get_cache_stats()
print(f"Total records in cache: {stats['total_records']}")
```

## Performance Tips

1. **Choose the Right Builder Method**: Match your access pattern to the builder method
2. **Set Appropriate TTL**: Balance data freshness vs. API call costs  
3. **Use Composite Primary Keys**: For multi-dimensional data access
4. **Implement Efficient Fetchers**: Minimize external API calls
5. **Monitor Cache Stats**: Use built-in statistics to optimize performance

## Error Handling

The cache includes comprehensive error handling:

```python
from cacheness.sql_cache import SQLCacheError, MissingDependencyError

try:
    data = stock_cache.get_data(symbol="AAPL", start_date="2024-01-01", end_date="2024-01-31")
except SQLCacheError as e:
    print(f"Cache error: {e}")
except MissingDependencyError as e:
    print(f"Missing dependency: {e}")
```
