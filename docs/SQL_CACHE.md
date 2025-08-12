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
from cacheness.sql_cache import SQLAlchemyDataAdapter
import pandas as pd

class StockDataAdapter(SQLAlchemyDataAdapter):
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
from cacheness.sql_cache import SQLAlchemyPullThroughCache

# Create cache instance
cache = SQLAlchemyPullThroughCache(
    db_url="stocks.db",  # DuckDB file
    table=stock_table,
    data_adapter=StockDataAdapter(),
    ttl_hours=24
)

# Get data (automatically fetches missing data)
data = cache.get_data(
    symbol="AAPL",
    start_date="2024-01-01",
    end_date="2024-01-31"
)

print(f"Retrieved {len(data)} records")
```

## Advanced Usage

### Custom Missing Data Detection

For time-series data, you'll want to implement intelligent gap detection:

```python
class TimeSeriesCache(SQLAlchemyPullThroughCache):
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

The cache supports various database backends:

```python
# DuckDB (recommended for analytics)
cache = SQLAlchemyPullThroughCache("data.db", ...)

# SQLite
cache = SQLAlchemyPullThroughCache("sqlite:///data.db", ...)

# PostgreSQL
cache = SQLAlchemyPullThroughCache(
    "postgresql://user:pass@localhost/db", ...
)

# In-memory (for testing)
cache = SQLAlchemyPullThroughCache("sqlite:///:memory:", ...)
```

### TTL Configuration

```python
# No expiration
cache = SQLAlchemyPullThroughCache(..., ttl_hours=0)

# 1 hour expiration
cache = SQLAlchemyPullThroughCache(..., ttl_hours=1)

# Daily refresh
cache = SQLAlchemyPullThroughCache(..., ttl_hours=24)
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
