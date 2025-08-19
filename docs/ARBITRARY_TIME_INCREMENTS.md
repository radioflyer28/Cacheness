# SqlCache Arbitrary Time Increments

The SqlCache now supports user-specified time increments for precise gap detection. This eliminates the need for auto-detection when you know your data's time intervals.

## Key Benefits

- **Precise Gap Detection**: No more guessing - specify exactly how your data is spaced
- **Better Performance**: Skip auto-detection when you know the increment
- **Multiple Formats**: Flexible input formats for convenience
- **Backward Compatible**: Auto-detection still works when no increment is specified

## Usage Examples

### 1. High-Frequency Sensor Data (Every 5 Minutes)

```python
from datetime import timedelta
from cacheness.sql_cache import SqlCache

# Using timedelta objects (most precise)
cache = SqlCache.with_sqlite(
    db_path="sensor_data.db",
    table=sensor_table,
    data_adapter=sensor_adapter,
    time_increment=timedelta(minutes=5)  # Exact 5-minute intervals
)

# The cache now knows data comes every 5 minutes
# Gap detection will be precise to this interval
data = cache.get_data(
    sensor_id="TEMP_001",
    start_time=datetime(2025, 8, 19, 10, 0),
    end_time=datetime(2025, 8, 19, 12, 0)
)
```

### 2. String Format for Common Intervals

```python
# Using convenient string formats
cache = SqlCache.with_sqlite(
    db_path="logs.db", 
    table=log_table,
    data_adapter=log_adapter,
    time_increment="30sec"  # Every 30 seconds
)

# Also supports: "5min", "2hour", "1day", etc.
```

### 3. Numeric Seconds for Precise Control

```python
# Using numeric seconds (useful for unusual intervals)
cache = SqlCache.with_sqlite(
    db_path="metrics.db",
    table=metrics_table, 
    data_adapter=metrics_adapter,
    time_increment=300  # 300 seconds = 5 minutes
)
```

### 4. Ordered Data with Custom Increments

```python
# For ordered data like order IDs, transaction numbers
cache = SqlCache.with_sqlite(
    db_path="orders.db",
    table=order_table,
    data_adapter=order_adapter,
    ordered_increment=10  # Orders increment by 10
)

# Perfect for batch processing systems
```

### 5. Auto-Detection (Backward Compatible)

```python
# No increment specified - will auto-detect from data
cache = SqlCache.with_sqlite(
    db_path="data.db",
    table=data_table,
    data_adapter=data_adapter
    # Will analyze data patterns to determine increment
)
```

## Supported Time Formats

### Timedelta Objects
- `timedelta(seconds=30)`
- `timedelta(minutes=5)`
- `timedelta(hours=1)`
- `timedelta(days=1)`

### String Formats
- **Seconds**: `"30sec"`, `"45second"`, `"60seconds"`
- **Minutes**: `"5min"`, `"15minute"`, `"30minutes"`
- **Hours**: `"1hour"`, `"2hours"`
- **Days**: `"1day"`, `"7days"`
- **Weeks**: `"1week"`, `"2weeks"`

### Numeric Formats
- Integers or floats representing seconds
- `300` = 5 minutes
- `3600` = 1 hour
- `86400` = 1 day

## Real-World Examples

### IoT Sensor Network
```python
# Weather stations report every 10 minutes
weather_cache = SqlCache.with_sqlite(
    "weather.db", weather_table, weather_adapter,
    time_increment=timedelta(minutes=10)
)
```

### Financial Data
```python
# Stock prices every 15 seconds during trading
stock_cache = SqlCache.with_sqlite(
    "stocks.db", stock_table, stock_adapter, 
    time_increment="15sec"
)
```

### E-commerce Analytics
```python
# Order processing system with incremental order IDs
order_cache = SqlCache.with_sqlite(
    "orders.db", order_table, order_adapter,
    ordered_increment=1  # Sequential order numbers
)
```

### Log Processing
```python
# Application logs every minute
log_cache = SqlCache.with_sqlite(
    "logs.db", log_table, log_adapter,
    time_increment=60  # 60 seconds
)
```

## Performance Benefits

**Without User-Specified Increment:**
- Cache must analyze existing data to guess increment
- May guess incorrectly for sparse data
- Extra processing time for auto-detection

**With User-Specified Increment:**
- ✅ No auto-detection overhead
- ✅ Precise gap detection
- ✅ Optimal cache hit rates
- ✅ Predictable behavior

## Migration Guide

Existing code continues to work unchanged:

```python
# OLD: Auto-detection (still works)
cache = SqlCache.with_sqlite("data.db", table, adapter)

# NEW: With explicit increment (better performance)
cache = SqlCache.with_sqlite(
    "data.db", table, adapter,
    time_increment=timedelta(minutes=5)
)
```

The enhancement is fully backward compatible while providing significant performance improvements when you know your data's time structure.