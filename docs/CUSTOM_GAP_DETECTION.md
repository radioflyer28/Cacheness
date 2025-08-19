# Custom Gap Detection in SqlCache

The SqlCache system provides multiple approaches for customizing gap detection logic to fit your specific use case. This flexibility allows you to optimize caching behavior for different data patterns and requirements.

## Overview

Gap detection determines when cached data is insufficient and new data needs to be fetched from the source. The system supports:

1. **Function-based custom detectors** (Recommended)
2. **Subclass-based custom logic** (Traditional inheritance)
3. **Built-in intelligent detection** (Automatic)

## Approach 1: Function-Based Custom Detectors (Recommended)

The most flexible approach uses a custom function passed to the `gap_detector` parameter:

```python
def conservative_gap_detector(query_params, cached_data, cache_instance):
    """Only fetch if cached data doesn't cover the full requested range."""
    if cached_data.empty:
        return [cache_instance._convert_query_to_fetch_params(query_params)]
    
    # Custom logic here - return list of fetch parameters or empty list
    for param_name, param_value in query_params.items():
        if param_name == 'trade_id' and isinstance(param_value, dict):
            requested_start = param_value['start']
            requested_end = param_value['end']
            
            if 'trade_id' in cached_data.columns:
                cached_min = cached_data['trade_id'].min()
                cached_max = cached_data['trade_id'].max()
                
                if requested_start < cached_min or requested_end > cached_max:
                    print(f"Gap detected: requested {requested_start}-{requested_end}, cached {cached_min}-{cached_max}")
                    return [cache_instance._convert_query_to_fetch_params(query_params)]
                else:
                    print("No gaps detected")
                    return []
    
    return []

# Use with SqlCache
cache = SqlCache.with_sqlite(
    db_path="cache.db",
    table=table,
    data_adapter=adapter,
    gap_detector=conservative_gap_detector
)
```

### Function Signature

Your custom gap detector function should have this signature:

```python
def custom_gap_detector(query_params: Dict[str, Any], cached_data: pd.DataFrame, cache_instance: 'SqlCache') -> List[Dict[str, Any]]:
    """
    Args:
        query_params: The original query parameters
        cached_data: Currently cached data that matches the query
        cache_instance: The SqlCache instance (provides access to increment settings and built-in methods)
    
    Returns:
        List of fetch parameter dictionaries. Empty list means no fetch needed.
    """
```

## Approach 2: Subclass-Based Custom Logic

For complex customization, you can subclass SqlCache and override `_find_missing_data`:

```python
class CustomGapSqlCache(SqlCache):
    def _find_missing_data(self, query_params, cached_data):
        """Custom gap detection via subclassing."""
        if cached_data.empty:
            return [self._convert_query_to_fetch_params(query_params)]
        
        # Your custom logic here
        # Return list of fetch parameters or empty list
        return []

# Use the custom subclass
cache = CustomGapSqlCache.with_sqlite(adapter=adapter)
```

## Approach 3: Built-in Intelligence

If no custom detector is provided, the system uses intelligent built-in gap detection:

```python
# Uses built-in intelligence automatically
cache = SqlCache.with_sqlite(adapter=adapter)
```

The built-in system:
- Detects time-series patterns and increments
- Handles ordered data (like trade IDs)
- Supports arbitrary time increments
- Provides sensible defaults for most use cases

## Example Custom Detectors

### Always Fetch (Testing/Development)
```python
def always_fetch_detector(query_params, cached_data, cache_instance):
    """Always fetch - useful for testing or ensuring fresh data."""
    return [cache_instance._convert_query_to_fetch_params(query_params)]
```

### Custom Increment Access
```python
def increment_aware_detector(query_params, cached_data, cache_instance):
    """Custom detector that can access and override increment settings."""
    
    # Access current increment settings
    current_time_increment = cache_instance.time_increment
    current_ordered_increment = cache_instance.ordered_increment
    
    # Override with custom increment logic
    if 'high_frequency' in str(query_params):
        custom_increment = timedelta(seconds=1)  # 1-second precision for HFT
        print(f"Using custom increment: {custom_increment}")
        # Implement gap detection with custom increment
    elif current_time_increment:
        print(f"Using user-specified increment: {current_time_increment}")
        # Use existing setting
    else:
        # Use built-in detection methods
        if not cached_data.empty and 'timestamp' in cached_data.columns:
            detected_increment = cache_instance._detect_granularity(cached_data, 'timestamp')
            print(f"Auto-detected increment: {detected_increment}")
    
    # Custom gap detection logic here
    return []  # Return appropriate fetch parameters
```

### Sparse Data Detector
```python
def sparse_detector(query_params, cached_data, cache_instance):
    """Detect missing individual records in sparse datasets."""
    if cached_data.empty:
        return [cache_instance._convert_query_to_fetch_params(query_params)]
    
    # Check for missing individual records
    for param_name, param_value in query_params.items():
        if param_name == 'trade_id' and isinstance(param_value, dict):
            start_id = param_value['start']
            end_id = param_value['end']
            expected_ids = set(range(start_id, end_id + 1))
            
            if 'trade_id' in cached_data.columns:
                cached_ids = set(cached_data['trade_id'].unique())
                missing_ids = expected_ids - cached_ids
                
                if missing_ids:
                    # Create fetch params for missing ranges
                    missing_ranges = []
                    current_start = None
                    current_end = None
                    
                    for trade_id in sorted(missing_ids):
                        if current_start is None:
                            current_start = current_end = trade_id
                        elif trade_id == current_end + 1:
                            current_end = trade_id
                        else:
                            # Gap found, close current range
                            range_params = query_params.copy()
                            range_params['trade_id'] = {'start': current_start, 'end': current_end}
                            missing_ranges.append(range_params)
                            current_start = current_end = trade_id
                    
                    # Add final range
                    if current_start is not None:
                        range_params = query_params.copy()
                        range_params['trade_id'] = {'start': current_start, 'end': current_end}
                        missing_ranges.append(range_params)
                    
                    return missing_ranges
    
    return []
```

## Accessing Cache Instance and Increment Settings

Custom gap detectors receive the cache instance as a third parameter, providing access to increment settings and built-in detection methods.

### Available Cache Instance Properties

```python
def advanced_gap_detector(query_params, cached_data, cache_instance):
    """Demonstrates accessing all available cache instance features."""
    
    # Access user-specified increment settings
    time_increment = cache_instance.time_increment
    ordered_increment = cache_instance.ordered_increment
    
    print(f"Time increment setting: {time_increment}")
    print(f"Ordered increment setting: {ordered_increment}")
    
    # Access built-in detection methods
    if not cached_data.empty:
        if 'timestamp' in cached_data.columns:
            detected_time_increment = cache_instance._detect_granularity(cached_data, 'timestamp')
            print(f"Auto-detected time increment: {detected_time_increment}")
        
        if 'order_id' in cached_data.columns:
            detected_order_increment = cache_instance._detect_ordered_granularity(cached_data, 'order_id')
            print(f"Auto-detected order increment: {detected_order_increment}")
    
    # Use built-in conversion methods
    fetch_params = cache_instance._convert_query_to_fetch_params(query_params)
    
    return [fetch_params]  # or [] for no fetch
```

### Custom Increment Specification

```python
def custom_increment_detector(query_params, cached_data, cache_instance):
    """Example of specifying custom increments based on context."""
    
    # Context-sensitive increment logic
    if 'symbol' in query_params:
        symbol = query_params['symbol'].get('value', '') if isinstance(query_params['symbol'], dict) else query_params['symbol']
        
        if symbol.startswith('BTC'):
            # Cryptocurrency: millisecond precision
            custom_increment = timedelta(milliseconds=100)
            print(f"Crypto detected: using {custom_increment} increment")
            
        elif symbol.startswith('FOREX'):
            # Forex: second precision
            custom_increment = timedelta(seconds=1)
            print(f"Forex detected: using {custom_increment} increment")
            
        else:
            # Regular stocks: use user setting or detect
            if cache_instance.time_increment:
                custom_increment = cache_instance.time_increment
                print(f"Using user setting: {custom_increment}")
            else:
                custom_increment = timedelta(minutes=1)  # Default for stocks
                print(f"Using default stock increment: {custom_increment}")
    
    # Implement gap detection with custom increment
    # ... your custom logic here ...
    
    return []
```

### Hybrid Approach with Built-in Fallback

```python
def hybrid_gap_detector(query_params, cached_data, cache_instance):
    """Combines custom logic with built-in detection as fallback."""
    
    try:
        # Custom logic for special cases
        if 'high_frequency' in str(query_params).lower():
            print("Using high-frequency custom logic")
            # Custom high-frequency gap detection
            return custom_hf_logic(query_params, cached_data, cache_instance)
        
        # For other cases, use built-in logic
        print("Using built-in gap detection logic")
        
        # Temporarily disable custom detector to get built-in result
        original_detector = cache_instance.gap_detector
        cache_instance.gap_detector = None
        
        try:
            return cache_instance._find_missing_data(query_params, cached_data)
        finally:
            cache_instance.gap_detector = original_detector
            
    except Exception as e:
        print(f"Custom logic failed: {e}, falling back to built-in")
        # Graceful fallback - return empty to let built-in handle it
        return []

def custom_hf_logic(query_params, cached_data, cache_instance):
    """Custom high-frequency logic implementation."""
    # Implement specialized logic for high-frequency data
    if cached_data.empty:
        return [cache_instance._convert_query_to_fetch_params(query_params)]
    
    # Custom gap detection with 1-second precision
    # ... implementation ...
    
    return []  # Return appropriate fetch parameters
```

### Multiple Increment Strategies

```python
def multi_strategy_detector(query_params, cached_data, cache_instance):
    """Demonstrates multiple increment strategies in one detector."""
    
    # Strategy 1: Use user-specified increments if available
    if cache_instance.time_increment or cache_instance.ordered_increment:
        print("Strategy 1: Using user-specified increments")
        time_inc = cache_instance.time_increment
        order_inc = cache_instance.ordered_increment
        return user_specified_logic(query_params, cached_data, time_inc, order_inc)
    
    # Strategy 2: Auto-detect from data patterns
    elif not cached_data.empty:
        print("Strategy 2: Auto-detecting from data")
        # Use built-in detection methods
        return auto_detect_logic(query_params, cached_data, cache_instance)
    
    # Strategy 3: Use contextual defaults
    else:
        print("Strategy 3: Using contextual defaults")
        return contextual_default_logic(query_params, cached_data, cache_instance)

def user_specified_logic(query_params, cached_data, time_inc, order_inc):
    """Logic when user specified increments."""
    # Implementation using user settings
    return []

def auto_detect_logic(query_params, cached_data, cache_instance):
    """Logic using auto-detection."""
    # Use cache_instance._detect_granularity() and _detect_ordered_granularity()
    return []

def contextual_default_logic(query_params, cached_data, cache_instance):
    """Logic using contextual defaults."""
    # Implement context-based default increments
    return []
```

## Factory Methods Support

All factory methods support the `gap_detector` parameter:

```python
# SQLite with custom detector
cache = SqlCache.with_sqlite(
    db_path="cache.db",
    table=table,
    data_adapter=adapter,
    gap_detector=my_detector
)

# DuckDB with custom detector  
cache = SqlCache.with_duckdb(
    db_path="analytics.db",
    table=table,
    data_adapter=adapter,
    gap_detector=my_detector
)

# PostgreSQL with custom detector
cache = SqlCache.with_postgresql(
    db_url="postgresql://user:pass@host/db",
    table=table,
    data_adapter=adapter,
    gap_detector=my_detector
)
```

## Error Handling

The system provides graceful fallback:
- If your custom detector raises an exception, it falls back to built-in logic
- This ensures cache operations continue even with detector errors
- Errors are logged for debugging

## Best Practices

1. **Use function-based detectors** for most customization needs
2. **Return empty list `[]`** when no fetch is needed
3. **Return list of parameter dictionaries** when fetch is needed
4. **Handle edge cases** like empty cached data
5. **Test thoroughly** with your specific data patterns
6. **Use built-in intelligence** as fallback for robustness

## Migration from Subclassing

If you're currently using subclassing, you can easily migrate to function-based approach:

```python
# Old way (still supported)
class MyCache(SqlCache):
    def _find_missing_data(self, query_params, cached_data):
        # Custom logic
        return fetch_params

# New way (recommended)
def my_gap_detector(query_params, cached_data, cache_instance):
    # Same custom logic
    return fetch_params

cache = SqlCache.with_sqlite(
    db_path="cache.db",
    table=table,
    data_adapter=adapter,
    gap_detector=my_gap_detector
)
```

The function-based approach provides the same flexibility with better composability and testability.