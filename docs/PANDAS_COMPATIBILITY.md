# Pandas Version Compatibility

## Overview

Cacheness supports **pandas 2.0 and newer**, including pandas 2.3.x and the upcoming pandas 3.0.x series. The library uses standard pandas APIs that are stable across these versions.

## Supported Versions

| pandas Version | Status | Notes |
|----------------|--------|-------|
| 2.0.x | ✅ Fully supported | Minimum version |
| 2.1.x | ✅ Fully supported | All features work |
| 2.2.x | ✅ Fully supported | All features work |
| 2.3.x | ✅ Fully supported | Tested with 2.3.1 |
| 3.0.x | ✅ Compatible | Future versions supported |

## Installation

Cacheness automatically supports all pandas 2.x and 3.x versions:

```bash
# Install with pandas support
pip install cacheness[dataframes]

# Or with all recommended dependencies
pip install cacheness[recommended]
```

The version constraint is `pandas>=2.0.0,<4.0.0`, providing maximum flexibility while ensuring compatibility.

## pandas 2.0+ Features Supported

### Nullable Data Types

pandas 2.0 introduced improved nullable integer, boolean, and string types:

```python
import pandas as pd
from cacheness import cacheness, CacheConfig

# Nullable integer dtype
df = pd.DataFrame({
    'nullable_int': pd.array([1, 2, None, 4, 5], dtype='Int64'),
    'regular': [10, 20, 30, 40, 50]
})

cache = cacheness(CacheConfig(cache_dir="cache"))
cache.put(df, key="nullable_data")
retrieved = cache.get(key="nullable_data")
```

### String Dtype

The dedicated string dtype is fully supported:

```python
df = pd.DataFrame({
    'string_col': pd.array(['a', 'b', 'c', None, 'e'], dtype='string'),
    'values': [1, 2, 3, 4, 5]
})

cache.put(df, key="string_data")
```

### PyArrow Backend

pandas 2.0+ PyArrow-backed string and numeric types work seamlessly:

```python
df = pd.DataFrame({
    'strings': pd.array(['a', 'b', 'c'], dtype='string[pyarrow]'),
    'ints': pd.array([1, 2, 3], dtype='int64[pyarrow]')
})

cache.put(df, key="pyarrow_data")
```

### Copy-on-Write (CoW)

pandas 2.0+ introduced Copy-on-Write mode. Cacheness works correctly regardless of CoW settings:

```python
# CoW mode (pandas 2.0+ default in future versions)
pd.options.mode.copy_on_write = True

df = pd.DataFrame({'a': [1, 2, 3]})
cache.put(df, key="cow_data")
```

## API Compatibility

Cacheness uses only stable pandas APIs that work across all 2.x and 3.x versions:

### DataFrame Methods Used

| Method | Version | Notes |
|--------|---------|-------|
| `to_parquet()` | 2.0+ | Primary storage format |
| `read_parquet()` | 2.0+ | Loading cached data |
| `to_frame()` | 2.0+ | Series to DataFrame conversion |
| `columns.tolist()` | 2.0+ | Metadata extraction |
| `shape` | 2.0+ | DataFrame dimensions |
| `dtypes` | 2.0+ | Column type information |

### Series Methods Used

| Method | Version | Notes |
|--------|---------|-------|
| `to_frame()` | 2.0+ | Series caching |
| `name` | 2.0+ | Series name preservation |

All these APIs are stable and work identically across pandas 2.0-3.x.

## Storage Format

Cacheness uses **Apache Parquet** as the primary storage format for DataFrames and Series:

- **Format**: Parquet (via `pyarrow` or `fastparquet`)
- **Compression**: Configurable (default: snappy)
- **Compatibility**: Works with all pandas 2.x and 3.x versions
- **Features**: Preserves dtypes, nullable values, and index

## Testing

Cacheness includes comprehensive pandas compatibility tests:

```bash
# Run pandas compatibility tests
pytest tests/test_pandas_compatibility.py -v

# Run all DataFrame-related tests
pytest tests/ -k "pandas or dataframe" -v
```

## Migration from pandas 1.x

If upgrading from pandas 1.x to 2.x:

1. **Minimum version**: Cacheness requires pandas 2.0+
2. **Deprecated APIs**: Cacheness doesn't use any pandas 1.x deprecated APIs
3. **Breaking changes**: See [pandas 2.0 release notes](https://pandas.pydata.org/docs/whatsnew/v2.0.0.html)

### Key pandas 2.0 Changes

Changes that **don't affect Cacheness** (handled automatically):

- ✅ Improved nullable dtypes - fully supported
- ✅ PyArrow backend - works seamlessly  
- ✅ Copy-on-Write mode - compatible
- ✅ Index behavior changes - Parquet preserves index correctly

## Performance

pandas 2.0+ improvements that benefit Cacheness:

- **Faster Parquet I/O**: Up to 2x faster reading/writing
- **Better memory efficiency**: Nullable dtypes use less memory
- **PyArrow integration**: Improved performance with PyArrow backend

## Version-Specific Considerations

### pandas 2.0.x

- First version with major dtype improvements
- PyArrow backend introduced
- All Cacheness features work perfectly

### pandas 2.1.x - 2.3.x

- Incremental improvements and bug fixes
- Enhanced Parquet support
- Better PyArrow integration
- All Cacheness features work perfectly

### pandas 3.0.x (Upcoming)

- Expected to maintain API compatibility
- Cacheness is designed to work with pandas 3.x
- Version constraint allows pandas 3.x: `pandas>=2.0.0,<4.0.0`

## Troubleshooting

### "pandas version too old"

```
Error: pandas>=2.0.0 is required
```

**Solution**: Upgrade pandas:
```bash
pip install --upgrade pandas
```

### PyArrow Not Available

If PyArrow-backed types fail:

```python
# Install PyArrow
pip install pyarrow>=21.0.0
```

### Parquet Compatibility

If you see Parquet-related errors:

```python
# Ensure PyArrow is installed
pip install pyarrow>=21.0.0

# Or use fastparquet as alternative
pip install fastparquet>=2023.0.0
```

## Best Practices

### Use Modern Dtypes

Take advantage of pandas 2.0+ nullable dtypes:

```python
# Good - nullable integer
df = pd.DataFrame({'col': pd.array([1, 2, None], dtype='Int64')})

# Old style - float64 with NaN
df = pd.DataFrame({'col': [1.0, 2.0, float('nan')]})
```

### Enable Copy-on-Write

For better performance with pandas 2.1+:

```python
import pandas as pd
pd.options.mode.copy_on_write = True
```

### Check DataFrame Compatibility

Before caching, verify your DataFrame can be stored as Parquet:

```python
import io

# Test if DataFrame is Parquet-compatible
try:
    df.to_parquet(io.BytesIO())
    print("✅ DataFrame is Parquet-compatible")
except Exception as e:
    print(f"❌ Not Parquet-compatible: {e}")
```

## Future Compatibility

Cacheness is designed to work with future pandas versions:

- Uses only stable, documented APIs
- Comprehensive compatibility test suite
- Version constraint allows pandas 3.x
- Regular testing against pandas development versions

## Related Documentation

- [Configuration Guide](CONFIGURATION.md) - Parquet compression settings
- [Performance Guide](PERFORMANCE.md) - DataFrame caching optimization
- [API Reference](API_REFERENCE.md) - Complete API documentation

## Changelog

### pandas Support History

- **v0.3.x**: Added explicit pandas 2.0-3.x compatibility
- **v0.3.x**: Comprehensive pandas compatibility test suite
- **v0.2.x**: pandas 2.0+ required
- **v0.1.x**: pandas 1.x support (deprecated)
