# Troubleshooting Guide

Common issues and their solutions when working with Cacheness.

## SQLite "database is locked"

**Symptoms:**
- `sqlite3.OperationalError: database is locked`
- Operations timing out after 30 seconds
- Errors under heavy concurrent write load

**Cause:** 
Heavy concurrent writes exceeding SQLite's 30-second timeout. SQLite uses file-level locking which can cause contention under high concurrency.

**Solutions:**
1. **Reduce concurrent writers** - Limit number of processes/threads writing simultaneously
2. **Increase timeout** - Configure longer timeout in SQLite connection settings
3. **Use PostgreSQL backend** - Better suited for high-concurrency scenarios
4. **Use write batching** - Group multiple cache operations into fewer transactions

**Prevention:**
- Use SQLite for <10 concurrent writers
- Use PostgreSQL for distributed/high-concurrency scenarios

## Orphaned Blob Files

**Symptoms:**
- Blob files exist in cache directory without corresponding metadata entries
- `list_entries()` shows fewer entries than files in cache directory
- Disk space used more than expected

**Cause:**
Hard crash (power loss, kill -9, system crash) between blob file write and metadata write. Cacheness writes blobs first, then metadata, so crashes leave orphaned blobs.

**Solutions:**
1. **Detect orphans:**
   ```python
   issues = cache.verify_integrity(repair=False)
   print(f"Found {len(issues)} orphaned blobs")
   ```

2. **Clean up orphans:**
   ```python
   issues = cache.verify_integrity(repair=True)
   print(f"Cleaned up {len(issues)} orphaned blobs")
   ```

**Prevention:**
- Use managed shutdowns when possible
- Run periodic integrity checks in production
- Consider using checkpoint/restart patterns for long-running jobs

## JSON Backend Performance Degradation

**Symptoms:**
- Cache operations get slower as cache grows
- Write operations taking seconds instead of milliseconds
- High CPU usage during cache writes

**Cause:**
JSON backend has O(n²) scaling - re-serializes entire metadata file on each write. With 500+ entries, this becomes prohibitively slow.

**Solutions:**
1. **Switch to SQLite backend:**
   ```python
   cache = UnifiedCache(
       cache_dir="./cache",
       metadata_backend="sqlite"  # Change from "json"
   )
   ```

2. **Migrate existing data:**
   ```python
   # Export from JSON backend
   old_cache = UnifiedCache(cache_dir="./cache", metadata_backend="json")
   entries = old_cache.list_entries()
   
   # Import to SQLite backend
   new_cache = UnifiedCache(cache_dir="./cache_sqlite", metadata_backend="sqlite")
   for entry in entries:
       obj = old_cache.get(entry['cache_key'])
       new_cache.put(obj, **entry['args'])
   ```

**Prevention:**
- Use JSON backend only for <200 entries
- Use SQLite backend for production deployments
- Monitor cache size and switch before performance degrades

## Handler Error Types Changed (v0.5.2+)

**Symptoms:**
- Code that previously caught `ImportError` or bare `Exception` from Parquet handler `get()`/`put()` now receives `CacheReadError` or `CacheWriteError`
- Applies to `PandasDataFrameHandler`, `PandasSeriesHandler`, `PolarsSeriesHandler` (in addition to `PolarsDataFrameHandler` which already had this behavior)

**Cause:**
As of v0.5.2, all four Parquet handlers consistently wrap errors in `CacheWriteError`/`CacheReadError` with `cache_operation_context`. Previously, only `PolarsDataFrameHandler` did this.

**Solution:**
Update exception handlers to catch the Cacheness error types:
```python
from cacheness.interfaces import CacheReadError, CacheWriteError

try:
    data = cache.get(cache_key="my_key")
except CacheReadError as e:
    print(f"Handler: {e.handler_type}, Message: {e}")
```

> **Note:** If you were catching bare `Exception`, no change is needed — `CacheReadError`/`CacheWriteError` are subclasses of `Exception`.

## Import Errors After Adding Dependencies

**Symptoms:**
- `ImportError: No module named 'pandas'` (or other optional dependencies)
- Tests failing with missing imports
- Handler registration errors

**Cause:**
Cacheness has optional dependency groups that aren't installed by default. Running `uv sync` without `--all-groups` only installs core dependencies.

**Solutions:**
1. **Install all dependencies:**
   ```bash
   uv sync --all-groups
   ```

2. **Install specific dependency group:**
   ```bash
   uv add pandas  # For dataframe support
   uv add boto3   # For S3 support
   ```

**Prevention:**
- Always run `uv sync --all-groups` when setting up development environment
- Check `pyproject.toml` for optional dependency groups
- Document which optional features your project uses

## Cache Key Collisions

**Symptoms:**
- Different function calls returning same cached value
- Cache hits when expecting misses
- Unexpected cache behavior with similar arguments

**Cause:**
Cache key generation doesn't properly distinguish between different argument combinations. Most common with:
- Mutable default arguments
- Functions with `**kwargs` where named params affect caching
- Custom objects without proper `__repr__` or `__hash__`

**Solutions:**
1. **Avoid mutable defaults:**
   ```python
   # Bad
   def process(data={}):
       ...
   
   # Good
   def process(data=None):
       data = data or {}
   ```

2. **Implement proper `__repr__`:**
   ```python
   class MyType:
       def __repr__(self):
           return f"MyType(field={self.field!r})"
   ```

3. **Use custom cache key:**
   ```python
   cache.put(result, x=x, y=y, cache_key="unique_operation")
   ```

## Test Failures After Changes

**Symptoms:**
- Previously passing tests now failing
- Test count different from baseline (1424 passed, 65 skipped)
- Intermittent test failures

**Common Causes & Fixes:**

1. **Stale test database:**
   ```bash
   rm -rf tests/__pycache__
   rm -rf .pytest_cache
   ```

2. **Dirty cache directory:**
   ```bash
   rm -rf cache/
   ```

3. **Import errors from new dependencies:**
   ```bash
   uv sync --all-groups
   ```

4. **Quality gate errors:**
   ```bash
   .\scripts\quality-check.ps1
   cat .quality-errors.log
   ```

## Cache Entries Disappearing on Get Errors

**Symptoms:**
- Cached entries vanish after deserialization failures
- `get()` returns `None` and the entry no longer exists

**Cause:** By default, `get()` auto-deletes entries that fail to load (corrupted files, handler mismatch, etc.). This is the `delete_on_error=True` behaviour.

**Solution — preserve entries for debugging:**

```python
from cacheness import cacheness, CacheConfig, CacheMetadataConfig

config = CacheConfig(
    metadata=CacheMetadataConfig(delete_on_error=False)
)
cache = cacheness(config)

# Now get() returns None on errors but keeps the entry intact
data = cache.get(experiment="broken")  # None, but entry still in metadata
```

> **Note:** In storage mode (`storage_mode=True`), entries are *never* deleted on errors regardless of this setting.

## Thread Safety Issues

**Symptoms:**
- Intermittent errors under concurrent access
- Data corruption in multi-threaded scenarios
- "Database is locked" with SQLite

**Concurrency Model (since v0.8.0):**
- `UnifiedCache` uses a reentrant lock (`threading.RLock`) that protects **all** public methods including `put()`, `get()`, `invalidate()`, batch operations, and management APIs
- JSON backend: Protected by its own `RLock` for in-memory dict and disk I/O
- SQLite backend: Uses its own `RLock` plus SQLAlchemy sessions with WAL mode
- PostgreSQL backend: Uses SQLAlchemy session management with connection pooling

**Solutions:**
1. **Single process, multiple threads:** Fully thread-safe — share a single `UnifiedCache` instance across threads
2. **Multiple processes, same cache directory:** Use SQLite backend (WAL mode handles concurrent access) or PostgreSQL. JSON backend is NOT safe for multi-process access
3. **"Database is locked" errors:** Ensure you're using SQLite backend (not JSON) and that WAL mode is enabled (default)

## Additional Resources

- **Performance Guide:** [`PERFORMANCE.md`](PERFORMANCE.md) - Optimization strategies and benchmarks
- **Backend Selection:** [`BACKEND_SELECTION.md`](BACKEND_SELECTION.md) - Choosing the right backend
- **Security Guide:** [`SECURITY.md`](SECURITY.md) - Cache signing and verification
- **API Reference:** [`API_REFERENCE.md`](API_REFERENCE.md) - Complete API documentation
