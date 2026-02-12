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
   cache.put(result, x=x, y=y, prefix="unique_operation")
   ```

## Test Failures After Changes

**Symptoms:**
- Previously passing tests now failing
- Test count different from baseline (787 passed, 70 skipped)
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

## Thread Safety Issues

**Symptoms:**
- Intermittent errors under concurrent access
- Data corruption in multi-threaded scenarios
- "Database is locked" with SQLite

**Important Notes:**
- `UnifiedCache` has `_lock` field but **never acquires it** - not thread-safe
- JSON backend: Protected by `threading.Lock()` at backend level
- SQLite backend: Uses WAL mode + per-operation locks
- PostgreSQL backend: Uses `ThreadPoolExecutor` for connection pooling

**Solutions:**
1. **Use backend-level concurrency protection** (already in place)
2. **Use process-level parallelism instead of threads** when possible
3. **Implement external locking** if coordinating across multiple cache instances

## Additional Resources

- **Performance Guide:** [`PERFORMANCE.md`](PERFORMANCE.md) - Optimization strategies and benchmarks
- **Backend Selection:** [`BACKEND_SELECTION.md`](BACKEND_SELECTION.md) - Choosing the right backend
- **Security Guide:** [`SECURITY.md`](SECURITY.md) - Cache signing and verification
- **API Reference:** [`API_REFERENCE.md`](API_REFERENCE.md) - Complete API documentation
