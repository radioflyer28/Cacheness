# Windows Compatibility

## Overview

Cacheness is fully compatible with Windows, Linux, and macOS. This document describes the Windows-specific considerations and fixes that were implemented to ensure reliable operation on Windows systems.

## Windows-Specific Challenges

### File Locking Behavior

Windows has stricter file locking semantics than Unix-based systems:

1. **SQLite Database Locks**: On Windows, SQLite database files cannot be deleted while connections are open, even if they've been "closed" at the Python level
2. **Temporary Files**: NamedTemporaryFile keeps file handles open by default, preventing writes to the file path
3. **Context Manager Cleanup**: Python's context managers (`with` statements) don't always release Windows file handles immediately

### Solutions Implemented

#### 1. Explicit Resource Cleanup in `UnifiedCache`

The `UnifiedCache` class now includes proper resource management:

```python
def close(self):
    """Explicitly close the metadata backend and release all resources."""
    if hasattr(self, 'metadata_backend') and self.metadata_backend:
        self.metadata_backend.close()
```

Key features:
- Context manager support (`__enter__`/`__exit__`)
- Automatic cleanup in `__del__` for garbage collection
- Explicit `close()` method for manual resource management

#### 2. Aggressive Connection Disposal in `SqliteBackend`

The SQLite backend ensures all connections are released:

```python
def close(self):
    """Close all database connections and clean up resources."""
    if hasattr(self, 'engine') and self.engine:
        self.engine.dispose()  # Close all connections in the pool
        import gc
        gc.collect()  # Force garbage collection to release file handles
```

#### 3. NamedTemporaryFile Pattern

When using `tempfile.NamedTemporaryFile` on Windows, the file handle must be closed before writing:

```python
# CORRECT - Windows compatible
with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_path = Path(temp_file.name)
# File handle closed here
temp_path.write_bytes(content)  # Now we can write

# INCORRECT - Fails on Windows
with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_path = Path(temp_file.name)
    temp_path.write_bytes(content)  # FAILS: file handle still open
```

#### 4. Test Fixture Cleanup Order

Test fixtures must respect resource cleanup order:

```python
@pytest.fixture
def cache_instance(cache_config, temp_cache_dir):
    """Create a cache instance for testing."""
    cache = cacheness(cache_config)
    yield cache
    cache.close()  # MUST close before temp_cache_dir cleanup
```

For fixtures that create temporary directories, use finalizers with delays:

```python
@pytest.fixture
def temp_cache_dir(request):
    """Create a temporary cache directory."""
    temp_dir = tempfile.mkdtemp()
    
    def cleanup():
        import time, gc
        gc.collect()
        time.sleep(0.2)  # Give Windows time to release locks
        if Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                time.sleep(0.5)  # Retry if needed
                shutil.rmtree(temp_dir)
    
    request.addfinalizer(cleanup)
    return temp_dir
```

## Best Practices for Cross-Platform Code

### 1. Always Use Context Managers or Explicit close()

```python
# Good - explicit resource management
cache = UnifiedCache(config)
try:
    cache.put(data)
finally:
    cache.close()

# Better - context manager
with UnifiedCache(config) as cache:
    cache.put(data)
```

### 2. Avoid Assumptions About File Deletion Timing

```python
# Don't assume file is immediately deletable after close()
cache.close()
# On Windows, give OS time to release handle
import gc; gc.collect()
os.remove(cache_file)
```

### 3. Use Platform-Agnostic Path Handling

```python
from pathlib import Path

# Good - works everywhere
cache_dir = Path("cache") / "data"

# Avoid - platform-specific
cache_dir = "cache\\data"  # Windows only
cache_dir = "cache/data"   # Unix-like only
```

## Testing on Windows

### Running Tests

```powershell
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_core.py -v
```

### Common Windows Test Issues

1. **PermissionError [WinError 32]**: File is being used by another process
   - Solution: Ensure `cache.close()` is called before cleanup
   - Solution: Add `gc.collect()` before file deletion

2. **Path length limitations**: Windows has a 260-character MAX_PATH limit
   - Solution: Use shorter temporary directory names in tests
   - Solution: Enable long path support in Windows 10+ (Group Policy)

3. **Symbolic link failures**: Require administrator privileges on Windows
   - Solution: Tests that use symlinks are skipped on Windows

## Platform-Specific Features

### Features That Work Differently

| Feature | Windows | Linux/macOS | Notes |
|---------|---------|-------------|-------|
| Symbolic links | Requires admin | Always available | Tests automatically skip |
| File locking | Strict (exclusive) | Permissive | Windows prevents deletion of open files |
| Path separators | `\` (backslash) | `/` (forward slash) | Use `pathlib.Path` for compatibility |
| Case sensitivity | Insensitive | Sensitive (usually) | Use consistent casing |
| SQLite WAL mode | Full support | Full support | Works identically |

### Features That Are Identical

- SQLite backend functionality
- Caching operations (get/put/delete)
- Metadata storage and querying
- Serialization (pickle, JSON, npz)
- Compression (gzip, lz4, zstd)
- Security (signing, encryption)
- Custom metadata with SQLAlchemy

## Continuous Integration

To ensure cross-platform compatibility, run tests on all platforms:

### GitHub Actions Example

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ["3.9", "3.10", "3.11", "3.12", "3.13"]

steps:
  - uses: actions/checkout@v4
  - name: Install uv
    uses: astral-sh/setup-uv@v4
  - name: Run tests
    run: uv run pytest tests/ -v
```

## Known Limitations

None! All features work identically on Windows, Linux, and macOS after the compatibility fixes.

## Performance Notes

Windows filesystem performance characteristics:

- **SQLite WAL mode**: Provides excellent concurrent read/write performance
- **File I/O**: Comparable to Linux when SSD is used
- **Temporary files**: Use in-memory temp directory for best performance (`TEMP` environment variable)

## Troubleshooting

### "PermissionError: [WinError 32]"

**Problem**: File is being used by another process.

**Solution**:
1. Ensure all `UnifiedCache` instances are properly closed
2. Check for unclosed file handles in custom code
3. Add `gc.collect()` before attempting file deletion
4. Wait briefly (0.1-0.5 seconds) before deletion

### "sqlite3.OperationalError: database is locked"

**Problem**: Another process or thread has a lock on the database.

**Solution**:
1. SQLite uses WAL mode for better concurrency
2. Default busy_timeout is 30 seconds - wait should be automatic
3. Ensure connections are closed properly with `cache.close()`

### Tests Failing on Windows Only

**Problem**: Tests pass on Linux/macOS but fail on Windows.

**Checklist**:
- [ ] Are you calling `cache.close()` in test cleanup?
- [ ] Are fixtures cleaning up in the correct order?
- [ ] Are you using `pathlib.Path` for path operations?
- [ ] Are you assuming files are immediately deletable after close?
- [ ] Are you using symbolic links without skipping on Windows?

## Contributing

When adding new features:

1. Test on both Windows and Unix-like systems
2. Use `pathlib.Path` for all file operations
3. Always provide `close()` methods for resources
4. Use context managers (`__enter__`/`__exit__`)
5. Add explicit resource cleanup in tests

## Related Documentation

- [Configuration](CONFIGURATION.md) - Cache configuration options
- [Performance](PERFORMANCE.md) - Performance optimization strategies
- [Security](SECURITY.md) - Security features and best practices
