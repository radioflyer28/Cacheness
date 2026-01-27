# Cross-Platform Development Guide

## Overview

Cacheness is a **pure Python library** that works identically across all major platforms:
- ✅ **Windows** (Windows 10+, tested on Windows 11)
- ✅ **Linux** (Ubuntu, Debian, RHEL, etc.)
- ✅ **macOS** (Intel and Apple Silicon)

The library uses platform-agnostic code and standard Python libraries, requiring **no platform-specific compilation or binary dependencies**.

## Wheel Distribution

### Universal Wheel

Cacheness builds as a **universal wheel** (`py3-none-any`):
```bash
uv build --wheel
# Produces: cacheness-x.x.x-py3-none-any.whl
```

This single wheel works on **all platforms** and Python 3.11+ interpreters because:
- Pure Python implementation (no C extensions)
- Platform-agnostic path handling using `pathlib`
- Cross-platform dependencies (SQLAlchemy, NumPy, etc.)
- No platform-specific code paths

### Building Distributions

```bash
# Build both wheel and source distribution
uv build

# Build only wheel
uv build --wheel

# Build only source distribution
uv build --sdist

# Custom output directory
uv build --out-dir dist/
```

## Cross-Platform Compatibility Strategy

### 1. Path Handling

**Always use `pathlib.Path`** for all file operations:

```python
from pathlib import Path

# ✅ GOOD - Works on all platforms
cache_dir = Path("cache") / "data"
config_file = Path.home() / ".config" / "app.json"

# ❌ BAD - Platform-specific
cache_dir = "cache\\data"  # Windows only
cache_dir = "cache/data"   # Unix-like only
```

**Path normalization for hashing:**
```python
# Always convert to POSIX paths for consistent hashing
relative_path = file_path.relative_to(base_path).as_posix()
```

### 2. File Locking and Resources

**Always explicitly close resources:**

```python
# ✅ GOOD - Explicit cleanup
cache = UnifiedCache(config)
try:
    result = cache.get(key="value")
finally:
    cache.close()

# ✅ BETTER - Context manager
with UnifiedCache(config) as cache:
    result = cache.get(key="value")
```

**Why this matters:**
- Windows requires exclusive file locks for deletion
- SQLite connections must be closed before database files can be deleted
- NamedTemporaryFile keeps handles open on Windows

### 3. Temporary Files

**Correct NamedTemporaryFile usage:**

```python
import tempfile
from pathlib import Path

# ✅ GOOD - Windows compatible
with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_path = Path(temp_file.name)
# File handle closed here, now safe to write
temp_path.write_bytes(content)

# ❌ BAD - Fails on Windows
with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_path = Path(temp_file.name)
    temp_path.write_bytes(content)  # File handle still open!
```

### 4. Line Endings

**Python handles line ending normalization automatically** when opening files in text mode:
- Windows: `\r\n` (CRLF)
- Unix/macOS: `\n` (LF)

```python
# Text mode - automatic conversion
with open("file.txt", "r") as f:
    content = f.read()  # Always uses \n internally

# Binary mode - no conversion
with open("file.txt", "rb") as f:
    content = f.read()  # Preserves original line endings
```

## Testing Across Platforms

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=cacheness --cov-report=html

# Run specific test categories
uv run pytest tests/ -m "not slow"
uv run pytest tests/ -k "test_cache"
```

### Platform-Specific Test Handling

Some tests automatically adapt to platform capabilities:

```python
# test_file_hashing.py handles Windows MAX_PATH limitations
import platform

if platform.system() == 'Windows':
    max_depth = 5  # Shorter for Windows MAX_PATH
    filename_length = 50
else:
    max_depth = 10
    filename_length = 100
```

**Symbolic link tests** gracefully skip on Windows when admin privileges aren't available:
```python
try:
    symlink_file.symlink_to(original_file)
    # Test symlink handling
except (OSError, NotImplementedError):
    pytest.skip("Symlinks not supported")
```

### Expected Test Results

| Platform | Passed | Failed | Skipped | Notes |
|----------|--------|--------|---------|-------|
| Windows  | 475    | 0      | 8       | Symlinks (6), TensorFlow (2) |
| Linux    | 475    | 0      | 2       | TensorFlow only |
| macOS    | 475    | 0      | 2       | TensorFlow only |

**Skipped tests on Windows:**
- `test_symlink_handling` (6 tests) - Requires admin privileges
- `test_tensorflow_*` (2 tests) - TensorFlow mutex issues (all platforms)

## Continuous Integration

### GitHub Actions Example

```yaml
name: Cross-Platform Tests

on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.11", "3.12", "3.13"]
    
    runs-on: ${{ matrix.os }}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install uv
      uses: astral-sh/setup-uv@v4
      with:
        version: "latest"
    
    - name: Set up Python
      run: uv python install ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: uv sync --all-extras
    
    - name: Run tests
      run: uv run pytest tests/ -v
    
    - name: Run linter
      run: uv run ruff check src/
```

## Platform-Specific Considerations

### Windows

**Advantages:**
- Excellent SQLite WAL mode performance
- Native path handling with `pathlib`
- Full feature parity with Unix

**Considerations:**
- Stricter file locking (requires explicit `close()`)
- MAX_PATH limitation (260 characters) - use short cache paths
- Symbolic links require administrator privileges
- Case-insensitive filesystem (usually)

**Recommendations:**
```python
# Use shorter cache directory names on Windows
if platform.system() == 'Windows':
    cache_dir = Path("C:/cache")
else:
    cache_dir = Path("/var/cache/myapp")
```

### Linux

**Advantages:**
- Permissive file locking
- Unlimited path lengths
- Native symbolic link support
- Case-sensitive filesystem

**Considerations:**
- File permissions may restrict cache directory access
- Different temp directory locations (`/tmp`, `/var/tmp`)

**Recommendations:**
```python
# Respect XDG Base Directory specification
import os
cache_dir = Path(os.environ.get('XDG_CACHE_HOME', 
                                 Path.home() / '.cache')) / 'myapp'
```

### macOS

**Advantages:**
- Similar to Linux for most operations
- Native symbolic link support
- Excellent SQLite performance

**Considerations:**
- Case-insensitive by default (but case-preserving)
- Apple Silicon (ARM64) vs Intel (x86_64) compatibility
- Gatekeeper may require notarization for distribution

**Recommendations:**
```python
# Use platform-standard cache location
cache_dir = Path.home() / 'Library' / 'Caches' / 'myapp'
```

## Performance Characteristics

### SQLite Performance

All platforms benefit from WAL mode:
- **Concurrent reads**: Unlimited
- **Concurrent writes**: Single writer, multiple readers
- **Performance**: Similar across platforms on SSD storage

### File I/O Performance

| Operation | Windows | Linux | macOS |
|-----------|---------|-------|-------|
| Sequential read | ✅ Excellent | ✅ Excellent | ✅ Excellent |
| Random read | ✅ Good | ✅ Excellent | ✅ Excellent |
| Sequential write | ✅ Good | ✅ Excellent | ✅ Excellent |
| Directory listing | ⚠️ Slower | ✅ Fast | ✅ Fast |

**Optimization tip**: On Windows, minimize directory scanning by using specific cache queries instead of `list_entries()` on large caches.

## Distribution and Installation

### PyPI Publishing

The universal wheel works on all platforms:

```bash
# Build distributions
uv build

# Publish to PyPI (requires authentication)
uv publish
```

### Installation

Users install the same way on all platforms:

```bash
# Basic installation
pip install cacheness

# With recommended dependencies
pip install cacheness[recommended]

# With all extras
pip install cacheness[recommended,dataframes]
```

## Debugging Platform Issues

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

cache = UnifiedCache(config)
```

### Check Platform Information

```python
import platform
import sys

print(f"Platform: {platform.system()}")
print(f"Architecture: {platform.machine()}")
print(f"Python: {sys.version}")
print(f"Path separator: {os.sep}")
```

### Verify Resource Cleanup

```python
import gc

cache = UnifiedCache(config)
# Use cache...
cache.close()
gc.collect()  # Force cleanup

# Verify no file handles remain open
# On Windows: Use Process Explorer
# On Linux: ls -la /proc/<pid>/fd
# On macOS: lsof -p <pid>
```

## Common Pitfalls

### ❌ Platform-Specific Paths

```python
# BAD - Won't work on Windows
config_file = "/etc/myapp/config.json"

# GOOD - Cross-platform
config_file = Path.home() / ".config" / "myapp" / "config.json"
```

### ❌ Assuming File Deletion Timing

```python
# BAD - May fail on Windows
cache.close()
os.remove(cache_file)  # Might still be locked

# GOOD - Explicit cleanup with retry
cache.close()
gc.collect()
time.sleep(0.1)  # Give OS time to release
try:
    os.remove(cache_file)
except PermissionError:
    time.sleep(0.5)
    os.remove(cache_file)
```

### ❌ Hardcoded Separators

```python
# BAD - Only works on Windows
path = "cache\\data\\file.json"

# GOOD - Cross-platform
path = Path("cache") / "data" / "file.json"
```

## Conclusion

Cacheness is **fully cross-platform** with:
- ✅ Single universal wheel for all platforms
- ✅ No platform-specific code or compilation
- ✅ Identical API and behavior everywhere
- ✅ 475/475 tests passing on Windows, Linux, macOS
- ✅ No need for platform-specific builds

The library follows Python best practices for cross-platform compatibility and leverages the standard library's platform abstraction layers.

## Related Documentation

- [Windows Compatibility](WINDOWS_COMPATIBILITY.md) - Windows-specific details and troubleshooting
- [Configuration Guide](CONFIGURATION.md) - Platform-agnostic configuration options
- [Performance Guide](PERFORMANCE.md) - Platform-specific performance tips
