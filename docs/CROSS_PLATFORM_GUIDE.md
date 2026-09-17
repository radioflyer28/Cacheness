# Cross-Platform Development Guide

## Overview

Cacheness supports Python 3.11 and newer, but platform qualification is
deliberately tiered. The current development checkout has these evidence
boundaries:

- **Linux** is the full stable-matrix qualification target.
- **macOS** has only public-topology boundary-smoke coverage.
- **Native Windows** is `UNAVAILABLE` / `NOT_QUALIFIED`.

The package source uses portable Python interfaces where practical, while its
installed dependency set can resolve platform-specific binary wheels. Refer to
[Release qualification](RELEASE_QUALIFICATION.md) for the sole detailed owner
of platform evidence and nonclaims.

## Wheel Distribution

### Universal Wheel

Cacheness builds a project wheel tagged `py3-none-any`:
```bash
uv build --wheel
# Produces: cacheness-x.x.x-py3-none-any.whl
```

The tag describes the Cacheness project wheel only. Installation still resolves
the declared dependency set for the target platform; those dependencies can
include native or platform-specific wheels. A successful installation or
portable wheel tag is not evidence of platform qualification.

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

#### Complete repository suite

Run the complete repository suite in a fresh, lockfile-backed environment:

```bash
uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false
```

This is the supported complete-suite command. It installs all declared extras and
the development test group in a new isolated environment from `uv.lock`, so it
does not modify packages beneath an already-running test process.

`uv run pytest` is intentionally not the complete-suite contract: a base
installation does not include the optional SQLAlchemy, pandas, or S3 dependencies
that their corresponding repository test modules exercise. During an earlier
shared-environment run, nested `uv run ruff` resolution removed live package
files, which then appeared as three public-import failures, 23 S3/botocore setup
errors, and 13 SQL/pandas failures. That was a test-invocation/test-isolation
cascade, not evidence of independent S3, pandas, or public-API product defects;
reproduce failures from the isolated command before treating them as
repository behavior failures.

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

Do not treat a historical count, a local test result, or an exploratory CI row
as a support claim. The current qualified boundaries are Linux full-matrix
coverage and macOS boundary smoke only. Native Windows has no qualification.
The supported commands and the evidence they can establish are maintained in
[Release qualification](RELEASE_QUALIFICATION.md).

## Continuous Integration

### Exploratory GitHub Actions Example

An exploratory operating-system matrix can find portability regressions. It is
not the qualification matrix and does not make a native Windows or full macOS
support claim.

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

Native Windows is currently `UNAVAILABLE` / `NOT_QUALIFIED`. The following
notes are development guidance for an exploratory environment, not evidence of
supported behavior:

- File locking is stricter and requires explicit `close()` calls.
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

Linux is the full stable-matrix qualification target. Qualification evidence
is tied to its recorded revision and command; it does not establish universal
performance or behavior equivalence.

**Recommendations:**
```python
# Respect XDG Base Directory specification
import os
cache_dir = Path(os.environ.get('XDG_CACHE_HOME', 
                                 Path.home() / '.cache')) / 'myapp'
```

### macOS

macOS is limited to Python 3.11/3.14 public-topology boundary smoke. A local
Darwin result does not establish Linux equivalence or a full macOS matrix.

**Recommendations:**
```python
# Use platform-standard cache location
cache_dir = Path.home() / 'Library' / 'Caches' / 'myapp'
```

## Performance Characteristics

Performance evidence is separate from functional qualification. Controlled
Linux performance remains deferred, and macOS timings are diagnostic only.
See [Release qualification](RELEASE_QUALIFICATION.md) rather than inferring
comparative I/O or concurrency performance from this guide.

## Distribution and Installation

### PyPI Publishing

Build and install the Cacheness project wheel using the normal packaging
commands. Installation does not qualify the target platform:

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

Cacheness has a Linux full-matrix target and macOS boundary-smoke evidence for
this checkout. Native Windows remains `UNAVAILABLE` / `NOT_QUALIFIED`.
Consult [Release qualification](RELEASE_QUALIFICATION.md) before treating any
test, wheel, or benchmark result as broader platform support.

## Related Documentation

- [Windows Compatibility](WINDOWS_COMPATIBILITY.md) - Windows-specific details and troubleshooting
- [Cache policy guide](CACHE_POLICY.md) - Current platform-agnostic cache construction
- [Performance Guide](PERFORMANCE.md) - Platform-specific performance tips
