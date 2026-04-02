# Testing Patterns

**Analysis Date:** 2026-04-02

## Test Framework

**Runner:**
- **pytest** >= 8.4.1 (configured in `pyproject.toml`)
- Config section: `[tool.pytest.ini_options]`

**Key Plugins:**
- `pytest-xdist` >= 3.8.0 — parallel test execution (`-n auto --dist loadgroup`)
- `pytest-cov` >= 6.2.1 — coverage reporting
- `hypothesis` >= 6.151.5 — property-based testing
- `moto[s3]` >= 5.0.0 — AWS S3 mocking

**Assertion Library:**
- Built-in pytest assertions (no third-party assertion library)

**Run Commands:**
```bash
# Full test suite (parallel, default)
uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py

# Sequential run (disable xdist)
uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py -p no:xdist

# Specific test file
uv run pytest tests/test_core.py -x -q

# With coverage
uv run pytest tests/ --cov=cacheness --cov-report=html

# With verbose logging
uv run pytest tests/ --log-cli-level=INFO
```

**Baseline:** 1427 passed, 102 skipped, 0 failures (~48s parallel, ~237s sequential)

## Test File and Directory Structure

**Location:** All tests in a top-level `tests/` directory (separate from source)

**Naming:** `test_*.py` — one file per concern area, not per source file

**Structure:**
```
tests/
├── conftest.py                         # Shared fixtures (PostgreSQL, S3, mocked S3)
├── test_core.py                        # Core UnifiedCache + CacheConfig
├── test_handlers.py                    # Handler system (Array, Object, DataFrame, Series)
├── test_metadata.py                    # Metadata backend integration
├── test_blob_store.py                  # BlobStore low-level API
├── test_decorators.py                  # @cached, @cache_if decorators
├── test_error_handling.py              # Error handling module
├── test_config_validation.py           # Config validation
├── test_config_options.py              # Config options
├── test_update_operations.py           # Update/upsert operations
├── test_cache_integrity.py             # Cache integrity verification
├── test_cache_integrity_verification.py # Integrity audit
├── test_cache_signing.py               # HMAC signing
├── test_concurrency_stress.py          # Thread safety stress tests
├── test_fault_injection.py             # Fault injection via mock.patch
├── test_property_based.py              # Hypothesis property-based tests
├── test_backend_parity.py              # Backend behavioral parity
├── test_backend_compatibility.py       # Backend compatibility
├── test_schema_versioning.py           # Schema migration
├── test_sqlite_schema_versioning.py    # SQLite-specific schema versioning
├── test_json_schema_versioning.py      # JSON-specific schema versioning
├── test_pg_schema_versioning.py        # PostgreSQL schema versioning
├── test_namespace_config.py            # Namespace configuration
├── test_namespace_integration.py       # Namespace integration
├── test_namespace_isolation.py         # Namespace isolation
├── test_namespace_plumbing.py          # Namespace plumbing
├── test_namespace_signing.py           # Namespace signing
├── test_s3_blob_backend.py             # S3 blob backend (moto-mocked)
├── test_s3_etag_integrity.py           # S3 ETag verification
├── test_s3_etag_metadata.py            # S3 ETag metadata
├── test_s3_orphan_cleanup.py           # S3 orphan cleanup
├── test_docker_integration.py          # Docker-based PostgreSQL/S3 tests
├── test_tensorflow_handler.py          # TensorFlow handler (Windows-excluded)
├── test_entry_list.py                  # EntryList wrapper
├── test_serialization.py              # Serialization
├── test_compress_pickle.py             # Compression
├── test_size_utils.py                  # Size/duration parsing
├── test_file_hashing.py                # File hashing
├── test_directory_sharding.py          # Directory sharding
├── test_storage_mode.py                # Storage mode (BlobStore standalone)
├── test_custom_metadata.py             # Custom metadata models
├── test_custom_metadata_backends.py    # Custom metadata backends
├── test_custom_metadata_namespace.py   # Custom metadata namespacing
├── ...                                 # ~65 test files total
```

## Source-to-Test File Mapping

| Source file changed | Primary test files |
|---------------------|--------------------|
| `src/cacheness/core.py` | `tests/test_core.py`, `tests/test_storage_mode.py` |
| `src/cacheness/storage/blob_store.py` | `tests/test_blob_store.py`, `tests/test_blob_namespace.py` |
| `src/cacheness/storage/backends/blob_backends.py` | `tests/test_blob_store.py`, `tests/test_s3_blob_backend.py` |
| `src/cacheness/storage/backends/s3_backend.py` | `tests/test_s3_blob_backend.py` |
| `src/cacheness/handlers.py` | `tests/test_handlers.py` |
| `src/cacheness/metadata.py` | `tests/test_metadata.py`, `tests/test_sqlite_schema_versioning.py` |
| `src/cacheness/size_utils.py`, `src/cacheness/config.py` | `tests/test_size_utils.py`, `tests/test_core.py` |
| `src/cacheness/security.py` | `tests/test_cache_signing.py` |
| Path/namespace logic | `tests/test_directory_sharding.py`, `tests/test_namespace_config.py` |

## Fixture Patterns

**Shared fixtures** in `tests/conftest.py`:
- **Session-scoped availability checks:** `postgres_available`, `s3_available` — check Docker containers with short timeouts, return `bool`
- **Session-scoped engines:** `postgres_engine` — creates SQLAlchemy engine, skips if unavailable
- **Function-scoped connections:** `postgres_connection`, `postgres_clean_db` — per-test isolation with schema teardown
- **Mocked AWS:** `mock_s3_client`, `mock_s3_bucket` — uses `moto.mock_aws()` context manager, no Docker required

**Common in-test fixture patterns:**

```python
@pytest.fixture
def temp_cache_dir(self):
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / "test_cache"

@pytest.fixture
def test_config(self, temp_cache_dir):
    """Create a test configuration."""
    config = CacheConfig(
        cache_dir=str(temp_cache_dir),
        metadata_backend="json",
        cleanup_on_init=False,
    )
    return config
```

**BlobStore fixtures** in `tests/test_blob_store.py`:
```python
@pytest.fixture
def blob_dir(tmp_path):
    return tmp_path / "blobs"

@pytest.fixture
def store(blob_dir):
    return BlobStore(cache_dir=blob_dir, backend="json")

@pytest.fixture
def signed_store(blob_dir):
    return BlobStore(cache_dir=blob_dir, backend="json",
                     enable_signing=True, use_in_memory_key=True)
```

**Cache lifecycle pattern:** Create + yield + close:
```python
@pytest.fixture
def stress_cache():
    with tempfile.TemporaryDirectory() as temp_dir:
        config = CacheConfig(cache_dir=temp_dir, metadata_backend="sqlite")
        cache = UnifiedCache(config=config)
        yield cache
        cache.close()
```

**Manual cleanup fixtures** (when `yield` cleanup isn't enough):
```python
@pytest.fixture
def temp_cache_signing_enabled():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        config = CacheConfig(cache_dir=str(temp_dir / "cache"), ...)
        cache = UnifiedCache(config)
        yield cache, temp_dir
        cache.close()
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
```

## Mocking / Patching

**Framework:** `unittest.mock` (stdlib) — `Mock`, `patch`, `patch.object`

**Common mocking patterns:**

1. **Fault injection** via `patch.object` (in `tests/test_fault_injection.py`):
```python
with patch.object(
    cache.metadata_backend,
    "put_entry",
    side_effect=RuntimeError("Simulated metadata write failure"),
):
    with pytest.raises(RuntimeError, match="Simulated metadata write failure"):
        cache.put(data, test_key="orphan_test")
```

2. **AWS S3 mocking** via `moto` (in `tests/test_s3_blob_backend.py`):
```python
@pytest.fixture
def s3_backend(aws_credentials, s3_bucket):
    from cacheness.storage.backends.s3_backend import S3BlobBackend
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=s3_bucket)
        yield S3BlobBackend(bucket=s3_bucket, ...)
```

3. **Module-level patching** for optional dependencies:
```python
with patch("cacheness.handlers.BLOSC2_AVAILABLE", False):
    # Test fallback behavior
```

## Test Markers and Parametrization

**Registered markers** (in `pyproject.toml`):
```python
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "optional_deps: tests that require optional dependencies",
    "xdist_group: group tests to run on the same xdist worker",
]
```

**Strict marker enforcement:** `--strict-markers` is in default addopts — unregistered markers cause errors.

**Module-level markers:**
```python
# Skip entire module if dependency missing
pytestmark = [
    pytest.mark.skipif(not MOTO_AVAILABLE, reason="moto not installed"),
    pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not installed"),
]

# Group all tests in module for same xdist worker
pytestmark = pytest.mark.xdist_group("docker")
```

**skipif pattern for optional deps:**
```python
@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not available")
def test_pandas_dataframe_handler(self):
    ...

@pytest.mark.skipif(not _has_polars(), reason="Polars Series tests require polars")
def test_polars_series_handler(self):
    ...
```

**importorskip pattern:**
```python
pd = pytest.importorskip("pandas")
```

**Parametrization** (used in `tests/test_update_operations.py`, `tests/test_namespace_integration.py`, `tests/test_schema_versioning.py`):
```python
@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_feature_across_backends(self, backend, tmp_path):
    ...

@pytest.mark.parametrize(
    "input_data, expected",
    [
        ({"key": "value"}, True),
        (42, True),
        (None, False),
    ],
)
def test_handler_variations(self, input_data, expected):
    ...
```

## Parallel Test Execution (xdist)

**Default config** (in `pyproject.toml`):
```ini
addopts = ["-n", "auto", "--dist", "loadgroup"]
```

**Worker distribution:** `loadgroup` — tests in the same `xdist_group` run on the same worker.

**Docker resource grouping:**
```python
pytestmark = pytest.mark.xdist_group("docker")
```
All tests in `tests/test_docker_integration.py` share the same worker to avoid concurrent Docker container access conflicts.

**Disable parallelism:** `uv run pytest ... -p no:xdist`

## Test Categories

**Unit Tests** (~majority of test suite):
- Test individual classes/functions in isolation
- Use `tmp_path`/`tempfile.TemporaryDirectory()` for filesystem isolation
- Examples: `tests/test_core.py`, `tests/test_handlers.py`, `tests/test_metadata.py`, `tests/test_size_utils.py`

**Integration Tests:**
- Test multi-component interaction (cache + backend + handlers)
- Examples: `tests/test_integration.py`, `tests/test_namespace_integration.py`, `tests/test_config_integration_example.py`

**Docker Integration Tests** (require `docker-compose up -d`):
- PostgreSQL backend operations
- S3/Garage blob storage
- Grouped via `@pytest.mark.xdist_group("docker")`
- Example: `tests/test_docker_integration.py`

**Property-Based Tests** (Hypothesis):
- Random input generation for invariant verification
- Handler round-trip, cache key determinism, metadata contract, compression round-trip
- Example: `tests/test_property_based.py`

**Stress Tests:**
- Concurrent access patterns (multi-threaded put/get/delete)
- Lock re-entrancy verification
- Example: `tests/test_concurrency_stress.py`, `tests/test_sqlite_concurrency.py`

**Fault Injection Tests:**
- Simulate I/O failures, mid-operation crashes, data corruption
- Use `unittest.mock.patch` to inject faults
- Example: `tests/test_fault_injection.py`

**Backend Parity Tests:**
- Verify SQLite and PostgreSQL backends behave identically
- Example: `tests/test_backend_parity.py`, `tests/test_backend_compatibility.py`

**Schema Versioning Tests:**
- Test schema migration across backend types
- Examples: `tests/test_schema_versioning.py`, `tests/test_sqlite_schema_versioning.py`, `tests/test_json_schema_versioning.py`, `tests/test_pg_schema_versioning.py`

## Test Class Organization

**Pattern:** Group related tests into classes with descriptive names:
```python
class TestCacheConfig:
    """Test CacheConfig class functionality."""

    def test_default_config(self):
        ...

    def test_custom_config(self):
        ...


class TestCacheness:
    """Test cacheness class functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        ...

    def test_put_get_roundtrip(self, temp_cache_dir):
        ...
```

**Fixtures scoped to class:** Define fixtures as methods inside the class using `@pytest.fixture`.

**Some files use module-level functions** instead of classes (e.g., `tests/test_docker_integration.py`):
```python
def test_postgresql_basic_operations(cacheness_cache_from_yaml):
    ...
```

## Coverage

**Configuration** (in `pyproject.toml`):
```ini
[tool.coverage.run]
source = ["cacheness"]
omit = ["*/tests/*", "*/test_*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
```

**Run coverage:**
```bash
uv run pytest tests/ --cov=cacheness --cov-report=html
```

**No enforced minimum** — coverage is informational, not gating.

## Import Pattern for Tests

**Critical gotcha:** `from cacheness import UnifiedCache` does NOT work for the `UnifiedCache` class directly. It is exported as `cacheness`:
```python
# Correct
from cacheness import CacheConfig, cacheness
from cacheness.core import UnifiedCache

# Wrong
from cacheness import UnifiedCache  # This does NOT work
```

**Standard test imports:**
```python
import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock

from cacheness import CacheConfig, cacheness
from cacheness.core import UnifiedCache
from cacheness.config import CacheStorageConfig, CacheMetadataConfig, ...
from cacheness.metadata import SqliteBackend, JsonBackend, create_metadata_backend
from cacheness.storage import BlobStore
```

## Known Test Quirks

**TensorFlow tests hang on Windows:**
- `tests/test_tensorflow_handler.py` must always be excluded on Windows
- Add `--ignore=tests/test_tensorflow_handler.py` to all test commands

**Optional dependency skipping:**
- Tests for Pandas, Polars, boto3, moto, psycopg gracefully skip when dependencies are missing
- ~102 tests are typically skipped in a standard run (optional deps not installed)

**`get()` is destructive on errors:**
- Auto-deletes entries that fail to load (except transient IO errors)
- Tests must account for this behavior — a failed `get()` removes the entry

**Logging suppression:**
- Default test log level is `WARNING` (set in `pyproject.toml`)
- Deprecation and PendingDeprecation warnings are filtered out

**Warning filters:**
```ini
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
]
```

## Tiered Testing Strategy

During development, use tiered testing to minimize feedback loops:

| Tier | When | What to run | Time |
|------|------|-------------|------|
| **Tier 1** | After each code change | Tests directly exercising modified code | ~5-15s |
| **Tier 2** | After all planned changes | Add regression-risk tests | ~30-60s |
| **Full suite** | Once before push | All tests | ~48s parallel |

---

*Testing analysis: 2026-04-02*
