# Testing Patterns

**Analysis Date:** 2026-04-07

## Test Framework

**Runner:**
- pytest >= 8.4.1
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`

**Parallel Execution:**
- `pytest-xdist` >= 3.8.0 — enabled by default via `-n auto --dist loadgroup`
- Tests sharing Docker resources grouped via `@pytest.mark.xdist_group("docker")`

**Assertion Library:**
- pytest native asserts (no third-party assertion library)

**JSON Report:**
- `pytest-json-report` >= 1.5.0 — writes `.test-results.json` (gitignored)
- Parse results: `(Get-Content .test-results.json | ConvertFrom-Json).summary`
- Configured in `pyproject.toml` addopts: `--json-report --json-report-file .test-results.json --json-report-omit log keywords collectors`

**Run Commands:**
```bash
# Full suite (parallel, ~48s)
uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py

# Sequential (disable xdist, ~237s — useful for debugging)
uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py -p no:xdist

# Targeted (single file)
uv run pytest tests/test_core.py -x -q --tb=short

# Parse JSON results after run
(Get-Content .test-results.json | ConvertFrom-Json).summary
```

**IMPORTANT:** Never pipe pytest output through `Select-String` or grep — use JSON report instead.

## Test Baseline

**Current:** 1773 passed, 122 skipped, 0 failures
**Windows:** Always add `--ignore=tests/test_tensorflow_handler.py` — TF tests hang

## Test File Organization

**Location:** All tests in top-level `tests/` directory (not co-located with source).

**Naming:** `test_<module_or_feature>.py` — mirrors source module names or feature areas.

**Source → test file mapping:**

| Source file | Primary test files |
|---|---|
| `src/cacheness/core.py` | `tests/test_core.py`, `tests/test_storage_mode.py` |
| `src/cacheness/storage/blob_store.py` | `tests/test_blob_store.py`, `tests/test_blob_namespace.py` |
| `src/cacheness/storage/blob_backends.py` | `tests/test_blob_store.py`, `tests/test_blob_namespace.py`, `tests/test_s3_blob_backend.py` |
| `src/cacheness/handlers/*.py` | `tests/test_handlers.py` |
| `src/cacheness/metadata/*.py` | `tests/test_metadata.py`, `tests/test_sqlite_schema_versioning.py` |
| `src/cacheness/security.py` | `tests/test_security.py` (via `test_namespace_signing.py`, `test_cache_signing.py`) |
| `src/cacheness/config.py` | `tests/test_config_validation.py`, `tests/test_core.py` |
| `src/cacheness/encryption.py` | `tests/test_encryption_at_rest.py` |
| `src/cacheness/write_intent.py` | `tests/test_write_intent.py` |
| `src/cacheness/decorators.py` | `tests/test_decorators.py` |

**Cross-cutting test files:**
- `tests/test_cache_integrity.py` — end-to-end integrity verification
- `tests/test_backend_parity.py` — cross-backend behavioral equivalence
- `tests/test_fault_injection.py` — I/O failure simulation
- `tests/test_concurrency_stress.py` — multi-threaded stress tests
- `tests/test_property_based.py` — Hypothesis property-based tests
- `tests/test_cross_phase_integration.py` — cross-feature integration

## Test Structure

**Suite Organization:**
```python
# tests/test_core.py
class TestCacheConfig:
    """Test CacheConfig class functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.storage.cache_dir == "./cache"
        assert config.metadata.default_ttl_seconds == 86400

    def test_custom_config(self):
        """Test custom configuration values."""
        ...
```

Tests are grouped into classes by feature/component. Each test method tests one behavior.

**Import pattern:**
```python
from cacheness import CacheConfig, cacheness
from cacheness.core import UnifiedCache as cacheness  # in tests needing UnifiedCache directly
from cacheness.config import CacheStorageConfig, CacheMetadataConfig, SecurityConfig
```

**IMPORTANT:** `from cacheness import UnifiedCache` does NOT work. The public export is `cacheness` (an alias for `UnifiedCache`). Use `from cacheness.core import UnifiedCache` when the class name is needed.

## Fixtures

**Shared fixtures** in `tests/conftest.py`:
```python
# Session-scoped — check Docker service availability
@pytest.fixture(scope="session")
def postgres_available() -> bool:
    """Check if PostgreSQL is available (with a short timeout)."""
    try:
        with psycopg.connect(..., connect_timeout=3):
            return True
    except Exception:
        return False

@pytest.fixture(scope="session")
def postgres_engine(postgres_available):
    """Create SQLAlchemy engine for PostgreSQL."""
    if not postgres_available:
        pytest.skip("PostgreSQL not available")
    engine = create_engine(get_postgres_url())
    yield engine
    engine.dispose()

# Function-scoped — per-test isolation
@pytest.fixture
def postgres_clean_db(postgres_connection):
    """Provide a clean PostgreSQL database for a test."""
    postgres_connection.execute(text("CREATE SCHEMA IF NOT EXISTS cache"))
    postgres_connection.commit()
    yield postgres_connection
    postgres_connection.execute(text("DROP SCHEMA IF EXISTS cache CASCADE"))
    postgres_connection.commit()
```

**Per-test cache factory** (common pattern across test files):
```python
def _make_cache(tmp_path, backend="json"):
    """Create a cacheness instance for testing."""
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_path)),
        metadata=CacheMetadataConfig(metadata_backend=backend),
        compression=CompressionConfig(use_blosc2_arrays=False),
    )
    return cacheness(config)
```

**`tmp_path` fixture:** pytest built-in, used for all cache directories. Ensures test isolation.

**Encrypted cache factory:**
```python
def _make_encrypted_cache(tmp_path, **security_overrides):
    """Create a UnifiedCache with encryption and signing enabled."""
    key_file = tmp_path / "cache_signing_key.bin"
    if not key_file.exists():
        key_file.write_bytes(secrets.token_bytes(32))
    defaults = {
        "enable_entry_signing": True,
        "enable_content_encryption": True,
        "encryption_key_file": "cache_signing_key.bin",
        "allow_unsigned_entries": True,
        "delete_invalid_signatures": False,
    }
    defaults.update(security_overrides)
    security = SecurityConfig(**defaults)
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_path)),
        metadata=CacheMetadataConfig(metadata_backend="json"),
        compression=CompressionConfig(use_blosc2_arrays=False),
        security=security,
    )
    return cacheness(config)
```

## Mocking

**Framework:** `unittest.mock` (standard library)

**Patterns:**
```python
# Patch metadata backend to simulate crash
with patch.object(
    cache.metadata_backend,
    "put_entry",
    side_effect=RuntimeError("Simulated metadata write failure"),
):
    with pytest.raises(RuntimeError, match="Simulated metadata write failure"):
        cache.put(data, test_key="orphan_test")

# Patch OS-level I/O
with patch("builtins.open", side_effect=OSError(errno.ENOSPC, "No space left")):
    ...
```

**What to mock:**
- Metadata backend methods (`put_entry`, `get_entry`) for fault injection
- OS/filesystem calls for I/O failure simulation
- External services (S3 via `moto`)

**What NOT to mock:**
- Handler logic (test with real data through the full put/get pipeline)
- Configuration validation (test with real `CacheConfig` instances)
- Compression/serialization (test roundtrip with actual data)

## Backend Parametrization

**Cross-backend parity tests** in `tests/test_backend_parity.py`:
```python
@pytest.mark.parametrize("backend", ["json", "sqlite", "postgresql"])
class TestEncryptionBackendParity_BlobStore:
    def test_encrypt_decrypt_roundtrip(self, tmp_path, backend):
        if backend == "postgresql":
            pg_url = _get_pg_url()
            if not pg_url:
                pytest.skip("PostgreSQL not available")
        cache = _make_encrypted_cache_for_backend(tmp_path, backend)
        ...
```

**Pattern:** Parametrize on `backend` string, skip PostgreSQL when Docker unavailable.

## Docker Integration

**External services:** PostgreSQL and S3 (Garage) via `docker-compose.yml`

**xdist grouping:** Tests that need Docker are grouped to run on the same worker:
```python
# Module-level marker for all tests in file
pytestmark = pytest.mark.xdist_group("docker")

# Or per-class
@pytest.mark.xdist_group("docker")
class TestPostgresqlSchemaV3ToV4:
    ...
```

**Availability check:** Session-scoped fixtures attempt connection with short timeouts, `pytest.skip()` on failure.

**Files using Docker:**
- `tests/test_docker_integration.py` — full PostgreSQL + S3 integration
- `tests/test_backend_parity.py` — cross-backend encryption tests
- `tests/test_pg_schema_versioning.py` — PostgreSQL schema migrations
- `tests/test_config_integration_example.py` — config with real backends

## Optional Dependency Guards

**Module-level `importorskip`:**
```python
# tests/test_encryption_at_rest.py
cryptography = pytest.importorskip("cryptography")

# Imports AFTER importorskip use # noqa: E402
from cacheness.config import CacheConfig, SecurityConfig  # noqa: E402
from cacheness.encryption import encrypt_blob, decrypt_blob  # noqa: E402
```

**`# noqa: E402` placement:** When using `pytest.importorskip()` at module level, ALL subsequent imports get `# noqa: E402`. Collapse to single-line imports — multi-line `# noqa: E402` on closing paren does NOT work.

**Per-test `skipif`:**
```python
@pytest.mark.skipif(not _has_pandas(), reason="Pandas not available")
def test_pandas_dataframe_caching(self):
    ...
```

**Per-test `importorskip`:**
```python
def test_entry_list_to_dataframe(self):
    pd = pytest.importorskip("pandas")
    ...
```

**Availability helper functions:**
```python
def _has_pandas():
    import importlib.util
    return importlib.util.find_spec("pandas") is not None
```

## Property-Based Tests (Hypothesis)

Located in `tests/test_property_based.py`:

```python
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

# Custom strategies for JSON-safe values
json_values = st.one_of(
    st.text(min_size=0, max_size=100),
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
)

@given(data=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_numpy_array_roundtrip(self, data, tmp_path):
    """Any valid float array survives put → get."""
    arr = np.array(data)
    ...
```

**Invariants tested:**
1. Handler round-trip: `put(obj) → get()` returns equivalent obj
2. Cache key determinism: same input → same key, always
3. Metadata backend contract: put → get → delete consistency
4. Compression round-trip: compress → decompress preserves data

## Fault Injection

Located in `tests/test_fault_injection.py`:

**Patterns tested:**
1. Orphaned blob on `put()` crash (metadata write failure after blob write)
2. `get()` auto-deletion on transient I/O errors
3. `get()` auto-deletion on handler exception (deserialization failure)
4. JSON backend corruption recovery
5. Disk full during blob write (patched `OSError(errno.ENOSPC)`)
6. TOCTOU race in `get()` (file exists → file gone on open)

```python
class TestOrphanedBlobOnPutCrash:
    def test_blob_cleaned_up_on_metadata_failure(self, tmp_path):
        cache = _make_cache(tmp_path)
        data = {"key": "value"}
        with patch.object(
            cache.metadata_backend, "put_entry",
            side_effect=RuntimeError("Simulated metadata write failure"),
        ):
            with pytest.raises(RuntimeError):
                cache.put(data, test_key="orphan_test")
        # Verify no orphaned blob files remain
        blob_files = list(tmp_path.rglob("*.pkl*"))
        assert blob_files == []
```

## Concurrency / Thread Safety Tests

Located in `tests/test_concurrency_stress.py` and `tests/test_thread_safety.py`:

```python
class TestConcurrentPutSameKey:
    def test_last_writer_wins_no_corruption(self, stress_cache):
        errors = []
        num_threads = 8

        def writer(thread_id):
            try:
                data = {"thread": thread_id, "value": thread_id * 100}
                cache.put(data, key="shared")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=writer, args=(i,))
                   for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors
        result = cache.get(key="shared")
        assert isinstance(result, dict)
```

**Thread safety fixtures:**
```python
@pytest.fixture
def stress_cache():
    with tempfile.TemporaryDirectory() as temp_dir:
        config = CacheConfig(cache_dir=temp_dir, metadata_backend="sqlite")
        cache = UnifiedCache(config=config)
        yield cache
        cache.close()
```

## Tiered Testing Strategy

During development, use tiered testing to minimize feedback time:

| Tier | When | What to run | Time |
|---|---|---|---|
| **Tier 1** | After each code change | Tests directly exercising modified code | ~5-15s |
| **Tier 2** | After all planned changes | Add regression-risk tests | ~30-60s |
| **Full suite** | Once before push | All tests | ~48s parallel |

**Example workflow:**
```bash
# Tier 1 — editing core.py and blob_store.py
uv run pytest tests/test_core.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py

# Tier 2 — add cross-cutting tests
uv run pytest tests/test_core.py tests/test_blob_store.py tests/test_cache_integrity.py tests/test_update_operations.py -x -q --ignore=tests/test_tensorflow_handler.py

# Full suite — before push
uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py
```

## pytest Configuration

From `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "6.0"
addopts = [
    "-q",
    "--strict-markers",
    "-n", "auto",
    "--dist", "loadgroup",
    "--tb", "short",
    "--json-report",
    "--json-report-file", ".test-results.json",
    "--json-report-omit", "log", "keywords", "collectors",
]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "optional_deps: tests that require optional dependencies",
    "xdist_group: group tests to run on the same xdist worker",
]
log_cli = true
log_cli_level = "WARNING"
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
]
```

**Key settings:**
- `--strict-markers` — undefined markers are errors
- `-n auto` — auto-detect CPU count for parallel
- `--dist loadgroup` — respect `xdist_group` markers
- `--tb short` — concise tracebacks

## Test Types

**Unit Tests:**
- Handler tests (`test_handlers.py`) — isolated handler put/get with real data
- Config tests (`test_config_validation.py`) — dataclass validation
- Schema tests (`test_sqlite_schema_versioning.py`, `test_json_schema_versioning.py`) — migration logic

**Integration Tests:**
- Core tests (`test_core.py`) — full put/get/delete through UnifiedCache
- Storage mode tests (`test_storage_mode.py`) — no-eviction mode
- Decorator tests (`test_decorators.py`) — `@cached` / `@cache_if` end-to-end

**Cross-Backend Tests:**
- `test_backend_parity.py` — same operations across JSON/SQLite/PostgreSQL
- `test_backend_compatibility.py` — data portability between backends

**Stress/Resilience Tests:**
- `test_concurrency_stress.py` — multi-threaded race conditions
- `test_fault_injection.py` — simulated I/O failures
- `test_thread_safety.py` — concurrent put/get smoke tests

**Property-Based Tests:**
- `test_property_based.py` — Hypothesis-generated random data roundtrips

## Common Test Patterns

**Cache factory function per test file:**
```python
def _make_cache(tmp_path, backend="json"):
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_path)),
        metadata=CacheMetadataConfig(metadata_backend=backend),
        compression=CompressionConfig(use_blosc2_arrays=False),
    )
    return cacheness(config)
```

**`use_blosc2_arrays=False`** in test configs — avoids optional-dependency failures and simplifies blob format assertions.

**Error assertion:**
```python
with pytest.raises(ValueError, match="must be positive"):
    CacheBlobConfig(max_inline_size=-1)

with pytest.raises(CacheConfigurationError):
    SecurityConfig(enable_content_encryption=True, enable_entry_signing=False)
```

**Round-trip verification:**
```python
cache.put(data, test_key="roundtrip")
result = cache.get(test_key="roundtrip")
assert result == data  # or np.array_equal for arrays
```

**Blob cleanup assertion:**
```python
blob_files = list(tmp_path.rglob("*.pkl*"))
assert blob_files == [], f"Orphaned blob files: {blob_files}"
```

---

*Testing analysis: 2026-04-07*
