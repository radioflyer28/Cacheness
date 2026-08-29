# Testing Patterns

**Analysis Date:** 2026-08-29

## Test Framework

**Runner:**
- `pytest` `>=8.4.1`, declared in the `dev` dependency group in `pyproject.toml`.
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`; it enables `-ra`, quiet output, strict markers, `tests/` as `testpaths`, and `test_*.py`/`Test*`/`test_*` discovery patterns.

**Assertion Library:**
- Use native Python `assert` statements and pytest helpers (`pytest.raises`, `pytest.skip`, `pytest.importorskip`, `caplog`). Array/dataframe tests additionally use `numpy.testing` and pandas comparison helpers, for example in `tests/test_core.py` and `tests/test_pandas_compatibility.py`.

**Run Commands:**
```bash
uv run pytest                         # Run the complete configured suite
uv run pytest tests/test_core.py -v   # Run one module verbosely
uv run pytest tests/ -m "not slow"    # Exclude tests marked slow
uv run pytest tests/ -k "test_cache" # Select tests by name expression
uv run pytest tests/ --cov=cacheness --cov-report=html  # Generate HTML coverage
```

No watch-mode command, tox/nox configuration, or separate test runner configuration was detected. The documented cross-platform commands in `docs/CROSS_PLATFORM_GUIDE.md` and `docs/WINDOWS_COMPATIBILITY.md` use the same `uv run pytest` entry point.

## Test File Organization

**Location:**
- Tests are separate from implementation in a flat `tests/` directory; there is no `tests/conftest.py` and no shared fixture module detected.
- Test modules are organized by implementation area (`tests/test_core.py`, `tests/test_metadata.py`, `tests/test_handlers.py`) and by behavior/feature (`tests/test_cache_integrity.py`, `tests/test_directory_sharding.py`, `tests/test_sqlite_concurrency.py`).

**Naming:**
- Files use `test_<subject>.py`; test classes use `Test<Subject>`; test methods and standalone tests use `test_<behavior>`.
- Test names describe the expected behavior, commonly with `test_<action>_<condition>` or `test_<action>_<result>`, such as `test_read_nonexistent_raises` in `tests/test_s3_blob_backend.py`.

**Structure:**
```text
tests/
├── test_core.py
├── test_<feature>.py
├── test_<backend>_backend.py
└── test_<integration_or_edge_case>.py
```

## Test Structure

**Suite Organization:**
```python
class TestCacheIntegrity:
    """Test cache integrity behavior."""

    @pytest.fixture
    def temp_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = cacheness(CacheConfig(cache_dir=temp_dir))
            yield cache
            cache.close()

    def test_file_hash_stored_in_metadata(self, temp_cache):
        temp_cache.put({"value": 1}, key="example")
        cache_key = temp_cache._create_cache_key({"key": "example"})
        entry = temp_cache.metadata_backend.get_entry(cache_key)
        assert "file_hash" in entry
```

The concrete fixture/lifecycle pattern appears in `tests/test_cache_integrity.py` and `tests/test_sqlite_concurrency.py`; use the existing module’s API and cleanup conventions when adding tests.

**Patterns:**
- Group related behavior in a `Test...` class with a short class docstring, then give each test a one-line docstring. Larger files use banner comments to separate fixture, core operation, error, and integration sections.
- Use function-scoped fixtures by default. Yield temporary paths or cache instances and close cache/backend resources after the yield; use `tempfile.TemporaryDirectory()` or `tmp_path` rather than repository paths.
- Use `@pytest.fixture(autouse=True)` for global registry isolation where every test must reset state. `tests/test_custom_metadata.py`, `tests/test_metadata_backend_registry.py`, and `tests/test_blob_backend_registry.py` explicitly save/reset/restore registries.
- Assert both result values and side effects such as cache statistics, metadata fields, persisted files, backend registration, and log output. End-to-end flows in `tests/test_integration.py` exercise put/get/stats/list behavior together.

## Mocking

**Framework:**
- Use `unittest.mock` (`patch`, `patch.object`, `Mock`, and `MagicMock`) from the standard library. No pytest-mock plugin was detected.
- Use `moto.mock_aws` for S3 behavior in `tests/test_s3_blob_backend.py`, with test credentials set by a fixture; do not call real cloud services.

**Patterns:**
```python
with patch("cacheness.handlers._lazy_import_tensorflow") as mock_import:
    mock_import.return_value = (None, False)
    handler = TensorFlowTensorHandler()
    assert not handler.can_handle([1, 2, 3])
```

Patch the symbol where the system under test looks it up, as in `tests/test_tensorflow_handler.py`, `tests/test_core.py`, and `tests/test_file_hashing.py`. Use `monkeypatch.setattr` when changing process/import state for one test, as in `tests/test_custom_metadata.py`.

**What to Mock:**
- Mock unavailable optional imports, clock/time, filesystem permission failures, process pools, and external SDK clients. Examples include `tests/test_postgresql_backend.py`, `tests/test_sql_cache.py`, `tests/test_file_hashing.py`, and `tests/test_s3_blob_backend.py`.
- Build small fake interface implementations for registry tests rather than mocking every method: `MockMetadataBackend` in `tests/test_metadata_backend_registry.py` and custom handlers/backends in `tests/test_handler_registration.py` and `tests/test_blob_backend_registry.py`.

**What NOT to Mock:**
- Keep serialization, cache-key generation, local filesystem persistence, SQLite behavior, and handler round trips as real operations when testing those contracts. `tests/test_cache_key_consistency.py`, `tests/test_serialization.py`, `tests/test_integration.py`, and `tests/test_sqlite_concurrency.py` follow this approach.
- Isolate external PostgreSQL/S3/TensorFlow dependencies with skips or service mocks; do not make ordinary unit tests require those services.

## Fixtures and Factories

**Test Data:**
```python
@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def valid_config():
    """Provide a valid configuration instance."""
    return CacheConfig()
```

This fixture style is used in `tests/test_config_validation.py`, `tests/test_postgresql_backend.py`, and `tests/test_handlers.py`. Inline dicts, NumPy arrays, pandas frames, and small fake classes are preferred over a global factory layer.

**Location:**
- Keep fixtures beside the tests that need them. No shared `conftest.py`, fixture package, or common factory directory was detected.
- Use `request.addfinalizer` for cleanup that needs custom retry/garbage-collection behavior, as in `tests/test_custom_metadata.py`; use a fixture `yield` for ordinary resource cleanup.
- Preserve registry state around registration tests and close every cache/backend created by a fixture. The registry isolation pattern is defined in `tests/test_metadata_backend_registry.py` and `tests/test_blob_backend_registry.py`.

## Coverage

**Requirements:**
- No minimum coverage threshold (`fail_under`) is configured or enforced.
- Coverage targets the `cacheness` package and omits `*/tests/*` and `*/test_*` in `pyproject.toml`. Reports exclude `pragma: no cover`, `__repr__`, `AssertionError`, and `NotImplementedError` lines.

**View Coverage:**
```bash
uv run pytest tests/ --cov=cacheness --cov-report=html
```

## Test Types

**Unit Tests:**
- Most tests are unit-style class suites around configuration, serialization, hashing, error handling, interfaces, handlers, decorators, and registry operations (`tests/test_config_validation.py`, `tests/test_serialization.py`, `tests/test_error_handling.py`, `tests/test_interfaces.py`).
- Use real small in-memory objects and temporary files for contract-level behavior, with mocks only at external/process boundaries.

**Integration Tests:**
- End-to-end cache persistence and backend combinations are covered in `tests/test_integration.py`, `tests/test_cross_system_compatibility.py`, `tests/test_backend_compatibility.py`, and `tests/test_sql_cache.py`.
- SQLite concurrency is exercised with `ThreadPoolExecutor` in both `tests/test_sqlite_concurrency.py` and `tests/test_sqlite_concurrency_temp.py`; S3 integration is simulated with moto in `tests/test_s3_blob_backend.py`.
- PostgreSQL tests use a `CACHENESS_TEST_POSTGRES_URL` environment variable and skip when the driver/service is unavailable (`tests/test_postgresql_backend.py`).

**E2E Tests:**
- No browser or user-interface E2E framework is used. The closest E2E coverage is Python-level cache workflows in `tests/test_integration.py` and documentation examples in `tests/test_sql_cache_documentation.py`.

## Common Patterns

**Async Testing:**
```python
# No async test functions or pytest-asyncio configuration were detected.
# Keep new tests synchronous unless the implementation introduces an async API.
```

**Error Testing:**
```python
with pytest.raises(ValueError, match="already registered"):
    registry.register_handler(duplicate_handler)
```

Use `pytest.raises` around the smallest operation that should fail and assert context/details from `exc_info` when the error contract exposes them. This pattern is common in `tests/test_handler_registration.py`, `tests/test_config_validation.py`, and `tests/test_error_handling.py`.

**Optional Dependencies and Skips:**
- Detect optional packages with a guarded import or `pytest.importorskip`, expose a module flag, and apply `@pytest.mark.skipif(..., reason=...)` to the affected test/class. Examples include `tests/test_postgresql_backend.py`, `tests/test_s3_blob_backend.py`, `tests/test_handlers.py`, and `tests/test_tensorflow_handler.py`.
- The configured markers are `slow`, `integration`, and `optional_deps`, but existing tests predominantly use `skipif` and do not consistently apply the custom markers.

**Observed Baseline:**
- On 2026-08-29, `uv run pytest -q` collected 777 tests: 749 passed, 26 skipped, and 2 failed in `tests/test_config_validation.py::TestYamlConfig` because relative YAML cache paths are normalized to absolute paths before comparison.
- Collection emits a `PytestCollectionWarning` for `TestDataClassForConsistency` in `tests/test_cache_key_consistency.py` because the dataclass has an `__init__` constructor. TensorFlow tests include explicit skips because importing TensorFlow can freeze the process (`tests/test_tensorflow_handler.py`).

---

*Testing analysis: 2026-08-29*
