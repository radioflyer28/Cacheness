# Deferred Items

## 32-01

- `uv run ty check src/cacheness/storage/blob_store.py tests/test_blob_store.py` reports two pre-existing `invalid-argument-type` diagnostics in `src/cacheness/storage/blob_store.py`:
  - `CacheConfig(cache_dir=self.cache_dir)` passes `Path` where `str | None` is declared.
  - `SqliteBackend(self.cache_dir / "cache_metadata.db")` passes `Path` where `str` is declared.
- The plan-level pytest command with repository addopts collected unrelated tests and surfaced existing failures outside `tests/test_blob_store.py`:
  - `tests/test_dunder_methods.py::TestDunderMethods::test_contains_expired_key`
  - `tests/test_fault_injection.py::TestOrphanedBlobOnPutCrash::test_failed_same_key_overwrite_preserves_previous_blob`
  - `tests/test_decorators.py::TestCacheIfDecorator::test_cache_if_supports_ttl_parameter`
