# Deferred Items

## 32-01

- `uv run ty check src/cacheness/storage/blob_store.py tests/test_blob_store.py` reports two pre-existing `invalid-argument-type` diagnostics in `src/cacheness/storage/blob_store.py`:
  - `CacheConfig(cache_dir=self.cache_dir)` passes `Path` where `str | None` is declared.
  - `SqliteBackend(self.cache_dir / "cache_metadata.db")` passes `Path` where `str` is declared.
- The plan-level pytest command with repository addopts collected unrelated tests and surfaced existing failures outside `tests/test_blob_store.py`:
  - `tests/test_dunder_methods.py::TestDunderMethods::test_contains_expired_key`
  - `tests/test_fault_injection.py::TestOrphanedBlobOnPutCrash::test_failed_same_key_overwrite_preserves_previous_blob`
  - `tests/test_decorators.py::TestCacheIfDecorator::test_cache_if_supports_ttl_parameter`

## 32-02

- `uv run ty check src/cacheness/storage/blob_store.py tests/test_blob_store.py` still reports the same two pre-existing `invalid-argument-type` diagnostics in `src/cacheness/storage/blob_store.py` constructor setup. They are unrelated to the POL-02 `_sanitize_key()` change.
- Including `tests/test_blob_namespace.py` in the full plan ty command also reports pre-existing `possibly-missing-attribute` warnings for tests that access filesystem-backend-only `_namespace` and `base_dir` attributes through the generic `BlobBackend` type.
- The initial RED pytest command with repository addopts collected unrelated tests instead of only the requested node IDs. The POL-02 regressions were therefore also verified with `-o addopts=''` for scoped evidence.
