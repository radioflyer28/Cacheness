# Deferred Items

## 32-01

- `uv run ty check src/cacheness/storage/blob_store.py tests/test_blob_store.py` reports two pre-existing `invalid-argument-type` diagnostics in `src/cacheness/storage/blob_store.py`:
  - `CacheConfig(cache_dir=self.cache_dir)` passes `Path` where `str | None` is declared.
  - `SqliteBackend(self.cache_dir / "cache_metadata.db")` passes `Path` where `str` is declared.
- The plan-level pytest command with repository addopts collected unrelated tests and surfaced existing failures outside `tests/test_blob_store.py`; these were later resolved by Phase 32 checkpoint fix commit `4d6e0d1`:
  - `tests/test_dunder_methods.py::TestDunderMethods::test_contains_expired_key`
  - `tests/test_fault_injection.py::TestOrphanedBlobOnPutCrash::test_failed_same_key_overwrite_preserves_previous_blob`
  - `tests/test_decorators.py::TestCacheIfDecorator::test_cache_if_supports_ttl_parameter`

## 32-02

- `uv run ty check src/cacheness/storage/blob_store.py tests/test_blob_store.py` still reports the same two pre-existing `invalid-argument-type` diagnostics in `src/cacheness/storage/blob_store.py` constructor setup. They are unrelated to the POL-02 `_sanitize_key()` change.
- Including `tests/test_blob_namespace.py` in the full plan ty command also reports pre-existing `possibly-missing-attribute` warnings for tests that access filesystem-backend-only `_namespace` and `base_dir` attributes through the generic `BlobBackend` type.
- The initial RED pytest command with repository addopts collected unrelated tests instead of only the requested node IDs. The POL-02 regressions were therefore also verified with `-o addopts=''` for scoped evidence.

## 32-05

- `uv run ty check src/cacheness/metadata/sqlite_backend.py tests/test_sqlite_schema_versioning.py tests/test_metadata.py` reports pre-existing SQLAlchemy dynamic-model diagnostics in `src/cacheness/metadata/sqlite_backend.py` plus a pre-existing `pytest.fail(...)` typing false positive in `tests/test_metadata.py`. These are unrelated to the SQLite PRAGMA lifecycle change.
- Running the plan pytest command without `-o addopts=''` collected the broader suite and surfaced existing failures in `tests/test_decorators.py::TestCacheIfDecorator::test_cache_if_supports_ttl_parameter`, `tests/test_dunder_methods.py::TestDunderMethods::test_contains_expired_key`, and `tests/test_fault_injection.py::TestOrphanedBlobOnPutCrash::test_failed_same_key_overwrite_preserves_previous_blob`. These failures were later resolved by Phase 32 checkpoint fix commit `4d6e0d1`; the POL-06 verification passed with the scoped command documented in the summary.

## 32-final-checkpoint

- `4d6e0d1` resolved the full-suite checkpoint failures discovered during 32-08:
  - decorator/cache-if TTL writes now persist the decorator TTL before stored `expires_at` is computed.
  - fractional per-entry TTLs are no longer truncated to zero.
  - explicit read-time TTL overrides still work without weakening stored-expiry precedence for normal reads.
  - failed same-key overwrites restore the previous committed blob in both cache and storage modes.
- Final checkpoint passed: `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` -> 1857 passed, 125 skipped, 27 warnings.
