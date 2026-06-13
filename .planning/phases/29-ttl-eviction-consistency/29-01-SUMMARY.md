---
phase: 29-ttl-eviction-consistency
plan: 01
subsystem: cache-metadata
tags: [ttl, expiry, cleanup, json, sqlite, postgresql, storage-mode]

requires:
  - phase: 29-ttl-eviction-consistency
    provides: Phase 29 context, research, patterns, and TTL-01 execution plan
provides:
  - Stored expires_at precedence for cache-mode reads
  - Stored expires_at precedence for public and backend cleanup paths
  - JSON, SQLite, PostgreSQL, and storage-mode TTL-01 regressions
affects: [phase-29, ttl-01, metadata-backends, storage-mode]

tech-stack:
  added: []
  patterns:
    - Stored expires_at is authoritative when present
    - Fallback TTL applies only to entries without stored expires_at

key-files:
  created:
    - .planning/phases/29-ttl-eviction-consistency/29-01-SUMMARY.md
  modified:
    - src/cacheness/core.py
    - src/cacheness/metadata/json_backend.py
    - src/cacheness/metadata/sqlite_backend.py
    - src/cacheness/storage/backends/postgresql_backend.py
    - src/cacheness/compress_pickle.py
    - src/cacheness/handlers/_compat.py
    - src/cacheness/metadata/_compat.py
    - tests/test_core.py
    - tests/test_metadata.py
    - tests/test_backend_parity.py
    - tests/test_storage_mode.py
    - tests/test_postgresql_backend.py

key-decisions:
  - "Stored expires_at is checked before ttl_seconds=None or config fallback branches."
  - "Backend cleanup deletes non-null past expires_at rows even when fallback TTL is zero; entries without expires_at keep existing no-TTL no-op behavior."
  - "Storage-mode get remains outside _is_expired and returns data with past stored expires_at."

patterns-established:
  - "Timestamp parsing normalizes naive datetimes to UTC before comparison."
  - "Public blob cleanup and backend metadata cleanup use the same stored-expiry predicate family."

requirements-completed: [TTL-01]

duration: 38min
completed: 2026-06-13T21:45:03Z
---

# Phase 29 Plan 01: Stored TTL Expiry Summary

**Stored per-entry expires_at now controls cache-mode reads and JSON, SQLite, and PostgreSQL cleanup while storage-mode reads remain non-expiring.**

## Performance

- **Duration:** 38 min
- **Started:** 2026-06-13T21:07:09Z
- **Completed:** 2026-06-13T21:45:03Z
- **Tasks:** 2
- **Files modified:** 12

## Accomplishments

- Added verify-first TTL-01 regressions for core read expiry, public cleanup, JSON/SQLite parity, PostgreSQL cleanup, and storage-mode bypass.
- Implemented stored `expires_at` precedence in `UnifiedCache._is_expired()` before fallback TTL handling.
- Aligned public cleanup and JSON/SQLite/PostgreSQL backend cleanup predicates so stored `expires_at` and fallback `created_at` semantics agree.
- Kept storage mode protected: `get()` still returns through `_storage_mode_get()` before cache expiry logic.

## Task Commits

1. **Rule 3 blocker fix:** `8f67268` - `fix(29-01): unblock optional dependency imports`
2. **Task 1 RED tests:** `12b2da0` - `test(29-01): add stored expiry precedence regressions`
3. **Task 2 implementation:** `ae27801` - `feat(29-01): honor stored expires_at precedence`

## Files Created/Modified

- `src/cacheness/core.py` - Added UTC timestamp parser and stored-expiry precedence in `_is_expired()` and public `cleanup_expired()`.
- `src/cacheness/metadata/json_backend.py` - Updated cleanup to prefer stored `expires_at`, with fallback TTL only when absent.
- `src/cacheness/metadata/sqlite_backend.py` - Updated SQL cleanup predicate for stored expiry and fallback TTL.
- `src/cacheness/storage/backends/postgresql_backend.py` - Updated PostgreSQL cleanup predicate to match SQLite/JSON semantics.
- `src/cacheness/compress_pickle.py` - Rule 3 optional `blosc2` permission fallback.
- `src/cacheness/handlers/_compat.py` - Rule 3 optional `blosc2` permission fallback.
- `src/cacheness/metadata/_compat.py` - Rule 3 `_CORE_TABLES` fallback for reduced SQLAlchemy environments.
- `tests/test_core.py` - Core read/cleanup regressions and deterministic TTL test updates.
- `tests/test_metadata.py` - JSON/SQLite stored-expiry cleanup regressions.
- `tests/test_backend_parity.py` - SQLite cleanup parity regression for both precedence directions.
- `tests/test_storage_mode.py` - Storage-mode past-expiry read bypass regression.
- `tests/test_postgresql_backend.py` - Docker-skipping PostgreSQL stored-expiry cleanup regression.

## Verification

- RED: targeted pytest reached the new regression and failed as expected in `tests/test_backend_parity.py::TestBackendParity::test_cleanup_expired_parity`.
- PASS: `uv run --python 3.12 pytest --override-ini testpaths= tests/test_core.py tests/test_metadata.py tests/test_backend_parity.py tests/test_storage_mode.py tests/test_postgresql_backend.py -x -q --ignore=tests/test_tensorflow_handler.py -p no:cacheprovider --basetemp .tmp\pytest`
  - Result: 160 passed, 33 skipped.
- PASS: `uv run --python 3.12 ruff format ...`
- PASS: `uv run --python 3.12 ruff check --fix ...`
- PASS: `uv run --python 3.12 ruff check ...`
- BASELINE FAIL: `uv run --python 3.12 ty check ...`
  - Result: failed with pre-existing touched-file diagnostics, including JSON backend broad metadata typing, core `Path | str` path unions, pytest `skip/fail` typing in tests, optional `polars`/`tensorflow` imports, and existing mock `blosc2` assignment typing.

## Decisions Made

- Preserved the plan's storage-mode guard by leaving `UnifiedCache.get()` ordering unchanged.
- Treated invalid stored `expires_at` values as expired in core/public cleanup paths, matching the defensive parsing threat mitigation.
- Used `--override-ini testpaths=` and workspace-local temp/cache environment variables for pytest because the sandbox otherwise collected unrelated tests and attempted denied user temp/cache paths.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed metadata compatibility import fallback**
- **Found during:** Task 1 RED verification
- **Issue:** Test collection failed because `metadata/__init__.py` imported `_CORE_TABLES` even when `_compat.py` did not define it in the no-SQLAlchemy fallback branch.
- **Fix:** Added `_CORE_TABLES = frozenset()` in the fallback branch.
- **Files modified:** `src/cacheness/metadata/_compat.py`
- **Verification:** Test collection progressed past the metadata import failure.
- **Committed in:** `8f67268`

**2. [Rule 3 - Blocking] Treated blosc2 cache permission failures like optional dependency absence**
- **Found during:** Task 1 RED verification
- **Issue:** `blosc2` import raised `PermissionError` while reading a denied user cache file, blocking test collection.
- **Fix:** Extended optional dependency fallback handlers to catch `PermissionError` in `compress_pickle.py` and `handlers/_compat.py`.
- **Files modified:** `src/cacheness/compress_pickle.py`, `src/cacheness/handlers/_compat.py`
- **Verification:** Test collection progressed and RED reached the stored-expiry assertion.
- **Committed in:** `8f67268`

---

**Total deviations:** 2 auto-fixed (Rule 3 blocking)
**Impact on plan:** Both were required to run the planned verification in this sandbox. TTL scope remained limited to TTL-01.

## Issues Encountered

- `uv sync --all-groups` attempted to download TensorFlow and failed due restricted network. Since TensorFlow tests are ignored by plan, verification used `uv sync --python 3.12 --group dev --group recommended` instead.
- Global uv, pytest, and blosc2 cache/temp paths under the user profile were inaccessible. Verification used project-local uv cache/Python directories and workspace-local pytest temp paths.
- `.planning/config.json` had a pre-existing unrelated local modification before execution and was preserved.

## Known Stubs

None.

## Threat Flags

None.

## User Setup Required

None.

## Next Phase Readiness

TTL-01 is implemented and covered. Plan 29-02 can build on the public cleanup semantics to route init-time cleanup through the same blob-deleting path.

## Self-Check: PASSED

- Found `.planning/phases/29-ttl-eviction-consistency/29-01-SUMMARY.md`.
- Found task commits `8f67268`, `12b2da0`, and `ae27801`.

---
*Phase: 29-ttl-eviction-consistency*
*Completed: 2026-06-13*
