---
phase: 32
plan: 05
plan_id: 32-05
subsystem: metadata/sqlite
tags: [sqlite, lifecycle, pragma, polish]
requires: [POL-06]
provides: [sqlite-close-time-optimize]
affects:
  - src/cacheness/metadata/sqlite_backend.py
  - tests/test_sqlite_schema_versioning.py
tech-stack:
  added: []
  patterns:
    - close-time SQLite PRAGMA optimize
    - best-effort cleanup guard before engine disposal
key-files:
  created:
    - .planning/phases/32-small-fixes-release-polish/32-05-SUMMARY.md
  modified:
    - src/cacheness/metadata/sqlite_backend.py
    - tests/test_sqlite_schema_versioning.py
    - .planning/phases/32-small-fixes-release-polish/deferred-items.md
key-decisions:
  - SQLite PRAGMA optimize now runs during SqliteBackend.close() before engine disposal, and optimize failures are logged at debug level without preventing disposal.
  - Connect-time PRAGMA optimize and PRAGMA page_size were removed because optimize belongs at close-time and page_size is ineffective after database creation.
requirements-completed: [POL-06]
metrics:
  duration: "approx. 35 min"
  completed: "2026-06-15T19:15:56Z"
---

# Phase 32 Plan 05: SQLite PRAGMA Lifecycle Summary

SQLite PRAGMA optimization now runs at backend close time in a guarded best-effort block before engine disposal.

## Completed Tasks

| Task | Status | Files |
|------|--------|-------|
| 32-05-01 Move SQLite optimize PRAGMA to close | Complete | `src/cacheness/metadata/sqlite_backend.py`, `tests/test_sqlite_schema_versioning.py` |

## Changes Made

- Removed connect-time `PRAGMA optimize` and connect-time `PRAGMA page_size`.
- Added close-time `PRAGMA optimize` before `engine.dispose()`.
- Kept disposal in `finally` so optimize failures cannot prevent cleanup.
- Added lifecycle regression coverage for optimize ordering and best-effort disposal.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking Test Compatibility] Accepted dict-form `metadata_dict` in existing SQLite migration tests**
- **Found during:** Task 32-05-01 verification
- **Issue:** Scoped target-file pytest failed because existing migration assertions called `json.loads()` on `metadata_dict`, while the current backend returns that value as a dict.
- **Fix:** Updated the two stale assertions to accept either dict or legacy JSON-string form.
- **Files modified:** `tests/test_sqlite_schema_versioning.py`
- **Verification:** `uv run pytest -o addopts='' tests/test_sqlite_schema_versioning.py tests/test_metadata.py -x -q --ignore=tests/test_tensorflow_handler.py` passed with 75 tests.

**Total deviations:** 1 auto-fixed blocking test issue. **Impact:** Test-only compatibility update; no production behavior expanded beyond POL-06.

## Verification

Passed:

- RED check before implementation: `uv run pytest tests/test_sqlite_schema_versioning.py -k "SqliteLifecyclePragmas" -x -q --ignore=tests/test_tensorflow_handler.py` failed as expected because close-time optimize was missing.
- Focused lifecycle check after implementation: `uv run pytest tests/test_sqlite_schema_versioning.py -k "SqliteLifecyclePragmas" -x -q --ignore=tests/test_tensorflow_handler.py` passed.
- Scoped plan verification: `uv run pytest -o addopts='' tests/test_sqlite_schema_versioning.py tests/test_metadata.py -x -q --ignore=tests/test_tensorflow_handler.py` passed: 75 passed.
- `uv run ruff check src/cacheness/metadata/sqlite_backend.py tests/test_sqlite_schema_versioning.py tests/test_metadata.py` passed.
- `rg -n "PRAGMA optimize|PRAGMA page_size" src/cacheness/metadata/sqlite_backend.py` shows only close-time optimize remains; `PRAGMA page_size` is absent.

Notes:

- The initial plan pytest command without `-o addopts=''` collected broader repository tests and surfaced unrelated existing failures. Scoped verification used `-o addopts=''` as allowed by the plan.
- `uv run ty check src/cacheness/metadata/sqlite_backend.py tests/test_sqlite_schema_versioning.py tests/test_metadata.py` still reports pre-existing dynamic SQLAlchemy and pytest typing diagnostics unrelated to this change; see `deferred-items.md`.

## Known Stubs

None.

## Threat Flags

None.

## Self-Check: PASSED

- Summary file created.
- POL-06 source and regression files exist.
- Required scoped verification passed.

## Next Phase Readiness

Ready for Phase 32 Plan 06.
