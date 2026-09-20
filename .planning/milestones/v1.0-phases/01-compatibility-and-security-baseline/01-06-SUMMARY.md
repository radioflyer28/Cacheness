---
phase: 01-compatibility-and-security-baseline
plan: "06"
subsystem: sql-pull-through-cache
tags: [sqlalchemy, sqlite, pandas, failure-contracts, logging]
requires:
  - "01-01: CacheReason values and CacheError hierarchy"
provides:
  - "Strict SqlCache completeness errors that retain every failed range and original cause"
  - "Opt-in best-effort SqlCacheResult reports with immutable ordered failure records"
  - "Structured, observable custom-gap and bulk-upsert fallback behavior"
affects: [phase-01-validation, sql-cache-regressions]
actuals:
  tokens: 6810
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - "Stage all pull-through fetch frames before strict-mode storage mutation"
    - "Use frozen result records and structured logging for opt-in partial outcomes"
    - "Run equivalent SQL fallbacks inside savepoints and preserve both failure causes"
key-files:
  created:
    - tests/test_sql_cache_failure_contract.py
  modified:
    - src/cacheness/sql_cache.py
    - tests/test_sql_cache.py
key-decisions:
  - "Strict SqlCache calls return a DataFrame only after all missing ranges resolve; every failure is collected before one typed error is raised."
  - "Best effort is an explicit keyword-only mode that returns SqlCacheResult with an immutable ordered failures tuple."
  - "Custom gap detection falls back to built-in detection only in best-effort mode, while bulk-to-row upsert remains an observable internal fallback."
patterns-established:
  - "Caller-controlled adapter and gap-detector failures carry a typed public reason and preserve their original exception as __cause__."
  - "Best-effort continuation emits exactly one structured warning per unresolved condition."
requirements-completed: [CACH-07]
coverage:
  - id: D1
    description: "SqlCache remains independently importable and retains its representative SQLite pull-through success workflow."
    requirement: CACH-07
    verification:
      - kind: integration
        ref: "uv run pytest -q -o log_cli=false tests/test_sql_cache.py tests/test_sql_cache_failure_contract.py -x"
        status: pass
    human_judgment: false
  - id: D2
    description: "Strict and best-effort missing-range, adapter, gap-detector, and upsert failure contracts are executable."
    requirement: CACH-07
    verification:
      - kind: integration
        ref: "tests/test_sql_cache_failure_contract.py"
        status: pass
    human_judgment: false
duration: 8min
completed: 2026-08-29
status: complete
---

# Phase 01 Plan 06: SqlCache Completeness Summary

**SqlCache now returns complete pull-through DataFrames by default, with typed staged failures and explicit inspectable partial results only when best effort is requested.**

## Performance

- **Duration:** 8min
- **Started:** 2026-08-29T20:24:54Z
- **Completed:** 2026-08-29T20:33:06Z
- **Tasks:** 2/2
- **Files modified:** 3

## Accomplishments

- Added a deterministic SQLite failure matrix for first, middle, last, multiple, and empty missing-range results, plus adapter, gap-detector, and write fallback faults.
- Staged all strict fetch frames before writes, returning one `SqlCacheFetchError` with ordered failure context and no committed partial rows.
- Added opt-in `failure_mode="best_effort"`, frozen `SqlCacheResult`/`SqlCacheFailure` reports, cause-preserving typed errors, and structured fallback logging.

## Task Commits

1. **Task 1: Specify strict completeness and explicit partial-result behavior**
   - `3ed74b3` `test(01-06): specify SQL cache failure contract`
2. **Task 2: Implement staged strict fetches and structured equivalent fallbacks**
   - `678ecba` `feat(01-06): implement SQL cache completeness contract`

## Files Created/Modified

- `src/cacheness/sql_cache.py` — typed failure/report models, staged fetching, explicit best-effort handling, and observable upsert/gap fallbacks.
- `tests/test_sql_cache_failure_contract.py` — deterministic failure ordering, rollback, partial-result, structured-log, and cause-preservation coverage.
- `tests/test_sql_cache.py` — confirms the public package aliases remain the independent SqlCache implementation.

## Decisions Made

- Made strict completeness the default and reserved partial data for a distinct opt-in result type, preventing a caller from mistaking incomplete data for a completed pull-through result.
- Preserved `SQLCacheError` as the public base name while integrating it with the project's `CacheError` and `CacheReason` contract.
- Kept bulk-to-row upsert as an internal equivalence fallback, isolating the failed bulk attempt with a savepoint before invoking row-level recovery.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed duplicate best-effort gap failure logging**
- **Found during:** Task 2 (focused failure-contract verification)
- **Issue:** A custom gap-detector failure logged once when selecting the built-in fallback and a second time during generic best-effort reporting.
- **Fix:** Kept the fallback-decision warning as the one structured gap failure record and limited generic reporting to fetch failures.
- **Files modified:** `src/cacheness/sql_cache.py`
- **Verification:** Focused failure-contract suite asserts one `gap_detection` structured log and passes.
- **Committed in:** `678ecba` (part of Task 2)

**Total deviations:** 1 auto-fixed (Rule 1)

**Impact on plan:** The fix enforces the planned one-log-per-failure observability contract without changing the public surface.

## Issues Encountered

- The sandbox denied access to uv's shared cache during normal verification. The same focused test and Ruff commands passed with the approved verification permission.
- `tests/test_sql_cache.py` retains two pre-existing Ruff unused-import findings outside this plan's changes; the changed source and new failure-contract test are clean.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase validation can rely on executable strict/partial SqlCache behavior and structured failure observability.
- SqlCache remains isolated from the BlobStore/UnifiedCache lifecycle migration.

## Self-Check: PASSED

- The three task files and this summary exist.
- Both task commits (`3ed74b3`, `678ecba`) are present in git history.
- `uv run pytest -q -o log_cli=false tests/test_sql_cache.py tests/test_sql_cache_failure_contract.py -x` passed: 27 tests.
- `uv run ruff check src/cacheness/sql_cache.py tests/test_sql_cache_failure_contract.py` passed.

---
*Phase: 01-compatibility-and-security-baseline*
*Completed: 2026-08-29*
