---
phase: 01-compatibility-and-security-baseline
plan: "05"
subsystem: metadata-query-security
tags: [sqlite, sqlalchemy, json1, validation, bound-parameters, query-meta]
requires:
  - phase: 01-01
    provides: Stable CacheQueryValidationError reason context for public validation failures
  - phase: 01-03
    provides: Compatibility-aware high-level cache safety boundaries
provides:
  - Bounded dotted-field validation and SQLite JSON path construction
  - Whole-request query field validation before backend or session access
  - Bound SQLAlchemy query_meta expressions with raw and legacy filter compatibility
affects: [UnifiedCache, metadata-backends, query-meta, Phase 01 quality gate]
actuals:
  tokens: 5325
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - Validate all caller-controlled query fields before constructing SQLAlchemy or SQLite work
    - Bind JSON paths and values separately; never interpolate caller input into SQL text
key-files:
  created:
    - src/cacheness/query_validation.py
    - tests/test_query_meta_security.py
  modified:
    - src/cacheness/core.py
    - tests/test_query_meta.py
key-decisions:
  - Query fields are strictly bounded dotted identifiers, while raw strings and booleans map to exact stored values and numeric filters preserve threshold semantics.
  - Historical serialized string filters remain exact-match compatible alongside the raw public filter inputs.
  - Invalid query fields are public typed errors and never become query_meta None results.
patterns-established:
  - Validate an entire request before opening a database session so a hostile late key cannot execute earlier predicates.
  - Capture compiled statements in tests to prove caller paths and values remain bound parameters.
requirements-completed: [SECU-06]
coverage:
  - id: D1
    description: Fixed field grammar rejects hostile, empty, over-depth, over-length, and control-character paths before database access.
    requirement: SECU-06
    verification:
      - kind: unit
        ref: uv run pytest -q -o log_cli=false tests/test_query_meta_security.py -x
        status: pass
    human_judgment: false
  - id: D2
    description: query_meta retains raw numeric/string/bool semantics and builds validated SQLite expressions with bound JSON paths and values.
    requirement: SECU-06
    verification:
      - kind: integration
        ref: uv run pytest -q -o log_cli=false tests/test_query_meta.py tests/test_query_meta_security.py tests/test_core.py -x
        status: pass
    human_judgment: false
duration: 6min
completed: 2026-08-29
status: complete
---

# Phase 01 Plan 05: Safe Metadata Queries Summary

**query_meta now validates bounded dotted field paths before any database work and compiles raw-compatible filters as bound SQLAlchemy SQLite expressions.**

## Performance

- **Duration:** 6min
- **Started:** 2026-08-29T21:44:21Z
- **Completed:** 2026-08-29T21:50:49Z
- **Tasks:** 2/2
- **Files modified:** 4

## Accomplishments

- Added a strict one-to-sixteen-segment dotted identifier grammar with a 255-character cap, stable invalid-field reason, and safe SQLite JSON path conversion.
- Validated every supplied field before the backend/session boundary, including invalid keys in first, middle, and last mapping positions.
- Replaced SQL string construction with SQLAlchemy select, JSON extraction, casts, and independently bound path/value parameters.
- Preserved raw numeric thresholds, exact raw string and boolean filters, query-all behavior, legacy serialized string filters, and the documented None-filter no-match result.
- Added hostile-field, call-order, raw-type, and compiled-statement parameter coverage.

## Task Commits

Each TDD task was committed atomically:

1. **Task 1: Freeze safe query semantics and pre-database rejection**
   - `fd9f8fe` `test(01-05): specify safe metadata query validation`
   - `e89ed7f` `feat(01-05): validate metadata query fields`
2. **Task 2: Build query_meta exclusively from validated bound expressions**
   - `a38e17f` `feat(01-05): bind validated metadata queries`

## Files Created/Modified

- `src/cacheness/query_validation.py` - bounded query-field grammar and validated SQLite path helper.
- `src/cacheness/core.py` - pre-session validation and bound `query_meta` expression construction.
- `tests/test_query_meta.py` - raw semantic and invalid-field contract coverage.
- `tests/test_query_meta_security.py` - hostile path, database-boundary, and bind-parameter coverage.

## Decisions Made

- Kept field validation independent of backend availability so invalid public fields always report a typed error rather than a backend-specific None.
- Bound stored `str:`, `bool:`, `int:`, and `float:` compatibility inputs exactly while converting raw strings and booleans to their persisted form.
- Applied numeric comparison only after checking bool explicitly, because bool is an int subclass but has categorical equality semantics.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Restored None-filter no-match behavior**
- **Found during:** Task 2 (Build query_meta exclusively from validated bound expressions)
- **Issue:** Initial exact-value normalization converted a `None` filter to the persisted literal `"None"`, returning a match where the documented behavior is an empty result.
- **Fix:** Bound SQL NULL directly for `None` so SQLite preserves the historical no-match result.
- **Files modified:** `src/cacheness/core.py`
- **Verification:** `uv run pytest -q -o log_cli=false tests/test_query_meta.py tests/test_query_meta_security.py tests/test_core.py -x`
- **Committed in:** `a38e17f` (part of Task 2)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug)
**Impact on plan:** The correction preserved the documented safe-result contract without expanding storage formats or query capabilities.

## Issues Encountered

- The passing pytest process emits existing `SqliteBackend.__del__` interpreter-shutdown `sys.meta_path is None` messages; they do not change its zero exit status and are outside this plan's query scope.
- The wider lint scope reports five existing unused-import/redefinition findings in legacy `core.py` and `test_query_meta.py`; the new validator and security-test files pass Ruff unchanged.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Metadata query fields are now a fixed, typed boundary suitable for subsequent compatibility and quality-gate work.
- Future query changes must validate the entire field set before session construction and keep all caller values and paths bound.

## Self-Check: PASSED

- All four implementation/test artifacts and this summary exist.
- All Task 1/Task 2 TDD commits (`fd9f8fe`, `e89ed7f`, and `a38e17f`) are present in git history.
- The modified files contain no intentional stubs, placeholders, skipped tests, or unrun required verification.

---
*Phase: 01-compatibility-and-security-baseline*
*Completed: 2026-08-29*
