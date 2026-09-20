---
phase: 01-compatibility-and-security-baseline
plan: "01"
subsystem: compatibility-and-security
tags: [public-api, exceptions, optional-dependencies, configuration, serialization]
requires: []
provides:
  - "Machine-readable CacheReason values and typed path, query, and legacy-format exceptions"
  - "Stable SQLAlchemy compatibility aliases that remain importable without optional dependencies"
  - "Exact JSON and YAML preservation of authored relative cache paths"
affects: [01-02, 01-03, 01-04, 01-05, 01-06, 01-07]
actuals:
  tokens: 3536
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - "Stable public reasons are stored as string Enum values in exception context"
    - "Optional-dependency annotations are deferred so imports fail only on feature use"
    - "Configuration paths remain authored data until a storage boundary resolves them"
key-files:
  created:
    - tests/test_public_api_contract.py
  modified:
    - src/cacheness/error_handling.py
    - src/cacheness/__init__.py
    - src/cacheness/sql_cache.py
    - src/cacheness/config.py
    - tests/test_config_validation.py
key-decisions:
  - "CacheReason uses stable lower-snake-case string values and typed boundary exceptions place the selected value in context['reason']."
  - "SQL cache classes stay importable without SQLAlchemy or pandas; construction raises the existing actionable dependency guidance."
  - "CacheStorageConfig preserves authored paths, leaving runtime resolution to planned storage boundaries."
patterns-established:
  - "Compatibility adapters stay exported as direct aliases for the full milestone."
  - "Public-surface tests use dependency-blocked subprocesses to verify deferred optional-feature failures."
requirements-completed: [MIGR-01]
coverage:
  - id: D1
    description: "Corrected public exports, SQLAlchemy aliases, optional imports, and typed reason hierarchy"
    requirement: MIGR-01
    verification:
      - kind: unit
        ref: "uv run pytest -q -o log_cli=false tests/test_public_api_contract.py tests/test_error_handling.py -x"
        status: pass
    human_judgment: false
  - id: D2
    description: "JSON and YAML preserve authored relative cache paths exactly"
    requirement: MIGR-01
    verification:
      - kind: unit
        ref: "uv run pytest -q -o log_cli=false tests/test_config_validation.py -x"
        status: pass
    human_judgment: false
duration: 5min
completed: 2026-08-29
status: complete
---

# Phase 01 Plan 01: Compatibility and Security Baseline Summary

**Public compatibility is frozen with reason-coded boundary errors, resilient optional imports, and exact authored-path configuration round trips.**

## Performance

- **Duration:** 5min
- **Started:** 2026-08-29T19:57:09Z
- **Completed:** 2026-08-29T20:01:49Z
- **Tasks:** 2/2
- **Files modified:** 6

## Accomplishments

- Added an executable public API matrix covering star imports, compatibility aliases, signatures, registries, reason codes, inheritance, and dependency-blocked imports.
- Added `CacheReason` plus typed unsafe-path, query-validation, and legacy-format errors whose contexts contain stable reason strings.
- Corrected the missing `SQLAlchemyDataAdapter` alias, retained `SQLAlchemySqlCacheAdapter`, and deferred SQLAlchemy-only annotation evaluation.
- Preserved user-authored relative `cache_dir` strings through JSON and YAML serialization without weakening configuration validation.

## Task Commits

Each task was completed through TDD commits:

1. **Task 1: Freeze the corrected public surface and typed reason contract**
   - `9ca0900` `test(01-01): freeze public API compatibility contract`
   - `415f05c` `feat(01-01): stabilize public compatibility contracts`
2. **Task 2: Preserve authored configuration values across serialization**
   - `c5c181a` `test(01-01): cover exact authored config paths`
   - `b8d64fc` `feat(01-01): preserve authored configuration paths`

## Files Created/Modified

- `tests/test_public_api_contract.py` - executable package-export, optional-import, alias, signature, and reason-code matrix.
- `src/cacheness/error_handling.py` - stable `CacheReason` enum and typed boundary-error subclasses.
- `src/cacheness/__init__.py` - corrected SQLAlchemy aliases and public error exports.
- `src/cacheness/sql_cache.py` - postponed annotations so unavailable SQLAlchemy cannot prevent import.
- `src/cacheness/config.py` - preserves authored `cache_dir` values.
- `tests/test_config_validation.py` - exact JSON/YAML relative-path round-trip coverage.

## Decisions Made

- Used lower-snake-case strings for `CacheReason` values so callers can branch and monitor without parsing exception prose.
- Kept legacy SQLAlchemy names as direct aliases for the milestone-long adapter window.
- Treated configured paths as serialized user data; future storage boundaries own runtime resolution and containment.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] Deferred SQLAlchemy-only annotation evaluation**
- **Found during:** Task 1 (dependency-blocked public import matrix)
- **Issue:** Blocking SQLAlchemy made `cacheness.sql_cache` fail at import time because a `Session` annotation was evaluated before the existing construction-time dependency check.
- **Fix:** Enabled postponed annotations in `src/cacheness/sql_cache.py`, preserving importability and the existing actionable construction-time dependency error.
- **Files modified:** `src/cacheness/sql_cache.py`
- **Verification:** Dependency-blocked subprocess cases pass for YAML, SQLAlchemy, and pandas.
- **Committed in:** `415f05c` (part of Task 1)

**Total deviations:** 1 auto-fixed (Rule 2)

**Impact on plan:** Required to satisfy the planned optional-dependency compatibility contract; no API or dependency expansion.

## Issues Encountered

- The requested Ruff scope reports 23 pre-existing findings: 15 in the legacy export barrel, two unused locals in `config.py`, and six pre-existing test imports. The new compatibility test and all newly added code are clean; no existing lint issue was changed because it is outside this plan's scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plans 01-02 through 01-06 can import the established `CacheReason` and typed boundary errors.
- Runtime storage boundaries can now resolve authored configuration paths without configuration serialization mutating them.

## Self-Check: PASSED

- All six implementation/test files and this summary exist.
- All four task commits (`9ca0900`, `415f05c`, `c5c181a`, `b8d64fc`) are present in git history.
- `uv run pytest -q -o log_cli=false tests/test_public_api_contract.py tests/test_config_validation.py tests/test_error_handling.py -x` passed: 106 tests.

---
*Phase: 01-compatibility-and-security-baseline*
*Completed: 2026-08-29*
