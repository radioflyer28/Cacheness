---
phase: 10-remove-sqlcache-pull-through-subsystem
plan: 04
subsystem: testing
tags: [pytest, ruff, documentation-contracts, sqlcache-removal]
requires:
  - phase: 09-adoption-and-release-surface-closure
    provides: BlobStore-first current documentation and quality-contract baseline
provides:
  - Retained static and frozen-suite contracts without live SqlCache test or source dependencies
  - Exact three-owner documentation cutover contract with non-owner promotion protection
affects: [phase-10-cutover, documentation, full-suite-validation]
actuals:
  tokens: 1886
  tasks: 3
  commits: 3
tech-stack:
  added: []
  patterns:
    - Rebind reusable AST sentinels to retained lifecycle boundaries before physical deletion
    - Give concise removal guidance explicit canonical owners while retaining current-guide negative checks
key-files:
  created: []
  modified:
    - tests/test_phase1_quality_gates.py
    - tests/test_full_suite_environment.py
    - tests/test_phase9_documentation.py
key-decisions:
  - "The no-direct-print AST sentinel now protects BlobStore, a retained lifecycle boundary, rather than the retiring SqlCache class."
  - "The frozen-suite diagnostic describes only supported optional-dependency behavior and retains the fixed all-extras command."
  - "Only the API reference, storage migration guide, and documentation index may own the concise SqlCache cutover statement."
patterns-established:
  - "Current-facing removal contracts use an explicit owner set and do not rewrite historical planning or dated audits."
requirements-completed: [CACH-07]
coverage:
  - id: D1
    description: Retained AST quality gates collect and lint without a deleted SqlCache source or failure-contract path.
    requirement: CACH-07
    verification:
      - kind: unit
        ref: uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase1_quality_gates.py -x
        status: pass
      - kind: other
        ref: uv run --isolated --group dev --frozen ruff check tests/test_phase1_quality_gates.py
        status: pass
    human_judgment: false
  - id: D2
    description: The exact frozen full-suite environment contract no longer diagnoses dedicated SqlCache modules.
    requirement: CACH-07
    verification:
      - kind: unit
        ref: uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_full_suite_environment.py -x
        status: pass
      - kind: other
        ref: uv run --isolated --group dev --frozen ruff check tests/test_full_suite_environment.py
        status: pass
    human_judgment: false
  - id: D3
    description: Documentation contracts name the exact future cutover-note owners and reject non-owner product promotion or dead retired-guide navigation.
    requirement: CACH-07
    verification:
      - kind: unit
        ref: uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only tests/test_phase9_documentation.py
        status: pass
      - kind: other
        ref: uv run --isolated --group dev --frozen ruff check tests/test_phase9_documentation.py
        status: pass
    human_judgment: true
    rationale: The newly precise assertions intentionally remain red until Plan 10-07 deletes retired guides and Plan 10-08 writes the canonical notes.
duration: 6 min
completed: 2026-09-17
status: complete
---

# Phase 10 Plan 04: Contract Inversion Summary

**Reusable quality, environment, and documentation contracts now describe the post-SqlCache boundary while preserving their independent protections.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-09-17T17:02:30Z
- **Completed:** 2026-09-17T17:08:46Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Removed obsolete SqlCache paths from the Phase 1 quality manifests and retained the direct-print AST sentinel against `BlobStore`.
- Kept the frozen all-extras suite command unchanged while removing the retired modules from its optional-dependency diagnosis.
- Replaced documentation blanket bans with exact canonical-note ownership, bounded use-case/no-replacement/caller-table assertions, non-owner guide protections, and retired-guide navigation checks.

## Task Commits

1. **Task 1: Retarget static quality gates to retained code** - `54c9273` (test)
2. **Task 2: Remove obsolete modules from the full-suite environment diagnosis** - `66b6347` (test)
3. **Task 3: Define exact documentation cutover-note ownership** - `aa01c10` (test)

## Files Created/Modified

- `tests/test_phase1_quality_gates.py` - Removes deleted manifest paths and applies the no-print sentinel to `BlobStore`.
- `tests/test_full_suite_environment.py` - Keeps the fixed frozen command while describing only supported optional dependency behavior.
- `tests/test_phase9_documentation.py` - Defines the three canonical note owners and guards other current guides and navigation.

## Decisions Made

- Kept broad AST analysis and direct-print protection rather than deleting the useful quality gate with the retired subsystem.
- Treated the new documentation assertions as deliberate pre-content RED contracts; collection and Ruff pass now, while Plans 10-07 and 10-08 make their runtime assertions green.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

The two new documentation assertions intentionally fail against the pre-cut current docs: the canonical note has not yet been added and `docs/PANDAS_API_AUDIT.md` still contains its scheduled-to-be-removed current SqlCache row. This is the planned Wave 0 RED state, not a collection or contract-helper failure.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plans 10-05 through 10-08 can delete the retired product/assets and make the focused Phase 10 contracts green without weakening the suite.
- The fixed full-suite invocation and retained lifecycle quality checks stay available throughout the cut.

## Self-Check: PASSED

- Confirmed all three modified contract files exist.
- Confirmed task commits `54c9273`, `66b6347`, and `aa01c10` exist in Git history.
- Confirmed the two green focused suites, documentation collection, and scoped Ruff verification pass.

---
*Phase: 10-remove-sqlcache-pull-through-subsystem*
*Completed: 2026-09-17*
