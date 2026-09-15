---
phase: 08-production-gates-and-performance-stabilization
plan: 17
subsystem: testing
tags: [blob-store, concurrency, adr-0001, sqlite, filesystem]
requires:
  - phase: 08-15
    provides: fixed Phase 8 local-readiness verifier inventory before the final closure plan
provides:
  - Exact-snapshot clear/delete test contract that distinguishes typed contention from corruption
  - Bounded 16-case event/barrier-ordered safety regression for the supported local topology
affects: [08-16, local-readiness, lifecycle-qualification]
actuals:
  tokens: 1737
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - Test-only event and barrier ordering separates permitted progress outcomes from mandatory safety and recovery assertions
key-files:
  created:
    - .planning/phases/08-production-gates-and-performance-stabilization/08-17-SUMMARY.md
  modified:
    - tests/test_blob_store_concurrency.py
key-decisions:
  - "Ordinary delete completion and CacheBlobLifecycleConflictError are both valid bounded outcomes after the exact clear snapshot."
  - "Every valid outcome must still converge to public absence, no authority entry, no cleanup debt, and bounded worker completion."
patterns-established:
  - "Concurrency tests use deterministic lifecycle seams and finite barriers, never sleeps, retries, winner assertions, or frequency targets."
requirements-completed: [QUAL-04]
coverage:
  - id: D1
    description: Preserved CR-01 accepts only ordinary delete completion or a typed lifecycle conflict while proving the same terminal safety state.
    requirement: QUAL-04
    verification:
      - kind: integration
        ref: tests/test_blob_store_concurrency.py#test_clear_and_delete_converge_after_an_exact_snapshot
        status: pass
      - kind: unit
        ref: tests/test_phase3_gap_acceptance.py#test_phase3_gap_acceptance_inventory
        status: pass
    human_judgment: false
  - id: D2
    description: Sixteen isolated exact-snapshot collisions reuse the safety oracle without scheduling or elapsed-time assertions.
    requirement: QUAL-04
    verification:
      - kind: integration
        ref: tests/test_blob_store_concurrency.py#test_clear_and_delete_exact_snapshot_stress_preserves_safety_across_valid_outcomes
        status: pass
    human_judgment: false
duration: 7m
completed: 2026-09-15
status: complete
---

# Phase 08 Plan 17: Clear/Delete Contract Correction Summary

**The exact-snapshot clear/delete regression now accepts ADR-valid typed contention while retaining strict final-state safety checks across sixteen ordered collisions.**

## Performance

- **Duration:** 7m
- **Started:** 2026-09-15T18:52:22Z
- **Completed:** 2026-09-15T18:59:12Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Refactored the preserved CR-01 selector around a shared, exact-snapshot clear/delete collision oracle.
- Accepted only an ordinary public-delete boolean or `CacheBlobLifecycleConflictError`; unexpected exceptions and incomplete workers remain failures.
- Added the required sixteen-case fresh-root regression and verified it repeatedly without sleeps, retries, or winner/frequency assertions.

## Task Commits

1. **Task 1: Correct the exact-snapshot clear/delete outcome contract** - `bae233a` (test)
2. **Task 2: Add a bounded repeated safety regression across valid outcomes** - `1eb02ff` (style)

## Files Created/Modified

- `tests/test_blob_store_concurrency.py` - shared exact-snapshot collision oracle, preserved CR-01 contract, sixteen-case regression, and focused Ruff formatting.

## Decisions Made

- The SQLite-authority/filesystem-payload topology may report a typed conflict under same-key contention; that is progress information, not corruption.
- Safety assertions are invariant across the allowed outcomes: absence through the public API, no authority row, and no cleanup debt.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Applied focused Ruff formatting required by the planned verification gate**
- **Found during:** Task 2
- **Issue:** `ruff format --check tests/test_blob_store_concurrency.py` reported formatting differences in the scoped module.
- **Fix:** Ran the repository's locked Ruff formatter on that module only; no selectors, test logic, or production files changed.
- **Files modified:** `tests/test_blob_store_concurrency.py`
- **Verification:** Focused pytest selectors, Ruff check, and Ruff format check all pass.
- **Committed in:** `1eb02ff`

---

**Total deviations:** 1 auto-fixed (1 blocking verification prerequisite).
**Impact on plan:** The test-only scope and all lifecycle boundaries remain unchanged.

## Issues Encountered

- The sandbox could not initialize the shared uv cache for the formatter. Re-running the same locked command with approved cache access completed successfully.

## User Setup Required

None - the regression uses the supported local SQLite/filesystem topology only.

## Next Phase Readiness

- Plan 08-16 can resume its fixed local-readiness inventory with both required clear/delete selectors.
- No remote qualification, publication, lifecycle implementation, dependency, workflow, or benchmark change was made.

## Self-Check: PASSED

- Confirmed the changed test module and both task commits exist.
- Confirmed the preserved CR-01 selector, the exact new regression selector, and the Phase 3 inventory collect successfully.
- Confirmed focused tests, three repeated stress invocations, Ruff lint, and Ruff format checks pass.
