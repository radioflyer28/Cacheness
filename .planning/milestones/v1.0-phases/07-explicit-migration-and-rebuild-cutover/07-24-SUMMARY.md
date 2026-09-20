---
phase: 07-explicit-migration-and-rebuild-cutover
plan: 24
subsystem: verifier contracts
tags: [migration, rebuild, cleanup-debt, threat-mapping, testing]
requires:
  - phase: 07-23
    provides: "Authenticated forward-fence and exact resume-settlement regression"
provides:
  - "Complete ordered Plan 21 terminal-state threat evidence"
  - "Fail-closed self-tests for forged-debt and forward-fence selectors"
affects: [Phase 7 verification, Phase 8 qualification]
actuals:
  tokens: 1002
  tasks: 1
  commits: 2
tech-stack:
  added: []
  patterns:
    - "Literal threat-to-selector maps use complete ordered tuple equality"
key-files:
  created: []
  modified:
    - tools/verify_phase7_contracts.py
    - tests/test_phase7_contract_verifier.py
key-decisions:
  - "T-07-21-03 retains forged-debt evidence and appends the exact forward-fence/resume selector."
  - "Plan 23 and current-three-gap verifier tests compare the complete ordered tuple so either selector removal fails closed."
patterns-established:
  - "Cross-plan threat ownership uses literal ordered evidence tuples, not membership-only checks."
requirements-completed: [MIGR-05]
coverage:
  - id: D1
    description: "Plan 21 terminal-state threat owns both forged-debt and forward-fence/resume evidence in the required order."
    requirement: MIGR-05
    verification:
      - kind: unit
        ref: tests/test_phase7_contract_verifier.py#test_fixed_manifest_maps_rebuild_cleanup_debt_fence_exactly
        status: pass
      - kind: unit
        ref: tests/test_phase7_contract_verifier.py#test_fixed_manifest_maps_current_three_gap_repairs_exactly
        status: pass
    human_judgment: false
  - id: D2
    description: "The fixed Phase 7 quick verifier retains its reviewed plan inventory and exact threat counts."
    requirement: MIGR-05
    verification:
      - kind: other
        ref: uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --quick
        status: pass
    human_judgment: false
duration: 4m
completed: 2026-09-12
status: complete
---

# Phase 07 Plan 24: Terminal-State Threat Evidence Summary

**The fixed verifier now binds the Plan 21 terminal-state threat to both forged-debt rejection and forward-fence/resume settlement evidence.**

## Performance

- **Duration:** 4m
- **Started:** 2026-09-12T02:52:39Z
- **Completed:** 2026-09-12T02:56:10Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- Expanded `T-07-21-03` to retain the original forged-debt selector and add the exact forward-fence/resume regression.
- Made both affected verifier self-tests assert the complete ordered two-selector tuple, closing the false-green mapping gap.
- Preserved the fixed Phase 7 inventory: 23 reviewed plan paths, 54 gap threats, and 100 total threats.

## Task Commits

1. **Task 1: Bind the exact forward-fence regression to the existing terminal-state threat (RED)** — `e08e134` (`test`)
2. **Task 1: Bind the exact forward-fence regression to the existing terminal-state threat (GREEN)** — `3f3929f` (`feat`)

## Files Created/Modified

- `tools/verify_phase7_contracts.py` — Associates the Plan 21 terminal-state threat with both exact recovery regressions.
- `tests/test_phase7_contract_verifier.py` — Enforces ordered complete-tuple equality in the Plan 23 and current-three-gap verifier contracts.

## Decisions Made

- Reused the existing literal selector table and test seams; no production lifecycle, state, authority, locking, queueing, journaling, or obstore integration work was introduced.
- Kept the optional WR-02 state-matrix warning and the pre-existing Phase 3 SQLite contention failure outside this verifier-only closure.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The initial exact test failure was the expected RED phase and was resolved by the planned selector-table change.

## Verification

- Passed `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase7_contract_verifier.py::test_fixed_manifest_maps_rebuild_cleanup_debt_fence_exactly -x -o log_cli=false -o addopts=`.
- Passed `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase7_contract_verifier.py::test_fixed_manifest_maps_current_three_gap_repairs_exactly -x -o log_cli=false -o addopts=`.
- Passed `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --quick`.
- Passed scoped Ruff for both verifier artifacts.

## Known Stubs

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 7's independent fixed-verifier MIGR-05 binding is complete. Live service, platform, packaging, and performance qualification remain Phase 8 work.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-12*

## Self-Check: PASSED

- `07-24-SUMMARY.md` exists at the required phase path.
- Both task commits (`e08e134`, `3f3929f`) exist in git history.
