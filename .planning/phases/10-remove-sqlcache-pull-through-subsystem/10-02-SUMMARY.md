---
phase: 10-remove-sqlcache-pull-through-subsystem
plan: 02
subsystem: testing
tags: [pytest, ruff, fixed-verifiers, sqlcache-removal]
requires:
  - phase: 10-remove-sqlcache-pull-through-subsystem
    provides: Collectable Phase 10 negative removal contract
provides:
  - Phase 4 verifier detached from historical SqlCache diagnostic paths
  - Phase 6 verifier scoped to CACH-01 through CACH-06
  - Phase 07.1 retirement selector bound to the Phase 10 negative contract
affects: [phase-10-cutover, historical-verifiers, test-collection]
actuals:
  tokens: 2551
  tasks: 3
  commits: 6
tech-stack:
  added: []
  patterns:
    - Closed verifiers preserve retained literal matrices while historical evidence stays immutable
    - Retired product selectors delegate to the current phase that owns their negative contract
key-files:
  created: []
  modified:
    - tools/verify_phase4_cutover.py
    - tests/test_phase4_cutover_verifier.py
    - tools/verify_phase6_contracts.py
    - tests/test_phase6_contract_verifier.py
    - tools/verify_phase071_contracts.py
    - tests/test_phase071_contract_verifier.py
key-decisions:
  - "Phase 4 parses only its retained owned matrix; dated deferred diagnostic evidence remains unchanged but is no longer executable input."
  - "Phase 6 owns CACH-01 through CACH-06, while Phase 10 solely owns the CACH-07 removal contract."
  - "Phase 07.1 validates the exact Phase 10 natural-absence node instead of a retired positive selector."
patterns-established:
  - "When a planned deletion removes a fixed test node, rebind closed verifier inventories before the physical deletion."
requirements-completed: [CACH-07]
coverage:
  - id: D1
    description: Phase 4's retained lifecycle verifier no longer parses removed SqlCache diagnostic paths.
    requirement: CACH-07
    verification:
      - kind: unit
        ref: uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase4_cutover_verifier.py -x
        status: pass
      - kind: other
        ref: uv run --isolated --group dev --frozen ruff check tools/verify_phase4_cutover.py tests/test_phase4_cutover_verifier.py
        status: pass
    human_judgment: false
  - id: D2
    description: Phase 6's literal contract manifest retains only CACH-01 through CACH-06 and supporting regressions.
    requirement: CACH-07
    verification:
      - kind: unit
        ref: uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase6_contract_verifier.py -x
        status: pass
      - kind: other
        ref: uv run --isolated --group dev --frozen ruff check tools/verify_phase6_contracts.py tests/test_phase6_contract_verifier.py
        status: pass
    human_judgment: false
  - id: D3
    description: Phase 07.1's fixed inventory selects Phase 10's natural SqlCache-absence assertion.
    requirement: CACH-07
    verification:
      - kind: unit
        ref: uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase071_contract_verifier.py -x
        status: pass
      - kind: other
        ref: uv run --isolated --group dev --frozen ruff check tools/verify_phase071_contracts.py tests/test_phase071_contract_verifier.py
        status: pass
    human_judgment: false
duration: 5 min
completed: 2026-09-17
status: complete
---

# Phase 10 Plan 02: Closed Verifier Inversion Summary

**Phase 4, Phase 6, and Phase 07.1 fixed verifiers now retain their real contracts without requiring the retiring SqlCache test nodes.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-09-17T13:11:00-04:00
- **Completed:** 2026-09-17T13:15:51-04:00
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Removed the Phase 4 verifier's live parsing and exception path for historical SqlCache diagnostics while preserving its owned lifecycle matrix and dated validation document.
- Scoped Phase 6's closed evidence inventory and diagnostic labels to CACH-01 through CACH-06.
- Rebound the Phase 07.1 all-node inventory to Phase 10's exact natural-absence test, retaining every Phase 07.1 plan, decision, threat, and non-claim inventory.

## Task Commits

1. **Task 1: Remove deleted diagnostics from the Phase 4 live verifier** - `138017a` (test RED), `99d4f3f` (feat GREEN)
2. **Task 2: Return Phase 6 verification to CACH-01 through CACH-06** - `d9a795c` (test RED), `f11a6df` (feat GREEN)
3. **Task 3: Point Phase 07.1 retirement coverage at the Phase 10 contract** - `ee00034` (test RED), `dff2e40` (feat GREEN)

## Files Created/Modified

- `tools/verify_phase4_cutover.py` - Executes only the retained Phase 4 matrix and fails any current collection error directly.
- `tests/test_phase4_cutover_verifier.py` - Locks the live Phase 4 matrix to retained paths.
- `tools/verify_phase6_contracts.py` - Removes the retired CACH-07 SqlCache node from Phase 6 ownership.
- `tests/test_phase6_contract_verifier.py` - Mirrors the CACH-01 through CACH-06-only literal inventory.
- `tools/verify_phase071_contracts.py` - Selects Phase 10's natural-absence boundary in its all-node inventory.
- `tests/test_phase071_contract_verifier.py` - Rejects the obsolete positive selector and requires the new owner.

## Decisions Made

- Historical validation documents remain truthful records rather than mutable dependencies of current verifier execution.
- The separate product's absence is proven by Phase 10, avoiding a misleading preservation of CACH-07 inside Phase 6.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Verification

- `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase4_cutover_verifier.py tests/test_phase6_contract_verifier.py tests/test_phase071_contract_verifier.py -x` — 48 passed.
- `uv run --isolated --group dev --frozen ruff check` over all six changed Python files — passed.

## Next Phase Readiness

The coordinated Phase 10 product cut can delete dedicated SqlCache test modules without breaking fixed verifier collection or obscuring retained lifecycle coverage.

## Self-Check: PASSED

- Confirmed the summary exists and all six TDD task commits are present in Git history.
- Confirmed the combined 48-test verifier suite and scoped Ruff gate pass.

---
*Phase: 10-remove-sqlcache-pull-through-subsystem*
*Completed: 2026-09-17*
