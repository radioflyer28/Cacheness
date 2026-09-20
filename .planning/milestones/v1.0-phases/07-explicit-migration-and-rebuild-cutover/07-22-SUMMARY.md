---
phase: 07-explicit-migration-and-rebuild-cutover
plan: 22
subsystem: migration-verification
tags: [migration, rebuild, verifier, threat-model, validation, pytest]
requires:
  - phase: 07-21
    provides: receipt-bound rebuild cleanup settlement and fail-closed debt evidence
provides:
  - Exact AST-validated evidence for all three repaired Phase 7 blockers
  - Literal ownership checks for Plans 01-22 and 96 Phase 7 threat IDs
  - Truthful gap-cycle validation evidence and a deferred WR-02 API-contract decision
affects: [phase-7-verification, phase-8-qualification, handler-registration]
actuals:
  tokens: 10039
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - Literal reviewed plan, threat, and exact pytest-selector inventories fail closed before execution.
    - Current payload identity tests use native NPZ fixtures when they need current-to-current migration semantics.
key-files:
  created:
    - .planning/phases/07-explicit-migration-and-rebuild-cutover/07-22-SUMMARY.md
  modified:
    - tools/verify_phase7_contracts.py
    - tests/test_phase7_contract_verifier.py
    - tests/test_migration_public_contract.py
    - tests/test_migration_run_evidence.py
    - .planning/phases/07-explicit-migration-and-rebuild-cutover/07-VALIDATION.md
    - .planning/phases/07-explicit-migration-and-rebuild-cutover/deferred-items.md
key-decisions:
  - The fixed verifier owns a literal Plans 01-22 inventory and 50 gap/96 total threat oracle; discovery cannot substitute for reviewed evidence.
  - MIGR-04 and MIGR-05 become PASS only after their complete exact selector maps pass in all mode.
  - WR-02 remains a future handler-registration API-contract choice rather than a Phase 7 migration claim.
requirements-completed: [MIGR-03, MIGR-04, MIGR-05, MIGR-06]
coverage:
  - id: D1
    description: Current migration, S3 abort, and rebuild-settlement repairs are bound to exact AST-validated selectors and cannot be silently removed.
    requirement: MIGR-04
    verification:
      - kind: integration
        ref: tests/test_phase7_contract_verifier.py#test_fixed_manifest_maps_current_three_gap_repairs_exactly
        status: pass
      - kind: other
        ref: uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --all
        status: pass
    human_judgment: false
  - id: D2
    description: The validation ledger distinguishes all-mode verified selector evidence from the unrelated direct-suite concurrency failure and retains Phase 8 boundaries.
    requirement: MIGR-05
    verification:
      - kind: other
        ref: .planning/phases/07-explicit-migration-and-rebuild-cutover/07-VALIDATION.md
        status: pass
    human_judgment: false
duration: 15 min
completed: 2026-09-11
status: complete
---

# Phase 07 Plan 22: Exact Gap Evidence Summary

**The Phase 7 verifier now fails closed if any of the destination-transform, typed-S3-abort, or rebuild-cleanup-settlement repairs disappear, while the validation ledger reports only evidence actually observed.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-09-11T23:56:17Z
- **Completed:** 2026-09-12T00:11:28Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Extended the literal verifier through Plan 22: 50 unique gap-plan threat rows, 96 total threat IDs, and exact AST selectors for every current repair.
- Added adversarial tests for removed current behavior, removed plan paths, missing/new threat rows, and false requirement PASS rendering.
- Recorded quick/all evidence (75/79 exact selectors; 92/96 passing cases), closed WR-01 with its exact accounting test, and explicitly deferred WR-02.

## Task Commits

1. **Task 1: Make the fixed verifier fail if any current gap behavior disappears** — `357245b` (test), `d43570b` (feat)
2. **Task 2: Run the exact gates and record current evidence plus WR-02 disposition** — `5029334` (docs)

## Files Created/Modified

- `tools/verify_phase7_contracts.py` — owns the reviewed Plans 01-22 inventory, 50-row gap oracle, 96-ID threat map, and current selectors.
- `tests/test_phase7_contract_verifier.py` — proves omissions of current behavior, paths, rows, or exact selectors fail before pytest.
- `tests/test_migration_public_contract.py`, `tests/test_migration_run_evidence.py` — use native NPZ values for current-to-current migration fixtures.
- `07-VALIDATION.md` — adds a dated, non-destructive current evidence section.
- `deferred-items.md` — preserves the Phase 3 concurrency observation and records the WR-02 public API decision.

## Decisions Made

- Source readability never satisfies exact destination-contract evidence; the same-version/different-format transform remains its own required selector.
- The all-mode requirement outcome depends on all mapped selectors; the direct full-suite conflict is visible evidence, not a fabricated PASS.
- No new authority, coordination mechanism, cross-resource ACID claim, or production obstore claim was added.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test fixture contract] Current-to-current migration fixtures retained a legacy object payload identity**

- **Found during:** Task 1
- **Issue:** The corrected destination-contract logic correctly rejected fixture sources written as `pickle@1` when the ordinary object handler declares `compressed_pickle@1`; four fixed-selector tests could not reach their intended migration behavior.
- **Fix:** Replaced only those fixture payloads with native NPZ arrays, whose source and destination handler contract is exactly current-to-current.
- **Files modified:** `tests/test_migration_public_contract.py`, `tests/test_migration_run_evidence.py`
- **Verification:** Four affected migration tests passed; quick verifier and the 79-selector all-mode union passed.
- **Committed in:** `d43570b`

---

**2. [Rule 2 - Tracking accuracy] Replaced the stale Roadmap claim that three gap plans were ready**

- **Found during:** Plan close-out
- **Issue:** The automatic plan-count update marked Plan 22 complete but retained wording that Plans 07-20 through 07-22 were merely ready.
- **Fix:** Recorded that their fixed-verifier evidence is complete while keeping independent phase re-verification pending.
- **Files modified:** `.planning/ROADMAP.md`
- **Verification:** Roadmap shows 22/22 executed and does not claim the stale verifier report is superseded automatically.
- **Committed in:** plan metadata commit

---

**Total deviations:** 2 auto-fixed (Rule 1 test-fixture contract; Rule 2 tracking accuracy).
**Impact on plan:** The tracking correction removes an inaccurate workflow claim without changing runtime behavior or Phase 8 boundaries.

## Issues Encountered

- The all-mode verifier exited zero. A subsequent direct deterministic non-live suite invocation produced the pre-existing `test_clear_and_delete_converge_after_an_exact_snapshot` `CacheBlobLifecycleConflictError` (1344 passed, 9 skipped, 1 failed). It is recorded as a non-PASS Phase 3 concurrency observation in `07-VALIDATION.md` and `deferred-items.md`; ADR 0001 prohibits reopening the race-patch loop in Phase 7.

## TDD Gate Compliance

- RED: `357245b` recorded the new exact-manifest test failing against the prior Plan 19/84-threat inventory.
- GREEN: `d43570b` added the literal Plan 20-22/threat/selector maps; the focused verifier module passed 24 tests.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Exact Phase 7 gap evidence is complete: the all-mode verifier and all current mapped selectors pass.
- Phase 8 retains live PostgreSQL/AWS S3, Windows, supported-Python, packaging, and performance qualification.
- WR-02 needs a separately scoped public handler-registration API-contract decision; no migration behavior is blocked on it.

## Self-Check: PASSED

- Confirmed all six modified artifacts and this summary exist.
- Confirmed task commits `357245b`, `d43570b`, and `5029334` exist in repository history.
- Confirmed the fixed all-mode verifier exited zero and the direct 79-selector union passed 96 cases; the separate full-suite conflict is recorded above as a non-PASS pre-existing observation.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-11*
