---
phase: 07-explicit-migration-and-rebuild-cutover
plan: 23
subsystem: storage lifecycle recovery
tags: [migration, rebuild, cleanup-debt, lifecycle-authority, verifier]
requires:
  - phase: 07-21
    provides: "Exact receipt-bound rebuild cleanup settlement and terminal ABORTED evidence invariant"
  - phase: 07-22
    provides: "Literal Phase 7 verifier inventory through Plan 22"
provides:
  - "Forward rebuild cleanup-debt fence before source, destination, authority, or evidence effects"
  - "State-limited explicit resume settlement for exact authenticated receipts"
  - "Plan 23 literal MIGR-05 and threat ownership in the fixed verifier"
affects: [Phase 7 completion, Phase 8 qualification]
tech-stack:
  added: []
  patterns:
    - "Authenticate evidence once, fence recovery debt before forward coordinator work, then reuse that evidence for validation"
    - "External cleanup remains receipt-bound and can be dispatched only from defined nonterminal recovery states"
key-files:
  created: []
  modified:
    - src/cacheness/storage/migration.py
    - src/cacheness/storage/migration_evidence.py
    - tests/test_rebuild_workflow.py
    - tools/verify_phase7_contracts.py
    - tests/test_phase7_contract_verifier.py
key-decisions:
  - "Authenticated rebuild cleanup debt fences direct stage, verify, and accept operations; explicit resume remains the only settlement dispatcher."
  - "Only REBUILDING and REBUILD_VERIFYING evidence may enter exact receipt cleanup; accepted evidence with debt is invalid at the model boundary."
actuals:
  tokens: 6297
  tasks: 1
  commits: 2
requirements-completed: [MIGR-05]
coverage:
  - id: D1
    description: "Authentic rebuild cleanup debt cannot progress through direct stage, verify, or accept calls and settles only by explicit resume."
    requirement: MIGR-05
    verification:
      - kind: integration
        ref: tests/test_rebuild_workflow.py#test_rebuild_cleanup_debt_fences_forward_methods_and_resume_settles_exact_receipts
        status: pass
    human_judgment: false
  - id: D2
    description: "Terminal accepted rebuild evidence cannot carry cleanup debt, and the fixed verifier owns all Plan 23 claims."
    requirement: MIGR-05
    verification:
      - kind: unit
        ref: tests/test_rebuild_workflow.py#test_rebuild_evidence_rejects_accepted_cleanup_debt
        status: pass
      - kind: unit
        ref: tests/test_phase7_contract_verifier.py#test_fixed_manifest_maps_rebuild_cleanup_debt_fence_exactly
        status: pass
    human_judgment: false
duration: 36m
completed: 2026-09-12
status: complete
---

# Phase 07 Plan 23: Rebuild Cleanup-Debt Fence Summary

**Authenticated rebuild cleanup debt now blocks forward progression, settles only through explicit receipt-bound resume, and is fail-closed at accepted evidence.**

## Performance

- **Duration:** 36m
- **Started:** 2026-09-12T01:13:05Z
- **Completed:** 2026-09-12T01:48:55Z
- **Tasks:** 1
- **Files modified:** 5

## Accomplishments

- Added one private coordinator guard immediately after authenticated evidence loading in direct rebuild operations, preventing revalidation, payload access, authority mutation, and evidence replacement while cleanup debt exists.
- Restricted `resume()` cleanup dispatch to `REBUILDING` and `REBUILD_VERIFYING`, preserving existing exact receipt plus lifecycle-authority replay settlement and rejecting other debt-bearing states before participant effects.
- Made `REBUILD_ACCEPTED` evidence with cleanup debt invalid and extended the literal verifier through Plan 23, 54 gap threats, and 100 total Phase 7 threats.

## Task Commits

1. **Task 1: Fence authentic rebuild debt end to end and bind it to the fixed verifier (RED)** — `c127c2d` (`test`)
2. **Task 1: Fence authentic rebuild debt end to end and bind it to the fixed verifier (GREEN)** — `1645a6a` (`feat`)

## Files Created/Modified

- `src/cacheness/storage/migration.py` — Fences forward rebuild actions and classifies eligible recovery states before cleanup settlement.
- `src/cacheness/storage/migration_evidence.py` — Rejects accepted rebuild evidence that retains cleanup debt.
- `tests/test_rebuild_workflow.py` — Covers authentic debt forward fences, unchanged authority/payload snapshots, exact settlement, and accepted-evidence rejection.
- `tools/verify_phase7_contracts.py` — Adds literal Plan 23 inventory, threat ownership, and exact MIGR-05/decision/assumption mappings.
- `tests/test_phase7_contract_verifier.py` — Validates Plan 23 counts, selectors, ownership, and adversarial threat/selector removals.

## Decisions Made

- Reused the existing lifecycle authority and exact receipt settlement path; no new lifecycle state, persistence field, lock, queue, lease, journal, sidecar, listing/adoption path, online-writer protocol, ACID claim, or obstore integration was introduced.
- Preserved the Phase 8 boundary: all verification is deterministic local evidence; live PostgreSQL/AWS S3, platform, packaging, and performance qualification remain unqualified Phase 8 work.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated the fixed-plan omission test for Plan 23**
- **Found during:** Task 1 full verifier run
- **Issue:** The existing adversarial test expected Plan 22 to be the final literal plan after the Plan 23 inventory extension.
- **Fix:** Updated the expected missing path to `07-23-PLAN.md` and extended adversarial checks to cover Plan 23 selectors and all four threat rows.
- **Files modified:** `tests/test_phase7_contract_verifier.py`
- **Verification:** Focused verifier tests passed; the full deterministic verifier passed.
- **Committed in:** `1645a6a`

**Total deviations:** 1 auto-fixed (Rule 1 bug)

## Verification

- Passed the exact nine-selector Plan 23 pytest gate.
- Passed `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --quick`.
- Passed `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --all`, including the deterministic non-live suite and scoped Ruff inventory.
- Passed scoped Ruff on all five Plan 23 files.

## Known Stubs

None. Stub-pattern matches were pre-existing optional/default values or test error expectations and do not feed a rendered deliverable.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 7's remaining MIGR-05 loss path is closed with deterministic local evidence. Phase 8 retains live PostgreSQL/AWS S3, supported-platform, packaging, and performance qualification.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-12*

## Self-Check: PASSED

- `07-23-SUMMARY.md` exists at the required phase path.
- Both task commits (`c127c2d`, `1645a6a`) exist in git history.
