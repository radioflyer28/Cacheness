---
phase: 07-explicit-migration-and-rebuild-cutover
plan: 21
subsystem: storage-migration
tags: [rebuild, recovery, lifecycle-authority, cleanup-debt, integrity]
requires:
  - phase: 07-20
    provides: destination-contract selection and typed migration-abort cleanup debt
provides:
  - Receipt-bound rebuild cleanup debt that remains resumable until exact settlement
  - Fail-closed cleanup validation for forged evidence and changed current ownership
affects: [07-22, phase-08-qualification]
actuals:
  tokens: 8112
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Existing lifecycle-authority replay corroborates bounded cleanup receipts before participant effects.
    - Changed current ownership is preserved while only the immutable recorded locator is settled.
key-files:
  created: []
  modified:
    - src/cacheness/storage/memory_lifecycle_authority.py
    - src/cacheness/storage/migration.py
    - src/cacheness/storage/migration_evidence.py
    - tests/test_rebuild_workflow.py
key-decisions:
  - "Cleanup debt is retired only after an exact receipt and authority replay corroborate deletion or absence."
  - "A later logical-key owner never authorizes current-key deletion; settlement addresses only the recorded immutable locator."
  - "Terminal ABORTED evidence requires empty debt and retirement of every recorded rebuild receipt."
patterns-established:
  - "Replay snapshots: an authority mutation retains its promoted entry snapshot so a later key owner cannot rewrite old-operation evidence."
  - "Fail-closed cleanup: malformed debt rejects before participant reads, deletes, or listing calls."
requirements-completed: [MIGR-05, MIGR-06]
coverage:
  - id: D1
    description: Receipt-bound rebuild cleanup remains resumable through response loss and reaches ABORTED only after exact settlement.
    requirement: MIGR-05
    verification:
      - kind: integration
        ref: tests/test_rebuild_workflow.py#test_rebuild_cleanup_debt_stays_resumable_until_exact_settlement
        status: pass
      - kind: unit
        ref: tests/test_rebuild_workflow.py#test_rebuild_evidence_rejects_terminal_aborted_cleanup_debt
        status: pass
    human_judgment: false
  - id: D2
    description: Forged receipt debt and changed current ownership fail closed without discovery or unrelated deletion.
    requirement: MIGR-06
    verification:
      - kind: integration
        ref: tests/test_rebuild_workflow.py#test_rebuild_cleanup_retry_rejects_forged_debt_without_payload_access
        status: pass
      - kind: integration
        ref: tests/test_rebuild_workflow.py#test_rebuild_cleanup_retry_preserves_changed_current_ownership
        status: pass
    human_judgment: false
duration: 18 min
completed: 2026-09-11
status: complete
---

# Phase 07 Plan 21: Receipt-bound Rebuild Cleanup Settlement Summary

**Rebuild cleanup now settles only authenticated immutable receipt locators, preserves later current owners, and reaches terminal evidence only after every recorded receipt retires.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-09-11T23:29:20Z
- **Completed:** 2026-09-11T23:47:21Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Kept failed cleanup in an existing resumable rebuild state and replayed exact lifecycle-authority receipts before retrying deletion or proving absence.
- Snapshotted promoted in-memory authority entries so a later key owner cannot alter the ownership replay for an earlier rebuild operation.
- Added regressions for every forged debt identity field, zero participant discovery on refusal, and exact old-locator cleanup while retaining a changed current owner.

## Task Commits

1. **Task 1: Keep one failed rebuild cleanup recoverable and settle it on resume** — `edb6bab` (test), `7ff1e52` (feat)
2. **Task 2: Reject forged debt and preserve changed ownership during settlement** — `cc97c4e` (test), `446c711` (test)

## Files Created/Modified

- `src/cacheness/storage/migration.py` — checkpoints exact rebuild debt, settles it through authority replay, and terminally aborts only after retirement.
- `src/cacheness/storage/migration_evidence.py` — rejects terminal rebuild evidence with outstanding debt or incomplete receipt retirement.
- `src/cacheness/storage/memory_lifecycle_authority.py` — retains the original promoted entry in operation replay.
- `tests/test_rebuild_workflow.py` — covers settlement, forged debt, current-owner preservation, and no-listing cleanup behavior.

## Decisions Made

- Cleanup ownership is authenticated by the existing operation replay plus the recorded `BlobReceipt`; evidence never discovers work from paths, listings, or current keys.
- An operational participant failure remains bounded explicit debt; integrity, ownership, and evidence conflicts continue to propagate fail closed.
- No lifecycle state, authority API, journal, lock, queue, lease, sidecar, or production obstore dependency was added.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Preserve historical promoted entries in in-memory authority replay**
- **Found during:** Task 1
- **Issue:** `read_mutation(operation_id)` reconstructed the promoted result from the mutable current-key entry, so a later owner could rewrite the old operation's replay evidence.
- **Fix:** Store a copied promoted `EntrySnapshot` with the mutation and use it for replay.
- **Files modified:** `src/cacheness/storage/memory_lifecycle_authority.py`
- **Verification:** Focused rebuild settlement selectors passed.
- **Committed in:** `7ff1e52`

**Total deviations:** 1 auto-fixed (Rule 1).
**Impact on plan:** The correction deepens the existing authority replay seam required for exact receipt-bound cleanup; it adds no coordination mechanism or lifecycle authority.

## Issues Encountered

- Task 2's new regressions passed on their first run because Task 1 already established the required authority-replay and exact-locator behavior. The tests were retained as independent fail-closed coverage.
- The workflow regression sweep excluded three Phase 8 live PostgreSQL/S3 modules because their qualification fixture is intentionally unavailable when invoked as standalone test files. The remaining deterministic prior-phase regression set exited successfully with two expected platform skips.

## Verification

- `uv run --isolated --all-extras --group dev --frozen pytest -q ...test_rebuild_* ...` — 10 passed.
- `uv run --isolated --all-extras --group dev --frozen ruff check src/cacheness/storage/migration.py src/cacheness/storage/migration_evidence.py tests/test_rebuild_workflow.py` — passed.
- Deterministic prior-phase regression set — passed; 2 expected platform skips.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 07-22 can now attach these exact rebuild selectors to the fixed Phase 7 contract verifier and perform final phase validation. Live PostgreSQL/S3 qualification remains Phase 8 work.

## Self-Check: PASSED

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-11*
