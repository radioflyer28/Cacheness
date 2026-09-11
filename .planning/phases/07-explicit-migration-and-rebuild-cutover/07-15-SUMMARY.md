---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "15"
subsystem: storage-migration
tags: [rebuild, lifecycle-authority, recovery, receipts, projections, pytest]
requires:
  - phase: 07-explicit-migration-and-rebuild-cutover
    provides: Exact lifecycle-authority operation replay and projection-suppressed maintenance puts
provides:
  - Authenticated bounded rebuild receipt batches containing exact canonical BlobReceipt identities
  - Deterministic rebuild operation IDs replayed only through the existing lifecycle authority
  - Receipt-bound rebuild resume, verification, explicit acceptance, and cleanup debt
affects: [phase-7-migration-cutover, phase-8-qualification, migration-maintenance]
actuals:
  tokens: 13527
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - Rebuild evidence corroborates only exact projection-free canonical receipts; it never records a pre-publication intent
    - Rebuild cleanup uses a receipt's key, generation, locator, and expectation as its sole deletion authority
key-files:
  created: []
  modified:
    - src/cacheness/storage/migration_evidence.py
    - src/cacheness/storage/migration.py
    - tests/test_rebuild_workflow.py
key-decisions:
  - "Rebuild operation IDs are deterministically derived from bounded plan and entry identity and can address only an existing exact lifecycle-authority replay record."
  - "Maintenance evidence stores projection-free BlobReceipt ownership only; response-loss recovery never discovers or adopts a destination entry."
  - "Rebuild cleanup deletes only an exact expectation match, while ownership conflicts and delete failures remain bounded explicit debt."
patterns-established:
  - "Checkpoint each canonical rebuild receipt before reading a later source entry; resume validates the receipt against both the authority replay and current exact entry."
  - "Derived projections remain an explicit post-REBUILD_ACCEPTED operation and cannot alter accepted canonical receipts."
requirements-completed: [MIGR-05, MIGR-06]
coverage:
  - id: D1
    description: A process loss between canonical rebuild commit and maintenance checkpoint re-derives the same operation ID and checkpoints the exact replayed receipt without derived work.
    requirement: MIGR-05
    verification:
      - kind: integration
        ref: tests/test_rebuild_workflow.py#test_rebuild_response_loss_rederives_operation_id_and_replays_authority_receipt
        status: pass
    human_judgment: false
  - id: D2
    description: Projection-equipped rebuild staging and replay remain projection-free until one explicit post-acceptance rebuild.
    requirement: MIGR-06
    verification:
      - kind: integration
        ref: tests/test_rebuild_workflow.py#test_projection_equipped_rebuild_replays_canonical_receipt_without_preacceptance_or_duplicate_derived_work
        status: pass
    human_judgment: false
  - id: D3
    description: Every resumable rebuild state validates exact receipt ownership, while conflicts and cleanup errors become bounded receipt-specific debt.
    requirement: MIGR-05
    verification:
      - kind: integration
        ref: tests/test_rebuild_workflow.py#test_rebuild_checkpoints_exact_destination_receipts_and_resumes_each_rebuild_state
        status: pass
      - kind: integration
        ref: tests/test_rebuild_workflow.py#test_rebuild_verification_failure_cleans_only_exact_receipts_and_persists_debt
        status: pass
    human_judgment: false
duration: 35min
completed: 2026-09-11
status: complete
---

# Phase 07 Plan 15: Receipt-Bound Rebuild Recovery Summary

**Interrupted rebuilds now resume through deterministic lifecycle-authority replay, checkpoint exact projection-free receipts, and retain ownership conflicts or delete failures as bounded cleanup debt.**

## Performance

- **Duration:** 35min
- **Started:** 2026-09-11T16:14:47Z
- **Completed:** 2026-09-11T16:49:47Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added authenticated, bounded `RebuildReceiptBatch` evidence with exact `BlobReceipt` identity and retirement progress, without payload bytes, manifests, signing material, or a second intent layer.
- Rebuild staging deterministically derives an operation ID for each planned entry, invokes only the internal projection-suppressed maintenance put, and replays that authority-owned operation after response loss.
- Resume validates REBUILDING, REBUILD_STAGED, REBUILD_VERIFYING, and REBUILD_VERIFIED from exact authority records and receipt-matching entries; no destination listing or adoption path is used.
- Failure cleanup uses the exact receipt expectation for deletion, treating absence as retired and recording changed ownership or operational errors as `rebuild:<operation>:<key>:<generation>:<locator>` debt.

## Task Commits

1. **Task 1: Re-derive one rebuild operation and checkpoint its canonical replay receipt** - `dc48086` (`feat`)
2. **Task 2: Resume every rebuild state and clean only exact receipt ownership** - `90e3e61`, `511725c` (`test`)

## Files Created/Modified

- `src/cacheness/storage/migration_evidence.py` - Encodes, authenticates, and bounds exact rebuild receipt batches and retirement identifiers.
- `src/cacheness/storage/migration.py` - Replays deterministic operations through the existing lifecycle authority, validates receipts, resumes legal rebuild states, and retires only exact ownership.
- `tests/test_rebuild_workflow.py` - Exercises response-loss replay, derived-work suppression, resumable states, conflict preservation, and operational cleanup debt.

## Decisions Made

- Destination authority state remains the single lifecycle authority. Offline maintenance evidence corroborates only exact authority results and cannot discover or adopt destination state.
- The crash window after canonical commit and before receipt checkpoint remains explicit: resume retries only the deterministic authority operation; it does not inspect payload paths or listings.
- Projection rebuild remains the sole derived path and starts only after explicit `REBUILD_ACCEPTED` evidence.

## Deviations from Plan

None - plan executed within the existing lifecycle-authority and bounded-evidence design. Receipt persistence and receipt-bound failure cleanup were implemented together because the first durable receipt makes its cleanup obligation immediately attributable.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Rebuild recovery now has durable, authenticated per-entry attribution without introducing a journal, second authority, coordination mechanism, or payload backend dependency. Live PostgreSQL and S3 qualification remains Phase 8 work.

## Self-Check: PASSED

- Confirmed the summary exists and all task commits (`dc48086`, `90e3e61`, `511725c`) are present.
- Confirmed the plan-owned diff has no whitespace errors.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-11*
