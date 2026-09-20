---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "06"
status: superseded
subsystem: storage-lifecycle
tags: [blobstore, clear, legacy-compatibility, lifecycle, recovery]
requires:
  - phase: 03-05
    provides: authenticated bounded clear targets and StoreAdmissionBarrier
provides:
  - Reopen-only handling for exact predecessor clear evidence
  - One current LifecycleEngine authority for every new BlobStore clear
affects: [03-07-reconciliation, 03-08-concurrency, 03-10-verification]
actuals:
  tokens: 7143
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Legacy control evidence is an explicit reopen-only compatibility input
    - New clear requests use authenticated Phase 3 operation records exclusively
key-files:
  created: []
  modified:
    - src/cacheness/storage/clear_recovery.py
    - src/cacheness/storage/blob_store.py
    - tests/test_clear_recovery.py
    - tests/test_blob_store_atomic_lifecycle.py
key-decisions:
  - "LegacyClearEvidenceAdapter exposes recovery only; it cannot create or publish a predecessor journal."
  - "BlobStore ordinary admission and JSON refresh do not acquire predecessor global clear admission."
requirements-completed: [STOR-04, STOR-05]
coverage:
  - id: D1
    description: Exact predecessor prepared evidence is recoverable through a reopen-only adapter, while malformed bytes are retained without destructive callbacks.
    requirement: STOR-04
    verification:
      - kind: integration
        ref: tests/test_clear_recovery.py -k "legacy or prepared or committed or malformed or topology or provenance"
        status: pass
    human_judgment: false
  - id: D2
    description: New BlobStore clears use only LifecycleEngine evidence and ordinary operations do not enter predecessor admission wrappers.
    requirement: STOR-05
    verification:
      - kind: integration
        ref: tests/test_clear_recovery.py tests/test_blob_store_atomic_lifecycle.py -k "clear and (legacy or journal or authority or admission or reopen)"
        status: pass
    human_judgment: false
duration: 8 min
completed: 2026-08-31
status: complete
---

# Phase 03 Plan 06: Legacy Clear Compatibility Boundary Summary

**Predecessor clear journals now reopen only through a narrow compatibility adapter, while every new BlobStore clear remains a signed Phase 3 LifecycleEngine operation.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-08-30T23:38:01-04:00
- **Completed:** 2026-08-31T03:45:45Z
- **Tasks:** 2/2
- **Files modified:** 4

## Accomplishments

- Added `LegacyClearEvidenceAdapter`, a recovery-only compatibility seam with no new-journal API.
- Kept strict legacy topology, field, path, and byte-bound validation before predecessor recovery can mutate payloads or metadata.
- Removed predecessor global mutation/read wrappers from BlobStore normal operations; fresh clears continue exclusively through `LifecycleEngine.clear()` and `StoreAdmissionBarrier`.
- Updated legacy recovery coverage so predecessor coordinator tests invoke its explicit frozen primitive, while direct BlobStore tests assert Phase 3 authority and immutable-generation semantics.

## Task Commits

1. **Task 1: Adapt exact predecessor clear evidence without acquiring new authority**
   - `c6c8bae` `test(03-06): add legacy clear adapter coverage`
   - `0792cbe` `feat(03-06): add reopen-only legacy clear adapter`
2. **Task 2: Route new clears exclusively through the Phase 3 engine**
   - `511bf5f` `test(03-06): add current clear authority coverage`
   - `6dadd5f` `feat(03-06): isolate legacy clear recovery from new clears`

## Files Created/Modified

- `src/cacheness/storage/clear_recovery.py` — adds the recovery-only legacy adapter and recognizes the frozen immutable-generation locator variant.
- `src/cacheness/storage/blob_store.py` — confines legacy evidence discovery to reopen and keeps ordinary work outside predecessor admission.
- `tests/test_clear_recovery.py` — separates frozen-coordinator compatibility tests from current lifecycle behavior.
- `tests/test_blob_store_atomic_lifecycle.py` — proves a new clear never invokes predecessor coordination.

## Decisions Made

- Legacy evidence may only be consumed at reopen; it is not read by normal operations and cannot start a new clear.
- Current lifecycle CAS and authenticated operation evidence remain the sole authority for new BlobStore clears.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Compatibility regression] Rebased predecessor recovery tests on immutable generation paths**
- **Found during:** Task 1 verification
- **Issue:** Phase 1 tests assumed candidate-form payload paths and injected faults into the retired public clear flow.
- **Fix:** Kept the predecessor coordinator test-only and explicit, while testing direct BlobStore interruption through the current lifecycle seams.
- **Files modified:** `tests/test_clear_recovery.py`, `src/cacheness/storage/clear_recovery.py`
- **Verification:** Both targeted legacy and authority test matrices pass.
- **Committed in:** `0792cbe`, `6dadd5f`

**2. [Rule 1 - Import failure] Preserved decorator support for ordinary admission**
- **Found during:** Task 2 verification
- **Issue:** Removing the retired predecessor wrappers also removed the `wraps` import still required by `_ordinary_admitted`.
- **Fix:** Restored the focused import.
- **Files modified:** `src/cacheness/storage/blob_store.py`
- **Verification:** Targeted lifecycle authority matrix and Ruff pass.
- **Committed in:** `6dadd5f`

**Total deviations:** 2 auto-fixed (2 Rule 1).

## Issues Encountered

- The broad historical clear-recovery file still contains additional Phase 1 contention tests that model the predecessor coordinator as normal BlobStore authority. The plan-required compatibility and direct-authority matrices pass; Plan 03-08 owns the wider concurrent-operation contract and should consolidate the remaining obsolete contention expectations.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 03-07 can inspect legacy and current evidence without letting either weak or malformed predecessor bytes authorize repair.
- Plan 03-08 can complete the per-key concurrent-operation model now that the predecessor global admission wrappers are absent from BlobStore operations.

## Self-Check: PASSED

- Verified all four plan-owned code and test files exist.
- Verified commits `c6c8bae`, `0792cbe`, `511bf5f`, and `6dadd5f` exist in git history.
