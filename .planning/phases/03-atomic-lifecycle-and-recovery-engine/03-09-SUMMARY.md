---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "09"
subsystem: storage-lifecycle
tags: [blobstore, close, admission, ownership, concurrency, lifecycle]
requires:
  - phase: 03-08
    provides: per-instance coordination and committed read admission seams
provides:
  - Instance-scoped OPEN/CLOSING/CLOSED admission with a configurable close drain
  - Ownership-aware exactly-once BlobStore resource release and retryable close failures
  - Typed post-close and close-timeout outcomes through storage error exports
affects: [03-10-verification, BlobStore, lifecycle, resource-ownership]
actuals:
  tokens: 6596
  tasks: 2
  commits: 5
tech-stack:
  added: []
  patterns:
    - Instance admission protects only one BlobStore's resources while normal work retains per-key concurrency
    - Closing remains retryable until every owned resource release succeeds
key-files:
  created:
    - tests/test_blob_store_close_contract.py
  modified:
    - src/cacheness/storage/coordination.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/__init__.py
    - src/cacheness/error_handling.py
key-decisions:
  - "A close waiter rechecks CLOSED after another closer releases resources, so terminal close never repeats an owned release sequence."
  - "Operation evidence is already durably persisted per mutation; close flushes a future repository hook before releasing the shared guarded descriptor."
patterns-established:
  - "InstanceAdmission keeps CLOSING live after a timeout or partial release failure, rejects new work, and lets later close calls converge."
requirements-completed: [STOR-05, STOR-07]
coverage:
  - id: D1
    description: "Close rejects post-close calls before resource access, drains admitted work to a caller-owned deadline, and detects reentrant self-waits."
    requirement: STOR-05
    verification:
      - kind: integration
        ref: "tests/test_blob_store_close_contract.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "Owned resources release once, injected backends remain open, partial release retries safely, and concurrent close waiters do not repeat release."
    requirement: STOR-07
    verification:
      - kind: integration
        ref: "tests/test_blob_store_close_contract.py#test_concurrent_close_waiter_does_not_repeat_owned_release"
        status: pass
      - kind: integration
        ref: "tests/test_blob_store_read_contract.py -k 'close or constructor'"
        status: pass
    human_judgment: false
duration: 7h 28m
completed: 2026-08-31
status: complete
---

# Phase 03 Plan 09: Close Admission and Ownership Summary

**BlobStore now drains admitted instance work through one bounded close state machine, then releases only its owned resources exactly once without clearing persisted data.**

## Performance

- **Duration:** 7h 28m
- **Started:** 2026-08-31T04:31:24Z
- **Completed:** 2026-08-31T11:59:14Z
- **Tasks:** 2/2
- **Files modified:** 5

## Accomplishments

- Added `InstanceAdmission` with OPEN/CLOSING/CLOSED state, condition-based in-flight draining, caller-owned `LifecycleLimits`, and stable typed close outcomes.
- Routed direct BlobStore entry points through instance admission, preserving the existing short clear barrier and per-key normal-operation coordination.
- Released only owned guarded I/O and metadata backends, preserved caller-injected backends and stored data, and made partial releases retryable without duplicate closes.
- Added deterministic event/clock close contracts, including a concurrent-close regression that proves a waiting closer never repeats repository flush or resource release.

## Task Commits

1. **Task 1: Reject new operations and drain admitted work deterministically**
   - `ad526cd` `test(03-09): add failing close lifecycle contracts`
   - `9ad0126` `feat(03-09): drain admitted BlobStore work before close`
2. **Task 2: Release only owned resources exactly once without clearing data**
   - `ea45ff7` `feat(03-09): release only owned BlobStore resources`

## Files Created/Modified

- `src/cacheness/storage/coordination.py` - owns the instance admission state, bounded drain, timeout, reentrant detection, and terminal close coordination.
- `src/cacheness/storage/blob_store.py` - admits direct operations, owns release ordering, and delegates context-manager exit to the same close path.
- `src/cacheness/error_handling.py` and `src/cacheness/storage/__init__.py` - expose stable closed and close-timeout error types.
- `tests/test_blob_store_close_contract.py` - proves admission, injected clock boundaries, ownership, partial release, data persistence, and concurrent close behavior.

## Decisions Made

- `CLOSED` is a terminal observation for concurrent close waiters; after waiting for another release attempt, they return without repeating flush or close calls.
- Close keeps resources live in `CLOSING` after timeout or partial release failure, so an explicit later close retries only outstanding owned work.
- Direct operation evidence writes are already durable; close invokes an optional future repository flush before closing the shared guarded descriptor and never clears or reconciles user data.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Concurrency bug] Prevented concurrent close waiters from repeating owned release**
- **Found during:** Task 2 final closeout audit
- **Issue:** A second closer could wake after the first reached `CLOSED`, then re-enter `_release_owned_resources()` and invoke an optional operation-repository flush again.
- **Fix:** `InstanceAdmission.begin_close()` now rechecks terminal `CLOSED` after waiting; added an event-driven two-closer regression contract.
- **Files modified:** `src/cacheness/storage/coordination.py`, `tests/test_blob_store_close_contract.py`
- **Verification:** Full close-contract suite, targeted read-contract suite, and targeted Ruff all pass.
- **Committed in:** `94a11e4`, `64e8a18`

---

**Total deviations:** 1 auto-fixed (1 Rule 1).
**Impact on plan:** The fix closes an exactly-once ownership race without broadening resource lifetime or serializing ordinary operations.

## Issues Encountered

- Carry forward the known Phase 03 final-closure item: `tests/test_blob_store_integrity.py` expects one manifest-authentication call, while the committed read contract deliberately validates M1 before and M2 after the private snapshot. Plan 03-10 owns reconciling that stale expectation; it is unrelated to close admission and remains unmodified here.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 03-10 can run the consolidated lifecycle matrix with close admission and ownership now covered by deterministic contracts.
- The only carried closure item is the deliberate pre/post-authentication expectation mismatch in the integrity test; it needs Phase 03-wide resolution, not a close-path change.

## Self-Check: PASSED

- Verified all five plan-owned source/test artifacts and this Summary exist in the working tree.
- Verified task and regression commits `ad526cd`, `9ad0126`, `ea45ff7`, `94a11e4`, and `64e8a18` exist in git history.

---
*Phase: 03-atomic-lifecycle-and-recovery-engine*
*Completed: 2026-08-31*
