---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "01"
subsystem: storage-lifecycle
tags: [blobstore, immutable-generations, cas, recovery, hmac, json]
requires:
  - phase: 02-canonical-storage-and-integrity-contract
    provides: signed canonical BlobManifestV1 records and authenticated reads
provides:
  - Immutable-generation BlobStore publication through exact manifest CAS
  - Authenticated operation evidence and reopen recovery for incomplete writes
  - Deterministic fault-boundary coverage for ordinary failures and process loss
affects: [03-02-manifest-cas, 03-04-delete-and-clear, 03-06-clear-compatibility, BlobStore]
actuals:
  tokens: 42974
  tasks: 2
  commits: 5
tech-stack:
  added: []
  patterns:
    - Native handler staging before durable lifecycle evidence
    - Authenticated operation record recovery driven by manifest authority
    - Exact CAS as the sole BlobStore publication authority point
key-files:
  created:
    - src/cacheness/storage/operation_record.py
    - src/cacheness/storage/operation_repository.py
    - src/cacheness/storage/lifecycle.py
  modified:
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/manifest_repository.py
    - src/cacheness/storage/guarded_handler_io.py
    - src/cacheness/storage/path_security.py
    - tests/test_blob_store_atomic_lifecycle.py
key-decisions:
  - "BlobStore writes create authenticated operation evidence before publishing native handler bytes to an immutable generation locator."
  - "Recovery trusts only an authenticated manifest and a validated, authenticated operation record; filenames never establish authority."
  - "JSON lifecycle writes refresh the live metadata view without reinstating Phase 1's global normal-operation admission lock."
patterns-established:
  - "LifecycleEngine owns direct BlobStore publication, checkpoints, and reopen convergence."
  - "Post-authority cleanup failures surface typed debt while retaining the committed winner."
requirements-completed: [STOR-03, STOR-04, STOR-05]
coverage:
  - id: D1
    description: Immutable JSON generation publication through private native serialization, durable evidence, exact CAS, and direct native reads.
    requirement: STOR-03
    verification:
      - kind: integration
        ref: tests/test_blob_store_atomic_lifecycle.py#test_tracer_json_put_uses_immutable_generation_cas_and_native_bytes
        status: pass
    human_judgment: false
  - id: D2
    description: Ordinary-exception and process-loss boundary recovery preserves either the old or the new complete generation.
    requirement: STOR-04
    verification:
      - kind: integration
        ref: tests/test_blob_store_atomic_lifecycle.py#test_lifecycle_failure_boundary_reopens_to_one_complete_generation
        status: pass
    human_judgment: false
  - id: D3
    description: Post-authority payload cleanup debt remains readable and resumes safely after reopening.
    requirement: STOR-05
    verification:
      - kind: integration
        ref: tests/test_blob_store_atomic_lifecycle.py#test_cleanup_failure_keeps_new_authority_and_resumes_after_reopen
        status: pass
    human_judgment: false
duration: 18min
completed: 2026-08-30
status: complete
---

# Phase 03 Plan 01: Atomic Lifecycle Tracer Summary

**BlobStore now publishes native handler payloads as immutable generations through authenticated operation evidence and exact manifest CAS, with reopen-safe cleanup recovery.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-08-30T18:13:33Z
- **Completed:** 2026-08-30T18:31:00Z
- **Tasks:** 2/2
- **Files modified:** 9

## Accomplishments

- Split private handler serialization from contained, exclusive immutable-generation publication; payload bytes remain native and unframed.
- Added bounded HMAC-authenticated operation records, exact manifest expectations, and LifecycleEngine as the direct BlobStore write coordinator.
- Added fault and process-loss reopen coverage that converges pre-authority residue and post-authority cleanup debt without revoking a winner.
- Preserved JSON fail-closed metadata refresh for lifecycle puts without reintroducing a global normal-operation lock.

## Task Commits

1. **Task 1: Publish and read one immutable generation through exact CAS**
   - `6296975` `test(03-01): add failing immutable lifecycle tracer`
   - `571dfd6` `feat(03-01): publish BlobStore generations through CAS`
2. **Task 2: Classify every tracer failure as recoverable pre- or post-authority state**
   - `89e12ec` `test(03-01): add failing lifecycle recovery matrix`
   - `92441b8` `feat(03-01): recover authenticated lifecycle debt`
   - `8d2e514` `fix(03-01): refresh JSON metadata before lifecycle put`

## Files Created/Modified

- `src/cacheness/storage/lifecycle.py` - coordinates evidence, immutable publication, CAS authority, checkpointing, and reopen recovery.
- `src/cacheness/storage/operation_record.py` - defines bounded signed lifecycle evidence and monotonic checkpoints.
- `src/cacheness/storage/operation_repository.py` - provides contained exact-byte evidence persistence and authenticated-recovery discovery.
- `src/cacheness/storage/manifest_repository.py` - adds exact raw-record manifest expectations and CAS publication.
- `src/cacheness/storage/guarded_handler_io.py` and `path_security.py` - retain private native staging and create immutable files with durability acknowledgements.
- `src/cacheness/storage/blob_store.py` - routes direct writes through LifecycleEngine and performs a non-locking JSON metadata refresh.
- `tests/test_blob_store_atomic_lifecycle.py` - exercises tracer, failure, crash/reopen, and resumable-cleanup behavior.

## Decisions Made

- A successful `publish_if_expected()` is the only authority transition; recovery compares authenticated manifest state rather than inferring authority from filenames or payload bytes.
- Unauthenticated, malformed, wrong-store, locator-invalid, or noncanonical operation evidence is ignored and never becomes a deletion target.
- Cleanup debt is typed only after authority has published; the committed generation remains readable while recovery retries owned cleanup.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Regression] Restore fail-closed JSON metadata refresh for lifecycle puts**
- **Found during:** Task 2 verification
- **Issue:** Removing Phase 1's global admission wrapper also removed the live JSON refresh that prevents a write from treating malformed metadata as a fresh empty document.
- **Fix:** Added a non-locking lifecycle metadata refresh before `BlobStore.put()` and translated refresh failure to the existing typed backend error.
- **Files modified:** `src/cacheness/storage/blob_store.py`
- **Verification:** `tests/test_blob_store_read_contract.py -k "admission_refresh_failures and put"`
- **Committed in:** `8d2e514`

---

**Total deviations:** 1 auto-fixed (1 regression)
**Impact on plan:** The fix preserves the intended no-global-lock architecture while retaining the Phase 2 fail-closed read/write contract.

## Issues Encountered

- The legacy clear-recovery parser accepts only its historical direct/candidate filename grammar, so its broad regression suite currently rejects the new generation-qualified locator. Extending that compatibility parser is intentionally deferred to Plan 03-06; this plan's tracer, targeted Phase 2 read-contract checks, and lint gate pass.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 03-02 can strengthen local repository CAS and conflict behavior using the established operation-record and immutable-generation seams.
- Plan 03-06 must update or replace the legacy clear-recovery locator grammar before the full historical clear suite can be green with the new lifecycle contract.

## Self-Check: PASSED

- Verified all nine plan files exist in the working tree and all five task commits are present in git history.

---
*Phase: 03-atomic-lifecycle-and-recovery-engine*
*Completed: 2026-08-30*
