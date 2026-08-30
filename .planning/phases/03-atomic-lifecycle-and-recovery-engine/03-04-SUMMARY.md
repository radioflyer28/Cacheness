---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "04"
subsystem: storage-lifecycle
tags: [blobstore, immutable-generations, cas, tombstone, recovery]
requires:
  - phase: 03-02
    provides: exact-record manifest publication and conditional retirement
  - phase: 03-03
    provides: bounded authenticated lifecycle evidence
provides:
  - Immutable create and overwrite cleanup that cannot delete a CAS winner
  - Signed tombstone-first deletion with exact conditional retirement and resume
affects: [03-05-clear-core, 03-06-clear-compatibility, 03-07-reconciliation, 03-08-concurrency]
actuals:
  tokens: 4205
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - CAS losers reclaim only their operation-bound immutable candidate
    - Delete authority is a signed tombstone, retired only with its exact record
key-files:
  created: []
  modified:
    - src/cacheness/storage/lifecycle.py
    - src/cacheness/storage/blob_store.py
    - tests/test_blob_store_atomic_lifecycle.py
key-decisions:
  - "A stale write cleans only its evidence-bound candidate before surfacing the exact CAS conflict."
  - "Delete retains the old payload identity in an authenticated tombstone and never removes a newer manifest during finalization."
patterns-established:
  - "Tombstone recovery authenticates both evidence and the current exact tombstone before payload reclamation."
requirements-completed: [STOR-03, STOR-04, STOR-05, STOR-07]
coverage:
  - id: D1
    description: Create and overwrite publish immutable generations, preserve CAS winners, and reclaim only stale contenders' owned residue.
    requirement: STOR-03
    verification:
      - kind: integration
        ref: tests/test_blob_store_atomic_lifecycle.py#test_stale_overwrite_conflict_reclaims_only_loser_candidate
        status: pass
      - kind: integration
        ref: tests/test_blob_store_atomic_lifecycle.py#test_lifecycle_failure_boundary_reopens_to_one_complete_generation
        status: pass
    human_judgment: false
  - id: D2
    description: Delete publishes signed tombstone authority before reclamation, resumes repeated cleanup, and preserves newer winners on stale CAS.
    requirement: STOR-05
    verification:
      - kind: integration
        ref: tests/test_blob_store_atomic_lifecycle.py#test_delete_publishes_signed_tombstone_before_payload_reclamation
        status: pass
      - kind: integration
        ref: tests/test_blob_store_atomic_lifecycle.py#test_repeated_delete_resumes_the_same_signed_tombstone
        status: pass
      - kind: integration
        ref: tests/test_blob_store_atomic_lifecycle.py#test_stale_delete_conflict_preserves_newer_committed_generation
        status: pass
    human_judgment: false
duration: 8 min
completed: 2026-08-30
status: complete
---

# Phase 03 Plan 04: Single-Key Atomic Lifecycle Summary

**BlobStore now publishes immutable create/overwrite generations through exact CAS and deletes through signed tombstones that can safely resume after interruption.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-08-30T19:06:44Z
- **Completed:** 2026-08-30T19:15:31Z
- **Tasks:** 2/2
- **Files modified:** 3

## Accomplishments

- Stale overwrite contenders retain the CAS winner and clean only their own operation-bound candidate.
- `BlobStore.delete()` now records authenticated intent, conditionally publishes a signed tombstone, then reclaims payload bytes and retires the exact tombstone.
- Repeated deletes resume a matching tombstone; true absence remains the compatible `False` result, and stale delete CAS failures preserve newer committed generations.

## Task Commits

1. **Task 1: Complete create and overwrite fault convergence**
   - `1b1f14a` `test(03-04): add stale overwrite conflict coverage`
   - `1af917b` `feat(03-04): reclaim stale overwrite candidates`
2. **Task 2: Publish tombstones before delete reclamation and converge repeated calls**
   - `9e01609` `test(03-04): add tombstone-first deletion coverage`
   - `9a1deb9` `feat(03-04): publish tombstones before blob deletion`

## Files Created/Modified

- `src/cacheness/storage/lifecycle.py` — completes stale-writer cleanup and signed tombstone publication, finalization, and authenticated recovery.
- `src/cacheness/storage/blob_store.py` — routes direct deletion through the lifecycle engine while retaining committed-only defaults for ordinary reads.
- `tests/test_blob_store_atomic_lifecycle.py` — covers stale overwrite/delete CAS, tombstone-before-reclamation, repeat-delete resume, and fresh-process recovery.

## Decisions Made

- Successful `publish_if_expected()` remains the sole authority transition; a losing writer never touches the previous or winning locator.
- Tombstone recovery requires an authenticated operation record plus a matching authenticated exact tombstone; filenames and payload contents never establish deletion authority.
- Ordinary direct delete no longer takes the predecessor's global clear admission lock; its repository CAS remains the cross-instance boundary.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `uv run pytest -q -o log_cli=false tests/test_blob_store_read_contract.py -k "delete" -x` fails in the pre-existing `test_delete_and_clear_preflight_authenticated_manifests_before_mutation`. Its generic `file_ops.delete` monkeypatch intentionally intercepts unsafe payload deletion but also intercepts later lifecycle-evidence retirement (`delete_durable -> delete`) during setup puts. This Plan did not modify that Phase 2 test file; the in-scope delete regressions excluding that fixture pass. Plan 03-08 owns its correction, with Plan 03-10 as consolidated closure.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 03-05 can use the same tombstone engine for bounded clear entries without serializing unrelated normal mutations.
- Plan 03-06 retains ownership of legacy clear-recovery compatibility, including its locator grammar and error translation.
- Plan 03-08 should narrow the existing payload-delete monkeypatch so it does not intercept lifecycle-evidence retirement.

## Self-Check: PASSED

- Verified `src/cacheness/storage/lifecycle.py`, `src/cacheness/storage/blob_store.py`, and `tests/test_blob_store_atomic_lifecycle.py` exist.
- Verified task commits `1b1f14a`, `1af917b`, `9e01609`, and `9a1deb9` exist in git history.
