---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "02"
status: superseded
subsystem: storage-lifecycle
tags: [blobstore, manifest, cas, json, sqlite, concurrency]
requires:
  - phase: 03-01
    provides: immutable generation publication and authenticated manifest authority
provides:
  - Exact-record CAS publication and retirement for admitted local manifest repositories
  - Cross-instance JSON compare/publish locking and SQLite conditional writer transactions
  - CAS-protected same-generation BlobStore metadata patches
affects: [03-04-delete-lifecycle, 03-05-clear-core, 03-08-concurrency, BlobStore]
actuals:
  tokens: 6206
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Opaque authenticated-record digest expectations at repository authority boundaries
    - Short JSON OS-lock refresh/compare/durable-publication critical sections
    - SQLite BEGIN IMMEDIATE exact-record conditional transactions
key-files:
  created:
    - tests/test_manifest_repository_cas.py
  modified:
    - src/cacheness/storage/manifest_repository.py
    - src/cacheness/storage/blob_store.py
    - tests/test_blob_store_read_contract.py
key-decisions:
  - "Generation and SHA-256 digest form one manifest expectation; adapters compare opaque bytes and never decode unauthenticated records."
  - "JSON instance-local locks are supplemented by a short adjacent POSIX lock plus durable refresh before conditional publication."
  - "BlobStore metadata patches publish re-signed user metadata only against the exact authenticated record that was observed."
patterns-established:
  - "Repository authority mutations use publish_if_expected or remove_if_expected rather than a read-then-unconditional-write sequence."
requirements-completed: [STOR-03, STOR-05, STOR-07]
coverage:
  - id: D1
    description: Exact-record create, replace, and conditional retirement have one local authority winner across memory, JSON, and SQLite repositories.
    requirement: STOR-03
    verification:
      - kind: integration
        ref: tests/test_manifest_repository_cas.py
        status: pass
    human_judgment: false
  - id: D2
    description: SQLite canonical bytes and compatibility projections roll back together when conditional publication fails.
    requirement: STOR-05
    verification:
      - kind: integration
        ref: tests/test_manifest_repository_cas.py#test_sqlite_cas_rolls_back_compatibility_projection_with_raw_record
        status: pass
    human_judgment: false
  - id: D3
    description: Competing same-generation BlobStore metadata patches retain one signed winner and surface a typed lifecycle conflict to the stale caller.
    requirement: STOR-07
    verification:
      - kind: integration
        ref: tests/test_blob_store_read_contract.py#test_update_metadata_rejects_a_stale_independent_store_patch
        status: pass
    human_judgment: false
duration: 7 min
completed: 2026-08-30
status: complete
---

# Phase 03 Plan 02: Exact Manifest CAS Summary

**Local BlobStore manifest repositories now enforce exact-record authority changes, and metadata patches can no longer overwrite an unseen same-generation update.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-08-30T18:32:00Z
- **Completed:** 2026-08-30T18:39:23Z
- **Tasks:** 2/2
- **Files modified:** 4

## Accomplishments

- Added validated `ManifestExpectation` tokens and atomic conditional retirement to memory, JSON, and SQLite manifest repositories.
- Made JSON CAS refresh its durable document under a narrow OS-backed lock and made SQLite acquire its writer transaction before comparison.
- Routed `BlobStore.update_metadata()` through authenticated exact-record CAS, with deterministic independent-store stale-patch coverage.

## Task Commits

1. **Task 1: Implement exact-record CAS for every admitted local repository**
   - `8fb9aec` `test(03-02): add failing exact manifest CAS coverage`
   - `edf3cff` `feat(03-02): enforce exact local manifest CAS`
2. **Task 2: Route metadata patching and lifecycle retirement through exact CAS**
   - `9230e28` `test(03-02): add stale metadata patch race coverage`
   - `48b306f` `feat(03-02): guard BlobStore metadata patches with CAS`

## Files Created/Modified

- `src/cacheness/storage/manifest_repository.py` — exact conditional create, replace, and remove operations for local raw manifest stores.
- `src/cacheness/storage/blob_store.py` — authenticates the observed record before conditionally publishing a metadata patch.
- `tests/test_manifest_repository_cas.py` — independent-instance winner, stale-record, conditional-removal, and transaction-rollback coverage.
- `tests/test_blob_store_read_contract.py` — deterministic stale same-generation metadata-patch race coverage.

## Decisions Made

- Exact raw canonical bytes, not a generation string alone, are the precondition for a same-generation metadata update or retirement.
- Repository adapters transport and compare opaque records; manifest authentication stays in the BlobStore lifecycle boundary.
- JSON only holds its OS-backed critical section for metadata refresh, comparison, and durable publication; it never covers payload I/O.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The separately tracked legacy clear-recovery locator compatibility failure remains assigned to Plan 03-06 and was not expanded into this CAS plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 03-03 can layer bounded lifecycle evidence and policy limits on truthful local authority operations.
- Plan 03-04 can consume conditional retirement when it introduces signed tombstone deletion semantics.

## Self-Check: PASSED

- Verified all four code/test artifacts and the Summary exist on disk.
- Verified task commits `8fb9aec`, `edf3cff`, `9230e28`, and `48b306f` exist in git history.
