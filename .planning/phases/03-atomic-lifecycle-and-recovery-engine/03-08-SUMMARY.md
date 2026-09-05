---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "08"
subsystem: storage-lifecycle
tags: [blobstore, lifecycle-authority, sqlite, json-projection, integrity]
requires:
  - phase: 03-07
    provides: Retired file-native scheduler and authority-only lifecycle foundation
provides:
  - Process-local coordination and payload-safe retained filesystem primitives
  - Projection-only manifest compatibility and exclusive BlobStore authority composition
affects: [phase-03-plans-09-10, BlobStore, lifecycle-authority]
actuals:
  tokens: 69305
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - LifecycleAuthority is the sole canonical state source; JSON is revision-bound derived output
    - Local coordination provides only per-key ordering and instance admission/close ownership
key-files:
  created: []
  modified:
    - src/cacheness/storage/coordination.py
    - src/cacheness/storage/path_security.py
    - src/cacheness/storage/manifest_repository.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/reconciliation.py
key-decisions:
  - "Retain immutable payload and path-safety primitives while deleting scheduler and platform lock authority."
  - "Keep JSON solely as an authority-snapshot projection; direct BlobStore operations never read it as canonical state."
  - "Select one LifecycleAuthority at construction and dispatch every lifecycle operation through its single engine."
patterns-established:
  - "Retirement gates inspect source symbols and branches as well as exercising runtime behavior."
  - "Obsolete private repository/admission test seams are replaced with authority-level contracts."
requirements-completed: [STOR-03, STOR-04, STOR-05, STOR-06, STOR-07]
coverage:
  - id: D1
    description: Scheduler-free local coordination and preserved payload safety
    requirement: STOR-03
    verification:
      - kind: unit
        ref: tests/test_blob_store_close_contract.py, tests/test_filesystem_containment.py, tests/test_phase3_scheduler_retirement.py
        status: pass
    human_judgment: false
  - id: D2
    description: Projection-only manifests and exclusive LifecycleAuthority runtime composition
    requirement: STOR-04
    verification:
      - kind: unit
        ref: tests/test_manifest_repository_cas.py, tests/test_blob_store_read_contract.py, tests/test_phase3_scheduler_retirement.py
        status: pass
    human_judgment: false
duration: 35min
completed: 2026-09-05
status: complete
---

# Phase 03 Plan 08: Seal One-Authority Composition Summary

**BlobStore now delegates every lifecycle operation to one transactional authority while JSON remains a revision-bound compatibility projection.**

## Performance

- **Duration:** 35min
- **Completed:** 2026-09-05
- **Tasks:** 2/2
- **Files modified:** 11
- **Verification:** focused Task 1 gate 101 passed, 3 skipped; focused Task 2 gate 59 passed; independent full suite passed; Phase 3 Ruff delta clean.

## Accomplishments

- Reduced coordination to bounded in-process per-key ordering and instance admission/close ownership; removed scheduler control, file/store lock authority, and Windows control branches.
- Preserved containment, no-follow/wrong-object checks, immutable publication/snapshot behavior, fsync, digest/size verification, and durable exact deletion.
- Replaced legacy manifest repositories with revision-aware JSON projection export and removed BlobStore's alternate manifest read/write/recovery branches.
- Converted affected tests to assert the authority boundary directly, including negative source gates for retired symbols and Windows-only lock branches.

## Task Commits

1. **Task 1: Trim lock/control helpers while preserving payload safety**
   - `881c35a` — `test(03-08): add retained helper retirement gate`
   - `91f1024` — `feat(03-08): trim retained lifecycle helpers`
2. **Task 2: Reduce manifest compatibility and seal one-authority composition**
   - `03a5d55` — `test(03-08): add exclusive authority projection gate`
   - `8a3040e` — `feat(03-08): seal one-authority composition`

## Decisions Made

- Canonical lifecycle state is available only from the selected `LifecycleAuthority`; a corrupt, stale, or missing JSON projection cannot answer a BlobStore operation or rebuild authority state.
- Retained signing-key initialization guards stay private to integrity key material and do not reintroduce BlobStore lifecycle coordination authority.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking test migration] Retired test imports and private seams prevented suite collection.**

- **Found during:** Task 2
- **Issue:** Existing repository/admission tests imported removed manifest repositories and `StoreAdmissionBarrier`, while one read-contract test monkeypatched an erased legacy manifest helper.
- **Fix:** Replaced those checks with authority/projection contracts and redirected retained key-lock tests to the private integrity guard.
- **Files modified:** `tests/test_blob_manifest_backends.py`, `tests/test_blob_store_concurrency.py`, `tests/test_blob_store_integrity.py`, `tests/test_blob_store_read_contract.py`
- **Verification:** Independent full suite passed.
- **Committed in:** `8a3040e`

**Total deviations:** 1 auto-fixed (Rule 3).

## Issues Encountered

- The sandbox initially denied uv cache access to the repository Ruff quality gate. The independent full-suite runner completed successfully with its existing cache access.
- Python shutdown emitted a non-failing `SqliteBackend.__del__` warning during one test invocation; it is already tracked as deferred and did not affect suite success.

## Known Stubs

None.

## Next Phase Readiness

The scheduler-free authority boundary is ready for the remaining Phase 03 validation and platform work. No migration or compatibility fallback remains in normal BlobStore operations.

## Self-Check: PASSED

- Task commits `881c35a`, `91f1024`, `03a5d55`, and `8a3040e` exist in history.
- Authority composition, projection compatibility, containment, and close contracts are present in the committed files.
