---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "07"
subsystem: storage-lifecycle
tags: [blobstore, lifecycle-authority, retirement, reconciliation]
requires:
  - 03-01
  - 03-06
provides:
  - physically retired file-native scheduler modules
  - authority-only reconciliation and lifecycle tests
  - AST, import, barrel, and runtime retirement denial gates
affects: [BlobStore, UnifiedCache, reconciliation, public-storage-api]
tech-stack:
  added: []
  patterns: [exact-name-type sentinel, authority-only lifecycle, negative reachability]
key-files:
  created: [tests/test_phase3_scheduler_retirement.py]
  modified:
    - src/cacheness/core.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/reconciliation.py
    - tests/test_blob_store_atomic_lifecycle.py
    - tests/test_blob_store_close_contract.py
    - tests/test_blob_store_read_contract.py
    - tests/test_manifest_repository_cas.py
  deleted:
    - src/cacheness/storage/operation_repository.py
    - src/cacheness/storage/operation_record.py
    - src/cacheness/storage/clear_recovery.py
decisions:
  - Retired development controls are classified only by exact name and filesystem type, then fail typed before authority bootstrap or mutation.
  - Reconciliation and lifecycle recovery use LifecycleAuthority exclusively; the superseded file-native reconciler is removed rather than retained behind a compatibility path.
metrics:
  duration: 27m
  completed: 2026-09-05
status: complete
actuals:
  tokens: 146891
  tasks: 2
  commits: 4
---

# Phase 03 Plan 07: Retire File-Native Scheduler Summary

The abandoned file-native scheduler is physically removed; BlobStore now fails known development controls closed before authority initialization and exposes only LifecycleAuthority recovery.

## Tasks Completed

1. Deleted the operation-record, repository, and clear-journal modules after confirming Plan 01's no-release evidence. Replaced their private tests with BlobStore and LifecycleAuthority lifecycle behavior plus a byte-preserving rebuild sentinel.
2. Removed residual reachability, including the dead pre-authority reconciler and private-test seams. Added production AST/import, barrel, importlib, and fresh/reopen runtime denial coverage.

## Verification

- `.venv/bin/pytest -q tests/test_public_api_contract.py tests/test_phase3_scheduler_retirement.py -x` — 12 passed
- `.venv/bin/python tools/verify_phase3_ruff_delta.py` — no unmatched findings
- `.venv/bin/pytest -q tests/test_clear_recovery.py tests/test_phase3_scheduler_retirement.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_reconciliation.py -x` — 17 passed
- `.venv/bin/pytest -q -o log_cli=false` — 749 passed, 28 skipped

## Decisions Made

- The bounded sentinel may inspect only exact control names and expected node types. It returns `blob_migration_required` without parsing bytes, creating directories, deleting evidence, or opening LifecycleAuthority.
- Runtime lifecycle repair and reconciliation are authority-only. No scheduler parser, replay adapter, compatibility wrapper, factory, barrel export, or fallback remains.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] Removed stale scheduler-era reconciliation and test dependencies**
- **Found during:** Task 2 verification
- **Issue:** Physical module retirement left an unreachable `_Reconciler` implementation and private test bodies that referenced the deleted types, producing stale construction paths and Ruff undefined-name errors.
- **Fix:** Replaced reconciliation with its authority-only implementation and reduced the affected test suites to public BlobStore/LifecycleAuthority behavior.
- **Files modified:** `src/cacheness/storage/reconciliation.py`, `tests/test_blob_store_atomic_lifecycle.py`, `tests/test_blob_store_close_contract.py`, `tests/test_manifest_repository_cas.py`
- **Commit:** ae71c12

**2. [Rule 1 - Bug] Restored UnifiedCache custom-query context management**
- **Found during:** Full-suite verification
- **Issue:** Scheduler import cleanup had removed the standard-library `contextmanager` import and left one custom-query session path calling the retired read-admission helper.
- **Fix:** Restored the stdlib import and made the session context directly manage its SQLAlchemy session without the retired lifecycle coordination path.
- **Files modified:** `src/cacheness/core.py`
- **Commit:** ae71c12

## Deferred Issues

- `SqliteBackend.__del__` can log an interpreter-shutdown `ImportError` after a successful suite. It is pre-existing/out of scope for this retirement plan and did not affect test outcomes.

## Known Stubs

None.

## Self-Check: PASSED

- Confirmed all three retired modules are absent from the worktree and unimportable.
- Confirmed task commits `780ecd8`, `b7b5c70`, `eb64562`, and `ae71c12` exist.
