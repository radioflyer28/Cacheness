---
phase: 28-silent-data-loss-remediation
plan: 01
subsystem: storage
tags: [blob-store, clear-all, namespaces, reliability]

requires: []
provides:
  - Reserved-aware recursive blob cleanup for active namespace clear_all()
  - clear_all_namespaces() coverage for default and non-default blob directories
affects: [storage, namespaces, reliability, REL-01]

tech-stack:
  added: []
  patterns: [reserved-aware filesystem cleanup, namespace-scoped blob traversal]

key-files:
  created:
    - .planning/phases/28-silent-data-loss-remediation/28-01-SUMMARY.md
  modified:
    - src/cacheness/storage/blob_store.py
    - tests/test_core.py

key-decisions:
  - "Source of truth was docs/CODE_REVIEW_FINDINGS.md R1 and docs/CODE_REVIEW_ACTIONS.md TASK-1."
  - "Kept clear_all() scoped to the active namespace because existing namespace tests define that public contract; clear_all_namespaces() remains the cross-namespace destructive operation."
  - "Used active blob backend directory traversal instead of root-level extension globs so sharded and custom-extension payload files are removed."

patterns-established:
  - "Blob cleanup should enumerate namespace blob roots and exclude reserved metadata, DB sidecars, signing keys, and intent files."

requirements-completed: [REL-01]

duration: 75min
completed: 2026-06-12
---

# Phase 28 Plan 01 Summary

**Namespace-recursive clear_all() blob cleanup without crossing namespace boundaries or deleting reserved files.**

## Performance

- **Duration:** 75 min
- **Started:** 2026-06-12T15:42:00-04:00
- **Completed:** 2026-06-12T16:56:51-04:00
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Replaced root-only extension glob cleanup with reserved-aware traversal of the active filesystem blob namespace directory.
- Added regressions for `clear_all()` removing active namespace blobs while preserving metadata/signing files and neighboring namespace data.
- Added `clear_all_namespaces()` coverage proving the nuclear operation clears default blobs and removes non-default namespace blob directories.

## Task Commits

1. **Task 1: Fix active namespace blob cleanup** - this commit

## Files Created/Modified

- `src/cacheness/storage/blob_store.py` - `_clear_blob_files()` now recursively deletes active namespace blob files and preserves reserved files.
- `tests/test_core.py` - Adds `clear_all()` and `clear_all_namespaces()` regressions for namespace blob cleanup.
- `.planning/phases/28-silent-data-loss-remediation/28-01-SUMMARY.md` - Captures execution evidence and decisions.

## Decisions Made

- Kept `clear_all()` namespace-scoped. TASK-1 wording asked for default plus custom namespace setup, but the existing namespace suite explicitly asserts that `clear_all()` must not delete another namespace's data. The implemented fix satisfies R1 by deleting blobs under the active namespace recursively; `clear_all_namespaces()` handles the cross-namespace case.
- Preserved `*_metadata.json` files in addition to `cache_metadata.json*` because JSON non-default namespaces use `{namespace_id}_metadata.json`.

## Deviations from Plan

### Auto-fixed Issues

**1. Scoped clear_all() to active namespace**
- **Found during:** Tier-1 verification
- **Issue:** A first recursive `cache_dir.rglob("*")` implementation deleted blobs from neighboring namespaces, failing existing namespace isolation tests.
- **Fix:** Traverse the active filesystem blob backend root instead of the whole cache root.
- **Files modified:** `src/cacheness/storage/blob_store.py`, `tests/test_core.py`
- **Verification:** Focused namespace tests and Task-1 Tier-1 file set pass.
- **Committed in:** this commit

---

**Total deviations:** 1 auto-fixed
**Impact on plan:** Preserves the established namespace contract while fixing the documented root-only cleanup bug.

## Issues Encountered

- The raw Tier-1 command inherited repo-wide pytest addopts and expanded to 1,897 tests. It exposed the over-broad first implementation; final verification used the same Tier-1 file set with addopts disabled.
- `ty check src/cacheness/storage/blob_store.py tests/test_core.py` reports pre-existing diagnostics in those files, including signing typed-dict calls in `blob_store.py` and optional dataframe imports/pytest skip typing in `tests/test_core.py`.

## Verification

- Verify-first before fix: `LEFTOVER BLOBS: [WindowsPath('.../default/a75e2ab759a78ad7.pkl')]`
- Verify after fix: `LEFTOVER BLOBS: []`
- Focused scoped/nuclear tests: `7 passed, 102 deselected`
- Tier-1 file set: `136 passed`
- `ruff format src/cacheness/storage/blob_store.py tests/test_core.py`: passed, reformatted files
- `ruff check --fix src/cacheness/storage/blob_store.py tests/test_core.py`: passed, fixed one lint issue
- `ruff check src/cacheness/storage/blob_store.py tests/test_core.py`: passed
- `ty check src/cacheness/storage/blob_store.py tests/test_core.py`: failed on existing diagnostics noted above

## User Setup Required

None.

## Next Phase Readiness

REL-01 is addressed for active namespace `clear_all()` and cross-namespace `clear_all_namespaces()`. Phase 28 can proceed to Plan 02 write-intent cleanup.

---
*Phase: 28-silent-data-loss-remediation*
*Completed: 2026-06-12*
