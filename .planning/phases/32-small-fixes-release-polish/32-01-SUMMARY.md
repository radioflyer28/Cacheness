---
phase: 32-small-fixes-release-polish
plan: 01
subsystem: storage
tags: [blobstore, metadata, regression-test, polish]

requires:
  - phase: 31-security-storage-mode-posture
    provides: BlobStore canonical signing and metadata contract preservation
provides:
  - BlobStore.put copies caller metadata before adding internal metadata fields.
  - Regression coverage for caller metadata immutability.
affects: [BlobStore, storage metadata, POL-01]

tech-stack:
  added: []
  patterns:
    - Copy caller-owned mutable metadata at API boundaries before internal mutation.

key-files:
  created:
    - .planning/phases/32-small-fixes-release-polish/deferred-items.md
  modified:
    - src/cacheness/storage/blob_store.py
    - tests/test_blob_store.py

key-decisions:
  - "BlobStore.put preserves the existing nested metadata contract by copying metadata with dict(metadata or {}) before adding internal fields."

patterns-established:
  - "Caller metadata boundary: copy mutable caller dictionaries before adding storage/signing metadata."

requirements-completed: [POL-01]

duration: 9min
completed: 2026-06-15
---

# Phase 32 Plan 01: BlobStore Metadata Immutability Summary

**BlobStore caller metadata is copied before Cacheness adds internal storage metadata, preserving public API behavior and stored nested metadata.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-06-15T18:10:40Z
- **Completed:** 2026-06-15T18:19:38Z
- **Tasks:** 1
- **Files modified:** 2 code/test files, 2 planning files

## Accomplishments

- Changed `BlobStore.put()` to use `dict(metadata or {})` before adding internal metadata fields.
- Added a regression test proving caller-owned metadata dictionaries remain unchanged after `put()`.
- Verified stored nested metadata still contains the caller key plus internal fields such as `actual_path`, `storage_format`, and `file_hash`.

## Task Commits

1. **Task 32-01-01: Copy BlobStore caller metadata before mutation** - `18b2629` (fix)

**Plan metadata:** pending SDK metadata commit or configured skip.

## Files Created/Modified

- `src/cacheness/storage/blob_store.py` - Copies caller metadata before adding internal BlobStore fields.
- `tests/test_blob_store.py` - Adds regression coverage for caller metadata immutability and stored nested metadata preservation.
- `.planning/phases/32-small-fixes-release-polish/deferred-items.md` - Records out-of-scope quality/test failures observed during verification.
- `.planning/phases/32-small-fixes-release-polish/32-01-SUMMARY.md` - This execution summary.

## Decisions Made

- Preserve the existing nested `"metadata"` storage contract and signing order; only the aliasing boundary changed.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The initial `uv` commands could not use the default user cache path or sandbox interpreter access. Verification was run with repo-local `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR`, and escalated command execution where required.
- `uv run ty check src/cacheness/storage/blob_store.py tests/test_blob_store.py` failed on two pre-existing type diagnostics in `src/cacheness/storage/blob_store.py` constructor setup, unrelated to this metadata-copy change.
- The literal plan pytest command collected unrelated tests under the repository xdist/addopts configuration and surfaced existing failures outside `tests/test_blob_store.py`. The scoped BlobStore suite was re-run with `-o addopts=''` to verify this plan's acceptance behavior.

## Verification

- RED regression: `uv run pytest tests/test_blob_store.py::TestBlobStoreBasic::test_put_does_not_mutate_caller_metadata -x -q --ignore=tests/test_tensorflow_handler.py` failed before the source fix with caller metadata containing internal fields.
- Post-fix direct probe: caller metadata printed `{'experiment': 'x42'}` and stored metadata printed `x42 default/sample.pkl pickle True`.
- `uv run ruff format src/cacheness/storage/blob_store.py tests/test_blob_store.py` passed; 2 files reformatted.
- `uv run ruff check --fix src/cacheness/storage/blob_store.py tests/test_blob_store.py` passed.
- `uv run ruff check src/cacheness/storage/blob_store.py tests/test_blob_store.py` passed.
- `uv run pytest -o addopts='' tests/test_blob_store.py::TestBlobStoreBasic::test_put_does_not_mutate_caller_metadata -q --ignore=tests/test_tensorflow_handler.py` passed: 1 passed.
- `uv run pytest -o addopts='' tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py` passed: 51 passed.

## Known Stubs

None.

## Deferred Issues

- Pre-existing `ty` diagnostics in `src/cacheness/storage/blob_store.py` are recorded in `deferred-items.md`.
- Existing unrelated pytest failures collected by the repository addopts path are recorded in `deferred-items.md`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

POL-01 is implemented and verified. Phase 32 can continue with the next independent small-fix plan.

## Self-Check: PASSED

- Found `.planning/phases/32-small-fixes-release-polish/32-01-SUMMARY.md`.
- Found `.planning/phases/32-small-fixes-release-polish/deferred-items.md`.
- Found implementation commit `18b2629`.

---
*Phase: 32-small-fixes-release-polish*
*Completed: 2026-06-15*
