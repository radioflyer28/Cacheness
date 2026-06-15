---
phase: 32-small-fixes-release-polish
plan: 04
plan_id: 32-04
subsystem: storage
tags: [filesystem, blob-backend, input-validation, pytest, pol-05]

requires:
  - phase: 32-small-fixes-release-polish
    provides: Phase 32 planning context and POL-05 validation contract
provides:
  - Filesystem blob backend absolute blob ID rejection before path construction
  - Backend-boundary regression coverage for absolute blob IDs
affects: [filesystem-blob-backend, blob-namespace-tests, release-polish]

tech-stack:
  added: []
  patterns:
    - Validate raw filesystem blob IDs before separator replacement or base-dir joins
    - Keep relative traversal neutralization behavior covered by regression tests

key-files:
  created:
    - .planning/phases/32-small-fixes-release-polish/32-04-SUMMARY.md
  modified:
    - src/cacheness/storage/backends/blob_backends.py
    - tests/test_blob_namespace.py

key-decisions:
  - "POL-05 rejects absolute raw blob IDs in FilesystemBlobBackend._get_blob_path before separator replacement and before joining with base_dir."
  - "Relative traversal-looking blob IDs continue to be neutralized rather than rejected, preserving existing behavior."
  - "Targeted pytest verification used -o addopts='' because the first targeted RED run broadened through repo addopts."

patterns-established:
  - "Filesystem path safety: reject absolute raw IDs at the backend boundary before any path construction."
  - "Regression tests cover both the rejected absolute case and preserved relative traversal neutralization."

requirements-completed: [POL-05]

duration: 6 min
completed: 2026-06-15
---

# Phase 32 Plan 04: Absolute Blob ID Rejection Summary

**Filesystem blob IDs now reject absolute paths before backend path construction while preserving relative ID sanitization.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-06-15T18:50:29Z
- **Completed:** 2026-06-15T18:56:24Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- Added `ValueError` validation in `FilesystemBlobBackend._get_blob_path()` before separator replacement and before joining with `base_dir`.
- Added backend-boundary regression coverage proving absolute blob IDs are rejected without writing outside the backend directory.
- Added coverage proving relative traversal-looking IDs still sanitize under `base_dir`.

## Task Commits

1. **Task 32-04-01: Reject absolute filesystem blob IDs** - `b46033a` (`fix`)

## Files Created/Modified

- `src/cacheness/storage/backends/blob_backends.py` - Rejects absolute raw blob IDs with `ValueError`.
- `tests/test_blob_namespace.py` - Covers absolute ID rejection and preserved relative traversal neutralization.
- `.planning/phases/32-small-fixes-release-polish/32-04-SUMMARY.md` - Execution summary.

## Decisions Made

- POL-05 is enforced at `FilesystemBlobBackend._get_blob_path()` because that is the filesystem path construction boundary.
- The guard checks the raw `blob_id` before separator replacement, preserving current handling for relative IDs.
- Pytest verification used `-o addopts=''` after the initial targeted RED run broadened beyond the requested file through repository addopts.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The initial `uv run pytest tests/test_blob_namespace.py -x -q --ignore=tests/test_tensorflow_handler.py` RED check broadened beyond the requested target through repository addopts. Subsequent verification used scoped `-o addopts=''` as allowed by the plan.
- Sandboxed `uv` could not query the Python interpreter and the default user uv cache path failed to initialize. Verification commands used repo-local `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` with approved escalation.
- `uv run ty check src/cacheness/storage/backends/blob_backends.py tests/test_blob_namespace.py` exited successfully but reported existing warnings in `tests/test_blob_namespace.py` about dynamic blob backend attributes. These warnings were not introduced by this plan.

## Verification

- RED: `uv run pytest tests/test_blob_namespace.py -x -q --ignore=tests/test_tensorflow_handler.py` failed as expected on `test_absolute_blob_id_is_rejected_before_filesystem_write` before the production fix. The command broadened through repo addopts, so later pytest runs were scoped.
- PASS: `uv run pytest tests/test_blob_namespace.py -x -q --ignore=tests/test_tensorflow_handler.py -o addopts=''` -> 22 passed.
- PASS: `uv run ruff format src/cacheness/storage/backends/blob_backends.py tests/test_blob_namespace.py` -> 1 file reformatted, 1 unchanged.
- PASS: `uv run ruff check --fix src/cacheness/storage/backends/blob_backends.py tests/test_blob_namespace.py`.
- PASS: `uv run ruff check src/cacheness/storage/backends/blob_backends.py tests/test_blob_namespace.py`.
- PASS: `uv run ty check src/cacheness/storage/backends/blob_backends.py tests/test_blob_namespace.py` -> exit 0 with existing warnings.
- PASS: `uv run pytest tests/test_blob_namespace.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py -o addopts=''` -> 76 passed.

## Known Stubs

None.

## Threat Flags

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

POL-05 is complete. Phase 32 can continue with the remaining independent small-fix plans.

## Self-Check: PASSED

- Found `src/cacheness/storage/backends/blob_backends.py`.
- Found `tests/test_blob_namespace.py`.
- Found implementation commit `b46033a`.
- Confirmed `Path(blob_id).is_absolute()` guard and regression tests are present.

---
*Phase: 32-small-fixes-release-polish*
*Completed: 2026-06-15*
