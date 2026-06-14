---
phase: 29-ttl-eviction-consistency
plan: 04
subsystem: cache-eviction
tags: [ttl, eviction, blobs, uri, memory-backend]

requires:
  - phase: 29-ttl-eviction-consistency
    provides: TTL-02 public cleanup routing and prior Phase 29 cache cleanup semantics
provides:
  - URI blob deletion through the active blob backend during size-limit eviction
  - Memory URI regression coverage for size eviction
  - Warning/no-crash behavior when remote-style deletion raises
affects: [phase-29, ttl-04, size-eviction, blob-backends]

tech-stack:
  added: []
  patterns:
    - URI blob cleanup is delegated to the blob backend rather than parsed in core.py
    - Local filesystem eviction continues to use resolved Path.unlink behavior

key-files:
  created:
    - .planning/phases/29-ttl-eviction-consistency/29-04-SUMMARY.md
  modified:
    - src/cacheness/core.py
    - tests/test_core.py

key-decisions:
  - "UnifiedCache._enforce_size_limit delegates URI actual_path deletion to self._blob_store.blob_backend.delete_blob(actual_path)."
  - "Remote-style deletion failures are warning-only during size enforcement so metadata eviction does not crash."
  - "The memory URI regression patches the cache's blob backend to InMemoryBlobBackend before writes, keeping this plan scoped to eviction behavior rather than broader blob-backend configuration plumbing."

patterns-established:
  - "Size eviction handles local and URI blobs through separate branches: backend delete for URI, Path.unlink for local files."

requirements-completed: [TTL-04]

duration: 2h 53m
completed: 2026-06-14T01:29:18Z
---

# Phase 29 Plan 04: Remote URI Eviction Summary

**Size-limit eviction now removes remote-style URI blobs through the active blob backend while preserving local unlink behavior.**

## Performance

- **Duration:** 2h 53m
- **Started:** 2026-06-13T22:36:10Z
- **Completed:** 2026-06-14T01:29:18Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added TTL-04 verify-first coverage for memory URI blob deletion, backend deletion warning behavior, and storage-mode no-eviction behavior.
- Updated `UnifiedCache._enforce_size_limit()` to call `self._blob_store.blob_backend.delete_blob(actual_path)` for URI paths.
- Preserved existing local filesystem cleanup behavior for non-URI paths.
- Confirmed the focused Phase 29 verification slice remains green.

## Task Commits

1. **Task 1: Add TTL-04 verify-first memory URI eviction regression** - `f11d7c0` (`test`)
2. **Task 2: Delete URI blobs through blob_backend during size enforcement** - `476df59` (`feat`)

## Files Created/Modified

- `src/cacheness/core.py` - `_enforce_size_limit()` now delegates URI blob deletion to the active blob backend and logs warning-only failures.
- `tests/test_core.py` - Added TTL-04 memory URI eviction, remote delete warning, and storage-mode no-op regressions.
- `.planning/phases/29-ttl-eviction-consistency/29-04-SUMMARY.md` - Execution summary and verification record.

## Verification

- RED: `uv run --python 3.12 pytest --override-ini testpaths= tests/test_core.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest -p no:cacheprovider`
  - Result: failed as expected at `test_size_limit_eviction_deletes_memory_uri_blob` because the evicted `memory://` blob remained, and at `test_size_limit_remote_delete_failure_logs_warning` because no warning was logged.
- PASS: `uv run --python 3.12 pytest --override-ini testpaths= -o addopts="" tests/test_core.py tests/test_blob_store.py -x -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest -p no:cacheprovider`
  - Result: 107 passed.
- PASS: `uv run --python 3.12 ruff format src/cacheness/core.py tests/test_core.py tests/test_blob_store.py`
- PASS: `uv run --python 3.12 ruff check --fix src/cacheness/core.py tests/test_core.py tests/test_blob_store.py`
- PASS: `uv run --python 3.12 ruff check src/cacheness/core.py tests/test_core.py tests/test_blob_store.py`
- BASELINE FAIL: `uv run --python 3.12 ty check src/cacheness/core.py tests/test_core.py tests/test_blob_store.py`
  - Result: failed with existing touched-file diagnostics in `core.py` and older `tests/test_core.py` sections, including `Path | str` path unions, optional `polars` imports, pytest `skip` typing, nullable metadata-entry edits, and an existing float passed to `max_cache_size_mb`.
  - New nullable-entry diagnostics from the TTL-04 regression were fixed before final verification.
- PASS: `uv run --python 3.12 pytest --override-ini testpaths= -o addopts="" tests/test_core.py tests/test_metadata.py tests/test_backend_parity.py tests/test_update_operations.py tests/test_storage_mode.py tests/test_blob_store.py tests/test_cache_signing.py tests/test_cache_integrity_verification.py -x -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest -p no:cacheprovider`
  - Result: 303 passed, 12 skipped.

## Decisions Made

- Kept the production change scoped to `_enforce_size_limit()`; no storage-mode eviction policy or public API behavior was added.
- Used `InMemoryBlobBackend` by patching the cache's active blob backend before writes for the regression, because this plan targets the eviction loop and not broader `UnifiedCache` blob backend construction.
- Used `-o addopts=""` for focused pytest runs after xdist startup proved disproportionately slow for this slice; TensorFlow ignore and workspace-local pytest temp paths were preserved.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The first memory URI regression setup initially relied on `CacheConfig(blob=CacheBlobConfig(blob_backend="memory"))`, but `UnifiedCache._init_blob_store()` did not forward that backend into `BlobStore`. The test was kept within plan scope by patching `cache._blob_store.blob_backend` to `InMemoryBlobBackend` before writes.
- The memory URI test initially asserted a full `get()` of the non-evicted entry. That overreached into memory-backed read behavior outside TTL-04, so the assertion was narrowed to metadata retention plus backend blob existence for the non-evicted entry.
- The exact xdist-based targeted command was slow in this environment; final task and focused Phase 29 verification used the same test files with `-o addopts=""` to run sequentially.
- `.planning/config.json` had a pre-existing unrelated newline modification and was preserved unstaged.

## Auth Gates

None.

## Known Stubs

None.

## Threat Flags

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

TTL-04 is implemented and covered. Phase 29 now covers TTL-01 through TTL-04 and is ready for phase-level verification or the next roadmap phase.

## Self-Check: PASSED

- Found `.planning/phases/29-ttl-eviction-consistency/29-04-SUMMARY.md`.
- Found task commit `f11d7c0`.
- Found task commit `476df59`.

---
*Phase: 29-ttl-eviction-consistency*
*Completed: 2026-06-14*
