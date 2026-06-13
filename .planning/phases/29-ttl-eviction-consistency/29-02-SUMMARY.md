---
phase: 29-ttl-eviction-consistency
plan: 02
subsystem: cache-cleanup
tags: [ttl, expiry, cleanup, hooks, blob-files]

requires:
  - phase: 29-ttl-eviction-consistency
    provides: TTL-01 stored expires_at precedence and public cleanup semantics
provides:
  - Init-time expired cleanup routed through public cleanup_expired
  - Blob deletion and on_evict hook coverage for constructor cleanup
affects: [phase-29, ttl-02, cache-init, cleanup-expired]

tech-stack:
  added: []
  patterns:
    - Constructor TTL cleanup delegates to the public cleanup boundary
    - Public cleanup remains responsible for blob deletion and eviction hooks

key-files:
  created:
    - .planning/phases/29-ttl-eviction-consistency/29-02-SUMMARY.md
  modified:
    - src/cacheness/core.py
    - tests/test_core.py

key-decisions:
  - "UnifiedCache._cleanup_expired keeps the default_ttl_seconds is None no-op guard, then delegates to cleanup_expired(ttl_seconds)."
  - "Read-path expired-entry deletion remained outside this plan per D-10; no get() behavior was changed."

patterns-established:
  - "Init-time cleanup should reuse public cleanup paths when metadata, blobs, and hooks must stay coherent."

requirements-completed: [TTL-02]

duration: 8min
completed: 2026-06-13T22:00:56Z
---

# Phase 29 Plan 02: Init Cleanup Public Path Summary

**Constructor TTL cleanup now uses the same blob-deleting and hook-invoking public cleanup path as manual expired cleanup.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-13T21:53:06Z
- **Completed:** 2026-06-13T22:00:56Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added a verify-first regression proving init-time expired cleanup must remove metadata, delete the associated blob file, and call `on_evict(cache_key, "expired")`.
- Changed `UnifiedCache._cleanup_expired()` to delegate to `self.cleanup_expired(ttl_seconds)` after preserving the no-TTL no-op guard.
- Confirmed initialization order: `_lock`, metadata backend, blob store, and write journal are initialized before `_cleanup_expired()` can run, and the call is not made while already inside the same lock.

## Task Commits

1. **Task 1: Add TTL-02 verify-first init cleanup regression** - `1c322d7` (`test`)
2. **Task 2: Route init cleanup through public cleanup_expired** - `a1d0404` (`feat`)

## Files Created/Modified

- `src/cacheness/core.py` - `_cleanup_expired()` now routes constructor cleanup through `cleanup_expired()`.
- `tests/test_core.py` - Added `test_init_cleanup_expired_deletes_blob_files_and_invokes_hook`.
- `.planning/phases/29-ttl-eviction-consistency/29-02-SUMMARY.md` - Execution summary and verification record.

## Verification

- RED: `uv run --python 3.12 pytest tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py`
  - Result: failed as expected at `tests/test_core.py::TestCacheness::test_init_cleanup_expired_deletes_blob_files_and_invokes_hook` because the blob still existed after init cleanup.
  - Note: this raw command over-collected due local pytest config; subsequent scoped runs used `--override-ini testpaths=`.
- PASS: `uv run --python 3.12 pytest --override-ini testpaths= tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest`
  - Result: 60 passed.
- PASS: `uv run --python 3.12 ruff format src/cacheness/core.py tests/test_core.py`
- PASS: `uv run --python 3.12 ruff check --fix src/cacheness/core.py tests/test_core.py`
- PASS: `uv run --python 3.12 ruff check src/cacheness/core.py tests/test_core.py`
- BASELINE FAIL: `uv run --python 3.12 ty check src/cacheness/core.py tests/test_core.py`
  - Result: failed with existing touched-file diagnostics: `Path | str` path unions in `core.py`, optional `polars` imports, pytest `skip` typing, pre-existing nullable metadata-entry edits in `tests/test_core.py`, and an existing float passed to `max_cache_size_mb`.
- PASS: `uv run --python 3.12 pytest --override-ini testpaths= tests/test_core.py tests/test_metadata.py tests/test_backend_parity.py tests/test_update_operations.py tests/test_storage_mode.py tests/test_blob_store.py tests/test_cache_signing.py tests/test_cache_integrity_verification.py -x -q --ignore=tests/test_tensorflow_handler.py --basetemp .tmp\pytest -p no:cacheprovider`
  - Result: 296 passed, 12 skipped.

## Decisions Made

- Kept the plan scope to TASK-6 / TTL-02 only; no read-path deletion or storage-mode policy changes were introduced.
- Used workspace-local uv cache, Python install, and pytest temp directories because user-profile cache/temp paths were not reliable in this sandbox.
- Disabled pytest cache provider for the focused Phase 29 command to avoid a denied `.pytest_cache` write warning.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The first targeted pytest command over-collected the broader suite despite the positional test file. Scoped verification used `--override-ini testpaths=` while preserving the plan's TensorFlow ignore.
- `uv` initially failed against the user-profile cache path. Verification continued with project-local `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR`.
- Pytest xdist initially attempted a denied user temp directory. Verification continued with workspace-local `--basetemp .tmp\pytest`.
- `gsd-tools query state.advance-plan` and `state.record-session` could not parse this STATE layout; metrics, roadmap, requirements, and decisions were updated through working handlers, then the stale STATE current-position text was patched manually.
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

TTL-02 is implemented and covered. Phase 29 Plan 03 can build on coherent TTL cleanup behavior while addressing access counters, provenance timestamps, TTL field preservation, and signatures.

## Self-Check: PASSED

- Found `.planning/phases/29-ttl-eviction-consistency/29-02-SUMMARY.md`.
- Found task commit `1c322d7`.
- Found task commit `a1d0404`.

---
*Phase: 29-ttl-eviction-consistency*
*Completed: 2026-06-13*
