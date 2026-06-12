---
phase: 28-silent-data-loss-remediation
plan: 03
subsystem: storage
tags: [write-intent, crash-recovery, seed-006, storage-mode]

requires:
  - phase: 28-02
    provides: Conservative stale-intent cleanup
provides:
  - Before-blob-write intent timing for cache mode
  - Before-blob-write intent timing for storage mode
affects: [storage, crash-recovery, storage-mode, R8]

tech-stack:
  added: []
  patterns: [planned-path write intents before blob IO]

key-files:
  created:
    - .planning/phases/28-silent-data-loss-remediation/28-03-SUMMARY.md
  modified:
    - src/cacheness/core.py
    - src/cacheness/_storage_mode_mixin.py
    - tests/test_write_intent.py

key-decisions:
  - "Source of truth was .planning/seeds/SEED-006-record-write-intent-before-blob-write.md and docs/CODE_REVIEW_FINDINGS.md R8."
  - "Timing tests assert the intent exists inside the interrupted _write_blob call; normal Python exception cleanup still removes the intent afterward."
  - "Inline writes continue to skip intents because they do not create filesystem blobs."

patterns-established:
  - "Use handler.get_file_extension(config) with the cache base path to record a planned cache-relative blob path before serialization starts."

requirements-completed: [REL-02, REL-03, REL-04]

duration: 20min
completed: 2026-06-12
---

# Phase 28 Plan 03 Summary

**Write intents now cover the blob-write crash window in both cache mode and storage mode.**

## Performance

- **Duration:** 20 min
- **Started:** 2026-06-12T16:43:30-04:00
- **Completed:** 2026-06-12T17:03:34-04:00
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Recorded planned cache-relative intent paths before `_blob_store._write_blob()` in normal cache mode.
- Mirrored the same before-write timing in storage mode.
- Added regressions proving the intent exists before blob I/O begins and successful/handled failures leave no residual intents.

## Task Commits

1. **Task 3: Record intents before blob writes** - this commit

## Files Created/Modified

- `src/cacheness/core.py` - Moves non-inline intent recording before cache-mode blob writes.
- `src/cacheness/_storage_mode_mixin.py` - Moves non-inline intent recording before storage-mode blob writes.
- `tests/test_write_intent.py` - Adds cache-mode and storage-mode timing regressions.
- `.planning/phases/28-silent-data-loss-remediation/28-03-SUMMARY.md` - Captures execution evidence and decisions.

## Decisions Made

- The planned path uses the already-selected handler's `get_file_extension()` so it matches the local file path expected during serialization.
- Tests model an interrupted `_write_blob()` by checking for the intent inside the monkeypatched write call. After the exception unwinds, cleanup still clears the intent, preserving the existing normal-exception behavior.

## Deviations from Plan

None - plan executed as specified.

## Issues Encountered

- The plan text says a failure should leave an intent file, but the existing exception handler clears intents for ordinary Python exceptions. The implemented test checks crash-window timing directly while preserving normal exception cleanup.
- `ty check src/cacheness/core.py src/cacheness/_storage_mode_mixin.py tests/test_write_intent.py` reports existing mixin/self-attribute diagnostics and `core.py` `Path | str` diagnostics.

## Verification

- Verify-first timing regressions before implementation: `2 failed, 16 deselected`
- Timing regressions after implementation: `2 passed, 16 deselected`
- Plan 03 Tier-1 file set: `117 passed`
- `ruff format src/cacheness/core.py src/cacheness/_storage_mode_mixin.py tests/test_write_intent.py`: passed, reformatted files
- `ruff check --fix src/cacheness/core.py src/cacheness/_storage_mode_mixin.py tests/test_write_intent.py`: passed
- `ruff check src/cacheness/core.py src/cacheness/_storage_mode_mixin.py tests/test_write_intent.py`: passed
- `ty check src/cacheness/core.py src/cacheness/_storage_mode_mixin.py tests/test_write_intent.py`: failed on existing diagnostics noted above

## User Setup Required

None.

## Next Phase Readiness

The write-intent sequence is complete through R8/SEED-006. Phase 28 can proceed to Plan 04.

---
*Phase: 28-silent-data-loss-remediation*
*Completed: 2026-06-12*
