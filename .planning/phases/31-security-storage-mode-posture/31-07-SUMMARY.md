---
phase: 31-security-storage-mode-posture
plan: 07
subsystem: storage
tags: [write-intent, storage-mode, crash-recovery, durability, regression-tests]

requires:
  - phase: 31-security-storage-mode-posture
    provides: STRG-01 storage-mode warning policy from Phase 31 Plan 05
provides:
  - STRG-03 regressions proving cache-mode intents exist before blob writes
  - STRG-03 regressions proving storage-mode intents exist before blob writes
  - Cleanup coverage for missing planned blobs with committed metadata protection
  - Verification that existing Phase 28 pre-blob intent implementation remains compliant
affects: [storage-mode, write-intent-journal, unified-cache, phase-31]

tech-stack:
  added: []
  patterns:
    - Verify-first preservation task for behavior already implemented by prior phases
    - Intent-payload assertions at the monkeypatched blob-write boundary

key-files:
  created:
    - .planning/phases/31-security-storage-mode-posture/31-07-SUMMARY.md
  modified:
    - tests/test_write_intent.py

key-decisions:
  - "STRG-03 production code remained unchanged because regressions passed against the existing pre-blob write-intent implementation."
  - "Task 2 was recorded with an empty verification commit to preserve the plan's per-task commit trail without source churn."

patterns-established:
  - "Pre-blob intent tests should assert the journal payload's cache_key and cache-dir-relative planned blob path inside the _write_blob boundary."
  - "Stale-intent cleanup tests should cover committed metadata with a missing never-created planned blob path."

requirements-completed: [STRG-03]

duration: 7min
completed: 2026-06-15
---

# Phase 31 Plan 07: Write-Intent Pre-Blob Coverage Summary

**STRG-03 is locked by regressions that prove write intents exist before blob writes and cleanup preserves committed data when planned blobs were never created.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-06-15T02:40:56Z
- **Completed:** 2026-06-15T02:47:50Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Strengthened cache-mode and storage-mode pre-blob tests to assert the intent payload exists before `_write_blob()` starts.
- Added cleanup coverage for a committed key whose stale intent points at a planned blob that was never created.
- Added storage-mode overwrite fault coverage proving a failure before replacement blob creation preserves the previous committed value.
- Verified `src/cacheness/core.py`, `src/cacheness/_storage_mode_mixin.py`, and `src/cacheness/write_intent.py` already satisfy STRG-03, so no production source changes were made.

## Task Commits

1. **Task 1: Add STRG-03 pre-blob intent and cleanup regressions** - `758a729` (test)
2. **Task 2: Preserve or repair pre-blob write-intent implementation** - `abe81a0` (chore, empty verification commit)

**Plan metadata:** pending final docs commit or skipped by GSD commit helper.

## Files Created/Modified

- `tests/test_write_intent.py` - Adds exact intent-payload assertions, missing-planned-blob cleanup coverage, and storage-mode pre-blob overwrite failure coverage.
- `.planning/phases/31-security-storage-mode-posture/31-07-SUMMARY.md` - Records Plan 31-07 execution and verification.

## Decisions Made

- Kept production source files unchanged because all STRG-03 regressions passed.
- Used an empty Task 2 verification commit so the plan has an explicit per-task audit trail without introducing no-op source edits.

## Deviations from Plan

None - plan executed as written. The verify-first path found the existing implementation compliant.

## Issues Encountered

- The literal plan pytest command failed before collection in this Windows environment because xdist tried to use an ACL-denied temp root at `AppData\Local\Temp\pytest-of-akriz`.
- Controlled verification used repo-local `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, cleared pytest addopts, disabled pytest cache, and set `--basetemp` under `.uv-cache`.
- `ty check` on the full plan file list failed on pre-existing mixin and legacy test typing diagnostics. `ty check tests/test_write_intent.py` passed, so the new STRG-03 tests did not add typing diagnostics.

## Verification

- Literal plan command: `uv run --python 3.12 pytest tests/test_atomic_writes.py tests/test_write_intent.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` - FAILED before collection on Windows xdist temp ACL (`PermissionError: ... AppData\Local\Temp\pytest-of-akriz`).
- Task 1 cache gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-07-task1 tests/test_atomic_writes.py tests/test_write_intent.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 90 passed.
- Task 1 storage gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-07-task1-storage tests/test_storage_mode.py tests/test_atomic_writes.py tests/test_write_intent.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 68 passed.
- Post-format focused gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-07-task1-postformat tests/test_write_intent.py -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 20 passed.
- Task 2 cache gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-07-task2-cache tests/test_atomic_writes.py tests/test_write_intent.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 90 passed.
- Task 2 storage gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-07-task2-storage tests/test_storage_mode.py tests/test_atomic_writes.py tests/test_write_intent.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 68 passed.
- Plan gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .uv-cache\pytest-31-07-plan-gate tests/test_atomic_writes.py tests/test_write_intent.py tests/test_core.py tests/test_storage_mode.py -x -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 131 passed.
- `uv run --python 3.12 ruff format src/cacheness/core.py src/cacheness/_storage_mode_mixin.py src/cacheness/write_intent.py tests/test_atomic_writes.py tests/test_write_intent.py tests/test_core.py tests/test_storage_mode.py` - PASSED, 7 files left unchanged after the Task 1 commit.
- `uv run --python 3.12 ruff check --fix src/cacheness/core.py src/cacheness/_storage_mode_mixin.py src/cacheness/write_intent.py tests/test_atomic_writes.py tests/test_write_intent.py tests/test_core.py tests/test_storage_mode.py` - PASSED.
- `uv run --python 3.12 ruff check src/cacheness/core.py src/cacheness/_storage_mode_mixin.py src/cacheness/write_intent.py tests/test_atomic_writes.py tests/test_write_intent.py tests/test_core.py tests/test_storage_mode.py` - PASSED.
- `uv run --python 3.12 ty check src/cacheness/core.py src/cacheness/_storage_mode_mixin.py src/cacheness/write_intent.py tests/test_atomic_writes.py tests/test_write_intent.py tests/test_core.py tests/test_storage_mode.py` - FAILED on pre-existing diagnostics in mixin/core/test typing.
- `uv run --python 3.12 ty check tests/test_write_intent.py` - PASSED.

## Known Stubs

None. Stub scan only matched existing `core.py` log text containing "not available"; no new stubs were introduced.

## Threat Flags

None. This plan adds tests around an existing write-intent file-I/O trust boundary and does not introduce new endpoints, auth paths, schema changes, or production file-access surfaces.

## User Setup Required

None - no external service configuration required.

## TDD Gate Compliance

- RED/verify-first test commit exists: `758a729` (`test(31-07): add write-intent coverage regressions`)
- GREEN/preservation commit exists after tests: `abe81a0` (`chore(31-07): verify write-intent implementation`)
- Production code was already green before Task 2, matching the plan's verify-first instruction.

## Next Phase Readiness

STRG-03 is complete. Plan 31-06 can rely on write-intent coverage spanning the full uncommitted-blob window while it handles transaction guarantees and opt-in fsync policy.

## Self-Check: PASSED

- Verified summary and key test file exist.
- Verified task commits `758a729` and `abe81a0` are present in git history.
- Verified task commits did not delete tracked files.

---
*Phase: 31-security-storage-mode-posture*
*Completed: 2026-06-15*
