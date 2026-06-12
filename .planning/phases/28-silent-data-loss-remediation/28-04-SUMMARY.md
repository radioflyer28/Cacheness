---
phase: 28-silent-data-loss-remediation
plan: 04
subsystem: metadata
tags: [json-backend, persistence, corrupt-metadata, reliability]

requires:
  - phase: 28-03
    provides: Completed write-intent crash-window coverage
provides:
  - JSON put_entry/remove_entry persistence failure propagation
  - Best-effort JSON telemetry persistence
  - Corrupt JSON metadata preservation
affects: [metadata, json-backend, REL-05, REL-06]

tech-stack:
  added: []
  patterns: [raise_on_error persistence switch, corrupt-file quarantine]

key-files:
  created:
    - .planning/phases/28-silent-data-loss-remediation/28-04-SUMMARY.md
  modified:
    - src/cacheness/metadata/json_backend.py
    - tests/test_metadata.py
    - tests/test_json_schema_versioning.py

key-decisions:
  - "Source of truth was docs/CODE_REVIEW_ACTIONS.md TASK-3 and docs/CODE_REVIEW_FINDINGS.md R3/R4."
  - "Only data-critical put_entry() and remove_entry() opt into raise_on_error."
  - "Stats and access-time writes remain best-effort and log failures without raising."

patterns-established:
  - "Corrupt JSON metadata is moved to *.corrupt-<timestamp> before an empty metadata store is created."

requirements-completed: [REL-05, REL-06]

duration: 25min
completed: 2026-06-12
---

# Phase 28 Plan 04 Summary

**JSON metadata writes now surface critical persistence failures and preserve corrupt files for recovery.**

## Performance

- **Duration:** 25 min
- **Started:** 2026-06-12T16:41:00-04:00
- **Completed:** 2026-06-12T17:06:20-04:00
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added `_save_to_disk(raise_on_error=False)` and enabled raising for `put_entry()` and `remove_entry()`.
- Kept access-time and hit/miss writes best-effort.
- Recreated missing metadata parent directories before writes.
- Preserved corrupt or invalid metadata files as `*.corrupt-<timestamp>` and logged the backup path at error level.

## Task Commits

1. **Task 4: Harden JSON metadata persistence** - this commit

## Files Created/Modified

- `src/cacheness/metadata/json_backend.py` - Adds critical-write failure propagation and corrupt-file preservation.
- `tests/test_metadata.py` - Adds save-failure and missing-parent regressions.
- `tests/test_json_schema_versioning.py` - Adds corrupt metadata preservation regression.
- `.planning/phases/28-silent-data-loss-remediation/28-04-SUMMARY.md` - Captures execution evidence and decisions.

## Decisions Made

- `put_entry()` and `remove_entry()` are treated as data-critical and raise when persistence fails.
- Telemetry/access-time saves keep the old best-effort contract.
- Invalid-schema metadata is handled like corrupt JSON because both cases would otherwise discard existing metadata without a recovery copy.

## Deviations from Plan

None - plan executed as specified.

## Issues Encountered

- Verify-first confirmed old behavior: data-critical saves logged but did not raise, while corrupt metadata started empty without a backup.
- `ty check src/cacheness/metadata/json_backend.py tests/test_metadata.py tests/test_json_schema_versioning.py` reports existing broad JSON metadata typing diagnostics and a pre-existing `pytest.fail` typing issue.

## Verification

- Verify-first focused regressions before implementation: `3 failed, 1 passed, 51 deselected`
- Focused regressions after implementation: `4 passed, 51 deselected`
- Plan 04 Tier-1 file set: `56 passed`
- `ruff format src/cacheness/metadata/json_backend.py tests/test_metadata.py tests/test_json_schema_versioning.py`: passed, reformatted files
- `ruff check --fix src/cacheness/metadata/json_backend.py tests/test_metadata.py tests/test_json_schema_versioning.py`: passed
- `ruff check src/cacheness/metadata/json_backend.py tests/test_metadata.py tests/test_json_schema_versioning.py`: passed
- `ty check src/cacheness/metadata/json_backend.py tests/test_metadata.py tests/test_json_schema_versioning.py`: failed on existing diagnostics noted above

## User Setup Required

None.

## Next Phase Readiness

REL-05 and REL-06 are addressed. Phase 28 can proceed to Plan 05.

---
*Phase: 28-silent-data-loss-remediation*
*Completed: 2026-06-12*
