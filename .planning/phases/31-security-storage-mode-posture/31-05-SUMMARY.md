---
phase: 31-security-storage-mode-posture
plan: 05
subsystem: storage
tags: [storage-mode, warnings, durability, destructive-api]

requires:
  - phase: 31-security-storage-mode-posture
    provides: Phase 31 storage-mode warning policy decisions D-29 through D-34
provides:
  - Storage-mode destructive API warnings via logger and RuntimeWarning
  - Warning-first coverage for clear_all, clear_all_namespaces, explicit TTL cleanup, and forced size cleanup
  - Storage-mode regression tests proving warning-first behavior and no implicit TTL or eviction behavior
  - Transaction guarantee documentation for storage-mode destructive API warnings
affects: [storage-mode, unified-cache, transaction-guarantees, phase-31]

tech-stack:
  added: []
  patterns:
    - Warning-first storage-mode destructive API helper
    - RuntimeWarning plus operational logger signal for durable-entry deletion risk

key-files:
  created:
    - .planning/phases/31-security-storage-mode-posture/31-05-SUMMARY.md
  modified:
    - src/cacheness/core.py
    - docs/TRANSACTION_GUARANTEES.md
    - tests/test_storage_mode.py
    - .planning/phases/31-security-storage-mode-posture/deferred-items.md

key-decisions:
  - "STRG-01 keeps destructive storage-mode APIs warning-first rather than hard-refusal by default."
  - "Storage-mode destructive warnings use both cacheness.core logger.warning and RuntimeWarning for operational and Python-level visibility."
  - "Implicit storage-mode TTL, eviction, and invalid-entry deletion behavior remains disabled."

patterns-established:
  - "Storage-mode destructive API guard: call _warn_storage_mode_destructive_api before explicit cleanup paths that can delete durable entries."

requirements-completed: [STRG-01]

duration: 9min
completed: 2026-06-15
---

# Phase 31 Plan 05: Storage-Mode Destructive API Warning Policy Summary

**Storage-mode destructive cleanup APIs now emit operational and RuntimeWarning signals before deleting durable entries while preserving warning-first compatibility.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-06-15T02:06:32Z
- **Completed:** 2026-06-15T02:15:41Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added RED regressions for storage-mode `cleanup_expired(ttl_seconds=...)`, `clear_all()`, `clear_all_namespaces()`, and forced size-limit cleanup warning behavior.
- Implemented a shared warning helper in `UnifiedCache` that emits both `logger.warning` and `warnings.warn(..., RuntimeWarning)`.
- Preserved the default warning-first behavior: destructive APIs still perform their documented deletion behavior and do not hard-raise.
- Updated `docs/TRANSACTION_GUARANTEES.md` to document the warning-first policy and the storage-mode no-implicit-delete invariants.

## Task Commits

1. **Task 1: Add storage-mode destructive warning regressions** - `655ade2` (test)
2. **Task 2: Implement warning-first policy for storage-mode destructive APIs** - `ce91e52` (feat)

**Plan metadata:** pending final docs commit or skipped by GSD commit helper.

## Files Created/Modified

- `src/cacheness/core.py` - Adds `_warn_storage_mode_destructive_api()` and calls it from destructive cleanup paths when storage mode is active.
- `tests/test_storage_mode.py` - Adds STRG-01 warning-first regressions and keeps no-implicit-TTL/eviction assertions separate.
- `docs/TRANSACTION_GUARANTEES.md` - Documents warning-first destructive API behavior for storage mode.
- `.planning/phases/31-security-storage-mode-posture/deferred-items.md` - Records out-of-scope verification blockers encountered during Plan 31-05.

## Decisions Made

- Kept warning-first semantics rather than default hard refusal, matching D-29 through D-34.
- Used both logging and Python warnings so service operators and test callers can both observe destructive storage-mode cleanup.
- Warn only when a storage-mode destructive cleanup path can delete entries; `_cleanup_expired()` remains silent when storage mode has no TTL configured.

## Deviations from Plan

None - plan executed as written. Out-of-scope baseline failures were documented rather than fixed.

## Issues Encountered

- Repository pytest addopts expanded the exact plan command into a broad suite, which hit unrelated baseline failures outside Plan 31-05.
- With addopts cleared, `tests/test_storage_mode.py tests/test_core.py` reaches the new STRG-01 tests successfully but later fails on the pre-existing same-key overwrite rollback regression `tests/test_storage_mode.py::TestNoAutoDelete::test_failed_same_key_overwrite_preserves_previous_blob`.
- `uv` default cache initialization failed on this Windows checkout; checks were rerun with repo-local `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR`.
- Pytest default temp discovery later hit Windows ACL errors; the green focused run used workspace-relative `--basetemp .pytest-tmp-31-05`.
- `ty check` still reports pre-existing mixin/test typing diagnostics on the plan file list. Ruff format/check pass.
- The generated `.pytest-tmp-31-05` directory could not be removed due Windows ACL denial. It was left unstaged.

## Verification

- RED: `uv run --python 3.12 pytest -o addopts='' tests/test_storage_mode.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` - FAILED as expected on missing `RuntimeWarning` for `cleanup_expired(ttl_seconds=...)`.
- Focused STRG-01 gate: `uv run --python 3.12 pytest -o addopts='' -p no:cacheprovider --basetemp .pytest-tmp-31-05 tests/test_storage_mode.py::TestStorageModeDestructiveWarnings tests/test_storage_mode.py::TestNoTTLExpiration::test_get_ignores_stored_past_expires_at tests/test_storage_mode.py::TestNoSizeEviction::test_enforce_size_limit_noop tests/test_storage_mode.py::TestStorageModeEndToEnd::test_entries_persist_after_cleanup_calls tests/test_core.py::TestCacheness::test_storage_mode_size_enforcement_remains_disabled tests/test_cache_integrity.py -q --ignore=tests/test_tensorflow_handler.py` - PASSED, 18 passed.
- Plan command with addopts cleared: `uv run --python 3.12 pytest -o addopts='' tests/test_storage_mode.py tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` - FAILED after the new STRG-01 tests passed, on pre-existing same-key overwrite rollback behavior.
- `uv run --python 3.12 ruff format src/cacheness/core.py src/cacheness/_storage_mode_mixin.py tests/test_storage_mode.py tests/test_core.py tests/test_cache_integrity.py` - PASSED after repo-local UV cache workaround.
- `uv run --python 3.12 ruff check --fix src/cacheness/core.py src/cacheness/_storage_mode_mixin.py tests/test_storage_mode.py tests/test_core.py tests/test_cache_integrity.py` - PASSED.
- `uv run --python 3.12 ruff check src/cacheness/core.py src/cacheness/_storage_mode_mixin.py tests/test_storage_mode.py tests/test_core.py tests/test_cache_integrity.py` - PASSED.
- `uv run --python 3.12 ty check src/cacheness/core.py src/cacheness/_storage_mode_mixin.py tests/test_storage_mode.py tests/test_core.py tests/test_cache_integrity.py` - FAILED on pre-existing diagnostics documented in deferred items.

## Known Stubs

None. Stub scan only matched existing initialization log text containing "not available".

## Threat Flags

None. This plan narrows an existing storage-mode destructive API trust boundary and does not add endpoints, auth paths, new file-access patterns beyond existing cleanup calls, or schema changes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

STRG-01 is ready for downstream Phase 31 storage-mode durability work. Plan 31-06 can document fsync and durability separately without needing to revisit the destructive API warning policy.

## Self-Check: PASSED

- Verified summary and key source/docs/test files exist.
- Verified task commits `655ade2` and `ce91e52` are present in git history.

---
*Phase: 31-security-storage-mode-posture*
*Completed: 2026-06-15*
