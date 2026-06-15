---
phase: 32-small-fixes-release-polish
plan: 08
subsystem: package-surface
tags: [python, public-api, imports, release-polish]
requires:
  - phase: 32-07
    provides: Runtime package version aligned to 0.12.0.
provides:
  - Root package export for UnifiedCache.
  - Preserved cacheness compatibility alias.
  - Root import regression in tests/test_core.py.
affects: [package-root, public-imports, tests]
tech-stack:
  added: []
  patterns: [root re-export with compatibility alias]
key-files:
  created:
    - .planning/phases/32-small-fixes-release-polish/32-08-SUMMARY.md
  modified:
    - src/cacheness/__init__.py
    - tests/test_core.py
key-decisions:
  - "UnifiedCache is imported by name from cacheness.core and cacheness remains an alias to the same class."
  - "The 0.12.0 runtime version from 32-07 was preserved unchanged."
requirements-completed: [POL-08]
duration: 15 min
completed: 2026-06-15
---

# Phase 32 Plan 08: Root UnifiedCache Export Summary

**Package root now exports UnifiedCache while preserving the legacy cacheness alias for compatibility.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-06-15T20:16:54Z
- **Completed:** 2026-06-15T20:31:43Z
- **Tasks:** 2
- **Files modified:** 2 source/test files plus this summary

## Accomplishments

- Added `UnifiedCache` to the package root import surface.
- Preserved `from cacheness import cacheness` as the same class object.
- Added a regression proving `UnifiedCache` and `cacheness` are both exported in `__all__`.

## Task Commits

1. **Task 32-08-01: Add root UnifiedCache export without removing cacheness alias** - `bd2f2f5` (`feat(32-08): export UnifiedCache from package root`)

**Plan metadata:** pending docs commit

## Files Created/Modified

- `src/cacheness/__init__.py` - Imports `UnifiedCache` by name, assigns `cacheness = UnifiedCache`, and adds `"UnifiedCache"` to `__all__`.
- `tests/test_core.py` - Adds root-import regression for `UnifiedCache`, the legacy alias, and `__all__`.
- `.planning/phases/32-small-fixes-release-polish/32-08-SUMMARY.md` - Captures execution results and checkpoint status.

## Decisions Made

- Kept the existing public alias intact instead of renaming it.
- Left `__version__ = "0.12.0"` unchanged from Plan 32-07.

## Deviations from Plan

None - plan implementation executed exactly as written.

## Issues Encountered

- The default global uv cache path failed with `Cannot create a file when that file already exists`; verification used workspace-local `.uv-cache` and `.uv-python`.
- Sandbox execution of uv commands failed with interpreter access errors, so uv verification commands were run outside the sandbox after approval.
- `uv run ty check src/cacheness/__init__.py tests/test_core.py` reported pre-existing diagnostics throughout `tests/test_core.py`; none were introduced by this plan's changed lines.
- The initial full-suite checkpoint failed outside POL-08 ownership:
  - `tests/test_decorators.py::TestCacheIfDecorator::test_cache_if_supports_ttl_parameter` expected `call_count == 2`, got `1`.
  - `tests/test_fault_injection.py::TestOrphanedBlobOnPutCrash::test_failed_same_key_overwrite_preserves_previous_blob` expected previous value preservation after metadata failure, got `None`.
  - `tests/test_dunder_methods.py::TestDunderMethods::test_contains_expired_key` expected an expired key to remain contained.

Ownership triage: these failures were not caused by the POL-08 package-root export, but the full-suite checkpoint is a Phase 32 release criterion. They were resolved by follow-up commit `4d6e0d1` (`fix(32): resolve final checkpoint regressions`) before phase verification.

## Verification

- RED check: `uv run pytest tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` failed as expected before implementation with `ImportError: cannot import name 'UnifiedCache' from 'cacheness'`.
- Formatting: `uv run ruff format src/cacheness/__init__.py tests/test_core.py` passed; 2 files reformatted.
- Ruff fix: `uv run ruff check --fix src/cacheness/__init__.py tests/test_core.py` passed.
- Ruff validation: `uv run ruff check src/cacheness/__init__.py tests/test_core.py` passed.
- Type check: `uv run ty check src/cacheness/__init__.py tests/test_core.py` failed on pre-existing `tests/test_core.py` diagnostics outside the new regression.
- Focused regression: `uv run pytest -o addopts='' tests/test_core.py::TestCacheConfig::test_unified_cache_root_export_preserves_alias -x -q --ignore=tests/test_tensorflow_handler.py` passed, 1 passed.
- Targeted test file: `uv run pytest -o addopts='' tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py` passed, 65 passed.
- Smoke: `uv run python -c "from cacheness import UnifiedCache, cacheness; assert UnifiedCache is cacheness; print(UnifiedCache.__name__)"` passed and printed `UnifiedCache`.
- Initial final Phase 32 checkpoint: `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` failed with 3 failed, 905 passed, 67 skipped in 100.20s.
- Checkpoint regression fix: `4d6e0d1` restored decorator/write-time TTL persistence, fractional per-entry TTL handling, explicit read-time TTL override behavior, and same-key overwrite rollback in cache and storage modes.
- Final Phase 32 checkpoint rerun: `uv run pytest tests/ -x -q --ignore=tests/test_tensorflow_handler.py` passed with 1857 passed, 125 skipped, 27 warnings in 77.77s.

## Known Stubs

None.

## Threat Flags

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

POL-08 is complete. Phase 32 release completion is ready for verifier review after the final full-suite checkpoint passed.

## Self-Check

PASSED - summary file exists on disk and implementation commit `bd2f2f5` is present in git history.

---
*Phase: 32-small-fixes-release-polish*
*Completed: 2026-06-15*
