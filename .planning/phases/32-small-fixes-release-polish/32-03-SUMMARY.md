---
phase: 32
plan: 03
plan_id: 32-03
subsystem: core-cache-hot-path
tags: [small-fix, cache-hit, metadata, ttl, tests]
requires: [POL-03]
provides: [TASK-18, cache-hit-single-metadata-read]
affects:
  - src/cacheness/core.py
  - tests/test_core.py
tech_stack:
  added: []
  patterns:
    - optional already-fetched metadata entry reuse
    - counting regression around metadata backend get_entry
key_files:
  created:
    - .planning/phases/32-small-fixes-release-polish/32-03-SUMMARY.md
  modified:
    - src/cacheness/core.py
    - tests/test_core.py
key_decisions:
  - "POL-03 keeps _is_expired() backward compatible by fetching metadata when no entry is provided, while UnifiedCache.get() passes its already-fetched entry on the cache-hit path."
requirements_completed: [POL-03]
metrics:
  started: "2026-06-15T18:40:06Z"
  completed: "2026-06-15T18:45:47Z"
  duration: "6 min"
  tasks: 1
  files: 2
---

# Phase 32 Plan 03: Metadata Read Hot-Path Summary

Cache-hit expiry checks now reuse the metadata entry already fetched by `UnifiedCache.get()`, avoiding the previous second backend read while preserving existing expiry behavior.

## Completed Work

| Task | Status | Commit | Files |
|------|--------|--------|-------|
| Task 32-03-01: Reuse fetched metadata entry for expiry checks | Complete | 3a471f8 | `src/cacheness/core.py`, `tests/test_core.py` |

## Implementation

- Updated `UnifiedCache._is_expired()` to accept an optional already-fetched metadata entry.
- Kept `_is_expired()` backward compatible for callers that do not provide an entry.
- Updated `UnifiedCache.get()` to pass its existing `entry` into `_is_expired()`.
- Added `test_get_cache_hit_reads_metadata_entry_once`, which wraps the real metadata backend and asserts a cache hit performs one `get_entry()` lookup.

## Verification

| Command | Result |
|---------|--------|
| `uv run pytest tests/test_core.py::TestCacheness::test_get_cache_hit_reads_metadata_entry_once -x -q --ignore=tests/test_tensorflow_handler.py` | RED: failed with `get_entry.call_count == 2` before implementation |
| `uv run pytest tests/test_core.py::TestCacheness::test_get_cache_hit_reads_metadata_entry_once -x -q --ignore=tests/test_tensorflow_handler.py -o addopts=''` | PASS: 1 passed |
| `uv run ruff format src/cacheness/core.py tests/test_core.py` | PASS: 2 files reformatted |
| `uv run ruff check --fix src/cacheness/core.py tests/test_core.py` | PASS: All checks passed |
| `uv run ruff check src/cacheness/core.py tests/test_core.py` | PASS: All checks passed |
| `uv run pytest tests/test_core.py -x -q --ignore=tests/test_tensorflow_handler.py -o addopts=''` | PASS: 64 passed |

## Verification Notes

- The first scoped pytest invocation unexpectedly broadened to the full configured testpaths because repo addopts were applied. Per the plan instruction, later scoped pytest runs used `-o addopts=''`; the plan-level targeted command passed with `64 passed in 9.81s`.
- `uv` could not use the default Windows cache path, so verification used repo-local `.uv-cache` and `.uv-python` directories.
- `uv run ty check src/cacheness/core.py tests/test_core.py` was run and failed on pre-existing diagnostics elsewhere in those files, including `Path | str` typing in `core.py` and older `tests/test_core.py` typing issues. No diagnostic was reported for the new regression test or the `_is_expired(..., entry=entry)` call.

## Deviations from Plan

None - implementation scope matched the plan. The scoped pytest command used `-o addopts=''` only after repo addopts broadened unexpectedly, which the plan explicitly allowed when documented.

## Known Stubs

None.

## Threat Flags

None. This plan did not introduce new network endpoints, auth paths, file access patterns, schema changes, or trust boundaries.

## Deferred Issues

- Existing `ty check src/cacheness/core.py tests/test_core.py` diagnostics remain outside this plan's scope. They predate POL-03 and are not caused by the metadata-entry reuse change.

## Self-Check: PASSED

- `src/cacheness/core.py` exists.
- `tests/test_core.py` exists.
- Commit `3a471f8` exists in git history.
- Summary file exists at `.planning/phases/32-small-fixes-release-polish/32-03-SUMMARY.md`.
