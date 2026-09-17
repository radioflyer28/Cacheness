---
phase: 10-remove-sqlcache-pull-through-subsystem
plan: 05
subsystem: public-api
tags: [sqlcache-removal, package-surface, pytest, ruff]
requires:
  - phase: 10-remove-sqlcache-pull-through-subsystem
    provides: Inverted verifier, quality, documentation, and dedicated-asset contracts
provides:
  - Physical removal of the SqlCache runtime module and top-level public names
  - Minimal shared CacheReason vocabulary with only SqlCache-exclusive values pruned
  - Removal of all dedicated SqlCache test modules without suite-manifest exclusions
affects: [phase-10-cutover, package-surface, test-collection, wheel-acceptance]
actuals:
  tokens: 32091
  tasks: 3
  commits: 3
tech-stack:
  added: []
  patterns:
    - Direct product removal uses ordinary Python absence rather than compatibility behavior
    - Shared error vocabularies are pruned only after repository-wide caller proof
key-files:
  created: []
  modified:
    - src/cacheness/__init__.py
    - src/cacheness/error_handling.py
  deleted:
    - src/cacheness/sql_cache.py
    - tests/test_sql_cache.py
    - tests/test_sql_cache_documentation.py
    - tests/test_sql_cache_failure_contract.py
key-decisions:
  - "Delete SqlCache, SqlCacheAdapter, and their source module outright; stale imports now fail through ordinary ImportError or ModuleNotFoundError."
  - "Retain BlobStore, UnifiedCache, handlers, and PostgreSQL lifecycle authority unchanged while deleting only four SqlCache-exclusive error reasons."
patterns-established:
  - "Dedicated tests are deleted only after fixed verifier and environment manifests no longer select them."
requirements-completed: [CACH-07]
coverage:
  - id: D1
    description: The SqlCache runtime module and top-level public names are physically absent while the package version remains 0.3.14.
    requirement: CACH-07
    verification:
      - kind: unit
        ref: "tests/test_phase10_sqlcache_removal.py::test_public_names_and_module_are_naturally_absent; tests/test_phase10_sqlcache_removal.py::test_package_version_remains_unchanged"
        status: pass
    human_judgment: false
  - id: D2
    description: The shared CacheReason hierarchy retains its exact supported vocabulary without the four SqlCache-only reasons.
    requirement: CACH-07
    verification:
      - kind: unit
        ref: "tests/test_public_api_contract.py tests/test_phase6_public_api_contract.py tests/test_phase6_suite_isolation.py"
        status: pass
    human_judgment: false
  - id: D3
    description: Dedicated SqlCache test modules are absent and the remaining suite collects without exclusions.
    requirement: CACH-07
    verification:
      - kind: unit
        ref: "tests/test_phase10_sqlcache_removal.py::test_removed_source_and_dedicated_tests_are_absent; pytest --collect-only"
        status: pass
    human_judgment: false
  - id: D4
    description: BlobStore composition, built-in handlers, and PostgreSQL authority contracts remain intact after the hard cut.
    requirement: CACH-07
    verification:
      - kind: integration
        ref: "tests/test_blob_store_composition.py tests/test_handlers.py tests/contracts/test_postgresql_lifecycle_authority.py"
        status: pass
    human_judgment: false
duration: 4 min
completed: 2026-09-17
status: complete
---

# Phase 10 Plan 05: Direct SqlCache Source and Public Cut Summary

**SqlCache is physically absent from the package and test suite; BlobStore, UnifiedCache, handlers, and PostgreSQL authority contracts remain verified.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-09-17T13:33:01-04:00
- **Completed:** 2026-09-17T13:36:57-04:00
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Deleted `cacheness.sql_cache`, `SqlCache`, and `SqlCacheAdapter` with no aliases, package hooks, tombstone, or replacement query-cache surface; package version remains `0.3.14`.
- Removed the three operation-specific SQL reasons and the missing-optional-dependency reason proved exclusive to the retired subsystem, retaining the shared typed error hierarchy.
- Deleted all dedicated SqlCache tests after the fixed manifests were inverted, and preserved full collection plus focused BlobStore, handler, and PostgreSQL lifecycle coverage.

## Task Commits

1. **Task 1: Delete the runtime and top-level public surface** - `9c67f59` (feat)
2. **Task 2: Prune only the orphan public error reasons** - `19b90a1` (refactor)
3. **Task 3: Delete the three dedicated SqlCache test modules** - `63b4fb4` (test)

## Files Created/Modified

- `src/cacheness/__init__.py` - Retains the BlobStore/UnifiedCache barrel without SQL pull-through exports.
- `src/cacheness/error_handling.py` - Retains the shared typed errors without retired-only reason values.
- `src/cacheness/sql_cache.py` - Deleted SQL pull-through implementation.
- `tests/test_sql_cache.py` - Deleted dedicated runtime suite.
- `tests/test_sql_cache_documentation.py` - Deleted dedicated documentation suite.
- `tests/test_sql_cache_failure_contract.py` - Deleted dedicated failure-contract suite.

## Decisions Made

- Direct physical absence is the stale-import contract; no compatibility behavior or tailored error was introduced.
- `MISSING_OPTIONAL_DEPENDENCY` was removed with the three SQL operation reasons because repository-wide caller inspection found no retained use.

## Verification

- `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase10_sqlcache_removal.py::test_public_names_and_module_are_naturally_absent tests/test_phase10_sqlcache_removal.py::test_package_version_remains_unchanged -x` — pass
- `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_public_api_contract.py tests/test_phase6_public_api_contract.py tests/test_phase6_suite_isolation.py -x` — 51 passed
- `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase10_sqlcache_removal.py::test_removed_source_and_dedicated_tests_are_absent -x` — pass
- `uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only` — pass
- `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_blob_store_composition.py tests/test_handlers.py tests/contracts/test_postgresql_lifecycle_authority.py -x` — 81 passed
- `uv run --isolated --group dev --frozen ruff check src/cacheness/__init__.py src/cacheness/error_handling.py tests/test_public_api_contract.py tests/test_phase6_public_api_contract.py` — pass

## Deviations from Plan

None - plan executed exactly as written. The Wave 0 public-contract test edits were already committed by Plan 10-01 and became green after this plan's runtime/error deletion, so no duplicate test edits were needed.

## Issues Encountered

None. The initial frozen test invocation needed the existing user-scoped `uv` cache, then ran successfully with the approved environment access.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The runtime/test product has been removed; dependency, packaging, and final acceptance plans can prove the pruned distribution boundary.
- Current storage and cache lifecycle ownership remains confined to BlobStore and UnifiedCache, with no lifecycle or topology change in this plan.

## Self-Check: PASSED

- Confirmed the surviving public barrel, error model, and summary exist.
- Confirmed the runtime module and three dedicated test modules are absent.
- Confirmed task commits `9c67f59`, `19b90a1`, and `63b4fb4` exist in Git history.

---
*Phase: 10-remove-sqlcache-pull-through-subsystem*
*Completed: 2026-09-17*
