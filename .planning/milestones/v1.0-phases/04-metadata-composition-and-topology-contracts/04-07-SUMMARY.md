---
phase: 04-metadata-composition-and-topology-contracts
plan: "07"
subsystem: testing
tags: [pytest, blobstore, storetopology, projections, public-api]
requires:
  - phase: 04-06
    provides: "StoreTopology direct-store test pattern and lifecycle contract"
provides:
  - "Mixed-scope cache and configuration consumers staged for the narrow BlobStore composition"
  - "Projection-only replacements for retired metadata registry and runtime ORM-session tests"
  - "Public API absence contract for retired metadata authority exports"
affects: [04-08, unified-cache, public-api, projections]
actuals:
  tokens: 49479.75
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - "Direct BlobStore tests construct StoreTopology explicitly."
    - "Derived consumers use the shared ProjectionRole and ProjectionSink protocol."
key-files:
  created: []
  modified:
    - "tests/test_config_validation.py"
    - "tests/test_phase3_local_workflows.py"
    - "tests/test_phase3_scheduler_retirement.py"
    - "tests/test_unified_cache_adversarial_lifecycle.py"
    - "tests/test_unified_cache_lifecycle_authority.py"
    - "tests/test_metadata.py"
    - "tests/test_metadata_backend_registry.py"
    - "tests/test_postgresql_backend.py"
    - "tests/test_public_api_contract.py"
    - "tests/test_custom_metadata.py"
    - "tests/test_cached_custom_metadata.py"
    - "tests/test_projection_mutation_contract.py"
    - "tests/test_projection_sql_atomicity.py"
key-decisions:
  - "Retain mixed-scope UnifiedCache policy assertions while changing only obsolete metadata-selector setup."
  - "Represent JSON, PostgreSQL, and former ORM read models solely as derived ProjectionSink consumers."
  - "Make retired metadata authority exports an explicit public-facade absence contract for Plan 04-08."
patterns-established:
  - "Test-only staging can be red solely for one planned atomic source cutover, but must remain collection-clean."
  - "Projection tests prove checkpoint-after-apply, idempotent replay, committed-partial, and isolated rebuild behavior without authority mutation."
requirements-completed: [BACK-02, BACK-03, BACK-06, BACK-07]
coverage:
  - id: D1
    description: "Mixed-scope cache and configuration consumers use explicit StoreTopology while retaining policy coverage."
    requirement: BACK-03
    verification:
      - kind: unit
        ref: "uv run --frozen pytest --collect-only -q tests/test_config_validation.py tests/test_phase3_local_workflows.py tests/test_phase3_scheduler_retirement.py tests/test_unified_cache_adversarial_lifecycle.py tests/test_unified_cache_lifecycle_authority.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "Legacy metadata and ORM consumers are staged as derived projection contracts without authority selection paths."
    requirement: BACK-02
    verification:
      - kind: unit
        ref: "uv run --frozen pytest --collect-only -q tests/test_metadata.py tests/test_metadata_backend_registry.py tests/test_postgresql_backend.py tests/test_public_api_contract.py tests/test_custom_metadata.py tests/test_cached_custom_metadata.py tests/test_projection_mutation_contract.py tests/test_projection_sql_atomicity.py"
        status: pass
    human_judgment: false
duration: 21min
completed: 2026-09-08
status: complete
---

# Phase 04 Plan 07: Test Consumer Preparation Summary

**Thirteen cache, metadata, public-facade, and projection consumer suites now target explicit StoreTopology and derived ProjectionSink contracts ahead of the atomic source cutover.**

## Performance

- **Duration:** 21 min
- **Started:** 2026-09-08T03:25:14Z
- **Completed:** 2026-09-08T03:46:09Z
- **Tasks:** 2
- **Files modified:** 13

## Accomplishments

- Kept configuration, lifecycle, scheduler, adversarial, and authority-delegation coverage while replacing only obsolete metadata-selector setup.
- Replaced metadata factories, registries, PostgreSQL authority claims, and runtime ORM-session mutation checks with shared projection-role contracts.
- Preserved public handler, CacheConfig, exception-reason, YAML optionality, and SqlCache import coverage while adding the deliberate absence contract for retired metadata exports.

## Task Commits

1. **Task 1: Preserve mixed-scope configuration and UnifiedCache regressions** - `3e08204` (`test`)
2. **Task 2: Preserve public, optional backend, and projection tests while retiring legacy assertions** - `fd4f36f` (`test`)

## Files Created/Modified

- `tests/test_config_validation.py` - removes selector-only metadata expectations while retaining configuration validation.
- `tests/test_phase3_local_workflows.py` and `tests/test_unified_cache_*.py` - preserve policy/lifecycle outcomes using topology-ready setup.
- `tests/test_metadata*.py` and `tests/test_postgresql_backend.py` - replace authority registry selection with explicit role and optional-dependency contracts.
- `tests/test_custom_metadata.py`, `tests/test_cached_custom_metadata.py`, and `tests/test_projection_*.py` - exercise derived reads, checkpoints, idempotency, committed partial outcomes, and offline rebuild publication.
- `tests/test_public_api_contract.py` - retains unrelated public API contracts and asserts the obsolete metadata surface disappears in Plan 04-08.

## Decisions Made

- Mixed-scope tests preserve TTL, eviction, statistics, decorator, global-cache, handler, and SqlCache assertions; this plan only retires obsolete Phase 4 setup and result shapes.
- PostgreSQL and former custom metadata behavior are asserted as derived projections; no test treats them as lifecycle authority or canonical-query completeness.
- The tests intentionally fail only for Plan 04-08’s source removal and internal `UnifiedCache` BlobStore wiring, preserving a clean atomic cutover target.

## Verification

- `uv run --frozen pytest --collect-only -q` over all 13 suites: **passed** (138 tests collected).
- Full targeted run: **121 passed, 17 failed as expected**. Failures are limited to Plan 04-08’s pending `UnifiedCache` StoreTopology wiring, the legacy-source absence guard, and retired public export removal.
- `uv run --frozen python tools/verify_phase4_ruff_delta.py`: **passed**.
- No plan-owned test was skipped or excluded.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test regression] Corrected typed-error construction and current reason-code coverage**
- **Found during:** Task 2
- **Issue:** The new backend-independent error assertion passed `reason=` to a generic error constructor, and the existing public reason-code set omitted Phase 3 catalog and committed-partial reasons.
- **Fix:** Passed the typed reason through the supported context mapping and extended the preserved public reason-code contract to the current enum values.
- **Files modified:** `tests/test_metadata.py`, `tests/test_public_api_contract.py`
- **Verification:** Targeted Task 2 run produced 31 passes and only the intentional Plan 04-08 export-absence failure.
- **Committed in:** `fd4f36f`

**Total deviations:** 1 auto-fixed (1 Rule 1 test regression)

## Issues Encountered

- The locked environment does not include SQLAlchemy, so the retained PostgreSQL test exercises only deterministic optional-import failure rather than a live service. This matches the Phase 4 non-claim and adds no skip.
- The expected red state is source-coupling only: current `UnifiedCache` still calls the retired `BlobStore(..., backend=...)` shape and the package still exports the retired metadata names. Plan 04-08 owns both removals.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 04-08 can atomically wire UnifiedCache through StoreTopology, remove legacy metadata selectors/exports, and delete the obsolete source paths. All known consumer suites now collect without import failures and identify the exact remaining source work.

## TDD Gate Compliance

Both tasks are intentional RED-stage test-consumer commits. The matching source GREEN transition belongs to Plan 04-08, which is the plan authorized to perform the atomic public cutover.

---
*Phase: 04-metadata-composition-and-topology-contracts*
*Completed: 2026-09-08*

## Self-Check: PASSED

- All 13 plan-owned test files exist.
- Task commits `3e08204` and `fd4f36f` exist in git history.
