---
phase: 06-unifiedcache-policy-composition
plan: "06"
subsystem: cache-public-api
tags: [python, unifiedcache, blobstore, configuration, public-api]
requires:
  - phase: 06-05
    provides: UnifiedCache policy facade composed over explicit BlobStore ownership
provides:
  - Canonical public cache surface with no singleton, factory, alias, or raw-result compatibility route
  - Nested ownership-aligned CacheConfig construction and BlobStore default configuration
  - Explicit stored-format version facts and typed offline-migration rejection coverage
affects: [07-offline-migration-and-rebuild, public-api, configuration]
actuals:
  tokens: 41757
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Explicit StoreTopology or caller-owned BlobStore selects cache storage identity.
    - Cache configuration changes are expressed through nested ownership sections.
key-files:
  created: []
  modified:
    - src/cacheness/__init__.py
    - src/cacheness/config.py
    - src/cacheness/core.py
    - src/cacheness/storage/blob_store.py
    - tests/test_phase6_public_api_contract.py
key-decisions:
  - "Remove development compatibility APIs outright; do not wrap aliases or revive singleton ownership."
  - "Keep format/schema dimensions authoritative and reject unsupported layouts before implicit mutation."
patterns-established:
  - "Public cache callers construct nested CacheConfig values and explicitly select a BlobStore composition."
requirements-completed: [CACH-01, CACH-02, CACH-03, CACH-04, CACH-05, CACH-06]
coverage:
  - id: D1
    description: Canonical public imports complete an explicit UnifiedCache lifecycle over an explicit topology.
    requirement: CACH-01
    verification:
      - kind: integration
        ref: tests/test_phase6_public_api_contract.py#test_canonical_imports_run_one_explicit_cache_lifecycle
        status: pass
    human_judgment: false
  - id: D2
    description: Removed singleton, factory, raw-result, and flat-config routes do not resolve or delegate.
    requirement: CACH-06
    verification:
      - kind: unit
        ref: tests/test_phase6_public_api_contract.py#test_removed_development_public_names_do_not_delegate
        status: pass
      - kind: unit
        ref: tests/test_phase6_public_api_contract.py#test_removed_cache_compatibility_methods_and_flat_config_are_absent
        status: pass
    human_judgment: false
  - id: D3
    description: Current stored format dimensions remain discoverable and unsupported store layouts require offline migration or rebuild.
    requirement: CACH-06
    verification:
      - kind: unit
        ref: tests/test_phase6_public_api_contract.py#test_explicit_version_dimensions_reject_unsupported_store_layouts
        status: pass
    human_judgment: false
duration: 13min
completed: 2026-09-09
status: complete
---

# Phase 06 Plan 06: Canonical Public API Cutover Summary

**One explicit UnifiedCache/BlobStore configuration and result surface replaces development-era aliases, singletons, factories, and raw cache observer shapes.**

## Performance

- **Duration:** 13 min
- **Started:** 2026-09-09T03:42:16Z
- **Completed:** 2026-09-09T03:55:22Z
- **Tasks:** 2/2
- **Files modified:** 10

## Accomplishments

- Published and exercised the canonical public import/configuration/lifecycle surface with an explicit topology and immutable result types.
- Removed global cache helpers, alternate factories, legacy raw `get`/`get_stats`/`list_entries` views, and all flat `CacheConfig` mapping and property paths.
- Retained authoritative format/schema dimensions and typed rejection of unsupported layouts, preserving Phase 7's offline migration/rebuild boundary.

## Task Commits

1. **Task 1: Import, configure, and run one object through the canonical public surface**
   - `37fc4f5` — `test(06-06): add failing canonical public API tracer`
   - `f6ac9fc` — `feat(06-06): publish canonical cache API tracer`
2. **Task 2: Remove overlapping pre-production APIs while retaining version boundaries**
   - `b746bbb` — `test(06-06): cover removed public compatibility APIs`
   - `679b50e` — `feat(06-06): remove overlapping cache compatibility APIs`

## Files Created/Modified

- `src/cacheness/__init__.py` — Deliberate canonical package exports from the tracer task.
- `src/cacheness/config.py` — Nested configuration only; removed backend selector, flat mapping, property, and factory compatibility paths.
- `src/cacheness/core.py` — Removed alternate cache constructor, singleton lifecycle, and raw observer APIs.
- `src/cacheness/storage/blob_store.py` — Builds its internal default CacheConfig through CacheStorageConfig.
- `tests/test_phase6_public_api_contract.py` — Enforces absent compatibility routes and explicit version rejection.
- `tests/test_phase6_lookup_contract.py`, `tests/test_phase6_removal_contract.py`, `tests/test_phase6_policy_contract.py`, `tests/test_phase6_decorator_contract.py`, and `tests/contracts/test_phase6_topology_policy.py` — Use canonical nested configuration in Phase 6 contracts.

## Decisions Made

- The selected `StoreTopology` or caller-provided `BlobStore`, not a configuration selector or capability flag, is the sole cache storage identity.
- Stored format/schema dimensions remain explicit; unsupported layouts raise a typed offline migration/rebuild error rather than receiving an in-place upgrade.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated BlobStore's internal default configuration path**

- **Found during:** Task 2
- **Issue:** Removing flat `CacheConfig(cache_dir=...)` left direct `BlobStore` construction unable to create its own default nested configuration.
- **Fix:** Constructed `CacheConfig(storage=CacheStorageConfig(cache_dir=...))` while retaining the existing compression configuration.
- **Files modified:** `src/cacheness/storage/blob_store.py`
- **Verification:** The complete Phase 6 contract suite passes, including direct and injected BlobStore cases.
- **Committed in:** `679b50e`

---

**Total deviations:** 1 auto-fixed (1 Rule 3 blocking issue)

**Impact on plan:** Necessary internal migration to make the approved removal functional; no compatibility shim or new lifecycle behavior was added.

## Issues Encountered

None beyond the expected RED test failure and the direct configuration callsite resolved above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 7 can rely on explicit format/schema dimensions and typed migration-or-rebuild rejection when adding offline cutover tooling.
- SqlCache and direct BlobStore remain intentionally separate supported surfaces.

## Verification

- `uv run --frozen pytest -q tests/test_phase6_public_api_contract.py tests/test_phase6_lookup_contract.py tests/test_phase6_removal_contract.py tests/test_phase6_policy_contract.py tests/test_phase6_decorator_contract.py tests/contracts/test_phase6_topology_policy.py -o log_cli=false` — 64 passed.
- `uv run --frozen ruff check src/cacheness/config.py src/cacheness/core.py src/cacheness/storage/blob_store.py tests/test_phase6_public_api_contract.py tests/test_phase6_lookup_contract.py tests/test_phase6_removal_contract.py tests/test_phase6_policy_contract.py tests/test_phase6_decorator_contract.py tests/contracts/test_phase6_topology_policy.py` — passed.

## Self-Check: PASSED

- All listed production and contract files exist.
- Task commits `37fc4f5`, `f6ac9fc`, `b746bbb`, and `679b50e` exist in Git history.

---

*Phase: 06-unifiedcache-policy-composition*
*Completed: 2026-09-09*
