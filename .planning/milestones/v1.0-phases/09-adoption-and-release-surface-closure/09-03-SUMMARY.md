---
phase: 09-adoption-and-release-surface-closure
plan: 03
subsystem: executable-adoption-examples
tags: [examples, blobstore, unified-cache, format-handler, catalog, pytest]
requires:
  - phase: 09-adoption-and-release-surface-closure
    provides: Alias-free FormatHandler protocol and store-local registration seam
provides:
  - Four exact, disposable BlobStore and UnifiedCache example journeys
  - Network-blocked repeatability harness for the published example files
affects: [09-04, 09-05, 09-06, 09-09, 09-10, phase-10]
actuals:
  tokens: 3172
  tasks: 1
  commits: 2
tech-stack:
  added: []
  patterns:
    - Published examples are the exact subprocess targets of their test harness.
    - Custom formats use store-local FormatHandler registration and private staged paths.
key-files:
  created:
    - tests/test_phase9_examples.py
    - examples/memory_blob_store.py
    - examples/durable_catalog_store.py
    - examples/unified_cache.py
    - examples/custom_mcap_format.py
  modified: []
key-decisions:
  - "Canonical journeys use only memory and local filesystem-plus-SQLite topologies, with no remote-service implication."
  - "The MCAP-style handler exposes stable data type, payload format, payload version, and .mcap suffix through the existing store-local registry."
patterns-established:
  - "Example tests run literal source paths twice from fresh child directories with a child-only socket guard and residue assertion."
requirements-completed: [CACH-06]
coverage:
  - id: D1
    description: Four published direct-storage, durable-catalog, UnifiedCache, and custom-format journeys run unchanged twice and emit deterministic success markers.
    requirement: CACH-06
    verification:
      - kind: integration
        ref: uv run pytest -q -o log_cli=false tests/test_phase9_examples.py -x
        status: pass
    human_judgment: false
  - id: D2
    description: The MCAP-style example proves store-local FormatHandler registration, stable native identity declarations, safe .mcap staging, and byte round trip.
    requirement: CACH-06
    verification:
      - kind: integration
        ref: tests/test_phase9_examples.py#test_canonical_examples_are_exact_repeatable_and_network_free
        status: pass
    human_judgment: false
duration: 18min
completed: 2026-09-17
status: complete
---

# Phase 9 Plan 03: Canonical Executable Examples Summary

**Four exact, self-verifying examples now demonstrate direct memory storage, durable local catalog storage, UnifiedCache policy reuse, and store-local MCAP-style format extension.**

## Performance

- **Duration:** 18 min
- **Completed:** 2026-09-17T03:10:16Z
- **Tasks:** 1
- **Files modified:** 5

## Accomplishments

- Added the four canonical published journeys, each using private temporary resources, explicit initialization, assertions, deterministic close, and one stable marker.
- Added a literal-file subprocess harness that runs every published journey twice from fresh directories with child-only socket blocking, credential/cache-environment sanitization, and residue checks.
- Demonstrated a custom `.mcap` handler registered with `store.handlers.register_handler(..., priority=0)` while retaining the guarded private staging and snapshot boundary.

## Task Commits

1. **Task 1: Publish and execute the exact four canonical example journeys**
   - `337a5df` — `test(09-03): add canonical example harness`
   - `bb9f750` — `feat(09-03): publish canonical storage examples`

## Verification

- `uv run pytest -q -o log_cli=false tests/test_phase9_examples.py -x` — passed (4 tests; each source file ran twice).
- `uv run ruff check tests/test_phase9_examples.py examples/memory_blob_store.py examples/durable_catalog_store.py examples/unified_cache.py examples/custom_mcap_format.py` — passed.
- Scoped diff confirms no Phase-10-owned SqlCache source, tests, or examples changed.

## Decisions Made

- The durable journey scopes its claim to an explicit local filesystem payload plus SQLite lifecycle authority; it does not imply remote or cross-resource ACID behavior.
- The MCAP example uses only the handler-provided private stage/snapshot paths. Its class name is not a persisted identity; `data_type`, `payload_format`, and `payload_format_version` are.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test setup] Created the nested child guard directory with parents.**
- **Found during:** Task 1 RED verification
- **Issue:** The new harness initially failed while creating its private socket guard before it could reach the intended missing-example failure.
- **Fix:** Made the private guard-directory creation create its parent directory.
- **Files modified:** `tests/test_phase9_examples.py`
- **Verification:** The RED test then failed only because the first planned published example did not yet exist; the completed harness passes all four journeys.
- **Committed in:** `337a5df`

**Total deviations:** 1 auto-fixed (Rule 1 test setup).
**Impact on plan:** Required for the exact-file harness to test the intended public example contract; no product or lifecycle scope changed.

## Known Stubs

None.

## Next Phase Readiness

- Plans 09-04 through 09-06 can now remove stale examples without losing an executable adoption baseline.
- No lifecycle authority, concurrency coordination, backend family, compatibility alias, or guarantee changed.

## Self-Check: PASSED

- All five created implementation/test files and both task commits are present in Git history.

*Phase: 09-adoption-and-release-surface-closure*
*Completed: 2026-09-17*
