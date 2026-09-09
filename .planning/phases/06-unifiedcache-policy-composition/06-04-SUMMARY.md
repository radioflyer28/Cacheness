---
phase: 06-unifiedcache-policy-composition
plan: "04"
subsystem: cache-policy
tags: [python, unified-cache, decorators, blobstore, cache-policy]
requires:
  - phase: 06-unifiedcache-policy-composition
    provides: Presence-bearing lookup results and bounded exact-generation removal reports.
provides:
  - Explicit-cache function decorator with normalized, namespace-isolated keys.
  - Outcome-aware decorator recomputation and observable lookup diagnostics.
  - Function-scoped bounded invalidation backed by canonical BlobStore truth.
affects: [06-05-topology-policy, 06-06-public-cutover, cache-policy-api]
actuals:
  tokens: 14538
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Decorators consume UnifiedCache lookup_call, put_call, and invalidate_function without owning lifecycle resources.
    - Function identity is a qualified, queryable catalog namespace with normalized bound arguments in the policy key.
key-files:
  created:
    - tests/test_phase6_decorator_contract.py
  modified:
    - src/cacheness/core.py
    - src/cacheness/decorators.py
    - tests/test_decorators.py
    - tests/test_cache_key_consistency.py
key-decisions:
  - "UnifiedCache derives normalized function keys and persists the qualified function namespace as an authenticated catalog field."
  - "The only decorator requires an explicit UnifiedCache and defaults to recomputing exactly absent and expired lookup outcomes."
  - "Function clearing delegates to bounded exact-generation invalidation and returns its unmodified CacheRemovalReport."
patterns-established:
  - "Cached None is determined exclusively by CacheLookupResult.outcome, never by truthiness or a second read."
  - "Failure suppression is an immutable, explicit outcome policy; the original lookup result remains observable on the wrapper."
requirements-completed: [CACH-02, CACH-03, CACH-04, CACH-06]
coverage:
  - id: D1
    description: Explicit decorators cache None and arbitrary values with qualified-function isolation and normalized call keys.
    requirement: CACH-04
    verification:
      - kind: unit
        ref: tests/test_phase6_decorator_contract.py
        status: pass
      - kind: unit
        ref: tests/test_decorators.py
        status: pass
    human_judgment: false
  - id: D2
    description: Default and explicit recomputation policies preserve typed lookup outcomes and causes.
    requirement: CACH-02
    verification:
      - kind: unit
        ref: tests/test_phase6_decorator_contract.py#test_default_decorator_preserves_failure_outcomes_without_recomputing
        status: pass
    human_judgment: false
  - id: D3
    description: Decorator clearing uses one bounded, exact-generation function namespace removal and reports its actual outcome.
    requirement: CACH-03
    verification:
      - kind: unit
        ref: tests/test_phase6_decorator_contract.py#test_function_clear_preserves_a_concurrently_replaced_generation
        status: pass
      - kind: unit
        ref: tests/test_phase6_removal_contract.py
        status: pass
    human_judgment: false
  - id: D4
    description: No decorator-owned cache, atexit cleanup, weak-reference registry, or close path remains.
    requirement: CACH-06
    verification:
      - kind: unit
        ref: tests/test_phase6_decorator_contract.py#test_explicit_decorator_module_has_no_implicit_lifecycle_owner
        status: pass
    human_judgment: false
duration: 10 min
completed: 2026-09-09
status: complete
---

# Phase 06 Plan 04: Explicit Decorator Policy Summary

**One explicit `cached` decorator now drives normalized function calls through caller-owned UnifiedCache policy, preserving stored None hits, typed failures, and truthful function-scoped clearing.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-09-09T03:04:59Z
- **Completed:** 2026-09-09T03:15:24Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Moved function qualification, signature binding/default application, and deterministic call-key generation into `UnifiedCache`.
- Persisted each decorated function's stable qualified namespace as authenticated, queryable catalog data and reused bounded exact removal for `cache_clear`.
- Replaced implicit decorator construction, cleanup registries, aliases, and broad error suppression with an explicit cache plus immutable recompute outcomes.
- Added proof for None hits, function isolation, canonical argument normalization, typed failure preservation, explicit fallback diagnostics, and replacement-safe clearing.

## Verification

- `uv run --frozen pytest -q tests/test_phase6_decorator_contract.py tests/test_phase6_removal_contract.py -o log_cli=false` — 18 passed.
- `uv run --frozen pytest -q tests/test_decorators.py tests/test_cache_key_consistency.py tests/test_phase6_decorator_contract.py tests/test_phase6_removal_contract.py -o log_cli=false` — 41 passed; one pre-existing pytest collection warning for `TestDataClassForConsistency`.
- `uv run --frozen ruff check src/cacheness/core.py src/cacheness/decorators.py tests/test_decorators.py tests/test_cache_key_consistency.py tests/test_phase6_decorator_contract.py tests/test_phase6_removal_contract.py` — passed.

## Task Commits

1. **Task 1: Cache one None-returning function through an explicitly supplied cache**
   - `cd272de` — test(06-04): add failing explicit decorator tracer
   - `d9e6d89` — feat(06-04): add explicit function cache policy
2. **Task 2: Enforce outcome-aware recomputation and truthful function clearing**
   - `d45a1ac` — test(06-04): cover decorator outcomes and clearing

## Files Created/Modified

- `src/cacheness/core.py` — owns qualified function namespace/key derivation, direct call lookup/write helpers, and scoped invalidation.
- `src/cacheness/decorators.py` — provides the single explicit-cache decorator and outcome policy.
- `tests/test_phase6_decorator_contract.py` — covers cache ownership, None, normalization, failure outcomes, and truthful clearing.
- `tests/test_decorators.py` and `tests/test_cache_key_consistency.py` — preserve canonical decorator regression coverage after the pre-production public cutover.

## Decisions Made

- `function_namespace` persists the stable `module.qualname` identity in the authoritative catalog rather than reverse-parsing opaque keys.
- The wrapper exposes its latest original lookup result after explicit recomputation; the result is not relabeled as a generic miss.
- `cache_clear` remains a bounded one-page operation, so callers retain continuation/retry truth from the canonical removal report.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Migrated directly conflicting legacy decorator regressions**
- **Found during:** Final plan verification.
- **Issue:** `tests/test_decorators.py` imported retired `cache_function`, `memoize`, and `CacheContext`, preventing collection after the approved explicit-only decorator cutover.
- **Fix:** Replaced only those legacy ownership/alias assertions with explicit-cache regressions and migrated the directly affected key-normalization assertions.
- **Files modified:** `tests/test_decorators.py`, `tests/test_cache_key_consistency.py`.
- **Verification:** Canonical decorator, key-normalization, and bounded-removal suite passed (41 tests).
- **Committed in:** `3532e11`.

### TDD Sequencing Note

Task 2's newly added outcome and clear tests passed immediately because Task 1's shared decorator boundary necessarily implemented the explicit immutable recompute policy and `invalidate_function` link. The test coverage was committed without artificial production churn.

**Total deviations:** 1 auto-fixed blocking regression migration.
**Impact on plan:** The approved pre-production cutover remains explicit-only; no compatibility adapter, lifecycle owner, lock, queue, or storage coordinator was added.

## Issues Encountered

- A sandboxed final verification could not write the existing `uv` cache. Re-running the same frozen command with the required cache access passed; this did not change project dependencies.

## Known Stubs

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Subsequent public-surface work can expose only the explicit decorator contract.
- BlobStore remains the sole storage lifecycle authority; decorator code adds no lifecycle coordination or ownership.

## Self-Check: PASSED

- Confirmed all five implementation/test artifacts and the summary exist on disk.
- Confirmed task commits `cd272de`, `d9e6d89`, `d45a1ac`, and `3532e11` exist in Git history.

---
*Phase: 06-unifiedcache-policy-composition*
*Completed: 2026-09-09*
