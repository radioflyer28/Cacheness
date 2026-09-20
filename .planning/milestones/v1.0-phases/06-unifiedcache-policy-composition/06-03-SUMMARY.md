---
phase: 06-unifiedcache-policy-composition
plan: "03"
subsystem: cache-policy
tags: [python, blobstore, unified-cache, catalog, bounded-maintenance, receipts]
requires:
  - phase: 06-unifiedcache-policy-composition
    provides: Authenticated catalog facts and bounded exact-generation removal reports.
provides:
  - Validated, topology-independent cache policy limits for TTL and size maintenance.
  - Caller-driven, bounded inventory, eviction, and verification continuations over authenticated catalog facts.
  - Immutable committed-write results that retain the BlobStore receipt beside one maintenance outcome.
affects: [06-04-decorators, 06-05-topology-policy, 06-06-public-cutover, cache-policy-api]
actuals:
  tokens: 13013
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - UnifiedCache policy resumes one bounded maintenance phase per explicit caller invocation.
    - CachePutResult separates immutable BlobStore commit truth from retryable policy follow-up truth.
key-files:
  created: []
  modified:
    - src/cacheness/config.py
    - src/cacheness/cache_policy.py
    - src/cacheness/core.py
    - tests/test_phase6_policy_contract.py
    - tests/test_phase6_lookup_contract.py
    - tests/test_phase6_removal_contract.py
    - tests/test_phase6_statistics.py
key-decisions:
  - "CachePutResult holds one unchanged BlobReceipt and one CacheMaintenanceResult, so post-commit policy work cannot rewrite canonical storage truth."
  - "UnifiedCache.put performs exactly one bounded maintenance step; later work requires an explicit validated resume."
  - "The pre-production put-result cutover exposes keys through receipt.key rather than retaining a string-return compatibility layer."
patterns-established:
  - "Cache policy uses only authenticated intrinsic catalog facts and BlobStore exact deletion; statistics and projections remain derived observers."
  - "A complete size-maintenance result is emitted only after a fresh stable verification scan proves the configured byte limit."
requirements-completed: [CACH-02, CACH-03, CACH-06]
coverage:
  - id: D1
    description: Finite, resumable size maintenance validates policy limits, obeys a work cap, preserves replacement winners, and converges for a quiescent multi-page inventory.
    requirement: CACH-03
    verification:
      - kind: unit
        ref: tests/test_phase6_policy_contract.py
        status: pass
    human_judgment: false
  - id: D2
    description: A successful UnifiedCache put returns the canonical BlobReceipt and one truthful complete or retryable maintenance result without rollback.
    requirement: CACH-02
    verification:
      - kind: unit
        ref: tests/test_phase6_policy_contract.py#test_put_retains_canonical_receipt_when_maintenance_is_incomplete
        status: pass
      - kind: integration
        ref: tests/test_blob_store_atomic_lifecycle.py
        status: pass
    human_judgment: false
  - id: D3
    description: Policy configuration remains a validated nested CacheConfig concern separate from BlobStore composition, handlers, and security settings.
    requirement: CACH-06
    verification:
      - kind: unit
        ref: tests/test_phase6_policy_contract.py#test_policy_configuration_rejects_non_finite_or_inconsistent_bounds
        status: pass
    human_judgment: false
duration: 14 min
completed: 2026-09-09
status: complete
---

# Phase 06 Plan 03: Truthful Bounded Size Maintenance Summary

**UnifiedCache now returns each canonical BlobStore receipt alongside one bounded, resumable size-maintenance result, keeping committed storage truth intact through partial policy work.**

## Performance

- **Duration:** 14 min
- **Started:** 2026-09-09T02:47:28Z
- **Completed:** 2026-09-09T03:00:55Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Added frozen `CachePolicyConfig` limits plus signed, composition-bound maintenance continuations for bounded inventory, eviction, and verification.
- Ensured each explicit maintenance call performs one finite step, reports incomplete or retryable outcomes under churn/conflict, and proves the byte limit only after fresh verification.
- Added immutable `CachePutResult`, preserving the canonical `BlobReceipt` when post-commit maintenance completes, remains pending, or reports a typed backend failure.

## Verification

- `uv run --frozen pytest -q tests/test_phase6_policy_contract.py -k "config or size or budget or multipage or resume or quiescent or churn or replacement or ordering" -o log_cli=false` — 7 passed.
- `uv run --frozen pytest -q tests/test_phase6_policy_contract.py tests/test_blob_store_atomic_lifecycle.py -k "put or maintenance or multipage or resume or quiescent or churn or conflict" -o log_cli=false` — 19 passed.
- `uv run --frozen pytest -q tests/test_phase6_lookup_contract.py tests/test_phase6_removal_contract.py tests/test_phase6_statistics.py tests/test_phase6_policy_contract.py -o log_cli=false` — 30 passed.
- `uv run --frozen ruff check src/cacheness/cache_policy.py src/cacheness/core.py tests/test_phase6_lookup_contract.py tests/test_phase6_removal_contract.py tests/test_phase6_statistics.py tests/test_phase6_policy_contract.py` — passed.

## Task Commits

1. **Task 1: Enforce one size limit within a finite maintenance budget**
   - `299f0be` — test(06-03): add failing bounded size policy tracer
   - `c2c50b2` — feat(06-03): add bounded resumable size maintenance
2. **Task 2: Preserve committed put truth across post-commit maintenance**
   - `3c00a7a` — test(06-03): add failing committed-put policy tests
   - `0d56807` — feat(06-03): preserve committed put truth

## Files Created/Modified

- `src/cacheness/config.py` — defines finite cache-policy limits separate from BlobStore composition.
- `src/cacheness/cache_policy.py` — owns immutable maintenance states, results, and receipt-bearing put results.
- `src/cacheness/core.py` — performs one canonical put followed by at most one bounded policy step.
- `tests/test_phase6_policy_contract.py` — proves budget, continuation, churn, replacement, and committed-put partial-success contracts.
- `tests/test_phase6_lookup_contract.py`, `tests/test_phase6_removal_contract.py`, and `tests/test_phase6_statistics.py` — consume canonical keys through `CachePutResult.receipt` after the cutover.

## Decisions Made

- `BlobReceipt` remains the sole canonical record of a committed generation; maintenance is a separate cache-policy result and never a lifecycle authority.
- A put begins but does not drain size maintenance. Callers explicitly provide the returned opaque state to run one later bounded step.
- The public pre-production cutover replaces the string `put()` result with `CachePutResult`; direct callers use `result.receipt.key`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Regression contract] Migrated direct Phase 6 callers to the receipt-bearing put result**
- **Found during:** Task 2
- **Issue:** Existing lookup, removal, and statistics tests passed the new `CachePutResult` where a raw cache key was required.
- **Fix:** Updated only the directly conflicting assertions to pass `result.receipt.key` under the approved pre-production API cutover.
- **Files modified:** `tests/test_phase6_lookup_contract.py`, `tests/test_phase6_removal_contract.py`, `tests/test_phase6_statistics.py`
- **Verification:** Phase 6 policy suite — 30 passed; focused Ruff check passed.
- **Committed in:** `0d56807`

**Total deviations:** 1 auto-fixed (Rule 1)

## Issues Encountered

None beyond the direct Phase 6 contract migrations documented above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Later cache policy and decorator plans can rely on an explicit committed-write result, caller-driven bounded maintenance, and receipt-derived cache keys without introducing another lifecycle owner.

---
*Phase: 06-unifiedcache-policy-composition*
*Completed: 2026-09-09*

## Self-Check: PASSED

- Confirmed the summary, policy implementation, and Phase 6 contract files exist.
- Confirmed task commits `299f0be`, `c2c50b2`, `3c00a7a`, and `0d56807` exist in git history.
