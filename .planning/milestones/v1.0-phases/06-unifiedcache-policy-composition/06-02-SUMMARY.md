---
phase: 06-unifiedcache-policy-composition
plan: "02"
subsystem: cache-policy
tags: [python, blobstore, unified-cache, catalog, invalidation, ttl]
requires:
  - phase: 06-unifiedcache-policy-composition
    provides: Presence-bearing cache lookup outcomes and immutable statistics.
provides:
  - Authenticated catalog policy facts coupled to exact lifecycle expectations.
  - Immutable bounded removal reports for expiry, key, predicate, and global policy removal.
affects: [06-03-size-policy, 06-04-decorators, 06-05-topology-policy, 06-06-public-cutover, 06-08-verifier]
actuals:
  tokens: 8102
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Policy selects authenticated BlobStore snapshots and delegates mutation through exact delete expectations.
    - A bounded catalog page reports opaque continuation or retryable stale-page truth without a policy-side lifecycle coordinator.
key-files:
  created:
    - tests/test_phase6_removal_contract.py
  modified:
    - src/cacheness/storage/catalog.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/cache_policy.py
    - src/cacheness/core.py
key-decisions:
  - "CacheRemovalReport separately accounts for removals, exact-generation conflicts, retryable traversal outcomes, and operational failures."
  - "A catalog cursor invalidated by a successful delete is reported as retryable rather than reused as if it still described current authority state."
patterns-established:
  - "Cache policy never deletes payloads, authority records, or projections directly; its sole mutation callback is BlobStore.delete(key, expected=...)."
  - "Cache entries persist a versioned queryable cache namespace and prefix in authoritative catalog descriptors."
requirements-completed: [CACH-01, CACH-02, CACH-03]
coverage:
  - id: D1
    description: Expired lookups retain CacheOutcome.EXPIRED while exact-generation cleanup reports success, conflict, or typed corruption truth.
    requirement: CACH-02
    verification:
      - kind: unit
        ref: tests/test_phase6_removal_contract.py#test_expired_lookup_reports_the_exact_lifecycle_removal
        status: pass
      - kind: unit
        ref: tests/test_phase6_removal_contract.py#test_expired_lookup_preserves_a_replacement_and_reports_conflict
        status: pass
      - kind: unit
        ref: tests/test_phase6_removal_contract.py#test_malformed_expiry_facts_fail_closed_without_deletion
        status: pass
    human_judgment: false
  - id: D2
    description: Single-key, predicate, and global invalidation share bounded exact-delete reporting with validation, pagination, and failure accounting.
    requirement: CACH-03
    verification:
      - kind: unit
        ref: tests/test_phase6_removal_contract.py
        status: pass
      - kind: unit
        ref: tests/test_catalog_query_contract.py
        status: pass
      - kind: unit
        ref: tests/test_blob_store_integrity.py
        status: pass
    human_judgment: false
duration: 13 min
completed: 2026-09-09
status: complete
---

# Phase 06 Plan 02: Bounded Exact Cache Removal Summary

**UnifiedCache now selects authenticated cache catalog snapshots and delegates truthful, bounded expiry and invalidation deletion to BlobStore's existing lifecycle authority.**

## Performance

- **Duration:** 13 min
- **Started:** 2026-09-09T02:23:54Z
- **Completed:** 2026-09-09T02:36:42Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Added immutable exact-removal reports and failure details while retaining the primary expired lookup outcome.
- Extended catalog pages with authenticated expectation, schema, byte-size, and timestamp facts required by cache policy.
- Made key, predicate, global, and TTL cleanup share BlobStore exact deletion, bounded page traversal, opaque continuation, and stale-page retryability.

## Verification

- `uv run --frozen pytest -q tests/test_phase6_removal_contract.py tests/test_catalog_query_contract.py tests/test_blob_store_integrity.py -o log_cli=false` — 87 passed.
- `uv run --frozen pytest -q tests/test_phase6_lookup_contract.py tests/test_phase6_statistics.py tests/test_blob_store_translation_seam.py -o log_cli=false` — 36 passed.
- Focused Ruff checks for modified source and test files — passed.

## Task Commits

1. **Task 1: Remove one expired entry through the canonical lifecycle**
   - `0aed9aa` — test(06-02): add failing removal policy tracer
   - `8fc83de` — feat(06-02): report exact expired cache removal
2. **Task 2: Expand the bounded removal primitive to explicit invalidation scopes**
   - `ae41869` — test(06-02): add failing bounded invalidation cases
   - `47bf403` — feat(06-02): bound exact cache invalidation

## Files Created/Modified

- `src/cacheness/storage/catalog.py` — exposes authenticated policy facts and exact delete expectations on catalog entries.
- `src/cacheness/storage/blob_store.py` — documents exact expectations on signed catalog pages.
- `src/cacheness/cache_policy.py` — owns frozen removal candidates, reports, and bounded outcome classification only.
- `src/cacheness/core.py` — persists cache catalog fields, validates policy facts, and routes all removal scopes through BlobStore exact deletion.
- `tests/test_phase6_removal_contract.py` — covers expiry, replacement, corrupt facts, all invalidation scopes, page caps, continuation, and backend failures.

## Decisions Made

- Cache policy treats a cursor invalidated by its own exact deletion as a typed retryable partial result; it does not pretend an authority-revision-bound cursor can safely continue after mutation.
- A legacy cache schema/policy-fact mismatch fails closed rather than being upgraded implicitly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Regression contract] Updated direct legacy removal assertions for the intentional report cutover**
- **Found during:** Task 2
- **Issue:** Existing lifecycle tests asserted `None`, an integer, and a direct `BlobStore.clear()` call even though D-10/D-11 require report-returning, exact `BlobStore.delete()` policy paths.
- **Fix:** Updated only the directly conflicting assertions and lifecycle hook to assert actual `CacheRemovalReport` conflict/empty outcomes.
- **Files modified:** `tests/test_unified_cache_lifecycle_authority.py`, `tests/test_unified_cache_adversarial_lifecycle.py`
- **Committed in:** `47bf403`

**Total deviations:** 1 auto-fixed (Rule 1)

## Issues Encountered

The broader legacy facade lifecycle suites still construct caches through the removed implicit `cacheness(config)` route, so they fail before reaching these assertions after Plan 06-01's explicit-store cutover. This is an existing Plan 06-01 compatibility follow-up, not a removal-policy failure; it is recorded in `deferred-items.md` for the later public-cutover/test reconciliation work.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 06-03 can consume `CacheRemovalReport`, authenticated `CatalogEntry` intrinsic facts, and the one bounded exact-delete primitive for resumable size maintenance.

---
*Phase: 06-unifiedcache-policy-composition*
*Completed: 2026-09-09*

## Self-Check: PASSED

- Confirmed summary, removal-policy implementation, and contract test files exist.
- Confirmed task commits `0aed9aa`, `8fc83de`, `ae41869`, and `47bf403` exist in git history.
