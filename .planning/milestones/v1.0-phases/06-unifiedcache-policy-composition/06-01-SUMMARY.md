---
phase: 06-unifiedcache-policy-composition
plan: "01"
subsystem: cache-policy
tags: [python, blobstore, unified-cache, cache-outcomes, statistics]
requires:
  - phase: 05-payload-backends-and-supported-topology-qualification
    provides: Explicit BlobStore compositions and typed direct-read failures.
provides:
  - Presence-bearing UnifiedCache lookup results over exactly one BlobStore observation.
  - Immutable six-outcome lookup statistics that never inspect or mutate storage.
affects: [06-02-removal-policy, 06-04-decorators, 06-06-public-cutover, 06-07-documentation, 06-08-verifier]
actuals:
  tokens: 6444
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - "Single BlobStore snapshot translated once into a presence-bearing cache result."
    - "Best-effort derived outcome recording with frozen, no-I/O snapshots."
key-files:
  created:
    - src/cacheness/cache_policy.py
    - tests/test_phase6_statistics.py
  modified:
    - src/cacheness/core.py
    - tests/test_phase6_lookup_contract.py
key-decisions:
  - "Cache policy maps only declared BlobStore read failures to CORRUPT, CONFLICT, or BACKEND_ERROR and preserves the original cause."
  - "CacheStatistics is derived solely from six classified outcomes; it never lists, queries, or mutates the catalog."
patterns-established:
  - "A stored None is a HIT because presence comes from BlobStore.open_entry(), not from the value."
  - "One lookup records one final outcome after its single storage observation."
requirements-completed: [CACH-02, CACH-04, CACH-05, CACH-06]
coverage:
  - id: D1
    description: "Explicit UnifiedCache composition reads a stored None through one BlobStore observation and returns a HIT."
    requirement: CACH-04
    verification:
      - kind: unit
        ref: "tests/test_phase6_lookup_contract.py#test_explicit_memory_composition_returns_hit_for_cached_none"
        status: pass
      - kind: unit
        ref: "tests/test_phase6_lookup_contract.py#test_lookup_observes_blob_store_once_for_absent_and_present_none"
        status: pass
    human_judgment: false
  - id: D2
    description: "Typed direct-read failures retain their cause and classify as corrupt, conflict, or backend error without cleanup."
    requirement: CACH-02
    verification:
      - kind: unit
        ref: "tests/test_phase6_lookup_contract.py#test_lookup_classifies_declared_storage_failures_without_cleanup"
        status: pass
      - kind: unit
        ref: "tests/test_phase6_lookup_contract.py#test_lookup_propagates_unclassified_programming_errors"
        status: pass
      - kind: unit
        ref: "tests/test_blob_store_translation_seam.py"
        status: pass
    human_judgment: false
  - id: D3
    description: "Frozen CacheStatistics separately counts every lookup outcome and has no catalog or lifecycle authority."
    requirement: CACH-05
    verification:
      - kind: unit
        ref: "tests/test_phase6_statistics.py#test_statistics_snapshot_starts_at_zero_and_is_frozen"
        status: pass
      - kind: unit
        ref: "tests/test_phase6_statistics.py#test_statistics_count_every_outcome_once_and_are_order_independent"
        status: pass
      - kind: unit
        ref: "tests/test_phase6_statistics.py#test_statistics_snapshot_never_observes_or_mutates_the_catalog"
        status: pass
    human_judgment: false
duration: 11 min
completed: 2026-09-09
status: complete
---

# Phase 06 Plan 01: Presence-Bearing Lookup and Statistics Summary

**One explicit UnifiedCache-to-BlobStore lookup now distinguishes stored None, absence, expiry, corruption, conflicts, and backend failures while exposing immutable derived statistics.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-09-09T02:05:11Z
- **Completed:** 2026-09-09T02:16:21Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added the frozen CacheOutcome, CacheLookupResult, and CacheStatistics policy contracts.
- Made lookup classify exactly one BlobStore entry observation, retaining declared typed causes without destructive cleanup for corrupt evidence.
- Added deterministic tests for all six outcomes, immutable/order-independent counters, and no-I/O statistics snapshots.

## Verification

- `uv run --frozen pytest tests/test_phase6_lookup_contract.py tests/test_phase6_statistics.py tests/test_blob_store_translation_seam.py -q -x -o log_cli=false` — 36 passed.
- `uv run --frozen ruff check src/cacheness/cache_policy.py src/cacheness/core.py tests/test_phase6_lookup_contract.py tests/test_phase6_statistics.py` — passed.

## Task Commits

1. **Task 1: Store and retrieve None through one explicit memory composition**
   - `ad9961a` — test(06-01): add failing cache lookup tracer
   - `43fd37b` — feat(06-01): add explicit cache lookup composition
2. **Task 2: Classify every lookup outcome into one immutable statistics snapshot**
   - `2ed6cba` — test(06-01): add failing lookup outcome statistics tests
   - `12f8ed9` — feat(06-01): classify cache lookups into immutable statistics

## Files Created/Modified

- `src/cacheness/cache_policy.py` — frozen cache result/statistics vocabulary and private derived observer.
- `src/cacheness/core.py` — single-read failure classification and statistics access without storage I/O.
- `tests/test_phase6_lookup_contract.py` — typed-error, cause-preservation, and non-destructive failure contracts.
- `tests/test_phase6_statistics.py` — six-outcome aggregate, immutability, ordering, and no-catalog contracts.

## Decisions Made

- Only explicit BlobStore integrity, conflict, and declared operational failures become public cache outcomes; arbitrary programming and control-flow errors still propagate.
- Outcome recording is best-effort derived state, so it cannot establish membership, select a generation, or influence lifecycle work.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 06-02 can consume CacheOutcome, CacheLookupResult, CacheStatistics, and the one-snapshot lookup contract for bounded lifecycle removal policy.

## Self-Check: PASSED

- All four plan implementation/test artifacts exist.
- All four Task 1 and Task 2 TDD commits are present in Git history.
- Coverage metadata validates with three fully automated deliverables.

---
*Phase: 06-unifiedcache-policy-composition*
*Completed: 2026-09-09*
