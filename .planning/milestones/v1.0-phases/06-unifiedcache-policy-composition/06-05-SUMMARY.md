---
phase: 06-unifiedcache-policy-composition
plan: "05"
subsystem: cache-storage-composition
tags: [python, unifiedcache, blobstore, topology, lifecycle, ownership]
requires:
  - phase: 06-04
    provides: Explicit cache decorator policy and exact-generation invalidation reports.
  - phase: 05-payload-backends-and-supported-topology-qualification
    provides: Qualified local topology profiles, deterministic remote candidate, and BlobStore composition.
provides:
  - One BlobStore identity per UnifiedCache, selected by injected store or StoreTopology constructor form.
  - Explicit caller-owned versus cache-owned lifecycle behavior without an ownership boolean.
  - A closed cache facade rejects policy statistics while leaving an injected store usable by its caller.
affects: [06-06, 06-07, 06-08, UnifiedCache, BlobStore]
actuals:
  tokens: 3175
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Constructor form expresses BlobStore ownership; UnifiedCache delegates lifecycle only for the topology-created form.
    - Cache policy observers share the explicit closed-facade boundary with cache storage operations.
key-files:
  created:
    - tests/contracts/test_phase6_topology_policy.py
  modified:
    - src/cacheness/core.py
key-decisions:
  - "An injected BlobStore remains caller-owned; a StoreTopology creates the one cache-owned store."
  - "Closing UnifiedCache also closes the policy-observer boundary without a lifecycle coordinator or changed BlobStore close outcomes."
patterns-established:
  - "Cache storage lifecycle delegates to one BlobStore and never closes topology participants directly."
requirements-completed: [CACH-01, CACH-02, CACH-06]
coverage:
  - id: D1
    description: One cache lifecycle is shared by memory/memory, SQLite/filesystem, and a deterministic PostgreSQL/S3 candidate without live-service claims.
    requirement: CACH-01
    verification:
      - kind: integration
        ref: uv run --frozen pytest -q tests/contracts/test_phase6_topology_policy.py -k "profile or preflight or identity or candidate" -o log_cli=false
        status: pass
    human_judgment: false
  - id: D2
    description: UnifiedCache holds one BlobStore and uses constructor form for caller-owned or cache-owned lifecycle delegation.
    requirement: CACH-02
    verification:
      - kind: integration
        ref: uv run --frozen pytest -q tests/contracts/test_phase6_topology_policy.py tests/test_blob_store_close_contract.py -o log_cli=false
        status: pass
    human_judgment: false
  - id: D3
    description: Closed cache facades reject policy observers while injected BlobStores remain usable by their callers.
    requirement: CACH-06
    verification:
      - kind: unit
        ref: tests/contracts/test_phase6_topology_policy.py#test_cache_close_closes_the_facade_boundary_without_releasing_injected_store
        status: pass
    human_judgment: false
duration: 5min
completed: 2026-09-09
status: complete
---

# Phase 06 Plan 05: UnifiedCache Policy Composition Summary

**UnifiedCache now composes exactly one BlobStore, preserves constructor-form lifecycle ownership, and closes all cache policy entry points without taking over caller-owned stores.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-09-09T03:30:50Z
- **Completed:** 2026-09-09T03:35:24Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Exercised the same cache lifecycle against qualified memory/memory and SQLite/filesystem profiles plus an explicitly deterministic PostgreSQL/S3 candidate.
- Restricted the cache facade to one injected or topology-created BlobStore and its existing AuthorityLifecycleEngine.
- Made a closed facade reject statistics as well as storage policy calls, without closing a caller-owned store.

## Task Commits

1. **Task 1: Run one cache lifecycle through each declared topology profile** — `99307e3` (RED), `92492dc` (GREEN)
2. **Task 2: Make store ownership and close outcomes explicit** — `99bfde2` (RED), `3945759` (GREEN)

## Files Created/Modified

- `src/cacheness/core.py` — Composes the selected BlobStore and prevents derived policy observation after facade close.
- `tests/contracts/test_phase6_topology_policy.py` — Covers local/candidate topology routing, ownership, and closed-facade behavior.

## Decisions Made

- Constructor form, not a compatibility ownership flag, determines whether UnifiedCache delegates lifecycle to the selected BlobStore.
- `statistics()` is a policy observer and rejects calls after facade close; this does not alter canonical BlobStore state or caller-owned lifecycle resources.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

The Task 2 RED test exposed that `statistics()` remained callable after cache close. Decorating that read-only facade method with the existing close guard resolved the mismatch without new coordination or changed storage ownership.

## Known Stubs

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

The one-store cache composition and ownership contract is covered for the supported local profiles. PostgreSQL/S3 remains deterministic-candidate evidence only; Phase 8 retains the live BACK-05 qualification gate.

## Self-Check

PASSED — both changed source artifacts and all four task commits are present.
