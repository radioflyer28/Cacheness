---
phase: 04-metadata-composition-and-topology-contracts
plan: "10"
subsystem: storage-composition
tags: [blob-store, topology, role-registry, lifecycle-authority, ownership]
requires:
  - phase: 04-09
    provides: selected payload-generation I/O and public catalog data flow
provides:
  - One topology-carried RoleRegistry for built-in and application participant selection
  - Runtime structural validation for payload, authority, and projection roles
  - Identity-deduplicated, reverse-order unwind of store-owned participants
affects: [phase-04-plan-11, phase-04-plan-12, phase-04-plan-13, phase-05, phase-06]
actuals:
  tokens: 8950
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - StoreTopology owns the sole application-extensible RoleRegistry.
    - Resolution uses a local identity ledger for normal close and exceptional unwind.
key-files:
  created: []
  modified:
    - src/cacheness/storage/composition.py
    - src/cacheness/storage/backends/blob_backends.py
    - src/cacheness/storage/backends/__init__.py
    - src/cacheness/storage/memory_lifecycle_authority.py
    - src/cacheness/storage/__init__.py
    - src/cacheness/__init__.py
    - tests/test_blob_store_composition.py
    - tests/test_topology_capabilities.py
    - tests/test_public_api_contract.py
key-decisions:
  - "StoreTopology owns exactly one RoleRegistry; BlobStore resolves only that composition root."
  - "Structural protocol checks and a local identity ledger replace concrete cross-role checks and duplicated close paths."
patterns-established:
  - "Record store-owned participants before validation, deduplicate by identity, and close in reverse acquisition order."
  - "Keep in-memory projection backup explicitly unsupported when its advertised projection capability is false."
requirements-completed: [BACK-02, BACK-03, BACK-06]
coverage:
  - id: D1
    description: "Application and built-in roles reach BlobStore through the topology-owned RoleRegistry only."
    requirement: BACK-03
    verification:
      - kind: integration
        ref: "tests/test_blob_store_composition.py"
        status: pass
      - kind: unit
        ref: "tests/test_public_api_contract.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "Payload, authority, and projection participants fail closed unless they satisfy their structural roles."
    requirement: BACK-02
    verification:
      - kind: unit
        ref: "tests/test_blob_store_composition.py"
        status: pass
      - kind: unit
        ref: "tests/test_topology_capabilities.py"
        status: pass
    human_judgment: false
  - id: D3
    description: "Owned invalid or repeated participants close once in reverse acquisition order while caller-owned injections remain open."
    requirement: BACK-06
    verification:
      - kind: unit
        ref: "tests/test_blob_store_composition.py"
        status: pass
      - kind: unit
        ref: "tests/test_topology_capabilities.py"
        status: pass
    human_judgment: false
duration: 13m 10s
completed: 2026-09-08
status: complete
---

# Phase 04 Plan 10: Composition Registry and Ownership Summary

**BlobStore now selects every built-in or application participant through one topology-owned registry, validates role structure before I/O, and unwinds only store-owned identities exactly once.**

## Performance

- **Duration:** 13m 10s
- **Started:** 2026-09-08T02:25:48-04:00
- **Completed:** 2026-09-08T02:38:58-04:00
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Made `RoleRegistry` an explicit `StoreTopology` field and removed the disconnected blob registry, its global state, and its public exports.
- Routed `BlobStore` through the one topology resolution path, including named application payload, authority, and projection participants.
- Enforced runtime structural role contracts and one close-once ownership ledger for normal close and failed construction.

## Task Commits

1. **Task 1 RED: Route application registrations through StoreTopology into BlobStore** - `832b91e` (`test`)
2. **Task 1 GREEN: Route application registrations through StoreTopology into BlobStore** - `ecd2534` (`feat`)
3. **Task 2 RED: Enforce structural roles and close owned invalid resources exactly once** - `b337e74` (`test`)
4. **Task 2 GREEN: Enforce structural roles and close owned invalid resources exactly once** - `e81d0f6` (`feat`)

## Files Created/Modified

- `src/cacheness/storage/composition.py` - Owns the sole registry, validates structural roles, and shares the identity-deduplicated close ledger between failure and normal close.
- `src/cacheness/storage/backends/blob_backends.py` and `src/cacheness/storage/backends/__init__.py` - Remove the retired global blob selector surface.
- `src/cacheness/storage/memory_lifecycle_authority.py` - Explicitly rejects isolated projection backup while satisfying the existing structural authority interface.
- `src/cacheness/storage/__init__.py` and `src/cacheness/__init__.py` - Expose the canonical `RoleRegistry` and `StoreTopology` selection surface only.
- `tests/test_blob_store_composition.py`, `tests/test_topology_capabilities.py`, and `tests/test_public_api_contract.py` - Cover high-level named registrations, structural rejection, unwind order, and retired-export absence.

## Decisions Made

- A `StoreTopology` owns one `RoleRegistry`; passing a registry into `resolve()` or maintaining a second/global fallback path is forbidden.
- Runtime role checks are protocol-based: payloads expose guarded generation I/O, authorities satisfy `LifecycleAuthority`, and projections expose derived apply/checkpoint operations.
- Ownership remains local to one resolution attempt. It is not a shared resource manager, reference counter, or lifecycle coordinator.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Completed the in-memory lifecycle authority's structural contract**
- **Found during:** Task 2
- **Issue:** `InMemoryLifecycleAuthority` was missing the existing `projection_backup` protocol member, so the built-in authority failed the runtime `LifecycleAuthority` check.
- **Fix:** Added an explicit context-manager method that rejects projection backup because the implementation advertises no projection capability.
- **Files modified:** `src/cacheness/storage/memory_lifecycle_authority.py`
- **Verification:** Focused structural composition suite passes.
- **Committed in:** `e81d0f6`

**2. [Rule 2 - Missing critical surface] Re-exported the canonical registry through the storage barrel**
- **Found during:** Task 2
- **Issue:** The package root could not expose `RoleRegistry` as the required sole public selection surface without the intermediate storage barrel exporting it.
- **Fix:** Added the direct storage-barrel re-export; no compatibility alias or second registry was introduced.
- **Files modified:** `src/cacheness/storage/__init__.py`
- **Verification:** Public export and focused composition suites pass.
- **Committed in:** `e81d0f6`

**Total deviations:** 2 auto-fixed (1 Rule 1, 1 Rule 2).

## Verification

- PASS — `uv run --frozen pytest -q tests/test_blob_store_composition.py tests/test_metadata_role_contract.py tests/test_public_api_contract.py -o log_cli=false` (40 passed).
- PASS — `uv run --frozen pytest -q tests/test_topology_capabilities.py tests/test_blob_store_composition.py tests/test_metadata_role_contract.py tests/test_public_api_contract.py -o log_cli=false` (51 passed).
- PASS — scoped `uv run --frozen ruff check` over all Plan 04-10 source and test files.
- PASS — `git diff --check` and source-absence checks for the retired blob registry and bypassable registry arguments.
- BLOCKED OUTSIDE THIS PLAN — `uv run --frozen python tools/verify_phase4_ruff_delta.py` rejects its frozen scope/baseline because it omits existing S3 files and encounters unrelated scope additions. This is already tracked as open Broken Windows ledger entry 33; the changed Plan 04-10 files have no scoped Ruff findings.

## Known Stubs

None.

## Issues Encountered

The phase Ruff-delta gate cannot produce a valid whole-phase result until its frozen baseline/scope is refreshed. This plan did not change the omitted S3 or unrelated paths, and the existing open ledger entry preserves the release gate.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 04-11 can rely on one composition root with validated participants and deterministic local ownership cleanup. Plan 04-12 should rewrite the remaining consumers directly to this public surface rather than retaining aliases or registry bridges.

## Self-Check: PASSED

- All nine implementation/test files and this summary exist.
- All four TDD commits are present in repository history.

---
*Phase: 04-metadata-composition-and-topology-contracts*
*Plan: 10*
*Completed: 2026-09-08*
