---
phase: 04-metadata-composition-and-topology-contracts
plan: "08"
subsystem: storage-composition
tags: [blobstore, store-topology, catalog, cache-policy, cutover, validation]
requires:
  - phase: 04-07
    provides: prepared direct consumers and negative cutover contracts
provides:
  - BlobStore-only public payload/catalog lifecycle composition
  - UnifiedCache policy facade composed over an internal BlobStore
  - Explicit release evidence and topology-qualified non-claims
affects: [phase-05-backend-qualification, phase-06-cache-policy, phase-07-migration]
actuals:
  tokens: 52597
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - Cache policy delegates to an internally composed BlobStore rather than a metadata authority.
    - JSON and PostgreSQL are derived projection participants, never lifecycle authorities.
key-files:
  created: []
  modified:
    - src/cacheness/core.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/metadata.py
    - docs/STORAGE_INITIALIZATION.md
    - .planning/phases/04-metadata-composition-and-topology-contracts/04-VALIDATION.md
key-decisions:
  - "Remove development-only metadata authorities and compatibility surfaces rather than preserving adapters."
  - "Keep UnifiedCache as a narrow BlobStore-backed policy facade; Phase 6 remains responsible for policy redesign."
  - "Record unsupported full-suite collection honestly instead of restoring retired APIs or overstating platform/service coverage."
patterns-established:
  - "One StoreTopology-rooted BlobStore owns payload and catalog lifecycle."
  - "Derived projections are rebuildable observers and cannot grant lifecycle or query-completeness authority."
requirements-completed: [BACK-02, BACK-03, BACK-06, BACK-07]
coverage:
  - id: D1
    description: BlobStore is the single StoreTopology-rooted payload and catalog lifecycle facade.
    requirement: BACK-02
    verification:
      - kind: integration
        ref: tests/test_blob_store_composition.py
        status: pass
      - kind: integration
        ref: tests/test_metadata_role_contract.py
        status: pass
    human_judgment: false
  - id: D2
    description: Catalog schemas, authenticated pages, capabilities, and projection-only roles retain their Phase 4 contracts.
    requirement: BACK-07
    verification:
      - kind: integration
        ref: tests/test_catalog_schema.py + tests/test_catalog_query_contract.py + tests/test_catalog_projection.py
        status: pass
    human_judgment: false
  - id: D3
    description: Release matrix documents supported interpreter evidence and explicit topology non-claims.
    requirement: BACK-06
    verification:
      - kind: other
        ref: .planning/phases/04-metadata-composition-and-topology-contracts/04-VALIDATION.md
        status: fail
    human_judgment: true
    rationale: Unexcluded full suites stopped in retired compatibility-test collection and optional pandas-dependent SQL-cache collection.
completed: 2026-09-08
status: complete
---

# Phase 04 Plan 08: Atomic BlobStore Cutover Summary

**Removed the development-only metadata authority surface so direct storage has one BlobStore/StoreTopology lifecycle, while UnifiedCache remains a small BlobStore-backed cache-policy facade.**

## Performance

- **Duration:** 5 min execution time, excluding git-permission handoffs
- **Started:** 2026-09-08T00:12:58-04:00
- **Completed:** 2026-09-08T04:17:53Z
- **Tasks:** 2/2
- **Files modified:** 17

## Accomplishments

- Removed metadata factories, registries, duplicate ABCs, runtime custom-metadata hooks, compatibility result shapes, and obsolete manifest/export modules.
- Reduced `UnifiedCache` to keying, TTL, eviction, invalidation, and statistics above an internally composed `BlobStore`; decorators use that same facade without a metadata-backend probe.
- Kept JSON and PostgreSQL strictly projection-only, retained typed unsupported-layout detection, and removed stale initialization guidance.
- Captured the exact focused, interpreter, lint, and external-service qualification evidence in `04-VALIDATION.md`.

## Task Commits

1. **Task 1: Atomically cut over source consumers and package exports** — `1351b9b` (`refactor`)
2. **Task 1 corrective deletion: remove superseded metadata modules** — `a6fb46a` (`refactor`)
3. **Task 2: Execute the supported-runtime release matrix and record honest evidence** — `151650f` (`docs`)

## Files Created/Modified

- `src/cacheness/core.py` — cache policy now composes the only storage engine through `StoreTopology` and `BlobStore`.
- `src/cacheness/storage/blob_store.py` and `src/cacheness/storage/lifecycle.py` — retain the canonical receipt/snapshot contract without legacy selectors or dictionary list filters.
- `src/cacheness/metadata.py` — reduced to the derived-only JSON projection participant.
- `src/cacheness/decorators.py` and `src/cacheness/storage/composition.py` — direct consumers no longer revive metadata authority selection.
- `docs/STORAGE_INITIALIZATION.md` — documents current `BlobReceipt`/`BlobEntry`, prefix-only listing, and projection-only failure boundaries.
- `04-VALIDATION.md` — records truthful supported-runtime and service/platform qualification evidence.

## Decisions Made

- Unsupported pre-cutover layouts are rejected with typed migration/rebuild-required evidence; normal initialization does not migrate them.
- The cutover does not add lifecycle locks, queues, coordinators, cross-resource ACID guarantees, or stronger progress claims than ADR 0001 permits.
- Retained tests that import the intentionally removed metadata authority are follow-up cleanup work, not justification for a runtime compatibility shim.

## Deviations from Plan

### Scope expansions required for the atomic cutover

**1. Updated direct consumers outside the initial file list**

- **Found during:** Task 1
- **Issue:** `decorators.py` still probed `metadata_backend` and retained a legacy decorator fallback; `composition.py` constructed the removed `JsonBackend`.
- **Fix:** Routed decorators exclusively through `UnifiedCache.get()` and changed composition to the derived-only `JsonProjection` participant.
- **Files modified:** `src/cacheness/decorators.py`, `src/cacheness/storage/composition.py`
- **Verification:** Focused Phase 4 suite: 185 passed; decorator suite: 20 passed, 1 optional pandas skip.
- **Committed in:** `1351b9b`

**2. Corrected stale initialization documentation**

- **Found during:** Task 2
- **Issue:** The guide still described removed dictionary filtering, `BlobEntryInfo`, legacy signature handling, and custom-ORM runtime metadata outcomes.
- **Fix:** Replaced it with the current BlobReceipt/BlobEntry, canonical-query, and derived-projection boundaries.
- **Files modified:** `docs/STORAGE_INITIALIZATION.md`
- **Verification:** Documentation audit in `04-VALIDATION.md`; stale contract terms removed from the guide.
- **Committed in:** `151650f`

**Total deviations:** 2 scope expansions required to prevent retained consumers or documentation from reintroducing the removed authority model.

## Issues Encountered

- The executor could not create `.git/index.lock`; the root agent committed each completed task after review.
- The exact unexcluded Python 3.11 and 3.13 suite commands both stopped in collection: legacy metadata-authority tests import removed names, and SQL-cache modules require pandas outside the locked default environment. This is documented as a release qualification gap, not hidden by exclusions.

## Known Stubs

None.

## Next Phase Readiness

- Phase 5 can qualify only explicitly named backend topology pairs; it must not infer S3 or PostgreSQL lifecycle support from the projection roles.
- Phase 6 can refine coherent cache policy while retaining the BlobStore engine boundary established here.
- Before any green complete-suite release gate, retire or rewrite the historical metadata-authority tests and run SQL-cache coverage with its declared optional dependency group.

## Self-Check: PASSED

`04-08-SUMMARY.md` exists, and task commits `1351b9b`, `a6fb46a`, and
`151650f` exist in Git history.
