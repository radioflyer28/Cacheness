---
phase: 04-metadata-composition-and-topology-contracts
plan: "09"
subsystem: storage-catalog-composition
tags: [blobstore, catalog, topology, payload-io, sqlite, memory]
requires:
  - phase: 04-08
    provides: BlobStore-only lifecycle composition
provides:
  - Public catalog commit, update, query, and reopen lifecycle
  - Payload-generation I/O materialized by the selected participant
affects: [phase-05-backend-qualification, phase-06-cache-policy, phase-07-migration]
tech-stack:
  added: []
  patterns:
    - Catalog mutations use the existing exact-record authority promotion.
    - Payload backends supply generation I/O while BlobStore retains lifecycle sequencing.
key-files:
  created: []
  modified:
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/lifecycle.py
    - src/cacheness/storage/composition.py
    - src/cacheness/storage/backends/blob_backends.py
    - src/cacheness/storage/guarded_handler_io.py
    - tests/test_catalog_schema.py
    - tests/test_catalog_query_contract.py
    - tests/test_blob_store_composition.py
    - tests/test_topology_capabilities.py
    - docs/CATALOG_AND_TOPOLOGY.md
decisions:
  - Public catalog updates preserve the payload generation and use exact authority compare-and-swap.
  - A configured cache directory cannot substitute for the selected payload backend's storage root.
metrics:
  duration: 44m
  completed: 2026-09-08
status: complete
actuals:
  tokens: 9682
  tasks: 2
  commits: 4
---

# Phase 04 Plan 09: Public Catalog and Selected Payload I/O Summary

`BlobStore` now exposes a validated, reopen-safe catalog lifecycle and uses the selected payload backend—not an incidental cache path—for immutable generation I/O.

## Outcomes

- Added optional `catalog_schema` and `catalog_values` to public `put_entry` and `put`, with validation before handler selection, authority dispatch, or payload staging.
- Added `update_catalog` for a same-generation, exact-record catalog patch or replacement that preserves payload locator, digest, byte size, handler fields, and generation.
- Made the public catalog tests prove default presence, explicit null, opaque values, stale-update conflict, portable query, and close/reopen behavior for memory and SQLite stores.
- Added a narrow payload-generation I/O provider contract. Filesystem and memory participants materialize their own handler I/O, and invalid payload participants fail while resolving topology.
- Documented the supported public catalog workflow and the explicit migration/rebuild boundary for future schema changes.

## Verification

| Command | Result |
| --- | --- |
| `uv run --frozen pytest -q tests/test_catalog_schema.py tests/test_catalog_query_contract.py -o log_cli=false` | Passed |
| `uv run --frozen pytest -q tests/test_blob_store_composition.py tests/test_topology_capabilities.py tests/test_catalog_query_contract.py -o log_cli=false` | Passed (52 tests) |
| `uv run --frozen pytest -q tests/test_catalog_schema.py tests/test_catalog_query_contract.py tests/test_blob_store_composition.py tests/test_topology_capabilities.py -o log_cli=false` | Passed (74 tests) |
| Focused Ruff across all Plan 04-09 source and test files | Passed |
| `uv run --frozen python tools/verify_phase4_ruff_delta.py` | Not clean: the frozen Phase 4 scope excludes pre-existing `src/cacheness/storage/backends/s3_backend.py` and `tests/test_s3_blob_backend.py`, which the verifier detects. Scoped Plan 04-09 Ruff passed. |

## Commits

- `16cc8f5` `test(04-09): add failing public catalog lifecycle test`
- `79811d9` `feat(04-09): expose public catalog lifecycle`
- `ae58726` `test(04-09): add failing selected payload I/O tests`
- `971fba4` `feat(04-09): use selected payload generation I/O`

## Decisions Made

- Catalog data is part of the signed canonical descriptor; projections remain derived-only and cannot authorize lifecycle or query completeness.
- Catalog update reuses the existing authority mutation seam instead of adding a catalog coordinator, lock, or second authority.
- `cache_dir` is not a payload routing override: the selected filesystem or memory participant owns the actual immutable-generation handler I/O.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] Updated the repository's actual guarded handler-I/O primitive**
- **Found during:** Task 2
- **Issue:** The plan named `src/cacheness/storage/guarded_io.py`, but the existing selected payload primitive is `src/cacheness/storage/guarded_handler_io.py`; the planned path does not exist.
- **Fix:** Added the required delete-or-prove-absent operation to `guarded_handler_io.py` and kept all selected-payload cleanup inside that primitive.
- **Files modified:** `src/cacheness/storage/guarded_handler_io.py`
- **Commit:** `971fba4`

## Known Stubs

None.

## Self-Check: PASSED

Verified the summary exists and all four task commits are present in the repository history.
