---
phase: 04-metadata-composition-and-topology-contracts
plan: "04"
subsystem: storage
tags: [blobstore, sqlite, catalog, hmac, keyset-pagination, topology]
requires:
  - phase: 04-03
    provides: StoreTopology role resolution, ownership, and truthful capability reporting
provides:
  - Explicit format-2 SQLite initialization and typed no-mutation rejection for incompatible layouts
  - One signed canonical descriptor per committed authority entry
  - Authenticated, revision-bound, bounded portable catalog scans for memory and SQLite
affects: [04-05, 04-06, 04-08, BlobStore, lifecycle-authority]
actuals:
  tokens: 17237
  tasks: 3
  commits: 3
tech-stack:
  added: []
  patterns:
    - Canonical signed descriptor bytes remain the only catalog representation.
    - Query pages enumerate authority identities and authenticate descriptors before evaluation.
    - Cursors bind store, format, schema, query, revision, and keyset identity with HMAC.
key-files:
  created: []
  modified:
    - src/cacheness/storage/sqlite_lifecycle_authority.py
    - src/cacheness/storage/memory_lifecycle_authority.py
    - src/cacheness/storage/lifecycle_authority.py
    - src/cacheness/storage/lifecycle.py
    - src/cacheness/storage/catalog.py
    - src/cacheness/storage/blob_store.py
    - tests/test_catalog_query_contract.py
key-decisions:
  - "SQLite user_version 7 is an authority-schema identifier independent of public store format 2."
  - "Signed canonical descriptors, rather than normalized catalog rows or indexes, are the sole query source."
  - "Work-capped pages resume at the last examined identity, including zero-match pages, and stale revisions require retry."
requirements-completed: [BACK-02, BACK-06, BACK-07]
coverage:
  - id: D1
    description: Explicit format-2 SQLite initialization rejects incompatible layouts unchanged.
    requirement: BACK-02
    verification:
      - kind: integration
        ref: tests/test_catalog_schema.py and tests/test_sqlite_bootstrap_concurrency.py
        status: pass
    human_judgment: false
  - id: D2
    description: Committed entries retain one signed canonical descriptor through SQLite reopen and authority transitions.
    requirement: BACK-07
    verification:
      - kind: integration
        ref: tests/test_blob_store_composition.py and tests/test_lifecycle_authority_contract.py
        status: pass
    human_judgment: false
  - id: D3
    description: Memory and SQLite provide bounded authenticated catalog pages with revision-bound cursors.
    requirement: BACK-06
    verification:
      - kind: integration
        ref: tests/test_catalog_query_contract.py and tests/test_topology_capabilities.py
        status: pass
    human_judgment: false
duration: 16m 51s
completed: 2026-09-08
status: complete
---

# Phase 04 Plan 04: Metadata Composition and Topology Contracts Summary

**Format-2 SQLite authority initialization, signed canonical descriptor commits, and bounded HMAC-protected catalog scans for memory and SQLite.**

## Performance

- **Duration:** 16m 51s
- **Started:** 2026-09-08T02:12:08Z
- **Completed:** 2026-09-08T02:28:59Z
- **Tasks:** 3
- **Files modified:** 12

## Accomplishments

- Added explicit, validation-only format-2 SQLite initialization with typed migration/rebuild rejection for incompatible evidence.
- Kept schema identity, catalog values/presence, opaque metadata, revision, and cleanup state in one signed descriptor committed through the existing authority transition.
- Added canonical-complete `BlobStore.query_catalog` pagination: memory and SQLite enumerate current identities in stable keyset order, authenticate descriptors, and bind cursors to the immutable query context.
- Made query capability truthful: portable canonical scans are supported; acceleration indexes are not.

## Task Commits

1. **Task 1: Establish format-2 SQLite initialization and rejection boundaries** — `23bfeef` (`feat`)
2. **Task 2: Persist and reopen canonical signed catalog descriptors** — `9ae1eed` (`feat`)
3. **Task 3: Implement canonical-complete revision-bound portable queries** — `34ad48a` (`feat`)

## Files Created/Modified

- `src/cacheness/storage/sqlite_lifecycle_authority.py` — explicit authority schema lifecycle and bounded read-snapshot keyset scan.
- `src/cacheness/storage/memory_lifecycle_authority.py` — same-process query parity over current authority entries.
- `src/cacheness/storage/lifecycle_authority.py` and `src/cacheness/storage/lifecycle.py` — opaque descriptor mutation contract and lifecycle composition.
- `src/cacheness/storage/catalog.py` — finite query/page validation, HMAC cursor codec, and descriptor-scan evaluator.
- `src/cacheness/storage/blob_store.py` — public `query_catalog` facade backed by the sole authority.
- `docs/STORAGE_INITIALIZATION.md` and affected catalog/bootstrap/composition tests — explicit initialization boundary and coverage.

## Decisions Made

- SQLite pages use an explicit read transaction, so the cursor revision and enumerated identities derive from one authority snapshot.
- A cursor stops at the last authenticated descriptor examined, never merely the last match; sparse and zero-match scans therefore resume safely.
- Descriptor schema mismatches do not become a secondary catalog state; only matching authenticated descriptors contribute to a query result.
- The existing local SQLite topology retains success, exact conflict, or typed retryable-timeout outcomes. This plan does not claim universal contender success.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] Removed stale unused `replace` import in `blob_store.py`**
- **Found during:** Task 3 verification
- **Issue:** The Phase 4 Ruff delta gate reported an unused import in a task-owned source file.
- **Fix:** Removed the unused import without changing runtime behavior.
- **Files modified:** `src/cacheness/storage/blob_store.py`
- **Verification:** Phase 4 Ruff delta and focused Ruff checks pass.
- **Committed in:** `34ad48a`

---

**Total deviations:** 1 auto-fixed (Rule 3)
**Impact on plan:** Verification-only cleanup; no architecture or behavioral scope changed.

## Issues Encountered

The shared Git index denied direct staging during each task boundary. The orchestrator committed the verified task files individually; no unrelated working-tree files were staged or modified.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 04-05 can build derived projections from `query_catalog` without making a projection authoritative. The durable scope remains the initialized local SQLite/filesystem topology; no claim is made for universal contention success, live PostgreSQL/S3 qualification, or a physical catalog index.

## Self-Check: PASSED

- Summary exists at the required phase path.
- Task commits `23bfeef`, `9ae1eed`, and `34ad48a` exist in Git history.

---
*Phase: 04-metadata-composition-and-topology-contracts*
*Completed: 2026-09-08*
