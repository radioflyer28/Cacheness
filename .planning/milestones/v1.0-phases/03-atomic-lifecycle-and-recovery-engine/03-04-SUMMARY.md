---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "04"
subsystem: storage lifecycle
tags: [blob-store, lifecycle-authority, sqlite, concurrency, cleanup]

requires:
  - phase: 03-03
    provides: durable SQLite authority state and authority transition contracts
provides:
  - Authority-backed BlobStore put, overwrite, read, metadata, delete, cleanup, and close paths
  - Exact cleanup-debt recovery with signed tombstone retirement
  - Bounded per-key local coordination and ownership-aware resource close
affects: [BlobStore, lifecycle authority, storage recovery, concurrency tests]

actuals:
  tokens: 119361
  tasks: 2
  commits: 8

tech-stack:
  added: []
  patterns:
    - short authority transactions around prepare and promotion only
    - immutable native payload generations with exact cleanup debts
    - authority-owned visibility with bounded local coordination

key-files:
  created: []
  modified:
    - src/cacheness/storage/lifecycle.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/lifecycle_authority.py
    - src/cacheness/storage/memory_lifecycle_authority.py
    - src/cacheness/storage/sqlite_lifecycle_authority.py
    - tests/test_blob_store_atomic_lifecycle.py
    - tests/test_blob_store_concurrency.py
    - tests/test_blob_store_close_contract.py
    - tests/test_blob_store_read_contract.py
    - tests/test_blob_manifest.py
    - tests/test_blob_manifest_backends.py
    - tests/test_blob_store_integrity.py
    - tests/test_blob_store_reconciliation.py
    - tests/test_clear_recovery.py
    - tests/test_filesystem_containment.py

key-decisions:
  - "LifecycleAuthority is the only committed-state authority; legacy manifest repositories are not initialized in authority mode."
  - "Promotion records exact cleanup debt atomically, while reclamation and debt retirement run outside authority transactions."
  - "Default BlobStore close owns and releases only the authority it constructs; injected resources stay caller-owned."

patterns-established:
  - "Use immutable generation locators and an exact lineage CAS for every authority mutation."
  - "Treat a signed tombstone as durable absence until authenticated cleanup debt has been safely retired."

requirements-completed: [STOR-03, STOR-04, STOR-05, STOR-07]

coverage:
  - id: D1
    description: "Authority-backed write, overwrite, metadata, and read paths retain old-or-new authenticated visibility."
    requirement: STOR-03
    verification:
      - kind: integration
        ref: "tests/test_blob_store_atomic_lifecycle.py"
        status: pass
      - kind: integration
        ref: "tests/test_blob_store_concurrency.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "Delete promotes an authenticated tombstone and reclaims only exact, non-owning cleanup debt."
    requirement: STOR-05
    verification:
      - kind: integration
        ref: "tests/test_blob_store_atomic_lifecycle.py"
        status: pass
    human_judgment: false
  - id: D3
    description: "Same-key CAS produces one winner while distinct keys overlap and reads retry only once for proven-newer state."
    requirement: STOR-04
    verification:
      - kind: integration
        ref: "tests/test_blob_store_concurrency.py"
        status: pass
    human_judgment: false
  - id: D4
    description: "Close blocks new work, drains safely, and releases only internally owned resources once."
    requirement: STOR-07
    verification:
      - kind: integration
        ref: "tests/test_blob_store_close_contract.py"
        status: pass
    human_judgment: false

duration: 103min
completed: 2026-09-05
status: complete
---

# Phase 03 Plan 04: Atomic Lifecycle and Recovery Completion Summary

**BlobStore now commits immutable native payload generations through LifecycleAuthority, with signed tombstone deletion, exact debt recovery, bounded local coordination, and ownership-aware close.**

## Performance

- **Duration:** 103 min
- **Started:** 2026-09-05T02:26:44Z
- **Completed:** 2026-09-05T04:09:22Z
- **Tasks:** 2
- **Files modified:** 15

## Accomplishments

- Replaced scheduler-mediated writes with durable authority prepare → native publish → verify → promote flow, keeping visibility in the authority alone.
- Implemented CAS-safe overwrite/metadata/delete paths, authenticated exact cleanup debt, signed tombstones, and bounded one-retry reads.
- Added deterministic concurrency and close coverage for one-winner mutation races, distinct-key overlap, tombstone recovery, and resource ownership.
- Migrated residual manifest, JSON projection, scheduler, and journal fixtures to LifecycleAuthority evidence; full repository verification is green.

## Task Commits

1. **Task 1: Complete put, overwrite, metadata update, and read-race semantics** - `def7305` (test), `bdab579` (feat)
2. **Task 2: Converge delete, cleanup, per-key coordination, and close** - `c653721` (test), `04831a3` (feat)
3. **Authority projection compatibility correction** - `f6bfc83` (fix)
4. **Authority-only constructor fixture migration** - `6295960` (fix)
5. **Authority-only read-contract compatibility migration** - `054c524` (fix)
6. **Authority-only repository-suite migration** - `9a8d962` (fix)

## Files Created/Modified

- `src/cacheness/storage/lifecycle.py` - Thin authority coordinator for mutation, read, cleanup, and tombstone flow.
- `src/cacheness/storage/blob_store.py` - Default authority wiring, public authority route, strict projection rejection, and owned-resource close.
- `src/cacheness/storage/lifecycle_authority.py` - Authority protocol support for pending mutation and exact cleanup-debt recovery.
- `src/cacheness/storage/memory_lifecycle_authority.py` - In-memory authority recovery implementation.
- `src/cacheness/storage/sqlite_lifecycle_authority.py` - Concurrent SQLite bootstrap and durable cleanup/recovery state.
- `tests/test_blob_store_atomic_lifecycle.py` - Authority fault, tombstone, and residue convergence coverage.
- `tests/test_blob_store_concurrency.py` - CAS, bounded-retry read, and distinct-key overlap coverage.
- `tests/test_blob_store_close_contract.py` - Close ownership and retry coverage.
- `tests/test_blob_store_read_contract.py` - Authority-constructor failure, cancellation, and caller-ownership coverage.
- `tests/test_blob_manifest.py` and `tests/test_blob_manifest_backends.py` - Native custom-handler and backend-neutral authority-entry coverage.
- `tests/test_blob_store_integrity.py`, `tests/test_blob_store_reconciliation.py`, and `tests/test_clear_recovery.py` - Signed authority-entry, exact-debt, and no-scheduler recovery coverage.
- `tests/test_filesystem_containment.py` - Authority-locator containment and all-or-error clear preflight coverage.

## Decisions Made

- Keep `LifecycleAuthority` as the sole committed truth. A legacy manifest repository must not be constructed as a parallel state authority in the default path.
- Record and authenticate exact cleanup debt with promotion; cleanup cannot use loose locators and cannot revoke a later winner.
- Keep local coordination bounded and per key, never as a store-, family-, or process-wide correctness lease.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bootstrap race] Repaired concurrent SQLite authority initialization.**
- **Found during:** Task 1
- **Issue:** Concurrent fresh stores could observe an incomplete bootstrap database before schema creation completed.
- **Fix:** Added exclusive schema recheck and bounded join behavior for an in-flight authority bootstrap.
- **Files modified:** `src/cacheness/storage/sqlite_lifecycle_authority.py`
- **Verification:** Repeated independent CAS concurrency runs and the authority behavior suites.
- **Committed in:** `bdab579`

**2. [Rule 2 - Recovery correctness] Completed authority recovery queries required by exact cleanup.**
- **Found during:** Task 1
- **Issue:** The authority protocol lacked pending-mutation and cleanup-debt queries needed to reconcile published candidates without a scheduler repository.
- **Fix:** Added bounded recovery operations to the protocol and its SQLite and in-memory implementations.
- **Files modified:** `src/cacheness/storage/lifecycle_authority.py`, `src/cacheness/storage/memory_lifecycle_authority.py`, `src/cacheness/storage/sqlite_lifecycle_authority.py`
- **Verification:** Tombstone/reconciliation fault coverage in `tests/test_blob_store_atomic_lifecycle.py`.
- **Committed in:** `bdab579`

**3. [Rule 1 - Close retry] Prevented repeat release of an already closed owned authority.**
- **Found during:** Task 2
- **Issue:** A partial close failure could cause a later retry to invoke `close()` again on a resource already released successfully.
- **Fix:** Track per-resource release completion before retrying remaining owned resources.
- **Files modified:** `src/cacheness/storage/blob_store.py`, `tests/test_blob_store_close_contract.py`
- **Verification:** `test_close_retry_does_not_reclose_an_already_released_owned_authority`.
- **Committed in:** `04831a3`

**4. [Rule 1 - Projection validation] Rejected unsupported legacy metadata adapters in authority mode.**
- **Found during:** Phase 2 read-contract regression
- **Issue:** An unsupported projection object was silently retained even though authority mode cannot use it as committed state.
- **Fix:** Accept only supported compatibility projections and fail closed for other adapters.
- **Files modified:** `src/cacheness/storage/blob_store.py`
- **Verification:** `tests/test_blob_store_read_contract.py -k 'authority_tracer or unsupported_backend'`.
- **Committed in:** `f6bfc83`

**5. [Rule 1 - Regression coverage] Migrated constructor ownership fixtures to the authority boundary.**
- **Found during:** Wave 4 regression verification
- **Issue:** The old fixtures forced `create_manifest_repository`, which authority mode correctly never constructs.
- **Fix:** Create and fail after the default `SqliteLifecycleAuthority` and in-memory projection are owned; assert injected authority and backend resources are untouched.
- **Files modified:** `tests/test_blob_store_read_contract.py`
- **Verification:** Exact reproduced assertion plus constructor/cancellation subset.
- **Committed in:** `6295960`

**6. [Rule 1 - Error taxonomy] Migrated residual projection fixtures and preserved unsupported authority-schema translation.**
- **Found during:** Wave 4 full historical read-contract verification.
- **Issue:** Direct-operation tests treated `cache_metadata.json` and retired manifest-repository seams as committed truth; malformed authority schema bytes no longer retained the established typed unsupported-version error.
- **Fix:** Assert direct reads and mutations use `LifecycleAuthority` despite corrupt or unavailable projections, move backend-failure coverage to authority operations, and translate an unsupported authority manifest schema to `CacheBlobManifestUnsupportedVersionError`.
- **Files modified:** `src/cacheness/storage/blob_store.py`, `tests/test_blob_store_read_contract.py`
- **Verification:** Entire `tests/test_blob_store_read_contract.py` suite (40 passed), all Plan 04 behavior suites (37 passed), and Phase 3 Ruff delta.
- **Committed in:** `054c524`

**7. [Rule 1 - Authority-only compatibility] Completed the stale full-suite migration and preserved fail-closed authority boundaries.**
- **Found during:** Wave 4 full repository verification.
- **Issue:** Remaining tests inspected retired manifest repositories, JSON projections, clear journals, or absolute payload paths. The sweep also exposed three authority-mode regressions: custom compatibility handlers lost their published native format, absent reads created a storage root, and a corrupt later authority entry could let `clear()` partially revoke earlier entries.
- **Fix:** Preserve signed native `storage_format` in handler metadata, defer managed-root materialization until payload work is necessary, authenticate/preflight the entire clear snapshot before deletion, and convert stale fixtures to `LifecycleAuthority` entry/debt contracts.
- **Files modified:** `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/lifecycle.py`, `tests/test_blob_manifest.py`, `tests/test_blob_manifest_backends.py`, `tests/test_blob_store_close_contract.py`, `tests/test_blob_store_integrity.py`, `tests/test_blob_store_read_contract.py`, `tests/test_blob_store_reconciliation.py`, `tests/test_clear_recovery.py`, `tests/test_filesystem_containment.py`.
- **Verification:** Full repository suite reached 100% with exit status 0; Plan 04 behavior suite (37 passed); authority/manifest/read/containment suite passed with 3 capability skips; compileall; Phase 3 Ruff delta.
- **Committed in:** `9a8d962`

**Total deviations:** 7 auto-fixed (6 Rule 1, 1 Rule 2).
**Impact on plan:** All changes preserve the authority-only architecture and are required for durable recovery, concurrency safety, or fail-closed compatibility.

## Issues Encountered

- `test_failed_initialization_closes_only_internally_owned_backend` and adjacent cancellation fixtures now exercise an internally created `LifecycleAuthority` plus projection, and verify that injected authority/backend resources stay caller-owned. This regression is fixed in `6295960` and `.planning/WINDOWS.md` entry 23 is resolved.
- The historical JSON-projection and manifest-repository fixtures now assert the authority-only contract. Corrupt or unavailable projections neither authorize nor block committed reads or mutations; `.planning/WINDOWS.md` entry 24 is resolved in `054c524`.
- The full repository suite now completes successfully. Capability-dependent PostgreSQL, TensorFlow, device-node, Windows-junction, and sandbox-denied AF_UNIX socket fixtures remain explicitly skipped without weakening authority validation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Authority-backed BlobStore mutations, recovery, and close semantics are ready for remaining Phase 3 integration. The complete repository suite is green under the authority-only contract.

## Self-Check: PASSED

All listed source/test artifacts exist and each recorded task commit, including `9a8d962`, is reachable from the repository history.

---
*Phase: 03-atomic-lifecycle-and-recovery-engine*
*Completed: 2026-09-05*
