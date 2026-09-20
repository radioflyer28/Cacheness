---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "14"
subsystem: storage-lifecycle
tags: [migration, rebuild, lifecycle-authority, replay, sqlite, postgresql, pytest]
requires:
  - phase: 07-explicit-migration-and-rebuild-cutover
    provides: Bounded authority-attributed maintenance recovery and the persisted PostgreSQL worker fence
provides:
  - Exact read-only operation replay records from memory, SQLite, and PostgreSQL authorities
  - Projection-free canonical maintenance puts that replay a persisted BlobReceipt after response loss
  - Prepared-intent completion that uses only the existing authority record and immutable participant locator
affects: [phase-7-rebuild-recovery, phase-8-qualification, migration-maintenance]
actuals:
  tokens: 8631
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - Operation-ID retries read one exact canonical authority record before staging or publishing a candidate
    - Derived projections remain outside the internal maintenance canonical-put receipt path
key-files:
  created: []
  modified:
    - src/cacheness/storage/lifecycle_authority.py
    - src/cacheness/storage/lifecycle.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/memory_lifecycle_authority.py
    - src/cacheness/storage/sqlite_lifecycle_authority.py
    - src/cacheness/storage/backends/postgresql_lifecycle_authority.py
    - tests/test_blob_store_atomic_lifecycle.py
    - tests/contracts/test_lifecycle_authority.py
    - tests/contracts/test_postgresql_lifecycle_authority.py
key-decisions:
  - "Maintenance replay reads the existing canonical authority record; a locator, listing, timing observation, or caller receipt never establishes ownership or completion."
  - "Only the internal maintenance wrapper suppresses automatic projections and it returns the canonical BlobReceipt directly; ordinary put_entry behavior is unchanged."
  - "Prepared replay observes only the authority-indexed immutable locator and accepts it only when digest and byte size corroborate the signed prepared descriptor."
patterns-established:
  - "Represent replay as an immutable authority-owned record containing prepared intent, optional verification, and optional promotion result."
  - "Complete a prepared replay through AuthorityLifecycleEngine rather than duplicating staging, publication, verification, promotion, or cleanup sequencing."
requirements-completed: [MIGR-05, MIGR-06]
coverage:
  - id: D1
    description: Projection-free maintenance writes replay the exact canonical receipt after a post-promotion lost response without another payload publication or derived projection attempt.
    requirement: MIGR-05
    verification:
      - kind: integration
        ref: tests/test_blob_store_atomic_lifecycle.py#test_blobstore_maintenance_canonical_put_replays_projection_free_receipt_after_response_loss
        status: pass
    human_judgment: false
  - id: D2
    description: Memory, SQLite, and deterministic PostgreSQL authorities expose one exact side-effect-free replay classification for absent, prepared, verified, and promoted operations.
    requirement: MIGR-06
    verification:
      - kind: unit
        ref: tests/contracts/test_lifecycle_authority.py#test_local_authorities_expose_exact_operation_replay_without_new_authority_state
        status: pass
      - kind: unit
        ref: tests/contracts/test_postgresql_lifecycle_authority.py#test_postgresql_read_mutation_returns_exact_prepared_and_promoted_replay
        status: pass
    human_judgment: false
duration: 19min
completed: 2026-09-11
status: complete
---

# Phase 07 Plan 14: Canonical Maintenance Replay Summary

**Offline rebuild writes can now retry one deterministic operation through the existing BlobStore engine and receive the exact projection-free canonical receipt after a lost response.**

## Performance

- **Duration:** 19 min
- **Started:** 2026-09-11T05:34:00Z
- **Completed:** 2026-09-11T05:53:12Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Added immutable `MutationReplay` records and read-only exact-operation lookup to the canonical authority contract, with memory, SQLite, and deterministic PostgreSQL parity.
- Added the narrow internal `BlobStore._put_entry_canonical_for_maintenance()` path, which forwards a validated operation ID into the existing lifecycle engine and returns the canonical receipt without automatic projection work.
- Made prepared replay reuse only authority-persisted intent, signed descriptor, immutable locator, digest, and size; it publishes only at exact absence and fails closed on disagreement.
- Proved a post-promotion lost response replays without a second generation, payload publication, authority revision, or projection attempt.

## Task Commits

Each TDD task was committed atomically:

1. **Task 1: Replay one projection-free maintenance put through the canonical memory authority** - `1419543` (`test`), `c31382f` (`feat`)
2. **Task 2: Implement the same exact operation replay in SQLite and PostgreSQL authorities** - `1de0566` (`feat`)

## Files Created/Modified

- `src/cacheness/storage/lifecycle_authority.py` - Defines the bounded immutable replay contract and authority method.
- `src/cacheness/storage/lifecycle.py` - Reads and completes exact replay state through the existing lifecycle sequencing.
- `src/cacheness/storage/blob_store.py` - Provides the internal projection-free maintenance receipt wrapper.
- `src/cacheness/storage/memory_lifecycle_authority.py`, `sqlite_lifecycle_authority.py`, and `backends/postgresql_lifecycle_authority.py` - Reconstruct read-only replay records from their existing authority state.
- `tests/test_blob_store_atomic_lifecycle.py` and authority contract tests - Cover response loss, zero derived effects, and durable replay parity.

## Decisions Made

- An operation ID addresses only an exact authority record. It cannot adopt an entry or infer lifecycle progress from a destination key, payload path, object listing, timing, or caller-provided receipt.
- Projection suppression is internal and maintenance-only. `put_entry()` and `put()` retain their existing post-commit projection behavior.
- PostgreSQL verification remains deterministic transcript coverage; this plan makes no live-service qualification claim.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] Corrected stale Phase 7 plan position after state advance**
- **Found during:** Plan tracking update
- **Issue:** `STATE.md` still reported Plan 3 after Plans 12 and 13 had completed, so the standard advance command moved it only to Plan 4 rather than the completed Plan 14.
- **Fix:** Preserved the tool-managed metrics/session updates and corrected the current-position display to Plan 14 of 19.
- **Files modified:** `.planning/STATE.md`
- **Verification:** `ROADMAP.md` reports 14/19 summaries and this plan's summary is present.

---

**Total deviations:** 1 auto-fixed (Rule 3)
**Impact on plan:** Documentation tracking only; no lifecycle or production code scope changed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

The rebuild coordinator can durably associate each stopped-worker destination write with a deterministic operation ID and recover its exact canonical receipt after response loss. Later plans remain responsible for persisting bounded rebuild receipts and orchestrating resume/abort from maintenance evidence; live PostgreSQL and S3 qualification remains Phase 8.

## Self-Check: PASSED

- Confirmed all nine modified production/test artifacts and this summary exist.
- Confirmed task commits `1419543`, `c31382f`, and `1de0566` exist in repository history.
- Passed the complete BlobStore lifecycle, local lifecycle-authority, deterministic PostgreSQL authority, topology, composition, and catalog-projection suites (108 tests).
- Passed scoped Ruff checks for every modified production and test file.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-11*
