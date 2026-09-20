---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "04"
subsystem: storage-migration
tags: [migration, inventory, sqlite, postgresql, offline-maintenance]
requires:
  - phase: 07-03
    provides: "Canonical migration plan model and read-only store inspection classifications"
provides:
  - "Bounded revision-bound raw authority inventory pages for memory, SQLite, and PostgreSQL"
  - "Topology-neutral authority identities with persisted capability and schema version"
  - "Deterministic PostgreSQL paging and typed progress-outcome coverage without a live-service claim"
affects: [07-05, 07-06, 07-07, migration, rebuild, lifecycle-authority]
actuals:
  tokens: 14886
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - "Raw maintenance scans page EntrySnapshot descriptors before manifest authentication or catalog classification."
    - "Inventory continuations bind store identity, captured authority revision, and key/generation position."
key-files:
  created: []
  modified:
    - src/cacheness/storage/migration_authority.py
    - src/cacheness/storage/memory_lifecycle_authority.py
    - src/cacheness/storage/sqlite_lifecycle_authority.py
    - src/cacheness/storage/backends/postgresql_lifecycle_authority.py
    - src/cacheness/storage/migration.py
    - src/cacheness/storage/migration_evidence.py
    - tests/test_migration_inspection.py
    - tests/contracts/test_postgresql_lifecycle_authority.py
key-decisions:
  - "Administrative inventory returns raw EntrySnapshot descriptors; maintenance alone authenticates and classifies their opaque manifest bytes."
  - "A continuation is rejected on identity or revision drift rather than silently refreshing source state."
  - "PostgreSQL inventory identifies itself through its persisted capability and schema version and preserves typed retryable progress causes."
patterns-established:
  - "Validate page and work bounds before opening an authority transaction, then issue indexed keyset reads of at most limit plus one rows."
  - "Use deterministic DB-API fakes for PostgreSQL contracts and state explicitly that Phase 8 owns real PostgreSQL/AWS S3 qualification."
requirements-completed: [MIGR-03, MIGR-04]
coverage:
  - id: D1
    description: "Memory and SQLite return every canonical raw entry once across bounded, revision-bound pages without catalog-schema filtering."
    requirement: MIGR-03
    verification:
      - kind: integration
        ref: "tests/test_migration_inspection.py#test_raw_inventory_pages_every_canonical_entry_without_manifest_filtering"
        status: pass
      - kind: unit
        ref: "tests/test_migration_inspection.py#test_inventory_continuation_rejects_revision_drift_without_refreshing"
        status: pass
    human_judgment: false
  - id: D2
    description: "PostgreSQL exposes the same bounded raw inventory semantics through deterministic driver contracts while retaining topology-specific retryable outcomes."
    requirement: MIGR-04
    verification:
      - kind: unit
        ref: "tests/contracts/test_postgresql_lifecycle_authority.py#test_inventory_pages_raw_rows_at_one_postgresql_revision_without_live_claims"
        status: pass
      - kind: unit
        ref: "tests/contracts/test_postgresql_lifecycle_authority.py#test_inventory_preserves_retryable_postgresql_progress_causes"
        status: pass
    human_judgment: false
duration: 11m 12s
completed: 2026-09-09
status: complete
---

# Phase 07 Plan 04: Bounded Authority Inventory Summary

**Memory, SQLite, and deterministic PostgreSQL authorities now expose revision-bound raw inventory pages that preserve opaque manifests for offline migration classification.**

## Performance

- **Duration:** 11m 12s
- **Started:** 2026-09-09T23:49:52Z
- **Completed:** 2026-09-10T00:01:04Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Replaced the former materialized maintenance inventory with bounded raw `EntrySnapshot` pages bound to store identity, authority revision, and a key/generation continuation.
- Added memory and SQLite inventory reads that leave authority state unchanged, retain schema-mismatched opaque manifest bytes, and reject a changed source as stale.
- Added deterministic PostgreSQL identity and inventory contracts with indexed keyset reads and preserved serialization, deadlock, lock-timeout, and statement-timeout causes.

## Task Commits

1. **Task 1: Add bounded raw inventory to memory and SQLite authorities** - `7be3f18` (test), `ec25746` (feat)
2. **Task 2: Implement the deterministic PostgreSQL inventory contract** - `4183f13` (test), `56b0eed` (feat)

## Files Created/Modified

- `src/cacheness/storage/migration_authority.py` - Defines bounded identity, cursor, raw page, and request-validation contracts.
- `src/cacheness/storage/memory_lifecycle_authority.py` - Provides same-process raw keyset paging.
- `src/cacheness/storage/sqlite_lifecycle_authority.py` - Provides a one-transaction SQLite identity and raw page read.
- `src/cacheness/storage/backends/postgresql_lifecycle_authority.py` - Provides persisted-capability-aware deterministic raw paging.
- `src/cacheness/storage/migration.py` and `migration_evidence.py` - Move existing maintenance consumers and canonical records to the bounded identity/inventory seam.
- `tests/test_migration_inspection.py` and `tests/contracts/test_postgresql_lifecycle_authority.py` - Cover bounded completeness, stale revisions, and deterministic PostgreSQL progress outcomes.

## Decisions Made

- Inventory is a narrow administrative read seam, not another source of lifecycle truth: it returns opaque canonical rows and never authenticates manifests, reads payloads, or filters by catalog schema.
- A changed authority revision is a typed conflict requiring reinspection; continuation never refreshes its snapshot.
- PostgreSQL's persisted capability and schema version are reported directly. The deterministic contract does not upgrade real PostgreSQL or AWS S3 qualification, which remains Phase 8 work.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] Removed materialized inventory consumers from the existing maintenance path**
- **Found during:** Task 1 (Add bounded raw inventory to memory and SQLite authorities)
- **Issue:** The pre-existing migration service and canonical maintenance evidence still used a whole-catalog inventory shape and would have bypassed the new bounded revision/identity contract.
- **Fix:** Updated the maintenance service, read-only inspection, and canonical identity records to page raw snapshots and retain capability/schema fields end-to-end.
- **Files modified:** `src/cacheness/storage/migration.py`, `src/cacheness/storage/migration_evidence.py`
- **Verification:** Focused migration, cutover, plan-contract, stored-compatibility, and authority-contract tests passed.
- **Committed in:** `ec25746` (part of Task 1)

---

**Total deviations:** 1 auto-fixed (1 Rule 2 missing critical functionality)
**Impact on plan:** Necessary to prevent the prior materialized administrative path from bypassing the new bounded inventory boundary; it adds no lifecycle authority, coordination mechanism, or live-service claim.

## Issues Encountered

- The local `uv` environment initially lacked the locked PostgreSQL extra after its virtual environment was recreated. Restoring the already-locked extra enabled deterministic contract tests; no project dependency or lockfile changed.

## Known Stubs

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 07 can authenticate, classify, and resume explicit maintenance evidence over bounded, entry-complete source pages. Later cutover and rebuild work must continue to treat these pages as administrative input only and must not infer live PostgreSQL/AWS S3 qualification.

## Self-Check: PASSED

- All eight implementation/test artifacts and this summary exist on disk.
- TDD commits `7be3f18`, `ec25746`, `4183f13`, and `56b0eed` exist in git history.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-09*
