---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "07"
subsystem: storage migration adapters
tags: [postgresql, s3, offline-migration, lifecycle-authority, moto]
requires:
  - phase: 07-06
    provides: shared authority publication receipts and offline cutover state machine
provides:
  - PostgreSQL schema-4 whole-store candidate, activation, rollback, and finalization transactions
  - run-owned S3 immutable candidate receipts with exact digest/size response-loss revalidation
  - deterministic non-live remote adapter contracts without a service qualification claim
affects: [phase-07-cutover, phase-08-qualification, storage-lifecycle]
actuals:
  tokens: 11862
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - receipt-bound PostgreSQL publication transitions within one authority transaction
    - S3 candidate writes verified by exact SHA-256 and byte size after ambiguous responses
key-files:
  created:
    - tests/test_migration_remote_contract.py
  modified:
    - src/cacheness/storage/backends/postgresql_lifecycle_authority.py
    - src/cacheness/storage/backends/s3_backend.py
    - src/cacheness/storage/migration.py
    - tests/contracts/test_postgresql_lifecycle_authority.py
key-decisions:
  - "PostgreSQL authority state remains the sole whole-store visibility decision; S3 candidate receipts only corroborate immutable external effects."
  - "S3 candidate locators carry an exact run marker and never permit listing-based discovery or adoption."
patterns-established:
  - "Remote response loss is classified through a read-and-rehash of the exact expected object, not ETag or list output."
  - "Offline migration revalidates plan-bound identities immediately before a remote candidate write without claiming global quiescence."
requirements-completed: [MIGR-04, MIGR-05]
coverage:
  - id: D1
    description: PostgreSQL authority records complete candidates, activates retained prior rows, and exposes only receipt-bound offline resolution.
    requirement: MIGR-04
    verification:
      - kind: integration
        ref: uv run --frozen --extra postgresql pytest -q tests/contracts/test_postgresql_lifecycle_authority.py -x -o log_cli=false
        status: pass
    human_judgment: false
  - id: D2
    description: S3 migration candidate writes retain exact run ownership, digest/size proof, and deterministic response-loss recovery without object discovery.
    requirement: MIGR-05
    verification:
      - kind: integration
        ref: uv run --frozen --extra cloud pytest -q tests/test_migration_remote_contract.py tests/test_s3_blob_backend.py -x -o log_cli=false
        status: pass
    human_judgment: false
duration: 11min
completed: 2026-09-10
status: complete
---

# Phase 07 Plan 07: Remote Migration Adapter Contracts Summary

**PostgreSQL-only whole-store publication transactions and S3 run-owned immutable candidate receipts with deterministic, non-live recovery evidence.**

## Performance

- **Duration:** 11 min
- **Started:** 2026-09-10T03:18:03Z
- **Completed:** 2026-09-10T03:29:07Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added receipt-bound PostgreSQL candidate recording, activation, lost-response classification, offline worker fencing, rollback, and finalization over the schema-4 authority state.
- Added bounded S3 migration-candidate writes with exact run ownership, SHA-256/size verification, and response-loss revalidation through the existing guarded snapshot path.
- Proved only deterministic adapter mechanics with psycopg transcript fakes and Moto; real PostgreSQL/Amazon S3 qualification remains Phase 8 work.

## Task Commits

1. **Task 1: Add PostgreSQL whole-store maintenance transactions** - `bd3baec` (test), `6d7bbc9` (feat).
2. **Task 2: Prove S3 candidate effects remain verifiable and non-authoritative** - `e48e7ac` (test), `3026d57` (feat).

## Files Created/Modified

- `src/cacheness/storage/backends/postgresql_lifecycle_authority.py` - PostgreSQL state/receipt transactions and offline fence for the shared migration contract.
- `src/cacheness/storage/backends/s3_backend.py` - immutable run-owned S3 candidate receipt and response-loss revalidation.
- `src/cacheness/storage/migration.py` - selects the guarded remote candidate writer and revalidates live identities immediately before its external effect.
- `tests/contracts/test_postgresql_lifecycle_authority.py` - deterministic whole-store transaction and offline worker-fence coverage.
- `tests/test_migration_remote_contract.py` - Moto/fake receipt ownership and response-loss tests explicitly labeled non-live.

## Decisions Made

- PostgreSQL transactions end at authority state; S3 creation and cleanup remain attributable immutable external effects.
- Candidate object presence, ETag, and listing remain non-authoritative. Only the exact run/plan/source-revision receipt plus authority activation can progress cutover.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Allowed an unmaterialized PostgreSQL authority through the offline worker guard.**
- **Found during:** Task 1
- **Issue:** `BlobStore.initialize()` checks ordinary worker access before explicit authority creation, so reading publication state first rejected a valid fresh authority.
- **Fix:** Treat only an unmaterialized authority as idle for the guard; explicit initialization still performs the schema/capability validation.
- **Files modified:** `src/cacheness/storage/backends/postgresql_lifecycle_authority.py`
- **Verification:** Full deterministic PostgreSQL authority contract suite passed.
- **Committed in:** `6d7bbc9`

**2. [Rule 2 - Missing critical functionality] Routed remote migration staging through guarded S3 candidate receipts.**
- **Found during:** Task 2
- **Issue:** The generic maintenance service used retired byte-CRUD staging, which S3 deliberately does not expose, preventing remote candidate staging despite the planned adapter contract.
- **Fix:** Added a narrow guarded S3 candidate-write receipt and selected it only when the payload participant supports it; local backends retain their existing path.
- **Files modified:** `src/cacheness/storage/backends/s3_backend.py`, `src/cacheness/storage/migration.py`
- **Verification:** Remote S3/Moto contracts and existing local cutover contracts passed.
- **Committed in:** `3026d57`

---

**Total deviations:** 2 auto-fixed (1 Rule 1, 1 Rule 2).
**Impact on plan:** Both fixes keep the single authority boundary intact and add no external transaction, lease, advisory lock, listing adoption, or live-service support claim.

## Issues Encountered

- The baseline `uv` environment lacked the optional psycopg/boto3 test extras. The frozen, declared `postgresql` and `cloud` extras supplied deterministic contract dependencies; no external service or credential was used.

## Known Stubs

None.

## User Setup Required

None - no external service configuration is required for this deterministic adapter plan.

## Next Phase Readiness

- The generic offline cutover service can stage immutable S3 candidates through exact receipts while PostgreSQL remains the authority boundary.
- Phase 8 still owns live PostgreSQL and Amazon S3 credentials, service behavior, compatible-service coverage, availability, and performance qualification.

## Self-Check: PASSED

- Confirmed all five implementation/test artifacts and this summary exist.
- Confirmed `bd3baec`, `6d7bbc9`, `e48e7ac`, and `3026d57` are present in git history.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-10*
