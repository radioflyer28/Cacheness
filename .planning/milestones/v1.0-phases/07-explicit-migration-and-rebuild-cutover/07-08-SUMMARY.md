---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "08"
subsystem: storage-migration
tags: [blobstore, offline-migration, rollback, finalize, purge, cleanup-debt]
requires:
  - phase: 07-02
    provides: "Accepted D-15 finalize-then-separate-purge retirement contract"
  - phase: 07-06
    provides: "Authority-owned activated_offline cutover with retained prior selection"
provides:
  - "Receipt-bound offline rollback and explicit finalization that seals rollback before workers restart"
  - "Run-owned unactivated candidate abort with authenticated evidence and no source deletion"
  - "Separately confirmed idempotent retained-prior purge with durable retryable cleanup debt"
affects: [migration, lifecycle-authority, blobstore, phase-08-qualification]
actuals:
  tokens: 10355
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - "Offline action confirmations are SHA-256 bindings to one authenticated run, plan, authority receipt, and exact cleanup set."
    - "Authority state stays canonical while evidence tracks external purge continuation and retryable cleanup debt."
key-files:
  created: []
  modified:
    - src/cacheness/storage/migration.py
    - src/cacheness/storage/migration_evidence.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/memory_lifecycle_authority.py
    - src/cacheness/storage/sqlite_lifecycle_authority.py
    - tests/test_migration_cutover.py
key-decisions:
  - "Rollback and finalize consume the authenticated activated receipt; finalization is replay-safe only with the same exact confirmation."
  - "Purge confirmation binds the authenticated retained-prior key, generation, locator, manifest digest, entry count, and byte count."
  - "Purge failures remain evidence-backed cleanup debt and never revert or alter the active candidate selection."
patterns-established:
  - "Expose retained migration rows only through a finalized authority state, then validate signed manifests again before external cleanup."
  - "Use delete-or-prove-absent for retry and response-loss recovery without locator discovery or time-based cleanup."
requirements-completed: [MIGR-04, MIGR-05]
coverage:
  - id: D1
    description: "Offline rollback restores the retained prior selection, while explicit finalization seals rollback and reopens ordinary worker access."
    requirement: MIGR-04
    verification:
      - kind: integration
        ref: "tests/test_migration_cutover.py#test_offline_service_rolls_back_only_the_activated_receipt"
        status: pass
      - kind: integration
        ref: "tests/test_migration_cutover.py#test_offline_service_finalize_requires_exact_confirmation_and_seals_rollback"
        status: pass
    human_judgment: false
  - id: D2
    description: "Abort and purge are exact, separately confirmed external cleanup actions whose partial failure remains retryable without changing activation."
    requirement: MIGR-05
    verification:
      - kind: integration
        ref: "tests/test_migration_cutover.py#test_offline_service_abort_removes_only_its_unactivated_candidate"
        status: pass
      - kind: integration
        ref: "tests/test_migration_cutover.py#test_offline_service_purge_is_separate_idempotent_retryable_cleanup"
        status: pass
      - kind: integration
        ref: "tests/test_migration_cutover.py#test_sqlite_purge_uses_the_finalized_authority_retention_rows"
        status: pass
    human_judgment: false
duration: 24m
completed: 2026-09-10
status: complete
---

# Phase 07 Plan 08: Offline Rollback, Finalize, Abort, and Purge Summary

**Offline migration now retains its prior selection through activation, supports receipt-bound rollback or final acceptance, and performs physical retirement only through separately confirmed, retryable purge.**

## Performance

- **Duration:** 24m
- **Started:** 2026-09-10T03:32:50Z
- **Completed:** 2026-09-10T03:56:50Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Added `OfflineMigrationService.rollback()` and `finalize()` over the authority-owned `activated_offline` state. Rollback restores the exact retained prior selection; finalization requires its own receipt-bound D-15 confirmation and permanently prevents rollback before ordinary workers resume.
- Added authenticated, run-owned candidate abort that only acts before activation and retains source plus maintenance evidence.
- Added separate retained-prior purge with a confirmation bound to exact identities, locators, manifest digests, counts, and bytes. It uses idempotent deletion/absence proof and records partial cleanup as evidence-backed retryable debt without changing activation success.
- Added finalized-only retained-prior access for the memory and SQLite authorities, and a narrow BlobStore maintenance cleanup primitive for exact authority-correlated locators.

## Task Commits

1. **Task 1: Enforce offline rollback eligibility and explicit finalize** - `31c83aa` (test), `190591d` (feat)
2. **Task 2: Abort owned candidates and purge retained prior data separately** - `e741a97` (test), `d546167` (feat)

## Files Created/Modified

- `src/cacheness/storage/migration.py` - Adds typed abort/purge receipts plus receipt, confirmation, rollback, finalization, abort, and purge orchestration.
- `src/cacheness/storage/migration_evidence.py` - Allows deterministic `purge_pending` continuation while retaining signed evidence bounds.
- `src/cacheness/storage/blob_store.py` - Adds the narrow exact-locator maintenance cleanup operation.
- `src/cacheness/storage/memory_lifecycle_authority.py` - Returns retained prior rows only for the matching finalized run.
- `src/cacheness/storage/sqlite_lifecycle_authority.py` - Reads retained prior rows transactionally from the schema-8 authority selection table.
- `tests/test_migration_cutover.py` - Covers rollback, confirmation-gated finalization, abort fencing, response-loss purge retry, stale source refusal, current-candidate protection, and SQLite purge.

## Decisions Made

- Finalization and purge use different confirmation tokens; the purge token is recomputed from the exact retained prior set rather than inferred from a candidate path or listing.
- A purge starts by recording `purge_pending`, so a response loss after an external delete can resume by proving the same locator absent.
- External cleanup is invoked only with authority-correlated locators. Authority state remains `active` throughout purge success or failure.

## Verification

- `uv run --frozen pytest -q tests/test_migration_cutover.py tests/test_lifecycle_authority_contract.py tests/test_migration_run_evidence.py -x -o log_cli=false` — passed (43 tests).
- `uv run --frozen ruff check src/cacheness/storage/migration.py src/cacheness/storage/migration_evidence.py src/cacheness/storage/blob_store.py src/cacheness/storage/memory_lifecycle_authority.py src/cacheness/storage/sqlite_lifecycle_authority.py tests/test_migration_cutover.py --output-format concise` — passed.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The sandbox could not read the existing shared `uv` cache for some verification attempts. The identical frozen commands passed with approved cache access; no dependency, lockfile, or environment change was made.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

The supported local migration lifecycle has explicit rollback, final acceptance, abort, and retryable physical-retirement contracts. Real PostgreSQL/S3 service and performance qualification remain separate Phase 8 work.

## Self-Check: PASSED

- Confirmed all six implementation/test files and this summary exist on disk.
- Confirmed TDD commits `31c83aa`, `190591d`, `e741a97`, and `d546167` exist in git history.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-10*
