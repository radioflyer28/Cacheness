---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "12"
subsystem: storage-migration
tags: [migration, recovery, lifecycle-authority, bounded-evidence, pytest]
requires:
  - phase: 07-explicit-migration-and-rebuild-cutover
    provides: Explicit offline migration authority, authenticated maintenance evidence, and immutable candidate payloads
provides:
  - Bounded migration-run partitioning before destination mutation
  - Authority-attributed candidate batch recovery with never-reused attempt locators
  - STAGING resume and retryable exact cleanup debt for attributed candidates only
affects: [phase-7-migration-cutover, phase-8-qualification, migration-maintenance]
actuals:
  tokens: 22958
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Authority-owned candidate descriptors are the only recovery input for external payload effects
    - Authenticated maintenance evidence stores bounded batch references and exact cleanup-debt identifiers
key-files:
  created: []
  modified:
    - src/cacheness/storage/migration_authority.py
    - src/cacheness/storage/migration_evidence.py
    - src/cacheness/storage/migration.py
    - src/cacheness/storage/memory_lifecycle_authority.py
    - src/cacheness/storage/sqlite_lifecycle_authority.py
    - src/cacheness/storage/backends/postgresql_lifecycle_authority.py
    - tests/test_migration_cutover.py
key-decisions:
  - "A payload written before its authority checkpoint remains an invisible, unattributed orphan; it is neither adopted nor promised exact cleanup."
  - "Migration recovery and abort use only authenticated descriptors retained by the existing lifecycle authority, never listings or deterministic-locator reconstruction."
  - "Oversized work returns deterministic split-required partitions before destination mutation instead of expanding evidence without a declared bound."
patterns-established:
  - "Checkpoint each external batch by first recording its exact descriptor in the sole authority, then recording only bounded progress references in maintenance evidence."
  - "Retry cleanup treats exact absence as success and retains digest-bound debt for payload mismatch or uncertain deletion outcomes."
requirements-completed: [MIGR-04, MIGR-05]
coverage:
  - id: D1
    description: Bounded migration runs split oversized inventories before mutation and retain only authority-attributed candidate batches for recovery.
    requirement: MIGR-04
    verification:
      - kind: unit
        ref: tests/test_migration_cutover.py#test_migration_run_enforces_entry_byte_and_evidence_limits_with_split_plan
        status: pass
      - kind: unit
        ref: tests/test_migration_cutover.py#test_uncheckpointed_candidate_orphan_remains_invisible_unadopted_and_outside_exact_cleanup
        status: pass
    human_judgment: false
  - id: D2
    description: STAGING resume and abort process only authority-attributed descriptors, preserve bounded cleanup debt, and retry already-absent candidates safely.
    requirement: MIGR-05
    verification:
      - kind: unit
        ref: tests/test_migration_cutover.py#test_resume_and_abort_staging_use_only_authority_attributed_batches
        status: pass
      - kind: unit
        ref: tests/test_migration_cutover.py#test_offline_service_abort_removes_only_its_unactivated_candidate
        status: pass
    human_judgment: false
duration: 23min
completed: 2026-09-11
status: complete
---

# Phase 07 Plan 12: Bounded Migration Recovery Summary

**Bounded offline migration now checkpoints exact candidate batches in the existing authority, splits oversized runs before mutation, and retries only attributed cleanup while leaving pre-checkpoint orphans invisible and unadopted.**

## Performance

- **Duration:** 23 min
- **Started:** 2026-09-11T02:07:12Z
- **Completed:** 2026-09-11T02:29:54Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Added validated candidate-entry and candidate-batch receipts, per-run entry/byte/evidence limits, and deterministic split-required partitions that return before destination mutation.
- Replaced deterministic retry locators with fresh attempt-local identifiers; only descriptors persisted by the existing lifecycle authority can be resumed, verified, activated, or aborted.
- Extended STAGING recovery so exact absence succeeds idempotently, byte or ownership mismatches become bounded cleanup debt, and the authority clears its unactivated candidate only after all attributed cleanup succeeds.

## Task Commits

Each TDD task was committed atomically:

1. **Task 1: Enforce bounded maintenance runs and checkpoint completed authority batches** - `98cf820` (`test`), `49c83cc` (`feat`)
2. **Task 2: Resume and abort STAGING with exact absence proof and durable cleanup debt** - `d381001` (`test`), `b693719` (`feat`)

## Files Created/Modified

- `src/cacheness/storage/migration_authority.py` - Validated, lossless candidate entry and batch descriptor contracts.
- `src/cacheness/storage/migration_evidence.py` - Versioned bounded checkpoint references, aggregate progress, and cleanup-debt capacity.
- `src/cacheness/storage/migration.py` - Run splitting, fresh candidate locators, authority-attributed stage/resume, and safe abort retry orchestration.
- `src/cacheness/storage/memory_lifecycle_authority.py` - In-process attributed candidate append/read/discard support.
- `src/cacheness/storage/sqlite_lifecycle_authority.py` - Durable local authority support for exact candidate append/read/discard.
- `src/cacheness/storage/backends/postgresql_lifecycle_authority.py` - Equivalent candidate append/read/discard behavior for the declared authority contract.
- `tests/test_migration_cutover.py` - TDD coverage for bounds, orphan handling, STAGING resume, idempotent absence, and cleanup debt.

## Decisions Made

- The accepted post-publication/pre-authority-checkpoint window is explicit: an immutable payload can remain invisible but unattributed, is never discovered or adopted, and has no Phase 7 exact-reclamation guarantee.
- Exact candidate descriptions remain in existing authority state; maintenance evidence stores only authenticated, bounded batch references and cleanup-debt identifiers.
- Candidate locators are minted per external attempt, so retries cannot reuse an earlier immutable generation locator.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Extended existing concrete authorities for attributed batch recovery**
- **Found during:** Task 1 and Task 2
- **Issue:** The existing candidate rows could record one complete candidate only, so a post-checkpoint resume or exact cleanup could not recover a bounded batch without reconstructing from guessed locators.
- **Fix:** Reused existing candidate evidence in memory, SQLite, and PostgreSQL authorities to append an exact candidate prefix, read only descriptors for the named run, and discard only the exact unactivated candidate after external retirement succeeds.
- **Files modified:** `memory_lifecycle_authority.py`, `sqlite_lifecycle_authority.py`, `postgresql_lifecycle_authority.py`
- **Verification:** Migration/evidence and local authority suites pass.
- **Committed in:** `49c83cc`, `b693719`

---

**Total deviations:** 1 auto-fixed (Rule 2)
**Impact on plan:** The extension reuses the existing lifecycle authority and candidate evidence; it adds no journal, lifecycle authority, coordination mechanism, or payload backend.

## Issues Encountered

- The first broader cutover run showed that authority-attributed candidates must be cleared after a successful abort before a later maintenance run can begin. Task 2 added exact authority discard after external cleanup; no visibility transition or new lifecycle state was introduced.

## Known Stubs

None. The modified production and test files contain no rendering-path placeholders, skipped tests, or intentional stubbed data flows.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Later migration plans can rely on bounded authority-attributed recovery and the documented orphan boundary. Live PostgreSQL/AWS S3 qualification, obstore adoption, and broader performance claims remain outside this plan and Phase 7’s accepted scope.

## Self-Check: PASSED

- Confirmed all eight modified production/test artifacts and `07-12-SUMMARY.md` exist.
- Confirmed task commits `98cf820`, `49c83cc`, `d381001`, and `b693719` exist in repository history.
- Re-ran 55 migration, evidence, local SQLite-authority, and lifecycle-authority tests: passed.
- Re-ran scoped Ruff checks for every modified production and test file: passed.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-11*
