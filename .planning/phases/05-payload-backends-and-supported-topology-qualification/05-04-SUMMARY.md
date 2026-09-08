---
phase: 05-payload-backends-and-supported-topology-qualification
plan: "04"
subsystem: database
tags: [postgresql, psycopg, lifecycle-authority, exact-cas, migrations]
requires:
  - phase: 03-atomic-lifecycle-and-recovery-engine
    provides: "AuthorityLifecycleEngine, semantic lifecycle values, and SQLite authority contract"
  - phase: 04-metadata-composition-and-topology-contracts
    provides: "PostgreSQL role boundary and topology-specific capability contract"
provides:
  - "Explicitly initialized and versioned direct-psycopg PostgreSQL authority schema"
  - "Core prepare, proof, promotion, abort, CAS, and cleanup-debt transactions"
  - "Driver-boundary regression contracts for unsafe initialization and uncertain commit recovery"
affects: [05-05, 05-06, 05-07, 05-08, 05-10, phase-07-migration]
actuals:
  tokens: 12917
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - "One fresh caller-supplied psycopg lease per short authority transaction"
    - "Schema identifiers via psycopg.sql.Identifier and all data via bound values"
    - "Exact mutation identity classification after uncertain PostgreSQL commits"
key-files:
  created:
    - src/cacheness/storage/backends/postgresql_lifecycle_authority.py
    - tests/contracts/test_postgresql_lifecycle_authority.py
  modified: []
key-decisions:
  - "PostgreSQL owns descriptor, intent, visibility, and cleanup-debt transactions only; payload I/O remains in the shared engine."
  - "Initialization is explicit and layout version mismatch remains a stopped-worker migration/rebuild boundary."
  - "Fresh lineage sentinels preserve ABA evidence without invalidating their own first create."
patterns-established:
  - "Use bounded transaction-local statement and lock timeouts, returning typed retryable outcomes rather than a distributed coordination layer."
  - "Classify ambiguous promotion by opening a fresh lease and reading the exact operation identity."
requirements-completed: [BACK-04, BACK-05]
coverage:
  - id: D1
    description: "Explicit PostgreSQL authority initialization and non-mutating reopen validation"
    requirement: BACK-04
    verification:
      - kind: unit
        ref: "tests/contracts/test_postgresql_lifecycle_authority.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "Exact PostgreSQL prepare, verification, promotion, abort, and uncertain-commit primitives"
    requirement: BACK-05
    verification:
      - kind: unit
        ref: "tests/contracts/test_postgresql_lifecycle_authority.py; tests/test_lifecycle_authority_contract.py"
        status: pass
    human_judgment: false
duration: 10min
completed: 2026-09-08
status: complete
---

# Phase 5 Plan 04: PostgreSQL Lifecycle Authority Summary

**Direct-psycopg authority primitives give the remote topology explicit schema ownership, exact transition CAS, and durable cleanup evidence without absorbing S3 lifecycle work.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-09-08T12:22:49Z
- **Completed:** 2026-09-08T12:32:18Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added a versioned PostgreSQL authority with explicit idempotent initialization and read-only reopen validation of store identity, capability, tables, and constraints.
- Added short, bound-value transactions for intent preparation, proof attachment, promotion, abort, authority revisions, and attributable cleanup debt.
- Added regression contracts for stale expectations, exact proof CAS, candidate debt, secret redaction, fresh-lineage ABA handling, and uncertain-commit classification through a fresh connection lease.

## Task Commits

1. **Task 1: Explicitly initialize and reopen one versioned PostgreSQL authority** - `6dd878f` (test RED), `49b0611` (feat GREEN)
2. **Task 2: Implement exact prepare-verify-promote-abort transitions in PostgreSQL** - `026d741` (feat)

## Files Created/Modified

- `src/cacheness/storage/backends/postgresql_lifecycle_authority.py` - Direct psycopg authority schema, validation, core mutation transitions, and typed driver boundary.
- `tests/contracts/test_postgresql_lifecycle_authority.py` - No-service driver transcripts for authority safety and recovery contracts.

## Decisions Made

- The adapter accepts only caller-owned connection factories/leases and never stores a connection, DSN, or pool globally.
- Version, identity, capability, table shape, and constraint shape are persisted authority facts; incompatible layouts fail with `CacheBlobMigrationRequiredError` without an implicit migration path.
- PostgreSQL transaction scope is the visibility boundary. The implementation has no S3 calls, advisory locks, SQLAlchemy lifecycle reuse, queues, or second sequence owner.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Prevented a new lineage sentinel from conflicting with its own absent create**
- **Found during:** Task 2
- **Issue:** Promotion creates the durable lineage record needed for future ABA detection. Treating that newly created record as a pre-existing lineage made a first create fail its own absent expectation.
- **Fix:** Used `INSERT ... RETURNING key` to distinguish a transaction's fresh sentinel, while retaining lineage evidence for any pre-existing deleted key.
- **Files modified:** `src/cacheness/storage/backends/postgresql_lifecycle_authority.py`, `tests/contracts/test_postgresql_lifecycle_authority.py`
- **Verification:** Focused driver contract plus shared lifecycle authority contract passed.
- **Committed in:** `026d741`

---

**Total deviations:** 1 auto-fixed (Rule 1)
**Impact on plan:** Required exact-CAS correctness only; it introduced no coordination mechanism or scope expansion.

## Issues Encountered

- The sandbox initially denied access to uv's shared cache and Git's index lock. Scoped approvals allowed the locked environment's existing dependency cache and the required atomic commits; no package was installed.

## User Setup Required

None - no live service configuration is required for this driver-boundary plan. Real PostgreSQL/AWS qualification remains a non-passing Phase 5 release gate.

## Next Phase Readiness

- Plan 05-05 can implement the remaining bounded catalog, clear, reconciliation, tombstone, debt retirement, and projection operations against this persisted authority schema.
- Plans 05-07 through 05-10 still require externally supplied real PostgreSQL and Amazon S3 resources before BACK-05 can be qualified as a supported topology claim.

## Self-Check: PASSED

- Found `src/cacheness/storage/backends/postgresql_lifecycle_authority.py` and `tests/contracts/test_postgresql_lifecycle_authority.py`.
- Found task commits `6dd878f`, `49b0611`, and `026d741` in Git history.

---
*Phase: 05-payload-backends-and-supported-topology-qualification*
*Completed: 2026-09-08*
