---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "06"
subsystem: storage migration authority
tags: [sqlite, postgresql, offline-migration, lifecycle-authority, projection-rebuild]
requires:
  - phase: 07-04
    provides: authenticated offline migration planning and source inventory contracts
  - phase: 07-05
    provides: explicit maintenance evidence and projection outcome contracts
provides:
  - SQLite schema-8 and PostgreSQL schema/capability-4 migration publication baseline
  - authority-owned candidate, activated_offline, active, and rolled_back selection lifecycle
  - idempotent local interruption recovery with derived-only projection rebuild results
affects: [phase-07-cutover, phase-08-qualification, storage-lifecycle]
actuals:
  tokens: 14564
  tasks: 3
  commits: 4
tech-stack:
  added: []
  patterns:
    - exact authority receipt classification for lost activation responses
    - immutable external candidate effects separated from authority ACID state
key-files:
  created: []
  modified:
    - src/cacheness/storage/migration_authority.py
    - src/cacheness/storage/sqlite_lifecycle_authority.py
    - src/cacheness/storage/memory_lifecycle_authority.py
    - src/cacheness/storage/migration.py
    - src/cacheness/storage/backends/postgresql_lifecycle_authority.py
    - src/cacheness/storage/blob_store.py
    - tests/test_migration_cutover.py
key-decisions:
  - "RQ-01 accepted: SQLite schema 8 and PostgreSQL schema/capability 4 are the initial authority publication baseline; development schemas 7/3 have no compatibility edge."
  - "Authority state alone selects candidate, retained prior, and canonical rows; payloads, evidence, and projection outcomes are immutable attributable external effects."
  - "activated_offline blocks every ordinary BlobStore worker entry point until an explicit authority rollback or finalization."
patterns-established:
  - "Recovery first asks the authority for an exact receipt, never infers activation from candidate paths or work files."
  - "Projection rebuild runs after canonical activation and returns a derived current/dirty result that retains the activation receipt."
requirements-completed: [MIGR-04, MIGR-05]
coverage:
  - id: D1
    description: "Complete whole-store candidate staging, verification, and authority-only publication retain the prior selection and block workers offline."
    requirement: MIGR-04
    verification:
      - kind: integration
        ref: "uv run --frozen pytest -q tests/test_migration_cutover.py tests/test_lifecycle_authority_contract.py -x -o log_cli=false"
        status: pass
      - kind: integration
        ref: "uv run --frozen --extra postgresql pytest -q tests/contracts/test_postgresql_lifecycle_authority.py tests/test_sqlite_metadata_bootstrap_atomicity.py -x -o log_cli=false"
        status: pass
    human_judgment: false
  - id: D2
    description: "Explicit maintenance evidence resumes local interruption boundaries, including lost activation observations, without making projections authoritative."
    requirement: MIGR-05
    verification:
      - kind: integration
        ref: "uv run --frozen pytest -q tests/test_migration_cutover.py tests/test_migration_run_evidence.py tests/test_projection_sql_atomicity.py -x -o log_cli=false"
        status: pass
    human_judgment: false
duration: 20min
completed: 2026-09-10
status: complete
---

# Phase 07 Plan 06: Offline Authority Publication and Recovery Summary

**SQLite schema-8 and PostgreSQL schema/capability-4 authority publication with retained prior selections, offline worker fencing, exact interruption recovery, and derived-only projection rebuilds.**

## Performance

- **Duration:** 20 min
- **Started:** 2026-09-10T02:39:31Z
- **Completed:** 2026-09-10T02:59:30Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments

- Recorded the explicit `proceed RQ-01` decision and published the first release authority baseline: SQLite 8 and PostgreSQL 4, without a development-schema compatibility route.
- Added authority-owned complete candidate/prior rows and whole-store activation with `activated_offline` worker fencing, explicit rollback/finalize paths, and no cross-resource ACID claim.
- Made recovery classify a response-lost activation from exact authority-owned receipt/state; projection rebuilds run separately and cannot revoke canonical activation.

## Task Commits

1. **Task 1: Confirm the one-way authority publication schema baseline** - accepted as `proceed RQ-01` (decision recorded in planning state; no code commit).
2. **Task 2: Define shared publication contracts and activate complete SQLite candidates** - `878fbaf` (test), `8ea92f1` (feat).
3. **Task 3: Resume every local interruption and rebuild projections separately** - `aaaf527` (test), `dffe75f` (feat).

## Files Created/Modified

- `src/cacheness/storage/migration_authority.py` - shared publication states, exact candidate receipts, prior-store, rollback, and finalize receipts.
- `src/cacheness/storage/sqlite_lifecycle_authority.py` - schema-8 authority transactions, retained candidate/prior rows, recovery receipt lookup, and offline fence.
- `src/cacheness/storage/memory_lifecycle_authority.py` - same-process implementation of the authority publication and recovery state machine.
- `src/cacheness/storage/migration.py` - exact receipt-driven resume classification and post-activation derived projection rebuilding.
- `src/cacheness/storage/backends/postgresql_lifecycle_authority.py` - schema/capability-4 authority DDL and contract target.
- `src/cacheness/storage/blob_store.py` - ordinary-worker access guard during `activated_offline`.
- `tests/test_migration_cutover.py` - whole-store, worker fencing, response-loss, and projection-separation contracts.

## Decisions Made

- RQ-01 was explicitly accepted before persistence semantics changed. Candidate activation remains wholly authority-owned; external payload and evidence effects remain immutable and attributable.
- `activated_offline` is intentionally not a worker-serving state. Only narrow maintenance status/receipt/rollback/finalize calls remain available until the operator resolves the cutover.
- PostgreSQL 3 and SQLite 7 are rejected as unsupported pre-release development layouts, rather than upgraded implicitly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Applied the PostgreSQL 4 baseline alongside the SQLite 8 implementation**
- **Found during:** Task 2
- **Issue:** The authority contract's required PostgreSQL capability/schema target was not yet represented by its singleton state and retained-entry DDL.
- **Fix:** Updated the PostgreSQL capability marker and schema contract to v4 with the same authority-owned publication fields and exact candidate/prior row contract.
- **Files modified:** `src/cacheness/storage/backends/postgresql_lifecycle_authority.py`, `tests/contracts/test_postgresql_lifecycle_authority.py`
- **Verification:** PostgreSQL/SQLite schema-contract suite passed.
- **Committed in:** `8ea92f1`

**2. [Rule 2 - Missing critical functionality] Routed ordinary BlobStore work through the offline authority fence**
- **Found during:** Task 2
- **Issue:** Authority-level fencing alone could not guarantee that every ordinary BlobStore operation was rejected during `activated_offline`.
- **Fix:** Added the canonical-store guard so ordinary open, read, query, initialization, and mutation paths consult the authority before proceeding.
- **Files modified:** `src/cacheness/storage/blob_store.py`
- **Verification:** Cutover and authority contract tests passed.
- **Committed in:** `8ea92f1`

**3. [Rule 1 - Bug] Classified a lost activation response before pre-activation destination revalidation**
- **Found during:** Task 3
- **Issue:** Resume first treated the authority's post-activation revision as stale, so it could not recognize a committed activation whose evidence observation was lost.
- **Fix:** Added exact authority receipt lookup and used it first for verified evidence; the resulting evidence is then revalidated against the activated revision.
- **Files modified:** `src/cacheness/storage/migration.py`, `src/cacheness/storage/memory_lifecycle_authority.py`, `src/cacheness/storage/sqlite_lifecycle_authority.py`, `tests/test_migration_cutover.py`
- **Verification:** Focused migration, evidence, projection, and authority suites passed.
- **Committed in:** `dffe75f`

---

**Total deviations:** 3 auto-fixed (2 Rule 2, 1 Rule 1).
**Impact on plan:** All changes enforce the accepted authority boundary and recovery behavior; no unsupported coordination mechanism or compatibility path was introduced.

## Issues Encountered

- The sandbox initially could not open the shared `uv` cache for the broader validation run. The unchanged validation command passed with approved access to that existing dependency cache.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Local same-backend offline cutover has a concrete schema baseline, deterministic recovery boundary, and explicit projection separation for downstream qualification.
- Real PostgreSQL/S3 service qualification remains the separate Phase 8 scope; this plan only fixes the PostgreSQL authority contract/version baseline.

## Self-Check: PASSED

- Confirmed all nine modified implementation/test files exist.
- Confirmed `878fbaf`, `8ea92f1`, `aaaf527`, and `dffe75f` exist in git history.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-10*
