---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "13"
subsystem: storage-lifecycle
tags: [postgresql, migration, lifecycle-authority, worker-fence, pytest]
requires:
  - phase: 07-explicit-migration-and-rebuild-cutover
    provides: Canonical activated-offline migration state and ordinary-worker maintenance fence
provides:
  - Persisted post-identity worker admission fence before BlobStore readiness
  - PostgreSQL mutation preflight that uses the same canonical worker guard
affects: [phase-7-migration-cutover, phase-8-qualification, migration-maintenance]
actuals:
  tokens: 2676
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Open and validate the remote authority before using its persisted worker-admission state
    - Reuse the canonical ordinary-worker guard at every remote mutation admission boundary
key-files:
  created: []
  modified:
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/backends/postgresql_lifecycle_authority.py
    - tests/contracts/test_postgresql_lifecycle_authority.py
key-decisions:
  - "A pre-identity idle observation is only an early fast check; persisted PostgreSQL state must authorize worker readiness after identity load."
  - "BlobStore initialization and direct PostgreSQL preflight reuse require_ordinary_worker_access instead of adding local coordination or a second authority."
patterns-established:
  - "Remote worker admission opens the canonical authority first, then performs a side-effect-free persisted-state guard before payload or signing work."
requirements-completed: [MIGR-04, MIGR-05]
coverage:
  - id: D1
    description: Fresh PostgreSQL-backed BlobStore initialization rejects persisted activated_offline before payload materialization, key access, or readiness.
    requirement: MIGR-04
    verification:
      - kind: unit
        ref: tests/contracts/test_postgresql_lifecycle_authority.py#test_fresh_blobstore_initialize_rechecks_activated_offline_after_postgresql_identity_load
        status: pass
    human_judgment: false
  - id: D2
    description: PostgreSQL mutation preflight reads persisted post-open state, rejects activated_offline, permits idle and active, and fails closed on malformed state.
    requirement: MIGR-05
    verification:
      - kind: unit
        ref: tests/contracts/test_postgresql_lifecycle_authority.py#test_postgresql_preflight_mutation_checks_persisted_worker_fence_after_open
        status: pass
      - kind: unit
        ref: tests/contracts/test_postgresql_lifecycle_authority.py#test_postgresql_preflight_mutation_preserves_worker_states_and_fails_closed
        status: pass
    human_judgment: false
duration: 2h 57m
completed: 2026-09-11
status: complete
---

# Phase 07 Plan 13: PostgreSQL Worker Admission Fence Summary

**Fresh PostgreSQL-backed BlobStores now reload canonical persisted migration state before readiness, and direct mutation preflight uses that same activated-offline worker fence.**

## Performance

- **Duration:** 2h 57m
- **Started:** 2026-09-11T02:33:17Z
- **Completed:** 2026-09-11T05:30:24Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Added a post-preflight ordinary-worker fence in `BlobStore.initialize()` before payload materialization, manifest-key access, or `_initialized = True`.
- Made `PostgresqlLifecycleAuthority.preflight_mutation()` open canonical identity and then invoke the existing persisted-state worker guard.
- Added deterministic adapter-backed coverage for activated-offline rejection, idle/active admission, malformed state rejection, call ordering, and zero payload materialization.

## Task Commits

Each TDD task was committed atomically:

1. **Task 1: Fence fresh BlobStore initialization after PostgreSQL identity load** - `3f30c89` (`test`), `690ce11` (`feat`)
2. **Task 2: Make PostgreSQL mutation preflight enforce the persisted worker fence** - `2e596d0` (`test`), `676bb1f` (`feat`)

## Files Created/Modified

- `src/cacheness/storage/blob_store.py` - Re-checks the canonical worker fence after authority preflight and before any readiness side effects.
- `src/cacheness/storage/backends/postgresql_lifecycle_authority.py` - Calls the existing ordinary-worker guard after persisted identity validation in mutation preflight.
- `tests/contracts/test_postgresql_lifecycle_authority.py` - Adds fresh-construction and direct-preflight deterministic state coverage, including normal and malformed persisted states.

## Decisions Made

- A synthetic pre-identity `idle` result remains only an early fast check. The persisted PostgreSQL state is authoritative after `open()` completes.
- The repair converges BlobStore initialization and direct preflight on `require_ordinary_worker_access`; it adds no lock, lease, sidecar, filesystem/object authority, online writer protocol, or live-service claim.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Restored injected payload cleanup to its owning fixture**
- **Found during:** Task 2 test-first setup
- **Issue:** A test insertion displaced the caller-owned remote payload cleanup into a later fixture, so the original injected participant was not released by its own test.
- **Fix:** Restored each injected payload's cleanup inside the fixture that owns it.
- **Files modified:** `tests/contracts/test_postgresql_lifecycle_authority.py`
- **Verification:** Focused selectors and the full PostgreSQL authority contract file pass.
- **Committed in:** `2e596d0`

**2. [Rule 1 - Bug] Extended an existing remote-initialize transcript for persisted guard reads**
- **Found during:** Task 2 full contract verification
- **Issue:** The deterministic fixture supplied only the former schema-validation transcript and therefore could not model the two now-required persisted idle-state reads.
- **Fix:** Added exact idle-state responses and updated the expected connection count without changing the production topology or lifecycle contract.
- **Files modified:** `tests/contracts/test_postgresql_lifecycle_authority.py`
- **Verification:** Full contract file and scoped Ruff pass.
- **Committed in:** `676bb1f`

---

**Total deviations:** 2 auto-fixed (Rule 1)
**Impact on plan:** Both corrections keep deterministic contract evidence aligned with the planned canonical PostgreSQL worker fence; no lifecycle authority or coordination surface was added.

## Issues Encountered

- The first Task 2 test insertion exposed a fixture-cleanup placement error before test collection. It was corrected before the intended red test run, then retained as an explicit regression-harness ownership fix.

## Known Stubs

None. The modified production and test files contain no intentional rendering-path placeholder, skipped test, or unwired mock-data flow.

## User Setup Required

None - no external service configuration required. PostgreSQL coverage remains deterministic and adapter-backed; live PostgreSQL qualification is still Phase 8 work.

## Next Phase Readiness

PostgreSQL workers cannot use a synthetic pre-identity idle state to bypass activated-offline maintenance. Phase 8 retains ownership of live PostgreSQL and S3 qualification; this plan makes no service availability claim.

## Self-Check: PASSED

- Confirmed all three modified code/test artifacts and this summary exist.
- Confirmed task commits `3f30c89`, `690ce11`, `2e596d0`, and `676bb1f` exist in repository history.
- Re-ran five focused fence cases, the full PostgreSQL lifecycle-authority contract file, and scoped Ruff: passed.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-11*
