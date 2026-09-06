---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "15"
subsystem: lifecycle
tags: [blobstore, unified-cache, concurrency, projection-cas, path-containment]
requires:
  - phase: 03-14
    provides: Generation-conditional committed-only compatibility projections
provides:
  - Exactly-once facade put admission and a bounded full-call clear state machine
  - Post-promotion-only projection and custom-link ownership transitions
  - Canonical empty-state cleanup and same-key hostile-locator preflight
affects: [03-16, 03-17, unified-cache, blobstore, metadata-projection]
actuals:
  tokens: 12279
  tasks: 3
  commits: 4
tech-stack:
  added: []
  patterns:
    - Active-clear ownership spans cleanup while snapshot ownership gates only begin_clear.
    - Compatibility projections are rendered only from the exact promoted authority lineage.
    - Canonical absence is handled by authority operations, not metadata fallback.
key-files:
  created:
    - tests/test_unified_cache_adversarial_lifecycle.py
  modified:
    - src/cacheness/storage/coordination.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/lifecycle.py
    - src/cacheness/core.py
    - tests/test_blob_store_concurrency.py
key-decisions:
  - "A clear's active owner remains admitted through cleanup, but ordinary work reopens immediately after the begin_clear snapshot window while OPEN."
  - "Only the operation-captured M1 locator may transition a projection after exact authority promotion; candidates never publish or adopt peer tokens."
  - "Empty canonical stores use BlobStore clear/delete semantics; legacy cleanup requires explicit recognized composition."
patterns-established:
  - "Use real facade Event hooks at lifecycle boundaries to force concurrent interleavings deterministically."
  - "Validate all supported persisted locator shapes before any same-key mutation."
requirements-completed: [STOR-03, STOR-04, STOR-05, STOR-06, STOR-07]
coverage:
  - id: D1
    description: "Facade puts, queued clears, and close share one bounded admission lifecycle."
    requirement: STOR-03
    verification:
      - kind: integration
        ref: "tests/test_unified_cache_adversarial_lifecycle.py -k admission"
        status: pass
    human_judgment: false
  - id: D2
    description: "Only a successful, exact authority promotion may replace M1 projection and link ownership."
    requirement: STOR-04
    verification:
      - kind: integration
        ref: "tests/test_unified_cache_adversarial_lifecycle.py -k linked_m1 or two_pending"
        status: pass
    human_judgment: false
  - id: D3
    description: "Canonical empty clear and invalidate remain authority operations during a peer first-put intent."
    requirement: STOR-05
    verification:
      - kind: integration
        ref: "tests/test_unified_cache_adversarial_lifecycle.py -k empty_authority or explicit_legacy"
        status: pass
    human_judgment: false
  - id: D4
    description: "Top-level and nested hostile same-key locators fail before any authority, projection, link, or payload mutation."
    requirement: STOR-06
    verification:
      - kind: integration
        ref: "tests/test_unified_cache_adversarial_lifecycle.py -k hostile_locator"
        status: pass
    human_judgment: false
  - id: D5
    description: "Clear admission remains bounded without serializing unrelated lifecycle and containment operations."
    requirement: STOR-07
    verification:
      - kind: integration
        ref: "tests/test_unified_cache_lifecycle_authority.py tests/test_blob_store_close_contract.py tests/test_blob_store_concurrency.py tests/test_filesystem_containment.py"
        status: pass
    human_judgment: false
duration: 31min
completed: 2026-09-06
status: complete
---

# Phase 03 Plan 15: Facade Admission and Exact Promotion Summary

**UnifiedCache now shares BlobStore's exact admission lifecycle, publishes compatibility state only after the winning authority generation, and treats empty or hostile projection state fail-closed.**

## Performance

- **Duration:** 31min
- **Started:** 2026-09-06T05:25:45Z
- **Completed:** 2026-09-06T05:56:00Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Added an exactly-once façade admission reference, a FIFO clear contender queue, and separate active-clear versus short snapshot ownership so close drains correctly without whole-cache payload serialization.
- Moved projection replacement and custom metadata linking behind the exact promoted authority lineage; pre-promotion M2 failures keep M1 projection, payload, and links intact.
- Made clear, invalidate, size cleanup, and expiry cleanup use canonical authority semantics even when empty, and rejected hostile top-level or nested locators before overwrite effects.

## Task Commits

1. **Task 1: Admit facade puts and exclude them from clear snapshot capture** - `2ef997a` (test), `6cd5c72` (feat)
2. **Task 2: Preserve M1 links until an exact promoted lineage owns projection replacement** - `c1d0dea` (fix)
3. **Task 3: Make empty canonical cleanup explicit and reject hostile put locators unchanged** - `bea8565` (fix)

## Files Created/Modified

- `src/cacheness/storage/coordination.py` - Tracks ordinary admission, FIFO clears, full-call ownership, and the bounded snapshot gate.
- `src/cacheness/storage/blob_store.py` - Separates public key-returning puts from private result-bearing admitted puts and composes two-phase clears.
- `src/cacheness/storage/lifecycle.py` - Splits clear snapshot establishment from post-snapshot cleanup.
- `src/cacheness/core.py` - Publishes exact authority-derived projections, preflights locators, and eliminates empty-state legacy inference.
- `tests/test_unified_cache_adversarial_lifecycle.py` - Exercises facade, promotion, empty-state, explicit-legacy, and hostile-locator schedules with deterministic barriers.
- `tests/test_blob_store_concurrency.py` - Aligns a prior clear race assertion with the new snapshot admission boundary.

## Decisions Made

- The active clear remains the only counted clear owner until the full clear call exits; the snapshot owner is deliberately shorter so unrelated operations can proceed during cleanup.
- Projection and custom metadata ownership changes happen only after the original operation's authority promotion remains current and its M1 token matches.
- An empty authority is valid canonical state. Legacy metadata cleanup is selected solely from explicit recognized legacy composition.

## Verification

- Passed the three task-specific deterministic façade filters on frozen Python 3.11/all extras/dev dependencies.
- Passed `tests/test_unified_cache_adversarial_lifecycle.py`, `tests/test_unified_cache_lifecycle_authority.py`, `tests/test_blob_store_close_contract.py`, `tests/test_blob_store_concurrency.py`, and `tests/test_filesystem_containment.py` together; two platform-specific filesystem tests skipped as expected (Windows junction and device-node fixture).
- `tests/test_unified_cache_adversarial_lifecycle.py` is Ruff-clean. `src/cacheness/core.py` retains two pre-existing unused-import findings at lines 379 and 476; no new lint findings were introduced.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Regression alignment] Updated a prior clear race assertion for the revised snapshot boundary**
- **Found during:** Task 1
- **Issue:** The previous test expected an ordinary delete to pass while the active clear still owned the snapshot gate, contradicting the corrected admission state machine.
- **Fix:** Released the test snapshot before racing the ordinary delete and accepted the resulting exact-snapshot outcome.
- **Files modified:** `tests/test_blob_store_concurrency.py`
- **Verification:** The deterministic clear, close, and concurrency suites pass.
- **Committed in:** `6cd5c72`

**Total deviations:** 1 auto-fixed (Rule 1)

## Issues Encountered

- The broad source-plus-test Ruff command reports two pre-existing unused imports in `src/cacheness/core.py`; they are outside this plan's behavior and left unchanged.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plans 03-16 and 03-17 can build on the committed exact-generation façade contract. Phase 3 remains in progress; this plan does not claim phase completion.

## Self-Check: PASSED

- Confirmed all six modified source/test files and this summary exist.
- Confirmed TDD and implementation commits `2ef997a`, `6cd5c72`, `c1d0dea`, and `bea8565` exist in repository history.

---
*Phase: 03-atomic-lifecycle-and-recovery-engine*
*Completed: 2026-09-06*
