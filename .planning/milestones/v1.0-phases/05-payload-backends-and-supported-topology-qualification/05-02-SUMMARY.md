---
phase: 05-payload-backends-and-supported-topology-qualification
plan: "02"
subsystem: storage-testing
tags: [blobstore, payload-generations, lifecycle-faults, integrity, recovery]
requires:
  - phase: 05-01
    provides: explicit local topology identities and qualified-profile catalog
provides:
  - reusable filesystem and memory generation-I/O participant contract
  - deterministic fault coverage across stage, publication, promotion, and cleanup boundaries
affects: [05-03, 05-04, s3-payload-adapter, topology-qualification]
actuals:
  tokens: 3960
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - tier-aware five-method payload-participant contracts
    - fault-only lifecycle boundaries that do not alter public timing hooks
key-files:
  created:
    - tests/contracts/test_payload_generation_io.py
    - tests/test_payload_faults.py
  modified:
    - src/cacheness/storage/backends/blob_backends.py
    - src/cacheness/storage/lifecycle.py
key-decisions:
  - "Memory generation publication rejects an existing locator rather than replacing bytes."
  - "Stage fault boundaries notify only the fault hook, preserving the existing public timing-observer sequence."
patterns-established:
  - "Classify integrity, recovery, progress, and performance in distinct contract assertions."
  - "Use named BoundaryHooks faults and authority snapshots, never sleeps or contender-success deadlines."
requirements-completed: [BACK-01, BACK-04]
coverage:
  - id: D1
    description: "Filesystem and memory participants implement the complete guarded generation-I/O contract at their declared tiers."
    requirement: BACK-01
    verification:
      - kind: unit
        ref: tests/contracts/test_payload_generation_io.py
        status: pass
    human_judgment: false
  - id: D2
    description: "Local lifecycle faults converge to an old or new complete generation with attributable cleanup debt."
    requirement: BACK-04
    verification:
      - kind: unit
        ref: tests/test_payload_faults.py
        status: pass
    human_judgment: false
duration: 56min
completed: 2026-09-08
status: complete
---

# Phase 05 Plan 02: Payload Generation and Fault Contracts Summary

**Filesystem and memory payload participants now share an executable immutable-generation contract, with deterministic engine-boundary recovery tests.**

## Performance

- **Duration:** 56 min
- **Completed:** 2026-09-08T11:55:16Z
- **Tasks:** 2/2
- **Files modified:** 4
- **Verification:** 49 focused lifecycle and payload tests passed; Ruff passed for all four changed files.

## Accomplishments

- Added a tier-aware contract that materializes each local participant exactly as `BlobStore` does, publishes native handler bytes once, checks private mode-0600 snapshots, validates signed manifest digest/size, and proves exact idempotent deletion.
- Made in-memory generation publication reject an existing locator, matching immutable filesystem generation semantics without adding lifecycle coordination.
- Added deterministic fault cases for staging, intent, publish, verification, promotion, and cleanup, with authority-state convergence and cleanup-debt assertions.

## Task Commits

1. **Task 1: Run one immutable local generation through the complete participant contract** — `c7b25ef` (red test), `ae07518` (immutable memory publication)
2. **Task 2: Classify lifecycle boundary faults without adding coordination mechanisms** — `d523f22` (red test), `3ba0acb` (fault-only stage boundaries)

## Files Created/Modified

- `tests/contracts/test_payload_generation_io.py` — reusable local generation-I/O contract and tier declaration checks.
- `tests/test_payload_faults.py` — table-driven deterministic lifecycle fault classification.
- `src/cacheness/storage/backends/blob_backends.py` — exclusive in-memory immutable generation publication.
- `src/cacheness/storage/lifecycle.py` — fault-only before/after stage test boundaries.

## Decisions Made

- The memory adapter keeps its same-process tier but must still reject replacement of an already published generation locator.
- Stage boundaries remain test-only fault seams and do not alter the existing `test_hook` timing sequence or lifecycle authority.
- Progress checks accept the topology's documented outcomes; no test imposes a timing deadline or requires every contender to succeed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Memory generation publication could overwrite an existing immutable locator**

- **Found during:** Task 1
- **Fix:** Reject an already-present `memory://` generation before writing bytes.
- **Files modified:** `src/cacheness/storage/backends/blob_backends.py`
- **Verification:** `tests/contracts/test_payload_generation_io.py`
- **Committed in:** `ae07518`

**2. [Rule 2 - Missing critical functionality] The sole lifecycle engine lacked deterministic stage-boundary fault injection**

- **Found during:** Task 2
- **Fix:** Added fault-only before/after-stage boundaries without changing lifecycle state, admission, visibility, or public timing hooks.
- **Files modified:** `src/cacheness/storage/lifecycle.py`
- **Verification:** `tests/test_payload_faults.py`
- **Committed in:** `3ba0acb`

**Total deviations:** 2 auto-fixed (1 Rule 1, 1 Rule 2). Both changes are scoped to immutable participant behavior and deterministic testing; neither adds coordination machinery.

## Known Stubs

None.

## Self-Check: PASSED

- Confirmed both contract files exist and the four task commits are reachable.

## Next Phase Readiness

S3 work has a precise local contract target: it must implement only the existing five-method participant seam, preserve immutable publication and verified private snapshots, and use the same named fault taxonomy without duplicating lifecycle sequencing.

---
*Phase: 05-payload-backends-and-supported-topology-qualification*
*Plan: 02*
*Completed: 2026-09-08*
