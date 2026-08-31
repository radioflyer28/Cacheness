---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "08"
subsystem: storage lifecycle
tags: [blobstore, concurrency, manifest-cas, read-integrity, lifecycle]
requires:
  - phase: 03-07
    provides: bounded reconciliation and clear lifecycle recovery
provides:
  - refcounted per-instance coordination for exact physical keys
  - deterministic same-key CAS race coverage and finite clear snapshot coverage
  - committed read acquisition with one authenticated generation retry
affects: [BlobStore, lifecycle, reconciliation, Phase 03 plans 09-10]
actuals:
  tokens: 11222
  tasks: 2
  commits: 7
tech-stack:
  added: []
  patterns:
    - per-key refcounted local coordination with independent-store CAS authority
    - M1 snapshot M2 read validation with one bounded retry
key-files:
  created:
    - tests/test_blob_store_concurrency.py
  modified:
    - src/cacheness/storage/coordination.py
    - src/cacheness/storage/lifecycle.py
    - src/cacheness/storage/blob_store.py
    - tests/test_blob_store_read_contract.py
key-decisions:
  - "Key locks are per BlobStore instance; independent instances remain governed solely by exact manifest CAS."
  - "BlobStore reads authenticate M1 and M2 around each private snapshot and retry only a proven newer generation once."
  - "Current clear tests target lifecycle evidence and snapshot seams, not predecessor clear-recovery admission."
patterns-established:
  - "Retain a registry entry before waiting and remove that exact entry only after its final user exits."
  - "Direct public clear boundaries translate unclassified lifecycle storage failures into CacheBlobBackendError."
requirements-completed: [STOR-03, STOR-07]
coverage:
  - id: D1
    description: "Same-key mutation races use local key ordering while independent stores select one winner through exact manifest CAS."
    requirement: STOR-03
    verification:
      - kind: integration
        ref: "tests/test_blob_store_concurrency.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "Committed reads verify M1 and M2 around a private snapshot, with one bounded retry only after generation movement."
    requirement: STOR-07
    verification:
      - kind: integration
        ref: "tests/test_blob_store_concurrency.py; tests/test_blob_store_read_contract.py"
        status: pass
    human_judgment: false
duration: 16min
completed: 2026-08-31
status: complete
---

# Phase 03 Plan 08: Deterministic Races and Committed Reads Summary

**BlobStore now coordinates only same-key local work, leaves independent winners to manifest CAS, and returns a stable committed snapshot after at most one proven generation retry.**

## Performance

- **Duration:** 16min
- **Started:** 2026-08-31T04:07:55Z
- **Completed:** 2026-08-31T04:24:06Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added refcounted `KeyCoordinatorRegistry` entries that survive waiting callers, retire at zero users, and acquire multiple physical identities in deterministic order.
- Routed direct same-key BlobStore lifecycle work through per-instance coordination while retaining exact manifest CAS as the independent-store authority.
- Reworked direct reads to authenticate M1, snapshot once, authenticate M2, and retry exactly once only when both authenticated generations differ.
- Added deterministic write/write, write/delete, distinct-key, clear-post-snapshot, read/write, and read/delete contracts without sleep-based race assertions.
- Migrated stale predecessor-clear test hooks to active lifecycle evidence and snapshot seams, preserving public error and recovery coverage.

## Task Commits

1. **Task 1: Coordinate keys locally and prove backend-CAS race winners** - `be52d4d`, `efa8a40` (test, feat)
2. **Task 2: Reacquire a moved committed generation exactly once** - `c120e42`, `ed54fac`, `576c914`, `d565543`, `7734026` (test, feat)

## Files Created/Modified

- `src/cacheness/storage/coordination.py` - refcounted exact-key locks and sorted multi-key acquisition.
- `src/cacheness/storage/blob_store.py` - local key admission, M1/M2 committed read loop, and typed clear boundary.
- `src/cacheness/storage/lifecycle.py` - exposes non-committed clear targets as direct lifecycle conflicts.
- `tests/test_blob_store_concurrency.py` - deterministic mutation, clear, and read race matrix.
- `tests/test_blob_store_read_contract.py` - current lifecycle seam contracts and M2 call ordering.

## Decisions Made

- Per-key coordination is deliberately per store instance, so it cannot disguise cross-instance CAS correctness.
- A missing or non-committed second authority record after M1 is a typed lifecycle conflict, never a payload miss or authority guess.
- Clear operation storage failures are translated at the direct BlobStore boundary while existing typed backend and conflict errors retain their identity.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Public error boundary] Restored typed clear failures for active lifecycle seams**
- **Found during:** Task 2
- **Issue:** Raw lifecycle storage errors from `clear()` bypassed the direct `CacheBlobBackendError` taxonomy.
- **Fix:** Wrapped unclassified lifecycle storage failures at the public boundary and preserved existing typed conflicts/backends.
- **Files modified:** `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/lifecycle.py`
- **Verification:** Full read-contract suite and targeted Ruff passed.
- **Committed in:** `d565543`

**2. [Rule 2 - Test migration] Replaced retired predecessor-clear hooks with active lifecycle evidence seams**
- **Found during:** Task 2
- **Issue:** Several read-contract tests expected the retired global predecessor clear coordinator to gate ordinary reads or perform current clears.
- **Fix:** Replaced obsolete hooks with operation-record creation, manifest snapshot, and clear-target seams while retaining typed error, cause, and recovery assertions.
- **Files modified:** `tests/test_blob_store_read_contract.py`
- **Verification:** `uv run pytest -q -o log_cli=false tests/test_blob_store_read_contract.py -x` passed.
- **Committed in:** `576c914`

**Total deviations:** 2 auto-fixed (1 Rule 1, 1 Rule 2).

## Verification

- `uv run pytest -q -o log_cli=false tests/test_blob_store_concurrency.py -x` — 8 passed
- `uv run pytest -q -o log_cli=false tests/test_blob_store_read_contract.py -x` — 48 passed
- `uv run ruff check src/cacheness/storage/coordination.py src/cacheness/storage/lifecycle.py src/cacheness/storage/blob_store.py tests/test_blob_store_concurrency.py tests/test_blob_store_read_contract.py` — passed

## Known Stubs

None.

## Next Phase Readiness

Phase 03 can build on bounded local key coordination and stable committed reads without relying on global normal-operation locks or read-side lifecycle evidence.

## Self-Check: PASSED

- Verified all five implementation/test artifacts exist.
- Verified task commits `be52d4d`, `efa8a40`, `c120e42`, `ed54fac`, `576c914`, `d565543`, and `7734026` exist in git history.

---
*Phase: 03-atomic-lifecycle-and-recovery-engine*
*Completed: 2026-08-31*
