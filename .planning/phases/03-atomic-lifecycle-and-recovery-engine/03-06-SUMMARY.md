---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "06"
subsystem: storage lifecycle
tags: [blob-store, lifecycle-authority, sqlite, json-projection, capability-validation]

requires:
  - phase: 03-05
    provides: authority-owned clear, reconciliation, and committed lifecycle state
provides:
  - revision-checked JSON compatibility projections streamed from private SQLite backups
  - authority-only BlobStore composition with topology capability validation
  - same-process-only memory authority claims without PostgreSQL or S3 adapters
affects: [BlobStore, LifecycleAuthority, metadata topology, Phase 4 adapters, Phase 7 rebuild tooling]

actuals:
  tokens: 8175
  tasks: 2
  commits: 3

tech-stack:
  added: []
  patterns:
    - private SQLite backup followed by keyset-streamed, atomic projection publication
    - semantic authority capabilities validated before selecting an internally owned adapter

key-files:
  created: []
  modified:
    - src/cacheness/storage/manifest_repository.py
    - src/cacheness/storage/sqlite_lifecycle_authority.py
    - src/cacheness/storage/lifecycle_authority.py
    - src/cacheness/storage/memory_lifecycle_authority.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/config.py
    - tests/test_manifest_repository_cas.py
    - tests/test_blob_store_read_contract.py

key-decisions:
  - "JSON is a best-effort, revision-tagged compatibility projection rebuilt solely from LifecycleAuthority; it never restores or arbitrates committed state."
  - "Topology selection validates semantic authority capabilities up front; memory is available only through explicit same-process topology and injected authorities retain ownership."

patterns-established:
  - "Projection renderers close the live SQLite source after a private backup, stream the snapshot by keyset, then publish atomically only if the captured authority revision remains current."
  - "BlobStore direct APIs use LifecycleAuthority as the sole committed-state source and refresh derived JSON only after a committed authority mutation."

requirements-completed: [STOR-03, STOR-04, STOR-05, STOR-06, STOR-07]

coverage:
  - id: D1
    description: JSON compatibility projections are private-backup snapshots, preserve the existing metadata shape, and cannot publish a stale revision over a newer authority revision.
    requirement: STOR-03
    verification:
      - kind: integration
        ref: tests/test_manifest_repository_cas.py
        status: pass
    human_judgment: false
  - id: D2
    description: BlobStore composes one authority, rejects unsupported durable or multiprocess memory topology before materialization, and keeps JSON derived from committed authority state.
    requirement: STOR-04
    verification:
      - kind: integration
        ref: tests/test_blob_store_read_contract.py
        status: pass
      - kind: integration
        ref: tests/test_lifecycle_authority_contract.py
        status: pass
    human_judgment: false
  - id: D3
    description: Projection and capability changes preserve authority lifecycle contracts without implementing PostgreSQL or S3 adapters.
    requirement: STOR-07
    verification:
      - kind: other
        ref: .venv/bin/python tools/verify_phase3_ruff_delta.py
        status: pass
      - kind: other
        ref: .venv/bin/python -m compileall -q src/cacheness
        status: pass
    human_judgment: false

duration: 13min
completed: 2026-09-05
status: complete
---

# Phase 03 Plan 06: JSON Projection and Authority Composition Summary

**SQLite-authority JSON projections now rebuild from private keyed backups, while BlobStore rejects unsupported topology and keeps memory authority explicitly same-process only.**

## Performance

- **Duration:** 13 min
- **Started:** 2026-09-05T19:17:52Z
- **Completed:** 2026-09-05T19:30:38Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Added a revision-aware JSON exporter that backs up SQLite privately, releases live authority access before rendering, keyset-streams canonical manifests, fsyncs an atomic replacement, and clears projection debt only through exact revision CAS.
- Made LifecycleAuthority the BlobStore composition root for direct operations; JSON is refreshed only after committed mutations and is never read as lifecycle, CAS, or recovery truth.
- Added capability-aware authority selection: unsupported durable, multiprocess, indexed-paging, or projection topology fails typed before store materialization, while explicit memory topology makes only same-process claims.

## Task Commits

1. **Task 1: Rebuild JSON as a revision-checked streaming projection** - `1e3e957` (test), `1c847f7` (feat)
2. **Task 2: Compose BlobStore through authority capabilities** - `1fb3bd4` (feat)

## Files Created/Modified

- `src/cacheness/storage/manifest_repository.py` - Exports compatible JSON solely from a private authority snapshot and guards final publication against stale revisions.
- `src/cacheness/storage/sqlite_lifecycle_authority.py` - Produces restrictive private SQLite projection backups detached from live authority access.
- `src/cacheness/storage/lifecycle_authority.py` - Defines projection and semantic topology capabilities.
- `src/cacheness/storage/memory_lifecycle_authority.py` - Makes memory's non-durable, non-multiprocess capabilities explicit.
- `src/cacheness/storage/blob_store.py` - Selects and validates one authority, preserves injection ownership, and schedules only best-effort derived projection refreshes after committed mutations.
- `src/cacheness/config.py` - Adds validated semantic topology requirements.
- `tests/test_manifest_repository_cas.py` - Covers projection backup, failure debt, and stale-revision publication races.
- `tests/test_blob_store_read_contract.py` - Covers explicit memory capability rejection and authority-derived JSON compatibility output.

## Decisions Made

- Keep JSON as a rebuildable compatibility artifact: failures leave authority state committed and projection debt dirty, rather than rolling back storage.
- Lock only derived-file publication and compare the live authority revision immediately before replacement, so a slow revision-R render cannot overwrite revision R+1 while no authority transaction spans export.
- Treat `backend="memory"` as explicit same-process selection; a supplied authority instance remains caller-owned and never gets replaced.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Stale projection publication] Prevented an older renderer from overwriting a newer projection.**
- **Found during:** Task 2 authority-composition integration testing.
- **Issue:** A slow revision-R renderer could finish after an automatic revision-R+1 export and replace the newer derived JSON even though its clean acknowledgement would fail.
- **Fix:** Serialized only JSON publication and rechecked the current authority revision while holding that derived-file lock; stale renderers now fail with projection debt left for a later rebuild.
- **Files modified:** `src/cacheness/storage/manifest_repository.py`, `tests/test_manifest_repository_cas.py`.
- **Verification:** Projection race tests and the complete authority/projection/read-contract suite pass.
- **Committed in:** `1fb3bd4`.

**2. [Rule 1 - Compatibility boundary] Corrected a read-contract expectation that conflicted with derived projection refresh.**
- **Found during:** Task 2 regression verification.
- **Issue:** A historical test expected direct mutation APIs to leave a corrupt JSON projection untouched, which contradicts JSON's new rebuildable-projection contract.
- **Fix:** Kept direct authority reads non-mutating, but asserted that a committed mutation repairs the derived JSON from authority state.
- **Files modified:** `tests/test_blob_store_read_contract.py`.
- **Verification:** `tests/test_blob_store_read_contract.py` and the combined 206-test suite pass.
- **Committed in:** `1fb3bd4`.

---

**Total deviations:** 2 auto-fixed (Rule 1).
**Impact on plan:** Both fixes protect the authority-only lifecycle boundary and compatibility behavior without adding new backend adapters or changing public metadata shapes.

## Issues Encountered

- The plan's `STOR-03` through `STOR-07` identifiers are not present in the current requirements ledger, so no ledger checkboxes or traceability rows could be marked; no unrelated requirements content was changed.
- State progress recalculation correctly left the phase aggregate untouched because Phase 03 remains in progress; the plan counter and roadmap now report Plan 07 next and 6/10 replacement-plan summaries.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 4 can consume the capability contract for later PostgreSQL/S3 work without inheriting a second lifecycle authority. JSON remains disposable and can be rebuilt from a committed SQLite authority snapshot.

## Self-Check: PASSED

All eight modified source/test artifacts and the three Task 1/Task 2 TDD commits are present in repository history.

---
*Phase: 03-atomic-lifecycle-and-recovery-engine*
*Completed: 2026-09-05*
