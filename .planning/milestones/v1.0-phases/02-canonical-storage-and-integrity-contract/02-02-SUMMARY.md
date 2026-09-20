---
phase: 02-canonical-storage-and-integrity-contract
plan: "02"
subsystem: storage
tags: [blobstore, canonical-manifest, json, sqlite, in-memory, integrity]
requires:
  - phase: 02-01
    provides: Signed canonical manifest bytes and the raw-record repository seam
provides:
  - Exact-byte local JSON, in-memory, and dedicated SQLite manifest repositories
  - Exact backend-identity selection at the BlobStore composition boundary
  - SQLite sidecar reconciliation after terminal clear recovery
affects: [phase-02-plans, BlobStore, phase-04-backend-composition, phase-06-unified-cache]
actuals:
  tokens: 6079
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Reversible base64 transport for canonical bytes in JSON and in-memory metadata
    - Dedicated SQLite BLOB table for canonical records, separate from legacy cache_entries
    - Exact concrete backend identities at the direct BlobStore composition root
key-files:
  created:
    - tests/test_blob_manifest_backends.py
  modified:
    - src/cacheness/storage/manifest_repository.py
    - src/cacheness/storage/blob_store.py
    - tests/test_clear_recovery.py
key-decisions:
  - "JSON and in-memory repositories preserve canonical bytes through reversible base64 metadata transport, while SQLite uses an isolated BLOB table."
  - "BlobStore admits only exact JsonBackend, SqliteBackend, and InMemoryBackend identities for Phase 2 manifest persistence."
  - "SQLite sidecar rows are removed only after the existing clear-recovery coordinator establishes terminal cache_entries authority."
patterns-established:
  - "Repository methods transport opaque bytes only; they never decode, authenticate, mutate access state, or clean evidence while reading."
  - "Unsupported custom and PostgreSQL metadata identities fail before BlobStore payload staging rather than inheriting unproven guarantees."
requirements-completed: [STOR-01, STOR-08]
coverage:
  - id: D1
    description: Local JSON, memory, and SQLite repositories preserve exact canonical bytes with identical CRUD semantics and persistent reopen coverage.
    requirement: STOR-01
    verification:
      - kind: integration
        ref: tests/test_blob_manifest_backends.py
        status: pass
    human_judgment: false
  - id: D2
    description: BlobStore selects only exact supported local manifest repositories and rejects unsupported custom or PostgreSQL identities before staging payloads.
    requirement: STOR-01
    verification:
      - kind: integration
        ref: tests/test_blob_manifest_backends.py tests/test_clear_recovery.py
        status: pass
    human_judgment: false
  - id: D3
    description: Repository absence remains distinct from typed backend-operation failures with retained causes.
    requirement: STOR-08
    verification:
      - kind: unit
        ref: tests/test_blob_manifest_backends.py
        status: pass
    human_judgment: false
duration: 8 min
completed: 2026-08-30
status: complete
---

# Phase 02 Plan 02: Local Canonical Manifest Repositories Summary

**BlobStore now persists canonical manifest bytes losslessly across exact local JSON, in-memory, and isolated SQLite adapters, with fail-closed backend selection.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-08-30T13:51:42Z
- **Completed:** 2026-08-30T14:00:10Z
- **Tasks:** 2/2
- **Files modified:** 4

## Accomplishments

- Added a local repository contract that round-trips opaque canonical bytes, including Unicode and nested metadata, across JSON, memory, and SQLite.
- Moved SQLite canonical records to a dedicated BLOB table so the legacy fixed-column `cache_entries` schema and rows are not used as a manifest transport.
- Made BlobStore select only exact local repository identities and reject custom or PostgreSQL backends before any payload staging.
- Preserved Phase 1 clear-recovery ownership while removing SQLite sidecar records only after a terminal clear outcome.

## Task Commits

1. **Task 1: Prove exact canonical bytes across local repositories and reopen** - `68dd1d9` (RED), `6b38918` (GREEN)
2. **Task 2: Select exact local repositories at the BlobStore composition boundary** - `238fb28` (RED), `3ea556c` (GREEN)

## Files Created/Modified

- `src/cacheness/storage/manifest_repository.py` - Lossless JSON, in-memory, and dedicated-table SQLite raw canonical record repositories.
- `src/cacheness/storage/blob_store.py` - Exact local repository selection and terminal SQLite sidecar cleanup around the existing clear coordinator.
- `tests/test_blob_manifest_backends.py` - CRUD byte parity, reopen, failure taxonomy, SQLite isolation, and selection contracts.
- `tests/test_clear_recovery.py` - Preserves clear-only rejection for SQLite memory while asserting early manifest rejection for unsupported custom, wrapped, and PostgreSQL identities.

## Decisions Made

- JSON and memory use a reversible base64 metadata transport because it preserves exact raw bytes while reusing their established durable/locked backend behavior.
- SQLite uses `cacheness_manifest_records_v1` with a `BLOB` column, keeping canonical data out of legacy `cache_entries` projections.
- Exact backend classes, rather than capability-shaped instances, define the Phase 2 local manifest contract; general backend composition remains Phase 4 work.

## TDD Gate Compliance

- RED commits: `68dd1d9`, `238fb28`
- GREEN commits: `6b38918`, `3ea556c`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test fixture] Corrected the canonical manifest HMAC fixture key length.**
- **Found during:** Task 1 GREEN
- **Issue:** The new contract fixture supplied a key that did not meet the established exact 32-byte canonical signer requirement.
- **Fix:** Replaced it with a stable 32-byte test-only key.
- **Files modified:** `tests/test_blob_manifest_backends.py`
- **Verification:** `uv run pytest -q -o log_cli=false tests/test_blob_manifest_backends.py -x`
- **Committed in:** `6b38918`

**2. [Rule 1 - Integration] Reconciled SQLite manifest sidecars after terminal clear recovery.**
- **Found during:** Task 2 verification
- **Issue:** The new dedicated SQLite manifest table retained raw records after the Phase 1 coordinator cleared their associated payload and `cache_entries` rows.
- **Fix:** Retained the coordinator unchanged, removed sidecars after a successful clear, and reconciled orphaned sidecars after recovery establishes terminal metadata authority.
- **Files modified:** `src/cacheness/storage/blob_store.py`, `tests/test_clear_recovery.py`
- **Verification:** `uv run pytest -q -o log_cli=false tests/test_blob_manifest_backends.py tests/test_clear_recovery.py -x`
- **Committed in:** `3ea556c`

**3. [Rule 1 - Regression contract] Aligned unsupported-topology coverage with fail-early selection.**
- **Found during:** Task 2 verification
- **Issue:** Existing clear-recovery tests expected custom and PostgreSQL metadata identities to construct before being rejected by `clear()`, contradicting this plan's required pre-staging construction rejection.
- **Fix:** Updated the regression contract to assert typed constructor rejection for custom, wrapped, and PostgreSQL identities while retaining SQLite-memory's separate clear-only refusal.
- **Files modified:** `tests/test_clear_recovery.py`
- **Verification:** Focused clear-recovery suite and targeted backend-selection gate passed.
- **Committed in:** `3ea556c`

---

**Total deviations:** 3 auto-fixed (3 Rule 1 correctness/integration repairs).
**Impact on plan:** All repairs enforce the stated local repository and fail-closed boundaries without adding Phase 4 backend composition or a general lifecycle engine.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 02-03 can build lifecycle behavior atop lossless canonical local repositories. The exact local-identity fence remains intentional; arbitrary injected, PostgreSQL, and S3 composition stay assigned to Phase 4 and later work.

## Self-Check: PASSED

- Confirmed all four implementation and TDD commits exist in git history.
- Confirmed the three local repository implementations, BlobStore composition changes, focused regression coverage, and this summary exist on disk.

---
*Phase: 02-canonical-storage-and-integrity-contract*
*Completed: 2026-08-30*
