---
phase: 04-metadata-composition-and-topology-contracts
plan: "06"
subsystem: testing
tags: [pytest, blobstore, storetopology, blobreceipt, format-2, security]
requires:
  - phase: 04-05
    provides: Derived projection outcomes on frozen BlobReceipt values
provides:
  - Direct BlobStore lifecycle and recovery regressions using explicit StoreTopology roles
  - Format-2 signed descriptor, BlobReceipt, integrity, and containment test coverage
  - Topology-qualified concurrency and portable Windows lifecycle evidence
affects: [04-07, 04-08, blobstore-test-consumers]
actuals:
  tokens: 32974
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - Direct stores use StoreTopology with filesystem payload and SQLite authority roles
    - Current assertions inspect signed BlobManifest descriptors and BlobReceipt values
key-files:
  created: []
  modified:
    - tests/test_blob_manifest.py
    - tests/test_blob_store_atomic_lifecycle.py
    - tests/test_blob_store_integrity.py
    - tests/test_filesystem_containment.py
    - tests/test_phase3_windows_contract.py
key-decisions:
  - "Direct BlobStore regression fixtures construct StoreTopology explicitly instead of selecting a backend."
  - "Current lifecycle assertions use signed format-2 BlobManifest descriptors and BlobReceipt, not BlobEntryInfo or repository shapes."
patterns-established:
  - "Containment and integrity checks run through the selected lifecycle authority without a compatibility adapter."
requirements-completed: [BACK-02, BACK-03, BACK-06, BACK-07]
coverage:
  - id: D1
    description: Direct BlobStore lifecycle, recovery, and receipt regressions use explicit composed topology.
    requirement: BACK-02
    verification:
      - kind: unit
        ref: "uv run --frozen pytest -q tests/test_blob_manifest.py tests/test_blob_manifest_backends.py tests/test_blob_store_atomic_lifecycle.py tests/test_blob_store_close_contract.py tests/test_blob_store_read_contract.py tests/test_blob_store_reconciliation.py tests/test_clear_recovery.py tests/test_manifest_repository_cas.py -x"
        status: pass
    human_judgment: false
  - id: D2
    description: Format-2 descriptor integrity, path containment, and legacy rejection remain fail-closed.
    requirement: BACK-03
    verification:
      - kind: unit
        ref: "tests/test_blob_store_integrity.py and tests/test_filesystem_containment.py"
        status: pass
    human_judgment: false
  - id: D3
    description: Contention and portable Windows authority assertions preserve topology-specific guarantees.
    requirement: BACK-06
    verification:
      - kind: unit
        ref: "tests/test_blob_store_concurrency.py and tests/test_phase3_windows_contract.py"
        status: pass
    human_judgment: false
duration: 26m 15s
completed: 2026-09-08
status: complete
---

# Phase 04 Plan 06: Direct BlobStore Regression Migration Summary

**Direct BlobStore lifecycle, security, recovery, and platform regressions now exercise explicit StoreTopology roles, frozen BlobReceipt values, and signed format-2 descriptors.**

## Performance

- **Duration:** 26m 15s
- **Started:** 2026-09-08T02:54:24Z
- **Completed:** 2026-09-08T03:20:39Z
- **Tasks:** 2/2
- **Files modified:** 13

## Accomplishments

- Rewrote the manifest, lifecycle, read, close, reconciliation, and CAS regression group around `StoreTopology`, `BlobReceipt`, and canonical signed `BlobManifest` data.
- Preserved rollback, cleanup debt, same-key contention, descriptor authentication, legacy rejection, traversal, and symlink containment evidence without test-only adapters.
- Kept portable Windows evidence honest: unprovisioned authority writes fail before any payload generation; live Windows validation remains explicitly unverified on this host.

## Task Commits

1. **Task 1: Rewrite retained manifest, atomic lifecycle, close, read, and recovery regressions** - `3d34c49` (`test`)
2. **Task 2: Rewrite retained concurrency, integrity, containment, and platform regressions** - `6bfd142` (`test`)

## Files Created/Modified

- `tests/test_blob_manifest.py`, `tests/test_blob_manifest_backends.py` - Current descriptor authenticity and topology construction contracts.
- `tests/test_blob_store_atomic_lifecycle.py`, `tests/test_blob_store_close_contract.py`, `tests/test_blob_store_read_contract.py`, `tests/test_blob_store_reconciliation.py`, `tests/test_clear_recovery.py`, `tests/test_manifest_repository_cas.py` - Lifecycle, read, close, recovery, and exact-CAS migrations.
- `tests/test_blob_store_concurrency.py`, `tests/test_blob_store_integrity.py` - ADR-qualified contention and format-2 integrity coverage.
- `tests/test_blob_store_legacy_contract.py`, `tests/test_filesystem_containment.py`, `tests/test_phase3_windows_contract.py` - Read-only unsupported-layout, containment, and portable Windows regressions.

## Decisions Made

- Direct stores receive one qualified `StoreTopology` with a filesystem payload role and SQLite lifecycle authority; no test selects a metadata backend through `BlobStore`.
- Current tests inspect immutable `BlobReceipt` and canonical signed descriptors rather than obsolete `BlobEntryInfo`, list-filter, repository, or V1 shape assumptions.
- Workspace-local SQLite WAL/SHM sidecars are excluded only when copying normative legacy fixture evidence, so untracked journals cannot alter fixture identity checks.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Existing workspace-local SQLite WAL/SHM files made the legacy fixture copy non-normative; the test fixture copier now excludes only those transient sidecars while keeping source and copied-tree non-mutation checks.
- Retired direct `BlobStore.backend` and stale high-level compatibility calls were removed from mixed containment/platform tests; equivalent direct lifecycle, containment, and no-payload-materialization assertions remain executable through `StoreTopology`.

## Known Stubs

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 04-07 can migrate the remaining mixed UnifiedCache and public-consumer suites while this direct-store regression group stays current-contract-only.

## Self-Check: PASSED

- All 13 declared test files and this summary exist.
- Task commits `3d34c49` and `6bfd142` exist in the repository history.
