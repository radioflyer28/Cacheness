---
phase: 04-metadata-composition-and-topology-contracts
plan: "12"
subsystem: storage-regression-tests
tags: [blob-store, role-registry, s3, sqlite, integrity, unified-cache]
requires:
  - phase: 04-11
    provides: receipt-preserving projection boundaries and bounded catalog cursors
provides:
  - Stable-path RoleRegistry and local S3 factory consumer coverage
  - Current authority-manifest integrity and BlobStore/UnifiedCache regression coverage
  - Explicit SQLite authority initialization and non-mutating unsupported-layout coverage
affects: [phase-04-plan-13, phase-05, phase-06, phase-07]
actuals:
  tokens: 16056
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - Stable regression modules bind current public composition and purpose-built security seams rather than retired registries or metadata adapters.
    - Mocked S3 registration proves factory construction only; it is not topology qualification.
key-files:
  created: []
  modified:
    - tests/test_blob_backend_registry.py
    - tests/test_s3_blob_backend.py
    - tests/test_public_api_contract.py
    - src/cacheness/storage/backends/s3_backend.py
    - tests/test_blob_store_integrity.py
    - tests/test_unified_cache_lifecycle_authority.py
    - tests/test_core.py
    - tests/test_phase3_gap_acceptance.py
    - tests/test_sqlite_metadata_bootstrap_atomicity.py
key-decisions:
  - "RoleRegistry is the sole registry consumer contract; retired process-global registry operations are not recreated in tests or production."
  - "S3 coverage is limited to local factory registration and exact option forwarding until Phase 5 qualifies service/topology behavior."
  - "Stable lifecycle tests preserve security and recovery outcomes through BlobStore and SqliteLifecycleAuthority rather than old metadata backend classes."
patterns-established:
  - "Inspect signed manifest test fixtures with BlobStore._authority_manifest_key(), the current authority-manifest test seam."
  - "Initialize SQLite stores explicitly before shared-worker tests and treat foreign or obsolete roots as non-mutating migration/rebuild evidence."
requirements-completed: [BACK-02, BACK-03, BACK-06, BACK-07]
coverage:
  - id: D1
    description: "Registry consumers retain typed role collision, replacement, validation, option forwarding, and fresh-instance behavior without a global selector."
    requirement: BACK-03
    verification:
      - kind: unit
        ref: "tests/test_blob_backend_registry.py tests/test_s3_blob_backend.py tests/test_public_api_contract.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "Stable integrity and cache-policy regressions use current authority-manifest and BlobStore seams while preserving fail-closed and replacement-safety assertions."
    requirement: BACK-07
    verification:
      - kind: integration
        ref: "tests/test_blob_store_integrity.py tests/test_unified_cache_lifecycle_authority.py tests/test_core.py"
        status: pass
    human_judgment: false
  - id: D3
    description: "SQLite authority initialization, close, schema, and non-mutating foreign/obsolete layout rejection remain executable at stable test paths."
    requirement: BACK-06
    verification:
      - kind: integration
        ref: "tests/test_phase3_gap_acceptance.py tests/test_sqlite_metadata_bootstrap_atomicity.py"
        status: pass
    human_judgment: false
duration: 40m
completed: 2026-09-08
status: complete
---

# Phase 04 Plan 12: Stable Consumer Migration Summary

**RoleRegistry, local S3 factory, integrity, cache-policy, and SQLite authority regressions now run at their stable paths without restoring retired global registries or metadata authorities.**

## Performance

- **Duration:** 40m
- **Started:** 2026-09-08T02:30:00-04:00
- **Completed:** 2026-09-08T03:10:36-04:00
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Rewrote the historical global blob-registry suite as a per-topology `RoleRegistry` contract covering built-ins, typed collisions, replacement, validation, options, and fresh factory identity.
- Limited mocked S3 evidence to local payload-role registration and exact construction-option forwarding, while making the Phase 5 qualification boundary explicit in code guidance and tests.
- Migrated retained integrity, lifecycle, cache, and bootstrap regression assertions to current `BlobStore`, `StoreTopology`, and `SqliteLifecycleAuthority` seams, preserving security, cleanup, initialization, and non-mutating format-rejection evidence.

## Task Commits

1. **Task 1: Rewrite blob and S3 registry consumers in place against RoleRegistry** — `2090608` (`test`)
2. **Task 2: Repair stable-path lifecycle, integrity, initialization, and cache regressions** — `5503f1c` (`test`)

## Files Created/Modified

- `tests/test_blob_backend_registry.py` — Replaces global mutable-registry assertions with meaningful per-instance role registry behavior.
- `tests/test_s3_blob_backend.py` and `src/cacheness/storage/backends/s3_backend.py` — Exercise and document local S3 factory registration without claiming a qualified lifecycle topology.
- `tests/test_public_api_contract.py` — Verifies canonical composition barrels are executable and retired selectors are absent.
- `tests/test_blob_store_integrity.py` and `tests/test_unified_cache_lifecycle_authority.py` — Preserve tamper, fail-closed, event-driven overlap, replacement, invalidation, and guarded-root behavior on current seams.
- `tests/test_core.py`, `tests/test_phase3_gap_acceptance.py`, and `tests/test_sqlite_metadata_bootstrap_atomicity.py` — Exercise public policy statistics plus explicit authority initialization, teardown, schema, and non-mutating layout rejection.

## Decisions Made

- Role/name registration and factory construction are meaningful consumer behavior; global unregister/list/reset behavior was intentionally retired with the former global selector.
- S3 is constructed locally through `RoleRegistry` only. It is neither passed to `StoreTopology` nor treated as a Phase 4 lifecycle participant.
- SQLite initialization regressions pre-initialize shared-worker roots and distinguish safety/recovery evidence from an unsupported universal first-use progress promise, in line with ADR 0001.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Tracking] Preserved the completed-phase count after plan advancement**
- **Found during:** Plan tracking
- **Issue:** The state advancement helper reduced `completed_phases` from the three completed roadmap phases to two while Phase 4 remains in progress.
- **Fix:** Restored the derived count to three.
- **Files modified:** `.planning/STATE.md`

**Total deviations:** 1 auto-fixed tracking correction. No implementation scope expanded.

## Verification

- PASS — `uv run --frozen pytest -q tests/test_blob_backend_registry.py tests/test_s3_blob_backend.py tests/test_public_api_contract.py -o log_cli=false`.
- PASS — `uv run --frozen pytest -q tests/test_blob_store_integrity.py tests/test_unified_cache_lifecycle_authority.py tests/test_core.py tests/test_phase3_gap_acceptance.py tests/test_sqlite_metadata_bootstrap_atomicity.py -o log_cli=false`.
- PASS — scoped `uv run --frozen ruff check` over all nine Plan 04-12 files and `git diff --check`.

## Issues Encountered

- The subagent sandbox cannot create `.git/index.lock`; the root executor reviewed and made the two atomic task commits. No task files were left uncommitted.

## User Setup Required

None — no external service configuration required. S3 service/topology qualification remains Phase 5.

## Next Phase Readiness

Plan 04-13 can audit the executable tree and interpreter matrix against current consumer surfaces. Phase 5 receives deliberately narrow S3 factory evidence, not a fabricated lifecycle qualification claim.

## Self-Check: PASSED

- All nine modified stable-path modules exist and no task module imports retired metadata-authority types or blob registry selectors.
- Task commits `2090608` and `5503f1c` exist in repository history.

---
*Phase: 04-metadata-composition-and-topology-contracts*
*Plan: 12*
*Completed: 2026-09-08*
