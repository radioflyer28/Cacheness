---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "10"
subsystem: storage-migration
tags: [offline-migration, public-api, documentation, api-coverage, rebuild]
requires:
  - phase: 07-07
    provides: "Deterministic PostgreSQL/S3 offline adapter contracts without live qualification"
  - phase: 07-09
    provides: "Explicit handler-backed rebuild and exact-confirmation workflow"
provides:
  - "One public Python maintenance surface for explicit migration and rebuild"
  - "Stopped-worker operator runbook with activated_offline, rollback/finalize, and separate purge boundaries"
  - "Detector-backed declaration that Phase 7 adds no external API integration"
affects: [phase-08-qualification, storage-migration, blobstore, unified-cache]
actuals:
  tokens: 9278
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - "Export one deliberate storage barrel for offline maintenance models, receipts, and typed errors."
    - "Record API-coverage detector output verbatim and justify a no-integration declaration without fabricating a capability matrix."
key-files:
  created:
    - docs/STORAGE_MIGRATION.md
    - tests/test_migration_public_contract.py
  modified:
    - src/cacheness/storage/__init__.py
    - docs/BACKEND_SELECTION.md
    - docs/STORAGE_INITIALIZATION.md
    - .planning/phases/07-explicit-migration-and-rebuild-cutover/07-COVERAGE.md
key-decisions:
  - "Offline migration and rebuild are published only as a Python library surface; no CLI, global service, or ordinary-open migration switch exists."
  - "The coverage artifact retains the real detector's public-API false positive and uses a reasoned no-external-integration declaration instead of a fabricated matrix."
  - "Initialization documentation names schema 8 and routes unsupported layouts to the explicit runbook rather than obsolete Phase 7 future-work wording."
patterns-established:
  - "Public maintenance examples require an explicit work directory, run ID, stopped-worker acknowledgement, evidence path, and destructive-action confirmation."
  - "Remote adapter tests remain deterministic evidence only; Phase 8 owns live PostgreSQL/AWS S3, Windows, and performance qualification."
requirements-completed: [MIGR-03, MIGR-04, MIGR-05, MIGR-06]
coverage:
  - id: D1
    description: "The one public Python maintenance surface exports safe models and typed errors, requires explicit offline inputs, and documents the full stopped-worker workflow."
    requirement: MIGR-04
    verification:
      - kind: integration
        ref: "tests/test_migration_public_contract.py::test_documented_public_workflow_uses_one_model_and_offline_fencing"
        status: pass
      - kind: integration
        ref: "uv run --frozen pytest -q tests/test_migration_public_contract.py tests/test_stored_compatibility.py -x -o log_cli=false"
        status: pass
    human_judgment: false
  - id: D2
    description: "The API-coverage declaration records the active detector result and explicitly preserves the Phase 8 live-service qualification boundary."
    requirement: MIGR-05
    verification:
      - kind: unit
        ref: "tests/test_migration_public_contract.py::test_external_api_coverage_declaration_is_detector_backed"
        status: pass
    human_judgment: false
duration: 12m
completed: 2026-09-10
status: complete
---

# Phase 07 Plan 10: Public Migration Contract Summary

**Cacheness now exposes one Python-only offline migration and rebuild surface, with an exact stopped-worker runbook and a detector-backed statement that no new external service integration was added.**

## Performance

- **Duration:** 12m
- **Started:** 2026-09-10T04:20:15Z
- **Completed:** 2026-09-10T04:31:47Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Re-exported `OfflineMigrationService`, immutable planning/evidence/authority result models, and typed migration errors through the deliberate `cacheness.storage` barrel without exposing backend tables, signing material, or a second lifecycle facade.
- Added the canonical migration runbook for inspect/plan/stage/verify/activate, activated-offline rollback or finalize, separately confirmed purge, exact-run resume, include-all rebuild, handler-owned transformation, derived projections, diagnostics, and topology-specific non-claims.
- Replaced legacy automatic migration wording, corrected stale initialization guidance, and added executable public-contract checks that forbid a migration switch on ordinary constructors or any CLI entry point.
- Recorded the active GSD API-coverage detector result verbatim and proved that the false-positive public-API signal does not create a PostgreSQL/S3 support claim; Phase 8 retains live-service, Windows, and performance qualification.

## Task Commits

1. **Task 1: Publish the single maintenance API and operator runbook** - `3b913fd` (test), `c6f93d2` (feat)
2. **Task 2: Declare deterministic external-API coverage scope** - `868f443` (docs)

## Files Created/Modified

- `src/cacheness/storage/__init__.py` - Public offline maintenance service, models, receipts, and typed-error exports.
- `src/cacheness/storage/migration.py` - Corrected the public service description to reflect declared migration edges and rebuild support.
- `docs/STORAGE_MIGRATION.md` - Canonical operator workflow and topology/non-qualification boundaries.
- `docs/BACKEND_SELECTION.md` and `docs/STORAGE_INITIALIZATION.md` - Remove automatic-migration claims and link unsupported layouts to the runbook.
- `tests/test_migration_public_contract.py` - Public import, signature, workflow, documentation, no-CLI, and detector-backed declaration coverage.
- `07-COVERAGE.md` - Exact detector output and reasoned no-new-integration declaration.

## Decisions Made

- Kept the operator entry point as a Python library API only. Any later CLI can only render the same models and must not own run discovery, evidence, or lifecycle state.
- Stored the detector's `detected: true` result exactly because the signal comes from the Phase 7 public-API wording; a capability matrix would falsely imply a new external service integration.
- Kept `BlobStore` as lifecycle authority, `UnifiedCache` as cache policy, and `SqlCache` separate throughout the documentation and tests.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected stale initialization migration guidance**
- **Found during:** Task 1
- **Issue:** `docs/STORAGE_INITIALIZATION.md` still named SQLite `user_version = 7` and told users to wait for future Phase 7 tooling, conflicting with the schema-8 baseline and newly published runbook.
- **Fix:** Updated the version reference and routed unsupported layouts to `STORAGE_MIGRATION.md`, explicitly preserving the no-CLI and no-implicit-constructor boundary.
- **Files modified:** `docs/STORAGE_INITIALIZATION.md`, `tests/test_migration_public_contract.py`
- **Verification:** Public workflow, stored-compatibility, and Ruff checks passed.
- **Committed in:** `c6f93d2`

---

**Total deviations:** 1 auto-fixed (Rule 1 bug).
**Impact on plan:** The correction removes contradictory public guidance without changing lifecycle architecture or adding an adapter, coordinator, compatibility path, or qualification claim.

## Known Stubs

None. Existing pre-plan empty `plan_digest` values only represent the deliberate inspected-evidence state and are not rendered placeholders.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration is required for this documentation and deterministic-contract plan.

## Next Phase Readiness

Phase 7 has one documented, tested public maintenance contract. Phase 8 can now qualify the explicitly deferred live PostgreSQL/AWS S3, Windows, and performance boundaries without interpreting deterministic adapter tests as release evidence.

## Self-Check: PASSED

- Confirmed all seven task artifacts and commits `3b913fd`, `c6f93d2`, and `868f443` exist.
