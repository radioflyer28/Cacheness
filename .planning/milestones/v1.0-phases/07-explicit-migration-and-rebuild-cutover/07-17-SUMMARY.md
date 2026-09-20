---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "17"
subsystem: storage-migration
tags: [migration, confidentiality, integrity, manifests, recovery, pytest]
requires:
  - phase: 07-explicit-migration-and-rebuild-cutover
    provides: Bounded authority-attributed candidate recovery and destination handler transforms
provides:
  - Digest-only, shareable migration-plan records bound to authenticated source catalog and manifest state
  - Pre-mutation source-state authentication and fail-closed drift rejection for migration and rebuild flows
  - Operator documentation separating public plans from protected authority-owned candidate evidence
affects: [phase-7-validation, phase-8-qualification, offline-maintenance]
actuals:
  tokens: 8624
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Shareable records use a fixed typed allowlist and canonical source-state digests instead of raw authenticated data
    - Execution treats re-read authenticated source manifests as the only source of catalog and handler metadata
key-files:
  created: []
  modified:
    - src/cacheness/storage/migration.py
    - tests/test_migration_plan_contract.py
    - tests/test_migration_run_evidence.py
    - docs/STORAGE_MIGRATION.md
key-decisions:
  - "Canonical plans bind source catalog and manifest state with digests but never serialize their raw values."
  - "Execution re-reads and authenticates source state before candidate mutation; source drift fails as source_state_drift."
  - "Only existing bounded authority-owned candidate evidence retains exact target descriptors; a pre-checkpoint payload orphan remains invisible and unattributed."
patterns-established:
  - "Use _canonical_source_binding() consistently at plan construction and live execution rather than comparing caller-supplied metadata."
  - "Carry unknown authenticated catalog attributes from the live manifest into destination projection without placing them in plans, reports, or logs."
requirements-completed: [MIGR-03, MIGR-05]
coverage:
  - id: D1
    description: Digest-only canonical plan records change with authenticated source state while excluding raw catalog and manifest values.
    requirement: MIGR-03
    verification:
      - kind: unit
        ref: tests/test_migration_plan_contract.py#test_machine_plan_digest_binds_sensitive_catalog_and_manifest_without_serializing_them
        status: pass
    human_judgment: false
  - id: D2
    description: Shared-plan execution authenticates fresh source state, preserves unknown authenticated attributes, and rejects catalog, manifest, or forged-source drift before candidate writes.
    requirement: MIGR-05
    verification:
      - kind: integration
        ref: tests/test_migration_run_evidence.py#test_execution_rereads_authenticates_and_rejects_plan_bound_manifest_or_catalog_drift
        status: pass
    human_judgment: false
duration: 10min
completed: 2026-09-11
status: complete
---

# Phase 07 Plan 17: Confidential Migration Plan Bindings Summary

**Offline migration plans now authorize exact authenticated source state through digests, while execution obtains live catalog and manifest data only from the source authority.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-09-11T17:22:02Z
- **Completed:** 2026-09-11T17:31:43Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Replaced raw catalog and base64 manifest plan fields with a strict identity/disposition/digest allowlist, including independent source catalog and manifest bindings.
- Re-read and authenticated the complete source inventory before migration candidates, rebuild candidates, rebuild verification, and rebuild acceptance; drift fails closed with `source_state_drift` before candidate mutation.
- Preserved unknown authenticated catalog attributes in the live destination projection while documenting the boundary between shareable plans, main run evidence, and existing protected authority-owned candidate descriptors.
- Kept the ADR 0001 limit explicit: a payload published before its authority checkpoint may be an invisible, unattributed orphan and has no guaranteed exact reclamation path.

## Task Commits

1. **Task 1: Bind sensitive source state without serializing it in the machine plan** - `b5fdca4` (`test`), `b62862d` (`feat`)
2. **Task 2: Re-read and authenticate digest-bound source state at execution** - `f48d893` (`test`), `4b528e2` (`feat`)

## Files Created/Modified

- `src/cacheness/storage/migration.py` - Encodes only source bindings in plan records and authenticates fresh source state for maintenance execution.
- `tests/test_migration_plan_contract.py` - Proves catalog/manifest secrecy and digest sensitivity in canonical plan bytes.
- `tests/test_migration_run_evidence.py` - Proves live carry-through, catalog drift rejection, and unauthenticated manifest rejection before candidate writes.
- `docs/STORAGE_MIGRATION.md` - Documents confidentiality, protected evidence, run bounds, and the accepted invisible-orphan limit.

## Decisions Made

- A decoded shareable plan deliberately contains no source catalog values or source manifest bytes. Its opaque revalidation entry can never drive payload handling.
- Existing authority-owned candidate evidence remains the sole bounded location for an exact target manifest needed by resume or abort; no journal, adoption mechanism, or lifecycle authority was added.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- An initial broader test command named a non-existent migration test file. The corrected explicit Phase 7 migration/rebuild suite passed; no test or production code was changed to accommodate it.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

The canonical plan is now safe to share and still fail-closed at execution. Phase 18 can bind fixed verification nodes without reopening source-data disclosure, lifecycle coordination, or the accepted pre-checkpoint orphan boundary.

## Self-Check: PASSED

- Confirmed all four plan-owned production, test, and documentation artifacts exist.
- Confirmed task commits `b5fdca4`, `b62862d`, `f48d893`, and `4b528e2` exist.
- Passed 58 focused migration/rebuild tests and scoped Ruff for the modified source and test files.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-11*
