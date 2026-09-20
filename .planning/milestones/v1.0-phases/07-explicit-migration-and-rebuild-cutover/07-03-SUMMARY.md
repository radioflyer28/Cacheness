---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "03"
subsystem: storage-migration
tags: [migration, compatibility, offline-maintenance, canonical-json, sqlite]
requires:
  - phase: 07-02
    provides: "Accepted D-02 current-plus-immediately-previous release-window contract"
provides:
  - "Bounded release-window compatibility matrix with exact directed per-dimension edges"
  - "Canonical immutable migration plans and report rendering from one validated model"
  - "Read-only current, historical, and corrupt store inspection classifications"
affects: [07-04, 07-08, migration, rebuild, lifecycle-authority]
actuals:
  tokens: 17149
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - "Compatibility is evaluated per persisted contract dimension through exact directed edges."
    - "Canonical machine plans are the sole input to bounded human-readable migration reports."
key-files:
  created:
    - tests/test_migration_plan_contract.py
    - tests/test_migration_inspection.py
  modified:
    - src/cacheness/storage/migration.py
    - src/cacheness/storage/migration_evidence.py
    - tests/test_stored_compatibility.py
key-decisions:
  - "The current baseline uses the abstract current release label and no historical or future persisted migration edge."
  - "Historical evidence receives a rebuild-only plan, while corrupt authority evidence is refused without an override."
patterns-established:
  - "Keep authority identity, totals, classifications, reasons, actions, and state in MigrationPlan; render reports only from that object."
  - "Use bounded canonical JSON parsing with duplicate-key and non-canonical-number rejection for maintenance artifacts."
requirements-completed: [MIGR-03, MIGR-04, MIGR-06]
coverage:
  - id: D1
    description: "Release support is bounded to current plus immediately previous releases, with exact directed edges for independently versioned contracts."
    requirement: MIGR-04
    verification:
      - kind: unit
        ref: "tests/test_migration_plan_contract.py#test_matrix_requires_one_exact_edge_per_changed_dimension"
        status: pass
    human_judgment: false
  - id: D2
    description: "Canonical migration plans round-trip byte-for-byte and reports render the same identities, totals, classifications, reasons, actions, and state."
    requirement: MIGR-03
    verification:
      - kind: unit
        ref: "tests/test_migration_plan_contract.py#test_canonical_plan_round_trips_and_human_report_uses_the_same_model"
        status: pass
    human_judgment: false
  - id: D3
    description: "Current SQLite, historical, and corrupt layouts are inspected without mutation; unsupported migration has no force path."
    requirement: MIGR-06
    verification:
      - kind: integration
        ref: "tests/test_migration_inspection.py"
        status: pass
      - kind: integration
        ref: "tests/test_stored_compatibility.py#test_initialized_current_memory_and_sqlite_stores_repeat_initialize_by_validation"
        status: pass
    human_judgment: false
duration: 16m 9s
completed: 2026-09-09
status: complete
---

# Phase 07 Plan 03: Compatibility and Inspection Contract Summary

**Bounded release compatibility and a canonical, read-only migration-plan model now classify current, historical, and corrupt stores without inventing a persisted migration target.**

## Performance

- **Duration:** 16m 9s
- **Started:** 2026-09-09T23:27:02Z
- **Completed:** 2026-09-09T23:43:11Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added immutable release-window, compatibility identity, exact directed-edge, and full per-dimension outcome models; the default baseline declares no historical development-layout edge.
- Added canonical `MigrationPlan` encoding, strict bounded parsing, exact aggregates, stable reason codes, and a human report rendered solely from the validated plan.
- Added read-only inspection for initialized SQLite stores plus rebuild-only historical and refused corrupt-authority path classifications, with fixture byte snapshots proving no implicit migration occurs.

## Task Commits

1. **Task 1: Define the release window and independent compatibility matrix** - `cfa2c62` (test), `f2e1f9f` (feat)
2. **Task 2: Produce deterministic non-mutating inspection plans and reports** - `a74afc4` (test), `be3275b` (feat)

## Files Created/Modified

- `src/cacheness/storage/migration.py` - Bounded compatibility data, canonical plan model, report rendering, and read-only inspection.
- `src/cacheness/storage/migration_evidence.py` - Shared bounded canonical JSON decoder for authenticated evidence and plans.
- `tests/test_migration_plan_contract.py` - Compatibility-edge and canonical plan/report contracts.
- `tests/test_migration_inspection.py` - Current, historical, and corrupt inspection non-mutation coverage.
- `tests/test_stored_compatibility.py` - Current SQLite authority inspection regression coverage.

## Decisions Made

- D-02 is implemented as an explicit `ReleaseWindow`; no direct edge exists for older releases, development schemas, or a fabricated next format.
- Store layout, authority, manifest, handler payload, and catalog contracts are classified independently; a changed dimension requires its own exact directed edge.
- Historical paths are rebuild-only and corrupt authority evidence is refused. Neither classification offers a force override or changes store state.

## Verification

- `uv run --frozen pytest -q tests/test_migration_plan_contract.py tests/test_migration_inspection.py tests/test_stored_compatibility.py tests/test_migration_cutover.py -x -o log_cli=false` — passed (22 tests).
- `uv run --frozen ruff check src/cacheness/storage/migration.py src/cacheness/storage/migration_evidence.py tests/test_migration_plan_contract.py tests/test_migration_inspection.py tests/test_stored_compatibility.py tests/test_migration_cutover.py --output-format concise` — passed.

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None.

## Issues Encountered

The sandbox could not open uv's shared cache during one final verification attempt; the identical focused commands passed when run with the required cache access.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 07 can now add bounded topology-specific inventory and later physical migration/rebuild actions against a stable, fail-closed plan contract. This plan does not publish an authority schema change, add legacy readers, or claim live PostgreSQL/S3 qualification.

## Self-Check: PASSED

- All five implementation/test artifacts and this summary exist on disk.
- TDD commits `cfa2c62`, `f2e1f9f`, `a74afc4`, and `be3275b` exist in git history.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-09*
