---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "01"
subsystem: storage-migration
tags: [blobstore, offline-migration, lifecycle-authority, sqlite, memory]
requires:
  - phase: 06-unifiedcache-policy-composition
    provides: "BlobStore-owned lifecycle with UnifiedCache as a narrow policy facade"
provides:
  - "Explicit memory/memory inspect-plan-stage-verify-activate migration tracer"
  - "Authenticated maintenance evidence and non-authoritative candidate receipts"
  - "Regression coverage proving ordinary entry points stay validation-only"
affects: [migration, rebuild, blobstore, unified-cache, topology-qualification]
actuals:
  tokens: 15394
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - "Offline service requests authority cutover; only the authority activates a verified candidate"
    - "Ordinary compatibility tests snapshot complete fixture trees before each entry point"
key-files:
  created:
    - src/cacheness/storage/migration.py
    - src/cacheness/storage/migration_authority.py
    - src/cacheness/storage/migration_evidence.py
  modified:
    - src/cacheness/storage/memory_lifecycle_authority.py
    - tests/test_migration_cutover.py
    - tests/test_stored_compatibility.py
key-decisions:
  - "The tracer uses a test-only current-to-current compatibility edge, so it proves the generic path without manufacturing a production format version."
  - "Maintenance evidence corroborates one explicit offline run, while authority activation alone selects visible store state."
  - "UnifiedCache preserves an unsupported-store failure as a typed lookup cause rather than adopting or changing the store."
patterns-established:
  - "Use explicit work directory, run ID, stopped-worker acknowledgement, and shared signing-provider identity for offline maintenance."
  - "Compare complete disposable fixture trees after each ordinary operation to guard fail-closed validation behavior."
requirements-completed: [MIGR-03, MIGR-04, MIGR-05]
coverage:
  - id: D1
    description: "Memory topology completes explicit inspect, plan, stage, verify, and separately requested authority activation."
    requirement: MIGR-03
    verification:
      - kind: integration
        ref: "tests/test_migration_cutover.py#test_memory_tracer_requires_explicit_whole_store_activation"
        status: pass
    human_judgment: false
  - id: D2
    description: "Unsupported and current stores retain validation-only ordinary entry behavior."
    requirement: MIGR-04
    verification:
      - kind: integration
        ref: "tests/test_stored_compatibility.py#test_ordinary_entry_points_never_adopt_or_modify_unsupported_roots"
        status: pass
      - kind: integration
        ref: "tests/test_stored_compatibility.py#test_initialized_current_memory_and_sqlite_stores_repeat_initialize_by_validation"
        status: pass
    human_judgment: false
  - id: D3
    description: "Candidate and evidence artifacts cannot independently authorize cutover."
    requirement: MIGR-05
    verification:
      - kind: integration
        ref: "tests/test_migration_cutover.py#test_candidate_and_evidence_never_authorize_activation"
        status: pass
    human_judgment: false
duration: 22m 28s
completed: 2026-09-09
status: complete
---

# Phase 07 Plan 01: Offline Migration Cutover Tracer Summary

**An explicit, authenticated memory-store migration flow now stages and verifies a complete candidate before the single lifecycle authority activates it, while ordinary store entry points remain validation-only.**

## Performance

- **Duration:** 22m 28s
- **Started:** 2026-09-09T22:28:39Z
- **Completed:** 2026-09-09T22:51:07Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Added an offline memory/memory tracer with explicit inspect, plan, stage, verify, and activate calls, bound to an operator run ID, work directory, and stopped-worker acknowledgement.
- Kept candidate artifacts and signed maintenance evidence non-authoritative; the in-memory lifecycle authority alone performs whole-store activation after receipt validation.
- Added a disposable-tree compatibility matrix covering historical, future, retired scheduler, corrupt authority, and foreign-root evidence across ordinary store and cache entry points.
- Proved repeat initialization preserves current memory entries and current SQLite bytes without an implicit format change.

## Task Commits

1. **Task 1: Execute one memory inspect-plan-stage-verify-activate run** - `6cdaff9` (test), `85224cb` (feat)
2. **Task 2: Preserve validation-only ordinary store entry points** - `e60636f` (test)

## Files Created/Modified

- `src/cacheness/storage/migration.py` - Explicit offline migration service and immutable plan/result vocabulary.
- `src/cacheness/storage/migration_authority.py` - Narrow authority protocol and bounded identity/candidate receipts.
- `src/cacheness/storage/migration_evidence.py` - Canonical authenticated maintenance evidence envelope.
- `src/cacheness/storage/memory_lifecycle_authority.py` - Same-process whole-store activation primitive for the tracer.
- `tests/test_migration_cutover.py` - End-to-end activation and non-authoritative evidence coverage.
- `tests/test_stored_compatibility.py` - Validation-only ordinary-entry regression matrix.

## Decisions Made

- Used a test-only current-to-current compatibility edge to exercise the real migration seam without creating a manufactured persisted version.
- Kept candidate visibility exclusively behind `MigrationAuthority.activate_verified_candidate`; evidence and candidate presence remain corroboration only.
- Treated `UnifiedCache.lookup`'s retained typed migration cause as the correct policy-layer behavior for unsupported roots.

## Verification

- `uv run --frozen pytest -q tests/test_stored_compatibility.py tests/test_migration_cutover.py -x -o log_cli=false` — passed (14 tests).
- `uv run --frozen ruff check tests/test_stored_compatibility.py` — passed.

## Deviations from Plan

None - plan executed as written. Existing ordinary lifecycle behavior already met the new regression contract, so Task 2 required no production change.

## Known Stubs

None. The empty `plan_digest` in inspected evidence is a deliberate pre-plan state; subsequent actions require the generated digest and verified evidence.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

The explicit memory tracer and validation-only boundary are ready for the planned compatibility-window, inventory, recovery, and rebuild expansions. This plan makes no live PostgreSQL, S3, performance, or native Windows qualification claim.

## Self-Check: PASSED

- All six plan files and this summary exist on disk.
- Task commits `6cdaff9`, `85224cb`, and `e60636f` exist in git history.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-09*
