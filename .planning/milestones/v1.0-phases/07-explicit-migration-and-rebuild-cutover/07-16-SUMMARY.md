---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "16"
subsystem: storage-migration
tags: [migration, handlers, guarded-handler-io, manifests, recovery, pytest]
requires:
  - phase: 07-explicit-migration-and-rebuild-cutover
    provides: Bounded authority-attributed candidate recovery and canonical maintenance replay
provides:
  - Exact source-and-destination compatibility-edge matching
  - One registered store-local handler transform for changed native payload contracts
  - Signed target manifests from independently re-opened and hashed guarded output
affects: [phase-7-validation, phase-8-qualification, migration-maintenance]
actuals:
  tokens: 9184
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - Exact destination handler contracts determine whether bytes copy or one handler-owned transform runs
    - Migration reopens transformed artifacts through GuardedHandlerIO and hashes those bytes before manifest signing
key-files:
  created: []
  modified:
    - src/cacheness/storage/migration.py
    - src/cacheness/handlers.py
    - tests/test_migration_plan_contract.py
    - tests/test_handler_registration.py
    - tests/test_migration_cutover.py
key-decisions:
  - "A directed compatibility edge matches only exact source and configured destination dimensions; a source-only match is rebuild-only."
  - "Byte copy remains limited to a destination handler that already reads the exact source payload identity; changed identities invoke one registered handler transform."
  - "Only authority-attributed transformed candidates are resumable or abortable; an earlier post-publication orphan remains invisible and unadopted without an exact-reclamation promise."
patterns-established:
  - "Validate declared transform edges at registration by rejecting CacheHandler's inherited rejection stub while allowing intermediate concrete overrides."
  - "Treat transform result size, digest, target format/version, catalog values, and runtime metadata as one signed destination manifest contract."
requirements-completed: [MIGR-04, MIGR-06]
coverage:
  - id: D1
    description: Exact source and destination matching prevents a source-only compatibility edge from authorizing a different target.
    requirement: MIGR-04
    verification:
      - kind: unit
        ref: tests/test_migration_plan_contract.py#test_compatibility_edge_requires_exact_destination_dimensions
        status: pass
      - kind: unit
        ref: tests/test_handler_registration.py#test_registered_handler_rejects_declared_edge_without_concrete_transform
        status: pass
    human_judgment: false
  - id: D2
    description: A changed custom payload contract transforms once through GuardedHandlerIO and preserves only checkpointed transformed candidates for recovery.
    requirement: MIGR-06
    verification:
      - kind: integration
        ref: tests/test_migration_cutover.py#test_migration_executes_one_handler_transform_and_publishes_destination_manifest_identity
        status: pass
      - kind: integration
        ref: tests/test_migration_cutover.py#test_transformed_candidate_checkpointed_metadata_recovers_within_limits_and_uncheckpointed_orphan_remains_invisible
        status: pass
    human_judgment: false
duration: 18min
completed: 2026-09-11
status: complete
---

# Phase 07 Plan 16: Destination-Compatible Handler Transform Summary

**Offline migration now distinguishes exact byte-copy contracts from one directed, handler-owned native transformation and signs the resulting target manifest identity.**

## Performance

- **Duration:** 18 min
- **Completed:** 2026-09-11T17:12:08Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Required compatibility edges to match both source and destination persisted dimensions, with stable rebuild-only classifications for source, destination, and non-executable-transform mismatches.
- Rejects a handler that advertises an edge but inherits `CacheHandler.transform_payload()`'s rejecting base implementation.
- Uses the source registry's exact directed transform once for changed native formats; the existing path-based handler and GuardedHandlerIO boundary remain intact.
- Reopens transformed output through GuardedHandlerIO, independently computes digest and byte size, checks the declared target identity, and signs the complete destination manifest with preserved catalog values and runtime handler metadata.
- Demonstrates that checkpointed transformed candidates resume and abort without another transform, while a pre-checkpoint immutable orphan stays invisible, unattributed, and unadopted.

## Task Commits

1. **Task 1: Require exact destination dimensions before classifying an edge executable** - `6d861f8` (`feat`)
2. **Task 2: Execute one handler-owned transform and publish its destination manifest identity** - `b8ec28c` (`feat`)

## Decisions Made

- Existing destination handlers may retain verified byte copies only when they explicitly read the source identity; a different destination identity requires a directed transform rather than coordinator conversion.
- The transform receives only a live private snapshot and destination guarded-I/O seam. No stream API, global transform registry, participant handle, obstore dependency, journal, lock, queue, or second authority was introduced.
- The accepted ADR 0001 boundary remains explicit: an immutable payload created before the authority checkpoint can be an invisible orphan, never an adopted candidate, and has no Phase 7 exact-cleanup guarantee.

## Deviations from Plan

None - plan executed within the existing single-authority and guarded path-handler boundaries.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 17 can keep runtime handler and catalog metadata out of shareable plans while preserving their authenticated destination-manifest and authority-evidence roles. Phase 8 retains live PostgreSQL/S3, Windows, supported-Python, and performance qualification.

## Self-Check: PASSED

- Confirmed all five plan-owned production/test artifacts and this summary exist.
- Confirmed task commits `6d861f8` and `b8ec28c` exist in repository history.
- Passed 41 focused migration-plan, handler-registration, cutover, and rebuild tests, plus the four exact Plan 07-16 selectors.
- Passed scoped Ruff and whitespace checks for all modified source and test files.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-11*
