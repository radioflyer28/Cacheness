---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "20"
subsystem: storage-migration
tags: [migration, handlers, s3, cleanup-debt, lifecycle-authority, pytest]
requires:
  - phase: 07-explicit-migration-and-rebuild-cutover
    provides: "Authority-attributed candidate evidence, handler-owned transforms, and explicit offline recovery"
provides:
  - "Destination-owned payload identities for exact migration transforms"
  - "Typed S3 abort debt checkpointing and explicit attributed retry settlement"
  - "Truthful per-call abort deletion accounting with fail-closed ownership checks"
affects: [phase-7-migration-cutover, phase-7-rebuild-recovery, phase-8-qualification]
actuals:
  tokens: 5412
  tasks: 2
  commits: 5
tech-stack:
  added: []
  patterns:
    - "Source readability establishes only readability; the destination handler's declared format/version is the immutable migration target."
    - "Abort retries only authority-attributed cleanup debt and classifies narrow participant operational failures without using listings."
key-files:
  created: []
  modified:
    - src/cacheness/storage/migration.py
    - tests/test_migration_cutover.py
    - tests/test_migration_remote_contract.py
key-decisions:
  - "A matching numeric version never makes different payload-format names identity-compatible; the destination handler declares both target dimensions."
  - "Only OSError and CacheBlobBackendError are retryable abort cleanup outcomes; manifest, ownership, and integrity disagreement remain fail closed."
  - "AbortReceipt.deleted_entries records effects deleted or proven absent in the current call, not candidates merely attempted."
patterns-established:
  - "Moto S3 participant tests use a deterministic authority fake solely to exercise payload mechanics and explicitly do not qualify live PostgreSQL/Amazon-S3."
  - "Current-to-current migration tracer fixtures use a stable test-only native contract rather than a legacy-readable object container as an identity target."
requirements-completed: [MIGR-04, MIGR-05]
coverage:
  - id: D1
    description: "A same-version native-format change resolves exactly one handler-owned transform and signs the destination format identity."
    requirement: MIGR-04
    verification:
      - kind: integration
        ref: tests/test_migration_cutover.py#test_same_version_different_format_uses_exact_directed_transform_and_destination_manifest
        status: pass
      - kind: unit
        ref: tests/test_migration_plan_contract.py#test_compatibility_edge_requires_exact_destination_dimensions
        status: pass
    human_judgment: false
  - id: D2
    description: "Typed S3 snapshot and delete interruptions become exact attributed cleanup debt, settle through explicit retry, and never use object listings."
    requirement: MIGR-05
    verification:
      - kind: integration
        ref: tests/test_migration_remote_contract.py#test_s3_abort_typed_operational_failures_checkpoint_exact_debt_and_retry
        status: pass
    human_judgment: false
  - id: D3
    description: "Partial abort receipts count only confirmed cleanup effects while ownership disagreement stays a typed fail-closed error."
    requirement: MIGR-05
    verification:
      - kind: unit
        ref: tests/test_migration_cutover.py#test_partial_abort_receipt_counts_only_deleted_or_proven_absent_candidates
        status: pass
      - kind: unit
        ref: tests/test_migration_cutover.py#test_abort_integrity_and_ownership_conflicts_remain_fail_closed
        status: pass
    human_judgment: false
duration: 12min
completed: 2026-09-11
status: complete
---

# Phase 07 Plan 20: Destination Identity and Abort Debt Summary

**Offline migration now treats the destination handler's full native contract as authoritative and turns typed S3 cleanup interruption into exact, explicit retry debt without adding coordination mechanisms.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-09-11T23:09:19Z
- **Completed:** 2026-09-11T23:21:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Prevented equal payload-version integers from collapsing different native format names into an identity byte copy; a declared directed handler transform now runs once and produces an authenticated destination manifest.
- Normalized only typed S3 participant failures during abort into existing authority-attributed `STAGING` cleanup debt, then proved a later explicit retry reaches `ABORTED` without listing or candidate adoption.
- Made partial abort receipts report only deletion or absence proofs established in the current call, while malformed or mismatched candidate evidence fails closed.

## Task Commits

1. **Task 1: Prove a same-version format change reaches one exact transform** - `0e69875` (`test`), `681f363` (`fix`)
2. **Task 2: Checkpoint typed S3 abort debt and report only completed deletions** - `873dea5` (`test`), `48318aa` (`test`), `17a3f16` (`fix`)

## Files Created/Modified

- `src/cacheness/storage/migration.py` - derives target identity only from the destination handler, preserves narrow typed abort debt, and accurately counts confirmed cleanup outcomes.
- `tests/test_migration_cutover.py` - covers same-version transform identity, truthful partial receipts, and fail-closed candidate ownership; current-to-current tracers use a stable test-native contract.
- `tests/test_migration_remote_contract.py` - exercises Moto-backed S3 snapshot/delete failure and retry settlement through the existing participant seam.

## Decisions Made

- Source contract support remains a read-eligibility question; it cannot redefine a configured destination contract.
- S3 object state remains an external effect only: authority evidence selects candidates and cleanup debt, while remote failure never becomes inferred absence.
- The accepted pre-checkpoint orphan boundary is unchanged: an immutable payload without authority attribution remains invisible, unadopted, and outside exact-cleanup guarantees.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test contract] Current-to-current lifecycle fixtures used a readable legacy object format as their implicit destination identity**

- **Found during:** Task 2
- **Issue:** After target selection correctly stopped substituting source identities, generic object fixtures described `pickle@1` source bytes while their handler declared `compressed_pickle@1` as the destination, making a deliberately current-to-current test edge rebuild-only.
- **Fix:** Registered a test-only stable native dictionary handler for memory/SQLite migration tracers; production handler behavior and migration authority semantics remain unchanged.
- **Files modified:** `tests/test_migration_cutover.py`
- **Verification:** `uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_migration_cutover.py tests/test_migration_remote_contract.py -x -o log_cli=false -o addopts=` (25 passed)
- **Committed in:** `873dea5`, `48318aa`

---

**Total deviations:** 1 auto-fixed (Rule 1 test contract).
**Impact on plan:** The correction keeps test fixtures aligned with the explicit destination-contract rule and introduces no production authority, backend, or coordination behavior.

## Issues Encountered

None after the test-fixture correction. Moto-backed coverage remains deterministic payload-participant evidence only, not live PostgreSQL/Amazon-S3 qualification.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 21 can add the existing receipt-bound rebuild-debt settlement path. Plan 22 can bind these selectors into the fixed Phase 7 verifier. No obstore adoption, listing-based recovery, second authority, or stronger cross-resource ACID claim was introduced.

## Self-Check: PASSED

- Confirmed all three modified production/test artifacts and this summary exist.
- Confirmed task commits `0e69875`, `681f363`, `873dea5`, `48318aa`, and `17a3f16` exist in repository history.
- Passed the seven exact Plan 20 selectors, the full local cutover/remote migration suite (25 passed), and scoped Ruff checks.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-11*
