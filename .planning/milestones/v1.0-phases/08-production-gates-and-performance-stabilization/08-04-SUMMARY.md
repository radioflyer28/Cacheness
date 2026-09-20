---
phase: 08-production-gates-and-performance-stabilization
plan: 04
subsystem: deterministic-lifecycle-testing
tags: [pytest, lifecycle, postgresql, cache-policy, integrity, recovery]
requires:
  - phase: 07.1-obstore-payload-participant-unification
    provides: One guarded obstore payload participant below BlobStore's lifecycle authority
  - phase: 08-production-gates-and-performance-stabilization
    provides: Exact-commit deterministic release-evidence tracer and fixed local gate
provides:
  - Named lifecycle integrity, recovery, bounded-clear, and ADR progress contracts
  - Deterministic PostgreSQL DB-API classification, replay, pagination, and rollback selectors
  - Public UnifiedCache policy contracts for typed causes and exact bounded removal
affects: [phase-08-coverage-baseline, phase-08-contract-verifier, QUAL-04, QUAL-05]
actuals:
  tokens: 8607
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Scripted DB-API transaction transcripts for non-live PostgreSQL boundary contracts
    - Public BlobStore and UnifiedCache behavior assertions over internal line coverage
key-files:
  created:
    - tests/test_phase8_lifecycle_coverage.py
    - tests/test_phase8_cache_policy_coverage.py
  modified: []
key-decisions:
  - "PostgreSQL DB-API contracts run without a live service and exercise only the declared transaction/error boundary."
  - "Typed conflict and retryable outcomes remain valid ADR 0001 progress results; tests never require every contender to succeed."
  - "UnifiedCache tests observe BlobStore receipts and typed causes without adding cache-side lifecycle coordination."
patterns-established:
  - "Phase 8 selector families use literal postgresql_error_classification, postgresql_replay, postgresql_pagination, and postgresql_transaction_rollback names."
  - "Maintenance continuations are rejected before storage I/O when their immutable signed evidence is malformed."
requirements-completed: [QUAL-04]
coverage:
  - id: D1
    description: Named lifecycle contracts prove integrity-before-deserialization, exact ambiguous publication settlement, cleanup debt recovery, bounded clear safety, and PostgreSQL authority fault boundaries.
    requirement: QUAL-04
    verification:
      - kind: unit
        ref: uv run pytest -q -o log_cli=false tests/test_phase8_lifecycle_coverage.py -x
        status: pass
      - kind: integration
        ref: uv run --isolated --all-extras --group dev --frozen python tools/verify_phase071_contracts.py --all
        status: pass
    human_judgment: false
  - id: D2
    description: Named UnifiedCache policy contracts prove typed cause preservation, exact invalidation accounting, stale restart, maintenance validation, and storage-free statistics.
    requirement: QUAL-05
    verification:
      - kind: unit
        ref: uv run pytest -q -o log_cli=false tests/test_phase8_cache_policy_coverage.py -x
        status: pass
    human_judgment: false
duration: 12m 14s
completed: 2026-09-13
status: complete
---

# Phase 08 Plan 04: Lifecycle and Cache-Policy Coverage Summary

**Named deterministic contracts now exercise lifecycle integrity/recovery and UnifiedCache policy boundaries before Phase 8 selects coverage floors.**

## Performance

- **Duration:** 12m 14s
- **Started:** 2026-09-13T23:59:54Z
- **Completed:** 2026-09-14T00:12:11Z
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments

- Added authority-first integrity, exact lost-response settlement, cleanup-debt recovery, and bounded clear contracts without adding lifecycle coordination.
- Added literal PostgreSQL selector families for SQLSTATE/driver error classification, immutable replay, bounded cursor progression, and rollback at every named transition boundary.
- Added public UnifiedCache contracts for declared typed lookup causes, exact deletion results, stale-cursor restart, immutable maintenance evidence, and storage-free statistics.

## Task Commits

1. **Task 1: Add named lifecycle integrity and recovery branch contracts** - `5b07bbe` (test RED), `84c5cb1` (test GREEN)
2. **Task 2: Add named cache-policy and invalidation branch contracts** - `f375505` (test RED), `6ead52f` (test GREEN)

## Files Created/Modified

- `tests/test_phase8_lifecycle_coverage.py` - Focused lifecycle/participant and PostgreSQL authority fault contracts.
- `tests/test_phase8_cache_policy_coverage.py` - Public cache-policy lookup, removal, continuation, maintenance, and statistics contracts.

## Decisions Made

- Deterministic PostgreSQL coverage uses a bounded scripted DB-API transaction transcript; it is not live-service qualification.
- Publication and cleanup tests assert exact immutable locator/digest evidence, not listing-based visibility or a cross-resource transaction.
- Policy tests assert original BlobStore causes and exact deletion expectations, preserving the BlobStore/UnifiedCache ownership split.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added a minimal SQL-composition fixture for default focused test execution.**

- **Found during:** Task 1
- **Issue:** The prescribed focused command does not install the optional PostgreSQL extra, so `PostgresqlLifecycleAuthority` rejects construction before its deterministic DB-API boundaries can be exercised.
- **Fix:** Added a test-local composable SQL fixture matching only the authority's identifier/query construction seam. The matrix still uses scripted DB-API transactions and no live service, package installation, or production fallback.
- **Files modified:** `tests/test_phase8_lifecycle_coverage.py`
- **Verification:** Both focused suites pass, and the inherited Phase 07.1 all-mode gate passes with its real optional-extra PostgreSQL selector.
- **Committed in:** `84c5cb1`

---

**Total deviations:** 1 auto-fixed (1 Rule 3)
**Impact on plan:** The fixture makes the plan's prescribed deterministic test command executable without changing production PostgreSQL behavior, dependency declarations, or topology guarantees.

## Known Stubs

None.

## Issues Encountered

None.

## Verification

- `uv run pytest -q -o log_cli=false tests/test_phase8_lifecycle_coverage.py -x` — 19 passed.
- `uv run pytest -q -o log_cli=false tests/test_phase8_cache_policy_coverage.py -x` — 11 passed.
- Combined focused suites — 30 passed.
- `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase071_contracts.py --all` — passed in 35.99s; live AWS/PostgreSQL, native platforms, packaging, and performance remain explicitly unqualified Phase 8 evidence classes.
- Ruff check and formatting check pass for both new test files.

## User Setup Required

None - this plan adds deterministic local contracts only. Live PostgreSQL and Amazon S3 qualification remain separate external evidence gates.

## Next Phase Readiness

- Plan 08-05 may now measure coverage and capture its read-only baseline because all four required PostgreSQL selector families collect and pass.
- QUAL-05 remains pending until Plan 08-05 captures and enforces the post-gap statement and branch floors.

## Self-Check: PASSED

- Both focused Phase 8 test files and this summary exist on disk.
- All four Task 1/Task 2 RED and GREEN commits resolve in Git history.

---
*Phase: 08-production-gates-and-performance-stabilization*
*Completed: 2026-09-13*
