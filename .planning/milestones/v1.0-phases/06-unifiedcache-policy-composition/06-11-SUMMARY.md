---
phase: 06-unifiedcache-policy-composition
plan: "11"
subsystem: testing
tags: [pytest, ruff, verification, cache-policy, suite-isolation]
requires:
  - phase: 06-09
    provides: Migrated cache format, containment, signing, array-security, and public API tests.
  - phase: 06-10
    provides: Migrated catalog-query, metadata-parameter, and quality-gate tests.
provides:
  - Deterministic all-extras local suite gate with exactly three Phase 8 live exclusions.
  - Fixed Phase 6 verifier with literal migrated-test inventory and fail-closed AST audit.
  - Executed local validation evidence for CACH-01 through CACH-07.
affects: [phase-06-verification, phase-08-qualification, BACK-05]
tech-stack:
  added: []
  patterns:
    - Fixed test-node inventories with independent completeness validation.
    - AST checks that distinguish executable compatibility use from deliberate absence tests.
key-files:
  created:
    - tools/run_phase6_local_suite.py
    - tests/test_phase6_suite_isolation.py
  modified:
    - tools/verify_phase6_contracts.py
    - tests/test_phase6_contract_verifier.py
    - .planning/phases/06-unifiedcache-policy-composition/06-VALIDATION.md
key-decisions:
  - "The local suite ignores only the three named Phase 8 PostgreSQL/S3 files; mocks and skips do not qualify live services."
  - "The fixed verifier audits a literal Plan 09-11 inventory and permits legacy syntax only in TypeError or explicit absence negatives."
  - "The locked all-extras local environment closes CACH-07 and the non-live suite gate; native Windows and BACK-05 remain unqualified."
actuals:
  tokens: 9164
  tasks: 2
  commits: 2
requirements-completed: [CACH-01, CACH-02, CACH-03, CACH-04, CACH-05, CACH-06, CACH-07]
coverage:
  - id: D1
    description: Exact three-module non-live suite selection with ordered public/SqlCache isolation coverage.
    requirement: CACH-06
    verification:
      - kind: integration
        ref: uv run --isolated --all-extras --group dev --frozen python tools/run_phase6_local_suite.py --repo-root .
        status: pass
    human_judgment: false
  - id: D2
    description: Fixed canonical-cutover verifier rejects retired configuration, construction, and cache-surface calls.
    requirement: CACH-07
    verification:
      - kind: integration
        ref: uv run --isolated --all-extras --group dev --frozen python tools/verify_phase6_contracts.py --repo-root .
        status: pass
    human_judgment: false
  - id: D3
    description: Scoped Plan 09-11 Python files satisfy the repository lint gate.
    verification:
      - kind: other
        ref: uv run --isolated --all-extras --group dev --frozen ruff check [Plan 09-11 paths]
        status: pass
    human_judgment: false
duration: 12 min
completed: 2026-09-09
status: complete
---

# Phase 6 Plan 11: Deterministic Local Suite Gate Summary

**A fixed all-extras local suite and canonical-cutover verifier now prove the migrated cache and SqlCache tests together while excluding only unqualified live service modules.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-09-09T08:00:38Z
- **Completed:** 2026-09-09T08:12:16Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Added a root-safe local runner that invokes the normal repository suite and fails closed unless exactly the three Phase 8 live PostgreSQL/S3 modules are ignored.
- Proved public API and SqlCache regressions are order-isolated in subprocesses, and repaired the retained Phase 1/3 node inventory.
- Added the nine migrated Plan 09/10 modules, isolation test, and gate tests to the fixed verifier; its AST audit rejects real flat `CacheConfig`, implicit `UnifiedCache`, and removed cache-surface calls while retaining deliberate structural negatives.
- Recorded final local evidence: 1,223 passed, 9 skipped, 0 failed from 1,232 collected nodes; fixed verifier and scoped Ruff passed.

## Task Commits

1. **Task 1: Run the complete local corpus with exact live-module exclusion and clean test-owned state** — `4038f30` (test)
2. **Task 2: Expand the fixed verifier and record the final local evidence** — `a43f536` (test)

## Files Created/Modified

- `tools/run_phase6_local_suite.py` — validates and executes the exact non-live suite selection.
- `tests/test_phase6_suite_isolation.py` — tests runner fail-closed behavior and bidirectional order isolation.
- `tests/test_full_suite_environment.py` — records the Phase 6 local command separately from the Phase 8 release command.
- `tests/test_phase3_gap_acceptance.py` — uses surviving deterministic integrity/recovery nodes.
- `tools/verify_phase6_contracts.py` — runs the fixed cutover manifest and source-aware AST compatibility audit.
- `tests/test_phase6_contract_verifier.py` — mutation tests for positive legacy calls, allowed negatives, and inventory omission.
- `.planning/phases/06-unifiedcache-policy-composition/06-VALIDATION.md` — final command, environment, count, and qualification evidence.

## Decisions Made

- The literal local-suite tuple is an integrity boundary: duplicate, missing, broad, or non-live exclusions fail before pytest starts.
- The verifier’s permitted legacy patterns are intentionally narrow: a `pytest.raises(TypeError)` block or an explicit `hasattr`/signature absence assertion. Comments and strings are not executable evidence.
- The all-extras locked local environment is the only source of this plan’s green suite verdict. PostgreSQL, Amazon S3, and native Windows remain `UNAVAILABLE`/`NOT_QUALIFIED`.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

The sandbox initially could not read uv’s existing cache outside the workspace. The authorized frozen commands then ran against that cache without changing dependencies or the lockfile.

## Known Stubs

None.

## Self-Check: PASSED

- All seven task artifacts and this summary exist on disk.
- Both task commits, `4038f30` and `a43f536`, exist in Git history.

## Next Phase Readiness

- Phase 6’s finite local verification gate is green, including CACH-07 and the exact non-live suite selection.
- BACK-05 real PostgreSQL/Amazon-S3 qualification and native Windows evidence remain Phase 8 obligations; no local mock or skipped result is a release claim.

---
*Phase: 06-unifiedcache-policy-composition*
*Completed: 2026-09-09*
