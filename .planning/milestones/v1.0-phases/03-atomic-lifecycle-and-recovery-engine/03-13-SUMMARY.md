---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "13"
subsystem: test-isolation-and-release-acceptance
tags: [pytest, uv, ruff, isolated-environment, release-gate, windows-qualification]
requires:
  - phase: 03-10
    provides: Measured lifecycle budgets and retained final release gates
  - phase: 03-11
    provides: Current-host D-32 qualification artifact and verifier
  - phase: 03-12
    provides: Immutable qualification command-role contract
provides:
  - A direct-interpreter Ruff gate that cannot mutate packages beneath pytest
  - One frozen isolated all-extras full-suite invocation for the repository
  - Evidence-backed Plan 03-10 closeout with D-32 still explicitly unqualified
affects: [phase-3-completion, phase-4, release-validation, phase-999.1]
actuals:
  tokens: 12500
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - In-suite tools execute from the active interpreter environment rather than nested dependency resolvers
    - Complete repository acceptance runs in a fresh frozen all-extras/dev environment
key-files:
  created:
    - tests/test_full_suite_environment.py
    - .planning/phases/03-atomic-lifecycle-and-recovery-engine/03-10-SUMMARY.md
  modified:
    - tests/test_phase1_quality_gates.py
    - docs/CROSS_PLATFORM_GUIDE.md
key-decisions:
  - "The Phase 1 Ruff gate invokes only the executable next to sys.executable and fails explicitly when that executable is absent."
  - "The sole complete-suite command is frozen, isolated, all-extras, and dev-group enabled; it does not redefine base dependencies."
  - "The fresh Darwin D-32 result remains exit 2/UNAVAILABLE/NOT_QUALIFIED with native_evidence false; Phase 999.1 still owns native Windows PASS/exit 0."
patterns-established:
  - "Environment-mutation cascades must be reproduced in a fresh isolated environment before they are classified as repository defects."
requirements-completed: [STOR-03, STOR-04, STOR-05, STOR-06, STOR-07]
full-suite:
  command: "uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false"
  result: pass
  exit_code: 0
  environment: fresh-frozen-isolated-all-extras-dev
  collection: complete-tests-tree-without-bypasses
plan-03-10-closeout:
  summary_path: ".planning/phases/03-atomic-lifecycle-and-recovery-engine/03-10-SUMMARY.md"
  status: complete
windows-qualification:
  runner_status: UNAVAILABLE
  runner_exit_code: 2
  milestone_status: NOT_QUALIFIED
  native_evidence: false
  backlog_phase: 999.1
  future_required_status: PASS
  future_required_exit_code: 0
coverage:
  - id: D1
    description: Phase 1's in-suite Ruff gate uses the active interpreter scripts directory and cannot invoke uv.
    requirement: STOR-03
    verification:
      - kind: unit
        ref: "tests/test_full_suite_environment.py::test_ruff_gate_executes_the_running_environment_binary_without_uv"
        status: pass
    human_judgment: false
  - id: D2
    description: The complete repository test tree passes in the supported frozen isolated all-extras/dev environment.
    requirement: STOR-05
    verification:
      - kind: integration
        ref: "uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false"
        status: pass
    human_judgment: false
  - id: D3
    description: Plan 03-10 closes only after its benchmark, Python 3.11, compatibility, qualification, Ruff-delta, and direct-Ruff gates pass.
    requirement: STOR-07
    verification:
      - kind: integration
        ref: "Plan 03-13 Task 2 retained-gate matrix"
        status: pass
    human_judgment: false
duration: 8min
completed: 2026-09-06
status: complete
---

# Phase 03 Plan 13: Full-Suite Isolation Closure Summary

**The in-suite Ruff quality gate no longer mutates pytest's environment, and Plan 03-10 now closes on a fully green frozen isolated repository suite plus every retained lifecycle release gate.**

## Performance

- **Duration:** 8min
- **Started:** 2026-09-06T00:52:36Z
- **Completed:** 2026-09-06T01:00:40Z
- **Tasks:** 2/2
- **Files modified:** 5, including the withheld Plan 03-10 closeout summary

## Accomplishments

- Added a TDD regression that captures the direct Ruff argv, forbids `uv` inside the quality gate, and records the historical test-isolation cascade as environment evidence rather than a product-defect waiver.
- Changed the quality gate to resolve `ruff` exclusively next to the running `sys.executable`, preserving its existing JSON parsing, scope, and exit-code contract while failing explicitly when Ruff is unavailable.
- Documented and passed the exact supported full-suite command, then passed all retained Plan 03-10 release gates before creating the Plan 03-10 summary.

## Verification

The affected post-Ruff slice passed:

```bash
uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase1_quality_gates.py tests/test_full_suite_environment.py tests/test_public_api_contract.py tests/test_s3_blob_backend.py tests/test_sql_cache.py tests/test_sql_cache_documentation.py tests/test_sql_cache_failure_contract.py -o log_cli=false
```

The complete repository suite then passed normally with the one supported command:

```bash
uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false
```

It used ordinary collection of the entire `tests/` tree and no ignore, deselection, xfail, error suppression, reordering, skip addition, or max-fail bypass. The remaining retained gates also passed: benchmark baseline verification, real Python 3.11 lifecycle focus, compatibility corpus validation, fresh D-32 qualification attestation, Phase 3 Ruff-delta verification, and direct Ruff on the declared scope.

The fresh Darwin D-32 command returned exactly exit 2 with `UNAVAILABLE`, `NOT_QUALIFIED`, and `native_evidence: false`. This remains evidence of an unavailable required platform—not Windows support. Phase 999.1 still requires an eligible native Windows Python 3.11 `PASS` with exit 0 before a Windows-qualified release.

## Task Commits

1. **Task 1: Keep optional dependencies intact across the in-suite Ruff gate**
   - `bd84071` — `test(03-13): add environment isolation regression`
   - `b2b5b7b` — `fix(03-13): isolate in-suite Ruff quality gate`
2. **Task 2: Run final release gates and close Plan 03-10 on evidence**
   - `78518a2` — `docs(03-13): close Plan 03-10 acceptance`

## Files Created/Modified

- `tests/test_full_suite_environment.py` — direct-interpreter tooling and complete-suite contract regression coverage.
- `tests/test_phase1_quality_gates.py` — direct scripts-directory Ruff execution with explicit absence failure.
- `docs/CROSS_PLATFORM_GUIDE.md` — exact frozen isolated all-extras/dev test command and test-isolation classification.
- `03-10-SUMMARY.md` — final evidence-backed Plan 03-10 acceptance record.

## Decisions Made

- Kept optional dependencies in their declared extras and the dev group; a complete repository test run uses them explicitly instead of altering the base installation contract.
- Treated the earlier public-import, S3/botocore, and SQL/pandas cascade as a mutable-environment issue because the same nodes passed from the fresh isolated environment.
- Preserved D-32's distinction between current-host unavailability evidence and future native Windows release qualification.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The successful full suite emitted the pre-existing `SqliteBackend.__del__` interpreter-shutdown diagnostic after pytest returned exit 0. It is protected baseline debt and was neither hidden nor altered by this plan.

## Known Stubs

None.

## Next Phase Readiness

- Plan 03-10 has a committed evidence record and ordinary Phase 3 completion tracking can now advance.
- Native Windows qualification is still deliberately outstanding in Phase 999.1; no current artifact makes a Windows support claim.

## Self-Check: PASSED

- The Task 1 and Task 2 commits exist, and both Plan 03-10 and Plan 03-13 summaries exist.
- The complete suite and every retained final gate completed with required exit codes from fresh frozen isolated environments.
- D-32 remains `UNAVAILABLE`/`NOT_QUALIFIED` with `native_evidence: false` and future native `PASS`/exit 0 owned by Phase 999.1.
