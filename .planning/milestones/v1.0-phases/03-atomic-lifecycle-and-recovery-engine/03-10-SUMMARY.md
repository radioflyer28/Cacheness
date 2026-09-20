---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "10"
subsystem: lifecycle-release-acceptance
tags: [lifecycle-authority, benchmark, configuration, python-311, ruff, windows-qualification]
requires:
  - phase: 03-11
    provides: Current-host Windows qualification attestation with an honest UNAVAILABLE disposition
  - phase: 03-12
    provides: Digest-bound distinction between runtime and native qualification commands
  - phase: 03-13
    provides: A non-mutating in-suite Ruff gate and fresh all-extras full-suite evidence
provides:
  - Measured lifecycle limits derived from the checked-in authority benchmark
  - Complete Phase 3 acceptance evidence from frozen isolated environments
  - Explicit D-32 deferral that preserves Phase 999.1 native Windows qualification
affects: [phase-3-completion, phase-4, phase-999.1, release-validation]
actuals:
  tokens: 39573
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - Lifecycle defaults are derived from checked-in benchmark percentiles and declared multipliers
    - Final acceptance uses a fresh isolated all-extras environment rather than a mutable nested tool invocation
key-files:
  created:
    - benchmarks/lifecycle_authority_benchmark.py
    - benchmarks/lifecycle_authority_baseline.json
  modified:
    - src/cacheness/config.py
    - tests/test_config_validation.py
    - tests/test_sqlite_lifecycle_authority.py
    - tests/test_blob_store_concurrency.py
    - tests/test_blob_store_reconciliation.py
key-decisions:
  - "LifecycleLimits maps only the measured reconciliation row/action/byte/time and busy dimensions; transaction, clear, and overlap percentiles remain release envelopes."
  - "The full repository suite runs only through the frozen isolated all-extras plus dev command; base-install coverage remains distinct."
  - "Darwin exit 2/UNAVAILABLE remains NOT_QUALIFIED with native_evidence false; Phase 999.1 still requires native Windows PASS/exit 0."
patterns-established:
  - "Release evidence is accepted only when a complete fresh suite and each retained gate exit normally."
requirements-completed: [STOR-03, STOR-04, STOR-05, STOR-06, STOR-07]
windows-qualification:
  artifact_path: ".planning/phases/03-atomic-lifecycle-and-recovery-engine/03-09-WINDOWS-QUALIFICATION.md"
  runner_status: UNAVAILABLE
  runner_exit_code: 2
  milestone_status: NOT_QUALIFIED
  native_evidence: false
  backlog_phase: 999.1
  future_required_status: PASS
  future_required_exit_code: 0
coverage:
  - id: D1
    description: Measured lifecycle defaults and release envelopes have baseline provenance and deterministic regression coverage.
    requirement: STOR-06
    verification:
      - kind: integration
        ref: "uv run --isolated --all-extras --group dev --frozen python benchmarks/lifecycle_authority_benchmark.py --verify-baseline benchmarks/lifecycle_authority_baseline.json"
        status: pass
    human_judgment: false
  - id: D2
    description: The entire repository test tree passes from the supported frozen isolated environment without collection exclusions.
    requirement: STOR-03
    verification:
      - kind: integration
        ref: "uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false"
        status: pass
    human_judgment: false
  - id: D3
    description: Current-host platform evidence is verified as explicitly unqualified rather than representing a Windows support claim.
    requirement: STOR-07
    verification:
      - kind: integration
        ref: "uv run --isolated --all-extras --group dev --frozen python tools/capture_phase3_windows_qualification.py verify …"
        status: pass
    human_judgment: false
duration: 4h 38min
completed: 2026-09-06
status: complete
---

# Phase 03 Plan 10: Lifecycle Budget and Final Acceptance Summary

**Measured lifecycle limits and frozen isolated acceptance gates now close Plan 03-10 without turning Darwin's unavailable qualification evidence into a Windows support claim.**

## Performance

- **Duration:** 4h 38min, including deferred final acceptance after Plans 03-11 through 03-13
- **Started:** 2026-09-06T00:22:58Z
- **Completed:** 2026-09-06T01:00:40Z
- **Tasks:** 2/2
- **Files modified:** 7 implementation/test files, plus this evidence summary

## Accomplishments

- Added production-schema lifecycle measurements, provenance, and checked-in distributions that derive only the explicit reconciliation/busy configuration limits.
- Preserved transaction, clear-snapshot, and distinct-key-overlap measurements as benchmark-only release envelopes rather than turning noisy timing into runtime policy.
- Closed the final release matrix with a freshly created frozen isolated all-extras environment: complete repository suite, benchmark, Python 3.11 focus, compatibility corpus, attestation verifier, Ruff delta, and direct Phase 3 Ruff scope all exited successfully.

## Final Acceptance Evidence

The supported complete-suite command is exactly:

```bash
uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false
```

It completed normally with no ignore, deselection, xfail, added skip, or max-fail bypass. The normal suite retains its pre-existing platform and service-dependent skips; it was run against all declared extras and the dev group, not by broadening base dependencies.

The retained Plan 03-10 gates also passed from frozen isolated environments:

- `python benchmarks/lifecycle_authority_benchmark.py --verify-baseline benchmarks/lifecycle_authority_baseline.json`
- Python 3.11 lifecycle/concurrency/close/scheduler/qualification-attestation focus
- `python tests/fixtures/compat/validate_corpus.py --expected-through sqlite-columns-v0314`
- `python tools/capture_phase3_windows_qualification.py verify …`
- `python tools/verify_phase3_ruff_delta.py`
- Direct Ruff over the declared Phase 3 new-file scope plus `tests/test_full_suite_environment.py`

The fresh qualification verifier re-ran the fixed Python 3.11 command on Darwin and required its actual exit 2, `UNAVAILABLE`, `NOT_QUALIFIED`, and `native_evidence: false` result. This closes the current-host evidence obligation only. Phase 999.1 still requires eligible native Windows Python 3.11 `PASS` at exit 0 before any Windows-qualified release.

## Task Commits

1. **Task 1: Measure production-schema lifecycle distributions**
   - `c5dc418` — `feat(03-10): measure lifecycle authority budgets`
2. **Task 2: Derive gates and run final Phase 3 acceptance**
   - `60f1850` — `test(03-10): require measured lifecycle defaults`
   - `e9f6fa6` — `feat(03-10): derive lifecycle limits from baseline`

Plan 03-13 supplied the final test-isolation repair required before this closeout:

- `bd84071` — `test(03-13): add environment isolation regression`
- `b2b5b7b` — `fix(03-13): isolate in-suite Ruff quality gate`

## Files Created/Modified

- `benchmarks/lifecycle_authority_benchmark.py` — production-schema measurement and verification harness.
- `benchmarks/lifecycle_authority_baseline.json` — recorded distributions, provenance, multipliers, and derived configuration values.
- `src/cacheness/config.py` — evidence-derived LifecycleLimits defaults.
- Lifecycle test modules — deterministic configuration, busy deadline, reconciliation bound, and overlap-envelope coverage.
- `tests/test_phase1_quality_gates.py` and `tests/test_full_suite_environment.py` — non-mutating in-suite Ruff regression contract consumed by final acceptance.

## Decisions Made

- Retained the base versus optional dependency contract; the all-extras/dev environment is the repository-suite contract, not a reason to make SQL, dataframe, S3, or TensorFlow packages base dependencies.
- Classified the earlier post-Ruff public-import, S3/botocore, and SQL/pandas cascade as shared-environment mutation, not individual product failures, because it did not reproduce in the fresh isolated environment.
- Kept D-32 strict: unavailable Darwin evidence is not native Windows evidence and cannot support a Windows-qualified release.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed nested dependency resolution from the Phase 1 Ruff quality gate.**

- **Found during:** Plan 03-13 Task 1, before this delayed closeout.
- **Issue:** `uv run ruff` started a second dependency resolution inside the pytest interpreter environment, mutating installed files that later optional-feature tests needed.
- **Fix:** Resolve Ruff only from the scripts directory beside `sys.executable`, fail explicitly if it is absent, and regression-test the argv.
- **Files modified:** `tests/test_phase1_quality_gates.py`, `tests/test_full_suite_environment.py`, `docs/CROSS_PLATFORM_GUIDE.md`.
- **Verification:** The affected post-Ruff slice and the complete frozen isolated suite both passed.
- **Committed in:** `b2b5b7b`.

**Total deviations:** 1 auto-fixed (Rule 1).
**Impact on plan:** Necessary for trustworthy final acceptance; no production API or dependency contract changed.

## Issues Encountered

- The successful full suite printed the already-known shutdown-time `SqliteBackend.__del__` interpreter-teardown diagnostic after pytest had exited 0. It is unrelated protected baseline debt and was not treated as a failed release gate or changed by this plan.

## Known Stubs

None.

## Next Phase Readiness

- Phase 3 final acceptance now has reproducible benchmark, compatibility, supported-Python, lint, and complete-suite evidence.
- Phase 999.1 remains the sole path to a Windows-qualified release; it must produce native Windows `PASS`/exit 0 evidence.

## Self-Check: PASSED

- Commits `c5dc418`, `60f1850`, `e9f6fa6`, `bd84071`, and `b2b5b7b` exist.
- The Plan 03-11 artifact and summary remain structured as `UNAVAILABLE`/`NOT_QUALIFIED` with `native_evidence: false` and Phase 999.1 as the future gate.
- Every retained final command exited with its required result in a fresh isolated environment.
