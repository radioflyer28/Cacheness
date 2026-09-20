---
phase: 08-production-gates-and-performance-stabilization
plan: 13
subsystem: release qualification
tags: [performance, preflight, git, fingerprint, verification]
requires:
  - phase: 08-10
    provides: exact-source release evidence and fixed Phase 8 verification inventory
provides:
  - read-only controlled-runner eligibility preflight
  - canonical bounded machine fingerprint and SHA-256 digest
  - literal Plan 08-13 threat-to-test bindings
affects: [08-11, release-qualification, controlled-performance]
actuals:
  tokens: 13863
  tasks: 1
  commits: 2
tech-stack:
  added: []
  patterns:
    - fixed allow-list evidence schemas with canonical JSON digests
    - optional-lock-free bounded Git read probes
key-files:
  created: []
  modified:
    - benchmarks/phase8_benchmarks.py
    - tests/performance/test_phase8_benchmarks.py
    - tools/verify_phase8_contracts.py
    - tests/test_phase8_contract_verifier.py
key-decisions:
  - "Preflight emits eligibility evidence only; it never measures workloads, writes a baseline, or claims performance stability."
  - "Only cacheness-perf-linux-x64, a clean detached exact SHA, and a fixed bounded fingerprint may qualify a runner."
  - "Git reads use --no-optional-locks and stable sanitised diagnostics so source checks cannot mutate or disclose repository details."
patterns-established:
  - "Controlled-runner checks validate exact record shape and canonical digest before later capture consumes them."
  - "Release verifier inventories remain literal; additive plans and threat selectors are never discovered dynamically."
requirements-completed: [QUAL-06]
coverage:
  - id: D1
    description: "Controlled-runner preflight accepts only fixed-label Linux x86-64, clean detached exact-commit eligibility and emits a bounded canonical record."
    requirement: QUAL-06
    verification:
      - kind: unit
        ref: "tests/performance/test_phase8_benchmarks.py#test_preflight_runner_emits_only_the_bounded_eligibility_record"
        status: pass
      - kind: unit
        ref: "tests/performance/test_phase8_benchmarks.py#test_preflight_runner_rejects_non_clean_or_attached_repository_state"
        status: pass
    human_judgment: false
  - id: D2
    description: "The Phase 8 fixed verifier binds Plan 08-13, every new threat, and the exact preflight selector."
    requirement: QUAL-06
    verification:
      - kind: unit
        ref: "tests/test_phase8_contract_verifier.py#test_fixed_manifest_covers_full_phase_decision_requirement_and_threat_sets"
        status: pass
    human_judgment: false
duration: 12min
completed: 2026-09-14
status: complete
---

# Phase 08 Plan 13: Controlled Runner Preflight Summary

**A read-only exact-commit controlled-runner preflight now emits a bounded canonical machine fingerprint/digest or fails closed before any performance capture.**

## Performance

- **Duration:** 12min
- **Started:** 2026-09-14T04:27:20Z
- **Completed:** 2026-09-14T04:39:00Z
- **Tasks:** 1
- **Files modified:** 4

## Accomplishments

- Added `--preflight-runner --expect-label --revision`, which accepts only the fixed logical runner, Linux x86-64, an exact clean detached SHA, and one schema-validated fingerprint.
- Bound only OS/architecture, CPU/governor, filesystem type, Python, uv, and SQLite fields into canonical SHA-256 eligibility evidence; repository, host, account, credential, timing, and raw-command details remain excluded.
- Added fail-closed regression coverage for source state, platform, labels, fingerprint drift, disclosure attempts, no-write behavior, and literal verifier inventory.

## Task Commits

1. **Task 1: Read-only exact-commit controlled-runner preflight** - `7ee3bac` (test), `1bf0446` (feat)

## Files Created/Modified

- `benchmarks/phase8_benchmarks.py` - Implements the isolated preflight CLI, bounded probes, canonical fingerprint validation, and optional-lock-free Git state checks.
- `tests/performance/test_phase8_benchmarks.py` - Exercises success, closed schema/digest, Git and platform failures, and no-write execution.
- `tools/verify_phase8_contracts.py` - Adds Plan 08-13 and its three threats to the fixed manifest.
- `tests/test_phase8_contract_verifier.py` - Locks the new literal plan and exact preflight selector binding.

## Decisions Made

- Preflight exits before workload selection, measurement, output creation, baseline capture/recalibration, or baseline verification.
- The sole admitted runner identity is the fixed logical label; all physical/host identities are deliberately excluded from the record.
- Platform/noise/stability and release performance-envelope validation remain exclusively Plan 08-11 work.

## Verification

- `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/performance/test_phase8_benchmarks.py tests/test_phase8_contract_verifier.py -x` — passed (34 tests).
- `uv run --isolated --group dev --frozen ruff check benchmarks/phase8_benchmarks.py tests/performance/test_phase8_benchmarks.py tools/verify_phase8_contracts.py tests/test_phase8_contract_verifier.py` — passed.
- `uv run --isolated --group dev --frozen ruff format --check benchmarks/phase8_benchmarks.py tests/performance/test_phase8_benchmarks.py tools/verify_phase8_contracts.py tests/test_phase8_contract_verifier.py` — passed.
- The documented command with the forced fixed logical label on this macOS host returned `runner preflight rejected: controlled runner platform is not eligible`, with no output/baseline file written.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test contract] Corrected the Git-status fixture to require `--untracked-files=all`.**
- **Found during:** Task 1
- **Issue:** The initial red fixture asserted a shorter status command and did not include the required untracked-file mode.
- **Fix:** Updated the fixture to require the explicit all-untracked porcelain invocation.
- **Files modified:** `tests/performance/test_phase8_benchmarks.py`
- **Verification:** All preflight source-state regressions pass.
- **Committed in:** `1bf0446`

**2. [Rule 3 - Verification] Formatted only the four plan-owned Python files required by the plan's Ruff format-check.**
- **Found during:** Task 1
- **Issue:** The required format-check found pre-existing formatting in the scoped harness and test file.
- **Fix:** Applied Ruff formatting only to the four files named by Plan 08-13; no source outside the plan scope changed.
- **Files modified:** `benchmarks/phase8_benchmarks.py`, `tests/performance/test_phase8_benchmarks.py`
- **Verification:** The named Ruff format-check passes.
- **Committed in:** `1bf0446`

---

**Total deviations:** 2 auto-fixed (1 Rule 1, 1 Rule 3).
**Impact on plan:** Both changes were required to make the documented preflight assertions and mandated scoped formatter gate truthful; no behavioral scope expanded.

## Issues Encountered

- The present macOS host correctly cannot qualify as the named controlled Linux runner. It remains a blocking `UNAVAILABLE` prerequisite for Plan 08-11 rather than substitute performance evidence.

## User Setup Required

None - this plan adds a repository-local preflight only. Plan 08-11 still requires its separately documented dedicated Linux runner.

## Next Phase Readiness

- Plan 08-11 can consume a retry-safe eligibility record from a clean detached `cacheness-perf-linux-x64` checkout before baseline capture.
- No lifecycle, storage, concurrency, TensorFlow, runner-provisioning, performance-threshold, or baseline change was made.

## Self-Check: PASSED

---
*Phase: 08-production-gates-and-performance-stabilization*
*Completed: 2026-09-14*
