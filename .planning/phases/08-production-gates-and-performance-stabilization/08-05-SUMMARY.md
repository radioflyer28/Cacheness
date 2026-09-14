---
phase: 08-production-gates-and-performance-stabilization
plan: 05
subsystem: testing
tags: [coverage, ruff, release-gate, pytest]
requires:
  - phase: 08-04
    provides: Named lifecycle and cache-policy coverage selectors
provides:
  - Branch-aware repository and critical-scope coverage ratchet
  - Direct Ruff lint and formatting gate for fixed critical and changed Python paths
  - Checked, reviewable Phase 8 coverage baseline
affects: [phase-08-quality-workflows, release-qualification]
actuals:
  tokens: 38159
  tasks: 2
  commits: 6
tech-stack:
  added: []
  patterns:
    - Strict canonical evidence parsing with literal AST-validated selectors
    - Read-only ratchet verification separated from explicit justified baseline capture
key-files:
  created:
    - tools/verify_phase8_coverage.py
    - tests/qualification/phase8_coverage_baseline.json
  modified:
    - pyproject.toml
    - tests/test_phase8_coverage_gate.py
    - .gitignore
key-decisions:
  - "Coverage floors compare covered and total statement/branch counts plus derived rates for repository and critical scopes."
  - "Coverage capture excludes all protected live-service markers; mocks, skips, and unavailable services never become baseline evidence."
  - "The baseline is an explicit canonical JSON artifact; ordinary verification never mutates it."
patterns-established:
  - "Release evidence uses a fixed selector inventory alongside numerical metrics."
  - "Ruff receives a bounded argv list from safe changed paths plus a literal critical inventory."
requirements-completed: [QUAL-03, QUAL-05]
coverage:
  - id: D1
    description: Branch-aware repository and critical-scope coverage ratchet with literal selector validation.
    requirement: QUAL-05
    verification:
      - kind: unit
        ref: tests/test_phase8_coverage_gate.py
        status: pass
      - kind: other
        ref: uv run --isolated --all-extras --group dev --frozen python tools/verify_phase8_coverage.py --report build/phase8/coverage.json --baseline tests/qualification/phase8_coverage_baseline.json --ruff
        status: pass
    human_judgment: false
  - id: D2
    description: Measured initial coverage floor from the deterministic non-live all-extras suite.
    requirement: QUAL-03
    verification:
      - kind: integration
        ref: uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false -m "not (live_postgresql or live_aws_s3 or live_remote)" -x --cov=cacheness --cov-branch --cov-report=json:build/phase8/coverage.json --cov-report=xml:build/phase8/coverage.xml
        status: pass
    human_judgment: false
duration: 1h 7m
completed: 2026-09-13
status: complete
---

# Phase 08 Plan 05: Branch-Aware Coverage and Direct Ruff Gates Summary

**A canonical, branch-aware coverage ratchet now protects named lifecycle and cache-policy contracts, with direct Ruff checks and a measured non-live baseline.**

## Performance

- **Duration:** 1h 7m
- **Started:** 2026-09-13T22:05:56-04:00
- **Completed:** 2026-09-13T23:12:00-04:00
- **Tasks:** 2/2
- **Files modified:** 23

## Accomplishments

- Enabled branch measurement and added a strict verifier that validates canonical Coverage.py JSON, source inventory, literal selector functions, raw counts, and derived rates.
- Established a checked macOS/CPython 3.13.15 non-live baseline: repository 75.34% statements / 59.04% branches; critical scope 77.94% / 60.13%.
- Added direct Ruff lint and format enforcement for the complete Phase 8 critical inventory plus safely parsed changed Python paths, without a lint-debt ledger.
- Normalized the critical lifecycle, cache-policy, qualification, packaging, and release-tracer source scopes so the direct gate passes.

## Task Commits

1. **Task 1: Build the strict branch-aware coverage ratchet** - `03f5e2c` (RED test), `6573226` (implementation)
2. **Task 2: Measure and record the post-gap coverage floors** - `0676e55`, `72b18c4`, `3d8df54` (gate repairs), `57b21a1` (baseline)

## Files Created/Modified

- `tools/verify_phase8_coverage.py` - strict parser, selector preflight, capture/verify separation, and direct Ruff runner.
- `tests/test_phase8_coverage_gate.py` - failure-mode and safe-default contracts.
- `tests/qualification/phase8_coverage_baseline.json` - canonical measured coverage floor and selector inventory.
- `pyproject.toml` - Coverage.py branch measurement.
- `.gitignore` - ignores local Coverage.py runtime files but permits the one checked qualification baseline.
- `src/cacheness/` critical modules and Phase 8 test scopes - Ruff-format normalized production and evidence surfaces.

## Decisions Made

- Coverage scope denominators are ratcheted as well as covered counts and rates, so reducing the measured surface cannot create an artificial improvement.
- Baseline capture explicitly excludes `live_postgresql`, `live_aws_s3`, and `live_remote`; protected live qualification remains a separate evidence class.
- The literal `HEAD^` default is accepted only as the verifier's documented parent reference; arbitrary revision expressions remain rejected.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Prevented a shrinking coverage scope from passing with an inflated rate**
- **Found during:** Task 1
- **Issue:** The initial comparison checked covered counts and rates but not total statement/branch counts.
- **Fix:** Ratcheted total counts and limited selector AST discovery to module-level tests.
- **Files modified:** `tools/verify_phase8_coverage.py`, `tests/test_phase8_coverage_gate.py`
- **Verification:** Focused coverage-gate tests pass.
- **Committed in:** `03f5e2c`, `6573226`

**2. [Rule 3 - Blocking] Made the deterministic measurement command exclude every protected live-service suite**
- **Found during:** Task 2
- **Issue:** The command excluded only `live_remote`, causing real PostgreSQL tests to run without their protected fixture scope.
- **Fix:** Added a literal non-live marker expression covering PostgreSQL, S3, and remote topology tests.
- **Files modified:** `tools/verify_phase8_coverage.py`, `tests/test_phase8_coverage_gate.py`
- **Verification:** The complete non-live all-extras coverage suite passed.
- **Committed in:** `0676e55`

**3. [Rule 1 - Bug] Accepted the verifier's documented safe `HEAD^` default**
- **Found during:** Task 2
- **Issue:** The direct Ruff path gate rejected its own default base revision before it could inspect paths.
- **Fix:** Allowed only the literal documented parent reference in addition to bounded branch/ref names.
- **Files modified:** `tools/verify_phase8_coverage.py`, `tests/test_phase8_coverage_gate.py`
- **Verification:** Focused tests and the real `--ruff` gate pass.
- **Committed in:** `3d8df54`

**4. [Rule 2 - Missing Critical Functionality] Kept required baseline evidence visible and generated coverage files out of the worktree**
- **Found during:** Task 2
- **Issue:** Global JSON ignore rules hid the required checked baseline, while Coverage.py runtime files were unignored.
- **Fix:** Added the exact baseline exception and ignored local `.coverage*` files.
- **Files modified:** `.gitignore`, `tests/qualification/phase8_coverage_baseline.json`
- **Verification:** The baseline is tracked and read-only verification passes.
- **Committed in:** `72b18c4`, `57b21a1`

**Total deviations:** 4 auto-fixed (2 bugs, 1 blocking issue, 1 missing critical function).

## Issues Encountered

- The full deterministic suite passed with the repository's pre-existing SQLite resource warnings and platform/TensorFlow skips. They were neither reclassified as qualification evidence nor expanded into lifecycle work.

## Next Phase Readiness

- Phase 8 workflows can invoke the fixed non-live coverage command and read-only verifier to detect coverage or critical-scope Ruff regressions.
- Protected PostgreSQL/Amazon-S3, Windows, packaging, and performance qualification remain independently scoped Phase 8 work.

## Self-Check: PASSED

- Confirmed required coverage verifier and baseline files exist.
- Confirmed all six task commits exist in Git history.
