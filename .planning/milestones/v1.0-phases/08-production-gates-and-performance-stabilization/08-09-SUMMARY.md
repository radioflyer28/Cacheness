---
phase: 08-production-gates-and-performance-stabilization
plan: 09
subsystem: testing
tags: [github-actions, release-qualification, coverage, ruff, platform-matrix]
requires:
  - phase: 08-02
    provides: package qualification and supported-installation probes
  - phase: 08-03
    provides: platform-matrix evidence producer
  - phase: 08-05
    provides: branch-aware coverage and quality verifier
  - phase: 08-06
    provides: structural call and memory bounds
provides:
  - Fixed pull-request and release-candidate quality workflow with class-separated evidence
  - Local core, packaging, platform, coverage, and structural gate orchestration
  - Operator-facing release qualification and nonclaim contract
affects: [phase-08, release-publication, qualification-collection]
actuals:
  tokens: 14594
  tasks: 2
  commits: 5
tech-stack:
  added: []
  patterns:
    - Class-separated evidence envelopes with no cross-class substitutions
    - Fixed CI commands delegated to version-controlled Python gate tools
key-files:
  created:
    - .github/workflows/quality.yml
    - docs/RELEASE_QUALIFICATION.md
    - tests/qualification/test_phase8_quality_workflow.py
  modified:
    - tools/run_phase8_local_gates.py
    - tools/phase8_evidence.py
    - tools/run_phase8_platform_gates.py
    - tools/run_phase8_scale_gates.py
key-decisions:
  - "Keep core quality evidence separate from TensorFlow wheel qualification so an unavailable optional dependency cannot downgrade core support evidence."
  - "Qualify retained TensorFlow support only on Python 3.11 and 3.12, matching the package probe's declared compatible range."
patterns-established:
  - "Release qualification: every evidence artifact binds one source revision and digest; absent or unavailable classes are never green substitutes."
requirements-completed: [QUAL-01, QUAL-02, QUAL-03, QUAL-04, QUAL-05, QUAL-07]
coverage:
  - id: D1
    description: "Pinned, least-privilege PR and release-candidate workflow runs fixed separated quality evidence classes."
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "tests/qualification/test_phase8_quality_workflow.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "Local core gate produces deterministic, coverage, and structural envelopes from one clean source identity."
    requirement: QUAL-05
    verification:
      - kind: integration
        ref: "uv run --isolated --all-extras --group dev --frozen python tools/run_phase8_local_gates.py core --output-dir <temporary-directory>/evidence"
        status: pass
    human_judgment: false
  - id: D3
    description: "Release guide declares qualification producers, terminal states, topology limits, and migration-fixture retention."
    requirement: QUAL-07
    verification:
      - kind: unit
        ref: "tests/qualification/test_phase8_quality_workflow.py -k documentation"
        status: pass
    human_judgment: false
duration: 32min
completed: 2026-09-14
status: complete
---

# Phase 08 Plan 09: Quality Workflow and Release Qualification Summary

**A pinned, evidence-class-separated CI workflow now qualifies core, packaging, platform, coverage, and structural evidence without promoting optional, live, or timing boundaries into one green result.**

## Performance

- **Duration:** 32 min
- **Started:** 2026-09-14T02:27:58Z
- **Completed:** 2026-09-14T02:59:32Z
- **Tasks:** 2/2
- **Files modified:** 8

## Accomplishments

- Added a least-privilege, SHA-pinned pull-request workflow covering Linux stable, macOS boundary, TensorFlow-compatible, and advisory prerelease roles without live credentials.
- Added exact-SHA release-candidate jobs with detached-checkout proof and fixed, separately uploaded deterministic, packaging, platform, coverage, and structural envelopes.
- Made local coverage and structural evidence first-class fixed gate outputs and documented the release evidence, platform scope, topology nonclaims, and migration-fixture boundary.

## Task Commits

1. **Task 1: Wire fixed quality commands into the supported CI matrix** - `33a43cf` (test), `d8216e0` (feat), `c4143d7` (fix), `7169394` (fix)
2. **Task 2: Publish the release qualification and nonclaim contract** - `b8a342e` (docs)

## Files Created/Modified

- `.github/workflows/quality.yml` - Pinned PR/push and exact-SHA release-candidate quality workflow.
- `tools/run_phase8_local_gates.py` - Fixed core, packaging, platform, coverage, and structural evidence dispatch.
- `tools/phase8_evidence.py` - Validation support for local coverage and structural envelopes.
- `tools/run_phase8_platform_gates.py` - Correct feature-aware platform contract execution.
- `tools/run_phase8_scale_gates.py` - Fixed structural observation collection for CI evidence.
- `tests/qualification/test_phase8_quality_workflow.py` - Workflow and release-guide contract suite.
- `tests/qualification/test_phase8_platform.py` - TensorFlow matrix truthfulness assertion.
- `docs/RELEASE_QUALIFICATION.md` - Evidence matrix, prerequisites, nonclaims, and retention rules.

## Decisions Made

- Core quality and TensorFlow packaging are separate evidence paths. The core path runs on the supported stable matrix, while the optional TensorFlow package path uses a compatible interpreter.
- Retained TensorFlow qualification is restricted to Python 3.11 and 3.12 because the completed package verifier truthfully marks later interpreter rows unavailable. This preserves handler support without inventing a compatibility claim.
- The workflow delegates selectors and thresholds to fixed Python tools; YAML orchestrates only reviewed commands and artifact collection.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Added first-class coverage and structural envelope production**
- **Found during:** Task 1 (Wire fixed quality commands into the supported CI matrix)
- **Issue:** The completed tools could validate supplied coverage/structural data, but the local workflow gate did not produce both evidence classes itself, leaving CI unable to upload the required class-separated artifacts.
- **Fix:** Added fixed coverage measurement plus ratchet/Ruff verification and fixed structural collection before writing validated envelopes.
- **Files modified:** `tools/run_phase8_local_gates.py`, `tools/phase8_evidence.py`, `tools/run_phase8_scale_gates.py`
- **Verification:** The composed `core` gate and focused workflow/structural test suite pass.
- **Committed in:** `d8216e0`

**2. [Rule 1 - Bug] Corrected the retained TensorFlow qualification matrix**
- **Found during:** Task 1 (Wire fixed quality commands into the supported CI matrix)
- **Issue:** The platform runner advertised TensorFlow through Python 3.13 while the package qualification implementation correctly treated TensorFlow compatibility as Python 3.11--3.12 only.
- **Fix:** Aligned the platform runner, CI matrix, workflow tests, and release guide to the package verifier's compatible range, while retaining all TensorFlow support.
- **Files modified:** `tools/run_phase8_platform_gates.py`, `.github/workflows/quality.yml`, `tests/qualification/test_phase8_platform.py`, `tests/qualification/test_phase8_quality_workflow.py`, `docs/RELEASE_QUALIFICATION.md`
- **Verification:** Platform and workflow qualification tests pass.
- **Committed in:** `c4143d7`

**3. [Rule 1 - Bug] Kept core quality gates independent from unavailable TensorFlow packaging**
- **Found during:** Task 1 (Wire fixed quality commands into the supported CI matrix)
- **Issue:** The generic all-gate path attempted optional TensorFlow package qualification on the core Python 3.13 row, turning a deliberate optional `UNAVAILABLE` state into a core quality failure.
- **Fix:** Added a `core` gate for deterministic, coverage, and structural evidence; retained packaging as a separately qualified compatible-Python path; and changed platform core rows to invoke only the deterministic contract suite.
- **Files modified:** `tools/run_phase8_local_gates.py`, `tools/run_phase8_platform_gates.py`, `.github/workflows/quality.yml`, `tests/qualification/test_phase8_quality_workflow.py`
- **Verification:** The composed local core gate, platform tests, workflow tests, tracer tests, structural tests, and Ruff checks pass.
- **Committed in:** `7169394`

**Total deviations:** 3 auto-fixed (2 Rule 1, 1 Rule 2).

**Impact on plan:** All changes are bounded to quality evidence production and matrix truthfulness. They add no storage lifecycle coordination, topology guarantee, dependency, or production runtime behavior.

## Issues Encountered

- An initial composed core-gate run ended before coverage output. Standalone coverage evidence and a subsequent complete core-gate run passed unchanged, confirming no persistent implementation failure.

## User Setup Required

None - no external service configuration required for the local/CI quality workflow. Protected live-service and controlled-performance qualifications remain explicit later prerequisites.

## Next Phase Readiness

- Fixed local and CI quality evidence is ready for the release collector and later protected live-service/controlled-performance workflows.
- TensorFlow remains supported and qualified only on the documented Python 3.11--3.12 rows; its eventual removal remains deferred to SEED-005.

## Self-Check: PASSED

Verified all eight created/modified task artifacts and all five task commits. The task diff contains no tracked-file deletions.

---
*Phase: 08-production-gates-and-performance-stabilization*
*Completed: 2026-09-14*
