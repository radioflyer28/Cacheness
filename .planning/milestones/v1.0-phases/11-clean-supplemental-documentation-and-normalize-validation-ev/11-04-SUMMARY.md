---
phase: 11-clean-supplemental-documentation-and-normalize-validation-ev
plan: 04
subsystem: qualification-tooling
tags: [ci, qualification, feature-profiles, github-actions, tensorflow-removal]
requires:
  - phase: 11-02
    provides: "Core-only platform and workflow contract tests."
  - phase: 11-03
    provides: "Removed TensorFlow runtime, package, and qualification surface."
provides:
  - "A literal core-only Phase 8 platform and local-gate profile contract."
  - "A TensorFlow-free quality workflow that retains non-live and protected-live boundaries."
affects: [11-06, 11-07, 11-08, 11-09, 11-10, milestone-audit]
actuals:
  tokens: 1776
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - "Qualification profile inventories stay literal and fail closed."
    - "Retired CI paths are deleted directly while retained jobs preserve frozen evidence semantics."
key-files:
  created: []
  modified:
    - tools/run_phase8_platform_gates.py
    - tools/run_phase8_local_gates.py
    - .github/workflows/quality.yml
    - tests/qualification/test_phase8_quality_workflow.py
key-decisions:
  - "Keep core as the sole literal profile instead of retaining compatibility aliases or an extension point."
  - "Run the retained release-packaging job on the pinned core interpreter while preserving frozen and protected-live workflow semantics."
  - "Treat the release-guide TensorFlow-presence assertion as stale after Plan 11-05 and enforce its absence instead."
requirements-completed: [D-05, D-08, D-18]
coverage:
  - id: D1
    description: "Phase 8 platform and local tooling accepts and emits only the core feature profile while preserving exact platform nonclaims."
    requirement: D-05
    verification:
      - kind: unit
        ref: "uv run --isolated --all-extras --group dev --frozen pytest -q tests/qualification/test_phase8_platform.py tests/qualification/test_phase8_release.py -x"
        status: pass
    human_judgment: false
  - id: D2
    description: "The quality workflow has no TensorFlow-only path and retains frozen, non-live, and protected-live job structure."
    requirement: D-08
    verification:
      - kind: unit
        ref: "uv run --isolated --all-extras --group dev --frozen pytest -q tests/qualification/test_phase8_quality_workflow.py tests/test_phase9_quality_workflow.py -x"
        status: pass
    human_judgment: false
  - id: D3
    description: "The combined platform, release, workflow, and Phase 9 parser contracts remain green without contacting an external service."
    requirement: D-18
    verification:
      - kind: integration
        ref: "uv run --isolated --all-extras --group dev --frozen pytest -q tests/qualification/test_phase8_platform.py tests/qualification/test_phase8_release.py tests/qualification/test_phase8_quality_workflow.py tests/test_phase9_quality_workflow.py -x"
        status: pass
    human_judgment: false
duration: 4m 42s
completed: 2026-09-19
status: complete
---

# Phase 11 Plan 04: Core-only Platform Gates and Quality Workflow Summary

**Core-only Phase 8 profile tooling and a TensorFlow-free quality workflow, retaining Linux/macOS evidence boundaries, native-Windows nonqualification, and protected-live semantics.**

## Performance

- **Duration:** 4m 42s
- **Started:** 2026-09-19T18:08:28Z
- **Completed:** 2026-09-19T18:13:10Z
- **Tasks:** 2/2
- **Files modified:** 4

## Accomplishments

- Reduced the platform and local gate profile inventory to literal `core`, rejecting both retired names and all unknown inputs before evidence emission.
- Deleted the TensorFlow-only workflow job and setup/probe wording while retaining frozen package, canonical-example, non-live, and protected-live jobs.
- Preserved exact Linux, macOS, Windows, source-identity, deferred-service, performance, and immutable-publication nonclaims.

## Task Commits

Each task was committed atomically:

1. **Task 1: Reduce platform and local gates to the sole core feature profile** — `b16f0c4` (`feat`)
2. **Task 2: Delete TensorFlow-only workflow jobs and repair retained job structure** — `868d5b1` (`feat`)

## Files Created/Modified

- `tools/run_phase8_platform_gates.py` — retains the exact core Linux matrix and platform nonclaim aggregation without retired profile branches.
- `tools/run_phase8_local_gates.py` — exposes only `core` as the local CLI profile choice.
- `.github/workflows/quality.yml` — removes the TensorFlow-only job and keeps the retained packaging job pinned to the core interpreter.
- `tests/qualification/test_phase8_quality_workflow.py` — corrects the stale release-guide assertion to require the retired surface's absence.

## Decisions Made

- Kept the profile inventory literal and fail closed; no alias, tombstone, generic profile discovery, or replacement CI path was introduced.
- Moved the retained packaging path to Python 3.13 because its former Python 3.12 selection existed solely as a TensorFlow compatibility carve-out.
- Kept the quality contract aligned with the completed documentation cutover by asserting that the canonical release guide no longer contains retired TensorFlow claims.

## Verification

- PASS — `uv run --isolated --all-extras --group dev --frozen pytest -q tests/qualification/test_phase8_platform.py tests/qualification/test_phase8_release.py -x` (36 passed).
- PASS — `uv run --isolated --all-extras --group dev --frozen pytest -q tests/qualification/test_phase8_quality_workflow.py tests/test_phase9_quality_workflow.py -x` (9 passed).
- PASS — combined non-live contract set: 45 passed.
- PASS — scoped Ruff for `tools/run_phase8_platform_gates.py`, `tools/run_phase8_local_gates.py`, and `tests/qualification/test_phase8_quality_workflow.py`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Repaired the stale release-guide TensorFlow-presence assertion.**

- **Found during:** Task 2 (Delete TensorFlow-only workflow jobs and repair retained job structure)
- **Issue:** The exact workflow verification still required `TensorFlow` in `RELEASE_QUALIFICATION.md`, contradicting completed Plan 11-05's direct removal of current TensorFlow documentation.
- **Fix:** Replaced that stale positive assertion with an absence assertion while retaining Linux/macOS and all nonclaim checks.
- **Files modified:** `tests/qualification/test_phase8_quality_workflow.py`
- **Verification:** The task-specific workflow suite passed 9/9 and the combined non-live suite passed 45/45.
- **Committed in:** `868d5b1` (part of Task 2)

---

**Total deviations:** 1 auto-fixed (1 Rule 1 bug).
**Impact on plan:** The fix restored the intended post-cutover contract without changing runtime behavior, lifecycle scope, or the retained workflow boundaries.

## Issues Encountered

None beyond the documented stale assertion, which was corrected during Task 2.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Later Phase 11 validation and acceptance work can rely on a single core-only platform profile and a TensorFlow-free current CI surface.
- Live PostgreSQL/Amazon-S3, controlled-Linux performance, native Windows, and immutable-publication evidence remain explicitly deferred or nonqualified.

## Self-Check: PASSED

- Found `11-04-SUMMARY.md` on disk.
- Found Task 1 commit `b16f0c4` and Task 2 commit `868d5b1` in Git history.

---
*Phase: 11-clean-supplemental-documentation-and-normalize-validation-ev*
*Completed: 2026-09-19*
