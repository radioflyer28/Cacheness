---
phase: 09-adoption-and-release-surface-closure
plan: 06
subsystem: adoption-example-ci
tags: [ci, github-actions, examples, pytest, documentation]
requires:
  - phase: 09-adoption-and-release-surface-closure
    provides: Four exact executable BlobStore and UnifiedCache example journeys
provides:
  - A blocking stable CI gate for the four canonical published examples
  - A static workflow/index contract that prevents example-surface drift
  - Removal of the duplicate Phase 6 example harness and final obsolete local scripts
affects: [09-07, 09-08, 09-09, 09-10, phase-10]
actuals:
  tokens: 5696
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - CI invokes the literal public example files through one reviewed pytest harness.
    - The examples index is an allowlist coupled to the same four-file CI contract.
key-files:
  created:
    - tests/test_phase9_quality_workflow.py
  modified:
    - .github/workflows/quality.yml
    - examples/README.md
  deleted:
    - tests/test_phase6_examples.py
    - examples/simple_function_caching.py
    - examples/simple_ml_pipeline.py
    - examples/simple_object_caching.py
key-decisions:
  - "The exact example gate runs on the Linux 3.13 stable row, keeping it non-live, blocking, and inside the existing frozen dependency environment."
  - "The supported examples index lists only the four canonical local journeys; SqlCache assets remain physically untouched for Phase 10."
patterns-established:
  - "Workflow source is statically tested for the fixed published-example command and its least-privilege non-live placement."
requirements-completed: [CACH-06]
coverage:
  - id: D1
    description: The stable non-live workflow runs the literal Phase 9 example harness, and the supported index links exactly the same four local journeys.
    requirement: CACH-06
    verification:
      - kind: integration
        ref: uv run pytest -q -o log_cli=false tests/test_phase9_quality_workflow.py tests/test_phase9_examples.py tests/qualification/test_phase8_quality_workflow.py -x
        status: pass
    human_judgment: false
  - id: D2
    description: The obsolete Phase 6 harness and three final non-SqlCache scripts are absent while the canonical contract stays green.
    requirement: CACH-06
    verification:
      - kind: integration
        ref: uv run pytest -q -o log_cli=false tests/test_phase9_quality_workflow.py tests/test_phase9_examples.py -x
        status: pass
    human_judgment: false
duration: 2min
completed: 2026-09-17
status: complete
---

# Phase 9 Plan 06: Canonical Example CI and Cleanup Summary

**The four published local BlobStore/UnifiedCache examples are now a blocking, exact-file CI contract, while the stale Phase 6 harness and remaining duplicate scripts are gone.**

## Performance

- **Duration:** 2 min
- **Started:** 2026-09-17T03:33:51Z
- **Completed:** 2026-09-17T03:36:15Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Added a static contract that binds the Linux 3.13 stable row to one frozen invocation of `tests/test_phase9_examples.py`, without secrets, live-service selectors, or advisory failure handling.
- Rewrote the examples index as the canonical four-link local adoption surface: memory store, durable filesystem-plus-SQLite catalog, UnifiedCache policy, and custom MCAP-style format registration.
- Deleted the superseded Phase 6 example harness and the obsolete function-caching, ML-pipeline, and object-caching scripts without modifying Phase-10-owned SqlCache assets.

## Task Commits

1. **Task 1: Make the exact published example surface a fixed CI gate**
   - `f8b4f42` — `test(09-06): add failing CI example contract`
   - `9f6a7af` — `feat(09-06): gate canonical examples in CI`
2. **Task 2: Remove the superseded harness and final non-SqlCache duplicates**
   - `bbba070` — `refactor(09-06): remove superseded example surface`

## Verification

- `uv run pytest -q -o log_cli=false tests/test_phase9_quality_workflow.py tests/test_phase9_examples.py tests/qualification/test_phase8_quality_workflow.py -x` — passed (12 tests).
- `uv run pytest -q -o log_cli=false tests/test_phase9_quality_workflow.py tests/test_phase9_examples.py -x` — passed (6 tests) after the deletions.
- `uv run ruff check tests/test_phase9_quality_workflow.py` — passed.
- Confirmed the four removed paths are absent and that no supported documentation references them.

## Decisions Made

- The example gate shares the existing Linux 3.13 stable row and its frozen isolated environment, preserving the Phase 8 workflow's live/release/packaging/coverage evidence boundaries.
- The public examples index promotes only qualified local journeys. It does not link or otherwise promote SqlCache, while leaving its Phase 10-owned files physically intact.

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None.

## User Setup Required

None - no external service configuration is required.

## Next Phase Readiness

- Plans 09-07 through 09-10 can link documentation and qualification material to one continuously verified example surface.
- Phase 10 retains ownership of direct SqlCache removal; this plan only removed stale non-SqlCache duplicates and stopped SqlCache promotion.

## Self-Check: PASSED

- The summary and all three Task commits are present in Git history.

*Phase: 09-adoption-and-release-surface-closure*
*Completed: 2026-09-17*
