---
phase: 11-clean-supplemental-documentation-and-normalize-validation-ev
plan: 07
subsystem: current-guidance-and-evidence
tags: [documentation, codebase-maps, package-extras, wheel-qualification, seed-resolution]
requires:
  - phase: 11-03
    provides: "TensorFlow-free runtime, package, and installed-wheel cutover"
  - phase: 11-04
    provides: "Core-only CI and qualification-profile cutover"
  - phase: 11-05
    provides: "Current documentation and qualification-claim cutover"
  - phase: 11-06
    provides: "Canonical Phase 3 validation evidence"
provides:
  - "Current agent guidance and codebase maps for the exact five-extra retained surface"
  - "Fulfilled SEED-005 with preserved Phase 8 rationale and non-promotable Phase 11 resolution evidence"
affects: [11-08, 11-09, 11-10, milestone-audit]
tech-stack:
  added: []
  patterns:
    - "Current maps name literal retained package inventories and negative source-free wheel boundaries."
    - "Fulfilled seeds preserve historical rationale while removing promotion fields and linking exact resolution evidence."
key-files:
  created: []
  modified:
    - AGENTS.md
    - .planning/codebase/ARCHITECTURE.md
    - .planning/codebase/CONCERNS.md
    - .planning/codebase/STACK.md
    - .planning/codebase/TESTING.md
    - .planning/seeds/SEED-005-remove-native-tensorflow-support.md
decisions:
  - "Keep BlobStore authority, UnifiedCache policy-only ownership, private handler staging, and all remote/platform/publication nonclaims unchanged while updating only the retired package surface."
  - "Fulfill SEED-005 in place so its Phase 8 deferral remains available as historical rationale without leaving a promotion path."
requirements-completed: [D-05, D-07, D-08, D-12, D-18]
coverage:
  - id: D1
    description: "Agent guidance and current codebase maps name the exact retained five-extra surface, source-free wheel absence contract, and unchanged lifecycle boundaries."
    requirement: D-05
    verification:
      - kind: unit
        ref: "uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_documentation.py tests/packaging/test_wheel_matrix.py -x"
        status: pass
    human_judgment: false
  - id: D2
    description: "SEED-005 is fulfilled with Phase 11 cutover links, preserved historical rationale, and no promotion fields."
    requirement: D-07
    verification:
      - kind: unit
        ref: "tests/test_phase9_evidence_metadata.py::test_phase11_seed_resolution_is_canonical"
        status: pass
    human_judgment: false
metrics:
  duration: 8m
  completed: 2026-09-19
status: complete
actuals:
  tokens: 2267
  tasks: 2
  commits: 2
---

# Phase 11 Plan 07: Retained-Surface Guidance and Seed Resolution Summary

**Current guidance now fixes Cacheness at its five retained extras and source-free wheel boundary, while SEED-005 records the completed TensorFlow removal without losing its Phase 8 rationale.**

## Performance

- **Duration:** 8m
- **Started:** 2026-09-19T18:30:25Z
- **Completed:** 2026-09-19
- **Tasks:** 2/2
- **Files modified:** 7

## Accomplishments

- Updated agent instructions and all current codebase maps to describe the retained dataframe, serialization, S3, and PostgreSQL surface with exactly `recommended`, `dataframes`, `s3`, `postgresql`, and `cloud` as published extras.
- Preserved the BlobStore sole-authority, UnifiedCache policy-only, private handler-staging, integrity, and deferred remote/platform/performance/publication boundaries while adding the negative source-free wheel contract.
- Fulfilled SEED-005 in place with Phase 11 cutover evidence, retained Phase 8 deferral history and breadcrumbs, and removed promotion fields.

## Task Commits

Each task was committed atomically:

1. **Task 1: Refresh agent guidance and current codebase maps to the retained surface** — `4043d9f` (`docs`)
2. **Task 2: Fulfill SEED-005 with a Phase 11 resolution link** — `6ab640f` (`docs`)

## Files Created/Modified

- `AGENTS.md` — names the exact five published extras and retained integrations without changing storage guardrails.
- `.planning/codebase/ARCHITECTURE.md`, `CONCERNS.md`, `STACK.md`, and `TESTING.md` — describe the retained implementation surface, negative fresh-wheel proof, and unchanged nonclaims.
- `.planning/seeds/SEED-005-remove-native-tensorflow-support.md` — preserves Phase 8 history while recording fulfilled Phase 11 resolution evidence.
- `.planning/phases/11-clean-supplemental-documentation-and-normalize-validation-ev/deferred-items.md` — records the later Plan 11-08 validation-normalization handoff found by the broad regression gate.

## Decisions Made

- Current guidance names only the literal manifest surface; lifecycle ownership and qualification limits remain their existing canonical contracts.
- A fulfilled seed stays as historical rationale, but its `trigger_when` and unknown scope fields are removed so it cannot be proposed as future work.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- The prescribed documentation and wheel contract passed (42 tests), and the targeted seed contract passed (1 test).
- The broader frozen non-live suite completed after the seed resolution but failed three out-of-scope canonical-validation selectors because Phase 01 and Phase 07 still declare `status: complete`. Plan 11-08 owns those validation records; the exact handoff is recorded in `deferred-items.md`.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 11-08 can normalize the remaining Phase 01, 05, 06, 07, 08, and 09 validation records without reopening their historical scope or lifecycle work.
- The frozen non-live gate has advanced beyond SEED-005; its remaining three failures are the explicitly deferred Plan 11-08 records.

## Self-Check: PASSED

- Found all six Task 1/2 files and this summary on disk.
- Found Task 1 commit `4043d9f` and Task 2 commit `6ab640f` in Git history.

---
*Phase: 11-clean-supplemental-documentation-and-normalize-validation-ev*
*Completed: 2026-09-19*
