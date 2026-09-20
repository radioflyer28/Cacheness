---
phase: 07-explicit-migration-and-rebuild-cutover
plan: "02"
subsystem: migration-governance
tags: [migration, rebuild, cutover, decision-gates, offline-maintenance]
requires:
  - phase: 07-01
    provides: "Explicit offline inspect-plan-stage-verify-activate tracer and validation-only entry-point boundary"
provides:
  - "Explicitly accepted D-02 previous-release support window"
  - "Explicitly accepted D-08 include-all rebuild exclusion contract"
  - "Explicitly accepted D-15 finalize-then-separate-purge retirement contract"
affects: [07-03, 07-08, 07-09, migration, rebuild, release-contract]
actuals:
  tokens: 3421
  tasks: 3
  commits: 1
tech-stack:
  added: []
  patterns:
    - "Blocking-human one-way decisions use exact resume signals and are summarized before dependent implementation starts."
    - "Finalize ends rollback eligibility; a separately confirmed purge is retryable cleanup debt and cannot revise activation success."
key-files:
  created:
    - .planning/phases/07-explicit-migration-and-rebuild-cutover/07-02-SUMMARY.md
  modified:
    - .planning/STATE.md
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md
key-decisions:
  - "D-02 accepted via `proceed D-02`: each release directly supports its current and immediately previous released layout; older stores advance through declared steps."
  - "D-08 accepted via `proceed D-08`: rebuild excludes nothing by default, and any exclusion needs an exact regenerated and reconfirmed plan."
  - "D-15 accepted via `proceed D-15`: finalize permanently ends rollback, while separately confirmed idempotent purge may physically delete the retained prior copy and leaves retryable debt on failure."
patterns-established:
  - "Do not infer destructive migration actions from time, candidate presence, or ordinary cleanup; require an explicit operator signal at each one-way boundary."
requirements-completed: [MIGR-04, MIGR-05, MIGR-06]
coverage:
  - id: D1
    description: "The locked D-02 release-window contract has an explicit proceed record before compatibility-window publication."
    requirement: MIGR-04
    verification:
      - kind: manual_procedural
        ref: "Execution history: proceed D-02"
        status: pass
    human_judgment: true
    rationale: "The gate approves the contract only; downstream implementation and qualification remain separate work."
  - id: D2
    description: "The locked D-08 include-all rebuild exclusion contract has an explicit proceed record before rebuild implementation."
    requirement: MIGR-06
    verification:
      - kind: manual_procedural
        ref: "Execution history: proceed D-08"
        status: pass
    human_judgment: true
    rationale: "The gate approves the contract only; downstream implementation and qualification remain separate work."
  - id: D3
    description: "The locked D-15 finalize and separately confirmed purge contract has an explicit proceed record before retirement implementation."
    requirement: MIGR-05
    verification:
      - kind: manual_procedural
        ref: "Execution history: proceed D-15"
        status: pass
    human_judgment: true
    rationale: "The gate approves the contract only; downstream implementation and qualification remain separate work."
duration: 2m 5s
completed: 2026-09-09
status: complete
---

# Phase 07 Plan 02: One-Way Migration Decisions Summary

**The developer explicitly accepted the bounded previous-release migration window, include-all-by-default rebuild policy, and a finalize-then-separate-purge retirement boundary before their dependent plans may implement them.**

## Performance

- **Duration:** 2m 5s
- **Started:** 2026-09-09T23:11:15Z
- **Completed:** 2026-09-09T23:13:20Z
- **Tasks:** 3
- **Files modified:** 0 product files; planning records only.

## Accomplishments

- Recorded `proceed D-02`: a release directly supports only its current and immediately previous released layout; older released stores proceed through successive declared migration steps. Phase 7 creates the first baseline without historical development-layout readers or a manufactured target version.
- Recorded `proceed D-08`: a rebuild starts with no exclusions. Any exclusion must come from a newly generated plan with exact keys or explicit categories, counts, bytes, reasons, and a matching confirmation; incompatible and cross-backend moves are rebuilds, not universal physical migration.
- Recorded `proceed D-15`: activation retains the prior valid store; finalize records acceptance and permanently ends rollback; a later separately confirmed purge is idempotent physical cleanup whose failure is retryable debt and never rewrites activation success.

## Task Commits

No task commits were created: all three tasks are blocking-human decision gates with no code, persisted-data, or task-artifact changes. Their exact resume signals are recorded here before dependent implementation proceeds.

**Plan metadata:** recorded in the final documentation commit.

## Files Created/Modified

- `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-02-SUMMARY.md` - Immutable execution record of the three accepted one-way contracts and their exact resume signals.
- `.planning/STATE.md` - Sequential plan position, decisions, metrics, and session continuity.
- `.planning/ROADMAP.md` - Phase 7 plan-progress count.
- `.planning/REQUIREMENTS.md` - Requirement tracking updated from the plan frontmatter.

## Decisions Made

- **D-02 — accepted (`proceed D-02`):** Publish the locked current-plus-immediately-previous released-layout support window; do not broaden it with unsupported development-layout readers.
- **D-08 — accepted (`proceed D-08`):** Keep rebuild include-all by default and require exact regenerated/reconfirmed exclusion evidence for any omission.
- **D-15 — accepted (`proceed D-15`):** Keep activation, finalize, and purge distinct. Finalize ends rollback only; later purge may remove the retained physical prior copy and reports failed cleanup as retryable debt.

## Verification

- `test -f .planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md` — passed for D-02 and D-08 source contract presence.
- `test -f docs/adr/0001-topology-specific-storage-guarantees.md` — passed for D-15 lifecycle guardrail presence.
- Execution history contains the exact accepted signals: `proceed D-02`, `proceed D-08`, and `proceed D-15`.

## Deviations from Plan

None - plan executed exactly as written. No irreversible implementation or physical purge was performed; this plan only records the explicit approvals that gate those later actions.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plans 07-03, 07-08, and 07-09 may use the exact accepted D-02, D-15, and D-08 contracts respectively. Each remains responsible for implementation and its own verification; no downstream action is authorized by inference from this approval record.

## Self-Check: PASSED

- This summary and the three referenced source artifacts exist on disk.
- The three exact proceed signals are recorded above without product-code or persisted-format changes.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-09*
