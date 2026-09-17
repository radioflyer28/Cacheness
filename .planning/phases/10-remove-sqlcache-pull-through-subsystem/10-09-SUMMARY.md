---
phase: 10-remove-sqlcache-pull-through-subsystem
plan: 09
subsystem: guidance-and-acceptance
tags: [codebase-maps, pytest, ruff, uv, wheel, lifecycle-guardrail]
requires:
  - phase: 10-remove-sqlcache-pull-through-subsystem
    provides: Direct runtime, dependency, documentation, and example cutover
provides:
  - Current agent and codebase maps for the BlobStore-first product boundary
  - Complete source, dependency, reference, fresh-wheel, and frozen non-live acceptance evidence
affects: [phase-10-closure, future-planning, release-qualification]
actuals:
  tokens: 44442
  tasks: 3
  commits: 2
tech-stack:
  added: []
  patterns:
    - Current maps describe the retained BlobStore authority and UnifiedCache policy boundary without rewriting historical evidence
    - Phase closure combines exact current-reference scanning with fresh isolated-wheel and frozen non-live regression evidence
key-files:
  created:
    - .planning/phases/10-remove-sqlcache-pull-through-subsystem/10-09-SUMMARY.md
  modified:
    - AGENTS.md
    - .planning/codebase/ARCHITECTURE.md
    - .planning/codebase/CONCERNS.md
    - .planning/codebase/CONVENTIONS.md
    - .planning/codebase/INTEGRATIONS.md
    - .planning/codebase/STACK.md
    - .planning/codebase/STRUCTURE.md
    - .planning/codebase/TESTING.md
    - tests/test_phase10_sqlcache_removal.py
key-decisions:
  - "Keep the complete current-facing absence scan narrow by allowing only exact negative verifier/test nodes, while canonical documentation remains separately phrase-checked."
  - "Record local/non-live acceptance without promoting it to live remote-service or controlled-Linux performance qualification."
patterns-established:
  - "Current maps describe retained roles and boundaries, while dated audits and completed phase artifacts remain historical evidence."
requirements-completed: [CACH-07]
coverage:
  - id: D1
    description: Current agent, architecture, concerns, and convention maps document a BlobStore-first lifecycle with UnifiedCache as policy only.
    requirement: CACH-07
    verification:
      - kind: unit
        ref: "uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase9_documentation.py -x"
        status: pass
    human_judgment: false
  - id: D2
    description: Integration, stack, structure, and testing maps match retained dependency and qualification ownership without stale current product claims.
    requirement: CACH-07
    verification:
      - kind: unit
        ref: "tests/test_phase10_sqlcache_removal.py::test_current_facing_references_match_allowlist"
        status: pass
      - kind: other
        ref: "uv lock --check"
        status: pass
    human_judgment: false
  - id: D3
    description: The final tree passes public/verifier/documentation/packaging contracts, scoped Ruff, the fresh isolated wheel, and the full frozen non-live suite.
    requirement: CACH-07
    verification:
      - kind: integration
        ref: "tests/packaging/test_wheel_matrix.py"
        status: pass
      - kind: other
        ref: "uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'"
        status: pass
    human_judgment: false
metrics:
  duration: 10 min
  completed: 2026-09-17
status: complete
---

# Phase 10 Plan 09: Guidance and Acceptance Summary

**Current maps now describe the retained BlobStore lifecycle and UnifiedCache policy boundary, and the complete frozen non-live and isolated-wheel acceptance chain is green.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-09-17T17:54:28Z
- **Completed:** 2026-09-17T18:04:57Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments

- Replaced stale current agent, architecture, concern, and convention guidance with the post-cut BlobStore-first authority model, ADR 0001 stop conditions, retained handler seam, and explicit qualification limits.
- Refreshed integration, dependency, structure, and testing maps for the remaining SQLAlchemy/PostgreSQL roles, six published extras, obstore participant, canonical examples, packaging proof, and non-live test boundary.
- Ran the complete Phase 10 evidence chain: scoped Ruff, lock integrity, current-reference scan, public/verifier/docs/packaging contracts, fresh wheel installation/round trips, and the required frozen non-live suite.

## Task Commits

1. **Task 1: Refresh current project, architecture, concern, and convention maps** - `cacba41` (docs)
2. **Task 2: Refresh integration, stack, structure, and testing maps** - `bd1ccdb` (docs)
3. **Task 3: Run the complete bounded Phase 10 acceptance chain** - verification-only; no source change or task commit

## Files Created/Modified

- `AGENTS.md` - Gives agents current product, lifecycle, dependency, validation, GSD, and historical-evidence guidance.
- `.planning/codebase/ARCHITECTURE.md` and `CONCERNS.md` - Describe one lifecycle authority, bounded reconciliation, and the guardrail against coordination accretion.
- `.planning/codebase/CONVENTIONS.md`, `INTEGRATIONS.md`, `STACK.md`, `STRUCTURE.md`, and `TESTING.md` - Map current handler, authority, package, example, and qualification ownership.
- `tests/test_phase10_sqlcache_removal.py` - Keeps the current-reference allowlist exact while admitting existing negative verifier/test selectors.

## Decisions Made

- Kept the current-reference scanner fail-closed and narrowly extended its allowlist only for existing negative contract nodes that prove absence; current maps remain unallowlisted and marker-free.
- Treated the local/non-live suite and fresh wheel as strong bounded evidence, while preserving the explicit nonclaim for live remote-service and controlled-Linux performance qualification.

## Verification

- `uv lock --check` — pass.
- Scoped Ruff across every Phase 10 changed surviving Python contract/packaging file — pass.
- `tests/test_phase10_sqlcache_removal.py::test_current_facing_references_match_allowlist` and `::test_manifest_lock_dependency_contract` — 2 passed.
- Frozen focused public/verifier/documentation/packaging acceptance set, including `tests/packaging/test_wheel_matrix.py` — exit 0; includes fresh isolated wheel archive/metadata/import/local-round-trip proof.
- `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'` — exit 0. Expected platform/TensorFlow skips and one existing collection warning were reported; no failures occurred.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical contract] Allow exact existing negative verifier/test references in the current-reference scanner**

- **Found during:** Task 2 (Refresh integration, stack, structure, and testing maps)
- **Issue:** The new scan correctly found four existing negative contract selectors, but its initial narrow allowlist omitted their exact paths and therefore blocked the map acceptance gate.
- **Fix:** Added only `tests/test_phase4_cutover_verifier.py`, `tests/test_phase071_contract_verifier.py`, `tests/test_phase9_quality_workflow.py`, and `tools/verify_phase071_contracts.py` as reviewed negative-contract owners.
- **Files modified:** `tests/test_phase10_sqlcache_removal.py`
- **Verification:** Current-reference and manifest/lock tests pass; the full focused and frozen non-live suites pass.
- **Committed in:** `bd1ccdb`

---

**Total deviations:** 1 auto-fixed (1 Rule 2 missing critical contract).
**Impact on plan:** The scanner remains fail-closed for every other current path and no product, lifecycle, dependency, or historical artifact behavior changed.

## Issues Encountered

The initial focused current-reference gate identified the omitted negative contract nodes above; the exact allowlist correction resolved it. The first non-interactive test capture ended before emitting a final status, so the same bounded suite was rerun in a monitored session and exited 0. No test failure or runtime defect was masked.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All nine Phase 10 plans now have current-map and acceptance evidence ready for independent phase verification and closeout.
- Live PostgreSQL/Amazon-S3 release qualification, immutable publication, and controlled-Linux performance remain explicitly deferred rather than implied by this plan.

## Self-Check: PASSED

- Confirmed all eight current maps, `AGENTS.md`, and this summary exist.
- Confirmed task commits `cacba41` and `bd1ccdb` exist in Git history.
- Confirmed the working tree has no unintended tracked change from this plan;
  user-owned `.planning/config.json`, `.claude/`, and `.planning/milestone.lock`
  remain unstaged.

---
*Phase: 10-remove-sqlcache-pull-through-subsystem*
*Completed: 2026-09-17*
