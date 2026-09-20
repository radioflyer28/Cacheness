---
phase: 08-production-gates-and-performance-stabilization
plan: 08
subsystem: live-service-qualification
tags: [postgresql, amazon-s3, obstore, github-actions, evidence, cleanup]
requires:
  - phase: 07.1-obstore-payload-participant-unification
    provides: One guarded ObstoreGenerationIO participant beneath BlobStore lifecycle authority
  - phase: 08-production-gates-and-performance-stabilization
    provides: Exact source/revision evidence vocabulary and protected performance-workflow precedent
provides:
  - Phase 8 fail-closed PostgreSQL/Amazon-S3 qualification runner
  - Exact marker-owned, bounded cleanup for Phase 8 qualification namespaces
  - Protected exact-SHA release-candidate and diagnostic scheduled workflows
affects: [BACK-05, QUAL-03, phase-08-release-verifier, release-qualification]
actuals:
  tokens: 13483.75
  tasks: 3
  commits: 3
tech-stack:
  added: []
  patterns:
    - Fixed real-service test inventory with exact revision/source-digest validation
    - Sanitized terminal evidence with explicit release-candidate versus scheduled roles
    - Marker-authorized bounded cleanup and protected GitHub Actions secret scoping
key-files:
  created:
    - tools/run_phase8_qualification.py
    - tests/qualification/test_phase8_evidence.py
    - tests/qualification/test_phase8_live_workflow.py
    - .github/workflows/live_qualification.yml
  modified:
    - tests/qualification/conftest.py
key-decisions:
  - "Only the release_candidate role can emit QUALIFIED evidence; scheduled diagnostics remain NOT_QUALIFIED even after a clean pass."
  - "Phase 8 accepts exactly one phase-specific run ID and maps it to a q8 schema/prefix without altering Phase 5 namespaces."
  - "Live source identity covers the current authority, obstore participant, fixed suite, qualification tools, and reviewed workflows; missing future contract files fail closed."
patterns-established:
  - "Real-service evidence validates literal sources, exact cleanup, standard AWS identity, and a fixed pytest selection before publication can use it."
  - "Live workflow secrets are scoped to one protected runner step after exact detached-SHA verification."
requirements-completed: [QUAL-03]
requirements-pending: [BACK-05]
coverage:
  - id: D1
    description: Fail-closed exact-revision live runner rejects missing, stale, incomplete, emulated, and owner-pinned evidence.
    requirement: BACK-05
    verification:
      - kind: unit
        ref: uv run pytest -q -o log_cli=false tests/qualification/test_phase8_evidence.py -k runner -x
        status: pass
    human_judgment: false
  - id: D2
    description: Phase 8 cleanup only touches exact marker-owned schema/prefix resources and preserves residue on ambiguity or failures.
    requirement: BACK-05
    verification:
      - kind: unit
        ref: uv run pytest -q -o log_cli=false tests/qualification/test_phase8_evidence.py -k 'cleanup or redact or secret' -x
        status: pass
    human_judgment: false
  - id: D3
    description: Protected exact-SHA RC and scheduled-diagnostic workflow separation is enforced statically.
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: uv run pytest -q -o log_cli=false tests/qualification/test_phase8_live_workflow.py tests/qualification/test_phase8_evidence.py -x
        status: pass
    human_judgment: false
duration: 13m 58s
completed: 2026-09-14
status: complete
---

# Phase 08 Plan 08: Live PostgreSQL and Amazon S3 Qualification Summary

**Protected real-service qualification now binds the frozen obstore topology to one clean candidate revision, exact bounded cleanup, and release-candidate-only evidence.**

## Performance

- **Duration:** 13m 58s
- **Started:** 2026-09-14T00:50:16Z
- **Completed:** 2026-09-14T01:04:14Z
- **Tasks:** 3/3
- **Files modified:** 5

## Accomplishments

- Added a Phase 8-owned runner for the exact live PostgreSQL, native obstore S3, and remote-topology modules. It records a reviewed source digest, obstore metadata, standard AWS identity, fixed test selection, and terminal cleanup state without emitting credentials or payload data.
- Extended qualification fixtures with phase-specific q8 schema/prefix naming while retaining Phase 5 compatibility, exact owner-marker checks, and existing page/object/byte/upload cleanup caps.
- Added a protected manual release-candidate workflow with detached 40-character SHA verification and a separately retained scheduled drift diagnostic that cannot produce release-eligible evidence.

## Task Commits

1. **Task 1: Rebind the frozen live runner to Phase 8 exact source and obstore evidence** - `158e648` (feat)
2. **Task 2: Preserve exact bounded live cleanup and secret-safe evidence** - `6a5bc6a` (feat)
3. **Task 3: Orchestrate protected release-candidate and scheduled drift evidence** - `928fc42` (feat)

## Files Created/Modified

- `tools/run_phase8_qualification.py` - Exact-source, no-skip live runner with sanitized evidence and atomic writes.
- `tests/qualification/conftest.py` - Preserves existing marker/cap cleanup while safely accepting one Phase 8 namespace.
- `tests/qualification/test_phase8_evidence.py` - Runner, source-binding, cleanup, redaction, and scheduled-role contracts.
- `.github/workflows/live_qualification.yml` - Protected exact-SHA RC and isolated scheduled diagnostic orchestration.
- `tests/qualification/test_phase8_live_workflow.py` - Static secret-boundary, action-pin, trigger, artifact, and role-separation tests.

## Decisions Made

- A scheduled real-service pass remains diagnostic: only the `release_candidate` role can emit `QUALIFIED` evidence.
- Phase 8 qualification derives q8 resource names from an exact `phase8-` run ID, keeping Phase 5 fixture behavior available to the frozen suite.
- The runner fails closed when a listed current or future reviewed qualification source is absent or dirty; an earlier artifact cannot prove a later candidate.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] Added release-role binding to live evidence.**

- **Found during:** Task 3
- **Issue:** A scheduled drift workflow could otherwise write a clean-looking live envelope with no intrinsic release-role distinction.
- **Fix:** Added exact `release_candidate` and `scheduled_diagnostic` roles; only the former can yield `QUALIFIED`.
- **Files modified:** `tools/run_phase8_qualification.py`, `tests/qualification/test_phase8_evidence.py`
- **Verification:** Full Phase 8 workflow/evidence contract suite passed.
- **Committed in:** `928fc42`

**2. [Rule 1 - Bug] Required the release role in the runner's final qualification predicate.**

- **Found during:** Task 3 scheduled-diagnostic regression test
- **Issue:** The new evidence validator rejected a scheduled `QUALIFIED` record, but the runner initially calculated qualification before considering the role.
- **Fix:** Included the exact `release_candidate` role in the final predicate.
- **Files modified:** `tools/run_phase8_qualification.py`
- **Verification:** Scheduled clean probes now write `NOT_QUALIFIED` diagnostic evidence and tests pass.
- **Committed in:** `928fc42`

**3. [Rule 3 - Blocking] Corrected the Phase 8 activity record after SDK state advancement.**

- **Found during:** Planning-state update
- **Issue:** The SDK advanced the plan counter but retained Plan 02 as the last recorded activity.
- **Fix:** Updated only the stale activity date and description to name completed Plan 08.
- **Files modified:** `.planning/STATE.md`
- **Verification:** State now names Plan 08 while preserving the SDK-selected `Plan: 8 of 12` position.
- **Committed in:** Plan metadata commit

---

**Total deviations:** 3 auto-fixed (1 Rule 1, 1 Rule 2, 1 Rule 3).
**Impact on plan:** Both fixes enforce the planned non-substitution boundary; no production storage, lifecycle coordination, endpoint override, or integrity semantics changed.

## Known Stubs

None.

## Issues Encountered

None. The local no-credential probe intentionally returned exit code 2 and sanitized `UNAVAILABLE`/`NOT_ATTEMPTED` evidence; it is not live qualification.

## User Setup Required

The actual BACK-05 service run remains Plan 08-11 work. A repository administrator must provide the protected `live-qualification` environment with the four named configuration values, an authoritative Amazon S3 bucket/region, and a least-privilege PostgreSQL service. Mocks, endpoint overrides, and scheduled diagnostics cannot substitute for it.

## Next Phase Readiness

- Plan 08-09 and Plan 08-10 can statically validate the workflow and collect only its fixed RC artifact by exact run ID.
- BACK-05 remains pending until Plan 08-11 obtains `QUALIFIED` plus `CLEAN` evidence for the exact final candidate revision.

## Self-Check: PASSED

- All five planned runner, fixture, workflow, and test artifacts exist on disk.
- Task commits `158e648`, `6a5bc6a`, and `928fc42` resolve in Git history.
