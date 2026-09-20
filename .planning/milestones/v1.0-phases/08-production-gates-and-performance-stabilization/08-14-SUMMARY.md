---
phase: 08-production-gates-and-performance-stabilization
plan: 14
subsystem: release-qualification
tags: [github-release, evidence, sha256, qualification, pytest]
requires:
  - phase: 08-13
    provides: controlled-Linux runner preflight retained for the future performance seed
provides:
  - Current-release collection and aggregation that exclude controlled performance while retaining an explicit QUAL-06 nonclaim
  - Fixed D-23 verifier inventories, public qualification boundary, and immutable publication controller
affects: [08-11, 08-12, release-qualification, SEED-006]
actuals:
  tokens: 17379
  tasks: 3
  commits: 4
tech-stack:
  added: []
  patterns:
    - Literal current-versus-deferred evidence inventories
    - Approval-bound immutable draft publication with post-publication incident records
key-files:
  created: []
  modified:
    - tools/verify_phase8_release.py
    - tools/verify_phase8_contracts.py
    - tests/qualification/test_phase8_release.py
    - tests/test_phase8_contract_verifier.py
    - docs/RELEASE_QUALIFICATION.md
key-decisions:
  - "Controlled performance is DEFERRED and NOT_QUALIFIED under QUAL-06/SEED-006; macOS diagnostics cannot substitute for Linux evidence."
  - "Current release evidence remains exact-SHA quality plus protected live-service qualification; all other non-deferred classes remain mandatory."
  - "Publication requires a digest-bound approval, exact draft assets, and a read-only immutable-state proof."
patterns-established:
  - "Release tooling must reject rather than ignore extra, diagnostic, or altered evidence."
  - "A post-publication mismatch becomes a NOT_QUALIFIED incident record and never triggers a repair attempt."
requirements-completed: [BACK-05, QUAL-01, QUAL-02, QUAL-03, QUAL-04, QUAL-05, QUAL-07]
coverage:
  - id: D1
    description: Current quality/live-only collection, exact QUAL-06 deferral, and rejection of timing substitutions
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: tests/qualification/test_phase8_release.py#test_required_release_collection_excludes_deferred_performance
        status: pass
      - kind: unit
        ref: tests/qualification/test_phase8_release.py#test_deferred_performance_artifacts_and_macos_diagnostics_are_rejected
        status: pass
    human_judgment: false
  - id: D2
    description: Fixed D-23 inventories and public controlled-performance nonclaim
    requirement: QUAL-05
    verification:
      - kind: unit
        ref: tests/test_phase8_contract_verifier.py#test_fixed_manifest_covers_deferred_performance_decision
        status: pass
      - kind: unit
        ref: tests/test_phase8_contract_verifier.py#test_release_documentation_preserves_seed006_nonclaim
        status: pass
    human_judgment: false
  - id: D3
    description: Approval-bound draft publication and immutable remote-state verification
    requirement: BACK-05
    verification:
      - kind: unit
        ref: tests/qualification/test_phase8_release.py#test_publish_requires_exact_approved_prepublication_digest
        status: pass
      - kind: unit
        ref: tests/qualification/test_phase8_release.py#test_publish_and_verify_requires_immutable_exact_remote_state
        status: pass
      - kind: unit
        ref: tests/qualification/test_phase8_release.py#test_postpublication_mismatch_records_unqualified_incident
        status: pass
    human_judgment: false
duration: 20min
completed: 2026-09-15
status: complete
---

# Phase 08 Plan 14: Release Evidence Deferral and Publication Controller Summary

**Exact-SHA release qualification now retains all quality and live-service proof, records the controlled-Linux performance nonclaim explicitly, and cannot publish without immutable, approval-bound evidence.**

## Performance

- **Duration:** 20 min
- **Started:** 2026-09-15T10:46:21-04:00
- **Completed:** 2026-09-15T11:06:02-04:00
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- Collected only the current quality and protected-live evidence set; aggregates require the closed D-23/QUAL-06/SEED-006 `DEFERRED` and `NOT_QUALIFIED` record.
- Bound Plan 14, D-23, seven current requirements, and four new threats into the fixed verifier and documented macOS as diagnostic only.
- Completed the draft, approval, publication, and immutable-state controller before exact-candidate collection, with an incident record for any postpublication mismatch.

## Task Commits

1. **Task 1: Cut exact-SHA collection and aggregation over to the non-deferred evidence set** - `1dfb7cd` (test), `1346474` (feat)
2. **Task 2: Bind the revised evidence boundary into the fixed verifier and public qualification contract** - `1447a88` (feat)
3. **Task 3: Finish the publication controller before selecting the exact candidate** - `e924d4c` (feat)

## Files Created/Modified

- `tools/verify_phase8_release.py` - Current evidence inventory, exact draft controller, approval binding, and read-only immutable release verification.
- `tests/qualification/test_phase8_release.py` - Adversarial contracts for preflight, exact assets/digests, approval, immutable state, and incident handling.
- `tools/verify_phase8_contracts.py` - Fixed Plan 14/D-23/current-requirement/threat inventory and deferred-status rendering.
- `tests/test_phase8_contract_verifier.py` - Literal verifier and documentation nonclaim tests.
- `docs/RELEASE_QUALIFICATION.md` - Public D-23/SEED-006, macOS diagnostic-only, retention, and asset-boundary contract.

## Decisions Made

- Current release collection never dispatches controlled performance; QUAL-06 stays explicitly `DEFERRED` / `NOT_QUALIFIED` for SEED-006.
- The digest supplied by the approving operator must match the canonical prepublication report before the draft can transition.
- Remote publication divergence is recorded as an incident and leaves qualification false; the controller does not attempt repair.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected publication CLI condition composition**
- **Found during:** Task 3
- **Issue:** New command-specific input validation initially omitted Boolean connectors.
- **Fix:** Restored explicit fail-closed argument guards before test execution.
- **Files modified:** `tools/verify_phase8_release.py`
- **Verification:** Focused publication-controller tests and `py_compile` passed.
- **Committed in:** `e924d4c`

**2. [Rule 1 - Bug] Persisted verified uploaded state in the prepublication record**
- **Found during:** Task 3
- **Issue:** The report retained asset names and digests but omitted the required API-confirmed `uploaded` state.
- **Fix:** Added and validated the state for every report asset before it can authorize publication.
- **Files modified:** `tools/verify_phase8_release.py`, `tests/qualification/test_phase8_release.py`
- **Verification:** All 21 release qualification tests passed.
- **Committed in:** `e924d4c`

**3. [Rule 2 - Missing Critical] Carried the approval digest into the future publication invocation**
- **Found during:** Task 3 handoff review
- **Issue:** The existing Plan 08-12 command omitted the required approval-bound report digest.
- **Fix:** Added the exact `--approved-report-sha256` argument to the future irreversible publication command.
- **Files modified:** `.planning/phases/08-production-gates-and-performance-stabilization/08-12-PLAN.md`
- **Verification:** The focused approval-digest adversarial test passes.
- **Committed in:** pending metadata amendment

**Total deviations:** 3 auto-fixed issues (2 Rule 1, 1 Rule 2).

**Impact on plan:** Both fixes enforce the planned fail-closed publication contract; no scope expansion occurred.

## Issues Encountered

- The sandbox initially denied read access to the existing `uv` cache. Re-running the same planned verification with approved toolchain access resolved it; no dependency changes were made.

## User Setup Required

None for this implementation. Plan 08-11/08-12 still requires the documented protected live-service and release-operator checkpoints.

## Next Phase Readiness

Plan 08-11 can collect exact-SHA quality and protected live-service evidence without a controlled Linux runner. Controlled performance remains deferred to SEED-006; macOS is not qualifying evidence. Real PostgreSQL/Amazon S3 evidence and immutable release authority remain mandatory external prerequisites.

## Self-Check: PASSED

- Confirmed task commits `1dfb7cd`, `1346474`, `1447a88`, and `e924d4c` exist.
- Confirmed the modified release tooling, verifier, tests, and qualification documentation exist and the complete Plan 08-14 verification set passes.
