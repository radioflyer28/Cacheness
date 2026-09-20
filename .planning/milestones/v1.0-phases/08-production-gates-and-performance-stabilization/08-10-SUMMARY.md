---
phase: 08-production-gates-and-performance-stabilization
plan: 10
subsystem: release-qualification
tags: [qualification, github-cli, sha256, github-actions, evidence, pytest]
requires:
  - phase: 08-07
    provides: controlled-performance evidence schema and benchmark harness
  - phase: 08-08
    provides: protected live-service evidence schema and runner
  - phase: 08-09
    provides: fixed quality workflows and local evidence producers
provides:
  - Closed Phase 8 plan, decision, threat, source, workflow, and selector verifier
  - Exact-SHA GitHub workflow dispatch, run-ID collection, and bounded artifact validation
  - Same-revision release-evidence aggregation and immutable-publication inspection
affects: [phase-08, release-collection, release-publication, qualification]
actuals:
  tokens: 19805
  tasks: 3
  commits: 10
tech-stack:
  added: []
  patterns:
    - Literal reviewed inventories with AST selector/source audits
    - Exact-SHA workflow provenance by new run ID and fixed artifact name
    - Bounded unavailable evidence remains a release-blocking nonclaim
key-files:
  created:
    - tools/verify_phase8_contracts.py
    - tools/verify_phase8_release.py
    - tests/test_phase8_contract_verifier.py
    - tests/qualification/test_phase8_release.py
  modified: []
key-decisions:
  - "Treat UNAVAILABLE packaging and platform envelopes as explicit release blockers, never generic failures or passes."
  - "Require one fresh exact-SHA workflow run and fixed artifact name per evidence class; latest-run lookup is prohibited."
  - "Keep final publication inspection read-only and require immutable release state plus matching SHA-256 assets."
patterns-established:
  - "Qualification evidence: record only bounded provenance and digest fields; never persist credentials, payloads, or unbounded logs."
requirements-completed: []
coverage:
  - id: D1
    description: "Fixed Phase 8 inventory verifier rejects omitted selectors, source regressions, and false qualification states."
    verification:
      - kind: unit
        ref: "tests/test_phase8_contract_verifier.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "Exact-SHA workflow collection, release aggregation, and immutable-publication inspection reject substituted evidence."
    verification:
      - kind: unit
        ref: "tests/qualification/test_phase8_release.py"
        status: pass
    human_judgment: false
duration: 55min
completed: 2026-09-14
status: complete
---

# Phase 08 Plan 10: Fixed Release Contract and Exact-SHA Evidence Summary

**Phase 8 now has a fail-closed fixed verifier and exact-SHA release-evidence collector that distinguishes unavailable prerequisites from invalid evidence without changing storage lifecycle behavior.**

## Performance

- **Duration:** 55 min
- **Started:** 2026-09-14T03:02:53Z
- **Completed:** 2026-09-14T03:58:14Z
- **Tasks:** 3/3
- **Files modified:** 4

## Accomplishments

- Bound literal Plan 01–12, D-01–D-22, all Phase 8 threats, requirements, sources, workflows, and named selectors to executable static and AST evidence.
- Added strict lower-case 40-character SHA dispatch, fresh run-ID provenance, fixed-name artifact download, envelope validation, canonical same-identity aggregation, and read-only immutable-release verification.
- Preserved truthful qualification boundaries: the final all-mode run reports packaging/platform/controlled/live as `UNAVAILABLE` and Windows as `NOT_QUALIFIED`, then exits 2 rather than claiming release readiness.

## Verification

- `uv run pytest -q -o log_cli=false tests/qualification/test_phase8_release.py tests/test_phase8_contract_verifier.py -x` — 23 passed.
- `uv run ruff check tools/verify_phase8_contracts.py tools/verify_phase8_release.py tests/test_phase8_contract_verifier.py tests/qualification/test_phase8_release.py` — passed.
- `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase8_contracts.py --all` — local classes completed; truthful blocking result: packaging/platform/controlled/live `UNAVAILABLE`, Windows `NOT_QUALIFIED`, exit 2.
- The exact non-live branch-coverage child completed successfully; expected platform and TensorFlow skips remained non-qualifying and did not substitute for evidence.

## Task Commits

1. **Task 1: Bind the complete fixed phase contract to exact selectors and source audits** — `236a328` (test), `abeb7b2` (feat)
2. **Task 2: Dispatch exact-SHA workflows and collect fixed artifacts by run ID** — `501d1a3` (test), `819411f` (feat)
3. **Task 3: Aggregate exact-commit evidence and verify immutable publication state** — `ee1d66d` (test), `0a46f82` (feat)
4. **Qualification-tool corrections** — `02c61d0`, `9d70271`, `01d84e1`, `1037909` (fix)

## Files Created/Modified

- `tools/verify_phase8_contracts.py` — reviewed fixed inventory, AST architecture audits, and bounded local-status reporting.
- `tools/verify_phase8_release.py` — exact-SHA dispatch/collection, canonical aggregate, and read-only release inspection.
- `tests/test_phase8_contract_verifier.py` — adversarial inventory and unavailable-evidence regressions.
- `tests/qualification/test_phase8_release.py` — fake-CLI adversarial workflow, aggregate, and publication assertions.

## Decisions Made

- A local `UNAVAILABLE` envelope is valid blocking evidence only when its bounded class and terminal status exactly match the reviewed gate; it can never make a release pass.
- GitHub orchestration records a newly observed run ID and revalidates its workflow, event, SHA, completion, and artifact name before aggregation.
- Publication verification is inspection-only: exact tag target, published immutable release, exact uploaded asset allow-list, and SHA-256 checks must all pass.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] All mode now invokes the complete local evidence set in its frozen qualification environment.**

- **Found during:** Tasks 1–3 integration verification.
- **Issue:** The initial all-mode implementation stopped after its self-tests and inherited a base environment that omitted optional test dependencies.
- **Fix:** Added fixed child-gate execution and used the frozen all-extras/dev environment for those children.
- **Files modified:** `tools/verify_phase8_contracts.py`, `tests/test_phase8_contract_verifier.py`.
- **Verification:** Focused suite passes; full non-live coverage child completed successfully.
- **Committed in:** `02c61d0`, `9d70271`.

**2. [Rule 1 - Bug] Preserve valid unavailable package/platform prerequisites as nonclaims.**

- **Found during:** Final all-mode verification on Darwin/3.13.
- **Issue:** The macOS 3.13 row is outside the required 3.11/3.14 boundary matrix and TensorFlow is unavailable on Darwin arm64. Their producers correctly emitted `UNAVAILABLE`, but the verifier reduced those envelopes to an opaque failure and suppressed the report.
- **Fix:** Added bounded, exact-class envelope recognition so unavailable packaging/platform prerequisites are rendered and retain the release-blocking exit code 2; malformed or failed evidence still exits 1.
- **Files modified:** `tools/verify_phase8_contracts.py`, `tests/test_phase8_contract_verifier.py`.
- **Verification:** Adversarial regression suite and final exact all-mode command pass with the documented blocking statuses.
- **Committed in:** `01d84e1`, `1037909`.

**Total deviations:** 2 auto-fixed (1 Rule 1, 1 Rule 2).

**Impact on plan:** Changes are bounded to qualification tooling and reporting. They add no storage lifecycle coordinator, no lock/lease/queue, no alternate authority, no backend behavior, and no integrity-digest change.

## Issues Encountered

- The full coverage suite is intentionally longer than focused contract tests because it executes the non-live repository suite with branch coverage. It completed successfully; its captured warnings and expected platform/TensorFlow skips do not establish qualifying substitute evidence.

## User Setup Required

None for this plan. Controlled runner, protected live services, exact GitHub workflow dispatch, and immutable-release permissions remain explicit later prerequisites.

## Next Phase Readiness

- Plan 11 can collect exact-run artifacts only when its controlled runner and protected live environment are configured; this plan intentionally does not simulate either one.
- `BACK-05` and `QUAL-01` through `QUAL-07` remain unchecked here because a self-tested collector is not release evidence. Their terminal state depends on subsequent exact-run and publication plans.

## Self-Check: PASSED

Verified the four task artifacts, this summary, all ten task commits, and no tracked-file deletions across the plan range.

---
*Phase: 08-production-gates-and-performance-stabilization*
*Completed: 2026-09-14*
