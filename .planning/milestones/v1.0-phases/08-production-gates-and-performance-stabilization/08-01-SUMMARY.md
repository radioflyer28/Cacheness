---
phase: 08-production-gates-and-performance-stabilization
plan: 01
subsystem: qualification
tags: [release-evidence, deterministic-gates, json-validation, source-identity]
requires:
  - phase: 07.1-obstore-payload-participant-unification
    provides: Fixed all-mode deterministic contract verifier for the qualified storage architecture
provides:
  - Exact-commit deterministic evidence envelopes with fail-closed parsing
  - Truthful class-by-class local qualification reporting
affects: [phase-08-release-verifier, packaging-gates, platform-gates, live-service-qualification]
actuals:
  tokens: 9740
  tasks: 2
  commits: 5
tech-stack:
  added: []
  patterns:
    - Canonical bounded JSON evidence with exact allow-lists
    - Exact revision and reviewed-source-digest identity validation
    - Non-qualifying evidence-class reporting through an explicit exit code
key-files:
  created:
    - tools/phase8_evidence.py
    - tools/run_phase8_local_gates.py
    - tests/test_phase8_release_tracer.py
  modified: []
key-decisions:
  - "A deterministic PASS evidences integrity, recovery, and progress only; performance remains NOT_QUALIFIED."
  - "A passing deterministic gate exits 2 while external evidence classes are UNAVAILABLE, preventing a partial release pass."
  - "Later release tooling must revalidate both the exact Git revision and reviewed-source SHA-256 digest."
patterns-established:
  - "Evidence producers validate one class and never aggregate release qualification."
  - "Terminal evidence states require an exact matching claim-category map."
requirements-completed: [QUAL-03, QUAL-04]
coverage:
  - id: D1
    description: Deterministic Phase 07.1 contract execution yields an exact-commit canonical PASS envelope.
    requirement: QUAL-04
    verification:
      - kind: e2e
        ref: uv run --isolated --all-extras --group dev --frozen python tools/run_phase8_local_gates.py deterministic --output /private/tmp/phase8-deterministic-evidence.json
        status: pass
      - kind: unit
        ref: tests/test_phase8_release_tracer.py
        status: pass
    human_judgment: false
  - id: D2
    description: Malformed, contradictory, stale, forged, unavailable, and unqualified evidence cannot become release proof.
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: tests/test_phase8_release_tracer.py
        status: pass
    human_judgment: false
duration: 16m
completed: 2026-09-13
status: complete
---

# Phase 08 Plan 01: Production Evidence Tracer Summary

**A fixed Phase 07.1 lifecycle contract now produces exact-commit deterministic evidence while making every unproduced release-evidence class visibly non-qualifying.**

## Performance

- **Duration:** 16m
- **Started:** 2026-09-13T22:56:35Z
- **Completed:** 2026-09-13T23:12:20Z
- **Tasks:** 2/2
- **Files modified:** 3

## Accomplishments

- Added a strict canonical evidence envelope for deterministic, packaging, platform, coverage, structural, controlled-performance, and live-service classes.
- Added the canonical local Phase 8 deterministic gate, bound to the fixed Phase 07.1 all-mode verifier and exact reviewed source identity.
- Hardened evidence parsing against duplicate keys, unknown fields, unsafe content, oversized input, forged source digests, terminal-state contradictions, and ADR claim-category swaps.
- Made the local command report all seven evidence classes and return a distinct unavailable-evidence exit code when external release prerequisites have not run.

## Task Commits

1. **Task 1: Carry one deterministic contract run into exact-commit evidence** - `f12b08d` (test), `ceaed58` (feat)
2. **Task 2: Harden evidence parsing and truthful non-claim states** - `5d4b0f0` (test), `8333dd7` (test), `1c2700d` (feat)

## Files Created/Modified

- `tools/phase8_evidence.py` - Canonical, bounded evidence envelope validation and exact source identity verification.
- `tools/run_phase8_local_gates.py` - Fixed deterministic gate plus truthful class-by-class reporting.
- `tests/test_phase8_release_tracer.py` - End-to-end and mutation contracts for pass, unavailable, unqualified, malformed, and forged evidence.

## Decisions Made

- Deterministic evidence records only integrity, recovery, and progress; it never promotes performance or another evidence class.
- A fully passing local deterministic run returns exit code `2` while external evidence remains unavailable, rather than manufacturing a whole-release success.
- Evidence consumers must independently bind an envelope to the required Git revision and source digest before using it as release proof.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Restored deterministic child mocking in the local gate**
- **Found during:** Task 2
- **Issue:** The default child callback was captured at function definition, so test monkeypatches still ran the full inherited verifier.
- **Fix:** Resolve the default fixed child callback inside `run_deterministic` while retaining the literal production command.
- **Files modified:** `tools/run_phase8_local_gates.py`
- **Verification:** `uv run pytest -q -o log_cli=false tests/test_phase8_release_tracer.py -x`
- **Committed in:** `1c2700d`

**2. [Rule 3 - Blocking] Repaired the legacy Phase 8 plan position manually**
- **Found during:** Plan state update
- **Issue:** The existing `STATE.md` Current Position did not contain the plan-counter fields required by `state.advance-plan`.
- **Fix:** Recorded Plan 01 completion and the next plan position directly after the SDK recorded metrics, decisions, session, roadmap, and requirement updates.
- **Files modified:** `.planning/STATE.md`
- **Verification:** State now identifies Plan 02 of 12 as the in-progress position.
- **Committed in:** Plan metadata commit

---

**Total deviations:** 2 auto-fixed (1 Rule 1, 1 Rule 3)
**Impact on plan:** The fixes preserve the fixed production command, keep contract tests deterministic, and restore accurate plan tracking; no storage behavior or topology guarantee changed.

## Known Stubs

None.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration is required for the deterministic tracer. PostgreSQL and AWS S3 qualification remain explicitly unavailable until their later Phase 8 gates run.

## Next Phase Readiness

- Later Phase 8 producers can use the shared strict envelope vocabulary without aggregating qualification themselves.
- Packaging, platform, coverage, structural, controlled-performance, and live-service evidence remain UNAVAILABLE and cannot satisfy release requirements until their dedicated plans produce exact-commit proof.

## Self-Check: PASSED

- Required qualification tools, tracer tests, and summary exist on disk.
- All five Task 1/Task 2 commits are present in Git history.

---
*Phase: 08-production-gates-and-performance-stabilization*
*Completed: 2026-09-13*
