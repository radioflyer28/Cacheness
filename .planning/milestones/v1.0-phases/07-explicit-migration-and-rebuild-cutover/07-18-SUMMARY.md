---
phase: 07-explicit-migration-and-rebuild-cutover
plan: 18
subsystem: testing
tags: [pytest, verifier, migration, threat-model, fail-closed]
requires:
  - phase: 07-17
    provides: Digest-bound, confidential migration-plan evidence
provides:
  - Exact, AST-validated pytest selectors for every fixed Phase 7 claim
  - An independently owned 38-row gap-plan threat inventory and exact 84-ID union
  - Fail-closed diagnostics for missing, renamed, malformed, or duplicate evidence
affects: [phase-7-validation, phase-8-qualification, offline-maintenance]
actuals:
  tokens: 13849
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Fixed claim evidence uses repository-relative path::test_name selectors validated by AST before pytest execution
    - Gap-plan threat rows use a literal ownership oracle rather than discovery or filename proxies
key-files:
  created: []
  modified:
    - tools/verify_phase7_contracts.py
    - tests/test_phase7_contract_verifier.py
key-decisions:
  - "Selector validation parses test source statically, so an unrelated passing test cannot satisfy a removed evidence node."
  - "The verifier executes a deterministic de-duplicated selector union and preserves the PostgreSQL quick-mode boundary."
  - "The accepted invisible, unattributed pre-checkpoint orphan remains a bounded ADR 0001 progress limit, not an atomicity claim."
patterns-established:
  - "Claim manifests must bind concrete tests and validate their symbols before subprocess execution."
  - "Plan-declared threats are cross-checked against an independently maintained per-plan ownership inventory."
requirements-completed: [MIGR-03, MIGR-04, MIGR-05, MIGR-06]
coverage:
  - id: D1
    description: Fixed Phase 7 requirement, decision, threat, assumption, and prohibition claims execute only validated exact test selectors.
    requirement: MIGR-05
    verification:
      - kind: unit
        ref: tests/test_phase7_contract_verifier.py#test_fixed_manifest_requires_exact_path_and_test_name_selectors
        status: pass
    human_judgment: false
  - id: D2
    description: Removed, renamed, duplicate, and unowned contract evidence fails before pytest can report a false green.
    requirement: MIGR-05
    verification:
      - kind: unit
        ref: tests/test_phase7_contract_verifier.py#test_fixed_manifest_rejects_removed_mapped_test_function_while_file_remains
        status: pass
      - kind: unit
        ref: tests/test_phase7_contract_verifier.py#test_fixed_manifest_rejects_renamed_mapped_test_function_while_file_remains
        status: pass
    human_judgment: false
duration: 20min
completed: 2026-09-11
status: complete
---

# Phase 07 Plan 18: Exact Contract Verifier Summary

**Phase 7 completion evidence now binds each claim to a validated test function and rejects changed or unowned evidence before running pytest.**

## Performance

- **Duration:** 20min
- **Completed:** 2026-09-11T17:53:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Replaced filename-level mappings with exact `tests/...py::test_name` selectors for all requirements, decisions, threat rows, flagged assumptions, and fixed prohibitions.
- Added the literal 38-row threat oracle for Plans 07-12 through 07-19, yielding a validated 84-ID Phase 7 inventory with no omission, duplicate, or unexpected-row tolerance.
- Validated selector syntax, inventory membership, and source symbols before subprocess execution; quick and all modes now send only a deterministic selector union to pytest.
- Added adversarial tests for removed/renamed functions, gap-row mutation, dependency-command rules, false obstore claims, and selector forwarding.

## Task Commits

1. **Task 1: Convert fixed claim mappings to exact validated selectors** - `6324fa0` (`test`), `f7ee9e1` (`feat`)
2. **Task 2: Fail closed when a mapped test is removed or renamed** - `ea2340b` (`test`), `07dfc95` (`feat`)

## Verification

- `uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false tests/test_phase7_contract_verifier.py` — 21 passed
- `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --quick` — PASS
- `uv run --isolated --all-extras --group dev --frozen ruff check tools/verify_phase7_contracts.py tests/test_phase7_contract_verifier.py` — PASS

## Decisions Made

- AST inspection is used for deterministic local symbol validation; it does not execute a potentially changed test module merely to discover its names.
- The exact threat oracle reads only the eight planned gap threat tables and retains the accepted scope: no production obstore adoption, no new authority, and no stronger orphan-reclamation guarantee.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected malformed Phase 7 context-path handling**
- **Found during:** Task 1 and Task 2 verification
- **Issue:** The one-item context tuple lacked a comma and was later combined as a path, causing character-by-character artifact checks and a tuple path error.
- **Fix:** Made the context path a true tuple and used its sole repository-relative member at the read boundary.
- **Files modified:** `tools/verify_phase7_contracts.py`
- **Verification:** Exact-manifest validation and selector-forwarding self-test pass.
- **Committed in:** `f7ee9e1`, `07dfc95`

**Total deviations:** 1 auto-fixed (Rule 1).
**Impact on plan:** Correctness-only verifier repair; no lifecycle, storage, or external behavior changed.

## Issues Encountered

None.

## Next Phase Readiness

Plan 19 can record quick/all verifier results as exact executable evidence. The verifier deliberately preserves Phase 8 boundaries for live PostgreSQL/S3, Windows, supported-Python qualification, performance distributions, obstore adoption, and perfect orphan reclamation.

## Self-Check: PASSED

- Confirmed both modified files exist.
- Confirmed all four task commits exist.
- Confirmed no known stubs, skipped tests, or unrun required verification remain.
