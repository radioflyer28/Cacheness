---
phase: 07-explicit-migration-and-rebuild-cutover
plan: 11
subsystem: testing
tags: [phase-7, verifier, migration, pytest, ruff, safety-contracts]
requires:
  - phase: 07-explicit-migration-and-rebuild-cutover
    provides: Phase 7 migration, rebuild, and storage-safety contract tests and validation artifacts
provides:
  - Fixed literal verifier inventory for all Phase 7 production, test, and planning artifacts
  - Adversarial contract audits for migration authority, security, documented topology limits, and Plan 01 prohibitions
  - Reproducible non-live final gate with explicit Phase 8 live-service exclusions
affects: [phase-8-performance, release-validation, migration-maintenance]
actuals:
  tokens: 14557
  tasks: 2
  commits: 3
tech-stack:
  added: []
  patterns:
    - Fixed literal test/path inventory with fail-closed manifest validation
    - Verifier self-tests mutate fixtures to prove its failure behavior
key-files:
  created:
    - tools/verify_phase7_contracts.py
    - tests/test_phase7_contract_verifier.py
    - .planning/phases/07-explicit-migration-and-rebuild-cutover/deferred-items.md
  modified:
    - .planning/phases/07-explicit-migration-and-rebuild-cutover/07-VALIDATION.md
key-decisions:
  - "The Phase 7 final gate uses a literal bounded inventory and reports live PostgreSQL/AWS S3 modules as NOT RUN / NOT QUALIFIED Phase 8 work."
  - "The observed clear/delete concurrency conflict remains truthful pre-existing evidence; no lifecycle coordination or storage race patch was added."
patterns-established:
  - "Safety assertions are paired with adversarial verifier tests, including self-tests that demonstrate failure on an omitted fixed-inventory member."
requirements-completed: [MIGR-03, MIGR-04, MIGR-05, MIGR-06]
coverage:
  - id: D1
    description: Fixed Phase 7 contract verifier closes migration, decision, security, assumption, and prohibition contracts.
    requirement: MIGR-03
    verification:
      - kind: unit
        ref: tests/test_phase7_contract_verifier.py#test_verifier_all_reports_fixed_inventory_pass
        status: pass
      - kind: other
        ref: uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --all
        status: pass
    human_judgment: false
  - id: D2
    description: Final validation artifact records a deterministic non-live gate and preserves explicitly unqualified topology and performance work.
    requirement: MIGR-06
    verification:
      - kind: other
        ref: .planning/phases/07-explicit-migration-and-rebuild-cutover/07-VALIDATION.md
        status: pass
    human_judgment: false
duration: 27min
completed: 2026-09-10
status: complete
---

# Phase 07 Plan 11: Fixed Contract Verification Summary

**A fixed, self-testing Phase 7 verifier now closes every declared migration and safety contract through adversarial tests while keeping live PostgreSQL, AWS S3, Windows, and Phase 8 performance explicitly unqualified.**

## Performance

- **Duration:** 27 min
- **Started:** 2026-09-10T04:36:46Z
- **Completed:** 2026-09-10T05:03:17Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added a fail-closed, literal-inventory verifier that checks all `MIGR-03` through `MIGR-06` contracts, decisions `D01` through `D22`, requirements `RQ01` through `RQ04`, all 46 declared threats, flagged assumptions, and every Plan 01 prohibition.
- Added adversarial verifier self-tests that prove missing fixed-inventory members and unsafe source fixtures fail, then ran the quick and all-environment gates successfully.
- Recorded the final deterministic non-live evidence: 1,311 tests with zero failures or errors, 9 expected platform/optional skips, and scoped Ruff passing; live PostgreSQL/AWS S3, Windows, and Phase 8 performance remain explicitly outside Phase 7 qualification.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add failing fixed-verifier tests** - `7ecc4e0` (`test`)
2. **Task 1: Implement the fixed Phase 7 contract verifier** - `3219924` (`feat`)
3. **Task 2: Bound the non-live final verification inventory** - `3a993bb` (`fix`)

## Files Created/Modified

- `tools/verify_phase7_contracts.py` - Fixed manifest, static audits, source-fixture checks, deterministic test orchestration, and scoped lint gate.
- `tests/test_phase7_contract_verifier.py` - Adversarial self-tests for positive contracts, eight prohibitions, manifest completeness, and verifier failure behavior.
- `.planning/phases/07-explicit-migration-and-rebuild-cutover/07-VALIDATION.md` - Final Phase 7 evidence ledger and qualification boundaries.
- `.planning/phases/07-explicit-migration-and-rebuild-cutover/deferred-items.md` - Truthful record of the one-off pre-existing concurrency observation.

## Decisions Made

- The verifier owns a fixed literal inventory rather than discovering a broad suite dynamically. Its `--all` gate excludes only the three named Phase 8 live-service modules and prints their NOT RUN / NOT QUALIFIED status; they are neither green skips nor Phase 7 evidence.
- The temporary `clear`/`delete` snapshot conflict observed before the inventory correction was not changed. ADR 0001 and the Phase 7 Wave 1 boundary prohibit reopening lifecycle coordination; a repeated fixed non-live run passed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Verification boundary] Bounded the final non-live inventory to its declared scope**
- **Found during:** Task 2 (Run final contract gate and record validation)
- **Issue:** The first final-gate inventory attempted three Phase 8 live PostgreSQL/AWS S3 modules, producing missing-fixture errors despite the Phase 7 non-live qualification boundary.
- **Fix:** Replaced the broad exclusion with the exact three literal Phase 8 modules, added an assertion that they remain NOT RUN / NOT QUALIFIED, and added the verifier's own test module to the fixed inventory.
- **Files modified:** `tools/verify_phase7_contracts.py`, `tests/test_phase7_contract_verifier.py`, `07-VALIDATION.md`, `deferred-items.md`
- **Verification:** `uv run --isolated --all-extras --group dev --frozen python tools/verify_phase7_contracts.py --all` passed.
- **Committed in:** `3a993bb` (part of Task 2)

---

**Total deviations:** 1 auto-fixed (Rule 1)
**Impact on plan:** The correction makes the final gate match its stated non-live scope without weakening any Phase 7 positive contract or prohibition.

## Issues Encountered

- One pre-correction final run also observed `test_clear_and_delete_converge_after_an_exact_snapshot` raise `CacheBlobLifecycleConflictError`. This is preserved in `deferred-items.md` as pre-existing evidence; the bounded verifier fix did not alter lifecycle or storage concurrency code. The repeated fixed non-live gate passed.

## Known Stubs

None. The created and modified plan files contain no rendering-path hardcoded empty values, placeholder content, or skipped tests.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 7 now has a reproducible final safety gate and validation ledger. Phase 8 must independently qualify live PostgreSQL/AWS S3 topology, Windows behavior, and performance budgets; this plan makes none of those integrity, availability, or progress guarantees.

## Self-Check: PASSED

- Confirmed `07-11-SUMMARY.md` exists.
- Confirmed task commits `7ecc4e0`, `3219924`, and `3a993bb` exist in repository history.
- Re-ran `uv run --frozen pytest -q tests/test_phase7_contract_verifier.py -x -o log_cli=false`: 14 passed.
- Re-ran `uv run --frozen ruff check tools/verify_phase7_contract_verifier.py tests/test_phase7_contract_verifier.py`: all checks passed.

---
*Phase: 07-explicit-migration-and-rebuild-cutover*
*Completed: 2026-09-10*
