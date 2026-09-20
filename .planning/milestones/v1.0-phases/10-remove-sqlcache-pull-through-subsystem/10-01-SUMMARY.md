---
phase: 10-remove-sqlcache-pull-through-subsystem
plan: 01
subsystem: testing
tags: [pytest, public-api, packaging-contract, sqlcache-removal]
requires:
  - phase: 09-adoption-and-release-surface-closure
    provides: BlobStore-first public contracts and isolated wheel qualification baseline
provides:
  - Fail-closed SqlCache removal contract covering source, imports, packaging, current references, and caller-table boundaries
  - Inverted public API and suite-order contracts for the surviving BlobStore and UnifiedCache surface
affects: [phase-10-cutover, packaging, documentation, public-api]
actuals:
  tokens: 3976
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - Literal negative contracts with subprocess import probes
    - Exact current-surface reference allowlist with historical exclusions
key-files:
  created:
    - tests/test_phase10_sqlcache_removal.py
  modified:
    - tests/test_public_api_contract.py
    - tests/test_phase6_public_api_contract.py
    - tests/test_phase6_suite_isolation.py
key-decisions:
  - "Treat physical absence and ordinary Python import failures as the SqlCache cutover contract; retain no compatibility behavior."
  - "Keep public contract order isolation only for the surviving BlobStore and UnifiedCache suites."
patterns-established:
  - "Removal contracts distinguish current product claims from preserved dated audits and planning history."
requirements-completed: [CACH-07]
coverage:
  - id: D1
    description: "Phase 10's negative source, import, manifest, current-reference, and caller-table contract is collectable and lint-clean."
    requirement: CACH-07
    verification:
      - kind: unit
        ref: "uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only tests/test_phase10_sqlcache_removal.py"
        status: pass
      - kind: other
        ref: "uv run --isolated --group dev --frozen ruff check tests/test_phase10_sqlcache_removal.py"
        status: pass
    human_judgment: true
    rationale: "The contract intentionally remains red until subsequent shared-requirement plans perform the physical product cut."
  - id: D2
    description: "Reusable public and order-isolation contracts preserve BlobStore and UnifiedCache while demanding SqlCache absence."
    requirement: CACH-07
    verification:
      - kind: unit
        ref: "uv run --isolated --all-extras --group dev --frozen pytest -q --collect-only tests/test_public_api_contract.py tests/test_phase6_public_api_contract.py tests/test_phase6_suite_isolation.py"
        status: pass
      - kind: other
        ref: "uv run --isolated --group dev --frozen ruff check tests/test_public_api_contract.py tests/test_phase6_public_api_contract.py tests/test_phase6_suite_isolation.py"
        status: pass
    human_judgment: true
    rationale: "The updated post-cut assertions are intentionally red until the later direct deletion plan lands."
duration: 3 min
completed: 2026-09-17
status: complete
---

# Phase 10 Plan 01: Removal Contract Summary

**Fail-closed contracts now define SqlCache's physical and import-level removal while preserving the BlobStore/UnifiedCache public boundary.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-09-17T16:42:25Z
- **Completed:** 2026-09-17T16:45:48Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added a literal negative contract for the removed module, dedicated assets, natural Python absence, unchanged version, pruned dependency metadata, bounded current-reference scanning, and no caller-table tooling.
- Inverted the reusable public API contracts to retain only BlobStore and UnifiedCache ownership, including removal of the four SqlCache-only error reasons.
- Reduced suite-order isolation to the two surviving public-contract modules in both relative orders.

## Task Commits

1. **Task 1: Add the fail-closed Phase 10 removal contract** - `6fa2446` (test)
2. **Task 2: Invert reusable public and suite-isolation contracts** - `b7dd4ae` (test)

## Files Created/Modified

- `tests/test_phase10_sqlcache_removal.py` - Exact product-removal and no-caller-data-mutation contract.
- `tests/test_public_api_contract.py` - Natural-absence, retained error-vocabulary, and API-note assertions.
- `tests/test_phase6_public_api_contract.py` - Canonical BlobStore/UnifiedCache-only surface contract.
- `tests/test_phase6_suite_isolation.py` - Both-order isolation for the two surviving public suites.

## Decisions Made

- Natural `ImportError` and `ModuleNotFoundError`, not a shim or tombstone, are the required legacy-import behavior.
- Dated audits and completed planning records are excluded from the current-surface scanner; each remaining reference owner is explicit.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Excluded binary fixtures from the current-surface text scanner**

- **Found during:** Task 1 (Add the fail-closed Phase 10 removal contract)
- **Issue:** The initial root scan attempted UTF-8 decoding of binary test fixtures, causing a helper exception instead of a product-boundary failure.
- **Fix:** Restricted the scanner to reviewed text suffixes (`.py`, `.md`, and `.toml`).
- **Files modified:** `tests/test_phase10_sqlcache_removal.py`
- **Verification:** The focused RED run reports only the existing SqlCache product surface; collection and Ruff both pass.
- **Committed in:** `6fa2446`

---

**Total deviations:** 1 auto-fixed (1 Rule 1 bug).
**Impact on plan:** The scanner now fails only on the intended current-surface claims and retains the required historical exclusions.

## Issues Encountered

None. The focused runtime assertions are intentionally RED until later Phase 10 plans delete the product and its related surface.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The direct deletion plan can now remove the SqlCache product without hiding stale exports, dependencies, documentation, or compatibility behavior.
- The current focused behavior suites remain intentionally RED only for the pre-cut SqlCache surface; collection and scoped Ruff are green.

## Self-Check: PASSED

- Confirmed all four changed contract files exist.
- Confirmed task commits `6fa2446` and `b7dd4ae` exist in Git history.
- Confirmed the four-module collection and scoped Ruff verification pass.

---
*Phase: 10-remove-sqlcache-pull-through-subsystem*
*Completed: 2026-09-17*
