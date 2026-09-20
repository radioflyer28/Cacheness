---
phase: 04-metadata-composition-and-topology-contracts
plan: "01"
subsystem: testing
tags: [ruff, pytest, catalog, topology, projections]
requires:
  - phase: 03-atomic-lifecycle-and-recovery-engine
    provides: SQLite lifecycle authority, immutable payload lifecycle, and derived projection boundary
provides:
  - Stable Phase 4 Ruff regression gate captured before Phase 4 production edits
  - Six assertion-level red contract suites for catalog, composition, topology, roles, and projections
affects: [04-02, 04-03, 04-04, 04-05, 04-06, 04-07, 04-08]
actuals:
  tokens: 14065.5
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - Line-independent Ruff diagnostic fingerprints with explicit planned-path scope
    - Assertion-gated red contracts that collect before Phase 4 source surfaces exist
key-files:
  created:
    - tests/fixtures/phase4_ruff_baseline.json
    - tools/verify_phase4_ruff_delta.py
    - tests/test_catalog_schema.py
    - tests/test_catalog_query_contract.py
    - tests/test_blob_store_composition.py
    - tests/test_topology_capabilities.py
    - tests/test_metadata_role_contract.py
    - tests/test_catalog_projection.py
  modified: []
key-decisions:
  - "Freeze Phase 4 lint debt from the plan-declared Python inventory, allowing only explicit retirement targets and clean new files."
  - "Use assertion-level import gates so Wave 0 contracts fail red without hiding missing clean APIs behind collection errors."
patterns-established:
  - "Portable catalog contracts distinguish stored absence from declared defaults and reject boolean-as-integer coercion."
  - "Composition contracts derive guarantees from active participants, preserve injected identity, and keep projections non-authoritative."
requirements-completed: [BACK-02, BACK-03, BACK-06, BACK-07]
coverage:
  - id: D1
    description: "Phase 4 Ruff delta gate detects changed diagnostics, scope drift, malformed baselines, and dirty new files."
    requirement: BACK-07
    verification:
      - kind: unit
        ref: "uv run --frozen python tools/verify_phase4_ruff_delta.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "Clean catalog, composition, topology, metadata-role, and projection behavior is specified before implementation."
    requirement: BACK-02
    verification:
      - kind: unit
        ref: "tests/test_catalog_schema.py and five companion Wave 0 contract suites"
        status: unknown
    human_judgment: false
duration: 5m 13s
completed: 2026-09-07
status: complete
---

# Phase 04 Plan 01: Wave 0 Contract Summary

**Frozen Phase 4 Ruff evidence and 79 collecting red contracts define the clean catalog, typed composition, topology, and projection cutover before production changes begin.**

## Performance

- **Duration:** 5m 13s
- **Started:** 2026-09-07T21:03:03-04:00
- **Completed:** 2026-09-07T21:08:16-04:00
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Captured a 46-diagnostic baseline across the 49 existing Phase 4 Python paths and added a verifier that allows only explicit retirement while requiring every new Phase 4 path to be Ruff-clean.
- Added six collection-safe red suites (79 tests) for the format-2 schema, typed query/cursor, one-root composition, topology capabilities, explicit metadata roles, and pull/checkpoint projection semantics.
- Locked Phase 4 contracts to typed retryable contention outcomes, projection non-authority, and non-mutating migration/rebuild-required handling without claiming live PostgreSQL/S3 qualification or cross-resource ACID.

## Task Commits

Each task was committed atomically:

1. **Task 1: Freeze the Phase 4 Ruff delta before source changes** - `53ad148` (test)
2. **Task 2: Specify the current-format catalog, composition, topology, and projection contracts** - `acd5fa8` (test)

## Files Created/Modified

- `tests/fixtures/phase4_ruff_baseline.json` - Checked-in stable Ruff diagnostic baseline for the declared Phase 4 scope.
- `tools/verify_phase4_ruff_delta.py` - Scope-aware Ruff verifier that rejects changed findings and malformed or drifted baseline data.
- `tests/test_catalog_schema.py` - Format-version, schema, default/presence, and unsupported-layout contracts.
- `tests/test_catalog_query_contract.py` - Predicate, bounds, ordering, cursor, stale-revision, and sparse-scan contracts.
- `tests/test_blob_store_composition.py` - One-root selection, exact injection, ownership, and legacy-selector retirement contracts.
- `tests/test_topology_capabilities.py` - Participant-derived capability and topology-qualified progress contracts.
- `tests/test_metadata_role_contract.py` - Authority/projection role separation across metadata families.
- `tests/test_catalog_projection.py` - Bounded pull, checkpoint, partial-result, and isolated rebuild contracts.

## Decisions Made

- The Ruff baseline is derived from the committed Phase 4 plan inventories so an undeclared Python scope change is itself a gate failure; line numbers are excluded from fingerprints while path, rule, message, and source remain bound.
- Wave 0 contract modules use explicit assertion gates for absent clean modules. They collect successfully and fail with assertions until later plans implement the surface, rather than failing during pytest collection.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Made the Ruff verifier compatible with the installed Ruff JSON schema.**

- **Found during:** Task 1 (Freeze the Phase 4 Ruff delta before source changes)
- **Issue:** Ruff 0.12 JSON diagnostics omit the historical `source` field, so the initial fingerprint parser could not capture the baseline.
- **Fix:** Derived the source line from each diagnostic's bounded file location, then recomputed and verified the fingerprint from path, rule, message, and source.
- **Files modified:** `tools/verify_phase4_ruff_delta.py`
- **Verification:** The verifier passes for the frozen baseline and rejected an intentional temporary unused-import diagnostic.
- **Committed in:** `53ad148` (part of Task 1)

---

**Total deviations:** 1 auto-fixed (1 Rule 1 bug fix)
**Impact on plan:** Necessary to make the planned stable source fingerprint work with the project's locked Ruff version; no production scope changed.

## Issues Encountered

- The repository ignores `*.json`; the intentionally checked-in Ruff baseline was staged explicitly as the plan's declared fixture.
- The main-branch safety policy required the orchestrator's approved commit path for both task commits; only their declared files were staged.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plans 04-02 through 04-05 can implement the clean catalog, composition, authority, and projection surfaces against the locked red contracts.
- The contracts deliberately do not claim live PostgreSQL/S3 topology qualification; that remains Phase 5 work.

---
*Phase: 04-metadata-composition-and-topology-contracts*
*Completed: 2026-09-07*

## Self-Check: PASSED

- All eight declared artifacts exist.
- Task commits `53ad148` and `acd5fa8` are present in git history.
- `uv run --frozen python tools/verify_phase4_ruff_delta.py` passes.
