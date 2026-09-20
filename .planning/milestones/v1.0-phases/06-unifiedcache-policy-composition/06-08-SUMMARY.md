---
phase: 06-unifiedcache-policy-composition
plan: "08"
subsystem: validation
tags: [python, pytest, ast, unifiedcache, blobstore, validation]
requires:
  - phase: 06-07
    provides: Canonical public cache-policy documentation and explicit BlobStore-powered examples
provides:
  - Root-safe fixed-scope verifier for the Phase 6 public-policy and lifecycle-authority contracts
  - Mutation-style verifier self-tests and an executed evidence ledger with open environment gates preserved
affects: [phase-06-acceptance, phase-08-qualification, release-verification]
actuals:
  tokens: 11650
  tasks: 2
  commits: 4
tech-stack:
  added: []
  patterns:
    - Fixed pytest node manifests and narrowly scoped AST checks provide deterministic cache-policy regression evidence.
    - Validation distinguishes local candidate evidence from unavailable live-service and platform qualification.
key-files:
  created:
    - tools/verify_phase6_contracts.py
    - tests/test_phase6_contract_verifier.py
  modified:
    - tests/test_phase6_statistics.py
    - tests/test_phase3_local_workflows.py
    - .planning/phases/06-unifiedcache-policy-composition/06-VALIDATION.md
key-decisions:
  - "The verifier executes a fixed CACH-01 through CACH-06 manifest plus the fixed SqlCache regression and exact hostile JSON-projection node; it never derives scope from a diff or all-test discovery."
  - "CACH-07 and the current-environment full-suite gate remain open when pandas is not installed; plan completion is not Phase 6 acceptance."
  - "PostgreSQL/S3 candidate fixtures and native-Windows evidence remain unavailable and are not BACK-05 or Phase 6 passing evidence."
patterns-established:
  - "Cache-policy source checks fail closed on extra lifecycle authority, direct resource mutation, queues or locks, compatibility routes, unbounded catalog traversal, and unsupported guarantee wording."
requirements-completed: [CACH-01, CACH-02, CACH-03, CACH-04, CACH-05, CACH-06]
coverage:
  - id: D1
    description: Fixed Phase 6 behavioral, architecture, and retained lifecycle manifests run under the repository-safe verifier.
    requirement: CACH-01
    verification:
      - kind: integration
        ref: tools/verify_phase6_contracts.py
        status: pass
    human_judgment: false
  - id: D2
    description: The exact hostile direct-projection node rejects format_version 999 with JsonProjectionError for an incompatible shape.
    requirement: CACH-06
    verification:
      - kind: unit
        ref: tests/test_catalog_projection.py::test_json_projection_rejects_incompatible_derived_documents
        status: pass
    human_judgment: false
  - id: D3
    description: The fixed SqlCache regression and full suite are required current-environment evidence but cannot collect without pandas.
    requirement: CACH-01
    verification:
      - kind: environment
        ref: .planning/phases/06-unifiedcache-policy-composition/06-VALIDATION.md
        status: open
    human_judgment: true
duration: 25min
completed: 2026-09-09
status: complete
---

# Phase 06 Plan 08: Fixed Contract Verification Summary

**A self-tested fixed verifier now proves the local Phase 6 cache-policy and one-authority contracts while preserving the missing-pandas full-suite, live-remote, and native-Windows gates as open evidence.**

## Performance

- **Duration:** 25 min
- **Started:** 2026-09-09T04:15:41Z
- **Completed:** 2026-09-09T04:40:31Z
- **Tasks:** 2/2
- **Files modified:** 5

## Accomplishments

- Added a repository-root-safe verifier with a fixed CACH-01 through CACH-06 manifest, the separate SqlCache regression, named Phase 3--5 retained contracts, and the exact hostile JSON-projection node.
- Added fail-closed AST and source-contract checks for a second lifecycle authority, direct policy-layer resource mutation, lifecycle locks or queues, compatibility routes, unbounded catalog traversal, hidden global ownership, and inflated topology guarantees.
- Added mutation-style self-tests for verifier diagnostics and updated two fixed retained fixtures to use the canonical nested cache configuration and result surface.
- Recorded executed commands, concrete decision coverage, strict JSON-projection rejection, candidate-only remote evidence, and all unresolved gates in `06-VALIDATION.md`.

## Task Commits

1. **Task 1: Prove one fixed Phase-6 contract manifest end to end**
   - `1f019fd` — `test(06-08): add failing Phase 6 verifier contract tests`
   - `c123749` — `feat(06-08): verify fixed Phase 6 cache contracts`
   - `91ce501` — `fix(06-08): execute fixed SQL regression in verifier`
2. **Task 2: Execute the validation matrix and record honest evidence**
   - `b58de9c` — `docs(06-08): record Phase 6 validation evidence`

## Files Created/Modified

- `tools/verify_phase6_contracts.py` — Explicit test manifest, contract/AST audits, executable diagnostics, and no live-service or dependency-install behavior.
- `tests/test_phase6_contract_verifier.py` — Mutation-style proof that the verifier rejects prohibited structure and honors fixed node identifiers.
- `tests/test_phase6_statistics.py` — Canonical nested configuration fixture for the retained policy statistic contract.
- `tests/test_phase3_local_workflows.py` — Canonical configuration/result fixture for the retained local lifecycle regression.
- `06-VALIDATION.md` — Executed Nyquist evidence matrix, open-gate records, and Phase 8/BACK-05 exclusions.

## Decisions Made

- `BlobStore` and its `AuthorityLifecycleEngine` remain the sole lifecycle authority. The verifier explicitly rejects a new lifecycle coordinator, policy-layer direct deletion, lifecycle lock, admission queue, readiness registry, or compatibility projection authority.
- The verifier reports local fixed-manifest results separately from CACH-07 and full-suite environment readiness. It does not treat mocked remote behavior as live qualification, and it does not promise cross-resource ACID, universal contender success, exact global-oldest eviction, or timing guarantees.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] Updated two retained fixed-manifest fixtures to the canonical cache surface.**
- **Found during:** Task 1
- **Issue:** The historical flat `CacheConfig(cache_dir=...)` fixtures and a legacy result expectation prevented retained Phase 3/6 contracts from exercising the canonical store/result policy.
- **Fix:** Used `CacheStorageConfig` through nested `CacheConfig` and the explicit result API; no production API or compatibility shim was restored.
- **Files modified:** `tests/test_phase6_statistics.py`, `tests/test_phase3_local_workflows.py`
- **Commit:** `c123749`

**2. [Rule 2 - Missing critical functionality] Executed the fixed SqlCache regression rather than merely naming it.**
- **Found during:** Task 1
- **Issue:** The initial manifest listed `tests/test_sql_cache.py` without running it, so it could not prove or expose the required regression result.
- **Fix:** Added the CACH-07 node to the verifier execution path and self-test coverage. In the default environment it now fails visibly because pandas is unavailable.
- **Files modified:** `tools/verify_phase6_contracts.py`, `tests/test_phase6_contract_verifier.py`
- **Commit:** `91ce501`

## Issues Encountered

- `uv run --frozen python tools/verify_phase6_contracts.py --repo-root .` is nonzero in the final default environment because `tests/test_sql_cache.py` cannot import `pandas`. The CACH-01 through CACH-06 and strict-projection portions pass; the CACH-07 node remains open.
- `uv run --frozen pytest -q --tb=no -o log_cli=false` has three collection errors: `tests/test_sql_cache.py`, `tests/test_sql_cache_documentation.py`, and `tests/test_sql_cache_failure_contract.py`, each due to `ModuleNotFoundError: No module named 'pandas'`. No dependency was installed or substituted.
- PostgreSQL/S3 live qualification and native-Windows evidence were unavailable and remain explicitly excluded. Candidate or mocked remote tests are not BACK-05 evidence.
- A proposed broad migration of 15 historic test modules was evaluated but not applied because it would have removed 5,299 lines including non-compatibility integrity and serialization assertions. The open full-suite gate is routed to the review/verification/gap-fix cycle instead of masking coverage or restoring compatibility APIs.

## Phase Verification Status

**PENDING — this plan is complete, but it is not a Phase 6 acceptance claim.** The required current-environment full suite and fixed CACH-07 regression are open until the qualified test environment supplies pandas and the resulting failures are independently reviewed. BACK-05 live PostgreSQL/S3 and native-Windows qualification remain outside this plan and unpassed.

## Verification

- `uv run --frozen pytest -q tests/test_phase6_contract_verifier.py -o log_cli=false` — 13 passed.
- `uv run --frozen pytest -q tests/test_catalog_projection.py::test_json_projection_rejects_incompatible_derived_documents -o log_cli=false` — 1 passed; `format_version` 999 raised `JsonProjectionError` matching incompatible shape.
- `uv run --frozen ruff check src/cacheness/__init__.py src/cacheness/config.py src/cacheness/core.py src/cacheness/cache_policy.py src/cacheness/decorators.py src/cacheness/storage/blob_store.py src/cacheness/storage/lifecycle.py tools/verify_phase6_contracts.py tests/test_phase6_lookup_contract.py tests/test_phase6_removal_contract.py tests/test_phase6_statistics.py tests/test_phase6_decorator_contract.py tests/test_phase6_public_api_contract.py tests/test_phase6_policy_contract.py tests/contracts/test_phase6_topology_policy.py tests/test_phase6_examples.py tests/test_phase6_contract_verifier.py tests/test_phase3_local_workflows.py` — passed.
- `uv run --frozen python tools/verify_phase6_contracts.py --repo-root .` — nonzero only for the CACH-07 SqlCache collection/import dependency gate; recorded open in the ledger.
- `uv run --frozen pytest -q --tb=no -o log_cli=false` — three pandas-related collection errors; recorded open in the ledger.

## Known Stubs

None.

## Next Phase Readiness

- Run the required review/verification/gap-fix cycle against an environment with the optional SQL test dependency available; preserve the current fixed manifest and do not relax its gates.
- Keep live PostgreSQL/S3 BACK-05 and native-Windows evidence in their dedicated Phase 8 qualification scope.

## Self-Check: PASSED

- All six listed created, modified, and ledger artifacts exist on disk.
- Task commits `1f019fd`, `c123749`, `91ce501`, and `b58de9c` exist in Git history.
- The changed implementation and test files contain no tracked placeholder or TODO/FIXME stub that prevents this plan's verifier outcome.

---

*Phase: 06-unifiedcache-policy-composition*
*Plan completed: 2026-09-09; Phase verification pending.*
