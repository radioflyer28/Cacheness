---
phase: 04-metadata-composition-and-topology-contracts
plan: "13"
subsystem: testing
tags: [blob-store, catalog, cutover, pytest, python-3.11, python-3.13, ruff]
requires:
  - phase: 04-12
    provides: Stable public consumer coverage for the BlobStore/catalog cutover
provides:
  - Remaining query, stored-format, and custom-catalog consumers on public BlobStore seams
  - AST audit of executable consumers for retired authority and selector use
  - Marker-bounded owned release matrix and separate deferred SQL-cache diagnostic
affects: [phase-05, phase-06, phase-07, phase-08, release-qualification]
actuals:
  tokens: 32985
  tasks: 2
  commits: 2
tech-stack:
  added: []
  patterns:
    - Release evidence derives a pytest matrix only from a marker-bounded owned list.
    - Deferred optional-dependency collection diagnostics are classified separately and never treated as a green suite.
key-files:
  created:
    - tools/verify_phase4_cutover.py
    - .planning/phases/04-metadata-composition-and-topology-contracts/04-13-SUMMARY.md
  modified:
    - tests/test_cached_query_meta.py
    - tests/test_stored_compatibility.py
    - examples/custom_metadata_demo.py
    - .planning/phases/04-metadata-composition-and-topology-contracts/04-VALIDATION.md
    - tests/fixtures/phase4_ruff_baseline.json
key-decisions:
  - "Only the 41-path Phase 4-owned list supplies PHASE4_MATRIX; the three pandas-dependent SQL-cache paths are diagnostic-only."
  - "A classified absent-pandas collection result remains explicitly non-green rather than a full-suite pass."
  - "Stable consumers prove public BlobStore/catalog behavior and current-format failures; they do not restore retired metadata authorities or compatibility adapters."
patterns-established:
  - "Use behavior-level public storage assertions rather than backend, ORM, or lifecycle-authority shape inspection."
  - "Treat Ruff scope inventory additions as clean-path maintenance; never refresh existing diagnostic fingerprints to manufacture a passing qualification gate."
requirements-completed: [BACK-02, BACK-03, BACK-06, BACK-07]
coverage:
  - id: D1
    description: "Remaining stable-path query, stored-format, and catalog-example consumers use only current public BlobStore/catalog/version seams."
    requirement: BACK-07
    verification:
      - kind: unit
        ref: "uv run --frozen pytest -q tests/test_cached_query_meta.py tests/test_stored_compatibility.py -o log_cli=false"
        status: pass
      - kind: other
        ref: "uv run --frozen python examples/custom_metadata_demo.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "The executable consumer tree has no retired metadata-authority or blob-selector use, and the owned matrix is disjoint from deferred SQL-cache diagnostics."
    requirement: BACK-02
    verification:
      - kind: other
        ref: "uv run --frozen --python 3.11 python tools/verify_phase4_cutover.py --all"
        status: pass
      - kind: other
        ref: "uv run --frozen --python 3.13 python tools/verify_phase4_cutover.py --all"
        status: pass
    human_judgment: false
  - id: D3
    description: "Phase 4 qualification preserves a frozen Ruff baseline while adding only verified-clean Plan 04-09 through 04-13 paths."
    requirement: BACK-06
    verification:
      - kind: other
        ref: "uv run --frozen --python 3.11 python tools/verify_phase4_ruff_delta.py"
        status: pass
      - kind: other
        ref: "uv run --frozen --python 3.13 python tools/verify_phase4_ruff_delta.py"
        status: pass
    human_judgment: false
duration: 2h
completed: 2026-09-08
status: complete
---

# Phase 4 Plan 13: Consumer Cutover and Two-Interpreter Evidence Summary

**Public BlobStore/catalog consumers and a 41-path release matrix now prove the cutover on Python 3.11 and 3.13, while pandas-dependent SQL-cache collection remains transparently diagnostic-only.**

## Performance

- **Duration:** 2h
- **Completed:** 2026-09-08
- **Tasks:** 2/2
- **Files modified:** 8
- **Evidence:** CPython 3.11.16: 588 passed, 6 skipped in 13.39s; CPython 3.13.15: 588 passed, 6 skipped in 9.78s.

## Accomplishments

- Replaced the final query, stored-format, and custom-metadata example consumers with public `BlobStore`, `CatalogSchema`, catalog query/update, current-format, and explicit migration-required seams.
- Added `verify_phase4_cutover.py`, which parses exact marker-bounded owned/deferred test sets, audits executable Python with AST, runs the owned matrix without test-selection filters, and reports deferred full-tree collection separately.
- Recorded reproducible two-interpreter matrix and Ruff-delta evidence. The only full-tree collection failures are exactly the three deferred paths caused by absent pandas; that diagnosis is explicitly not a full-suite pass.

## Task Commits

1. **Task 1: Migrate the remaining query, stored-format, and catalog-example consumers** — `cf247e6` (`test`)
2. **Task 2: Enforce disjoint owned-matrix and deferred-diagnostic sets on Python 3.11 and 3.13** — `cb5f167` (`test`)

## Files Created/Modified

- `tests/test_cached_query_meta.py` — public catalog query, current-generation, cleanup, and close coverage.
- `tests/test_stored_compatibility.py` — current format-2 reopen plus typed non-mutating unsupported-layout evidence.
- `examples/custom_metadata_demo.py` — direct BlobStore/catalog schema/query/update workflow.
- `tools/verify_phase4_cutover.py` — consumer audit, owned-matrix runner, and deferred collection classifier.
- `04-VALIDATION.md` — exact owned/deferred path sections and interpreter evidence.
- `tests/test_filesystem_containment.py` and `tests/test_lifecycle_authority_contract.py` — existing invariants migrated to the final public topology/cleanup seams.
- `tests/fixtures/phase4_ruff_baseline.json` — clean Plan 04-09 through 04-13 paths added to the declared qualification inventory only.

## Decisions Made

- The 41 test paths bounded by `phase4-owned-matrix` markers are the whole Phase 4 release matrix. The bounded three-path SQL-cache list is structurally excluded and is only valid for full-tree diagnostic classification.
- The full-tree collection result does not overclaim optional dependency coverage: exact absent-pandas failures in the three deferred paths are documented as non-green until Phase 8.
- Scope drift in the Ruff gate is resolved by adding only individually Ruff-clean declared paths, preserving every prior baseline diagnostic and fingerprint.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Stale public-surface assertions] Migrated final topology and cleanup contract tests**
- **Found during:** Task 2
- **Issue:** Two containment cases asserted an obsolete nested cleanup exception, and two authority-contract tests constructed `BlobStore` with the retired `lifecycle_authority` argument.
- **Fix:** Assert the current exact cleanup `OSError` message and construct the store with the public `StoreTopology` seam. The root-creation assertion now accounts for topology construction while preserving zero additional mutation by read-only calls and reopen.
- **Files modified:** `tests/test_filesystem_containment.py`, `tests/test_lifecycle_authority_contract.py`
- **Verification:** Focused contract tests pass as members of both 41-path interpreter matrices.
- **Committed in:** `cb5f167`

**2. [Rule 3 - Qualification inventory] Added declared clean Phase 4 paths to the frozen Ruff scope**
- **Found during:** Task 2
- **Issue:** The delta tool rejected its inventory because plans 04-09 through 04-13 declared new Python paths outside the original 04-01 through 04-08 baseline.
- **Fix:** Added only ten individually Ruff-clean declared paths as `new_paths`; no existing finding, fingerprint, or hash was refreshed, removed, or forgiven.
- **Files modified:** `tests/fixtures/phase4_ruff_baseline.json`
- **Verification:** The Ruff-delta command passes on CPython 3.11.16 and 3.13.15.
- **Committed in:** `cb5f167`

**Total deviations:** 2 auto-fixed (1 Rule 1, 1 Rule 3).

## Issues Encountered

The final full-tree collection cannot import pandas for exactly `tests/test_sql_cache.py`, `tests/test_sql_cache_documentation.py`, and `tests/test_sql_cache_failure_contract.py`. The verifier classifies this bounded outcome, but it deliberately does not call it a successful full suite. Phase 8 owns optional pandas/SQL-cache installation qualification.

## User Setup Required

None - no external service configuration is required for this plan.

## Next Phase Readiness

Phase 4 now has deterministic current-surface consumer and two-interpreter evidence. Phase 5 still needs live PostgreSQL/S3 topology qualification, Phase 6 owns cache-policy composition, Phase 7 owns offline migration/rebuild execution, and Phase 8 owns pandas/SQL-cache installation qualification. This plan makes no native Windows, live-service, cross-resource ACID, or universal-progress claim.

## Self-Check: PASSED

- `04-13-SUMMARY.md` exists.
- Task commits `cf247e6` and `cb5f167` exist in the repository history.
