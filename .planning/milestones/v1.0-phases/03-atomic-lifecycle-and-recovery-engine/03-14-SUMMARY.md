---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "14"
subsystem: lifecycle-authority-projection-parity
tags: [lifecycle-authority, projection-cas, json, sqlite, postgresql, concurrency]
requires:
  - phase: 03-13
    provides: Frozen isolated full-suite and release-gate execution contract
  - phase: 03-08
    provides: Canonical SQLite lifecycle authority and reconciliation evidence
provides:
  - Generation/locator-conditional projection mutation across memory, JSON, SQLite, and PostgreSQL adapters
  - Committed-only cache/query/custom-metadata facade views with bounded coherent authority/projection reads
  - Deterministic stale-projection and backend-parity race coverage
affects: [phase-3-completion, blobstore, unified-cache, metadata-backends, release-validation]
actuals:
  tokens: 19194
  tasks: 3
  commits: 4
tech-stack:
  added: []
  patterns:
    - Projection rows use exact generation-specific locator tokens for compare-and-mutate operations
    - JSON coordinates only a short document refresh/compare/atomic-replace interval with a sibling descriptor lock
    - SQLite reconciles schema only during bounded first use, then relies on its native short writer transactions
key-files:
  created:
    - tests/test_projection_mutation_contract.py
  modified:
    - src/cacheness/core.py
    - src/cacheness/metadata.py
    - src/cacheness/storage/lifecycle.py
    - src/cacheness/storage/blob_store.py
    - src/cacheness/storage/backends/postgresql_backend.py
    - src/cacheness/storage/sqlite_lifecycle_authority.py
    - tests/test_unified_cache_lifecycle_authority.py
key-decisions:
  - "Facade projections are rebuildable compatibility state: only an authenticated committed authority manifest may make a row live."
  - "A stale projection writer proves the exact observed generation-specific locator; a mismatch evicts cached state and changes no replacement row or custom link."
  - "JSON uses a projection-document-only descriptor lock, PostgreSQL verifies the token in its transaction, and SQLite avoids any facade-wide operation lock."
  - "Darwin D-32 evidence remains UNAVAILABLE/NOT_QUALIFIED; Phase 999.1 alone owns native Windows PASS qualification."
patterns-established:
  - "Use authority snapshots plus exact projection tokens for facade repair; return a coherent pair after one retry or raise a typed conflict."
  - "Keep schema/bootstrap coordination bounded to authority initialization, not ordinary lifecycle or payload work."
requirements-completed: [STOR-03, STOR-04, STOR-05, STOR-06, STOR-07]
coverage:
  - id: D1
    description: Stale facade projection teardown cannot replace or remove a concurrent generation or its custom links.
    requirement: STOR-03
    verification:
      - kind: integration
        ref: "tests/test_unified_cache_lifecycle_authority.py"
        status: pass
    human_judgment: false
  - id: D2
    description: Public cache, metadata-query, custom-metadata, and aggregate surfaces expose only committed authority entries.
    requirement: STOR-04
    verification:
      - kind: integration
        ref: "tests/test_unified_cache_lifecycle_authority.py"
        status: pass
    human_judgment: false
  - id: D3
    description: Memory, cached, JSON cross-instance, and PostgreSQL projection adapters implement token-conditional parity.
    requirement: STOR-05
    verification:
      - kind: integration
        ref: "tests/test_projection_mutation_contract.py"
        status: pass
    human_judgment: false
  - id: D4
    description: Complete supported-environment acceptance, stored-data compatibility, and Ruff-delta gates pass without changing Windows qualification evidence.
    requirement: STOR-07
    verification:
      - kind: integration
        ref: "uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q -o log_cli=false"
        status: pass
      - kind: integration
        ref: "tests/fixtures/compat/validate_corpus.py --expected-through sqlite-columns-v0314"
        status: pass
      - kind: unit
        ref: "tests/test_phase3_ruff_delta.py"
        status: pass
    human_judgment: false
duration: 1h 29min
completed: 2026-09-06
status: complete
---

# Phase 03 Plan 14: Projection Backend Parity Summary

**Compatibility projections now follow exact authority generation/locator tokens across local, JSON, cached, and PostgreSQL metadata paths, so stale facade work cannot expose or erase another lifecycle generation.**

## Performance

- **Duration:** 1h 29min
- **Started:** 2026-09-06T03:07:57Z
- **Completed:** 2026-09-06T04:37:17Z
- **Tasks:** 3/3
- **Files modified:** 8

## Accomplishments

- Carried the operation-captured expectation and promoted authority snapshot through private BlobStore/lifecycle seams while preserving public `BlobStore.put()` key-only returns.
- Made projection replacement, deletion, cache invalidation, JSON cross-instance updates, and PostgreSQL custom-link insertion conditional on the exact observed token; all public facade surfaces now filter to authenticated committed authority state.
- Added deterministic two-facade, cross-instance JSON, cached-wrapper, mocked-PostgreSQL, and delayed custom-link contracts; passed fresh frozen Python 3.11, compatibility, and Ruff acceptance gates.

## Verification

Passed focused lifecycle/parity acceptance:

```bash
uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q tests/test_unified_cache_lifecycle_authority.py tests/test_projection_mutation_contract.py tests/test_blob_store_concurrency.py tests/test_sqlite_lifecycle_authority.py -x -o log_cli=false
```

Passed the exact complete-suite gate:

```bash
uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q -o log_cli=false
```

The full suite passed with existing documented skips for unavailable native Windows junction evidence, unavailable PostgreSQL service, and TensorFlow safety coverage. Its existing interpreter-shutdown `SqliteBackend.__del__` diagnostics occurred only after pytest returned exit 0.

Also passed:

```bash
uv run --isolated --all-extras --group dev --frozen python tests/fixtures/compat/validate_corpus.py --expected-through sqlite-columns-v0314
uv run --isolated --all-extras --group dev --frozen pytest -q tests/test_phase3_ruff_delta.py -x -o log_cli=false
uv run --isolated --all-extras --group dev --frozen python tools/verify_phase3_ruff_delta.py
```

## Task Commits

1. **Task 1: Preserve a concurrent M2 projection and its custom links through stale teardown**
   - `bb1c4ea` — `test(03-14): expose stale projection teardown race`
   - `347ecb3` — `fix(03-14): bind projections to lifecycle tokens`
2. **Task 2: Hide tombstones and return only coherent authority/projection pairs**
   - `db6c54d` — `fix(03-14): hide non-live authority projections`
3. **Task 3: Seal backend parity and rerun complete Phase 3 acceptance**
   - `e1494bf` — `feat(03-14): seal projection backend parity`

## Files Created/Modified

- `src/cacheness/core.py` — committed-only authority facade filtering, token-aware repair, bounded coherent reads, and promoted-token custom metadata binding.
- `src/cacheness/metadata.py` — JSON document-level conditional projection mutation guard and adapter contract.
- `src/cacheness/storage/backends/postgresql_backend.py` — exact PostgreSQL row/link mutation and delayed custom-metadata token verification.
- `src/cacheness/storage/sqlite_lifecycle_authority.py` — bounded same-instance bootstrap and first-use schema reconciliation without an operation-wide lock.
- `src/cacheness/storage/lifecycle.py` and `src/cacheness/storage/blob_store.py` — private result-preserving lifecycle seam that leaves the public BlobStore API intact.
- `tests/test_unified_cache_lifecycle_authority.py` and `tests/test_projection_mutation_contract.py` — deterministic race, adapter, and PostgreSQL wiring regression coverage.

## Decisions Made

- Treat projection metadata as derived compatibility state; lifecycle authority remains the sole canonical recovery source.
- Require exact generation-specific locator comparison for all projection mutation and custom-link work. Same-token retries converge, while stale different-token work is a no-op/conflict.
- Use only short backend-local coordination: JSON's document commit guard, PostgreSQL's transactional row lock, and SQLite first-use schema setup. No facade/global lock spans payload work, authority access, or unrelated keys.
- Kept D-32 unchanged: this Darwin host remains `UNAVAILABLE`/`NOT_QUALIFIED` with `native_evidence: false`; Phase 999.1 requires native Windows Python 3.11 PASS/exit 0.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Broke the metadata-to-storage import cycle for JSON commit locking**
- **Found during:** Task 3
- **Issue:** Importing the cross-platform descriptor-lock helper eagerly from `metadata.py` re-entered the package's metadata compatibility exports.
- **Fix:** Resolved the helper lazily inside the JSON projection commit guard, preserving the short lock scope.
- **Files modified:** `src/cacheness/metadata.py`
- **Verification:** JSON two-instance stale-writer contract passes.
- **Committed in:** `e1494bf`

**2. [Rule 1 - Concurrency] Repaired same-instance SQLite authority bootstrap and first-use schema contention**
- **Found during:** Task 3 complete-suite acceptance
- **Issue:** Parallel first puts could classify bounded bootstrap artifacts as legacy evidence; repeatedly running an EXCLUSIVE schema reconciliation before each transition also exhausted the 187ms writer-admission bound.
- **Fix:** Added bootstrap-only admission and one-time schema reconciliation. Ordinary state transitions remain native SQLite bounded transactions, with no whole-cache operation lock.
- **Files modified:** `src/cacheness/storage/sqlite_lifecycle_authority.py`
- **Verification:** Both collected SQLite concurrency files pass, focused lifecycle tests pass, and the frozen full suite passes.
- **Committed in:** `e1494bf`

**3. [Rule 1 - Safety] Restored size-cleanup containment preflight before its projection-size fast path**
- **Found during:** Task 3 complete-suite acceptance
- **Issue:** The fast return for an under-limit authority projection skipped the existing multi-entry path-containment check.
- **Fix:** Run the established metadata preflight before evaluating the fast-path size bound.
- **Files modified:** `src/cacheness/core.py`
- **Verification:** `tests/test_filesystem_containment.py::test_high_level_locator_preflight_blocks_multi_entry_mutation` and the full suite pass.
- **Committed in:** `e1494bf`

**4. [Rule 1 - Test Isolation] Re-register custom-metadata schemas within projection race fixtures**
- **Found during:** Task 3 focused-suite execution
- **Issue:** Other test modules reset the process-global custom metadata registry, making the independent race fixture order-sensitive.
- **Fix:** Re-register the fixture's model before each contract scenario.
- **Files modified:** `tests/test_projection_mutation_contract.py`
- **Verification:** Focused contract and complete suites pass under ordinary collection.
- **Committed in:** `e1494bf`

**Total deviations:** 4 auto-fixed (3 Rule 1, 1 Rule 3).

**Impact on plan:** All changes preserve the planned canonical-authority architecture and were required to make the new projection behavior safe under ordinary complete-suite contention.

## Issues Encountered

- The initially restored full suite exposed existing high-contention SQLite tests beyond the narrower two-test gate. The final repair reduces repeated schema-exclusive work rather than relaxing lifecycle timeouts or serializing cache operations globally.

## Known Stubs

None. The changed-file scan found only intentional optional/contract values, not a user-visible placeholder or unwired data path.

## Next Phase Readiness

- Plan 03-14 is complete with fresh end-to-end evidence for STOR-03 through STOR-07, but this summary does not mark Phase 3 itself complete.
- D-32 remains honestly unqualified on Darwin. Native Windows Python 3.11 evidence is still required in Phase 999.1.

## Self-Check: PASSED

- All four Task commits (`bb1c4ea`, `347ecb3`, `db6c54d`, `e1494bf`) exist.
- The created parity contract test and every modified implementation path listed above exist.
- Focused, frozen full-suite, compatibility-corpus, and Ruff-delta acceptance commands completed successfully; no changed qualification artifact claims Windows support.
