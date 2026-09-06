---
phase: 03-atomic-lifecycle-and-recovery-engine
plan: "16"
subsystem: lifecycle
tags: [sqlite, postgresql, projection-cas, custom-metadata, signing]
requires:
  - phase: 03-15
    provides: Exact-generation authority promotion and compatibility projections
provides:
  - Database-native projection comparison and mutation for independent SQLite writers
  - PostgreSQL per-key transaction locking before projection comparison
  - Cached SQL custom-metadata protocol delegation and signed PostgreSQL metadata parity
affects: [03-17, unified-cache, blobstore, metadata-projection]
actuals:
  tokens: 13165
  tasks: 3
  commits: 6
tech-stack:
  added: []
  patterns:
    - Projection comparison, mutation, link ownership, and result classification share one backend transaction.
    - Metadata wrappers delegate explicit custom-metadata protocols instead of exposing backend session factories.
    - Signature-visible mapping inputs are canonicalized before signing and restored at their nested read path.
key-files:
  created:
    - tests/test_projection_sql_atomicity.py
  modified:
    - src/cacheness/core.py
    - src/cacheness/metadata.py
    - src/cacheness/storage/backends/postgresql_backend.py
    - tests/test_cached_custom_metadata.py
key-decisions:
  - "SQLite acquires BEGIN IMMEDIATE before its projection read so independent writers classify one exact winner and one mismatch."
  - "PostgreSQL takes a bound, stable per-key advisory transaction lock before both absent and present-row comparisons."
  - "CachedMetadataBackend owns no raw SessionLocal or engine aliases; it delegates exact-current custom-metadata operations and managed sessions."
  - "Authority signing canonicalizes cache-key parameter order because the signer renders mappings through str(dict)."
patterns-established:
  - "Use deterministic transaction-boundary hooks and process barriers, never timing-only thread races, to prove database coordination."
  - "Fail closed when persisted signature inputs cannot be decoded; do not mutate canonical authority evidence during read failure."
requirements-completed: [STOR-03, STOR-04, STOR-05, STOR-06, STOR-07]
coverage:
  - id: D1
    description: "Independent SQLite and PostgreSQL-mapped projection schedules produce one applied mutation and one exact mismatch without raw race errors."
    requirement: STOR-03
    verification:
      - kind: integration
        ref: "tests/test_projection_sql_atomicity.py -k 'sqlite and (absent or existing) or postgresql and (absence or existing or transaction or links)'"
        status: pass
    human_judgment: false
  - id: D2
    description: "Memory-cached SQLite and PostgreSQL-mapped facades preserve exact-current custom metadata and retire stale links."
    requirement: STOR-04
    verification:
      - kind: integration
        ref: "tests/test_cached_custom_metadata.py -k 'sqlite or postgresql or cached or replacement or stale'"
        status: pass
    human_judgment: false
  - id: D3
    description: "A signed PostgreSQL-mapped projection retains nested cache-key parameters and remains live across public reads and statistics."
    requirement: STOR-05
    verification:
      - kind: integration
        ref: "tests/test_cached_custom_metadata.py -k 'signed and cache_key_params or malformed_key_params'"
        status: pass
    human_judgment: false
duration: 33min
completed: 2026-09-06
status: complete
---

# Phase 03 Plan 16: SQL Atomicity and Metadata Parity Summary

**SQLite and PostgreSQL projection paths now make per-key decisions transactionally, while cached SQL metadata and signed PostgreSQL entries retain one exact, live authority projection.**

## Performance

- **Duration:** 33min
- **Started:** 2026-09-06T06:27:55Z
- **Completed:** 2026-09-06T07:00:06Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- SQLite begins its writer transaction before reading a projection, and its comparison, row transition, `CacheMetadataLink` retirement, and mismatch classification complete together. Deterministic independent-adapter and spawned-process schedules prove exactly one winner for both absent and existing-row races.
- PostgreSQL takes a bound `pg_advisory_xact_lock(hashtextextended(...))` transaction guard before selection. Dialect compilation and controlled schedules cover expected absence, replacement, same-token refresh, links, and mismatch behavior without claiming a live PostgreSQL service test.
- `CachedMetadataBackend` delegates explicit exact-current storage and managed custom-metadata session protocols to SQLite/PostgreSQL backends. Public memory-cached facade workflows retain only M2 metadata and reject delayed M1 linkage.
- PostgreSQL now restores decoded `cache_key_params` under nested metadata used by signature verification, retains a same-object top-level compatibility alias, and rejects malformed persisted data without retirement. Authority signing canonicalizes map order so a valid entry survives public get, list, and stats reads.

## Task Commits

1. **Task 1: Prove one atomic SQL projection winner across independent adapters** - `375ced8` (test), `d80a8b3` (feat)
2. **Task 2: Delegate cached SQL custom metadata through explicit backend protocols** - `317da28` (test), `d31e40e` (feat)
3. **Task 3: Restore PostgreSQL nested signing inputs and prevent valid-entry retirement** - `6ca32e0` (test), `fafd682` (feat)

## Files Created/Modified

- `src/cacheness/metadata.py` - Adds SQLite transaction admission plus explicit custom-metadata protocol implementations and cached-wrapper delegation.
- `src/cacheness/storage/backends/postgresql_backend.py` - Adds advisory-lock transaction ordering, exact current metadata handling, and fail-closed nested parameter decoding.
- `src/cacheness/core.py` - Uses backend custom-metadata protocols and canonicalizes authority signing inputs.
- `tests/test_projection_sql_atomicity.py` - Exercises independent SQLite process/adaptor races and PostgreSQL transaction ordering.
- `tests/test_cached_custom_metadata.py` - Covers public cached custom-metadata and signed PostgreSQL-mapped workflows.

## Decisions Made

- A database writer transaction, rather than an instance lock, is the atomicity boundary for SQLite projection CAS.
- PostgreSQL uses a fixed-seed, bound key-derived advisory lock before both presence and absence decisions.
- Cached wrappers preserve SQL custom-metadata capability only through explicit methods and context managers; wrappers never surface a raw session/engine alias.
- Signing data must have deterministic mapping order before HMAC creation, because the existing signer serializes mapping values with `str(dict)`.

## Verification

- Passed `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q tests/test_projection_sql_atomicity.py -k "sqlite and (absent or existing) or postgresql and (absence or existing or transaction or links)" -x -o log_cli=false` (4 passed).
- Passed `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q tests/test_cached_custom_metadata.py -k "sqlite or postgresql or cached or replacement or stale" -x -o log_cli=false` (4 passed).
- Passed `uv run --isolated --python 3.11 --all-extras --group dev --frozen pytest -q tests/test_cached_custom_metadata.py -k "signed and cache_key_params or malformed_key_params" -x -o log_cli=false` (2 passed).
- Passed the combined SQL projection/custom-metadata regression group (45 passed): `tests/test_projection_sql_atomicity.py`, `tests/test_cached_custom_metadata.py`, `tests/test_projection_mutation_contract.py`, and `tests/test_custom_metadata.py`.
- `tests/test_cached_custom_metadata.py` is Ruff-clean. The PostgreSQL source module retains five pre-existing Ruff findings for optional-import and import-placement conventions; none were introduced by this plan.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Database transaction bug] Avoided opening a second SQLite connection while a writer transaction owns projection mutation**
- **Found during:** Task 1
- **Issue:** Link-table inspection used a separate engine/bind connection after `BEGIN IMMEDIATE`, which could contend with the transaction that was meant to own the entire decision.
- **Fix:** Inspect the current session connection so comparison, link ownership, and classification remain in the writer transaction.
- **Files modified:** `src/cacheness/metadata.py`, `tests/test_projection_sql_atomicity.py`
- **Verification:** Independent thread/process race schedules and projection-contract regressions pass.
- **Committed in:** `d80a8b3`

**2. [Rule 1 - Typed conflict dependency] Imported the custom metadata conflict error at the backend boundary**
- **Found during:** Task 2
- **Issue:** The new exact-current SQLite custom-metadata protocol attempted to raise the typed lifecycle conflict without importing it.
- **Fix:** Added the domain error import and verified stale insertion rolls back without an object or link.
- **Files modified:** `src/cacheness/metadata.py`
- **Verification:** Cached SQLite/PostgreSQL-mapped facade tests pass.
- **Committed in:** `d31e40e`

**3. [Rule 1 - Signature parity] Canonicalized signed cache-key parameter ordering**
- **Found during:** Task 3
- **Issue:** Lifecycle manifest canonical JSON and PostgreSQL read-back could present the same parameter mapping in different insertion orders, while HMAC input used order-sensitive `str(dict)`.
- **Fix:** Canonicalize the map before signing and restore the decoded value at the nested signature-visible metadata path.
- **Files modified:** `src/cacheness/core.py`, `src/cacheness/storage/backends/postgresql_backend.py`, `tests/test_cached_custom_metadata.py`
- **Verification:** Signed public PostgreSQL-mapped get/list/stats workflow and malformed fail-closed coverage pass.
- **Committed in:** `fafd682`

**Total deviations:** 3 auto-fixed (Rule 1)

## Issues Encountered

- Native Windows and a live PostgreSQL service are unavailable in this environment. The PostgreSQL coverage is deliberately database-dialect and SQLite-mapped protocol parity evidence, not live-service qualification.
- The known `SqliteBackend` interpreter-shutdown diagnostic remains deferred to Plan 03-17; this plan neither masks nor claims to resolve it.

## User Setup Required

None - no external service configuration was required for deterministic Python 3.11 coverage.

## Next Phase Readiness

Plan 03-17 can consume database-native projection and metadata protocol contracts. It remains responsible for the explicitly deferred SQLite shutdown diagnostic and any separate native Windows/live PostgreSQL qualification.

## Self-Check: PASSED

- Confirmed all five plan source/test artifacts and this summary exist.
- Confirmed the TDD and implementation commits `375ced8`, `d80a8b3`, `317da28`, `d31e40e`, `6ca32e0`, and `fafd682` exist in repository history.
- Scanned plan-owned source and test files for intentional placeholders or rendering stubs; none were introduced.

---
*Phase: 03-atomic-lifecycle-and-recovery-engine*
*Completed: 2026-09-06*
