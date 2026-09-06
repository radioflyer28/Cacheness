---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-06T10:36:07Z
depth: deep
files_reviewed: 23
files_reviewed_list:
  - benchmarks/lifecycle_authority_baseline.json
  - benchmarks/lifecycle_authority_benchmark.py
  - src/cacheness/core.py
  - src/cacheness/metadata.py
  - src/cacheness/storage/backends/postgresql_backend.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/coordination.py
  - src/cacheness/storage/lifecycle.py
  - src/cacheness/storage/sqlite_lifecycle_authority.py
  - tests/test_blob_store_concurrency.py
  - tests/test_blob_store_read_contract.py
  - tests/test_blob_store_reconciliation.py
  - tests/test_cache_integrity.py
  - tests/test_cached_custom_metadata.py
  - tests/test_cached_query_meta.py
  - tests/test_phase3_gap_acceptance.py
  - tests/test_phase3_postreview_concurrency.py
  - tests/test_projection_sql_atomicity.py
  - tests/test_sqlite_authority_admission.py
  - tests/test_sqlite_bootstrap_concurrency.py
  - tests/test_sqlite_concurrency.py
  - tests/test_sqlite_concurrency_temp.py
  - tests/test_unified_cache_adversarial_lifecycle.py
findings:
  critical: 2
  warning: 0
  info: 0
  total: 2
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-06T10:36:07Z
**Depth:** deep
**Files Reviewed:** 23
**Status:** issues_found

## Summary

This fresh review covers the Phase 3 implementation from `f410621` through
current HEAD `74ee9fe`, with a full re-review of Plan 03-18 and the four findings
recorded at `28755ab`. All four of those findings are closed in their original
forms: failed pre-entry clear tickets retire exactly, SQLite reapplies a
conservative remaining busy budget before `BEGIN IMMEDIATE`, child processes
replace both the copied writer registry and its potentially locked global lock,
and cached metadata queries delegate through an explicit capability.

The phase is still not ready to ship. The new constructor-owned metadata
bootstrap is not atomic across concurrent constructors and reproducibly fails
with a check-then-create schema race. The new query implementation also turns
corrupt committed `cache_key_params` into an apparently valid empty mapping,
contradicting the phase's fail-closed integrity and narrow-exception policy.

The focused Plan 03-18 suites passed (40 tests). The reconciliation clock test
now isolates its own two-call work deadline under normal authority setup. The
revised stress modules keep ordinary schedules strict and limit typed timeout
acceptance to the three named extreme schedules; no raw lock string or unrelated
error is accepted. Benchmark data and envelopes were not changed by Plan 03-18.
Native Windows remains `UNAVAILABLE` / `NOT_QUALIFIED`, and live PostgreSQL is
not claimed.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Concurrent fresh SQLite metadata constructors race during schema creation

**Classification:** BLOCKER

**File:** `src/cacheness/metadata.py:1703-1724`

**Issue:** `_bootstrap_database()` runs `Base.metadata.create_all(engine)` with
no database transaction or other cross-constructor serialization. SQLAlchemy's
`create_all()` checks whether a table exists and then issues `CREATE TABLE` as
separate steps. Two fresh engines can both observe absence before either creates
the table, after which one constructor fails with `sqlalchemy.exc.OperationalError:
table cache_entries already exists`. This is not only theoretical: 64 concurrent
fresh `SqliteBackend` constructors targeting one empty database produced 53
successes and 11 `OperationalError` failures. The provisional-engine disposal
prevents a pool leak, but does not make initialization idempotent. This violates
Plan 03-18's explicit requirement that constructor-owned WAL/schema bootstrap
be safe for concurrent fresh processes, and can prevent independent cache
instances from opening the same new root.

**Fix:** Serialize the schema check-and-create sequence with SQLite itself. Open
one bootstrap connection, acquire a bounded writer transaction before schema
inspection, and execute the `CREATE TABLE IF NOT EXISTS`/schema validation,
stats-row initialization, and required bootstrap work through that same
connection before commit. The solution must coordinate independent processes,
not only threads in one Python process, and should translate exhausted SQLite
busy waits into the domain metadata error while preserving the DBAPI cause.
Add deterministic thread and process tests that release multiple constructors
against one empty path simultaneously, require every constructor to succeed,
verify one valid schema/stats row and WAL mode, and prove all provisional engines
are disposed on a forced bootstrap failure.

### CR-02: `query_meta()` silently converts corrupt committed parameters to `{}`

**Classification:** BLOCKER

**File:** `src/cacheness/metadata.py:2257-2286`

**Issue:** The backend catches every exception from `json_loads()` and returns
`cache_key_params: {}`. An empty-filter `query_meta()` does not invoke a SQLite
JSON function, so malformed stored JSON reaches this handler and is presented
as valid empty parameters. A deterministic reproduction changed one live row's
`cache_key_params` to the invalid JSON string `"{"`; `query_meta()` returned the
live entry with `cache_key_params: {}` instead of failing closed. Filtered
queries can instead surface SQLite's `malformed JSON` as `DBAPIError`, which is
translated to `CacheMetadataError` and then suppressed by the facade. Thus the
same integrity defect is inconsistently falsified or hidden according to filter
shape. This directly contradicts Plan 03-18's requirement that programmer and
integrity failures propagate while only operational database failures retain
the logged-`None` policy.

**Fix:** Never substitute `{}` for undecodable persisted parameters. Validate
the JSON for every exact live `(cache_key, actual_path)` row before applying
filters and raise a domain `CacheIntegrityError` containing the affected key
when decoding fails or the decoded value is not the required mapping shape.
Keep that integrity exception outside the facade's `CacheMetadataError`
fallback. Distinguish operational lock/I/O `DBAPIError` from corruption-related
SQLite errors rather than treating every DBAPI failure as an ordinary query
miss. Add regressions for empty, string, and numeric filters over malformed live
metadata, asserting the same fail-closed exception and zero pooled connections
left checked out.

## Prior Finding Disposition

| Finding | Disposition | Reviewed evidence |
|---|---|---|
| `28755ab` CR-01: failed clear FIFO tombstone | CLOSED | `clear_operation()` discards the exact unentered ticket in `finally`; timeout and close-cancellation coverage proves later FIFO progress and zero accounting. |
| `28755ab` CR-02: additive SQLite busy budget | CLOSED | Remaining time is converted conservatively and installed immediately before `BEGIN IMMEDIATE`; combined observer-plus-writer coverage preserves `stage=sqlite_busy` and the SQLite cause within one deadline. |
| `28755ab` CR-03: copied admission state after fork | CLOSED | The child at-fork hook replaces both global registry objects; tests cover an owned per-path gate and a globally locked registry while inherited instances still fail the PID guard. |
| `28755ab` WR-01: cached `query_meta` capability loss | CLOSED | The facade uses `supports_entry_metadata_query()` and `query_entries_by_key_params()`; `CachedMetadataBackend` delegates both without exposing `SessionLocal`. |
| Earlier CR-01 through CR-09 | CLOSED | Facade admission, post-promotion cleanup, generation convergence, SQL projection serialization, bootstrap reclassification, cached custom metadata, canonical empty-state handling, locator containment, and PostgreSQL nested signed parameters remain intact. |
| Earlier WR-01/WR-02 | CLOSED | Explicit idempotent close remains in place and deterministic publication/projection/bootstrap/admission schedules remain covered. |

## Additional Plan 03-18 Assessment

- Exact `(cache_key, actual_path)` SQL filtering prevents a later same-key
  projection from being substituted for an earlier authority observation; a
  mismatch is omitted and no projection repair occurs on the query path.
- Pool checkout now runs connection-local PRAGMAs only. WAL/schema/stats and
  `PRAGMA optimize` moved to constructor bootstrap, and sessions return their
  connections on both success and translated DBAPI failure.
- The three extreme stress schedules preserve exact attempt accounting, at
  least 75 percent aggregate success, per-worker progress, committed-value
  checks, bounded child termination, and admission-registry retirement.
  Query/get-only and ordinary write/mixed/file/stats/integrity tests retain
  all-success behavior.
- The reconciliation time-limit fixture now completes setup using the normal
  authority budget before installing its finite reconciliation clock, and
  asserts exactly the two monotonic observations used to return a resume token
  without consuming an operation record.
- PostgreSQL projection locking and signed key-parameter compatibility were not
  changed by Plan 03-18. Static/fake-dialect evidence remains consistent, but no
  live PostgreSQL qualification is inferred from it.

---

_Reviewed: 2026-09-06T10:36:07Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
