---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-06T08:34:02Z
depth: deep
files_reviewed: 18
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
  - tests/test_cache_integrity.py
  - tests/test_cached_custom_metadata.py
  - tests/test_phase3_gap_acceptance.py
  - tests/test_projection_sql_atomicity.py
  - tests/test_sqlite_authority_admission.py
  - tests/test_sqlite_bootstrap_concurrency.py
  - tests/test_unified_cache_adversarial_lifecycle.py
findings:
  critical: 3
  warning: 1
  info: 0
  total: 4
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-06T08:34:02Z
**Depth:** deep
**Files Reviewed:** 18
**Status:** issues_found

## Summary

This review covered the Phase 3 implementation delta from `f410621` through
`6402707`, with an adversarial focus on Plans 03-15, 03-16, and 03-17. The
implementation closes the previously reported CR-01 through CR-09 and WR-01
through WR-02 defects in their original forms. In particular, publication is
now admitted at the facade boundary, deferred cleanup follows promotion,
projection writers serialize their compare-and-write transaction, cached
custom metadata delegates its capability and session, PostgreSQL preserves
nested signed key parameters, bootstrap conflicts are reclassified, and
explicit backend close replaces destructor cleanup.

The phase is not ready to ship. Three newly demonstrated concurrency/deadline
defects can strand the process or violate the lifecycle contract: an abandoned
clear ticket permanently blocks later clears, SQLite lock acquisition can spend
a fresh timeout after earlier stages have consumed the absolute budget, and a
forked child inherits the parent's process-local admission registry. A separate
public API compatibility defect makes `query_meta()` silently unusable through
the supported cached metadata wrapper.

Native Windows remains `UNAVAILABLE` / `NOT_QUALIFIED`. Live PostgreSQL behavior
is not claimed by this review; the PostgreSQL assessment is limited to code,
compiled SQL, and the checked fake-dialect tests. The checked benchmark baseline
retains the legacy additive scenarios and records its source/harness provenance.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: A timed-out queued clear leaves a permanent FIFO tombstone

**Classification:** BLOCKER

**File:** `src/cacheness/storage/coordination.py:205-241`

**Issue:** `clear_operation()` appends its ticket before waiting, but its
`finally` block removes the ticket only when `entered` is true. If
`_wait_for_gate()` raises `CacheBlobCloseTimeoutError` while this clear is
queued behind another clear, the unentered ticket remains at the head of
`_clear_tickets`. No live operation owns that ticket, so every subsequent clear
queues behind it and times out as well. The registry also grows by one ticket
per retry. A deterministic public-API reproduction paused one clear at its
snapshot, let a second clear time out, released the first, and observed a third
clear time out with the stale queue growing from one to two tickets. This
violates FIFO progress, bounded bookkeeping, and repeatable clear convergence.

**Fix:** Remove the exact ticket on every exit path, including failure before
entry, and notify all waiters after removal. For example, make the `finally`
block acquire the condition and call the existing exact-ticket discard helper
when `entered` is false; retain the current active-clear teardown when it is
true. Add a deterministic regression test that queues a timed-out clear behind
an active clear, releases the active clear, then proves a later clear succeeds
and the queue/refcount state is empty.

### CR-02: SQLite lock acquisition can exceed the single absolute deadline

**Classification:** BLOCKER

**File:** `src/cacheness/storage/sqlite_lifecycle_authority.py:1133-1142`

**File:** `src/cacheness/storage/sqlite_lifecycle_authority.py:1235-1253`

**Issue:** `_connection()` calculates the remaining budget once and uses it as
the SQLite connection `timeout`. `_transaction()` then spends time in
connection preflight, process-local FIFO admission, and observer dispatch before
executing `BEGIN IMMEDIATE`, but the connection's busy timeout is never reduced
to the budget remaining at that point. Consequently SQLite can wait for the
original timeout after earlier stages have already consumed most of the same
absolute deadline. A deterministic reproduction with a 0.187-second deadline,
a 0.12-second admission-observer delay, and an external `BEGIN IMMEDIATE`
writer raised `stage=sqlite_busy` after approximately 0.368 seconds. This breaks
Plan 03-17's central promise that preflight, FIFO admission, dispatch, and
SQLite lock acquisition share one bounded deadline.

**Fix:** Immediately before `BEGIN IMMEDIATE`, derive SQLite's busy timeout from
`_remaining_for_stage(absolute_deadline)` and apply that remaining duration to
the connection (for example with `PRAGMA busy_timeout`, using a conservative
millisecond conversion that cannot extend the absolute deadline). Preserve the
underlying `sqlite3.OperationalError` when translating a genuine busy failure.
Add a deterministic combined-delay regression: consume budget in dispatch while
an external writer owns SQLite, then assert total monotonic elapsed time remains
within one configured deadline plus a small scheduler tolerance. The existing
tests exercise queued admission and SQLite busy independently, so they cannot
detect this additive timeout.

### CR-03: Forked children inherit stale process-local admission locks and tickets

**Classification:** BLOCKER

**File:** `src/cacheness/storage/sqlite_lifecycle_authority.py:93-96`

**File:** `src/cacheness/storage/sqlite_lifecycle_authority.py:631-640`

**File:** `src/cacheness/storage/sqlite_lifecycle_authority.py:163-170`

**Issue:** The writer-admission registry and its lock are module globals keyed
only by canonical database path. The instance PID guard correctly rejects an
authority object inherited across `fork()`, but it does not protect a fresh
authority constructed in the child: that new object reuses the inherited gate.
If the parent forked while another thread owned a ticket, the child waits behind
a copied owner that can never release in the child's memory. If the vanished
thread held the registry or condition lock at the instant of fork, the child can
deadlock before reaching the bounded wait. A deterministic reproduction forked
while a parent writer was paused at `writer_admission.eligible`; a fresh child
authority for the same root timed out at `stage=writer_admission` instead of
operating with fresh child coordination state. This contradicts the documented
fork contract: inherited authorities fail closed, while a newly constructed
child authority must work.

**Fix:** Make admission state process-scoped and reinitialize it after fork.
Keying gates by `(pid, canonical_path)` is necessary but not sufficient if the
global registry lock itself was inherited while locked; register an
`os.register_at_fork(after_in_child=...)` handler that replaces both the
registry and its lock in the child. Add a regression that forks while a parent
gate is actively owned, verifies the inherited authority is rejected, and
verifies a fresh child authority completes without observing the parent's
ticket/refcount state.

## Warnings

### WR-01: `query_meta()` silently fails through `CachedMetadataBackend`

**Classification:** WARNING

**File:** `src/cacheness/core.py:648-670`

**File:** `src/cacheness/metadata.py:553-557`

**Issue:** `UnifiedCache.query_meta()` directly requires and accesses
`self.metadata_backend.SessionLocal`. `CachedMetadataBackend`, used by the
supported `enable_memory_cache=True` configuration, deliberately exposes its
custom metadata capability and session through delegation but does not expose
`SessionLocal`. As a result, `query_meta()` logs that the backend does not
support SQL queries and returns `None` even when the wrapped SQLite backend does
support them. This is an API/capability mismatch adjacent to the Plan 03-16
custom-metadata delegation fix: custom metadata now works through the wrapper,
but the built-in metadata-query API still does not.

**Fix:** Define a backend-neutral query/session capability for built-in entry
metadata and delegate it through `CachedMetadataBackend`, then route
`query_meta()` through that capability instead of inspecting a concrete
`SessionLocal` attribute. Add tests for `store_cache_key_params=True` with
memory caching enabled, covering both a matching query and an unsupported
backend's explicit error/result policy.

## Prior Finding Disposition

| Prior finding | Disposition | Reviewed evidence |
|---|---|---|
| CR-01 admitted publication gap | CLOSED | Public `put()` enters admission before authority publication and delegates nested work without reacquiring the facade gate. |
| CR-02 destructive pre-promotion hook | CLOSED | Projection hooks prepare state; unlink/deferred cleanup happens only after promotion. |
| CR-03 peer-token adoption | CLOSED | Publishers retain their own token and converge only after verifying the winner's exact promoted generation. |
| CR-04 projection compare/write race | CLOSED | SQLite uses `BEGIN IMMEDIATE`; PostgreSQL obtains a transaction advisory lock before the row compare/write sequence. Live PostgreSQL remains unqualified. |
| CR-05 fresh-root bootstrap race | CLOSED | Concurrent root/leaf creation conflicts are caught and reclassified through canonical bootstrap inspection. |
| CR-06 cached custom metadata | CLOSED | The cache wrapper delegates support, storage, and scoped custom-metadata sessions. |
| CR-07 empty-state inference | CLOSED | Canonical roots use authority state; only recognized legacy layouts use legacy inference. |
| CR-08 hostile observed locator | CLOSED | Observed payload locators are normalized/contained before mutation or cleanup. |
| CR-09 PostgreSQL signed key parameters | CLOSED | Nested `key_params` are preserved with the compatibility alias and malformed non-mappings fail closed. |
| WR-01 destructor cleanup | CLOSED | Explicit idempotent close is the lifecycle mechanism; the backend destructor was removed. |
| WR-02 deterministic interleavings | CLOSED AS ORIGINALLY FILED | The required publication/projection/bootstrap/admission interleavings were added, though the newly identified timeout-plus-busy and active-gate fork schedules need their own regressions. |

---

_Reviewed: 2026-09-06T08:34:02Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
