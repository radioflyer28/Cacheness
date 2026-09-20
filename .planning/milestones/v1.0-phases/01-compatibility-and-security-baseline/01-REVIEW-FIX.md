---
phase: 01
fixed_at: 2026-08-30T04:15:39Z
review_path: .planning/phases/01-compatibility-and-security-baseline/01-REVIEW.md
iteration: 4
findings_in_scope: 2
fixed: 2
skipped: 0
status: all_fixed
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-08-30T04:15:39Z
**Source review:** `.planning/phases/01-compatibility-and-security-baseline/01-REVIEW.md`
**Iteration:** 4

## Summary

- Findings in scope: 2
- Fixed: 2
- Skipped: 0

## Prior Iteration Closures

### CR-R1: A durably committed JSON entry is misclassified as uncommitted

**Files modified:** `src/cacheness/metadata.py`, `tests/test_clear_recovery.py`
**Commit:** `08b3f99`

JSON backup retirement now logs an unacknowledged cleanup barrier after the new
live document has already been file- and directory-synced; it does not report the
completed publication as a failure. That preserves high-level candidate ownership
at metadata authority. Tests cover first write for a new key and cross-format
overwrite through BlobStore and UnifiedCache, with backup-unlink and post-unlink
directory-fsync faults.

**Status:** fixed: requires human verification

### CR-R2: Failed committed-journal publication leaves prepared work live

**Files modified:** `src/cacheness/storage/clear_recovery.py`, `tests/test_clear_recovery.py`
**Commit:** `08b3f99`

Committed-journal replacement is now inside the rollback boundary. If the durable
journal is still exactly prepared, the coordinator restores payloads and the
complete metadata snapshot before returning the publication error. If committed or
unknown evidence may have been published, it retains that evidence, poisons the
live owner, and rejects normal operations until terminal recovery. Caught
BaseException interruptions after metadata clear poison the owner as well. The
matrix covers JSON, SQLite, and in-memory metadata at serialization,
candidate-write, replacement, and post-replace acknowledgement boundaries.

**Status:** fixed: requires human verification

### CR-R3: Ordinary writes bypass clear admission

**Files modified:** `src/cacheness/storage/clear_recovery.py`, `src/cacheness/storage/blob_store.py`, `src/cacheness/core.py`, `tests/test_clear_recovery.py`
**Commit:** `08b3f99`

BlobStore and UnifiedCache now take the same root-scoped in-process/advisory
admission around lifecycle operations. Normal operations block and serialize behind
another writer or clear; same-thread reentrancy rejects rather than deadlocking,
and startup recovery remains fail-fast.
Every mutation rechecks/reconciles journal evidence before publication. Reads reject
prepared/poisoned evidence but may observe the authoritative empty view after a
committed clear awaiting tombstone reclamation. Tests cover same instance,
preconstructed two-instance, and JSON/SQLite subprocess contenders. The live
threaded regressions pause clear immediately after its prepared journal is durable,
prove the put cannot finish while clear owns admission, then prove clear followed by
the put is linearizable: the new entry is readable and is the sole remaining
candidate. JSON writers refresh their live metadata under the acquired admission so
the preconstructed contender cannot resurrect the cleared snapshot. The existing
core concurrent-put contract remains passing.

The ordinary-operation policy intentionally blocks rather than returning a typed
contention error: it preserves the historical success semantics for concurrent
callers while making their outcome serializable. This is not a retry loop or an
accidental indefinite-wait regression. Advisory and process locks are released on
normal owner completion and process exit. A live in-process thread that hangs while
holding admission can block peers indefinitely by design; Phase 1 chooses that
correctness-over-availability tradeoff explicitly.

**Status:** fixed: requires human verification

## Fixed Issues (Iteration 4)

### CR-R4: Public metadata queries bypass prepared-clear admission

**Files modified:** `src/cacheness/core.py`, `tests/test_clear_recovery.py`
**Commit:** `cf3be4b`

`query_meta()`, `query_custom()`, and
`get_custom_metadata_for_entry()` now enter the same root read admission as
`get()`. The deprecated `query_custom_metadata()` reaches the decorated
`query_custom()` path. `query_custom_session()` holds admission from context
entry through the caller's complete `with` body, rather than only while it
constructs the returned context manager. Admission happens outside the query
methods' broad exception handlers, so prepared or poisoned recovery evidence
cannot become `None`, `[]`, or `{}`.

The SQLite regressions preconstruct a second cache, interrupt clear with a
prepared journal, and verify each public surface rejects the intermediate state.
They repeat that matrix for a poisoned local owner, verify a committed journal
exposes only the authoritative empty query result, and prove a live clear blocks
until a custom-query context exits.

**Status:** fixed: requires human verification

### CR-R5: JSON close can republish stale metadata after another instance clears

**Files modified:** `src/cacheness/metadata.py`, `tests/test_clear_recovery.py`
**Commit:** `cf3be4b`

`JsonBackend.close()` no longer writes its in-memory metadata snapshot. JSON
mutations already publish synchronously, so close has no pending state to flush;
republishing on close could make a preconstructed second instance resurrect
metadata that a first instance had authoritatively cleared. High-level
`BlobStore.close()` and `UnifiedCache.close()` inherit this non-mutating backend
close behavior without introducing a broader lifecycle or compare-and-swap
protocol.

Deterministic BlobStore and UnifiedCache tests perform A-write, B-preconstruct,
A-clear, B-close, then reopen and prove the store remains empty and the old
payload is absent.

**Status:** fixed: requires human verification

## Verification

Verification ran in the **main checkout**; no isolated worktree was used.

- Passed: `tests/test_clear_recovery.py -x`, including the 8-case live
  put-versus-clear JSON/SQLite × BlobStore/UnifiedCache × same/second-instance
  matrix.
- Passed: `tests/test_metadata.py tests/test_clear_recovery.py tests/test_cache_integrity.py tests/test_filesystem_containment.py tests/test_core.py::TestCacheness::test_concurrent_access -x` (one expected Windows-junction skip).
- Passed: `.venv/bin/ruff check src/cacheness/storage/clear_recovery.py src/cacheness/storage/blob_store.py tests/test_clear_recovery.py`.
- Passed Phase 1 quality gate in its normal `uv` environment:
  `uv run pytest -q -o log_cli=false tests/test_phase1_quality_gates.py`.
- Passed after iteration 4: `tests/test_clear_recovery.py`,
  `tests/test_metadata.py`, `tests/test_cache_integrity.py`,
  `tests/test_filesystem_containment.py`, and
  `tests/test_core.py::TestCacheness::test_concurrent_access` (one expected
  Windows-junction skip); `tests/test_query_meta.py`,
  `tests/test_query_meta_security.py`, and `tests/test_custom_metadata.py`.
- Passed after iteration 4: `.venv/bin/ruff check
  src/cacheness/storage/clear_recovery.py src/cacheness/storage/blob_store.py
  tests/test_clear_recovery.py`; the pre-existing nine Ruff findings in
  `core.py`/`metadata.py` remain unchanged.
- Passed complete suite in the same environment:
  `uv run pytest -q -o log_cli=false` (expected optional PostgreSQL/TensorFlow
  and Windows-junction skips, plus the existing dataclass collection warning).
- Parse checks and `git diff --check` passed.

---

_Fixed: 2026-08-30T04:15:39Z_
_Fixer: the agent (gsd-code-fixer)_
_Iteration: 4_
