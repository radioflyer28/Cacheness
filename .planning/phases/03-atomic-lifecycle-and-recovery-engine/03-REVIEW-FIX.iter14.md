---
phase: 03-atomic-lifecycle-and-recovery-engine
fixed_at: 2026-09-01T20:37:31Z
review_path: .planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 12
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-01T20:37:31Z  
**Source review:** `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md`  
**Iteration:** 12

## Summary

- Findings in scope: 5
- Fixed: 5
- Skipped: 0
- Source and regression-test commit: `8ca6209` (`fix(03): harden lifecycle recovery contracts`)

## Fixed Issues

### CR-01: Manifest-key acknowledgement has no independently durable completion state

**Files modified:** `src/cacheness/storage/integrity.py`, `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/path_security.py`, `tests/test_blob_store_integrity.py`, `tests/test_manifest_repository_cas.py`  
**Commit:** `8ca6209`

**Applied fix:** Added a keyed, exact-identity readiness record that is published only after a retained-handle acknowledgement; a same-process initialization lock and interprocess file lock serialize first initialization. Unacknowledged keys are unusable and safely resumable after close or acknowledgement failures. Windows flushes re-attest the regular, non-reparse, single-link file contract on the retained handle. BlobStore validates key-provider returns and translates ordinary provider failures with provider and operation context while preserving typed manifest-repository failures.

### CR-02: Pending lifecycle-control recovery inventory is not stably bounded

**Files modified:** `src/cacheness/storage/operation_repository.py`, `src/cacheness/storage/reconciliation.py`, `tests/test_blob_store_reconciliation.py`  
**Commit:** `8ca6209`

**Applied fix:** Added bounded pending-control pages and a durable pending-recovery cursor. Invalid, unreadable, digest-mismatched, and blocked candidates advance scheduling without consuming recovery-action capacity; valid work remains reachable on later calls. Reconciliation now reports pending residue in dry-run mode without granting it mutation authority, and resume tokens carry an independent pending cursor.

### CR-03: Reconciliation sidecars can hide debt or overrun a shared apply budget

**Files modified:** `src/cacheness/storage/lifecycle.py`, `src/cacheness/storage/operation_repository.py`, `src/cacheness/storage/reconciliation.py`, `tests/test_blob_store_reconciliation.py`  
**Commit:** `8ca6209`

**Applied fix:** Sidecars now have independent bounded pages/cursors and authenticated classification. Startup defers a tombstone only for a signature- and primary-byte-bound sidecar; malformed matching sidecars remain visible, inert, and non-fatal. Apply uses one total action budget across orphan-sidecar retirement and primary work. A signed post-effect tombstone checkpoint can resume terminal work after the primary manifest has been removed, without replaying a destructive effect.

### WR-01: Tombstone reconciliation does not acknowledge post-effect interruption seams

**Files modified:** `src/cacheness/storage/reconciliation.py`, `tests/test_blob_store_reconciliation.py`  
**Commit:** `8ca6209`

**Applied fix:** Payload deletion and manifest removal now prove their observable effect and durably checkpoint it before propagating a `BaseException`. Added before/inside/after fault seams for payload, tombstone removal, checkpoint transitions, primary retirement, and sidecar retirement. Parametrized regression coverage proves convergence with exactly one payload delete and one manifest removal for every seam.

### WR-02: Empty reconciliation apply initializes a signing key

**Files modified:** `src/cacheness/storage/reconciliation.py`, `tests/test_blob_store_reconciliation.py`  
**Commit:** `8ca6209`

**Applied fix:** Reconciliation inventories sidecars and pending controls before loading a manifest key. A pristine `reconcile(apply=True)` is now an idempotent no-op for memory, JSON, and SQLite metadata backends and does not create a trust root.

## Verification

All final gates below ran from the **main checkout** after fast-forwarding `8ca6209`; the earlier isolated worktree was removed transactionally.

- Deterministic Phase 3 regression subset: `207 passed`.
- Python 3.13: compile check and complete repository suite passed (with live logs disabled).
- Python 3.11: compile/import check and focused lifecycle regression: `28 passed`.
- Changed-file Ruff: passed.
- `uv lock --check`: passed.
- `git diff --check HEAD^ HEAD`: passed.

## Residuals

- Native Windows execution was not available on this host. The Windows source contract is covered by simulated native API tests; a real Windows runner remains required evidence.
- The complete local suite retains 31 expected skips for unavailable Polars, PostgreSQL/psycopg, TensorFlow, Windows-junction, and device-node capabilities. No test failed.

---

_Fixed: 2026-09-01T20:37:31Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 12_
