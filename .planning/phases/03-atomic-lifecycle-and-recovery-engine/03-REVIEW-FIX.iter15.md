---
phase: 03-atomic-lifecycle-and-recovery-engine
fixed_at: 2026-09-01T22:05:00Z
review_path: .planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 13
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-01T22:05:00Z  
**Source review:** `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md`  
**Iteration:** 13

## Summary

- Findings in scope: 5
- Fixed: 5
- Skipped: 0
- Source and regression-test commit: `36a3110` (`fix(03): harden lifecycle recovery publication`)

## Fixed Issues

### CR-01: Key readiness is observed before exact initialization authority

**Files modified:** `src/cacheness/storage/integrity.py`, `src/cacheness/storage/path_security.py`, `tests/test_blob_store_integrity.py`  
**Commit:** `36a3110`

**Applied fix:** Added a separate stable initialization lock that is acquired before any EEXIST key bytes are read. Short crash-partial key and ready inodes are retired only under that authority and exact identity checks; complete ready bytes re-execute their platform acknowledgement before use. Windows now flushes the same retained descriptor that supplied its native volume/file identity, rather than trusting a second pathname stat. Added cross-process crash-partial-key and partial-ready recovery coverage.

### CR-02: Pending and sidecar paging is not name-bounded or reliably resumable

**Files modified:** `src/cacheness/storage/path_security.py`, `src/cacheness/storage/operation_repository.py`, `src/cacheness/storage/reconciliation.py`, `tests/test_blob_store_reconciliation.py`  
**Commit:** `36a3110`

**Applied fix:** Replaced `nsmallest` materialization for pending and reconciliation-sidecar pages with a managed streaming bounded directory slice, including opaque encrypted cursors that can advance past malformed candidates. Pending is now a first-class resume source, so a pending-only partial report emits a v2 resume token and reaches its final page.

### CR-03: Authenticated sidecars can bind the wrong primary and abort later work

**Files modified:** `src/cacheness/storage/reconciliation.py`, `tests/test_blob_store_reconciliation.py`  
**Commit:** `36a3110`

**Applied fix:** Sidecars are now checked against the exact current primary bytes, proposed recovery action, and legal checkpoint state before either sidecar or primary is treated as attached/safe. A signed mismatch becomes a stable blocked finding, preserves both records, does not consume the shared action budget, and no longer prevents later safe reconciliation work. Apply isolates checkpoint binding conflicts per finding.

### WR-01: Released v1 resume tokens are rejected

**Files modified:** `src/cacheness/storage/reconciliation.py`, `tests/test_blob_store_reconciliation.py`  
**Commit:** `36a3110`

**Applied fix:** The compatibility decoder now accepts the released three-field v1 schema (`manifest`, `operation`, `priority`), maps v2-only cursors to `None`, and restricts v1 priority to its historic values. A fixed authenticated pre-v2 token vector verifies continuation behavior.

### WR-02: Payload deletion can be called again after a checkpoint-bound interruption

**Files modified:** `src/cacheness/storage/reconciliation.py`, `tests/test_blob_store_reconciliation.py`  
**Commit:** `36a3110`

**Applied fix:** Tombstone reconciliation proves payload absence before issuing a delete on every reopen. After any post-effect checkpoint-bound `BaseException`, it attempts the monotonic durable acknowledgement without replaying fault hooks and preserves the original interruption. The seam matrix now counts delete invocations separately from successful effects and requires exactly one call.

## Verification

All focused checks below ran from the **main checkout** after the commit.

- `tests/test_blob_store_integrity.py` and `tests/test_blob_store_reconciliation.py`: passed.
- Complete focused Phase 3 regression command: passed, including atomic lifecycle, close, concurrency, integrity, read-contract, reconciliation, clear, containment, CAS, configuration, and public API suites.
- Complete repository suite: passed with the declared `recommended`, `dataframes`, `sql`, `s3`, and `postgresql` test dependencies.
- CPython 3.11.16: `compileall` passed; `import cacheness` passed with the declared `recommended` extra.
- Changed-file Ruff: passed.
- `uv lock --check` and `git diff --check`: passed before commit.

## Residuals

- Native Windows execution remains unavailable on this host. The source now binds the retained Win32 handle identity/flush contract, but a Windows runner is still required evidence.
- A bare install still cannot collect the full suite without NumPy and related optional test dependencies. This existing packaging contract issue is recorded in protected planning concerns and was not changed by this lifecycle fix.

---

_Fixed: 2026-09-01T22:05:00Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 13_
