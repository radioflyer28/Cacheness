---
phase: 03-atomic-lifecycle-and-recovery-engine
fixed_at: 2026-09-01T23:40:00Z
review_path: .planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 14
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-01T23:40:00Z  
**Source review:** `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md`  
**Iteration:** 14

## Summary

- Findings in scope: 3
- Fixed: 3
- Skipped: 0
- Source commit: `04395ab` (`fix(03): bound lifecycle initialization and reconciliation inventory`)
- Follow-up injected-Windows regression test: `d6bfb0b` (`test(03): cover win32 initialization lock retry`)

## Fixed Issues

### CR-01: First-key initialization could wait indefinitely

**Files modified:** `src/cacheness/config.py`, `src/cacheness/error_handling.py`, `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/coordination.py`, `src/cacheness/storage/integrity.py`, `src/cacheness/storage/__init__.py`, `tests/test_blob_store_integrity.py`, `tests/test_public_api_contract.py`  
**Commits:** `04395ab`, `d6bfb0b`

**Applied fix:** Added policy-owned initialization deadline/retry limits and a stable public lifecycle-timeout outcome. First-key authority now retries nonblocking `LOCK_NB`/`LOCKFILE_FAIL_IMMEDIATELY` acquisition until the deadline, closes the exact descriptor before timing out, and never reads or retires a winner key while admission is unavailable. Replaced the global guard with reference-counted per-key guards that retire at zero references. BlobStore preserves the timeout instead of translating it to an authentication failure. Regression coverage exercises live cross-process timeout, release-before-deadline, process-loss recovery, parallel independent stores, and guard retirement.

### CR-02: Operation, pending, and sidecar cursors could skip or livelock

**Files modified:** `src/cacheness/config.py`, `src/cacheness/storage/path_security.py`, `src/cacheness/storage/operation_repository.py`, `src/cacheness/storage/manifest_repository.py`, `src/cacheness/storage/reconciliation.py`, `tests/test_manifest_repository_cas.py`  
**Commit:** `04395ab`

**Applied fix:** Replaced mixed filesystem-order/lexical paging with an explicit separately bounded lexical inventory. The independent `max_inventory_items` policy bounds directory/key inspection; pages continue to bound raw record reads and reconciliation actions. Over-limit inventories fail closed instead of silently skipping evidence. Operation, pending, and sidecar pages all use the same full-filename cursor namespace; old bare sidecar IDs remain accepted when decoding prior tokens. Removed whole-namespace `nsmallest` paging and added reverse-creation traversal coverage (`z`, `a`, `b`).

### CR-03: Apply double-charged completed orphan sidecars and leaked exact CAS races

**Files modified:** `src/cacheness/storage/reconciliation.py`, `tests/test_blob_store_reconciliation.py`  
**Commit:** `04395ab`

**Applied fix:** Reconciliation now applies a single normalized action stream. Completed orphan sidecars are exact revalidated retirement actions in that stream, so an attempted action consumes the shared mutation budget once and cannot be revisited as a no-op primary action. Exact lifecycle conflicts are caught per finding, reported as stable blocked conflict dispositions, and do not prevent later independent actions. Coverage proves one orphan sidecar plus one safe primary completes exactly two actions under a budget of two.

## Verification

All gates below ran from the **main checkout** after the source/test commit.

- Complete Phase 3 lifecycle corpus: passed; two documented skips (Windows junction and unavailable device node).
- Full Python 3.13 suite with `recommended`, `dataframes`, `sql`, `s3`, and `postgresql` test dependencies: passed; only documented platform/service/TensorFlow skips.
- CPython 3.11.16: `compileall`, import with `recommended`, and focused integrity/reconciliation/manifest-repository tests passed.
- Changed-path Ruff: passed. `src/cacheness/config.py` retains exactly two pre-existing F841 findings, verified against `HEAD`; this fix added none.
- `uv lock --check` and `git diff --check`: passed.

## Residuals

- Native Windows execution is unavailable on this host. POSIX live-holder coverage and injected Windows nonblocking-adapter semantics are present; native Windows CI is still required evidence.
- Inventory is intentionally fail-closed at `LifecycleLimits.max_inventory_items` (default 4,096). A backend-native durable unbounded inventory index remains a future backend-scaling concern; no evidence is silently skipped beyond the current bounded local contract.

---

_Fixed: 2026-09-01T23:40:00Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 14_
