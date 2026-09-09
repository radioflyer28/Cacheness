---
phase: 03-atomic-lifecycle-and-recovery-engine
fixed_at: 2026-09-01T20:22:00Z
review_path: /Users/akriz/code/cacheness/.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 11
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-01T20:22:00Z
**Source review:** `03-REVIEW.md`
**Iteration:** 11

## Summary

- Findings in scope: 7
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: Lifecycle payload cleanup bypasses the Win32 handle-delete protocol

**Files modified:** `src/cacheness/storage/blob_store.py`,
`src/cacheness/storage/path_security.py`,
`tests/test_blob_store_atomic_lifecycle.py`,
`tests/test_blob_store_reconciliation.py`,
`tests/test_manifest_repository_cas.py`
**Commit:** `921324d`
**Applied fix:** All lifecycle cleanup now calls `delete_durable()`. The Win32
path performs regular/reparse/link validation, identity capture, disposition, and
close through one retained handle; it no longer pre-verifies a pathname then
deletes a potentially substituted name. Regressions cover overwrite, delete,
clear, reconciliation candidate cleanup, and the native adapter contract.

### CR-02: Signing-key initialization can publish manifests before its trust root is durable

**Files modified:** `src/cacheness/storage/integrity.py`,
`src/cacheness/storage/blob_store.py`,
`tests/test_blob_store_integrity.py`
**Commit:** `921324d`
**Applied fix:** New file keys are closed and then acknowledged before use. POSIX
checks the exact key inode before and after an fsync of the identity-checked parent
directory. The D-22 Windows path documents and uses a reparse-safe regular-handle
flush primitive, while a public durability-provider seam permits a stronger
application keystore acknowledgement. Descriptor and injected-provider operational
failures are translated to `CacheBlobManifestUnauthenticatedError` with cause and
provider/operation context.

### CR-03: Tombstone reconciliation can retire resume evidence before checkpoint completion

**Files modified:** `src/cacheness/storage/reconciliation.py`,
`src/cacheness/storage/lifecycle.py`,
`tests/test_blob_store_reconciliation.py`
**Commit:** `921324d`
**Applied fix:** `COMPLETE_TOMBSTONE` is now a durable state machine:
`prepared` → `payload_deleted` → `tombstone_removed` → `completed`.
The primary lifecycle record remains until the completed sidecar is durable.
Startup recovery defers a checkpoint-owned tombstone to reconciliation, so every
fault seam can reopen and continue forward without replaying payload deletion or
stranding a prepared sidecar.

### WR-01: Digest-invalid pending controls starve the bounded recovery budget

**Files modified:** `src/cacheness/storage/operation_repository.py`,
`tests/test_blob_store_reconciliation.py`
**Commit:** `921324d`
**Applied fix:** Pending control candidates receive a bounded read and digest
check before consuming an action slot. Invalid bytes remain untouched, while a
later valid candidate can progress under a one-action limit.

### WR-02: Normal operations accept noncanonical raw manifests

**Files modified:** `src/cacheness/storage/blob_store.py`,
`tests/test_blob_store_integrity.py`
**Commit:** `921324d`
**Applied fix:** The shared authenticated manifest loader now requires the
signed model's exact canonical bytes to equal the repository bytes before any
handler, locator, or CAS expectation is used. Get, put, delete, metadata update,
exists, list, and reconciliation now all fail closed on reordered or
whitespace-padded records.

### WR-03: One malformed reconciliation sidecar can starve completed orphan cleanup

**Files modified:** `src/cacheness/storage/operation_repository.py`,
`src/cacheness/storage/reconciliation.py`,
`tests/test_blob_store_reconciliation.py`
**Commit:** `921324d`
**Applied fix:** Sidecar parsing is isolated. Malformed/untrusted sidecars remain
untouched and are surfaced as bounded blocked findings; only authenticated,
eligible completed-orphan retirement consumes the reconciliation action budget.

### WR-04: Reopen tombstone retirement can lose the authoritative conflict state

**Files modified:** `src/cacheness/storage/lifecycle.py`,
`tests/test_blob_store_atomic_lifecycle.py`
**Commit:** `921324d`
**Applied fix:** Reopen recovery now uses the same conflict-aware retirement
helper as direct delete. A later winner is preserved and a retirement failure is
a `CacheBlobRecoverableCleanupError` with explicit post-authority,
later-winner, conflict-type, and retirement-error context.

## Verification

Verification ran in the **main checkout**.

- Complete 11-module Phase 3 focused suite passed; only the documented Windows
  junction and unavailable device-node fixtures skipped.
- Complete repository suite passed on CPython 3.13.3, with documented optional and
  platform skips plus the existing collection warning.
- CPython 3.11.16 compileall passed. A base-install import still fails because
  NumPy is an existing undeclared mandatory import; this known packaging issue was
  not changed by Phase 3. With the declared `recommended` extra, import and the
  changed integrity/reconciliation/adapter regressions passed.
- Changed-path Ruff, `uv lock --check`, and `git diff --check` passed.
- Native Windows filesystem execution remains an explicit later platform/CI
  evidence gate. The source and adapter tests cover the D-22 one-user/session
  contract but do not claim a Windows-host run.

---

_Fixed: 2026-09-01T20:22:00Z_
_Fixer: the agent (gsd-code-fixer)_
_Iteration: 11_
