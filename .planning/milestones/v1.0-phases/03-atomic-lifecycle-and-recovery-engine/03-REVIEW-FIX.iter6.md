---
phase: 03
fixed_at: 2026-09-01T14:54:23Z
review_path: /Users/akriz/code/cacheness/.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 4
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-01T14:54:23Z  
**Source review:** `03-REVIEW.md`  
**Iteration:** 4

**Summary:**

- Findings in scope: 6
- Fixed: 6
- Skipped: 0

## Fixed Issues

### CR-01: Nested fixed-stripe evidence locks can deadlock the clear lifecycle

**Files modified:** `src/cacheness/storage/operation_repository.py`, `src/cacheness/storage/coordination.py`, `tests/test_blob_store_reconciliation.py`  
**Commit:** `37730b1`  
**Applied fix:** Clear control page, checkpoint, reference, and retirement transitions now use one canonical operation-scoped lease. A nested child lifecycle operation inherits an already-active root lease rather than reopening an independently hashed descriptor, preventing both a same-stripe self-deadlock and a parent-A/child-B vs parent-B/child-A cycle. The in-process stripe calculation now matches the durable lock-stripe calculation. Regressions force same-stripe and opposite-order child topology, and the SQLite clear/put concurrency case passes 20 repeated runs without a timeout.

### CR-02: A crash during exclusive clear-control creation permanently poisons reopen

**Files modified:** `src/cacheness/storage/path_security.py`, `tests/test_blob_store_atomic_lifecycle.py`  
**Commit:** `37730b1`  
**Applied fix:** Immutable control records now write and fsync an unpredictable contained temporary, install it with an atomic no-replace hard link, fsync the parent directory, then remove and fsync the temporary directory entry. Failed writes clean their temporary and never expose a partial final record. Fault injection interrupts page, checkpoint, and both reference-chunk positions after a short write; reopen authenticates the primary record, aborts safely, preserves the original payload, and retires the remaining valid control evidence.

### CR-03: Default JSON BlobStore mutations fail on Windows

**Files modified:** `src/cacheness/storage/coordination.py`, `src/cacheness/storage/manifest_repository.py`, `tests/test_manifest_repository_cas.py`  
**Commit:** `37730b1`  
**Applied fix:** JSON canonical-manifest CAS now uses the common POSIX `flock`/Win32 `LockFileEx` adapter rather than importing `fcntl` directly. The Windows seam rejects `fcntl` imports, uses the actual injected Win32 adapter, and races two same-key JSON writers to exactly one winner.

### CR-04: JSON authority locking bypasses managed path containment

**Files modified:** `src/cacheness/storage/manifest_repository.py`, `src/cacheness/storage/path_security.py`, `src/cacheness/storage/blob_store.py`, `tests/test_manifest_repository_cas.py`  
**Commit:** `37730b1`  
**Applied fix:** A JSON repository creates its authority lock through `ManagedFileOps`, opens it no-follow once, captures its device/inode identity, and retains that descriptor for the repository lifetime. Each CAS verifies the name still resolves to that exact safe inode before locking; same-repository threads are serialized before acquiring the OS authority lock. Repository close releases the owned descriptor and failed BlobStore initialization releases it as well. Tests reject a lock symlink placed before creation, a replacement made between managed creation and no-follow open, and a live lock-name replacement after construction, without creating an outside file; the separate spawned two-process CAS test proves exact-winner behavior.

### WR-01: Barrier acquisition can register a recreated root under the wrong identity

**Files modified:** `src/cacheness/storage/coordination.py`, `tests/test_blob_store_close_contract.py`  
**Commit:** `37730b1`  
**Applied fix:** Admission-barrier acquisition now constructs the managed descriptor boundary first, keys the registry under that verified identity while holding the registry lock, closes a losing candidate, and releases by its captured registry key. The deterministic root-replacement seam proves the retired and recreated roots get independent barriers and both descriptors close after release.

### WR-02: Lock release failures can mask the real lifecycle failure

**Files modified:** `src/cacheness/error_handling.py`, `src/cacheness/storage/coordination.py`, `src/cacheness/storage/__init__.py`, `tests/test_blob_store_close_contract.py`, `tests/test_public_api_contract.py`  
**Commit:** `37730b1`  
**Applied fix:** Lock acquisition, protected-body execution, and release are now separate paths. Handles are always closed in a nested `finally`; an active body exception wins over an unlock failure, while a standalone unlock/close failure raises the new typed `CacheBlobLockReleaseError` with the original `OSError` as its cause. The public reason contract and both failure paths are tested.

## Verification

Verification ran in the **main checkout** at `/Users/akriz/code/cacheness` after commit `37730b1`.

- `uv run python -m py_compile` on all changed source and test files: passed.
- `uv run ruff check` on all changed source and test files: passed.
- Focused manifest CAS and atomic-control tests: passed.
- Focused reconciliation, concurrency, close, manifest-backend, and containment tests: passed (one expected Windows-junction skip).
- Full Phase 3 test suite: passed (one expected Windows-junction skip).
- Compatibility corpus through `sqlite-columns-v0314`: passed.
- Full repository suite: passed; expected optional PostgreSQL, TensorFlow, and Windows-junction skips remained, plus the pre-existing dataclass collection warning.

---

_Fixed: 2026-09-01T14:54:23Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 4_
