---
phase: 03
fixed_at: 2026-09-01T15:56:55Z
review_path: /Users/akriz/code/cacheness/.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 5
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-01T15:56:55Z  
**Source review:** `03-REVIEW.md`  
**Iteration:** 5

**Summary:**

- Findings in scope: 6
- Fixed: 6
- Skipped: 0

## Fixed Issues

### CR-01: Root-wide lease inheritance removed exact-CAS exclusion

**Files modified:** `src/cacheness/storage/operation_repository.py`, `src/cacheness/storage/lifecycle.py`, `tests/test_blob_store_reconciliation.py`  
**Commit:** `b109531`  
**Applied fix:** Reentrancy now applies only to the exact operation ID. Clear continuation uses a separate aggregate-control lease and acquires every child delete/checkpoint through its own exact evidence lease; nested distinct operation leases fail closed. Fixed stripe descriptors are bounded by physical stripe name, and same/opposite-stripe completion plus real two-process exact-CAS winner tests cover the boundary.

### CR-02: JSON authority could switch to a replacement root after validation

**Files modified:** `src/cacheness/storage/manifest_repository.py`, `src/cacheness/storage/blob_store.py`, `src/cacheness/metadata.py`, `tests/test_manifest_repository_cas.py`  
**Commit:** `b109531`  
**Applied fix:** JSON refresh, comparison, and publication now run through a retained managed-root descriptor while the retained authority descriptor is locked. BlobStore delegates its pre-lifecycle JSON refresh to this same repository boundary. Root replacement seams immediately before and immediately after lock acquisition fail closed and create no replacement-root metadata.

### CR-03: Mutable or hard-linked lock inodes could split advisory authority

**Files modified:** `src/cacheness/storage/path_security.py`, `src/cacheness/storage/coordination.py`, `src/cacheness/storage/operation_repository.py`, `src/cacheness/storage/manifest_repository.py`, `src/cacheness/storage/blob_store.py`, `tests/test_blob_store_close_contract.py`, `tests/test_manifest_repository_cas.py`  
**Commit:** `b109531`  
**Applied fix:** Control locks reject non-regular and multi-linked inodes, retain verified descriptors, and verify both name and root identity before and after OS lock acquisition. Admission candidates and operation stripe descriptors are closed when not retained; store shutdown closes all retained authority handles. Hard-link and regular-inode replacement regressions cover short-lived, JSON, admission, and evidence-lock families.

### CR-04: Windows durability path made unsupported directory fsync claims

**Files modified:** `src/cacheness/storage/path_security.py`, `tests/test_manifest_repository_cas.py`  
**Commit:** `b109531`  
**Applied fix:** A non-descriptor Windows topology now fails during lifecycle-control setup with a typed unsupported-capability error before canonical JSON authority or POSIX directory I/O is attempted. The Windows branch is exercised with a platform seam and an assertion that directory `fsync` is never called. This is intentionally fail-closed rather than a claim that the unsupported fallback is durable; actual-Windows put/delete/clear/CAS matrix execution remains required when native Win32 durability support is introduced.

### CR-05: Hard-link no-replace publication could strand unauthenticated temporary residue

**Files modified:** `src/cacheness/storage/path_security.py`, `src/cacheness/storage/operation_repository.py`, `src/cacheness/storage/lifecycle.py`, `tests/test_blob_store_atomic_lifecycle.py`  
**Commit:** `b109531`  
**Applied fix:** Control publication now uses native consuming no-replace rename (`renameatx_np`/`renameat2`) rather than hard-link installation. Pending candidates bind the exact final control name, payload digest, and random token; recovery validates that provenance before promotion or retirement. Subprocess `os._exit` tests cover temp creation, final installation, both directory-durability boundaries, and retirement for primary operations and clear sidecars, then reopen to no control residue.

### WR-01: Failed direct JSON repository construction leaked owned descriptors

**Files modified:** `src/cacheness/storage/manifest_repository.py`, `tests/test_manifest_repository_cas.py`  
**Commit:** `b109531`  
**Applied fix:** Constructor setup is one ownership-aware `try` boundary. Failures during durable lock creation, open, identity capture, or post-validation close the owned root descriptor and any retained lock handle before reraising; direct-construction close-spy regressions cover each failure point.

## Verification

Verification ran in the **main checkout** at `/Users/akriz/code/cacheness` after commit `b109531`.

- Python syntax checks passed for all changed source modules.
- Changed-module and changed-test Ruff checks passed. `metadata.py` retains seven pre-existing F401 findings; it passed with that existing baseline excluded and introduced no new lint class.
- The complete Phase 03 suite passed, with the expected Windows-junction fixture skipped.
- The repository suite passed in four command-time-bounded chunks. Expected optional PostgreSQL, TensorFlow, and Windows-junction skips, the pre-existing dataclass collection warning, and known SQLite shutdown-only destructor logging remained.
- Compatibility corpus validation through `sqlite-columns-v0314` passed.

---

_Fixed: 2026-09-01T15:56:55Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 5_
