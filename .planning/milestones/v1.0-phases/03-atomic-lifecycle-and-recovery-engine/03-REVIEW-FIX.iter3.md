---
phase: 03
fixed_at: 2026-08-31T14:18:46Z
review_path: /Users/akriz/code/cacheness/.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 2
findings_in_scope: 6
fixed: 6
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-08-31T14:18:46Z  
**Source review:** `03-REVIEW.md`  
**Iteration:** 2

**Summary:**

- Findings in scope: 6
- Fixed: 6
- Skipped: 0

## Fixed Issues

### CR-01: A crash while paging a clear snapshot lets recovery delete later writes

**Files modified:** `lifecycle.py`, `operation_repository.py`, `operation_record.py`, `test_blob_store_atomic_lifecycle.py`  
**Commit:** `37c1afb`  
**Applied fix:** A `PREPARED` clear no longer resumes a lexical inventory after its admission epoch is lost. Recovery authenticates and retires only its non-authoritative pages, checkpoints, and proven sidecar chunks, without reading or deleting manifests. The deterministic process-loss test proves an already-open independent writer's later key survives reopen.

### CR-02: Clear sidecars can be written larger than the configured read limit and poison reopen

**Files modified:** `lifecycle.py`, `operation_record.py`, `operation_repository.py`, `test_blob_store_atomic_lifecycle.py`  
**Commit:** `37c1afb`  
**Applied fix:** Exact manifests that exceed the caller's operation-record limit are represented as authenticated bounded chunks. The signed page binds ordered chunk digests, count, and total length; reads validate this contract before allocation, and clear preflights representability before creating the primary `PREPARED` record. The injected-small-limit crash/reopen test confirms valid oversized evidence clears and retires safely.

### CR-03: The new admission/evidence locks disable BlobStore on Windows

**Files modified:** `coordination.py`, `operation_repository.py`, `blob_store.py`, `test_blob_store_close_contract.py`  
**Commit:** `37c1afb`  
**Applied fix:** Shared/exclusive lifecycle locks now use POSIX `flock` or a Win32 `LockFileEx` adapter, failing closed only when neither lock topology is available. The Windows seam test exercises lock acquisition and release through canonical `BlobStore` construction and clear.

### WR-01: Every lifecycle transition permanently creates another advisory-lock file

**Files modified:** `coordination.py`, `operation_repository.py`, `test_blob_store_reconciliation.py`  
**Commit:** `37c1afb`  
**Applied fix:** Exact evidence transitions hash into a fixed set of 64 root-scoped lock stripes rather than allocating one file per transition. The high-cardinality lifecycle test proves the lock namespace remains bounded.

### WR-02: The global admission registry leaks root descriptors and becomes stale after root recreation

**Files modified:** `coordination.py`, `blob_store.py`, `test_blob_store_close_contract.py`  
**Commit:** `37c1afb`  
**Applied fix:** Admission barriers are leased by `(device, inode)`, release and close their root descriptor when the last `BlobStore` closes, and create a distinct barrier after a root is removed and recreated. Regression coverage checks both many closed roots and root identity replacement.

### WR-03: The evidence-CAS regression test never exercises the cross-process primitive

**Files modified:** `operation_repository.py`, `test_blob_store_reconciliation.py`  
**Commit:** `37c1afb`  
**Applied fix:** Spawned subprocesses construct their own managed root and evidence repository, reconstruct the signed exact record from canonical bytes, and race checkpoint/checkpoint plus checkpoint/retire. Each race has exactly one winning transition.

## Verification

Verification ran with the **isolated review-fix worktree source** at
`/Users/akriz/code/cacheness/.claude/worktrees/rf-03-fix2-1788184323`, using
the main checkout's existing virtualenv for the project dependency set.

- Python syntax compilation of all five changed storage modules: passed.
- Changed-file Ruff: passed.
- Focused lifecycle, cross-process CAS, fixed-stripe, barrier lifetime, and Windows-seam tests: passed.
- Full Phase 3 suite: passed (one expected Windows-junction skip).
- Full repository suite: passed in bounded test groups because the terminal's 30-second command limit truncates a monolithic run; expected PostgreSQL and TensorFlow skips remained.
- Compatibility corpus through `sqlite-columns-v0314`: passed.

---

_Fixed: 2026-08-31T14:18:46Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 2_
