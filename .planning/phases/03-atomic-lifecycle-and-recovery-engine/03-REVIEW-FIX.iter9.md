---
phase: 03
fixed_at: 2026-09-01T17:33:54Z
review_path: /Users/akriz/code/cacheness/.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 7
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-01T17:33:54Z  
**Source review:** `03-REVIEW.md`  
**Iteration:** 7

**Summary:**

- Findings in scope: 4
- Fixed: 4
- Skipped: 0

## Fixed Issues

### CR-01: A new ordinary reader could lose its OS lock during final-reader close

**Files modified:** `src/cacheness/storage/coordination.py`, `tests/test_blob_store_concurrency.py`  
**Commit:** `5d42b5c`  
**Applied fix:** Ordinary admission now treats `closing` as a first-class state: a replacement reader waits until the final retained shared-lock context has released before starting a new `0 -> 1` acquisition. The deterministic handoff regression pauses the final unlock, proves the replacement cannot enter early, and proves an external exclusive contender remains blocked while the replacement reader is active.

### CR-02: A replaceable lock sidecar could partition authority

**Files modified:** `src/cacheness/storage/path_security.py`, `tests/test_manifest_repository_cas.py`, `tests/test_blob_store_close_contract.py`  
**Commit:** `25b3454`  
**Applied fix:** Removed the in-root lock-authority sidecar. Descriptor-capable POSIX stores bind the accepted lock inode to the anchored root directory through an exact extended attribute; Windows binds it through an immutable HKCU registry value protected during first publication by an abandon-safe named mutex. Later openers compare that root-bound record before accepting a lock descriptor, so swapping mutable control files fails before JSON CAS can enter. The native Windows adapter is exercised through an injected contract in this non-Windows environment.

### CR-03: A partial authority sidecar could permanently brick reopen

**Files modified:** `src/cacheness/storage/path_security.py`, `tests/test_blob_store_atomic_lifecycle.py`  
**Commit:** `25b3454`  
**Applied fix:** Authority publication is now one root-object xattr/registry value, not a direct write to a final sidecar filename. Pre- and post-publication process-loss seams prove reopening converges from either no binding or a complete binding; the kernel operation cannot expose a partial authority value. Ambiguous legacy sidecar bytes are neither trusted nor deleted, and no new sidecar or pending residue is created.

### CR-04: Windows fallback did not implement the required lifecycle contract

**Files modified:** `src/cacheness/storage/path_security.py`, `tests/test_manifest_repository_cas.py`, `tests/test_blob_store_close_contract.py`  
**Commit:** `25b3454`  
**Applied fix:** Native `ERROR_FILE_EXISTS` and `ERROR_ALREADY_EXISTS` now translate to `FileExistsError`; directory flush handles request generic write access and always close; and the fallback promotes or retires only exact digest-bound pending controls after a pre-`MoveFileExW` loss. Unit tests cover native error mapping, directory-handle flags/closure, registry-authority contract injection, and exact fallback promotion.

## Verification

Verification ran in the **main checkout** at `/Users/akriz/code/cacheness` after commits `5d42b5c` and `25b3454`.

- Focused Phase 3 lifecycle suite passed on CPython 3.13: atomic lifecycle, close contract, concurrency, integrity/read/reconciliation, clear recovery, configuration, containment, manifest CAS, and public API tests. The expected Windows-junction and unprivileged-device fixtures skipped.
- All repository tests passed on CPython 3.13 in four bounded runs. Expected optional PostgreSQL, TensorFlow, Polars, Windows-junction, and device-node skips occurred; the known shutdown-only `SqliteBackend.__del__` `sys.meta_path is None` messages also remain.
- CPython 3.11.16 passed `python -m compileall -q src/cacheness`, `import cacheness`, and focused atomic lifecycle, close, concurrency, containment, and manifest-CAS suites.
- `uv lock --check`, changed-path Ruff, and `git diff --check` passed before commit.

## Residual Risks

- The native Windows adapter and registry authority protocol are covered through injected contract tests on macOS. A real Windows filesystem/CI run remains required to establish platform-specific `CreateFileW`, registry, `MoveFileExW`, and durability behavior; this is an evidence gap, not a disabled Windows path.
- Root-authority bindings intentionally remain while their storage root exists. Deleting a bound lock control file is fail-closed rather than rebinding it; deleting the root itself removes POSIX xattrs, while Windows registry values are retained to avoid recreating authority for a reused identity namespace.
- Process-loss probes validate protocol convergence, not physical power-loss persistence after acknowledged filesystem calls.
- The documented undeclared NumPy base-import dependency and the existing shutdown-only SQLite destructor diagnostic remain outside this finding set.

---

_Fixed: 2026-09-01T17:33:54Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 7_
