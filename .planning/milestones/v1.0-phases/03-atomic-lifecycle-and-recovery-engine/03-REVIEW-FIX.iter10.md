---
phase: 03
fixed_at: 2026-09-01T18:08:00Z
review_path: /Users/akriz/code/cacheness/.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 8
findings_in_scope: 6
fixed: 4
skipped: 2
status: partial
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-01T18:08:00Z  
**Source review:** `03-REVIEW.md`  
**Iteration:** 8

**Summary:**

- Findings in scope: 6
- Fixed: 4
- Skipped: 2

## Fixed Issues

### CR-03: Windows control durability used an unsupported directory flush

**Files modified:** `src/cacheness/storage/path_security.py`, `tests/test_manifest_repository_cas.py`, `tests/test_blob_store_close_contract.py`  
**Commits:** `714983d`, `57f43bb`  
**Applied fix:** Native moves now request `MOVEFILE_WRITE_THROUGH`; ordinary replacement uses the documented write-through replacement mode; surviving regular files are flushed through a regular, reparse-safe file handle; and durable deletion uses the write-through move primitive. The code no longer calls `FlushFileBuffers` on a directory handle. The non-Windows contract adapters assert flags and handle closure. Native Windows filesystem/ACL/crash evidence remains an execution gate rather than a simulated claim.

### CR-04: Windows pending-control recovery could delete a substituted pathname

**Files modified:** `src/cacheness/storage/path_security.py`, `tests/test_manifest_repository_cas.py`  
**Commit:** `714983d`  
**Applied fix:** When no-replace promotion finds an identical final record on the portable Windows fallback, the pending candidate is now blocked and retained. It is never unlinked by pathname after its verified read has ended. The substitution regression replaces the pending name with a victim at the former destructive seam and proves the victim remains present.

### WR-01: Uncertain admission unlock reopened the barrier

**Files modified:** `src/cacheness/storage/coordination.py`, `tests/test_blob_store_concurrency.py`  
**Commits:** `800d26f`, `3caba6a`  
**Applied fix:** Every release failure invokes a barrier-owned poison callback, including when the guarded body also fails. The original body exception remains public-compatible, while subsequent ordinary and aggregate admissions fail with `CacheBlobLockReleaseError` until all owning stores close and a new barrier is constructed. The regression exercises the body-error-plus-unlock-error path and both re-admission modes.

### WR-02: Reopen could consume a live clear and change `clear()`'s return count

**Files modified:** `src/cacheness/storage/lifecycle.py`, `tests/test_blob_store_concurrency.py`  
**Commit:** `800d26f`  
**Applied fix:** `clear()` now takes its continuation lease before publishing the snapshot and retains it through reclamation and retirement. A deterministic seam after aggregate admission releases proves a concurrently constructing store blocks on that live lease; the creator returns the correct cleared count.

## Skipped Issues

### CR-01: POSIX exact-CAS authority cannot be made immutable against its store owner with the current architecture

**File:** `src/cacheness/storage/path_security.py:1042`  
**Reason:** The current root-xattr binding is mutable by the same principal that can mutate the store root, so it cannot satisfy the review's stronger adversarial contract. Replacing it with an in-root lock file simply moves the same deletion/rebinding problem. Locking the root directory itself gives one non-rebindable kernel identity while that root survives, but it is one store-wide advisory lock and cannot supply the required independent per-key CAS authority; using it exclusively for every JSON/evidence transition conflicts with the locked no-global-normal-operation/per-key-concurrency contract. An ordinary process race that does not delete the root authority remains fail-closed today; an actor able to remove all authority state is outside that strongest truthful guarantee.

**Required contract choice:** either (1) define a trusted-store-owner boundary that forbids deletion/rebinding of lifecycle authority while the store is live, (2) replace JSON/filesystem exact-CAS coordination with an external durable authority service/backend, or (3) explicitly serialize local JSON/evidence transitions through a root-wide kernel lock and accept the resulting concurrency change. None is an implementation-only choice under the locked Phase 3 contracts.

### CR-02: Windows cross-user/session authority needs a platform support contract and native evidence

**File:** `src/cacheness/storage/path_security.py:149`  
**Reason:** The existing HKCU plus `Local\\` mutex binding is scoped to one user/session and therefore does not meet a shared-store cross-user/service contract. A truthful replacement requires a global kernel object or retained root handle protocol with an explicit cross-principal security descriptor, namespace lifetime and identity-reuse rules, and real Windows ACL/session/service tests. The current macOS adapter tests cannot prove those Windows kernel contracts. Replacing names without deciding which store principals may join would silently broaden the supported security topology.

**Required contract choice:** either (1) support only a single Windows SID/session and reject other sharing topologies, (2) support a documented set of store ACL principals through a `Global\\` namespace/retained-handle implementation validated on Windows, or (3) route cross-principal storage through an external coordinator. This requires product/platform-contract direction plus a Windows execution environment, not a speculative macOS-side patch.

## Verification

Verification ran in the **main checkout** at `/Users/akriz/code/cacheness`.

- CPython 3.13 full suite: passed (expected optional/platform skips and the existing collection warning only).
- CPython 3.13 focused lifecycle suite: concurrency, close, and manifest-CAS contracts passed (60 tests).
- CPython 3.11.16: `python -m compileall -q src/cacheness` and `import cacheness` passed.
- Changed-path Ruff, `uv lock --check`, and `git diff --check` passed.

## Residual Risks

- CR-01 and CR-02 are explicit unresolved architecture/platform contracts above; this report does not treat the current xattr/HKCU mechanism as a fix for them.
- The Windows implementation has injected adapter coverage only. Native NTFS/ReFS/local-volume and supported network-topology durability, ACL, session, and crash/reopen verification remains required.
- Blocked Windows pending candidates are intentionally retained rather than risking deletion of substituted data. They require a future handle-based retirement protocol or explicit reconciliation policy.

---

_Fixed: 2026-09-01T18:08:00Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 8_
