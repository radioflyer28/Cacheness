---
phase: 03
fixed_at: 2026-09-01T16:32:13Z
review_path: /Users/akriz/code/cacheness/.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 6
findings_in_scope: 8
fixed: 8
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-01T16:32:13Z  
**Source review:** `03-REVIEW.md`  
**Iteration:** 6

**Summary:**

- Findings in scope: 8
- Fixed: 8
- Skipped: 0

## Fixed Issues

### CR-01: Concurrent ordinary admissions released the shared OS lock too early

**Files modified:** `src/cacheness/storage/coordination.py`, `tests/test_blob_store_concurrency.py`  
**Commit:** `41012b2`  
**Applied fix:** Ordinary admission now keeps one retained OS shared lock across the aggregate local reader count, acquiring it only for the `0 -> 1` transition and releasing it only for `1 -> 0`. Aggregate admission waits for both acquisition and release transitions. A fresh external process cannot obtain exclusive admission until both nested ordinary readers exit.

### CR-02: Replaced lock names could partition JSON, admission, and evidence authority

**Files modified:** `src/cacheness/storage/path_security.py`, `src/cacheness/storage/coordination.py`, `src/cacheness/storage/manifest_repository.py`, `src/cacheness/storage/operation_repository.py`, `tests/test_manifest_repository_cas.py`  
**Commit:** `41012b2`  
**Applied fix:** Every lifecycle lock now has a root-bound, exact identity record separate from its mutable lock pathname. New openers validate that binding before retaining or acquiring a descriptor; a replacement pathname therefore fails closed rather than joining a second authority partition. A deterministic two-process JSON swap test proves that the original descriptor can publish once while the replacement opener is blocked before CAS.

### CR-03: Default BlobStore deliberately refused Windows

**Files modified:** `src/cacheness/storage/path_security.py`, `tests/test_manifest_repository_cas.py`, `tests/test_blob_store_close_contract.py`  
**Commit:** `41012b2`  
**Applied fix:** Added a native Win32 adapter that uses a consuming `MoveFileExW` no-replace move and `CreateFileW` directory handles with `FlushFileBuffers`; the normal fallback path now routes JSON and SQLite BlobStore lifecycle operations through it. Injected adapter tests exercise default JSON/SQLite put/get/delete/clear and authority publication without POSIX directory I/O.

### CR-04: Process loss could leave pending lifecycle-lock residue

**Files modified:** `src/cacheness/storage/path_security.py`, `src/cacheness/storage/coordination.py`, `src/cacheness/storage/manifest_repository.py`, `src/cacheness/storage/operation_repository.py`, `tests/test_blob_store_atomic_lifecycle.py`  
**Commit:** `41012b2`  
**Applied fix:** Fixed lock files now use direct exclusive final-name creation and durable directory acknowledgement. Their bytes are non-authoritative; ownership is the verified descriptor identity. Subprocess loss tests cover admission, JSON authority, clear-resume, and conditional-stripe creation, then reopen without any `.pending.*.tmp` residue.

### CR-05: Candidate hashing followed substitutions outside managed I/O

**Files modified:** `src/cacheness/storage/guarded_handler_io.py`, `src/cacheness/storage/lifecycle.py`, `src/cacheness/storage/path_security.py`, `tests/test_blob_store_atomic_lifecycle.py`  
**Commit:** `41012b2`  
**Applied fix:** Immutable publication captures the installed file identity. Verification hashes through one no-follow, single-link managed descriptor and validates the identity before and after reading, before manifest CAS. Symlink, hard-link, FIFO, and regular-inode substitution tests prove no manifest is committed and no outside bytes are read.

### CR-06: Special files could block unauthenticated reads

**Files modified:** `src/cacheness/storage/path_security.py`, `tests/test_filesystem_containment.py`  
**Commit:** `41012b2`  
**Applied fix:** Descriptor and fallback reads now obtain initial descriptors nonblocking, require a regular single-linked `fstat` result, and reject every special/reparse object before read or parse. `exists` and `get_size` use the same fail-closed check. FIFO, Unix socket, directory, and—where the test user may create one—device-node regressions complete without blocking.

### CR-07: Python 3.11 could not compile the public package

**Files modified:** `src/cacheness/serialization.py`, `pyproject.toml`  
**Commit:** `41012b2`  
**Applied fix:** Rewrote the dict serialization f-string expressions into Python 3.11-compatible intermediate values and changed Ruff’s target to `py311`. CPython 3.11.16 now compiles the source, imports `cacheness`, and passes the focused lifecycle suite in the recommended dependency environment.

### WR-01: Crash seams did not match primitive durability boundaries

**Files modified:** `src/cacheness/storage/path_security.py`, `tests/test_blob_store_atomic_lifecycle.py`  
**Commit:** `41012b2`  
**Applied fix:** Control-publication seams now fire after temporary-file fsync, after consuming rename and before directory fsync, and after directory fsync. Process-loss tests use those exact boundaries for operation and clear sidecar recovery. Test documentation explicitly limits `os._exit` to process-loss modelling rather than physical power-loss proof.

## Verification

Verification ran in the **main checkout** at `/Users/akriz/code/cacheness` after commit `41012b2`.

- `uv run --group recommended pytest -q -o log_cli=false` passed: 777 collected tests, with expected optional/environment skips and the existing dataclass collection warning.
- The focused lifecycle, containment, concurrency, manifest-CAS, and close-contract suite passed on CPython 3.13; the Windows-junction and unprivileged device-node fixtures skipped as expected.
- CPython 3.11.16 checks passed: `python -m compileall -q src/cacheness`, `import cacheness`, and the focused lifecycle suite under `--group recommended`.
- `uv lock --check`, changed-file Ruff, and `git diff --check` passed.

## Residual Risks

- The native Win32 adapter is covered through an injectable protocol on this host; execution on a real Windows filesystem/CI worker is still required to establish filesystem-specific durability behavior.
- Process-loss seams validate reconciliation protocol states. A filesystem or power-loss harness remains necessary for claims about hardware persistence after acknowledged writes.
- A bare base install still cannot import because NumPy is eagerly imported while declared only in optional dependency groups. This pre-existing packaging concern is outside this review finding set and remains documented separately.

---

_Fixed: 2026-09-01T16:32:13Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 6_
