---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-01T17:08:54Z
depth: deep
files_reviewed: 28
files_reviewed_list:
  - src/cacheness/__init__.py
  - src/cacheness/config.py
  - src/cacheness/error_handling.py
  - src/cacheness/metadata.py
  - src/cacheness/serialization.py
  - src/cacheness/storage/__init__.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/clear_recovery.py
  - src/cacheness/storage/coordination.py
  - src/cacheness/storage/guarded_handler_io.py
  - src/cacheness/storage/lifecycle.py
  - src/cacheness/storage/manifest_repository.py
  - src/cacheness/storage/operation_record.py
  - src/cacheness/storage/operation_repository.py
  - src/cacheness/storage/path_security.py
  - src/cacheness/storage/reconciliation.py
  - tests/test_blob_store_atomic_lifecycle.py
  - tests/test_blob_store_close_contract.py
  - tests/test_blob_store_concurrency.py
  - tests/test_blob_store_integrity.py
  - tests/test_blob_store_read_contract.py
  - tests/test_blob_store_reconciliation.py
  - tests/test_clear_recovery.py
  - tests/test_config_validation.py
  - tests/test_filesystem_containment.py
  - tests/test_manifest_repository_cas.py
  - tests/test_public_api_contract.py
  - pyproject.toml
findings:
  critical: 4
  warning: 0
  info: 0
  total: 4
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-01T17:08:54Z
**Depth:** deep
**Files Reviewed:** 28
**Status:** issues_found

## Summary

Commit `41012b2` correctly fixes Python 3.11 syntax, no-follow regular-file
reads, candidate hashing through managed descriptors, fixed-lock pending-file
residue, and the previously misaligned durability seams. The full CPython 3.13
suite and focused CPython 3.11 lifecycle suite pass. Those tests still do not
establish Phase 3 convergence.

Four blockers remain. The new admission reference count has a zero-reader
closing race that releases a newly acquired shared lock; a deterministic probe
observed an external exclusive lock while one ordinary reader was active. The
new lock-authority sidecar is described as immutable but is an ordinary,
replaceable file, so replacing the lock and its predictable sidecar still lets
a second authority partition open. The same sidecar is written directly to its
final name, so process loss during a partial write leaves an unrecoverable file
that permanently blocks reopen. Finally, the native Windows fallback is not a
truthful implementation of the tested fake adapter: its error mapping breaks
`FileExistsError` contracts, its directory flush handle lacks the required
write access, and its pending-control recovery explicitly refuses every
fallback topology.

The review reconfirmed exact manifest CAS on the admitted local backends,
committed-only normal reads, managed no-follow candidate verification,
single-link special-file rejection, bounded operation parsing, authenticated
clear/reconciliation evidence, idempotent close ownership, native
handler-owned formats, and the absence of payload provenance guessing or a
Cacheness payload wrapper. The documented base-install NumPy packaging concern
and optional PostgreSQL/TensorFlow skips are not counted.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: BLOCKER — A new ordinary reader can lose its OS lock during the last-reader close transition

**File:** `src/cacheness/storage/coordination.py:443-502`

**Issue:** `ordinary_admission()` waits for `_ordinary_lock_opening` but not
`_ordinary_lock_closing`. When the last reader decrements `_ordinary_active` to
zero, it sets `_ordinary_lock_closing` and releases the condition before calling
the old context's `__exit__`. A new reader can enter in that interval, observe
zero readers, acquire `LOCK_SH` on the same retained open-file description, and
publish its new context. The old reader then calls `LOCK_UN` on that same open
file description and removes the new reader's lock. A deterministic probe
paused the old context immediately before unlock, entered a new ordinary
context, released the old context, and then acquired `LOCK_EX|LOCK_NB` from a
separate descriptor while `_ordinary_active == 1`. This recreates the clear
snapshot race that CR-01 intended to fix. The nested-reader regression exits
from two readers to one and never exercises the `0 -> closing -> 1` handoff.

**Fix:** Make ordinary admission wait while `_ordinary_lock_closing`, or keep
the condition held across the final unlock and next `0 -> 1` acquisition with
an explicit state machine. Add a deterministic test that pauses the final
reader's `__exit__`, starts a replacement reader, and proves external exclusive
admission remains blocked throughout the handoff.

### CR-02: BLOCKER — The lock-authority sidecar is replaceable and does not prevent authority partitioning

**Files:** `src/cacheness/storage/path_security.py:847-898`, `src/cacheness/storage/path_security.py:1346-1380`

**Issue:** `ensure_lifecycle_lock()` calls the sidecar immutable, but nothing
makes `.cacheness-lock-authorities/<digest>.authority` immutable or
authenticated. Its locator and bytes are predictable from the managed-root
identity, lock pathname, and replacement inode identity. A writer with the
same pathname-manipulation ability modeled by the lock-swap tests can replace
both the lock file and its sidecar with a correctly encoded pair. A later
`ManagedFileOps` instance then accepts that replacement in
`ensure_lifecycle_lock()` and `retain_lock_identity()` while the first process
still holds the retired descriptor. If the first process has passed its final
identity check, both descriptors can enter independent JSON CAS critical
sections. The new test swaps only the lock name, so it proves mismatch
detection rather than immutable cross-process authority.

**Fix:** Anchor synchronization in an identity that a later opener cannot
rebind by replacing another pathname, such as a retained/backend-native root
authority object with an enforceable creation/reopen contract. Do not solve the
replacement problem with another replaceable unauthenticated file. Extend the
two-process in-flight swap test to replace every mutable control pathname and
prove the contender still cannot enter.

### CR-03: BLOCKER — Process loss can leave a partial lock-authority sidecar that permanently bricks reopen

**Files:** `src/cacheness/storage/path_security.py:781-825`, `src/cacheness/storage/path_security.py:872-898`

**Issue:** The direct-final-name primitive is appropriate for fixed lock bytes
because those bytes are explicitly non-authoritative, but
`ensure_lifecycle_lock()` also uses it for the authoritative identity sidecar.
`_create_bytes_direct_exclusive_descriptor()` creates the final pathname before
looping through `_write_all()` and fsync. Process loss after creation or a
partial write therefore leaves an empty/truncated final sidecar. Every later
opener sees `FileExistsError`, reads the partial bytes, and raises
`CacheUnsafePathError(PATH_RACE)` forever; no operation evidence or
reconciliation action owns or repairs it. A direct probe placing partial bytes
at the exact sidecar locator reproduced the permanent reopen failure. The new
process-loss tests cover fixed lock creation only after its bytes are fsynced,
not interruption while publishing the authority sidecar.

**Fix:** Publish authoritative lock binding bytes with a recoverable,
old-or-new-complete protocol, or eliminate the sidecar in favor of an authority
primitive that does not require separately published bytes. Add subprocess
loss tests before, during, and after sidecar publication and require reopen to
converge without deleting ambiguous evidence.

### CR-04: BLOCKER — The production Windows fallback does not satisfy the lifecycle contract exercised by its fake adapter

**Files:** `src/cacheness/storage/path_security.py:53-130`, `src/cacheness/storage/path_security.py:900-927`, `src/cacheness/storage/path_security.py:1072-1095`, `src/cacheness/storage/operation_repository.py:1102-1143`

**Issue:** Three production-only mismatches keep the advertised Windows path
from providing the Phase 3 lifecycle:

1. `_WindowsFileApi._raise_last_error()` raises `OSError(winerror, ...)` rather
   than translating `ERROR_FILE_EXISTS`/`ERROR_ALREADY_EXISTS` to
   `FileExistsError`. Every exclusive-create caller relies on
   `except FileExistsError` for conflict or idempotent behavior, while the fake
   adapter raises that subclass explicitly.
2. `flush_directory()` opens the directory with `_GENERIC_READ` and then calls
   `FlushFileBuffers`; the Win32 contract requires a handle with generic write
   access. The fake adapter records a call and never exercises native handle
   access or filesystem behavior.
3. A process loss before `MoveFileExW` leaves the digest-bound pending file by
   design, but `promote_durable_pending_control()` unconditionally rejects
   every non-descriptor topology. Actual Windows always uses that fallback, so
   `recover_pending_operation_records()` cannot promote or retire the evidence
   the Windows writer creates.

Thus basic injected put/get/delete/clear tests do not validate native Windows
publication, conflict handling, or crash/reopen convergence, and CR-03 from the
prior review is not fully fixed.

**Fix:** Map native already-exists errors to `FileExistsError`, implement and
verify a supported Windows directory-durability primitive with the required
handle access, and provide fallback-native promotion/retirement of exact
digest-bound pending evidence. Run real Windows multiprocess and process-loss
tests for JSON and SQLite before treating the platform contract as restored.

## Warnings

None.

## Verification

- `uv run --python 3.13 --group recommended pytest -q -o log_cli=false` —
  passed; expected optional/environment skips and one existing collection
  warning.
- Focused 11-module Phase 3 suite on CPython 3.13 — passed with two expected
  containment fixture skips.
- `uv run --python 3.11 --group recommended python -m compileall -q src/cacheness`
  — passed on CPython 3.11.16.
- `uv run --python 3.11 --group recommended python -c 'import cacheness'` —
  passed (`0.3.14`).
- Focused lifecycle, close, concurrency, containment, and manifest-CAS suite on
  CPython 3.11 — passed with two expected containment fixture skips.
- Deterministic admission handoff probe — reproduced
  `external_exclusive_while_reader_active=True` with `_ordinary_active == 1`.
- Partial exact lock-authority sidecar probe — reproduced fail-closed permanent
  reopen as `CacheUnsafePathError(PATH_RACE)`.

---

_Reviewed: 2026-09-01T17:08:54Z_
_Reviewer: the agent (general code reviewer)_
_Iteration: 7_
