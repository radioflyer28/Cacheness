---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-08-31T14:29:52Z
depth: deep
files_reviewed: 27
files_reviewed_list:
  - src/cacheness/__init__.py
  - src/cacheness/config.py
  - src/cacheness/error_handling.py
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
  - uv.lock
findings:
  critical: 4
  warning: 2
  info: 0
  total: 6
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-08-31T14:29:52Z
**Depth:** deep
**Files Reviewed:** 27
**Status:** issues_found

## Summary

Commits `6905569` and `37c1afb` materially close the earlier cursor-token,
selected-record read, interrupted-`PREPARED` inventory, bounded chunk sidecar,
barrier lease, and subprocess-CAS findings. The focused 11-module phase suite
passes (with the Windows junction test skipped). The implementation is still
not shippable: the fixed evidence stripes can self-deadlock during routine clear
recovery; exclusive control-file creation can leave a partial authenticated-page
artifact that permanently prevents reopen; and the default JSON authority path
remains both POSIX-only and outside the managed containment boundary.

This review does not count the documented repository-wide Ruff baseline or the
known shutdown-only `SqliteBackend.__del__` failure. It reconfirms that the AEAD
resume token is versioned, nonce-fresh, domain-separated, pre-decode bounded,
and backed by the declared `cryptography` dependency. It also reconfirms exact
manifest checks in `exists`/`list`, digest-bound chunk assembly and retirement,
no payload wrapper/header, no native-format breach, no read-time recovery, and
no filename-only payload provenance.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: BLOCKER — Nested fixed-stripe evidence locks can deadlock the clear lifecycle

**Files:** `src/cacheness/storage/operation_repository.py:126-170`, `src/cacheness/storage/coordination.py:128-194`, `src/cacheness/storage/lifecycle.py:555-625`, `src/cacheness/storage/lifecycle.py:702-782`

**Issue:** `_continue_clear()` and `_abort_prepared_clear()` hold an
`operation-run:<id>` advisory stripe while calling checkpoint/retirement helpers
that acquire additional advisory stripes. The in-process lock and file-lock
stripe calculations are not even the same: `_conditional_lock_for()` uses the
first digest byte, while `lock_stripe_index()` reduces the first four bytes.
When an outer and inner transition land on the same file stripe, the same thread
opens the lock file again and requests another exclusive `flock`/`LockFileEx`.
On POSIX, a second `flock` through an independently opened descriptor blocks
behind the first descriptor owned by that same process; a direct nonblocking
probe on this runtime returns `EWOULDBLOCK`. With 64 stripes, each nested
transition has a 1/64 collision chance, and concurrent clears can also form an
A-then-B/B-then-A cycle. The existing subprocess tests race one unnested exact
transition and cannot expose this hang.

**Fix:** Do not nest independently hashed advisory stripes. Give one operation a
single process-wide/cross-process lease and perform its page/checkpoint CAS under
that lease, or implement a stripe guard that is keyed by the exact file-stripe
index, tracks per-thread recursion, acquires the OS lock only at recursion depth
zero, and enforces a total order for multiple stripes. Add deterministic tests
that force outer/inner same-stripe collisions and opposite-order two-operation
collisions; assert bounded completion on POSIX and Windows.

### CR-02: BLOCKER — A crash during exclusive clear-control creation permanently poisons reopen

**Files:** `src/cacheness/storage/path_security.py:621-665`, `src/cacheness/storage/operation_repository.py:378-418`, `src/cacheness/storage/operation_repository.py:474-535`, `src/cacheness/storage/lifecycle.py:470-505`, `src/cacheness/storage/lifecycle.py:702-782`

**Issue:** Clear pages, checkpoints, and reference chunks are created directly at
their final names with `O_EXCL`/`open(..., "xb")`, then filled and fsynced. A
process loss during `_write_all()` leaves a short final file. On reopen, the
signed primary record sends recovery into `_abort_prepared_clear()`: a partial
page fails canonical authentication, while a partial chunk fails its signed
digest check. Both errors abort construction and leave the same artifacts in
place, so every future reopen fails identically. The new crash test interrupts
before the first chunk write, and the chunk-size test observes only fully
returned writes; neither exercises process loss during a page/checkpoint/chunk
write. Bounded reads prevent allocation abuse but do not provide crash atomicity.

**Fix:** Add an atomic no-replace control-record publish primitive: fully write
and fsync an unguessable contained temporary, atomically install it only if the
final name is absent (for example `linkat`/platform equivalent or a guarded
no-replace rename), fsync the directory, then remove the temporary. Keep payload
candidate semantics separate if partial final candidates are intentionally
operation-owned. Add fault-injection tests for short page, checkpoint, and every
chunk position, followed by close/crash, reopen, safe abort, and artifact
retirement.

### CR-03: BLOCKER — Default JSON BlobStore mutations still fail on Windows

**Files:** `src/cacheness/storage/manifest_repository.py:275-303`, `src/cacheness/storage/blob_store.py:284-307`, `tests/test_blob_store_close_contract.py:337-361`

**Issue:** `37c1afb` adds Win32 locking for admission and operation evidence, but
the canonical JSON manifest compare/publish lock still imports `fcntl` and
explicitly raises when it is absent. JSON is the default `BlobStore` backend, so
Windows can now construct a store but its first `put`, `delete`, or metadata CAS
fails at the sole authority publication point. The new “Windows” test only
monkeypatches `coordination._platform_name` while executing on POSIX; the JSON
manifest repository therefore imports the host's real `fcntl`, producing a
false end-to-end pass.

**Fix:** Route JSON manifest CAS through the same truthful POSIX/Win32 lock
abstraction (or a backend-native transactional primitive) and exercise the
actual Windows branch in CI. The Windows test must run with `fcntl` unavailable
and prove concurrent same-key JSON publication has one exact winner, not merely
that the coordination adapter recorded shared/exclusive calls.

### CR-04: BLOCKER — JSON authority locking bypasses managed path containment

**File:** `src/cacheness/storage/manifest_repository.py:289-299`

**Issue:** The JSON CAS lock uses raw `Path.mkdir()` and `open(lock_path,
"a+b")`, rather than `ManagedFileOps`. A symlink/reparse point at the lock name
is followed; a dangling link can make `O_CREAT` create a file outside the store,
and swapping the link or regular file lets two writers lock different inodes and
enter refresh/compare/publish concurrently. That defeats exact manifest CAS and
can lose a committed generation, in addition to violating the project's
fail-closed containment boundary. None of the containment tests targets this
authority lock.

**Fix:** Resolve and open the lock through the managed no-follow boundary, keep
one stable inode for the store lifetime, and use the common cross-platform lock
adapter on that descriptor. Add symlink/junction substitution tests before open
and between validation/open, plus a two-process CAS test proving substitution is
rejected without any out-of-root creation.

## Warnings

### WR-01: WARNING — Barrier acquisition can register a recreated root under the wrong identity

**File:** `src/cacheness/storage/coordination.py:212-277`

**Issue:** `acquire()` stats the pathname before taking the registry lock, then
constructs `ManagedFileOps` and independently stats it again, but stores the new
barrier under the first identity. If the root is removed/recreated in that
window, the dictionary key differs from `barrier._root_identity`; `release()`
cannot find itself, so the descriptor and registry entry leak. A later acquire
of the recreated root creates a second barrier for the same live root. The
current recreation test performs removal only after close and cannot exercise
this race.

**Fix:** Construct the descriptor-anchored barrier first, then under the registry
lock key it by its verified `_root_identity`; if another barrier already won,
close the unused candidate. On release, remove by the exact key captured during
registration. Add a deterministic replacement seam between the initial stat and
registration.

### WR-02: WARNING — Lock release failures can mask the real lifecycle failure

**File:** `src/cacheness/storage/coordination.py:156-194`

**Issue:** The `try` covers the caller's entire `yield`, so an `OSError` raised by
the protected lifecycle mutation is mislabeled as “lock could not be acquired.”
Then `unlock()` runs unguarded in `finally`; if unlock also fails, it replaces
the original exception and may skip the subsequent handle close. This makes
recovery evidence and operator diagnostics ambiguous at exactly the failure
boundary the phase is intended to preserve.

**Fix:** Separate acquisition, body, and release error handling. Always close the
handle in a nested `finally`; preserve an active body exception, and translate a
standalone unlock failure to a distinct typed lock-release error with its cause.

---

_Reviewed: 2026-08-31T14:29:52Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
