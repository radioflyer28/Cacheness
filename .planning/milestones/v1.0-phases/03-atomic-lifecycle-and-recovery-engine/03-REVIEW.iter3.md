---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-08-31T14:42:00Z
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
  critical: 3
  warning: 3
  info: 0
  total: 6
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-08-31T14:42:00Z  
**Depth:** deep  
**Files Reviewed:** 27  
**Status:** issues_found

## Summary

Commit `6905569` closes the original deterministic cursor cipher, same-process
evidence-CAS, selected-record `exists`/`list`, and default-limit clear cleanup
cases, and all 11 phase-focused test modules pass (one Windows-only containment
test is skipped). It does not close the phase contract. A crash during clear
inventory construction lets recovery absorb and delete post-snapshot writes;
caller-owned evidence limits make valid large manifests poison clear and every
subsequent reopen; and the new locking layer makes every canonical `BlobStore`
unusable on Windows despite the public cross-platform contract. The lock design
also leaks a descriptor-backed barrier per root and an unbounded file per
evidence transition, while the purported CAS regression test never crosses a
process boundary.

This review does not count the documented repository-wide Ruff baseline or the
known shutdown-only `SqliteBackend.__del__` issue. It also confirms that the
AEAD token uses a 32-byte domain-separated key, 12-byte fresh nonce, 16-byte
tag, pre-decode bounds, and a declared base dependency, and found no new payload
wrapper/header or native-format violation.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: BLOCKER — A crash while paging a clear snapshot lets recovery delete later writes

**Files:** `src/cacheness/storage/lifecycle.py:328-451`, `src/cacheness/storage/lifecycle.py:636-656`, `tests/test_blob_store_concurrency.py:271-320`, `tests/test_blob_store_atomic_lifecycle.py:672-705`

**Issue:** Aggregate admission is held while one live manifest page at a time is
persisted, but the primary clear record remains `PREPARED` until all pages are
written. If the process dies after one page, the OS lock is released. An
already-open independent store can then commit a new key. Reopen reacquires the
lock, reloads the persisted page, resumes live paging from that page's lexical
cursor, and adds the new key to the old clear operation. `_continue_clear`
subsequently tombstones and deletes it. The resulting target set is neither the
pre-crash snapshot nor the post-crash state.

This was reproduced with `manifest_page_size=1`: clear persisted the first of
keys `a` and `z`, simulated process loss occurred, a second already-open store
committed key `m`, and reopen recovery returned an empty store and `get("m") is
None`. The new subprocess test only pauses after the *entire* snapshot has been
persisted; the resume test interrupts after destructive target progress. Neither
exercises process loss during inventory construction.

**Fix:** Never resume a live lexical inventory after its admission epoch was
lost. Either persist a backend-native immutable snapshot identity and page only
that snapshot, or treat an interrupted `PREPARED` inventory as an aborted clear:
authenticate and retire/roll back its non-authoritative pages and sidecars
without deleting any manifest generation, then require a fresh clear to take a
new snapshot. Add a crash test after the first persisted page with an already
open independent writer, and prove the later generation survives recovery.

### CR-02: BLOCKER — Clear sidecars can be written larger than the configured read limit and poison reopen

**Files:** `src/cacheness/config.py:353-391`, `src/cacheness/storage/lifecycle.py:369-440`, `src/cacheness/storage/operation_repository.py:192-204`, `src/cacheness/storage/operation_repository.py:422-479`, `src/cacheness/storage/operation_record.py:447-504`

**Issue:** On page overflow, clear writes the complete canonical manifest to a
reference sidecar without checking `max_operation_record_bytes`. Every sidecar
read then goes through `_read_bounded`, which applies that exact operation-record
limit. A manifest may be valid under the manifest codec while larger than a
caller's smaller operation-evidence policy, so the implementation persists bytes
that it is structurally incapable of reading. The signed page and `PREPARED`
main record remain, and constructor recovery fails on every reopen.

This was reproduced with a valid 15,837-byte manifest and
`max_operation_record_bytes=10_000`: `clear()` wrote a 15,837-byte reference and
a 501-byte signed page, then raised `CacheManifestIntegrityError`; reopening the
store raised the same error. The added large-manifest test uses only the default
1 MiB operation limit, while the injected-small-limit test reads an unrelated
65-byte hostile operation file and never exercises sidecar creation/recovery.

**Fix:** Give referenced manifest evidence its own explicit bounded contract or
chunk exact manifest bytes into individually bounded immutable records whose
ordered identities and total digest are bound by the signed page. Enforce the
same bound before every sidecar write as well as before allocation on read. If a
caller policy cannot represent even the minimum control structure, reject it
before persisting the primary clear record; never leave an unreopenable
`PREPARED` operation. Test a manifest larger than an injected operation-record
limit through clear, crash, reopen, and artifact retirement.

### CR-03: BLOCKER — The new admission/evidence locks disable BlobStore on Windows

**Files:** `src/cacheness/storage/coordination.py:65-97`, `src/cacheness/storage/operation_repository.py:143-184`, `src/cacheness/storage/lifecycle.py:47-63`, `src/cacheness/storage/blob_store.py:300-308`, `README.md:13`

**Issue:** Both new lock primitives import `fcntl` and deliberately raise
`CacheBlobBackendError` when it is unavailable. More importantly,
`LifecycleEngine.__init__` unconditionally enters aggregate admission to run
recovery, so on Windows *construction* of every non-legacy canonical
`BlobStore` fails before the caller can use even in-memory or SQLite storage.
This is not a typed rejection of one unsupported distributed topology; it
removes an already advertised platform from the public API. The repository
still describes the library as fully compatible with Windows, Linux, and macOS,
and the lock dependency was not reflected as a platform restriction.

**Fix:** Introduce a lock abstraction with truthful Windows and POSIX
implementations (for example, a stable Win32/`msvcrt` file lock plus `flock`), or
use backend-native transactional admission/CAS where available. Capability
failure should be scoped to the unsupported backend topology rather than all
store construction. Run the same clear-admission and exact-evidence CAS tests in
the Windows matrix; do not skip the only target-platform behavior test.

## Warnings

### WR-01: WARNING — Every lifecycle transition permanently creates another advisory-lock file

**Files:** `src/cacheness/storage/operation_repository.py:133-190`, `src/cacheness/storage/operation_repository.py:314-351`, `src/cacheness/storage/operation_repository.py:481-550`, `src/cacheness/storage/operation_repository.py:663-730`, `tests/test_blob_store_atomic_lifecycle.py:316-384`

**Issue:** `_conditional_lock_locator` hashes the full transition ID into a unique
file under `operations/.conditional-locks`, and no path ever retires or reuses
those files. Operation UUIDs make ordinary put/delete records unique; clear also
creates distinct run, checkpoint, page, and reference locks. The new cleanup
assertions glob only top-level `operations/*.json`, so they miss the accumulating
nested lock namespace. Removing a live lock file is unsafe because it permits
two inode generations, but retaining a unique inode forever causes unbounded
disk/inode growth in normal production use.

**Fix:** Use a fixed, bounded cross-process stripe set derived from the same hash
as the in-process stripes, or a backend transaction/lock table with bounded
retention. Assert that high-cardinality successful operations do not increase
the lock-file count beyond the fixed bound.

### WR-02: WARNING — The global admission registry leaks root descriptors and becomes stale after root recreation

**Files:** `src/cacheness/storage/coordination.py:35-63`, `src/cacheness/storage/path_security.py:228-271`, `src/cacheness/storage/blob_store.py:853-898`

**Issue:** `StoreAdmissionBarrier._instances` strongly retains one barrier for
every path ever opened. Each barrier owns a separate `ManagedFileOps`, which on
POSIX owns a root directory descriptor, but `BlobStore.close()` releases only
its own guarded I/O and backend. Thus closed stores leak descriptors. If a
closed store root is later removed and recreated at the same path, `for_root`
returns the barrier anchored to the old inode; `_assert_root_identity` then makes
the new store unusable rather than constructing a barrier for the new root.

**Fix:** Reference-count barriers by `(device, inode)` and close/remove them when
the last store releases the root, or keep only weak registry values with an
explicit owned-resource lifetime. Add close/reopen tests for many distinct roots
and for remove/recreate of the same path.

### WR-03: WARNING — The evidence-CAS regression test never exercises the cross-process primitive

**File:** `tests/test_blob_store_reconciliation.py:580-615`

**Issue:** The test creates two repository objects but races them in two threads
inside one interpreter. The module-wide `RLock` alone is sufficient for this
test to pass even if every `flock` call is removed or broken. Consequently the
test cited as proof of cross-process exact CAS does not validate the behavior
that motivated CR-01. The clear test does spawn a process, but it exercises the
admission lock, not checkpoint-vs-checkpoint or checkpoint-vs-retire evidence
CAS.

**Fix:** Spawn independent processes with separately constructed
`ManagedFileOps`/repositories, synchronize them immediately before the exact
transition, and assert one winner for checkpoint-vs-checkpoint and
checkpoint-vs-retire. Run this on each supported lock implementation.

---

_Reviewed: 2026-08-31T14:42:00Z_  
_Reviewer: the agent (gsd-code-reviewer)_  
_Depth: deep_
