---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-01T15:06:57Z
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
  critical: 5
  warning: 1
  info: 0
  total: 6
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-01T15:06:57Z
**Depth:** deep
**Files Reviewed:** 27
**Status:** issues_found

## Summary

The full 27-file Phase 3 scope, every source/test change in `37730b1`, and the
relevant JSON-backend callers were reviewed again without trusting the fix
report. The 11 focused phase modules pass, with only the existing Windows
junction fixture skipped. Those tests do not establish convergence: adversarial
probes reproduced a lost exact-CAS update through operation-lease inheritance,
an authority publication into a replacement JSON root, acceptance of an
outside hard link as the authority-lock inode, and provenance-free temporary
residue after a process loss between no-replace installation and cleanup.

Five blockers remain. The new nested-lock shortcut avoids the reported
deadlock by discarding cross-process exclusion for nested operation IDs; JSON
authority is still path-based after its lock check and its lock inode remains
replaceable; the purported Windows path still executes directory `open`/`fsync`
operations that are not implemented by the actual Windows fallback; and the
new crash-atomic create protocol leaves unreportable hard-link temporaries at a
real process-loss boundary. Failed direct JSON-repository construction also
leaks its internally opened root descriptor.

This review does not count the documented repository-wide Ruff baseline or the
known shutdown-only `SqliteBackend.__del__` failure. The review reconfirmed the
bounded/versioned AEAD resume token, authenticated interrupted-inventory abort,
bounded evidence/chunk parsing, typed `exists`/`list` behavior, signed clear
artifact retirement, close ownership tracking, Python 3.11-compatible syntax,
public dependency/export updates, native handler-owned formats, absence of a
Cacheness payload wrapper/header, non-mutating normal reads, and the prohibition
on payload provenance guessing.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: BLOCKER — Root-wide lease inheritance removes exact-CAS exclusion for nested operation IDs

**Files:** `src/cacheness/storage/operation_repository.py:151-227`, `src/cacheness/storage/lifecycle.py:555-626`, `src/cacheness/storage/lifecycle.py:1280-1400`, `tests/test_blob_store_reconciliation.py:714-778`

**Issue:** `_has_held_operation_lease()` returns true when the thread holds
*any* operation lease for the root. `_conditional_transition()` then yields
without either the child operation's process-wide `RLock` or its advisory file
lock. During `_continue_clear()`, the outer clear ID is leased while
`_delete_clear_target()` creates and advances a distinct child delete ID. A
concurrent process recovering or reconciling that child still locks the child
stripe, but the clear thread does not, so their read/compare/write sequences do
not exclude one another. The outer lock is only one hashed operation stripe; it
is not root-wide authority and cannot protect the child record.

A deterministic probe against the current code forced the nested writer and a
normal exact writer to read the same child bytes before publication. Both
`checkpoint_if_exact()` calls returned success (`['exact-won', 'nested-won']`)
and the last writer silently replaced the first. The regression at lines
714-778 asserts that nested calls open no lock, so it codifies the lost-CAS
condition rather than proving safety. This can regress/retire recovery evidence
and repeat or lose destructive progress after a clear/recovery race.

**Fix:** Never infer authority from “some lease on this root.” Reentrant bypass
may apply only to the exact same operation lease. Restructure clear so child
delete work does not acquire a second unordered lock while the parent is held:
for example, durably claim/checkpoint one target under the parent lease, release
it, perform the child lifecycle under its own exact lease, then reacquire the
parent and exact-CAS the completion; or introduce an explicit, common lease
token whose lock actually synchronizes every participant. Add a deterministic
two-repository/process test that forces both writers past the child comparison
and proves exactly one succeeds, in addition to bounded same-stripe and
opposite-order completion tests.

### CR-02: BLOCKER — JSON authority publication can switch to a replacement root after validation

**Files:** `src/cacheness/storage/manifest_repository.py:350-408`, `src/cacheness/storage/path_security.py:278-319`, `src/cacheness/metadata.py:961-1064`

**Issue:** `_json_compare_publish_lock()` verifies the managed lock name/root
identity before acquiring the retained lock descriptor, but the actual JSON
authority refresh and durable replacement are delegated to `JsonBackend` using
raw pathname operations. There is no managed-root check after the authority
lock is acquired and no descriptor-relative read/publish for the metadata file.
If the configured root is renamed and recreated after `file_identity()` returns,
`_read_live_document()` and `_save_to_disk()` operate in the replacement root.
The retained descriptor still locks the old root's inode and therefore provides
neither containment nor synchronization for the new authority file.

A direct probe wrapped `file_identity()` so it returned the valid old inode and
then renamed/recreated the root. `publish_if_expected()` succeeded and created
`metadata.json` in the replacement root (`published True False` for new versus
retired root). In a real `BlobStore.put`, that boundary occurs after native
candidate publication and manifest signing, producing authority in one root
that names payload bytes in the retired descriptor-anchored root.

**Fix:** Move JSON canonical-byte refresh and durable compare/publication into a
descriptor-anchored repository primitive. Keep the metadata directory/file
identity inside the same managed boundary as the retained authority descriptor,
revalidate the root after lock acquisition, and never call path-based
`JsonBackend` I/O after that check. Add a fault seam immediately after lock
validation and after lock acquisition; replacing the root at either seam must
raise a typed path/backend failure and create no document in the new root.

### CR-03: BLOCKER — Mutable and hard-linked lock inodes can split every advisory authority boundary

**Files:** `src/cacheness/storage/coordination.py:223-261`, `src/cacheness/storage/coordination.py:365-373`, `src/cacheness/storage/manifest_repository.py:216-227`, `src/cacheness/storage/manifest_repository.py:350-365`, `src/cacheness/storage/path_security.py:399-425`, `src/cacheness/storage/path_security.py:884-907`

**Issue:** Managed lock opens reject symlinks/reparse points but accept regular
files with `st_nlink > 1`, and the JSON identity check is separated from the
actual lock acquisition. A hard link from outside the store can therefore be
installed as the lock before construction; the current repository accepts it
and retains the outside inode. A probe confirmed construction with link count
2 and `_json_lock_identity` equal to the outside file. A second probe replaced
the live lock name with another regular inode after `file_identity()` returned;
the old repository still published successfully. A repository opened after the
replacement locks the new inode, so two writers can concurrently execute
refresh/compare/publish against the same metadata document.

The shorter `interprocess_file_lock()` path is weaker: admission and evidence
stripes open the mutable name afresh for every operation and retain no expected
inode at all. Substitution between two process opens lets a clear's exclusive
admission overlap an ordinary shared admission, or lets two evidence CAS
writers lock different stripe files. That breaks the barriers used to justify
snapshot completeness and exact mutation.

**Fix:** Treat lock identity as authority data, not merely a no-follow path.
Reject multi-link lock files, retain one verified descriptor per root/lock
identity, and make validation plus acquisition resistant to name substitution
(or use an immutable descriptor-anchored lock object/backend-native primitive
whose identity cannot be swapped independently). Fail closed if the topology
cannot supply that guarantee. Add hard-link fixtures and two-process tests that
swap a regular lock inode between validation/open/acquisition and prove no two
critical sections overlap.

### CR-04: BLOCKER — The claimed Windows path still depends on unsupported directory fsync behavior

**Files:** `src/cacheness/storage/path_security.py:240-251`, `src/cacheness/storage/path_security.py:584-605`, `src/cacheness/storage/path_security.py:677-702`, `src/cacheness/metadata.py:941-959`, `tests/test_manifest_repository_cas.py:178-241`, `tests/test_blob_store_close_contract.py:339-363`

**Issue:** Windows deliberately disables descriptor mode, so durable exclusive
lock/control creation takes `_create_bytes_durable_exclusive_fallback()`. After
the hard-link install it calls `_fsync_containing_directory()`, whose fallback
uses `os.open(directory, O_RDONLY)` followed by `os.fsync(directory_fd)`. The
CRT-backed Windows `os.open` does not open directories with the required
`FILE_FLAG_BACKUP_SEMANTICS`, and directory `fsync` is not implemented by this
path. `JsonBackend._fsync_metadata_directory()` repeats the same assumption.
Consequently default JSON construction/publication can fail before the new
`LockFileEx` adapter is useful.

Both “Windows” tests run on POSIX and only monkeypatch
`coordination._platform_name`; `os.name`, `ManagedFileOps.descriptor_mode`, hard
link publication, directory durability, and the real `_NativeWindowsLockApi`
remain the host POSIX implementations. They are not evidence that the actual
Win32 branch works.

**Fix:** Implement truthful Windows durability with Win32 directory handles and
the appropriate flush/replace/no-replace primitives, or fail construction with
a typed unsupported-capability error before claiming the lifecycle. Run an
actual Windows CI job that constructs a default JSON store, races two processes
on exact same-key CAS, creates operation/admission lock files, and completes
put/delete/clear/reopen using the production `LockFileEx` adapter.

### CR-05: BLOCKER — Crash-atomic no-replace publication creates provenance-free temporary residue

**Files:** `src/cacheness/storage/path_security.py:530-612`, `src/cacheness/storage/operation_repository.py:832-865`, `src/cacheness/storage/operation_repository.py:1037-1103`

**Issue:** The new control-record primitive uses a random
`.final.<uuid>.tmp` hard link, installs the final name, fsyncs the directory,
then unlinks the temporary. A real process loss after the durable link and
before temporary retirement bypasses the Python exception cleanup. A forked
probe stopped at that exact boundary and left both the valid final record and
`.aaaaaaaa...json.<uuid>.tmp` as hard links to the same inode. Operation and
clear-page inventory accept only exact final grammars, so recovery and
reconciliation ignore the temporary. Once the authenticated final is retired,
the hidden link keeps the evidence bytes alive with no persisted exact locator
and no evidence-gated cleanup path. Repeating this boundary accumulates
unbounded, unreportable residue and violates STOR-04/D-06 rather than merely
leaving resumable operation debt.

**Fix:** Make temporary identity part of authenticated operation provenance
before it can persist, or use a publication primitive/protocol whose pre-final
temporary is deterministically and safely inventoryable by exact inode/digest
binding to authenticated evidence. Reconciliation must report and retire only
the exact proven temporary, never a glob match. Add subprocess `os._exit` tests
after temp creation, final link, first directory fsync, temp unlink, and second
directory fsync; reopen must converge with no final or temporary control residue.

## Warnings

### WR-01: WARNING — Failed direct JSON repository construction leaks its owned root descriptor

**File:** `src/cacheness/storage/manifest_repository.py:193-235`

**Issue:** When no `file_ops` is supplied, the constructor creates and owns a
`ManagedFileOps` at lines 201-203. Cleanup is installed only around the later
`open_read()` block. If `create_bytes_durable_exclusive()` fails for any reason
other than `FileExistsError`—including unsupported no-replace/durability on the
filesystem—the constructor exits without closing `_json_lock_file_ops`. In
`BlobStore`, assignment to `self.manifest_repository` has not completed, so its
failed-initialization cleanup cannot reach this partially constructed owner.
Repeated failed opens leak raw root descriptors until process limits are hit.

**Fix:** Wrap authority-lock creation and open in one constructor-level
`try`/`except BaseException` that closes any lock handle and every internally
owned `ManagedFileOps` before reraising. Add descriptor-count/close-spy tests for
failures during create, open, identity capture, and post-open validation.

---

_Reviewed: 2026-09-01T15:06:57Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
