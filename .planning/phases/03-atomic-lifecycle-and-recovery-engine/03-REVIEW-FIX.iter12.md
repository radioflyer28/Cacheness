---
phase: 03-atomic-lifecycle-and-recovery-engine
fixed_at: 2026-09-01T19:04:40Z
review_path: /Users/akriz/code/cacheness/.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.md
iteration: 10
findings_in_scope: 9
fixed: 9
skipped: 0
status: all_fixed
---

# Phase 03: Code Review Fix Report

**Fixed at:** 2026-09-01T19:04:40Z
**Source review:** `03-REVIEW.md`
**Iteration:** 10

## Summary

- Findings in scope: 9
- Fixed: 9
- Skipped: 0

## Fixed Issues

### CR-01: Stale overwrite/delete could form a CAS expectation from a newer read

**Files modified:** `src/cacheness/storage/blob_store.py`,
`src/cacheness/storage/lifecycle.py`,
`tests/test_blob_store_atomic_lifecycle.py`
**Commits:** `684a658`, `44dcbb3`
**Applied fix:** Authenticated manifest loading now returns the exact raw record
observed by the operation. Overwrite, delete, tombstone, and metadata-update CAS
expectations are constructed from that raw record rather than a later repository
read. A later winner therefore makes the original operation lose with a typed
lifecycle conflict; it cannot revoke or adopt the winner. Operation evidence uses
the same captured authority snapshot.

### CR-02: Authenticated pre-CAS PUT residue could remain permanently blocked

**Files modified:** `src/cacheness/storage/lifecycle.py`,
`tests/test_blob_store_atomic_lifecycle.py`
**Commit:** `684a658`
**Applied fix:** Recovery now recognizes the narrow safe case in which authenticated
record/provenance/containment/candidate ownership are valid and current authority
has both a distinct generation and locator. It checkpoints any unfinished
candidate publication, deletes only that exact candidate, and retires the exact
evidence. Same-locator/different-generation residue remains blocked.

### CR-03: The one-user/session Windows topology lacked a persistent key-provider seam

**Files modified:** `src/cacheness/storage/blob_store.py`,
`src/cacheness/storage/integrity.py`, `tests/test_blob_store_integrity.py`
**Commit:** `684a658`
**Applied fix:** `BlobStore` accepts an injected manifest key provider. The default
provider now has a portable strict path: it rejects unsafe/reparse/non-regular or
multi-linked key files, uses no-follow open where available plus identity checks,
and retains atomic exclusive creation and file durability. POSIX ownership/mode
checks remain additional POSIX policy rather than an unconditional non-POSIX
rejection. Tests cover injected providers and the non-POSIX source contract.

### CR-04: Windows deletion used unsupported `MoveFileExW(path, NULL, ...)`

**Files modified:** `src/cacheness/storage/path_security.py`,
`tests/test_manifest_repository_cas.py`
**Commit:** `684a658`
**Applied fix:** The native adapter now opens the target with DELETE access and
reparse-safe flags, validates the opened handle as a single-link regular target,
then uses the documented `SetFileInformationByHandle(FileDispositionInfo)`
immediate-disposition protocol. Native failures preserve error context through the
typed backend boundary; the acknowledgement is accurately limited to successful
handle-based disposition rather than an unclaimed directory-flush guarantee.

### WR-01: Failed retained-resource cleanup could be forgotten before a retry

**Files modified:** `src/cacheness/storage/operation_repository.py`,
`src/cacheness/storage/coordination.py`,
`tests/test_blob_store_close_contract.py`
**Commit:** `684a658`
**Applied fix:** Lock handles, root descriptors, and barrier registrations remain
tracked until their close/release succeeds. A failure remains retryable (or keeps
the barrier poisoned), so a second close cannot falsely report a clean CLOSED
state while a retained native resource is still live. First, middle, and final
failure positions are exercised.

### WR-02: Unrelated pending-control names could exhaust the recovery bound

**Files modified:** `src/cacheness/storage/operation_repository.py`,
`tests/test_manifest_repository_cas.py`
**Commit:** `684a658`
**Applied fix:** Pending recovery filters exact eligible operation-control names
before its stable bounded selection. Ordinary, malformed, and clear-control names
therefore cannot starve eligible operation recovery across repeated reopens.

### WR-03: Windows registry/mutex errors escaped the backend error taxonomy

**Files modified:** `src/cacheness/storage/coordination.py`,
`src/cacheness/storage/path_security.py`,
`tests/test_manifest_repository_cas.py`
**Commit:** `684a658`
**Applied fix:** Registry/mutex creation, wait, query/set, release, and close now
translate capability/policy statuses to `CacheBlobBackendError` with the stable
capability reason; other statuses use the stable backend-failure reason. Causes
and operation context are retained, including release/close uncertainty.

### WR-04: Tombstone conflict cleanup could mask the authoritative conflict

**Files modified:** `src/cacheness/storage/lifecycle.py`,
`tests/test_blob_store_atomic_lifecycle.py`
**Commit:** `684a658`
**Applied fix:** Both delete conflict-retirement paths now translate a failed
evidence retirement into a typed recoverable-cleanup error whose cause/context
retain the original conflict and post-authority state. The later winner remains
the authoritative manifest.

### WR-05: Completed reconciliation checkpoints accumulated indefinitely

**Files modified:** `src/cacheness/storage/operation_repository.py`,
`src/cacheness/storage/reconciliation.py`,
`tests/test_blob_store_reconciliation.py`
**Commit:** `684a658`
**Applied fix:** Reconciliation records a completed checkpoint, retires primary
evidence, then retires that exact checkpoint using its raw bytes. Reopen
authenticates and retires an orphaned completed checkpoint without repeating the
destructive action. Successful steady state leaves no primary/action sidecars.

## Verification

Verification ran in the **main checkout** for the initial focused suite and in
isolated temporary environments for the final cross-version probes, so shared
work was not disrupted.

- A clean isolated CPython 3.13.3 full suite passed, with only the documented
  optional/platform skips and the existing collection warning.
- CPython 3.13 targeted review regressions: 15 passed.
- CPython 3.11.16: `python -m compileall -q src/cacheness` and `import cacheness`
  passed; the same 15 targeted review regressions passed.
- The complete Phase 3 focused module suite passed after `684a658`; its
  `test_blob_store_read_contract.py` compatibility follow-up passed after
  `44dcbb3`.
- Changed-path Ruff passed. `uv lock --check` and `git diff --check` passed.

## Remaining Platform Verification Gap

The native Windows filesystem, ACL, one-user/session topology, and crash/reopen
path have not executed on a Windows host. Adapter and source-contract tests cover
the native calls and typed error mapping, but do not claim native-platform proof.
The later platform/CI gate must run that verification.

---

_Fixed: 2026-09-01T19:04:40Z_
_Fixer: the agent (gsd-code-fixer)_
_Iteration: 10_

