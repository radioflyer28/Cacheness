---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-01T20:50:15Z
depth: deep
files_reviewed: 33
files_reviewed_list:
  - docs/SECURITY.md
  - docs/WINDOWS_COMPATIBILITY.md
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
  - src/cacheness/storage/integrity.py
  - src/cacheness/storage/lifecycle.py
  - src/cacheness/storage/manifest.py
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
  warning: 2
  info: 0
  total: 5
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-01T20:50:15Z
**Depth:** deep
**Files Reviewed:** 33
**Status:** issues_found

## Summary

This review accepts D-21 and D-22 as the canonical product boundary: the local
store owner is trusted for lifecycle-control availability, and Windows sharing is
limited to one OS user in one interactive or service session. Native Windows
execution remains a later evidence gate. The findings below are deterministic
implementation contradictions inside that supported topology, including observable
substitution that the contract explicitly promises to reject.

Several iteration-12 fixes hold. Key-provider results are now checked for exact
32-byte material and ordinary provider exceptions are translated with provider and
operation context. Pending recovery advances past a Windows identical-final
candidate instead of charging the same lexical candidate forever. Startup defers a
tombstone only for a signed, primary-byte-bound checkpoint. Malformed sidecars are
visible in dry-run and do not abort a matching primary action. Completed orphan
sidecars and primary actions share one mutation counter. Pristine JSON, memory, and
SQLite apply is an idempotent no-op without creating a key. Exact raw manifests,
native handler formats, containment, CAS, clear/admission, close ownership, and
normal-read non-mutation remain green.

Three blockers remain. First-key readiness still reads an EEXIST winner before
taking the interprocess lock, cannot recover a crash-partial key or readiness
record, and treats a fully written but unacknowledged readiness filename as
complete after close/directory-ack failure. Its Windows flush still does not bind
the retained native handle's identity to the expected key identity. Pending and
sidecar “pages” bound stored results and file reads but scan every directory name;
moreover a pending-only page with more entries emits no resume token and loses the
remaining candidates from that report traversal. Finally, an authenticated
sidecar whose digest/action does not bind its primary is reported merely as
attached, then raises a reconciliation conflict that aborts every later apply
action.

Two warnings cover recovery compatibility and the exact destructive-call oracle.
The version-2 decoder claims version-1 support but expects a field that released
version-1 tokens never contained, invalidating every outstanding resume token.
Tombstone interruption before or inside the post-delete checkpoint calls
delete_durable again after reopen; the new test counts successful deletions, not
calls, so it does not establish the claimed no-replay contract.

The complete focused Phase 3 corpus passes with the two documented platform
skips. Python 3.11.16 compile/import, lock, diff, and changed-path Ruff gates pass.
The full suite passes when run outside the filesystem sandbox; the sandbox-only
Unix-socket and nested-uv failures were rerun in the supported environment. Native
Windows was not executed. Green suites omit the cross-process partial-key window,
readiness-publication crash seams, retained-handle identity correlation,
pending-only final page, directory-name call bound, mismatched authenticated
sidecar, released v1 token, and exact delete-call cases reproduced below.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: BLOCKER — The key-readiness protocol can fail early, brick recovery, or authorize an unacknowledged record

**Files:** `src/cacheness/storage/integrity.py:126-370`,
`src/cacheness/storage/integrity.py:437-474`,
`src/cacheness/storage/path_security.py:234-287`,
`tests/test_blob_store_integrity.py:75-203`

**Issue:** an EEXIST initializer calls
`_read_existing_key_and_identity()` before it enters
`_acknowledgement_lock()`. A separate process can therefore observe the
exclusive-created file while the winner has not written 32 bytes and fail
immediately instead of waiting for readiness. A deterministic fork probe blocked
the winner's first `os.write`; the loser returned
`ManifestKeyError: Canonical manifest key must contain exactly 32 bytes`, then
the winner completed normally.

Crash recovery is not complete on either record. A process loss during the key
write leaves a regular, single-link, short key that every later initializer rejects
before it can lock or safely retire that exact inode. A process loss or write/fsync
failure during `_publish_ready_record()` leaves an exclusive-created partial
`.ready` file; later `_ready_matches()` raises permanently and publication
cannot replace it. Conversely, once the ready bytes happen to be complete,
`_complete_key_acknowledgement()` returns immediately on a byte match. If ready
close, parent-directory fsync, or the Windows readiness flush failed, a retry skips
the failed durability acknowledgement. A probe failed the first ready-directory
acknowledgement: the first initialization raised, the second returned the key, and
the ready acknowledgement call count remained one.

The Windows adapter now checks reparse and link invariants on the handle, but does
not compare that handle's native volume/file ID with the caller's expected
identity. Its `os.lstat(locator)` check is a separate pathname observation.
Replacing the name before CreateFile and restoring it before lstat can flush a
different retained regular handle while both pathname checks see the expected key.
That is an observable check/open/check substitution inside D-22's supported
one-user/session topology.

This violates D-04, D-05, D-07, D-11, D-18, SECU-04, and D-22.

**Fix:** acquire the exact key initialization authority before reading winner
bytes, and wait/retry boundedly while another holder is publishing. Model key and
readiness creation as recoverable states: under the lock, safely retire only the
exact unacknowledged partial inode or finish its write/acknowledgement, and never
treat ready bytes as complete until their namespace/platform acknowledgement is
durably recorded or re-executed. Bind the Win32 handle's native file identity to
the expected identity, not to a second pathname stat. Add true cross-process
partial-write, process-loss, short-ready, ready-close, ready-directory-ack, and
native-handle substitution tests with descriptor/resource accounting.

### CR-02: BLOCKER — Pending/sidecar inventory is not name-bounded and pending-only pagination loses work

**Files:** `src/cacheness/storage/operation_repository.py:533-593`,
`src/cacheness/storage/operation_repository.py:1243-1371`,
`src/cacheness/storage/reconciliation.py:318-415`,
`src/cacheness/storage/reconciliation.py:948-955`,
`tests/test_blob_store_reconciliation.py:1031-1089`

**Issue:** both new page methods use `nsmallest(limit + 1, generator)`.
They retain and read only a bounded result page, but `nsmallest` must enumerate
every directory entry to establish lexical order. With page size two, a probe
containing 100 eligible pending names returned two records after inspecting all
100 names. The same algorithm is used for sidecars. Large crash residue therefore
still has unbounded directory-call/CPU latency on startup and every report page,
contrary to the strict inventory/call bound.

Pending resume also has a termination bug. `next_pending` can be non-null while
manifest, operation, and sidecar sources are all exhausted, but
`next_priority` is selected only from those three primary sources. It remains
None, so `_encode_resume_token()` returns None and discards the pending cursor.
With `operation_page_size=2`, three digest-invalid pending candidates, and no
other evidence, a deterministic dry-run returned two pending findings and no
resume token. The third candidate is unreachable in that traversal. This also
means “resume=no” is not a truthful completion signal.

Windows identical-final scheduling itself now advances and the durable startup
cursor prevents that candidate from permanently starving a later valid candidate.
The defect is the report cursor and inventory bound, not a demand to delete the
blocked Windows pathname.

This violates D-05, D-13 through D-16, STOR-04, STOR-06, and D-22's bounded local
recovery contract.

**Fix:** use an actually bounded backend enumeration/index primitive with a stable
opaque cursor; bound names inspected, bytes read, and actions separately. Include
pending as a first-class resume source (or encode a pending cursor independently
of primary priority), and emit a token whenever any source has remaining work.
Exercise pending-only and sidecar-only multi-page traversal to terminal completion,
insertion around cursors, restart/wrap behavior, 100x invalid prefixes, and
simulated/native Windows identical-final plus later-valid progression while
asserting exact enumeration/read/action counts.

### CR-03: BLOCKER — A validly signed but primary-mismatched sidecar aborts apply and starves later work

**Files:** `src/cacheness/storage/reconciliation.py:430-468`,
`src/cacheness/storage/reconciliation.py:470-513`,
`src/cacheness/storage/reconciliation.py:832-864`,
`src/cacheness/storage/reconciliation.py:1206-1255`,
`src/cacheness/storage/lifecycle.py:1097-1126`,
`tests/test_blob_store_reconciliation.py:1095-1185`

**Issue:** startup correctly requires the checkpoint signature, operation ID,
COMPLETE_TOMBSTONE action, and exact primary digest before deferring recovery.
Dry-run classification does not apply the same binding. Any signed sidecar with a
matching operation ID is reported as
`reconciliation_checkpoint_attached`, even if its evidence digest or action is
for different primary bytes.

The corresponding primary can still classify as a safe action. Apply reaches
`_prepare_action_checkpoint()`, detects the digest/action mismatch, and raises
`CacheBlobReconciliationConflictError`. `_apply_findings()` isolates only
`CacheBlobReconciliationCheckpointError`, so the whole bounded apply aborts and
later independent safe findings are never reached. A deterministic tombstone
probe installed a correctly signed `prepared` checkpoint with the same operation
ID and a different digest. Dry-run reported it as attached plus the primary as
safe; apply raised the conflict and retained both records. Repeated apply has the
same result.

This is authenticated evidence, but it is not authority for this primary. It must
be classified as a bound conflict and isolated, just as malformed sidecars are,
rather than becoming a global denial of reconciliation progress.

This violates D-13 through D-16 and STOR-06.

**Fix:** classify sidecars against the current exact primary bytes and expected
action/state before any primary is proposed safe. Emit a stable blocked/conflict
finding for a mismatched signed sidecar, keep both exact records untouched, and
continue later independent work. Catch per-finding binding conflicts at the apply
loop only after preserving their explicit report disposition. Add wrong-digest,
wrong-action, completed/attached/orphan, later-valid, repeated apply, and reopen
tests under the one shared action budget.

## Warnings

### WR-01: Released version-1 reconciliation tokens are rejected by the compatibility decoder

**Files:** `src/cacheness/storage/reconciliation.py:284-286`,
`src/cacheness/storage/reconciliation.py:1340-1398`,
`tests/test_blob_store_reconciliation.py:923-965`

**Issue:** commit 8ca6209 increments `_TOKEN_VERSION` to 2 and explicitly accepts
version 1, but its version-1 schema requires
`{"manifest","operation","sidecar","priority"}`. The released parent implementation
encoded version 1 with only `manifest`, `operation`, and `priority`. A token
constructed by the exact prior encoder decrypts successfully with the unchanged
domain/key and is then rejected as malformed. Outstanding operator resume tokens
cannot resume after upgrade, despite the apparent compatibility branch.

**Fix:** decode the actual three-field v1 schema, map missing sidecar and pending
cursors to None, restrict its priority to the two values v1 emitted, and keep v2's
five-field validation exact. Add a fixed pre-8ca6209 token vector and prove it
resumes the same cursor without weakening authentication or byte bounds.

### WR-02: Post-delete checkpoint faults replay the payload deletion call

**Files:** `src/cacheness/storage/reconciliation.py:691-730`,
`src/cacheness/storage/reconciliation.py:810-830`,
`tests/test_blob_store_reconciliation.py:1235-1333`

**Issue:** after `_delete_or_prove_absent()` succeeds, the normal path calls
`_advance_tombstone_checkpoint()` outside the deletion try/except. A
`BaseException` at
`reconcile_tombstone_before_payload_deleted_checkpoint` or
`...inside_payload_deleted_checkpoint` leaves the sidecar in `prepared`.
Reopen enters the payload-delete stage again and calls
`delete_durable()` on the absent locator. A deterministic before-checkpoint
probe observed two calls for the same locator.

The parameterized regression increments its counter by `int(deleted)`, so a
second call returning False is invisible and the test's `payload_deletes == 1`
assertion proves only one successful unlink, not the claimed exact call count.
This contradicts D-14's “without repeating destructive work” and the fix report's
exactly-one-call statement even though the second call is currently idempotent.

**Fix:** once absence is observed, durably acknowledge that post-effect fact before
propagating every checkpoint-boundary BaseException, without allowing a checkpoint
fault to erase the original interruption. Count invocations separately from
successful effects. Parameterize persistence failures and every before/inside/after
effect and checkpoint seam, reopen repeatedly, and assert exact calls, successful
effects, checkpoint bytes, primary evidence, and sidecar retirement ordering.

## Info

None.

## Verification

- The complete focused Phase 3/compatibility command passed; only the documented
  Windows-junction and unavailable device-node fixtures skipped.
- The complete repository suite passed outside the filesystem sandbox with its
  documented optional/platform skips and existing collection warning. The first
  sandbox run's Unix-socket PermissionError and nested-uv cache error were
  environmental and were rerun rather than treated as product findings.
- CPython 3.11.16 isolated `compileall` and `import cacheness` passed with the
  declared `recommended` extra (`0.3.14`).
- `uv lock --check`, `git diff --check`, `git diff --check HEAD^ HEAD`, and
  changed-path Ruff passed.
- A forked EEXIST probe failed the loser on the winner's zero-byte key before the
  winner completed. A failed ready-directory acknowledgement was not retried.
- A page-size-two probe inspected 100 names to return two pending records. A
  pending-only three-record report returned two findings with no resume token.
- A signed wrong-digest sidecar classified as attached, then aborted apply with
  `CacheBlobReconciliationConflictError`.
- A pre-8ca6209 v1 token was rejected. A post-delete/pre-checkpoint interruption
  invoked `delete_durable()` twice for the same payload.
- Pristine JSON, memory, and SQLite apply remained no-op without key creation.
  Exact CAS, clear/admission, close ownership, containment, native/no-wrapper
  formats, normal-read non-mutation, and conflict-provenance gates remain green.
- Native Windows execution was unavailable; source/adapter inspection found the
  retained-handle identity contradiction in CR-01.

---

_Reviewed: 2026-09-01T20:50:15Z_
_Reviewer: the agent (general code reviewer)_
_Iteration: 13_
