---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-01T21:31:01Z
depth: deep
files_reviewed: 35
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
  - tests/test_blob_manifest.py
  - tests/test_blob_manifest_backends.py
  - tests/test_clear_recovery.py
  - tests/test_config_validation.py
  - tests/test_filesystem_containment.py
  - tests/test_manifest_repository_cas.py
  - tests/test_public_api_contract.py
  - pyproject.toml
  - uv.lock
findings:
  critical: 3
  warning: 0
  info: 0
  total: 3
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-01T21:31:01Z
**Depth:** deep
**Files Reviewed:** 35
**Status:** issues_found

## Summary

This is a fresh review of commit `36a3110`. It accepts D-21 and D-22 as the
canonical boundary: the local store owner is trusted for lifecycle-control
availability, and Windows sharing is limited to one OS user in one interactive or
service session. Native Windows execution remains a later evidence gate. The
findings below do not require a hostile store owner or unsupported Windows
topology.

Several iteration-13 fixes are real. Every EEXIST key read now occurs after the
stable initialization authority is acquired. A completed crash-partial key or
readiness record can be retired only after exact identity validation, and complete
ready bytes re-execute their platform acknowledgement. The Windows key path derives
native identity and flushes the same retained descriptor. Exact signed sidecars are
checked against primary bytes, action, and legal state before stable-snapshot apply.
The decoder accepts the exact released three-field v1 token schema; its checked-in
constant matches the encoder immediately preceding v2. Tombstone recovery observes
absence before issuing deletion and durably advances after every tested post-effect
or checkpoint interruption, so the exact payload-delete call is not replayed.
Pristine apply remains non-mutating.

Three blockers remain. First, key initialization waits on blocking `flock` or
`LockFileEx` with no timeout, so the advertised bounded attempt loop begins only
after an arbitrarily long wait. Second, the new bounded directory slice combines
unspecified filesystem order with lexical cursors, counts only filtered matches,
and uses the wrong sidecar cursor namespace after partial consumption. This skips
pending evidence and can duplicate/livelock sidecar reports. Third, apply retires
completed orphan sidecars in a pre-pass and then charges their unchanged SAFE
findings again; exact repository conflicts are also outside the per-finding catch.
A shared action budget can therefore expire without applying a separately safe
primary, or one sidecar race can abort every later action.

The focused Phase 3 corpus and complete repository suite pass outside the sandbox;
the initial sandbox run failed only at its prohibited Unix-socket fixture. Python
3.11.16 compile/import, the lockfile, current/commit whitespace checks, and
changed-file Ruff pass. Native Windows was not executed. Green tests do not assert
a lock-acquisition deadline, directory names inspected, non-lexical enumeration,
partial safe-sidecar continuation, sidecar-plus-primary shared-budget accounting,
or exact sidecar CAS races. Deterministic probes reproduced the skip, livelock, and
double-budget defects below.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: BLOCKER — First-key initialization can wait forever before its bounded loop begins

**Files:** `src/cacheness/storage/integrity.py:139-152`,
`src/cacheness/storage/integrity.py:233-295`,
`src/cacheness/storage/coordination.py:155-213`,
`tests/test_blob_store_integrity.py:145-170`,
`tests/test_blob_store_integrity.py:220-248`

**Issue:** the stable initialization authority correctly precedes every EEXIST key
read, but `_initialization_lock()` calls
`interprocess_open_file_lock()` without a deadline. POSIX uses blocking
`fcntl.flock(..., LOCK_EX)`; Windows uses blocking `LockFileEx`. The constant
`_MAX_KEY_INITIALIZATION_ATTEMPTS = 4` bounds only create/read/repair attempts
after that lock returns. An alive but stalled initializer, durability provider, or
filesystem lock holder therefore hangs every first use indefinitely. The
process-global `_INITIALIZATION_GUARD` also makes that unbounded wait serialize
unrelated stores in the same interpreter.

The concurrency test deliberately proves only that a loser is still alive after 50
ms and then manually releases the winner. The cross-process test starts recovery
only after the child has exited; it never exercises a live partial writer, a
bounded acquisition deadline, or a typed timeout. Crash recovery, ready
re-acknowledgement, exact-identity retirement, retained Win32-handle flush, and
normal descriptor cleanup otherwise hold under source inspection and focused
tests.

This violates D-05's bounded retry/backoff requirement and the deterministic
failure side of STOR-04/SECU-04.

**Fix:** add an explicit key-initialization deadline to the lifecycle policy and
use nonblocking `LOCK_NB` retry/backoff on POSIX and
`LOCKFILE_FAIL_IMMEDIATELY` retry/backoff on Windows. On expiry, close the exact
handle and raise the stable typed key/lifecycle outcome without reading or
retiring winner bytes. Scope the in-process guard by store/key authority rather
than one module-global mutex. Add true cross-process live-writer, timeout,
release-before-deadline, process-loss, exception, and descriptor-count tests on
POSIX and native Windows.

### CR-02: BLOCKER — Pending/sidecar cursors are neither stable nor truly bounded and can skip or livelock

**Files:** `src/cacheness/storage/path_security.py:885-950`,
`src/cacheness/storage/operation_repository.py:560-616`,
`src/cacheness/storage/operation_repository.py:1266-1304`,
`src/cacheness/storage/operation_repository.py:1512-1563`,
`src/cacheness/storage/manifest_repository.py:566-600`,
`src/cacheness/storage/reconciliation.py:977-995`,
`src/cacheness/storage/reconciliation.py:1422-1522`,
`tests/test_blob_store_reconciliation.py:269-307`,
`tests/test_blob_store_reconciliation.py:1070-1208`

**Issue:** `list_directory_names_bounded()` iterates `os.scandir()` in
filesystem order, stops after `max_names` filtered matches, sorts only that
slice, and resumes by rejecting names lexically less than or equal to its maximum.
Filesystem enumeration order is not lexical. A deterministic probe created
`z`, `a`, `b` in that order with limit two. Page one was `(a, z)` with
cursor `z`; page two was empty, permanently skipping `b`. Pending and sidecar
pages both use this primitive. Deletion, reinsertion, restart, and filesystem-order
changes can similarly skip or duplicate candidates.

The call bound is also false: names rejected by `name_filter` and names at or
before the lexical cursor are inspected without incrementing any counter. An
arbitrarily large unrelated or malformed prefix can make one “two-name” page scan
the entire directory. The unchanged primary operation inventory still uses
`nsmallest` over every directory entry, and JSON/memory manifest paging uses
`nsmallest` over every key, so complete reconciliation is not call-bounded even
after the new pending/sidecar helper.

There is an independent sidecar resume bug. When the shared report budget consumes
only part of a sidecar page, `_next_sidecar_cursor()` stores the bare 32-hex
operation ID. The repository cursor is otherwise a full filename such as
`reconcile-action-<id>.json`. On the next call every sidecar filename compares
greater than the bare ID, so the page restarts. With three completed orphan
sidecars and action budget one, two consecutive dry runs both returned only
sidecar `a...` and decoded to the same `a...` cursor. This is a deterministic
livelock despite AEAD authentication of the outer token.

The pending-only regression happens to create names in lexical order and checks
raw file reads, not directory names inspected. The sidecar budget test has one
blocked and one safe sidecar, so it consumes the entire safe set and never takes a
partial safe-sidecar cursor. No test covers reverse/insertion order, deletion,
restart/wrap, a large ineligible namespace, or terminal no-skip/no-dup traversal.

This violates D-05, D-13 through D-16, STOR-04, STOR-06, and the documented
bounded local-recovery contract.

**Fix:** introduce a genuinely stable backend inventory primitive: an indexed or
bucketed ordered namespace, or a durable authenticated snapshot whose continuation
position is independent of `scandir` order. Bound every directory entry
inspected, every record byte read, and every authorized action separately. Keep
cursor values in one namespace end-to-end (full filename or operation ID, never
both), and encode only validated positions in the AEAD token. Replace the remaining
operation and JSON/memory `nsmallest` scans or narrow the claimed bound. Test
pending-only and sidecar-only completion under reverse creation order,
insert-before/after, deletion, restart and wrap, partial safe pages, malformed and
unrelated prefixes, and 100x namespaces with exact name/read/action counters and a
terminal no-skip/no-duplicate invariant.

### CR-03: BLOCKER — Apply double-charges orphan sidecars and does not isolate exact sidecar races

**Files:** `src/cacheness/storage/reconciliation.py:453-494`,
`src/cacheness/storage/reconciliation.py:625-690`,
`src/cacheness/storage/reconciliation.py:873-927`,
`src/cacheness/storage/reconciliation.py:1247-1374`,
`src/cacheness/storage/operation_repository.py:633-713`,
`tests/test_blob_store_reconciliation.py:1168-1262`

**Issue:** stable-snapshot binding is now exact: digest, primary ID, action, and
legal checkpoint state are checked, and a signed mismatch blocks both the sidecar
and its primary without charging the report action budget. The apply accounting
around valid completed or racing sidecars is still wrong.

`_apply_findings()` first calls
`_retire_orphaned_completed_checkpoints()` and decrements `remaining` for each
retired sidecar. It then iterates the original findings. The same orphan finding is
still SAFE, `_apply_finding()` returns because no primary exists, and line 491
decrements `remaining` a second time. A deterministic budget-two probe installed
one completed orphan sidecar and one safe uncommitted primary. Apply retired the
sidecar, charged it twice, left the safe candidate and primary evidence untouched,
and returned findings that described both as safe. The report's resume decision is
computed before apply and can be terminal even though its advertised mutation
budget did not execute the second authorized action.

Per-finding conflict isolation also catches the wrong exception family. Exact
sidecar checkpoint/retirement CAS methods raise
`CacheBlobLifecycleConflictError`, while the loop catches only
`CacheBlobReconciliationCheckpointError` and
`CacheBlobReconciliationConflictError`. A concurrent exact sidecar advance during
`_advance_action_checkpoint()`, or a sidecar change during orphan retirement,
therefore aborts the aggregate apply and starves every later independent finding.
The mismatch regression is static and cannot exercise these exact-byte races.

This violates D-13 through D-15 and STOR-06's resumable, isolated, shared-budget
apply contract.

**Fix:** represent apply work as one normalized action stream and charge the shared
budget exactly once only when an action is actually attempted. Remove already
retired orphan findings from the later pass or mark them consumed, and generate
the continuation token from unapplied actions as well as scanned evidence. Convert
or catch repository exact-byte lifecycle conflicts at the per-finding boundary,
preserve both current records, emit a stable blocked/conflict disposition, and
continue later work. Add sidecar-plus-primary budgets of one/two, exact update and
retire races, completed/attached/orphan mixes, repeated apply, reopen, and
fault-injection tests that assert exact attempts, effects, remaining artifacts,
tokens, and later-action progress.

## Warnings

None.

## Info

None.

## Verification

- Preserved the exact previous live report as
  `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.iter12.md`;
  `cmp` returned zero before this report replaced the live file.
- Focused integrity and reconciliation suites passed.
- The complete Phase 3/compatibility command passed outside the filesystem
  sandbox with only the documented Windows-junction and unavailable-device-node
  skips. Its first sandbox run failed only because Unix-socket creation was
  prohibited.
- The complete repository suite passed outside the sandbox with documented
  optional/platform skips and the existing collection warning.
- CPython 3.11.16 isolated `compileall` and `import cacheness` passed with the
  declared `recommended` extra (`0.3.14`).
- `uv lock --check`, `git diff --check`, `git diff --check HEAD^ HEAD`, and
  changed-file Ruff passed.
- A bounded-directory probe returned `(a, z)`, cursor `z`, then no second page
  from a `z, a, b` directory, proving a skipped candidate.
- Three completed sidecars with action budget one returned sidecar `a...` twice
  across consecutive authenticated resume tokens, proving cursor livelock.
- One completed orphan sidecar plus one safe primary under budget two retired only
  the sidecar and left the primary/candidate intact, proving double charging.
- The exact released v1 encoder from `8ca6209^` has the same three-field schema,
  token domain, version byte, key derivation, and AEAD layout as the fixed test
  vector; current decoding accepts it.
- Exact stable-snapshot sidecar binding, pristine apply, crash-partial key/ready
  recovery, ready acknowledgement re-execution, retained-handle Windows source
  contract, resource cleanup paths, and exact no-replay payload-deletion seams
  passed inspection/tests.
- Native Windows execution was unavailable; no Windows-only behavioral claim is
  upgraded beyond source-contract review.

---

_Reviewed: 2026-09-01T21:31:01Z_
_Reviewer: Codex (general code reviewer)_
_Iteration: 14_
