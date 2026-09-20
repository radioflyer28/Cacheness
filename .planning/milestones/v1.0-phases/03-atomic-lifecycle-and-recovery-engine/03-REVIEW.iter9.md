---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-01T22:15:00Z
depth: deep
files_reviewed: 31
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
  warning: 5
  info: 0
  total: 9
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-01T22:15:00Z
**Depth:** deep
**Files Reviewed:** 31
**Status:** issues_found

## Summary

The trusted-store-owner and one-user/one-session Windows boundaries in D-21 and
D-22 are accepted as canonical. This review does not reintroduce the superseded
hostile-owner or cross-principal/session requirement. Within that declared
boundary, however, Phase 3 is not ready to pass.

Four blockers remain. A stale overwriter can authenticate generation A, reread
generation B, and then use B's exact bytes as its CAS expectation while retaining
A's generation and locator; a deterministic two-store probe made that stale
writer revoke B. Authenticated pre-authority PUT residue becomes permanently
blocked when any later generation wins. The supported Windows topology cannot
create or reopen a canonical store because its only signing-key provider rejects
every non-POSIX runtime. Even if a key were supplied, Windows durable deletion
calls `MoveFileExW(path, NULL, MOVEFILE_WRITE_THROUGH)`, although the documented
NULL-destination delete form requires `MOVEFILE_DELAY_UNTIL_REBOOT`; normal
payload/evidence deletion therefore has no supported native implementation.

Five warnings cover cleanup and recovery truthfulness. Retained descriptor owners
discard bookkeeping before close succeeds, so a retry can report `CLOSED` with a
live descriptor/lock. Pending-control recovery applies its budget before filtering
eligible names and can be starved forever. Windows registry/mutex failures still
escape as raw `OSError`. Delete conflict handlers can replace the required conflict
or recoverable-cleanup outcome with an evidence-retirement failure. Successful
reconciliation permanently leaves completed action checkpoints that no later
inventory can reach or retire.

The full and focused suites pass, as do Python 3.11 compile/import, lock, diff, and
focused Ruff checks. Those green checks do not exercise the missing race or native
Windows contracts. Native Windows execution remains a later owned verification
gap; CR-03 and CR-04 are source-level contradictions inside the topology D-22 says
is supported, not merely missing platform evidence.

The review reconfirmed committed-only normal reads, no normal-read mutation or
operation-record authority guessing, managed descriptor hashing with regular-file
and single-link enforcement, native handler formats without a Cacheness wrapper,
live clear continuation leasing, poisoned admission after uncertain unlock,
fixed-control convergence, POSIX xattr disappearance/mismatch fail-closed behavior,
and the current pending-control substitution safeguard.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: BLOCKER — A stale overwrite can bind CAS to a newer winner and revoke it

**Files:** `src/cacheness/storage/lifecycle.py:1130-1149`,
`src/cacheness/storage/lifecycle.py:1163-1181`,
`src/cacheness/storage/manifest_repository.py:83-98`,
`tests/test_blob_store_atomic_lifecycle.py:468-505`

**Issue:** `LifecycleEngine.put()` authenticates one `existing` manifest, then
performs a second independent `get_raw()`. If another store publishes between
those reads, the method combines the old manifest's generation/locator with the
new record's digest. `ManifestExpectation.matches()` compares only the digest, so
the stale writer's CAS is authorized against the intervening winner rather than
the exact authenticated record it observed. The persisted operation record also
contradicts itself: `expected_generation` names the old generation while
`expected_record_digest` names the new record.

A deterministic probe paused store A on its second authority read, let store B
commit `{'v': 1}`, then resumed A. A returned success and the final value was
`{'v': 2}`. B's valid winner was silently revoked. If the second read instead sees
absence, the `assert raw_expected is not None` path exposes an untyped
`AssertionError` rather than the required lifecycle conflict. Existing stale-
overwrite coverage injects at manifest publication, after the expectation is
already built, so it cannot detect this read-pair bug.

This violates D-02, D-18, and D-19: overwrite must expect the exact authenticated
record read by that operation, and stale contenders must conflict without revoking
the winner.

**Fix:** Build `ManifestExpectation` from `previous_manifest.canonical_bytes()` (or
retain the exact raw bytes returned by the authenticated loader) and reject any
second-read absence/change with `CacheBlobLifecycleConflictError`; never adopt a
newer unauthenticated-to-this-operation record as the expectation. Persist the
same generation/digest pair in operation evidence. Add deterministic create,
overwrite, and delete interleavings between authenticated load and expectation
construction, asserting one winner and typed losers.

### CR-02: BLOCKER — Superseded authenticated PUT residue can never converge

**Files:** `src/cacheness/storage/lifecycle.py:928-972`,
`src/cacheness/storage/reconciliation.py:650-727`,
`tests/test_blob_store_reconciliation.py:495-535`

**Issue:** Recovery cleans an authenticated PUT candidate only when authority is
absent, equals the operation's generation, or still equals its expected generation.
If a process dies before CAS and any later writer commits another generation,
`_recover_record()` falls through without action and reconciliation classifies the
same evidence as `operation_authority_ambiguous`. The operation-owned immutable
candidate and signed record remain forever even when the current authenticated
manifest has both a different generation and a different locator.

A deterministic probe interrupted an absent-key PUT at `manifest_publish`, opened
another store and committed a winner, then reopened and reconciled one hour later.
The winner stayed readable, but the first candidate and signed operation record
survived with `blocked/report_only/operation_authority_ambiguous`.

This violates STOR-04/STOR-06, D-07, D-11, D-14, and D-19: an authenticated loser
must clean or explicitly converge only its own candidate, not become permanent
unactionable residue after a normal crash/winner interleaving.

**Fix:** When record authentication, topology, locator containment, and immutable
candidate ownership are proven, and current authenticated authority has a different
generation *and* different locator, classify the operation candidate as safe stale
residue, checkpoint deletion, and retire exact evidence. Keep a same-locator/
different-generation case blocked. Cover create and replace crashes before CAS,
followed by a later winner and reopen/apply.

### CR-03: BLOCKER — The advertised same-user/session Windows store cannot sign or reopen

**Files:** `src/cacheness/storage/blob_store.py:311-315`,
`src/cacheness/storage/blob_store.py:988-1006`,
`src/cacheness/storage/integrity.py:87-102`,
`src/cacheness/storage/integrity.py:167-174`,
`docs/WINDOWS_COMPATIBILITY.md:5-19`

**Issue:** Every `BlobStore` constructs the file-backed `ManifestKeyProvider`, but
both initialization and reading explicitly reject `os.name != "posix"`. On native
Windows, the first signed lifecycle record/manifest fails with
`manifest_signing_key_invalid`, and a store cannot reopen a persisted key. The new
Win32 lifecycle coordination and durability adapters are therefore unreachable for
ordinary canonical use.

This is not a cross-principal/session objection. It contradicts D-22's statement
that advertised Windows compatibility otherwise remains in force for one OS user
and one interactive/service session. A direct non-POSIX provider probe reproduces
the rejection; current simulated Windows tests patch filesystem adapters while
still executing the POSIX key provider.

**Fix:** Implement or inject a Windows-safe persistent signing-key provider with
the declared same-user/session ACL, no-follow/reparse, atomic initialization,
durability, and reopen contract. Add native Windows fresh-store write/read and
reopen tests before claiming support.

### CR-04: BLOCKER — Windows durable deletion calls an unsupported `MoveFileExW` form

**Files:** `src/cacheness/storage/path_security.py:140-143`,
`src/cacheness/storage/path_security.py:1726-1743`,
`tests/test_blob_store_close_contract.py:378-405`,
`tests/test_manifest_repository_cas.py:784-800`

**Issue:** `_WindowsFileApi.delete_write_through()` calls
`MoveFileExW(existing, NULL, MOVEFILE_WRITE_THROUGH)`. Microsoft's `MoveFileExW`
contract documents a NULL destination as deletion-at-reboot only when
`MOVEFILE_DELAY_UNTIL_REBOOT` is set; `MOVEFILE_WRITE_THROUGH` describes a move and
has no documented immediate-delete form. The implementation comment claiming this
is a documented pathname deletion boundary is therefore false. On native Windows
the call has no supported success contract (normally an invalid-parameter failure),
so payload reclamation, tombstone completion, evidence retirement, clear, and
reconciliation cannot complete.

The injected adapter accepts `delete_write_through()` without exercising its native
arguments, while the source-level flag test covers only no-replace rename. A green
POSIX-host adapter suite cannot establish this operation.

**Fix:** Use a documented, identity-retaining immediate deletion protocol (for
example, a reparse-safe handle opened with `DELETE` plus
`SetFileInformationByHandle(FileDispositionInfo)`) and define its durability/
acknowledgement semantics. Preserve fail-closed identity checks and typed errors.
Add native Windows put/delete, overwrite cleanup, clear, operation retirement, and
crash/reopen tests under ordinary non-administrator ACLs.

## Warnings

### WR-01: Failed retained-lock cleanup is forgotten, so retry can report CLOSED with live descriptors

**Files:** `src/cacheness/storage/operation_repository.py:131-137`,
`src/cacheness/storage/coordination.py:417-437`,
`src/cacheness/storage/blob_store.py:907-939`,
`tests/test_blob_store_close_contract.py:220-252`

**Issue:** `FileOperationRecordRepository.close()` clears `_lock_handles` before
closing them. If any close raises, that handle and all later handles are absent from
retry bookkeeping. `StoreAdmissionBarrier.release()` similarly decrements/removes
its final registry lease before closing the retained lock handle and root descriptor.
BlobStore's outer resource ledger cannot recover state already discarded by those
owners.

Deterministic fault probes left an evidence-lock FD and, separately, a barrier root
FD live after the first typed close failure. A second `close()` reached `CLOSED`
without retrying either leaked descriptor. A retained advisory lock can block other
processes until garbage collection or process exit. The existing partial-close test
injects only `backend.close()` and misses both internal discard-before-close paths.

This violates D-12's ownership-aware, idempotent resource-release contract.

**Fix:** Retain per-resource state until each close has a known terminal outcome;
remove successfully closed handles individually and preserve failed/uncertain ones
for retry (or poison permanently when close ownership is unknowable). Do not remove
the final barrier registry lease until both retained resources are accounted for.
Add first/middle/final retained-handle and barrier-root close fault tests.

### WR-02: Pending-control recovery can be permanently starved by unrelated names

**Files:** `src/cacheness/storage/operation_repository.py:1075-1118`,
`tests/test_manifest_repository_cas.py:568-604`

**Issue:** `recover_pending_operation_records()` takes
`nsmallest(max_reconcile_actions, all_directory_names)` before checking exact pending
syntax. With a configured limit of one, a lexically earlier unrelated or malformed
entry consumes the only slot on every reopen, so a valid digest-bound pending record
is never inspected. No cursor advances past the blocker.

This violates D-14's resumable convergence and the Phase 3 bounded-inventory
decision that adjacent/non-operation records must not consume the recovery action
budget.

**Fix:** Filter exact eligible pending-control names before applying the action bound,
and/or persist a stable cursor so repeated bounded runs advance. Add repeated-reopen
tests with malformed, ordinary, clear-sidecar, and valid pending siblings.

### WR-03: Windows lifecycle-authority failures still escape the typed backend taxonomy

**Files:** `src/cacheness/storage/path_security.py:258-334`,
`src/cacheness/storage/path_security.py:1292-1316`,
`src/cacheness/storage/coordination.py:346-364`,
`tests/test_manifest_repository_cas.py:734-745`

**Issue:** `_WindowsRegistryAuthorityApi._raise_status()` raises raw `OSError`, and
mutex/registry creation, wait, query, set, release, and close failures are not
translated at the `ManagedFileOps` or barrier construction boundary. Access denial,
policy restriction, or missing capability can therefore leak a raw platform error
from `BlobStore(...)`. The new typed xattr taxonomy does not cover this Windows path,
despite the fix report's claim that the adapter has a stable capability error.

This violates D-18's typed unsupported-capability outcome and the repo error-
translation convention.

**Fix:** Translate registry/mutex capability and policy statuses to
`CacheBlobBackendError` with `BLOB_BACKEND_CAPABILITY_UNSUPPORTED`, translate other
native I/O to `BLOB_BACKEND_FAILURE`, and preserve the original cause and operation/
scope context. Treat release/close uncertainty explicitly. Add injected access-
denied, wait-failed, set/query, release, and close tests.

### WR-04: Delete cleanup failure can mask the required conflict or recoverable outcome

**Files:** `src/cacheness/storage/lifecycle.py:1410-1420`,
`src/cacheness/storage/lifecycle.py:1426-1448`,
`tests/test_blob_store_atomic_lifecycle.py:571-600`

**Issue:** A tombstone publication conflict calls `_retire(record)` directly in the
conflict handler and then reraises. A later tombstone-removal conflict also calls
`_retire(record)` from its conflict handler. If exact evidence retirement fails, the
backend/cleanup exception replaces the winner-preserving conflict (first branch) or
the post-authority partial-success result (second branch). The surviving durable
evidence is recoverable debt, but callers do not receive
`CacheBlobRecoverableCleanupError` with operation/generation/key context.

This violates D-08's rule that cleanup failure after evidence persistence is surfaced
as a typed recoverable outcome.

**Fix:** Wrap both conflict-retirement sites in the same explicit recoverable-cleanup
translation used by PUT; preserve the publication/removal conflict as cause or
structured context. Add deterministic retirement-failure tests at both sites and
assert the winner remains authoritative.

### WR-05: Successful reconciliation leaks completed action checkpoints permanently

**Files:** `src/cacheness/storage/reconciliation.py:424-447`,
`src/cacheness/storage/reconciliation.py:449-503`,
`src/cacheness/storage/operation_repository.py:420-520`,
`tests/test_blob_store_reconciliation.py:495-535`

**Issue:** Apply mode writes a prepared checkpoint, performs the destructive action,
advances the checkpoint to `completed`, and retires the primary operation record.
It never calls the implemented `retire_reconciliation_checkpoint_if_exact()`. The
checkpoint filename is deliberately excluded from operation inventory, so once the
primary record is gone no later recovery/reconciliation pass can discover or retire
the completed sidecar.

A deterministic safe-candidate apply left no candidate and no operation record but
left `operations/reconcile-action-<operation>.json` in completed state. Repeating
this flow creates one permanent authenticated sidecar per reconciled operation.

This violates D-11/D-14 cleanup convergence and the bounded-evidence/resource-
lifetime intent.

**Fix:** After both the action checkpoint and primary evidence reach a recoverable
terminal ordering, retire the exact completed checkpoint. Add reopen seams before
and after operation retirement; recovery must authenticate and retire an orphaned
completed checkpoint without repeating the destructive action. Add a steady-state
test asserting successful apply leaves no operation/action sidecars.

## Info

None.

## Verification

- `uv run --python 3.13 --group recommended pytest -q -o log_cli=false` — full
  suite passed with the documented optional/platform skips and the existing
  collection warning.
- Focused 11-module Phase 3 suite on CPython 3.13 — passed; only the Windows
  junction and unavailable device-node containment fixtures skipped.
- `uv run --python 3.11 --group recommended python -m compileall -q
  src/cacheness` and supported-environment `import cacheness` — passed on
  CPython 3.11.16 (`0.3.14`).
- `uv lock --check`, `git diff --check`, and relevant Phase 3 source/test Ruff
  scope — passed.
- Deterministic two-store overwrite probe — reproduced CR-01: an intervening
  committed winner was revoked by a stale writer.
- Deterministic crash-before-CAS/later-winner/reopen probe — reproduced CR-02:
  the signed candidate remained permanently `operation_authority_ambiguous`.
- Direct non-POSIX manifest-key provider probe — reproduced CR-03's unconditional
  platform rejection. Native Windows itself was not executed.
- Primary Win32 API contract review — confirmed CR-04's NULL-destination
  `MoveFileExW` form is documented only with `MOVEFILE_DELAY_UNTIL_REBOOT`, not
  the supplied `MOVEFILE_WRITE_THROUGH` immediate-delete claim.
- Retained-lock and admission-barrier close fault probes — reproduced WR-01:
  retry reached `CLOSED` with live descriptors not retried.
- Deterministic reconciliation apply probe — reproduced WR-05: candidate and
  primary evidence retired while the completed action checkpoint remained.
- Native Windows status: not executed. Existing Windows coverage is injected
  adapter coverage only. Native filesystem/ACL/session/crash verification remains
  a later owned gap, while CR-03/CR-04/WR-03 are current source contradictions.

---

_Reviewed: 2026-09-01T22:15:00Z_
_Reviewer: the agent (general code reviewer)_
_Iteration: 10_
