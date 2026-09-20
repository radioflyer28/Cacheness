---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-01T19:17:29Z
depth: deep
files_reviewed: 32
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
findings:
  critical: 3
  warning: 4
  info: 0
  total: 7
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-01T19:17:29Z
**Depth:** deep
**Files Reviewed:** 32
**Status:** issues_found

## Summary

The canonical D-21 and D-22 boundaries are accepted: the local store owner is
trusted not to deliberately destroy or rebind all live authority, and Windows is
limited to one OS user in one interactive/service session. No finding below asks
for hostile-owner or cross-principal/session protection. Native Windows execution
remains a later evidence gate; source contradictions in the supported topology are
still blockers.

The exact authenticated raw-record fix is effective for put, delete, clear-target
delete, tombstone recovery, and metadata CAS. The superseded-candidate rule now
requires both a different generation and locator and converges create/replace
residue. Retained operation-lock/barrier resources remain reachable across close
failure. Windows registry/mutex failures are typed, including release/close
uncertainty. Direct delete conflict retirement now preserves conflict versus
post-authority state. Those iteration-10 claims were independently confirmed.

Three blockers remain. First, the new Win32 handle-disposition implementation is
not called by lifecycle payload cleanup: the engine still reaches the generic
pathname `delete()`. Second, file-backed signing-key creation acknowledges only
the file, not the directory entry containing the sole trust root, and provider or
descriptor-close failures can escape the typed manifest boundary. Third,
`COMPLETE_TOMBSTONE` reconciliation retires the primary operation evidence before
marking its action checkpoint complete; an interruption at that seam leaves a
prepared sidecar that no inventory can resume or retire.

Four warnings cover bounded recovery and fail-closed consistency. A syntax-valid
but digest-invalid pending control can still consume the first action slot forever.
Normal BlobStore operations accept noncanonical raw manifest JSON that
reconciliation calls untrusted. Malformed reconciliation sidecars can block every
apply and starve later completed orphans. Reopen tombstone recovery still loses the
authoritative removal conflict when exact evidence retirement fails.

The clean focused and full suites pass, along with Python 3.11 compile/import,
lock, diff, and focused Ruff. Deterministic probes reproduce CR-01, CR-03, WR-01,
WR-02, and WR-03. Green tests do not exercise the missing call edge, crash-durable
key-directory acknowledgement, or the uncovered interruption order.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: BLOCKER — Lifecycle payload cleanup bypasses the Win32 handle-delete protocol

**Files:** `src/cacheness/storage/blob_store.py:1015-1028`,
`src/cacheness/storage/path_security.py:177-221`,
`src/cacheness/storage/path_security.py:1843-1857`,
`src/cacheness/storage/path_security.py:1926-1938`,
`src/cacheness/storage/lifecycle.py:1066-1079`,
`src/cacheness/storage/lifecycle.py:1342-1350`,
`src/cacheness/storage/lifecycle.py:1474-1481`

**Issue:** `_WindowsFileApi.delete_write_through()` now uses `CreateFileW` plus
`SetFileInformationByHandle(FileDispositionInfo)`, but it is reachable only from
`ManagedFileOps.delete_durable()`. Every lifecycle candidate, superseded payload,
and tombstone cleanup calls `BlobStore._delete_or_prove_absent()`, which calls
`file_ops.delete()`. In Windows fallback mode that method executes `Path.unlink()`
and never invokes the native adapter.

A deterministic simulated-Windows probe constructed descriptor-less
`ManagedFileOps`, injected a recording `_windows_file_api`, and called
`file_ops.delete(payload)`. It returned `True` with `native_calls == []`. Thus the
iteration-10 adapter test proves the isolated adapter, not the production lifecycle
call chain. Payload reclamation, overwrite cleanup, clear, and reconciliation still
use the pathname behavior CR-04 was intended to replace.

Even `delete_durable()` records a pathname identity before opening but does not
pass it to the native adapter; `_ByHandleFileInformation` exposes volume/file index
fields, yet `delete_write_through()` checks only reparse and link count. The
path-to-handle substitution check claimed by the fix is therefore incomplete.

This violates D-09, D-11, D-18, D-19, and the supported D-22 topology.

**Fix:** Route `_delete_or_prove_absent()` through `delete_durable()` and make the
Win32 adapter compare the opened handle's `(VolumeSerialNumber, FileIndexHigh,
FileIndexLow)` with the exact pre-open identity, or eliminate the pathname precheck
and retain one handle-based identity throughout. Preserve truthful disposition/
close acknowledgement and typed uncertainty. Add production-call-chain tests plus
native Windows put/overwrite/delete/clear/reconcile crash/reopen tests.

### CR-02: BLOCKER — Signing-key initialization can durably publish manifests without a durable trust root

**Files:** `src/cacheness/storage/integrity.py:95-142`,
`src/cacheness/storage/integrity.py:168-203`,
`src/cacheness/storage/blob_store.py:988-1013`

**Issue:** `ManifestKeyProvider.initialize_new_store()` exclusively creates the
key, writes it, and `fsync()`s the file, but never durably acknowledges the parent
directory entry before the lifecycle publishes durable operation evidence and a
manifest. A crash can therefore leave durable signed control records while losing
the only key pathname needed to authenticate them. Reopen then correctly fails
closed, but the last valid generation is permanently unavailable because the
transaction ordering never made the trust root durable first.

The portable-provider path also has incomplete typed failure behavior. Both
`finally: os.close(descriptor)` sites may expose raw `OSError`, and an injected
provider raising an operational error is not translated: a direct probe with
`get_key()` raising `OSError("kms unavailable")` leaked that raw exception from
`BlobStore.put()`. The fix's source-inspection test only proves removal of the
explicit non-POSIX rejection; it does not exercise atomic first-open, crash
durability, reopen, ACL deployment expectations, or provider failures on Windows.

The reparse/regular/single-link and POSIX owner/mode checks are present, and the
documented same-user/session ACL responsibility is consistent with D-22. The
blocker is the missing durable trust-root ordering and typed operational boundary,
not a request for cross-principal key protection.

This violates D-04, D-07, D-11, D-18, and the project's fail-closed-but-recoverable
integrity contract.

**Fix:** Introduce a platform key-provider durability primitive. On POSIX, close
the fully synced file and fsync an identity-checked parent directory before any
signed lifecycle record may publish. On Windows, use a documented handle-based
write-through/flush protocol and document exactly what is acknowledged under the
deployment ACL. Translate file-provider close failures and injected-provider
operational failures to `CacheBlobManifestUnauthenticatedError` with preserved
cause. Test concurrent initialization, interruption after every persistence step,
reopen, reparse/single-link/permission failures, and injected-provider ownership.

### CR-03: BLOCKER — Tombstone reconciliation can lose its only resume evidence before checkpoint completion

**Files:** `src/cacheness/storage/reconciliation.py:388-433`,
`src/cacheness/storage/reconciliation.py:470-528`,
`src/cacheness/storage/operation_repository.py:433-570`

**Issue:** for `COMPLETE_TOMBSTONE`, `_apply_finding()` first calls
`LifecycleEngine._recover_tombstone()`. That call reclaims the payload, removes the
tombstone, and retires the primary operation record. Only after it returns does
the reconciler advance the action checkpoint from `prepared` to `completed`. If
checkpoint completion is interrupted, the destructive work and primary-evidence
retirement have already succeeded, but the remaining prepared sidecar is excluded
from normal operation inventory. `_retire_orphaned_completed_checkpoints()` ignores
it because it retires only `completed` checkpoints.

A deterministic probe created post-authority tombstone debt, applied reconciliation,
and injected failure in `_complete_action_checkpoint()` after
`_recover_tombstone()` returned. The manifest and primary operation record were
absent, while `reconcile-action-<id>.json` remained authenticated in `prepared`
state. A second apply reported zero findings and left it forever. The iteration-10
regression covers candidate deletion and a completed orphan only.

This violates D-11, D-14, STOR-05, and STOR-06.

**Fix:** Make tombstone completion checkpoint-aware as one resumable state machine.
Either checkpoint each destructive substep before primary retirement, or retain
primary evidence until the action checkpoint is completed and make reopen consume
prepared checkpoints safely. Add BaseException seams before/after payload delete,
tombstone remove, primary retire, checkpoint completion, and checkpoint retire;
repeated apply must neither replay a completed action nor strand the only sidecar.

## Warnings

### WR-01: Digest-invalid pending controls still starve the bounded recovery budget

**Files:** `src/cacheness/storage/operation_repository.py:1123-1195`,
`src/cacheness/storage/operation_repository.py:1197-1221`

**Issue:** the iteration-10 fix filters exact filename grammar before
`nsmallest(max_reconcile_actions, ...)`, but content eligibility is checked after
the bound. With a limit of one, a lexically first syntax-valid pending name whose
bytes do not match its encoded digest consumes the sole slot on every call. A later
valid pending record is never inspected.

A deterministic probe called recovery twice with one bad-digest `0...` candidate
and one valid `f...` candidate. Both returned `()`, and the valid final remained
absent. The existing test covers malformed names, not syntax-valid ineligible
content.

**Fix:** Define the action budget over candidates that pass bounded read and digest
validation, or persist a stable cursor so repeated runs advance. Preserve invalid
bytes untouched and report them separately.

### WR-02: Normal operations accept noncanonical raw manifests that reconciliation rejects

**Files:** `src/cacheness/storage/manifest.py:188-228`,
`src/cacheness/storage/blob_store.py:1110-1189`,
`src/cacheness/storage/reconciliation.py:827-850`

**Issue:** `_load_authenticated_manifest_with_raw()` authenticates the canonical
semantic projection but never requires `manifest.canonical_bytes() == raw_manifest`.
Reconciliation does. Reformatting a valid signed manifest with whitespace/order
changes therefore leaves the signature valid; ordinary `get()` accepts it, while
dry-run reports `manifest_untrusted`.

A deterministic probe pretty-printed the signed JSON and observed
`get("k") == {"x": 1}` followed by reconciliation reason `manifest_untrusted`.
Put/delete/metadata CAS can also adopt those noncanonical bytes as expectations
instead of failing closed on the observable control substitution.

**Fix:** Require exact canonical raw bytes at the common authenticated loader before
locator/handler access. Add get/put/delete/update/exists/list and reconciliation
consistency tests for reordered and whitespace-padded records.

### WR-03: One malformed reconciliation sidecar can block every apply and starve completed orphans

**Files:** `src/cacheness/storage/operation_repository.py:455-488`,
`src/cacheness/storage/reconciliation.py:368-386`,
`src/cacheness/storage/reconciliation.py:513-528`

**Issue:** checkpoint inventory applies `max_reconcile_actions` after filename
syntax only. `_retire_orphaned_completed_checkpoints()` then parses selected records
without isolating malformed/untrusted sidecars. A malformed lexically early
`reconcile-action-<32hex>.json` raises
`CacheBlobReconciliationCheckpointError` on every apply and prevents later
authenticated completed orphans from retirement. It is not a blocked report
finding because these sidecars are excluded from normal evidence reports.

A probe persisted `b"{"` under a valid checkpoint name. Every apply failed typed
and preserved it, but made no progress. D-15 requires malformed evidence to be
reported untouched, not to become permanent global apply denial.

**Fix:** Parse/authenticate sidecars independently under a stable cursor; report
invalid entries as blocked without consuming all future progress. Apply the action
bound to eligible work or advance a durable cursor.

### WR-04: Reopen tombstone retirement failure still loses the authoritative conflict state

**Files:** `src/cacheness/storage/lifecycle.py:1035-1079`,
`src/cacheness/storage/lifecycle.py:1116-1157`

**Issue:** direct delete conflict branches use `_retire_after_conflict()`, but
reopen recovery does not. `_recover_tombstone()` catches a conditional removal
conflict and calls `_retire(record)`. If exact evidence retirement fails, the
retirement exception escapes and `recover()` wraps it as generic cleanup debt. The
removal conflict is not preserved as explicit cause/context, so callers cannot
distinguish a later authoritative winner from ordinary evidence cleanup failure.

**Fix:** Route recovery through a conflict-aware retirement helper recording that
tombstone authority was reclaimed, the later winner survived, and only evidence
retirement remains. Add a recovery removal-conflict plus retirement-failure test.

## Info

None.

## Verification

- Clean focused Phase 3 11-module suite passed; only the documented Windows
  junction and unavailable device-node fixtures skipped.
- `uv run pytest -q -o log_cli=false` passed cleanly with documented optional/
  platform skips and the existing collection warning.
- CPython 3.11.16 `compileall` and `import cacheness` passed (`0.3.14`).
- `uv lock --check`, `git diff --check`, and focused Phase 3 Ruff passed.
- Exact authenticated snapshot interleavings for overwrite/delete/metadata are
  covered by deterministic regressions; the iteration-10 raw-record fix is confirmed.
- Create/replace later-winner tests and source inspection confirm generation-and-
  locator-distinct convergence; same-locator remains blocked.
- Retained operation-lock/barrier first/middle/final retry tests pass.
- Windows registry/mutex tests confirm typed create/wait/query/set/release/close
  outcomes. Native Windows itself was not executed.
- Simulated Windows production-call probe reproduced CR-01: deletion returned
  success without calling the native disposition adapter.
- Tombstone checkpoint interruption reproduced CR-03: primary evidence and manifest
  retired, prepared action sidecar permanently unreachable.
- Bad-digest pending-control, noncanonical-manifest, and malformed-sidecar probes
  reproduced WR-01 through WR-03.
- Native handler/no-wrapper, containment, committed-only reads, no read-side evidence
  mutation, clear/admission, and close ownership suites remain green.

---

_Reviewed: 2026-09-01T19:17:29Z_
_Reviewer: the agent (general code reviewer)_
_Iteration: 11_
