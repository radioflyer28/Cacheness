---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-01T19:59:07Z
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
  warning: 2
  info: 0
  total: 5
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-01T19:59:07Z
**Depth:** deep
**Files Reviewed:** 32
**Status:** issues_found

## Summary

This review accepts the canonical D-21 and D-22 product boundaries. The local
store owner is trusted not to deliberately destroy or rebind every live authority,
and Windows support is one OS user in one interactive/service session. Native
Windows execution remains a later evidence gate. The findings below are source or
deterministic-contract contradictions inside that supported topology; none asks
for hostile-owner or cross-session protection.

Three iteration-11 claims hold. Every active canonical lifecycle payload cleanup
now reaches `delete_durable()`, and the Win32 deletion adapter disposes and closes
one retained, reparse-checked, single-link handle rather than reopening a verified
pathname. Exact canonical raw manifests now fail closed through ordinary get, put,
delete, metadata update, exists, list, clear, and reconciliation. Reopen tombstone
removal conflicts also retain the later winner and surface explicit post-authority
retirement context.

Three blockers remain. First-key publication has no durable initialization state:
a concurrent opener can consume the file before acknowledgement, and any failed
acknowledgement or close is bypassed by the next ordinary read. Second, the
pending-control starvation fix makes inventory unbounded and still permanently
starves later valid work on the supported Windows fallback when an identical final
already exists. Third, a malformed sidecar matching valid primary evidence causes
startup to defer that evidence and every apply to abort; it is invisible in dry-run
and can starve all later actions.

Two warnings cover the new tombstone state machine's incomplete interruption
oracle and empty-store apply behavior. The `prepared` post-delete seam replays the
payload deletion call on reopen despite the implementation's no-replay claim, and
`reconcile(apply=True)` on a pristine empty store fails merely because it eagerly
loads a signing key for an empty sidecar inventory.

The complete focused and repository suites pass. Python 3.11.16 compile/import
passes with the declared `recommended` extra; the known base-install NumPy issue is
not duplicated here. Lock, diff, and changed-path Ruff gates pass. Deterministic
probes reproduced all five findings. Green tests omit the concurrent/failing
trust-root acknowledgement, identical-final Windows pending recovery, matching
malformed sidecar, per-stage destructive-call count, and pristine apply cases.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: BLOCKER — First-key initialization can publish signed authority before durability acknowledgement

**Files:** `src/cacheness/storage/integrity.py:121-180`,
`src/cacheness/storage/integrity.py:182-271`,
`src/cacheness/storage/path_security.py:234-257`,
`src/cacheness/storage/blob_store.py:988-1021`,
`tests/test_blob_store_integrity.py:73-91`,
`tests/test_blob_store_integrity.py:113-143`

**Issue:** `initialize_new_store()` publishes the key pathname at `O_EXCL` open,
then writes/fsyncs/closes and only afterwards calls the durability provider. A
losing concurrent initializer handles `FileExistsError` by immediately reading and
returning that visible file. It neither waits for nor proves the winner's
acknowledgement. A deterministic two-thread probe blocked the winner inside
`acknowledge_new_key()` and observed the loser return all 32 key bytes while the
winner remained blocked. That loser can sign lifecycle evidence and a manifest
before the trust root's required parent-entry/platform acknowledgement completes.

Failure recovery has the same ordering hole. If key-file close or durability
acknowledgement raises, the file remains. `get_or_initialize_new_store()` then
returns it through `_read_existing_key()` and never retries acknowledgement. A
probe injected one `OSError` acknowledgement failure: the first call raised typed
`ManifestKeyError`, the key remained, and the second call returned the key with the
provider call count still one. A crash at the same boundary has the same reopen
shape. The existing race test writes a winner file and raises `FileExistsError`
without any acknowledgement, then asserts it is immediately trusted; its oracle
therefore encodes the bug.

The advertised Windows acknowledgement is also not the claimed exact-handle
contract. `_flush_windows_key_entry()` checks pathname identity before and after,
but `_WindowsFileApi.flush_regular_file()` performs no handle information, reparse,
link-count, or expected-identity validation on the handle it actually flushes.
This is a check/open/check substitution gap for the sole trust root. In addition,
`BlobStore._manifest_key()` catches only three exception classes. A deterministic
injected provider raising a custom keystore exception leaked that exception
unchanged, and native `CacheBlobBackendError` acknowledgement failures likewise do
not become the report's claimed manifest-authentication failure with provider and
operation context.

This violates D-04, D-07, D-11, D-18, SECU-04, and the D-22 supported topology.

**Fix:** Serialize first-key initialization through one durable initialization
protocol whose completed state is independently recognizable. Do not let an
`EEXIST` loser or reopen consume the key until the winner's exact identity has an
acknowledged completion record; safely resume acknowledgement or retire only the
exact unacknowledged inode after failure. Bind the Windows flush to one retained
handle and verify its native identity/reparse/link contract against the expected
key identity. Validate provider return values and translate every ordinary
`Exception` from file/injected providers to
`CacheBlobManifestUnauthenticatedError` with cause and provider/operation context.
Add concurrent init, crash/reopen, failed-close, failed-ack, unsafe-object, and
custom-provider exception tests.

### CR-02: BLOCKER — Pending recovery is unbounded and still permanently starves valid Windows work

**Files:** `src/cacheness/storage/operation_repository.py:1125-1202`,
`src/cacheness/storage/path_security.py:1749-1788`,
`tests/test_blob_store_reconciliation.py:1001-1027`,
`tests/test_manifest_repository_cas.py:45-70`

**Issue:** the fix moved `max_reconcile_actions` after digest validation, but it
first materializes every syntax-valid name with `sorted(...)` and reads every
lexically preceding digest-invalid file. With a configured action limit of one, a
probe containing 20 digest-invalid candidates followed by one valid candidate
performed 21 bounded file reads before returning the one valid result. Thus valid
work progresses only by abandoning the bounded inventory/I/O contract; sufficiently
large crash residue can exhaust memory or monopolize startup.

The supported Windows fallback still has true permanent starvation. When a
digest-valid pending control has an identical final, the fallback deliberately
leaves the pending pathname and returns `False`. The repository nevertheless
counts it as the sole eligible action. With that candidate lexically first and a
later promotable valid candidate, two consecutive simulated-Windows recovery calls
both returned no work, retained the first pending file, and never created the
later final. The new digest-invalid regression cannot expose this because it runs
the POSIX descriptor path and has no identical-final eligible candidate.

Neither digest-invalid nor identical-final pending evidence is included in the
reconciliation report despite the repository docstring promising it remains for
reporting. The bytes are untouched, but operators receive no bounded finding or
resume cursor with which to advance past them.

This violates D-05, D-11, D-14, D-15, STOR-04, STOR-06, and D-22.

**Fix:** Replace whole-directory sorting with a stable bounded cursor/page for
pending controls. Bound names inspected, bytes read, and actions independently so
repeated calls advance without rescanning one invalid prefix. On Windows, either
retire an identical pending object through a retained handle that validates and
disposes the same native identity, or advance a durable cursor without charging it
forever. Surface untouched invalid/blocked candidates in dry-run reports. Add
large-prefix call-count tests and repeated simulated/native Windows identical-final
plus later-valid recovery tests.

### CR-03: BLOCKER — A malformed matching sidecar becomes unauthenticated recovery authority and aborts every apply

**Files:** `src/cacheness/storage/lifecycle.py:1038-1056`,
`src/cacheness/storage/operation_repository.py:448-490`,
`src/cacheness/storage/reconciliation.py:378-440`,
`src/cacheness/storage/reconciliation.py:522-563`,
`src/cacheness/storage/reconciliation.py:674-706`,
`tests/test_blob_store_reconciliation.py:1030-1070`

**Issue:** startup `_recover_tombstone()` treats mere sidecar pathname existence
as sufficient reason to defer valid primary evidence. It does not authenticate or
bind the sidecar before allowing it to change recovery ordering, contradicting its
own comment that invalid sidecars are inert.

Apply logs a malformed sidecar as blocked during orphan cleanup, but then processes
the matching safe primary finding. `_prepare_action_checkpoint()` parses the same
malformed bytes and raises `CacheBlobReconciliationCheckpointError`; the per-finding
loop does not isolate that failure. A deterministic tombstone probe wrote `b"{"`
under its matching sidecar name. Two consecutive apply calls each raised the same
typed error and retained both the valid tombstone primary evidence and malformed
sidecar. Any later findings in those calls were never reached. The regression only
uses a malformed sidecar whose ID has no primary evidence, so it proves orphan
retirement can pass it, not that normal safe work can.

Sidecars are inspected only inside `apply=True`, so default dry-run does not report
the malformed evidence at all. `list_reconciliation_checkpoint_raws()` also
materializes and reads every syntactically valid sidecar before applying any bound;
a limit-one probe with 20 malformed sidecars returned/read all 20. Completed-orphan
retirement has a separate action counter from primary findings, so one apply can
also perform up to the configured limit in each category rather than in total.

This violates D-13 through D-16, STOR-04, and STOR-06.

**Fix:** Inventory sidecars through their own stable bounded cursor and include
blocked sidecar findings in dry-run. Authenticate and bind a sidecar before startup
defers primary recovery. Isolate malformed/untrusted sidecars per finding so they
remain untouched without aborting later eligible work, and use one shared total
action budget across sidecar retirement and primary actions. Add matching-primary,
later-valid, dry-run visibility, repeated apply/reopen, and bounded-call tests.

## Warnings

### WR-01: The tombstone checkpoint state machine replays payload deletion after its documented post-delete seam

**Files:** `src/cacheness/storage/reconciliation.py:589-614`,
`tests/test_blob_store_reconciliation.py:1073-1118`

**Issue:** in `prepared`, `_apply_complete_tombstone()` deletes the payload, invokes
`reconcile_tombstone_after_payload_delete`, and only then persists
`payload_deleted`. A `BaseException` at the named post-delete seam therefore leaves
the sidecar in `prepared`. Reopen calls `_delete_or_prove_absent()` for the same
payload again. A deterministic probe recorded one payload `delete_durable()` call
before interruption and another for the same locator after reopen. The second call
is idempotent because the payload is absent, so winner safety is preserved, but the
implementation comment and fix report explicitly claim the deletion is not
replayed. The regression checks final convergence but no destructive-call count and
only one of the new state machine's before/after seams.

**Fix:** use the candidate-delete pattern: catch `BaseException`, prove the payload
is absent, durably advance to `payload_deleted`, then re-raise. Apply equivalent
post-effect acknowledgement at the manifest-removal seam. Parameterize every
before/inside/after payload, tombstone, checkpoint, primary-retirement, and sidecar-
retirement seam; reopen/apply repeatedly and assert exact call counts and evidence.

### WR-02: Reconciliation apply on a pristine empty store fails on an absent key

**Files:** `src/cacheness/storage/reconciliation.py:378-397`,
`src/cacheness/storage/reconciliation.py:522-526`

**Issue:** `_retire_orphaned_completed_checkpoints()` loads the manifest signing
key before it knows whether any sidecars exist. A pristine `BlobStore` deliberately
has no key until its first canonical write. A probe observed dry-run return an empty
report, while `reconcile(apply=True)` on the same untouched store raised
`CacheBlobManifestUnauthenticatedError`. Repeated apply is therefore not an
idempotent no-op for the simplest consistent store, and the apply path creates a
different failure surface despite having zero proposed actions.

**Fix:** obtain a bounded sidecar page first and return immediately when both the
report and sidecar inventory contain no eligible work. Load the key only when bytes
must actually be authenticated; do not initialize a trust root merely to apply an
empty report. Add pristine JSON, memory, and SQLite dry/apply/repeated-close tests.

## Info

None.

## Verification

- Complete focused Phase 3 11-module suite passed; only the documented Windows
  junction and unavailable device-node fixtures skipped.
- Full repository suite passed with 32 documented platform/optional skips and the
  existing collection warning.
- CPython 3.11.16 `compileall` and `import cacheness` passed with the declared
  `recommended` extra (`0.3.14`). The known NumPy base-install concern is omitted.
- `uv lock --check`, `git diff --check`, and focused 921324d changed-path Ruff passed.
- Production overwrite, delete, clear, and reconciliation candidate probes reached
  `delete_durable()`; Win32 disposition/close adapter tests remain green. Native
  Windows itself was not executed.
- A blocked durability-provider probe proved a concurrent key reader returns before
  acknowledgement; a failed-ack probe proved retry bypasses acknowledgement.
- A custom injected keystore exception escaped the manifest error boundary.
- Limit-one pending recovery performed 21 reads for 20 invalid plus one valid
  candidate. Simulated Windows identical-final recovery starved a later valid
  candidate on consecutive calls.
- Limit-one sidecar inventory read all 20 malformed sidecars. A malformed sidecar
  matching authenticated tombstone evidence aborted two consecutive apply calls.
- Tombstone post-delete interruption invoked durable deletion for the same payload
  once before and once after reopen. Empty-store dry-run succeeded while apply failed.
- Exact raw manifest, native handler/no-wrapper, containment, normal-read
  non-mutation, exact CAS, clear/admission, close ownership, and conflict-provenance
  suites remain green.

---

_Reviewed: 2026-09-01T19:59:07Z_
_Reviewer: the agent (general code reviewer)_
_Iteration: 12_
