---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-02T01:26:50Z
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
  critical: 4
  warning: 0
  info: 0
  total: 4
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-02T01:26:50Z
**Depth:** deep
**Files Reviewed:** 35
**Status:** issues_found

## Summary

This is a fresh independent review of the complete Phase 3 implementation at
`d6bfb0b`, after the source fix `04395ab`. The live fix report was treated as an
unverified claim set. D-21 and D-22 are the canonical deployment boundaries: the
local store owner is trusted for lifecycle-control availability, Windows support
is one OS user/session, and native Windows execution remains a later evidence
gate. None of the findings below assumes a hostile store owner or unsupported
Windows topology.

The iteration-14 work does fix several previously reported defects. POSIX and the
native Win32 adapter make nonblocking kernel-lock attempts; the cross-process path
has a monotonic timeout/backoff and closes its contender handle; the public timeout
type is preserved by `BlobStore`; initialization guards are authority-scoped,
reference-counted, and retire at zero; unrelated stores initialize concurrently;
and the checked tests cover process loss and release before the kernel-lock
deadline. Sidecar cursors now stay in the full-filename namespace while accepting
old bare IDs, and v1/v2 tokens decode. Apply no longer double-charges an orphan
sidecar, and exact primary/sidecar lifecycle conflicts are isolated per finding.

Four blockers remain. The in-process guard is still a blocking `RLock` acquired
before the deadline exists, so a same-process contender can exceed the configured
timeout and later succeed. The new inventory cap is a permanent denial threshold,
not resumable paging: ordinary stores above 4,096 JSON/memory manifests and shared
operation directories above the cap cannot reconcile, recover, or clear. Lexical
cursors over reconstructed mutable inventories also skip entries inserted before
the cursor. Finally, apply computes and authenticates its continuation before
revalidation; when exact evidence changes and no action is attempted, the returned
token can be terminal while safe debt remains.

The complete focused Phase 3 corpus and full repository suite pass. Python 3.11.16
compile/import passes with the declared `recommended` extra. `uv lock --check`,
working-tree/commit whitespace checks, and changed-path Ruff were run; Ruff reports
the same two pre-existing `config.py` F841 findings present before `04395ab`.
Native Windows was not executed. Green suites do not cover the four deterministic
reproductions below.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: BLOCKER — The same-process initialization guard sits outside the configured deadline

**Files:** `src/cacheness/storage/integrity.py:31-61`,
`src/cacheness/storage/integrity.py:180-193`,
`src/cacheness/storage/integrity.py:278-340`,
`tests/test_blob_store_integrity.py:202-287`

**Issue:** `_KeyInitializationGuardRegistry.acquire()` increments the correct
per-authority refcount but then enters `with lock:` using an unbounded `RLock`.
`initialize_new_store()` acquires that guard before `_initialization_lock()` creates
the monotonic deadline. If one thread is stalled while acknowledging the first key,
a second thread for the same store never reaches the nonblocking POSIX/Win32 loop
and cannot raise `CacheBlobLifecycleTimeoutError` at the configured wall deadline.

A deterministic probe used a durability provider blocked on an `Event`, a 50 ms
timeout, and a second same-key provider. The contender was still alive after 200 ms
and, after the winner was released, returned success at 217 ms rather than timing
out. It did not read or retire winner bytes, and guard retirement/unrelated-store
concurrency remain correct, but the advertised authority deadline is false for a
supported same-process topology.

The existing timeout test holds the kernel authority in another process. The guard
test covers only two unrelated stores. Neither places two same-key threads on the
in-process guard while the first initializer is alive.

This violates the bounded retry/deadline contract selected under D-05 and the
deterministic failure requirement of STOR-04/SECU-04.

**Fix:** establish one absolute deadline before acquiring any same-key admission
primitive. Acquire the per-authority guard with bounded/nonblocking retries against
the remaining time (or remove it and rely on the already-correct kernel authority),
then pass the same absolute deadline into the POSIX/Win32 loop so the layers cannot
each receive a fresh budget. Preserve refcount retirement on timeout and add
same-process live-winner timeout, release-before-deadline, exception, descriptor,
and registry-zero tests.

### CR-02: BLOCKER — `max_inventory_items` permanently denies valid large stores and conflates unrelated control families

**Files:** `src/cacheness/config.py:353-370`,
`src/cacheness/storage/path_security.py:885-945`,
`src/cacheness/storage/manifest_repository.py:565-607`,
`src/cacheness/storage/operation_repository.py:575-619`,
`src/cacheness/storage/operation_repository.py:1280-1315`,
`src/cacheness/storage/operation_repository.py:1523-1578`,
`src/cacheness/storage/lifecycle.py:414-434`,
`src/cacheness/storage/lifecycle.py:1163-1180`

**Issue:** the new limit bounds one inventory construction, but it is a hard
namespace-size ceiling rather than a continuation bound. JSON and memory
`ManifestRepository.list_page()` first materialize every string key and raise when
the store has more than `max_inventory_items` (default 4,096). No cursor can move
past that exception. The same rule counts every entry in the shared `operations/`
directory before filtering, so four unrelated names can block primary operations,
pending controls, and reconciliation sidecars when the injected limit is three.
Each recovery/reconcile call repeats the same denial forever.

Deterministic probes created four valid memory-backed manifests under a limit of
three: both `reconcile()` and `clear()` raised `CacheBlobBackendError` without a
report or continuation. An otherwise empty store with four unrelated files in
`operations/` also made `reconcile()` fail. The manifest adapter wraps the original
error and drops `max_inventory_items` from the public context, leaving only the
generic backend-failure reason. At the default, an ordinary 4,097-entry local store
therefore loses the Phase 3 clear/recovery contract. Raising the configurable limit
only moves the permanent threshold and recreates whole-namespace work.

The implementation separately bounds record bytes and apply attempts, which is
good, but D-05/D-10/D-13-D-16 and STOR-04/STOR-06 require bounded resumable work,
not permanent refusal of a valid large store. The checked reverse-order test uses a
namespace below the cap and cannot distinguish the two contracts.

**Fix:** replace the ceiling with a backend-native ordered inventory/index or a
durable authenticated chunked snapshot. Page JSON/memory manifests without
re-materializing the entire store, and separate primary, pending, sidecar, and
clear-control namespaces (or their durable indices) so one family cannot deny the
others. Keep distinct per-call counters for names/keys inspected, raw bytes read,
and actions attempted. If an inventory itself is unavailable, return a stable typed
blocked finding and continuation rather than aborting the entire report. Add stores
well above 4,096 entries and 100x unrelated namespaces with terminal no-skip,
no-duplicate, bounded-call assertions across reconcile, recover, and clear.

### CR-03: BLOCKER — Lexical cursors over mutable reconstructed inventories skip insertions

**Files:** `src/cacheness/storage/path_security.py:895-945`,
`src/cacheness/storage/manifest_repository.py:582-607`,
`src/cacheness/storage/operation_repository.py:582-619`,
`src/cacheness/storage/operation_repository.py:1287-1315`,
`src/cacheness/storage/operation_repository.py:1544-1578`,
`src/cacheness/storage/reconciliation.py:958-998`

**Issue:** sorting each bounded inventory fixes the earlier filesystem-order/lexical
cursor contradiction for a static namespace, and the sidecar filename namespace is
now consistent. It does not create a stable inventory across calls. Every page
rebuilds membership and resumes with `name > cursor`. A record inserted before the
cursor after page one is invisible for the rest of that resume chain. Pending
recovery persists the same kind of lexical name cursor, so restart does not repair
the omission until that scheduling cursor is explicitly wrapped or removed.

A deterministic directory probe started with `a,b,c`, read page `(a,b)` with cursor
`b`, inserted `aa`, and resumed. Page two returned only `c`, produced a terminal
cursor, and never reported `aa`. Delete/reinsert and concurrent operation/sidecar
publication have the same missing-membership problem. Static reverse-creation order
passes because sorting a single unchanged membership is not the disputed case.

Clear protects its manifest snapshot with aggregate admission, but reconciliation
scans before acquiring aggregate apply admission, and pending recovery/operation
inventory can change across calls and reopen. This contradicts the stable cursor and
resumable no-skip contract in D-10/D-13-D-16 and STOR-06.

**Fix:** bind continuation to an authenticated inventory generation/snapshot whose
membership cannot change under the cursor, or use backend-assigned monotonic
sequence positions plus an explicit high-water mark. Define insertion/deletion
semantics and wrap only after the current high-water mark completes. Test insert
before/after, delete current/next, delete-and-reinsert, restart, terminal wrap, and
old bare-ID/v1/v2 tokens across manifest, primary operation, pending, sidecar, JSON,
memory, and SQLite sources with an exact once-per-snapshot invariant.

### CR-04: BLOCKER — Apply can return a terminal cursor for exact evidence it scanned but did not apply

**Files:** `src/cacheness/storage/reconciliation.py:396-449`,
`src/cacheness/storage/reconciliation.py:451-512`,
`src/cacheness/storage/reconciliation.py:514-558`,
`src/cacheness/storage/reconciliation.py:560-596`,
`tests/test_blob_store_reconciliation.py:1212-1249`

**Issue:** orphan and primary work now share one normalized stream, and a true
attempt consumes one slot once. However, the resume token is computed from scanned
findings before `_apply_findings()` runs. `_apply_finding()` and orphan retirement
return `False` when exact bytes changed or disappeared during revalidation. Those
unattempted findings consume no apply budget, but the precomputed source cursor has
already advanced past them. An exact conflict is reported, yet the same precomputed
token also advances past the current record.

A deterministic probe scanned one safe uncommitted candidate, exact-CAS advanced
its primary record to another valid checkpoint immediately before apply, and then
ran the real apply loop. The report still contained the original SAFE
`delete_candidate` finding, returned `resume_token=None`, and left both the updated
primary and candidate intact. The updated record remained independently safe and
actionable. Re-running from scratch happens to rediscover it, but a caller following
the authenticated continuation contract is told the run is complete.

The orphan-plus-primary budget-two regression proves double charging is fixed; it
does not race exact evidence between scan and apply or assert that every unapplied
action remains in the returned continuation. This violates D-14 and STOR-06's
resumable apply guarantee.

**Fix:** make apply return structured per-finding outcomes (`attempted`, `completed`,
`conflicted`, `stale/unapplied`) and derive continuation only afterward. Retain the
earliest source position containing stale or conflicted current work, or persist an
authenticated apply work queue/checkpoint before advancing scan cursors. Keep one
budget charge per actual repository/destructive attempt and preserve current bytes.
Add exact primary update/retire and sidecar update/retire races under budgets one and
two, later-action progress, repeated apply, reopen, and every fault seam; assert
effects, attempts, remaining artifacts, dispositions, and nonterminal continuation.

## Warnings

None.

## Info

None.

## Verification

- Preserved the exact prior live report as
  `.planning/phases/03-atomic-lifecycle-and-recovery-engine/03-REVIEW.iter13.md`;
  both files had SHA-256
  `8101756aa479a4592d4a04687661b85243d6bc1a42045aa4d10dede9e01c99ca`
  before the live report was replaced.
- Complete focused Phase 3/compatibility corpus passed with only the documented
  Windows-junction and unavailable-device-node skips.
- Full repository pytest passed with documented PostgreSQL, TensorFlow, and
  platform skips plus the existing collection warning.
- CPython 3.11.16 `compileall` and import passed with the declared `recommended`
  extra (`0.3.14`). The known base-install NumPy concern is intentionally not
  duplicated here.
- `uv lock --check`, `git diff --check`, and `git diff --check 04395ab^..HEAD`
  passed.
- Changed-path Ruff reports only the same two `src/cacheness/config.py` F841
  findings present in `04395ab^`; no new Ruff finding was introduced.
- Same-key same-process initialization remained blocked 200 ms under a configured
  50 ms timeout and then succeeded at 217 ms after release.
- Four manifests under `max_inventory_items=3` made both reconcile and clear fail;
  four unrelated `operations/` names made an empty store's reconcile fail.
- A page `(a,b)` followed by insertion of `aa` and resume from `b` returned only
  `c` and a terminal cursor, proving the insertion skip.
- An exact safe-primary transition between scan and apply returned a terminal token
  while preserving an independently safe primary and candidate.
- Source inspection and focused tests support the fixed claims for nonblocking
  POSIX/native-Win32 attempts, handle cleanup, cross-process timeout/release/loss,
  public timeout propagation, per-authority guard retirement, old v1/v2/bare-ID
  token decoding, full-filename sidecar cursors, orphan/primary one-charge behavior,
  exact conflict isolation, current-record preservation, later-action progress,
  native handler bytes, normal-read non-mutation, containment, and Python 3.11.
- Native Windows execution was unavailable; no Windows-only behavioral claim is
  upgraded beyond source-contract review.

---

_Reviewed: 2026-09-02T01:26:50Z_
_Reviewer: Codex (general code reviewer)_
_Iteration: 15_
