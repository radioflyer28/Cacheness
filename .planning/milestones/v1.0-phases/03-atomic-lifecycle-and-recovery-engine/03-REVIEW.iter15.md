---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-02T04:02:37Z
depth: deep
head: 8a502a0
files_reviewed: 35
files_reviewed_list:
  - docs/SECURITY.md
  - docs/WINDOWS_COMPATIBILITY.md
  - pyproject.toml
  - uv.lock
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
  - tests/test_phase1_quality_gates.py
  - tests/test_public_api_contract.py
findings:
  critical: 5
  warning: 0
  info: 0
  total: 5
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-02T04:02:37Z
**Depth:** deep
**HEAD:** `8a502a0`
**Status:** issues_found

## Summary

This is a fresh review of the complete Phase 3 implementation, including the
high-water inventory and clear-page rewrite in `afde03f`, `aafa79b`, and
`8a502a0`. The live fix report and green suites were treated as claims, not proof.
D-21 and D-22 were respected: none of the findings assumes a hostile local store
owner or an unsupported Windows cross-user/session topology.

The rewrite does preserve high-water and next-sequence fields in signed v2 clear
pages, keeps v1 pages readable, binds v2 page IDs to sequence state, uses the same
deadline object for local and kernel key admission, revalidates exact evidence
before apply, and isolates caught primary/sidecar CAS exceptions from later
actions. Those improvements do not converge end to end. Five blockers remain:
ordinary clear rejects valid publication order and stale high-water pages;
reconciliation cannot advance within partially consumed or empty high-water
pages; append-only JSON inventories still perform whole-history work and have a
permanent byte ceiling; apply can return terminal after an exact conflict leaves
safe debt; and both admission loops can acquire after their absolute deadline.

The focused Phase 3 corpus completed successfully, as did Python 3.11.16 package
import, `uv lock --check`, and Ruff on the structural rewrite paths. These green
checks do not cover the deterministic failures below. One independent full-suite
attempt in this review was invalidated when the virtual environment was changed
during execution for the Python 3.11 check; it is not cited as evidence. The full
development environment was restored afterward.

## Critical Issues

### CR-01: Clear cannot consume valid high-water inventory order or stale-only pages

**Files:** `src/cacheness/storage/operation_record.py:762-768`,
`src/cacheness/storage/lifecycle.py:520-657`,
`src/cacheness/storage/manifest_repository.py:694-734`,
`src/cacheness/storage/manifest_repository.py:1112-1145`

**Issue:** Manifest pages now yield current members in append-only publication
sequence, but `ClearTargetPage` still requires every page's keys to be lexically
sorted. `_snapshot_clear_targets()` passes the sequence-ordered entries directly
to the page constructor. A valid store that publishes `z` and then `a` therefore
cannot be cleared with the default page size. A deterministic probe produced
`CacheManifestIntegrityError: Clear target page keys must be unique and ordered`
for memory, JSON, and SQLite backends.

The same consumer also treats an empty page with a nonterminal high-water cursor
as corruption. Empty nonterminal pages are a normal result when one bounded
inventory window contains only deleted or replaced events. With
`max_inventory_items=2`, three overwrites of one key produced an empty first page
with `ManifestCursor(key='k', snapshot_high_water=3, next_sequence=3)`; `clear()`
then raised `Clear manifest page has an empty non-terminal cursor` instead of
advancing to the live third event.

This breaks D-10/D-11 and STOR-05 for ordinary valid stores and means the v2 cursor
fix is not end-to-end. It also prevents the requested page-size-one/interruption
proof from generalizing to default pages, deletion, replacement, and large stale
prefixes.

**Fix:** define target-page canonical order independently of source inventory
order (for example, sort each durable target page while keeping its signed source
and next cursors), and advance through empty nonterminal source pages rather than
rejecting them. Add all three metadata backends, reverse publication order,
multiple replacements/deletions, more than one inspection window of stale events,
encoded-size splits, and interruption at each page boundary.

### CR-02: Reconciliation continuation loops on partial and empty high-water pages

**Files:** `src/cacheness/storage/reconciliation.py:339-456`,
`src/cacheness/storage/reconciliation.py:1081-1150`,
`src/cacheness/storage/manifest_repository.py:657-740`,
`src/cacheness/storage/operation_repository.py:519-566`

**Issue:** The repositories expose only a page-level sequence cursor to the
reconciler. When the shared finding budget consumes fewer entries than the page,
`_next_manifest_cursor()`, `_next_operation_cursor()`, and
`_next_sidecar_cursor()` return the incoming cursor. For a first page that is
`None`, the v4 token therefore encodes no source position and the next call reads
the same first entry forever. A deterministic dry-run probe with two live
manifests, `manifest_page_size=2`, and `max_reconcile_actions=1` returned the
finding for key `a` and decoded continuation `(None, None, None, None,
'manifest')` on four consecutive calls; key `b` was never reached.

Empty nonterminal repository pages have the same failure: `consumed == 0` returns
the incoming cursor instead of the repository's `page.next_cursor`, so a stale
inventory window is reread forever. Apply does not repair manifest/report-only
pages, and dry-run intentionally mutates nothing, making this a deterministic
nonconvergence rather than a transient replay.

This violates D-13/D-14, STOR-06, and the stable no-skip/no-replay continuation
contract. A bounded result page is not useful if the authenticated resume chain
cannot advance.

**Fix:** expose exact post-entry cursors for every inventory family, as the
manifest repository already does for clear, and derive continuation from the last
actually consumed inventory position. When a page contains no eligible entries,
advance to its nonterminal page cursor. Add dry-run and apply chains where page
size exceeds the action budget, stale windows exceed `max_inventory_items`, and
all three sources interleave; assert terminal completion and exactly-once
membership per snapshot.

### CR-03: The inventories remain whole-history, permanently capped, and incomplete for legacy/pending evidence

**Files:** `src/cacheness/storage/operation_repository.py:440-566`,
`src/cacheness/storage/operation_repository.py:1309-1317`,
`src/cacheness/storage/operation_repository.py:1394-1512`,
`src/cacheness/storage/operation_repository.py:1550-1572`,
`src/cacheness/storage/manifest_repository.py:478-540`,
`src/cacheness/storage/manifest_repository.py:657-740`,
`src/cacheness/storage/path_security.py:885-951`

**Issue:** Primary and sidecar scheduling use one append-only JSON array per
family. Every append and page rereads, parses, and validates the entire history,
then rewrites the entire array. The file is limited by
`max_operation_record_bytes`; once it reaches that unrelated per-record limit,
`_append_inventory_event()` raises `Lifecycle inventory requires bounded
compaction`, but no compaction exists. Because checkpointing appends before each
record transition, a store eventually cannot finish cleanup or accept later
operations even though retired evidence no longer exists. A probe with an 8 KiB
record bound failed permanently on the sixteenth simple put; smaller valid bounds
failed earlier. The default merely postpones the same finite store-lifetime cap.

JSON and memory manifest inventories likewise validate the full append-only event
list on every page/publication, so `max_inventory_items` limits returned work but
not actual work. SQLite uses an indexed table and does not have this particular
problem.

Pending controls were not moved to a high-water index at all. Their cursor remains
lexical and stores only a name; `list_directory_names_bounded()` calls
`os.scandir()` across the entire directory and sets `max_inventory_names=None`.
Family filtering prevents unrelated names from entering the returned page, but it
does not bound the number inspected, so a large unrelated namespace still makes
startup/reconciliation unbounded and insert-before-cursor semantics remain
mutable.

Finally, a missing operation inventory is treated as an empty inventory even when
valid pre-index operation files exist. There is no bootstrap, rebuild-required
signal, or authenticated migration path, so upgrade-era recovery debt can become
invisible. Manifest repositories at least reject a wholly missing index when raw
rows exist, although their error is wrapped as a generic backend failure.

This does not satisfy D-05/D-10/D-13-D-16 or STOR-04/STOR-06. Scheduling state is
not destructive authority, but it has become an unbounded availability dependency
with a finite lifetime and incomplete legacy coverage.

**Fix:** use a durable chunked/indexed append log with bounded reads/writes and an
explicit safe compaction protocol, or backend-native sequence storage. Give
primary, sidecar, and pending families independent high-water cursors while
bounding actual directory inspection. Detect pre-index evidence and either build
the index from strictly validated names without assigning authority or return a
typed migration/rebuild-required outcome. Test well beyond the default 4,096
inspection window and beyond the current 1 MiB inventory ceiling, plus unrelated
names, restart, compaction interruption, and pre-index stores.

### CR-04: An exact apply conflict can return a terminal token while safe debt remains

**Files:** `src/cacheness/storage/reconciliation.py:457-490`,
`src/cacheness/storage/reconciliation.py:499-599`,
`src/cacheness/storage/reconciliation.py:1024-1079`

**Issue:** Apply correctly catches exact primary/sidecar CAS conflicts and
continues later independent work, but records a conflict as `attempted=True`.
`stale_or_unapplied` is therefore false and the retain helpers advance past it.
When it is the last scanned item, `next_priority` and every cursor remain `None`,
so the report is terminal even though current exact bytes still represent safe
debt.

A deterministic probe created one authenticated completed orphan sidecar, injected
an exact retirement conflict, and ran apply with a one-action budget. The report
contained both `completed_reconciliation_checkpoint_orphan` and
`reconciliation_action_conflict`, the exact safe sidecar remained, and
`resume_token` was `None`. Starting a brand-new scan might later rediscover it,
but following the returned authenticated continuation cannot, contradicting the
resumable contract.

This violates D-14 and STOR-06. Charging one actual CAS attempt is correct; treating
that charge as completed source progression is not.

**Fix:** distinguish attempted-budget accounting from completed source progression.
After a conflict, re-read/reclassify exact current primary and sidecar bytes and
retain a continuation at that source whenever safe or unresolved debt remains.
The same pass may continue later independent work, but it must not return terminal
until every scanned safe action is completed or represented by a continuation.

### CR-05: Both key-admission loops can acquire after the absolute deadline

**Files:** `src/cacheness/storage/integrity.py:46-82`,
`src/cacheness/storage/integrity.py:202-218`,
`src/cacheness/storage/integrity.py:310-376`,
`src/cacheness/storage/coordination.py:170-228`

**Issue:** One deadline object is now passed through local and kernel admission,
and timeout paths retire the registry reference and close the contender handle.
However, both retry loops attempt acquisition before checking whether the deadline
has expired. After a failed attempt sleeps for the remaining interval, the next
iteration can acquire a guard or POSIX/Win32 lock even though monotonic time is
already beyond the configured deadline. The current tests cover a live holder
that stays locked through timeout and a release before expiry; they do not cover a
release after expiry but before the contender is rescheduled for its next attempt.

The injected Win32 compatibility fallback also retries a blocking `lock()` call
when a shim rejects the `nonblocking` keyword. Native `_NativeWindowsLockApi`
supports `FAIL_IMMEDIATELY`, but this fallback means the claimed injected-adapter
deadline is not structurally guaranteed.

This leaves the absolute admission deadline selected for deterministic failure
under D-05/STOR-04/SECU-04 false at a boundary race.

**Fix:** check the shared monotonic deadline before every retry after the initial
immediate attempt, then perform only a nonblocking acquisition. Reject a Win32
adapter that cannot honor nonblocking acquisition instead of falling back to a
blocking call. Add deterministic fake-clock/fake-lock tests for release just
after expiry, plus timeout exception, guard count, descriptor, and Win32 token
cleanup assertions.

## Verification Evidence

- Focused Phase 3 corpus: terminal pass for atomic lifecycle, integrity, manifest
  CAS, reconciliation, concurrency, close, read contract, and clear recovery tests.
- Adversarial probes: deterministic failures reproduced for CR-01, CR-02, CR-03,
  and CR-04. CR-05 follows directly from acquisition-before-deadline-check control
  flow and is amenable to a fake-clock test.
- Python 3.11.16: managed-environment package import completed successfully with
  the declared `recommended` dependency group.
- `uv lock --check`: passed.
- Ruff: passed on the structural rewrite modules and their focused tests. Broader
  changed-file output contains known pre-existing repository lint debt and is not
  reported as a new Phase 3 issue.
- Native Windows was unavailable; injected adapter coverage was inspected. D-22
  remains the accepted one-user/session boundary.

---

_Reviewed: 2026-09-02T04:02:37Z_
_Reviewer: independent Phase 3 deep review_
