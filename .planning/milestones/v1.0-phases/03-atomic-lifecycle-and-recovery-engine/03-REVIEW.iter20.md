---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-02T18:41:27Z
depth: deep
head: f04eb3f
archived_as: iteration-21-input
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
  critical: 3
  warning: 0
  info: 0
  total: 3
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-02T18:41:27Z
**Depth:** deep
**HEAD:** `f04eb3f`
**Status:** issues_found

## Summary

Iteration 20 fixed the canonical primary filename grammar and now bounds each
individual sparse primary/pending recovery page. The repair still does not
establish the claimed compatibility or convergence boundaries. A lone sibling
head is accepted as proof for an absent family even though the older lazy-head
implementation could create exactly that mixed state while leaving pre-index
evidence in another family. Manifest maintenance consumes only one fixed
window before clear snapshots, so repeated supported post-authority failures
still make clear admission proportional to failure history. Finally, the fixed
operation-maintenance windows are insufficient when one long-lived operation
pins the floor: a successful clear produces hundreds of stale primary slots,
and the new inspected-position budget repeatedly reads the same first gap
because lifecycle recovery has no continuation or stale-page compaction.

These are deterministic states at `max_inventory_items=1`, not hostile store-
owner substitution. The full suite can remain green while genuine legacy
evidence is hidden and recovery never converges.

## Critical Issues

### CR-01: A sibling v2 head is not proof that an absent family contains no legacy evidence

**Files:** `src/cacheness/storage/operation_repository.py:746-758`,
`src/cacheness/storage/operation_repository.py:760-821`

**Issue:** `_has_current_v2_sibling_head()` makes `_read_inventory()` return an
empty v2 head for an absent family without checking that family's v1 inventory
or exact raw evidence. `initialize_new_store()` likewise skips the all-family
compatibility proof whenever *any* head exists, then publishes the marker and
missing empty heads. That inference is not justified for the explicitly
supported older lazy-head layout: the old implementation could create one
family head while pre-index evidence already existed in another family.

A bounded probe created a valid primary v2 head with no marker/sidecar head and
retained `operations/reconcile-action-<32hex>.json`. Reopening the sidecar
inventory returned an empty terminal page while the legacy bytes remained on
disk. The same branch hides a target-family `.cacheness-inventory-v1` file.
This is precisely mixed upgrade evidence, not an unrelated filename, and it can
make reconciliation falsely terminal. A marker-only state has the analogous
race if raw old-version evidence is published after the compatibility scan but
before/after marker publication because the initialization lock is disjoint
from the family transition locks.

**Fix:** Replace the sibling-head shortcut with an authenticated/versioned
store-level initialization record that proves the all-family compatibility
decision, or treat markerless lazy-head stores with absent families as an
explicit migration state and inspect/migrate them under a bounded documented
path. Do not turn ambiguity into an empty family merely to ignore unrelated
names. Serialize the compatibility decision and marker publication against all
family evidence publication, and test mixed raw/v1 families plus marker/head/key
crash seams, reopen, constructor recovery, first mutation, and public
reconciliation for default and injected keys.

### CR-02: One manifest maintenance window does not make clear work proportional to live members

**Files:** `src/cacheness/storage/manifest_repository.py:735-816`,
`src/cacheness/storage/lifecycle.py:520-570`

**Issue:** A JSON/memory publication always attempts compaction, but every
post-authority compaction failure can durably add another stale manifest event.
`compact_inventory_for_recovery()` consumes only one 64-position window, and
`_snapshot_clear_targets()` calls it only once before synchronously following
every manifest cursor, writing a zero-target clear page for each stale window.
There is no inspected-position total, maintenance continuation, or direct
retirement of the superseded publication event.

With one live JSON key, `max_inventory_items=1`, and 80 injected failures at the
supported boundary after authority publication but before compaction, the head
was `first_live_sequence=1, next_sequence=82`. Restoring compaction and calling
`clear()` still made 18 manifest `list_page()` calls to clear that one key. The
count grows with prior failures beyond the fixed 64-slot pass. Reopening once
only moves one window; clear on that reopened store has the same tail for a
sufficiently long history.

**Fix:** Make each authority transition durably identify/retire its superseded
event, or persist bounded maintenance debt with a continuation that mutating
recovery can consume before aggregate traversal without turning clear into a
history scan. Prove same-process repeated post-authority failures, one/multiple
reopens, later live keys, old high-water cursors, JSON and memory, and page
limits one/two. Clear page calls after maintenance must be proportional to live
snapshot members, not failure count; normal reads must remain non-mutating.

### CR-03: Fixed retirement windows leave successful lifecycle history permanently ahead of the live floor

**Files:** `src/cacheness/storage/operation_repository.py:976-1105`,
`src/cacheness/storage/operation_repository.py:2345-2367`,
`src/cacheness/storage/lifecycle.py:1347-1420`

**Issue:** Retirement performs two fixed 64-position compaction windows. A live
record at `first_live_sequence` prevents the floor from advancing even while
round-robin maintenance deletes later stale events. A clear naturally creates
this condition: its own primary record remains live while each target runs a
successful child delete lifecycle. When the clear record finally retires, two
windows cannot advance across all of the already-deleted child history.

A successful clear of 140 JSON keys at manifest/operation/inventory size one
left the primary head at `first_live_sequence=768, next_sequence=1406`; the
first page was empty but nonterminal, leaving 638 stale positions after a fully
successful public operation. More seriously, three calls to
`LifecycleEngine.recover()` left the head byte-for-byte unchanged. The new
`inspected_positions` decrement correctly caps each individual call, but
recovery resets its cursor to `None`, does not compact inspected stale primary
positions, and persists no continuation, so it charges the same first gap on
every invocation. A later live tombstone after this prefix is consequently
unreachable to constructor recovery and `_find_tombstone_record()` can report
"no matching operation" forever. Reconciliation tokens can advance through
the prefix, but require history-proportional calls and do not repair the
recovery floor.

**Fix:** Retire all authenticated inventory positions owned by the completing
operation/aggregate lifecycle, or persist a crash-safe compaction continuation
that advances the floor across exact-proven gaps. Lifecycle recovery and
tombstone lookup must either carry an exact continuation or durably compact
each charged sparse page; a per-call counter without forward progress is not
convergence. Test a pinned live operation plus long successful
create/replace/delete traffic, large successful clear, interruption during
maintenance, reopen, later live tombstone, old cursors, and limits one/two.
Repeated recovery must monotonically advance and reach terminal in work
proportional to live evidence.

## Warnings

None.

## Verification Performed

- Iteration-20 focused compatibility tests passed (14 selected tests), which
  confirms the new exact `<32hex>.json` recognizer but does not cover mixed
  lazy-v2/legacy families.
- Mixed-family probe: one valid primary head plus a raw exact sidecar produced
  an empty sidecar inventory page while retaining the sidecar bytes.
- Manifest-debt probe: 80 post-authority compaction failures, one live key, and
  a restored maintenance path required 18 clear manifest pages at limits one.
- Successful-clear probe: 140 keys left primary floor 768 / next 1406 and a
  nonterminal empty first page; three explicit recovery calls made no floor
  progress. Pending inventory did reach its terminal floor, isolating the
  defect to primary floor retirement/continuation.
- The `inspected_positions` values themselves matched the number of primary
  positions read in the probes. Pending recovery persists an exact cursor and
  reconciliation resume tokens advance exact source cursors; no additional
  arithmetic defect was established there. The actionable defect is discarded
  primary recovery/tombstone continuation plus maintenance debt, not the new
  decrement operation.
- The parent workflow separately reports a terminal green full suite and
  Python 3.11 smoke at this head. Those tests do not cover the adversarial
  states above. `uv.lock` is unchanged. Known unrelated repository-wide lint
  debt, shutdown-only SQLite destructor behavior, and base-install NumPy
  packaging were excluded.

---

_Reviewer: independent deep rereview, iteration 21_
