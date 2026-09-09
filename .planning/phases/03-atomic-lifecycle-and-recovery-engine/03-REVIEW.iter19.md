---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-02T17:50:31Z
depth: deep
head: 34f1058
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
  warning: 1
  info: 0
  total: 4
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-02T17:50:31Z
**Depth:** deep
**HEAD:** `34f1058`
**Status:** issues_found

## Summary

Iteration 19's focused regressions pass, and the new live floors do advance in
the tested wrap/reopen cases. The repairs are not yet sufficient at the actual
compatibility and work-budget boundaries. Two independent initialization paths
can relabel genuine pre-index evidence as an empty current-v2 family. In
addition, ordinary successful operations generate sparse primary history faster
than retirement compacts it, so recovery's action budget does not bound its
actual work. The manifest floor has the analogous lifetime-history behavior
after a supported post-authority failure.

These failures are deterministic with `max_inventory_items=1`; they do not
depend on a malicious local store owner, unsupported topology, or malformed
payload data. A green full suite at this head therefore does not establish the
Phase 3 crash-convergence and bounded-work contracts.

## Critical Issues

### CR-01: Primary pre-index evidence is never recognized because the detector uses the wrong filename grammar

**Files:** `src/cacheness/storage/operation_repository.py:560-598`,
`src/cacheness/storage/operation_repository.py:1108-1118`,
`src/cacheness/storage/operation_repository.py:1976-1982`

**Issue:** A primary operation record is stored as
`operations/<32-hex-operation-id>.json`, but `_has_preindex_evidence("primary")`
passes each complete directory name to `_is_hex_identifier()`. That predicate
accepts exactly 32 hexadecimal characters and therefore rejects every canonical
primary filename because of its `.json` suffix.

A fresh repository containing only `operations/aaaa...aaaa.json` returned
`False` from `_has_preindex_evidence("primary")`; `list_page()` then returned a
terminal empty `OperationPage`. Public reconciliation consequently treats real
unindexed primary lifecycle evidence as absence rather than raising the typed
migration-required outcome promised by `_read_inventory()`. This can hide
unfinished lifecycle work and falsely claim terminal recovery.

**Fix:** Add and use an exact primary-filename recognizer that validates the
`.json` suffix and its 32-hex stem, matching `locator_for()`. Cover a lone raw
primary record through repository paging, public dry-run/apply reconciliation,
and lifecycle recovery, with default/custom signing keys, reopen, unrelated
names, and `max_inventory_items=1/2`. Each path must fail closed with the typed
family-specific migration error and must not create current-v2 heads or a
marker.

### CR-02: Key absence authorizes the current-v2 marker before genuine legacy evidence is checked

**Files:** `src/cacheness/storage/blob_store.py:997-1053`,
`src/cacheness/storage/integrity.py:208-222`,
`src/cacheness/storage/operation_repository.py:718-767`

**Issue:** `_manifest_key(initialize_new_store=True)` defines freshness as
"manifest page empty and default key inode absent." It creates/acknowledges a
new key and then calls `initialize_new_store()`, whose first durable transition
is the marker that makes all absent family heads read as empty. No operation
family is checked for pre-index evidence before that marker is published.

The failure is independent of CR-01. With one correctly recognized legacy
`reconcile-action-<32hex>.json` sidecar, `reconcile()` initially raised
`CacheBlobMigrationRequiredError(family="sidecar")`. On an equivalent store,
calling `put()` first succeeded, created the marker/empty heads, and subsequent
reconciliation no longer surfaced the sidecar migration requirement; the
sidecar remained on disk but outside the claimed current inventory. A missing
key is not proof that an existing operation namespace is a fresh store.

**Fix:** Establish freshness as one fail-closed atomic compatibility decision,
not a pre-lock key probe. Before publishing the marker or any empty head, prove
that every legacy family has no evidence (and no v1 inventory), under the same
initialization transition that publishes current-v2 provenance. Ambiguous or
over-bound legacy namespaces must remain typed migration-required; validated
current-v2 marker/head stores must still ignore unrelated names without a
legacy scan. Test every crash seam including marker creation, each family head,
key acknowledgement, constructor/first mutation failure, and reopen for both
default and injected keys.

### CR-03: Normal operation history outruns compaction, and recovery does not charge sparse-page work

**Files:** `src/cacheness/storage/operation_repository.py:924-1035`,
`src/cacheness/storage/operation_repository.py:1037-1107`,
`src/cacheness/storage/operation_repository.py:2190-2260`,
`src/cacheness/storage/lifecycle.py:1294-1327`,
`src/cacheness/storage/lifecycle.py:1329-1387`

**Issue:** Each normal put appends several primary inventory events as its
operation record advances checkpoints, but terminal retirement performs only
one compaction window capped by `max_inventory_items`. With the supported value
`max_inventory_items=1`, a put generated five sequence positions and retired
only one. Ten ordinary successful puts left no live primary operation records
but persisted `first_live_sequence=11` and `next_sequence=51`.

This is not merely a resumable-report presentation issue. With
`operation_page_size=1` and `max_reconcile_actions=1`, one direct
`LifecycleEngine.recover()` invocation made 40 `list_page()` calls across those
stale positions. Empty pages do not decrement `remaining_actions`, so the
action budget does not bound actual inventory reads. Dry-run reconciliation
made one read but returned a resume token for nonexistent debt, forcing history-
proportional invocations to reach terminal. The gap grows under normal success
at `max_inventory_items=1/2`, contradicting the durable live-floor purpose and
the bounded recovery contract.

**Fix:** Make retirement/compaction amortize at least all sequence positions
created by the retiring exact operation (or directly retire its authenticated
event positions) so successful traffic cannot grow a sparse prefix without
bound. Independently charge every inspected inventory position/page against a
recovery work budget; an empty sparse page must not permit an unbounded loop in
one call. Preserve stable old high-water cursors and no normal-read mutation.
Test long successful create/replace/delete sequences for primary, sidecar, and
pending families at limits 1/2, then prove both public recovery and repeated
reconciliation reach terminal in work proportional to live evidence rather
than total historical checkpoints.

## Warnings

### WR-01: Manifest live floors are not crash-convergent after authority publication

**Files:** `src/cacheness/storage/manifest_repository.py:732-783`,
`src/cacheness/storage/manifest_repository.py:785-827`,
`src/cacheness/storage/manifest_repository.py:908-984`,
`src/cacheness/storage/lifecycle.py:506-706`

**Issue:** The memory/JSON live floor advances only in best-effort compaction
after the manifest authority has already been published. A process loss or
backend failure at that exact post-authority boundary leaves the publication's
old scheduling events durable but leaves `first_live_sequence` unchanged.
Recovery of the operation record does not compact the manifest inventory, and
clear only pages it.

A crash-equivalent bounded probe injected failure immediately when post-publish
compaction began. After one initial put and eight same-key overwrites whose new
manifests all became authority before the reported backend error, the JSON head
was `first_live_sequence=1, next_sequence=10`. Clearing the one live key then
required 10 manifest `list_page()` calls at page/budget size one. Repeating the
supported post-authority failure makes aggregate clear admission proportional
to failure history, recreating the lifetime-scan problem the new floor was
intended to remove.

**Fix:** Give mutating recovery or clear snapshot setup a bounded,
exact-revalidated way to advance crash-left manifest floors before traversing
the full snapshot; alternatively durably checkpoint compaction debt and consume
it under an explicit inspected-position budget. Do not mutate normal reads.
Cover process loss after JSON/memory authority publication but before
compaction, reopen, later live records, old high-water cursors, malformed/read-
failure floor pinning, and clear page-call bounds proportional to current live
members.

## Verification Performed

- Iteration-19 focused regressions and adjacent pre-index tests: 19 passed.
- Changed-path Ruff passed for the four modified source modules and the
  reconciliation test.
- Primary legacy grammar probe: `_has_preindex_evidence("primary") == False`
  and terminal empty `list_page()` for a real `<32hex>.json` record.
- Sidecar initialization probe: migration-required before first write; first
  write then created current-v2 provenance and hid the still-present sidecar.
- Normal-success operation probe: ten puts yielded primary head floor 11 / next
  51; one recovery call performed 40 empty inventory pages despite action budget
  one.
- Post-authority manifest-failure probe: one live JSON key, head floor 1 / next
  10, and ten clear page calls.
- The parent workflow separately obtained a terminal green full-suite result on
  the restored Python 3.13 environment. Those tests do not cover the adversarial
  states above.
- `uv.lock` is unchanged. Known unrelated repository-wide lint debt,
  shutdown-only SQLite destructor behavior, and base-install NumPy packaging
  were excluded.

---

_Reviewer: independent deep rereview, iteration 20_
