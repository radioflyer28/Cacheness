---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-02T18:05:00Z
depth: deep
head: dee9f01
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
  critical: 2
  warning: 1
  info: 0
  total: 3
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-09-02T18:05:00Z
**Depth:** deep
**HEAD:** `dee9f01`
**Status:** issues_found

## Summary

Iteration 18's two repairs are effective. Memory and JSON manifest compaction
retain malformed live projections through unrelated overwrite/removal, later
valid events, repeat reconciliation, and JSON reopen. Operation inventory event
read denial and bounded-read failures are typed, retain family/sequence context
and cause, preserve membership, and cannot produce a clean terminal result.
The ten new regressions pass.

Three adjacent inventory defects remain. A lazily absent family head can turn a
current v2 interrupted operation into a spurious migration requirement; the
operation compaction floor can strand a fresh primary page on an invalid cursor;
and memory/JSON manifest scans still pay store-lifetime rather than live-store
cost while holding clear admission. These are independent of the accepted
trusted-owner and Windows topology boundaries.

## Critical Issues

### CR-01: A missing sibling-family head blocks current v2 recovery as a false migration

**Files:** `src/cacheness/storage/operation_repository.py:543-581`,
`src/cacheness/storage/operation_repository.py:607-631`

**Issue:** Inventory heads are created lazily per family. If a current store has
a primary record but has never created a sidecar head, reading the sidecar family
falls back to `_has_preindex_evidence()`. That fallback scans the shared
`operations/` namespace with `max_scanned_names=max_inventory_items`; unrelated
infrastructure and primary names consume the bound even though the sidecar
filter rejects them. The resulting bounds error is translated into
`CacheBlobMigrationRequiredError` before the current indexed primary can be
reconciled.

A deterministic public probe used `max_inventory_items=2`, interrupted one put
at `candidate_publish`, and called `BlobStore.reconcile()`. The directory
contained only `.cacheness-inventory-v2`, `.conditional-locks`, and the one
current primary record. Reconciliation raised
`CacheBlobMigrationRequiredError(family="sidecar")`; there was no legacy
sidecar. This recreates the cross-family budget collision the v2 family split
was intended to remove and blocks STOR-06 recovery at an ordinary crash seam.

**Fix:** Durably establish empty current-version heads for all operation families
when a new store/repository is initialized, before any family can publish
evidence, while preserving fail-closed explicit migration for genuinely old
stores. Alternatively make legacy detection use a separately bounded family
namespace whose unrelated entries cannot consume its proof-of-absence budget.
Cover each lone primary/sidecar/pending family, both sibling heads absent,
one-more-than-bound unrelated names, reopen, and the public recovery/reconcile
paths with `max_inventory_items=1/2`.

### CR-02: Compaction can strand the live floor and make a sparse primary page construct an invalid cursor

**Files:** `src/cacheness/storage/operation_repository.py:816-859`,
`src/cacheness/storage/operation_repository.py:924-989`

**Issue:** `_compact_inventory_window()` advances `first_live_sequence` only
when it exactly equals the window's `start`. Once an earlier window establishes
a live floor and the compaction cursor moves past it, a later retirement of that
floor cannot advance it when the cursor wraps behind the floor. All events can
therefore be deleted while the durable floor remains on a sparse position.

A fresh primary page starting from that floor has no prior name. When its first
bounded window consists only of gaps but the high-water lies later,
`_inventory_page()` constructs `OperationCursor("~", ...)`; `OperationCursor`
requires a 32-hex identifier and raises `CacheUnsafePathError`. Public
`BlobStore.reconcile()` reproduced that failure with a v2 head containing an
oldest live primary followed by retired traffic, after the oldest primary was
retired. The head remained `first_live_sequence=2` with every event absent.
Sidecar and pending cursors accept `~`, but their stale floor still causes
repeated lifetime-prefix paging.

This blocks deterministic recovery after the supported pattern of one
interrupted oldest operation plus later completed traffic and violates STOR-06.

**Fix:** Advance the floor whenever a bounded compaction window actually covers
the current floor, deriving the next proven live/lower-bound position from the
subrange beginning at that floor. Preserve monotonicity across wrap and reopen.
For an empty nonterminal primary page, use a valid authenticated before-first or
last-inspected sequence cursor rather than the sidecar/pending `~` sentinel.
Test primary, sidecar, and pending families with a live floor, later stale tail,
retirement after cursor wrap, all-sparse and later-live windows, page size one,
reopen, and public terminal convergence.

## Warnings

### WR-01: Memory and JSON clear still scan deleted manifest history under aggregate admission

**Files:** `src/cacheness/storage/manifest_repository.py:512-542`,
`src/cacheness/storage/manifest_repository.py:692-725`,
`src/cacheness/storage/manifest_repository.py:850-923`,
`src/cacheness/storage/lifecycle.py:506-706`

**Issue:** The memory/JSON manifest head tracks no `first_live_sequence`, and
`list_page()` walks every integer position from sequence one. Compaction deletes
stale events but cannot let a fresh scan skip those proven gaps. Clear must build
its complete target chain while holding aggregate admission, so ordinary
overwrite history directly lengthens the interval during which unrelated
operations are excluded.

With one key overwritten 40 times, `max_inventory_items=2`, and manifest page
size one, `clear()` required 21 manifest page calls on both memory and JSON
(one preflight plus twenty snapshot windows) to clear one live key. SQLite used
two calls because its indexed query skips deleted rows. Increasing overwrite
history increases the memory/JSON admission interval without increasing live
data. This is the manifest analogue of the already-recognized pending lifetime
prefix problem and conflicts with the short bounded aggregate-snapshot contract.

**Fix:** Add a backward-compatible durable live floor (or an equally bounded
indexed sparse successor) for memory/JSON manifest inventories. Advance it only
through exact-revalidated absence/digest mismatch, retain it on decode/migration
failure, and preserve existing high-water cursors. Add large overwrite/delete
prefixes, one later live manifest, tiny windows, reopen, concurrent append, and
clear instrumentation proving work is proportional to current indexed members
rather than lifetime sequence count. Keep SQLite behavior unchanged.

## Verification Performed

- Iteration-18 targeted regressions: 10 passed.
- Complete `test_blob_store_reconciliation.py` plus
  `test_manifest_repository_cas.py`: 140 tests passed.
- Malformed manifest compaction probes passed for memory/JSON overwrite,
  removal, repeat reconciliation, later valid events, and JSON reopen.
- Primary/sidecar/pending inventory event permission and bounds injections
  remained typed with exact context/cause and retained event files.
- Public current-v2 missing-sidecar-head probe reproduced CR-01 with one
  interrupted primary and a two-name inspection policy.
- Floor-wrap/all-sparse public reconciliation probe reproduced CR-02.
- Forty-overwrite clear probe reproduced WR-01 on memory/JSON and established
  the SQLite comparison baseline.
- Changed-path Ruff and `git diff --check` passed. The active environment is
  Python 3.11.16 and imports `cacheness 0.3.14` successfully.
- The parent workflow holds clean full-suite evidence for the prior head; this
  pass prioritized bounded adversarial probes and did not repeat it.

No additional actionable integrity, concurrency, compatibility, or performance
defect was found in the inspected iteration-18 boundaries. Known unrelated
repository-wide lint debt, shutdown-only destructor behavior, and base-install
NumPy packaging were excluded.

---

_Reviewer: independent deep rereview, iteration 19_
