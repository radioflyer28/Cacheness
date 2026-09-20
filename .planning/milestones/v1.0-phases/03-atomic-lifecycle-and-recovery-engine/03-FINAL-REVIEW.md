---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-06T05:45:00Z
depth: deep
files_reviewed: 8
files_reviewed_list:
  - src/cacheness/core.py
  - src/cacheness/metadata.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/lifecycle.py
  - src/cacheness/storage/lifecycle_authority.py
  - src/cacheness/storage/sqlite_lifecycle_authority.py
  - tests/test_unified_cache_lifecycle_authority.py
  - tests/test_phase3_windows_contract.py
findings:
  critical: 2
  warning: 1
  info: 0
  total: 3
status: issues_found
---

# Phase 03: Final Focused Repair Review

**Reviewed:** 2026-09-06T05:45:00Z  
**Depth:** deep  
**Files Reviewed:** 8  
**Status:** issues_found

## Summary

The 4143242 repair correctly derives fields within `_sync_authority_projection()`
from one authenticated authority snapshot, routes normal public payload operations
through `BlobStore`, and moves the UnifiedCache Windows preflight before public-root
or facade-metadata materialization. The 9088df7 fallback is limited to compatibility
objects without `_lock`; initialized `UnifiedCache` instances retain their `RLock`.
The targeted authority, Windows-contract, integrity, and serialization tests passed
on this host (one native-Windows target skipped as designed).

However, facade projection *removal* remains unconditional. A stale operation can
erase a concurrently published replacement's compatibility row and, on SQLite,
cascade-delete its custom-metadata links. A tombstoned authority entry is also
rendered as a live projection during recoverable cleanup debt. Those violations leave
the public cache/query surface disagreeing with the authority and cause durable custom
metadata loss. Phase 3 is not ready to ship.

The reviewed D-32 artifacts remain honest: the current host is
`UNAVAILABLE`/`NOT_QUALIFIED` with `native_evidence: false`; nothing here treats it
as Windows support.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Stale authority projection cleanup can delete a concurrent replacement and its custom metadata [BLOCKER]

**File:** `src/cacheness/core.py:1321-1326`, `src/cacheness/core.py:1345-1358`, `src/cacheness/metadata.py:1693-1719`

**Issue:** `_sync_authority_projection()` reads the authority, and if it observes
absence, calls `metadata_backend.remove_entry(cache_key)` without binding that delete
to the observed absence. `_repair_projection_after_failed_put()` has the same
snapshot-then-unconditional-remove shape. A replacement can promote after the stale
read and before the removal:

1. An invalidate/TTL/read-failure path exactly deletes M1 and enters `_sync...`.
2. `_sync...` reads no entry; another `UnifiedCache` promotes M2, writes its
   projection, and stores its custom metadata.
3. The first facade unconditionally removes M2's projection. With `SqliteBackend`,
   `remove_entry()` also cascades/explicitly cleans `CacheMetadataLink` rows.

M2's payload remains authority-owned, but its compatibility projection is absent and
its custom metadata is permanently deleted. A later `get()` may recreate only the
plain projection; it cannot reconstruct the custom metadata. Thus the repair still
allows a stale cleanup to delete concurrent replacement state, precisely the prior
CR-01 failure class, and violates the public metadata/query contract.

**Fix:** Make projection writes/removals conditional on the exact authority
expectation/generation they represent. The compatibility backend needs a
generation-aware compare-and-set/delete primitive (including custom-link ownership),
or projection teardown must be performed by the durable lifecycle authority in the
same transition. Re-read-and-remove is insufficient: the final deletion must be
conditional in the metadata store. Add deterministic two-facade tests that interleave
an M2 put (including `custom_metadata`) after M1 deletion/failed-first-put observes
absence and before projection teardown; assert M2 remains queryable and all its links
remain intact.

### CR-02: A recoverable tombstone is projected as a live cache entry [BLOCKER]

**File:** `src/cacheness/core.py:1234-1245`, `src/cacheness/core.py:1321-1343`, `src/cacheness/storage/lifecycle.py:332-390`

**Issue:** `AuthorityLifecycleEngine.delete()` promotes a signed tombstone before
settling cleanup debt. If the payload cleanup raises, the tombstone remains the
canonical authority state and the error is propagated. On the next facade read,
`_authority_snapshot_entry()` sees that tombstone and calls
`_sync_authority_projection()`. The latter does not reject `manifest.state ==
"tombstoned"`; it writes a normal compatibility entry containing the tombstone
locator, handler type, size, and old user metadata.

This makes a logically deleted key visible again to `list_entries()`/`query_meta()`
and can make `get()` attempt a nonexistent tombstone payload. The state is not a
legacy entry: it is an authority-owned recoverable deletion, so treating it as live
breaks invalidation/cleanup/read-failure semantics until an external reconciliation
finishes. It also leaves any custom-metadata link visible for a key the authority has
already deleted.

**Fix:** Treat every non-`committed` authority manifest as absent at the facade
boundary. `_authority_snapshot_manifest()`/`_sync_authority_projection()` should
return a typed non-live result for tombstones and remove the corresponding projection
through the same generation-conditional mechanism as CR-01, while retaining only the
authority's cleanup debt for reconciliation. Add regressions that force
`_settle_debts()` to fail during `invalidate`, TTL cleanup, and corrupt-read cleanup;
assert `get`, `list_entries`, `query_meta`, and custom-metadata lookup do not expose
the tombstoned generation and cannot delete a concurrent replacement.

## Warnings

### WR-01: `_authority_snapshot_entry()` can return an authority snapshot and projection from different generations [WARNING]

**File:** `src/cacheness/core.py:1103-1119`

**Issue:** The method reads `snapshot` first, then may invoke
`_sync_authority_projection()`, which performs a fresh authority read. If M2 is
promoted between those reads, the method returns the original M1 snapshot with an M2
projection. Exact CAS prevents that stale M1 from deleting M2, but cache-policy code
then evaluates expiry/read-failure behavior using incoherent inputs and can produce a
spurious miss rather than a coherent M2 result. The existing regression only exercises
`_sync_authority_projection()` directly and deliberately accepts an old projection;
it does not exercise this returned-pair race.

**Fix:** After a repair, re-read the authority and projection as a bounded retry and
return only a pair whose projection generation/locator matches that final snapshot.
If stability cannot be established, fail as a lifecycle conflict rather than applying
policy to mixed observations. Add a deterministic interleaving at the
`read_entry`/projection-repair boundary and assert the returned pair is generation
coherent.

---

_Reviewed: 2026-09-06T05:45:00Z_  
_Reviewer: gsd-code-reviewer_  
_Depth: deep_
