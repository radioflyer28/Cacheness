---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-05T22:10:00-04:00
depth: deep
files_reviewed: 11
files_reviewed_list:
  - src/cacheness/core.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/lifecycle.py
  - src/cacheness/storage/lifecycle_authority.py
  - src/cacheness/storage/memory_lifecycle_authority.py
  - src/cacheness/storage/sqlite_lifecycle_authority.py
  - src/cacheness/storage/path_security.py
  - tests/test_cache_integrity.py
  - tests/test_filesystem_containment.py
  - tests/test_phase3_windows_contract.py
  - tests/test_unified_cache_lifecycle_authority.py
findings:
  critical: 3
  warning: 0
  info: 0
  total: 3
status: issues_found
---

# Phase 03: Post-fix Code Review Report

**Reviewed:** 2026-09-05T22:10:00-04:00
**Depth:** deep
**Files Reviewed:** 11
**Status:** issues_found

## Summary

Reviewed the actual source and test changes in `a00f9d7`, `8642823`, and
`79a61bb`, including cross-instance authority/projection call chains. The native
Win32 error mapping correctly turns `ERROR_FILE_NOT_FOUND`/`ERROR_PATH_NOT_FOUND`
into `FileNotFoundError` and existing-path errors into `FileExistsError`; the
non-Windows seams do not claim Darwin `UNAVAILABLE` evidence proves Windows
support. However, the UnifiedCache facade still has three release-blocking
integrity defects: two projection races can remove a committed replacement or
leave a failed candidate visible, and its public Windows facade path mutates the
configured cache directory before the authority preflight rejects an unprovisioned
root.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Projection sync can label an old locator as the newer authority generation and then delete the replacement

**File:** `/Users/akriz/code/cacheness/src/cacheness/core.py:1221-1244`

**Issue:** `_sync_authority_projection()` reads the authority twice without
binding the projection contents to one `EntrySnapshot`: `BlobStore.get_metadata()`
can return manifest M1, another cache instance can promote M2, and the subsequent
`read_entry()` returns M2. The method then writes M1's `actual_path` with M2's
`authority_generation`. A following `get()` accepts that forged matching generation,
opens the now-cleaned M1 path, and `_retire_exact_authority_snapshot()` deletes M2
using its exact expectation. This is a concrete cross-instance data-loss race, not
just a transient cache miss.

Reproduced by interleaving a second `put()` between `get_metadata()` and
`read_entry()` in `_sync_authority_projection()`: the first cache returned `None`,
and the second cache's committed replacement and projection were both removed.

**Fix:** Derive every projected field from the same authenticated authority
snapshot, not from `get_metadata()` plus a later snapshot. Add a lifecycle helper
that returns `{snapshot, manifest}` atomically under the authority read contract,
or read a snapshot, authenticate its `manifest` bytes, render it, then re-read and
retry unless `expectation` is unchanged before writing the projection. Do not write
the projection at all when the observed generation differs from the manifest being
rendered. Add a deterministic regression test for this exact interleaving and
assert the replacement remains readable.

### CR-02: Failed non-conflict promotions leave stale metadata pointing to a deleted payload candidate

**File:** `/Users/akriz/code/cacheness/src/cacheness/core.py:1206-1219,1261-1277`

**Issue:** `_prepare_authority_projection()` commits compatibility metadata before
the SQLite authority promotion. If a later lifecycle step fails (for example,
`record_verification()` or `promote_mutation()` raises an I/O error), the lifecycle
engine aborts and reclaims the candidate, but `UnifiedCache.put()` only repairs the
projection for `CacheBlobLifecycleConflictError`. All other failed puts return with
metadata carrying the candidate generation and an `actual_path` that no longer
exists, while the authority still owns the previous generation.

This was reproduced by injecting an `OSError` at `put.before_promotion`: after the
failed call, metadata contained the candidate generation and a nonexistent payload
path, while the SQLite authority retained the old generation. A later `get()`
happens to repair it, but direct compatibility metadata consumers and any policy
operation before that repair observe a payload/metadata disagreement.

**Fix:** On every exception from `_cache_blob_store.put()` after the facade has
installed the projection hook, restore/synchronize the projection from the current
authority entry before re-raising the original error. Preserve the original error
as the direct failure if repair also fails, and explicitly handle an absent
authority by removing the projection. Add regression coverage for both a
pre-promotion verification failure and a post-promotion cleanup failure.

### CR-03: The public UnifiedCache facade mutates its configured Windows cache path before authority preflight

**File:** `/Users/akriz/code/cacheness/src/cacheness/core.py:103-106,128-140`

**Issue:** The new authority route preflights `BlobStore.put()`, but public
`UnifiedCache` construction still executes `self.cache_dir.mkdir(...)` and creates
`GuardedHandlerIO` before it constructs the inner authority root. On Windows with
an absent/unprovisioned configured `cache_dir`, `UnifiedCache(...); put(...)` thus
creates the configured path before `SqliteLifecycleAuthority.preflight_mutation()`
rejects the absent `.cacheness/unified-cache-v1` root. The added contract test
covers direct `BlobStore.put()` only, so it misses the primary public facade now
promoted by CR-01.

The facade path was reproduced with the Windows platform seam: `put()` raised the
expected provisioning error, but the configured cache directory already existed.
That violates the stated non-mutating Windows preflight boundary and leaves an
operator with a partially materialized root that the offline-provisioning workflow
was supposed to own.

**Fix:** Make `UnifiedCache` defer cache-directory and metadata-backend
materialization until after its BlobStore authority has successfully preflighted the
same configured root, or add a facade-level non-mutating preflight before either
`mkdir` or metadata backend initialization. Cover `UnifiedCache` (JSON and SQLite
metadata configurations) with an absent Windows root and assert no configured-root
or metadata artifact is created.

---

_Reviewed: 2026-09-05T22:10:00-04:00_
_Reviewer: gsd-code-reviewer_
_Depth: deep_
