---
phase: 01-compatibility-and-security-baseline
reviewed: 2026-08-30T03:35:34Z
depth: deep
files_reviewed: 13
files_reviewed_list:
  - src/cacheness/core.py
  - src/cacheness/metadata.py
  - src/cacheness/query_validation.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/clear_recovery.py
  - src/cacheness/storage/guarded_handler_io.py
  - src/cacheness/storage/path_security.py
  - tests/test_cache_integrity.py
  - tests/test_clear_recovery.py
  - tests/test_filesystem_containment.py
  - tests/test_phase1_quality_gates.py
  - tests/test_query_meta.py
  - tests/test_query_meta_security.py
findings:
  critical: 3
  warning: 0
  info: 0
  total: 3
status: issues_found
---

# Phase 1: Renewed Code Review Report

**Reviewed:** 2026-08-30T03:35:34Z
**Depth:** deep
**Files Reviewed:** 13
**Range:** `494d661..a73d880`
**Status:** issues_found

## Summary

The gap plans close the exact CR-01 staged-inode substitution and CR-06 signed-64
query-boundary reproductions, and candidate naming avoids directly overwriting the
prior committed locator. The renewed review still found three release-blocking
lifecycle failures. One reopens candidate ownership specifically for durable JSON
metadata. Two make the new clear protocol unsafe in the live process: failure to
publish the committed journal is not rolled back, and admission excludes only other
clear/recovery callers rather than normal writes. All three were reproduced against
the current implementation. Validation must remain draft/pending.

The explicitly deferred `STOR-03..STOR-06`, `CACH-03`, `BACK-03`, and `BACK-06`
generalizations were not treated as findings. The findings below are failures of
the narrower Phase 1 candidate and global-clear contracts themselves.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-R1: A durably committed JSON entry is misclassified as uncommitted and its payload is deleted

**Classification:** BLOCKER

**Files:** `src/cacheness/metadata.py:1080-1089`, `src/cacheness/storage/blob_store.py:230-235`, `src/cacheness/core.py:1208-1213`

**Issue:** `JsonBackend._save_to_disk()` has already replaced and directory-fsynced
the live metadata document before it removes the rollback backup. If backup unlink
or the following directory fsync fails, lines 1080-1089 re-read the new live
document and raise. Both high-level writers set `metadata_committed` only after
`put_entry()` returns, so they interpret this post-authority exception as a failed
metadata commit and delete the candidate now referenced by the authoritative JSON
document. A direct overwrite reproduction failed the third metadata-directory
fsync: `put()` raised, `get_entry("k")` pointed to the replacement candidate, and
that candidate no longer existed. The prior payload remained as an unowned file.
The operation leaves durable metadata/payload disagreement and makes the key
unreadable after a reported failure.

**Fix:** Make metadata publication return an explicit outcome that distinguishes
`not_committed`, `committed`, and `uncertain`, or make cleanup after the durable
commit non-throwing/recoverable without reporting publication failure. High-level
candidate ownership must end as soon as the new live JSON document is authoritative,
not only when every backup-retirement barrier returns. For an uncertain outcome,
re-read the authoritative entry and delete only a candidate proven not to be
referenced. Add first-write and cross-format overwrite tests that inject backup
unlink and post-unlink directory-fsync failures through both `BlobStore.put()` and
`UnifiedCache.put()`.

### CR-R2: Failure to publish the committed clear journal leaves a prepared transaction live and later recovery destroys successful writes

**Classification:** BLOCKER

**Files:** `src/cacheness/storage/clear_recovery.py:206-235`, `src/cacheness/storage/blob_store.py:420-440`, `src/cacheness/core.py:1499-1527`

**Issue:** The rollback guard ends immediately after `_clear_backend()` returns.
`_replace_journal()` is outside that guard. If the metadata clear succeeds but the
durable `prepared -> committed` journal replacement fails, `clear()` raises while
all metadata is cleared, originals are absent, tombstones and a prepared journal
remain, and the live store is still usable. A reproduced JSON case then completed
`put("new")` successfully. Recovering the retained prepared journal restored the
old exact metadata snapshot, erased the successful new entry, and left its
candidate payload orphaned. Even without the intervening write, a normal
`Exception` at committed-state publication violates the pre-commit contract because
the failed call does not restore the exact prior visible state.

**Fix:** Treat durable committed-journal publication as part of the pre-commit
region. If `_replace_journal()` fails, immediately run `_rollback_prepared()` and
return only after exact rollback; if rollback cannot complete, mark the coordinator
and owning store poisoned and reject every normal operation until recovery reaches
a terminal state. Add JSON, SQLite, and memory fault tests at committed-journal
serialization, candidate write, replace, and directory-fsync boundaries, plus a
test proving no intervening operation can be accepted while prepared evidence
remains unresolved.

### CR-R3: Normal writes bypass clear admission and can return success while clear deletes their metadata and strands their payload

**Classification:** BLOCKER

**Files:** `src/cacheness/storage/clear_recovery.py:136-159`, `src/cacheness/storage/blob_store.py:151-245,420-440`, `src/cacheness/core.py:1084-1230,1499-1527`

**Issue:** The process/advisory admission lock is acquired only by constructor
recovery and `clear`; neither `put()` implementation participates. Consequently a
clear snapshots its mappings, publishes `prepared`, and can pause before staging
while a same-root `put()` commits a new entry. The clear then stages only its old
snapshot and `clear_all()` deletes the newly committed metadata. A deterministic
threaded reproduction had `BlobStore.put(..., key="raced")` return `"raced"`, the
clear return successfully, `get("raced")` return `None`, and the raced candidate
remain in the root without an owner. The same ordering exists in `UnifiedCache`.
This is not the deferred general generation/CAS model: it is a data-loss hole in
the newly claimed atomic global-clear boundary.

**Fix:** Coordinate every metadata/payload mutation with the clear state machine.
Use a root-scoped reader/writer protocol in-process and a compatible shared/exclusive
OS admission protocol across processes: ordinary puts hold shared admission from
preflight through metadata authority and prior-payload cleanup; clear/recovery holds
exclusive admission for the full transaction. Recheck for unresolved journal state
after admission and before publication. Add deterministic same-instance, two-instance,
and subprocess put-vs-clear tests for JSON and SQLite, asserting a linearizable
outcome with no orphan payload and no successful write that is silently discarded.

## Prior Review History (preserved)

The previous standard review at `2026-08-30T00:32:17Z` reported six blockers:

1. **CR-01:** descriptor-mode staging did not bind publication to the validated regular-file inode.
2. **CR-02:** `BlobStore.put()` overwrote the live deterministic payload before metadata commit.
3. **CR-03:** `UnifiedCache.put()` left candidates after metadata-publication failure.
4. **CR-04:** `UnifiedCache.clear_all()` irreversibly deleted a prefix before metadata clear.
5. **CR-05:** failed clear finalization retained anonymous tombstones without durable recovery identity.
6. **CR-06:** Python integers outside SQLite's signed 64-bit bind range collapsed into an untyped miss.

Plans 01-13 through 01-15 were reviewed as attempted closure of that set. The
renewed findings above supersede the prior release decision but do not erase that
iteration history.

---

_Reviewed: 2026-08-30T03:35:34Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
