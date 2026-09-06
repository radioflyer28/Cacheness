---
phase: 03-atomic-lifecycle-and-recovery-engine
reviewed: 2026-09-06T04:50:34Z
depth: deep
files_reviewed: 12
files_reviewed_list:
  - src/cacheness/core.py
  - src/cacheness/metadata.py
  - src/cacheness/storage/backends/postgresql_backend.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/lifecycle.py
  - src/cacheness/storage/lifecycle_authority.py
  - src/cacheness/storage/memory_lifecycle_authority.py
  - src/cacheness/storage/sqlite_lifecycle_authority.py
  - tests/fixtures/phase3_ruff_baseline.json
  - tests/test_projection_mutation_contract.py
  - tests/test_unified_cache_lifecycle_authority.py
  - tools/verify_phase3_ruff_delta.py
findings:
  critical: 9
  warning: 2
  info: 0
  total: 11
status: issues_found
---

# Phase 03: Code Review Report

**Reviewed:** 2026-09-06T04:50:34Z
**Depth:** deep
**Files Reviewed:** 12
**Status:** issues_found

## Summary

The Plan 03-14 implementation closes several previously reported visibility gaps, and the two focused test modules pass under Python 3.11 (25 tests). However, deep call-chain review found nine ship-blocking correctness, integrity, concurrency, compatibility, and fail-closed defects. The most consequential problems are that `UnifiedCache.put()` bypasses `BlobStore` admission, projection/link state is destructively changed before authority promotion, the retry path can overwrite another operation's pending projection, and SQL projection compare-and-mutate is not atomic across independent adapters. The deterministic tests do not exercise those interleavings.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: UnifiedCache mutation bypasses BlobStore operation admission [BLOCKER]

**File:** `src/cacheness/storage/blob_store.py:96-103,309-335`; `src/cacheness/core.py:1730-1747`

**Issue:** Public `BlobStore.put()` is protected by `_ordinary_admitted`, which checks the canonical store and enters `InstanceAdmission.operation()`. The new result-returning `_put_with_result()` seam has no equivalent admission, and `UnifiedCache.put()` calls that private method directly. A facade put is therefore invisible to the admission counter. `BlobStore.close()` can observe zero in-flight operations and close the authority/guarded I/O while the facade put is still running, and `BlobStore.clear()` does not exclude that mutation. This reintroduces lifecycle races at the exact composition seam Plan 03-14 added.

**Fix:** Put admission around the result-returning entry point and separate the already-admitted implementation to avoid double admission. For example:

```python
def _put_with_result(self, *args, **kwargs):
    self._require_canonical_store()
    with self._instance_admission.operation():
        return self._put_with_result_admitted(*args, **kwargs)

@_ordinary_admitted
def put(self, *args, **kwargs):
    return self._put_with_result_admitted(*args, **kwargs).locator
```

Add a barrier test that pauses a facade put after admission and proves `close()`/`clear()` cannot pass it.

### CR-02: A failed overwrite permanently deletes the previous generation's custom-metadata links [BLOCKER]

**File:** `src/cacheness/core.py:1563-1594,1690-1708`; `src/cacheness/storage/lifecycle.py:258-289`; `src/cacheness/metadata.py:1941-1950,2001-2005`; `src/cacheness/storage/backends/postgresql_backend.py:321-333,384-387`

**Issue:** The lifecycle hook publishes the candidate projection before signing/verification and authority promotion. Both SQL adapters delete all custom-metadata links when the projected locator changes, and commit that deletion with the candidate projection. If signing, verification, or promotion then fails, `_repair_projection_after_failed_put()` can restore the M1 projection but cannot reconstruct the deleted M1 link rows. The failed M2 operation thus destroys user-visible custom metadata belonging to the still-authoritative M1 generation. Current failure coverage starts from an absent key and never proves link preservation on failed replacement.

**Fix:** Do not perform destructive link ownership transition before authority promotion. Stage the candidate projection without retiring old links, or compute/sign the projection first and publish projection plus link transition only after exact promotion succeeds. The post-promotion transaction must condition deletion/insertion on the promoted locator/token. Add a failed-overwrite regression beginning with an M1 entry that has custom metadata and assert its links survive every pre-promotion failure hook.

### CR-03: Projection retry can steal another operation's pending candidate token [BLOCKER]

**File:** `src/cacheness/core.py:1563-1587`

**Issue:** When the initial projection compare-and-mutate mismatches, `_prepare_authority_projection()` checks that the authority expectation is still current and then replaces its expected projection token with whatever row is now visible. That row can be a competing operation's pre-promotion candidate, not a committed projection. In a forced schedule, B installs its candidate projection and pauses before promotion; A observes B's candidate, adopts B's locator as its expectation, and overwrites the row. B can then win authority promotion but fail its post-promotion custom-link write because A replaced its projection. A later loses promotion and repairs. A successful canonical B mutation is reported as failed and its custom metadata is lost.

**Fix:** An operation must never substitute an observed peer token for its captured projection token. A mismatch must remain a typed conflict/no-op. If a repair/retry is required, prove that the observed row corresponds to the same committed authority snapshot and is not a pending candidate, preferably by persisting an operation token/state in the projection. Add a barrier test with two overlapping pre-promotion candidates; the existing replacement test lets the peer complete promotion before the hook and does not cover this case.

### CR-04: SQL projection compare-and-mutate is not atomic across independent adapters [BLOCKER]

**File:** `src/cacheness/metadata.py:1967-2007`; `src/cacheness/storage/backends/postgresql_backend.py:351-387`

**Issue:** SQLite protects a SELECT-then-ORM-mutate sequence only with a per-instance Python lock. Independent `SqliteBackend` objects and processes do not share that lock, so two writers can both observe the same locator/absence and then race. The loser may receive `database is locked`/a unique-key error, or an ORM update keyed only by primary key can replace a newer row; it is not guaranteed to receive the contract's mismatch/no-op result. PostgreSQL uses `FOR UPDATE` for existing rows, but expected absence locks no row: two creators can both observe absence and one receives `IntegrityError` instead of an exact conflict result. The tests are sequential for both SQL adapters and therefore cannot establish the claimed CAS contract.

**Fix:** For SQLite, acquire `BEGIN IMMEDIATE` before the comparison or use one conditional SQL mutation whose affected-row count defines success. For PostgreSQL, serialize each cache key with an advisory lock, or use `INSERT ... ON CONFLICT` plus a conditional update and deterministic result classification. Convert expected contention into the typed lifecycle conflict/no-op result. Add deterministic tests using independent SQLite connections and real PostgreSQL dialect semantics for both absent and existing rows.

### CR-05: Fresh-root SQLite bootstrap still races across authority instances and processes [BLOCKER]

**File:** `src/cacheness/storage/sqlite_lifecycle_authority.py:80-82,400-406,433-434`

**Issue:** `_bootstrap_lock` is owned by one authority instance. Two separately constructed authorities/processes can both classify a root as missing. The winner creates the directory; the loser executes `mkdir(..., exist_ok=False)` and raises `FileExistsError` without reclassifying/joining the winner. This affects independently constructed facades performing their first mutations, including distinct keys. Existing process tests seed the authority first, and the same-facade test shares one authority and lock, so neither covers this bootstrap race.

**Fix:** Catch `FileExistsError` at root creation and immediately repeat the exact safe classification. Reject wrong-object/symlink states and join only a bounded recognized bootstrap state before continuing the O_EXCL leaf protocol. Add fresh-root tests with two authority instances and two processes, without pre-seeding the database.

### CR-06: Cached SQL metadata wrappers silently disable custom-metadata APIs [BLOCKER]

**File:** `src/cacheness/metadata.py:477-498,2505-2512`; `src/cacheness/core.py:387-464,567-595,637-638`

**Issue:** When memory caching is enabled, `create_metadata_backend()` wraps SQLite/PostgreSQL in `CachedMetadataBackend`. The wrapper delegates projection mutation but exposes neither `store_custom_metadata_if_current` nor the wrapped backend's `SessionLocal`/`engine`. Core therefore reports the underlying SQL backend as custom-metadata-capable, but `put(..., custom_metadata=...)` silently inserts no link, `query_custom()` cannot query it, and `query_custom_session()` rejects the backend. This is a supported configuration and a public API compatibility failure.

**Fix:** Give `CachedMetadataBackend` explicit custom-metadata delegation methods (including exact-current checks and cache invalidation) and a safe query/session abstraction, rather than relying on attribute sniffing. Add SQLite and PostgreSQL parity tests with `enable_memory_cache=True` covering put, get, query, stale-link rejection, and replacement ownership.

### CR-07: Empty committed-key snapshots make clear/invalidate delete an in-flight canonical generation as “legacy” [BLOCKER]

**File:** `src/cacheness/core.py:2027-2031,2273-2289,2294-2318`

**Issue:** `clear_all()` treats an empty `BlobStore.list()` result as permission to run compatibility cleanup directly against every projection row. A first put can already have installed its candidate projection while authority still has no committed key. `clear_all()` then deletes that candidate payload and row as “legacy”; the put can subsequently promote the already-verified manifest and return success, leaving authority committed to a missing payload. The analogous absent-snapshot/key-only fallback in `invalidate()` and `_retire_exact_authority_snapshot()` can also delete a newer projection. The Plan 03-14 facade mutation is not admitted (CR-01), but the momentary-empty heuristic is unsafe across separate facades even after admission is fixed.

**Fix:** Enter legacy cleanup only when the store was explicitly classified as a recognized legacy backend, never because the committed key list happens to be empty. Canonical stores must invoke authority clear semantics even for an empty snapshot. Projection teardown must always use an exact locator/token compare-and-mutate. Add an empty-store `clear_all()` versus first-put barrier schedule and the corresponding invalidation schedule.

### CR-08: Same-key put overwrites an unsafe persisted locator without fail-closed containment validation [BLOCKER]

**File:** `src/cacheness/core.py:1563-1569,1725-1728`

**Issue:** `UnifiedCache.put()` reads the current projection's locator and passes it directly as the CAS expectation. It does not run `_entry_locator()` containment validation before payload/projection mutation. A tampered same-key projection whose `actual_path` points outside the managed root is therefore silently replaced (and its links can be deleted) instead of blocking the mutation at the integrity boundary. The code does not need to open the external path for this to violate the project's fail-closed rule: it destroys evidence and accepts malformed persisted state as a valid mutation token.

**Fix:** Validate every observed projection locator, including nested metadata forms, with `_entry_locator(..., operation="put")` before writing a payload or mutating projection state. On failure, raise the domain integrity/path exception without modifying authority, projection, links, or payloads. Add same-key hostile top-level and nested-locator tests; unrelated-entry size preflight does not cover this path.

### CR-09: PostgreSQL returns cache-key parameters at the wrong nesting and invalidates correctly signed entries [BLOCKER]

**File:** `src/cacheness/storage/backends/postgresql_backend.py:576-581`; `src/cacheness/core.py:1547-1553,2010-2017`

**Issue:** Projection signing reads `metadata["cache_key_params"]`. PostgreSQL persists that value, but `_entry_to_dict()` restores parsed parameters at the result's top level instead of inside `metadata`, unlike the other backends. Signature authorization later reads only the nested metadata value, omits the parameters, and rejects a valid signature whenever `store_cache_key_params=True`. With the default invalid-signature policy, a read can retire the valid canonical entry.

**Fix:** Restore parsed `cache_key_params` into the nested metadata mapping used at signing, consistently with SQLite/JSON/in-memory. Preserve any compatibility top-level field only as an alias. Add an end-to-end PostgreSQL parity test with signing and stored cache-key parameters enabled, verifying get/list/stats do not retire the entry.

## Warnings

### WR-01: SqliteBackend destructor can raise during interpreter shutdown [WARNING]

**File:** `src/cacheness/metadata.py:2397-2409`

**Issue:** `__del__()` calls `close()` directly, while `close()` performs a late `import gc`, collection, and logging. During interpreter finalization, imports and module globals may already be unavailable, producing the observed `Exception ignored in: SqliteBackend.__del__` after an otherwise-green suite. Destructors must not surface exceptions, and noisy finalization can hide real test/process failures.

**Fix:** Prefer explicit/context-managed close and remove the destructor. If compatibility requires it, make it a minimal best-effort guard that catches `BaseException`, does not import or log during finalization, and nulls/disposes the engine idempotently.

### WR-02: Deterministic regression coverage does not exercise the critical cross-operation schedules [WARNING]

**File:** `tests/test_projection_mutation_contract.py:1-275`; `tests/test_unified_cache_lifecycle_authority.py:1-634`

**Issue:** SQL projection contract tests are sequential; only JSON gets a cross-instance race. The “PostgreSQL” helper runs on SQLite and tests `_store_custom_metadata()` directly, while the core dispatch test only proves a method call, not real `UnifiedCache.put(..., custom_metadata=...)` transaction behavior. Failure coverage starts from an absent key rather than an M1 generation with links; replacement-before-projection lets the peer finish promotion before the hook; and distinct-key coverage uses one facade/authority. The suite therefore passes without forcing CR-01 through CR-07.

**Fix:** Add hook/barrier tests for admitted close/clear, two pending candidate projections, failed overwrite with old links, empty-store clear versus first promotion, independent SQLite adapters/fresh authorities, and cached-wrapper custom metadata. Exercise PostgreSQL-specific SQL/result classification with a real service where available; the required deterministic core-to-adapter path should not be replaced by direct private-method calls.

---

_Reviewed: 2026-09-06T04:50:34Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: deep_
