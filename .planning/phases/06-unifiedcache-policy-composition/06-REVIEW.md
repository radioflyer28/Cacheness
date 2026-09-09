---
phase: 06-unifiedcache-policy-composition
reviewed: 2026-09-09T05:29:03Z
depth: standard
files_reviewed: 25
files_reviewed_list:
  - src/cacheness/__init__.py
  - src/cacheness/cache_policy.py
  - src/cacheness/config.py
  - src/cacheness/core.py
  - src/cacheness/decorators.py
  - src/cacheness/storage/blob_store.py
  - src/cacheness/storage/catalog.py
  - docs/CACHE_POLICY.md
  - examples/simple_object_caching.py
  - examples/configurable_serialization_demo.py
  - examples/api_request_caching.py
  - tools/verify_phase6_contracts.py
  - tests/test_phase6_lookup_contract.py
  - tests/test_phase6_statistics.py
  - tests/test_phase6_removal_contract.py
  - tests/test_phase6_policy_contract.py
  - tests/test_phase6_decorator_contract.py
  - tests/contracts/test_phase6_topology_policy.py
  - tests/test_phase6_public_api_contract.py
  - tests/test_phase6_examples.py
  - tests/test_phase6_contract_verifier.py
  - tests/test_decorators.py
  - tests/test_cache_key_consistency.py
  - tests/test_phase3_local_workflows.py
  - tests/test_config_validation.py
findings:
  critical: 4
  warning: 0
  info: 0
  total: 4
status: issues_found
---

# Phase 6: Code Review Report

**Reviewed:** 2026-09-09T05:29:03Z
**Depth:** standard
**Files Reviewed:** 25
**Status:** issues_found

## Summary

Fix iteration 1 fully resolves prior CR-04 and WR-01 through WR-03, and the
public predicate/global/function continuation and custom injected-handler cases
from prior CR-01/CR-02 now behave correctly. Prior CR-03 is resolved for a
cache-owned store, but the same close-after-commit boundary remains incorrect
for a caller-owned injected store. The stale-cursor defect also remains in the
bounded TTL cleanup path, direct `BlobStore` construction still discards its
configured handler policy, and the Phase 6 cutover leaves the required full
suite unable to collect because current repository tests still import removed
development APIs.

The focused Phase 6 and directly affected regression scope passes when the
uncollectable `tests/test_config_validation.py` module is excluded (one
Pandas-dependent test skips). Scoped Ruff and the verifier self-tests pass. The
fixed verifier reports CACH-01 through CACH-06, canonical decorator/key
regressions, and strict projection rejection as PASS; its only fixed-manifest
failure is the missing-`pandas` CACH-07 gate, which remains Phase 8 environment
qualification. Repository-wide collection has 17 errors: three are that
missing optional package, while fourteen are stale cutover imports or tests and
are Phase 6 gaps.

All recommended fixes preserve ADR 0001. They require no lock, queue,
coordinator, readiness mechanism, compatibility restoration, cross-resource
ACID claim, or timing/progress guarantee.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Bounded TTL cleanup still returns a cursor invalidated by its own deletion

**Classification:** BLOCKER

**File:** `src/cacheness/core.py:417-445`

**Issue:** Fix `6be21e1` restarts public predicate/global/function invalidation,
but `_cleanup_expired()` still returns `page.cursor` after deleting an expired
entry from that page. The successful delete advances the catalog revision, so
passing the reported continuation into the next bounded cleanup raises
`CacheCatalogStaleCursorError` instead of making progress. A memory/memory
reproduction with three expired entries and `page_size=work_cap=1` removes one
entry, returns `complete=False` with an authority cursor, then fails on the
second call. This violates the Phase 6 requirement that TTL cleanup share the
bounded resumable removal semantics.

**Fix:** Apply the same restart-continuation rule used by
`invalidate_where()`: consume the cache-policy restart token as a fresh scan,
translate stale authority cursors to typed restart evidence, and replace a
post-delete non-exhausted authority cursor with the restart token. Add a
three-page `_cleanup_expired()` regression that follows every returned
continuation to completion.

### CR-02: Close-after-commit still ignores facade closure for an injected store

**Classification:** BLOCKER

**File:** `src/cacheness/core.py:789-803`

**Issue:** `_start_size_maintenance()` bypasses the facade close guard and relies
on the underlying store to raise `CacheBlobStoreClosedError`. That works for the
cache-owned topology tested by `test_close_after_commit_has_declared_derived_outcome`,
because `cache.close()` closes that store. It does not work for a caller-owned
injected `BlobStore`, which correctly remains open. If `put_entry()` commits and
then closes the facade, `cache.put()` continues catalog maintenance on the open
store and returns an ordinary continuation with `cause=None`, rather than the
typed close restart promised by the fix. The receipt is retained, but the
derived result does not truthfully report the concurrent facade close.

**Fix:** At the internal post-commit boundary, check the facade's closed state
and return `_maintenance_restart(CacheBlobStoreClosedError("Cache is closed"))`
before policy I/O. Keep the receipt unchanged. Add the same deterministic
commit-then-close test for an injected, caller-owned store and assert that the
store remains usable afterward.

### CR-03: Direct BlobStore construction discards configured handler policy

**Classification:** BLOCKER

**File:** `src/cacheness/storage/blob_store.py:206-217`

**Issue:** `BlobStore` retains the supplied `CacheConfig` as `self.config`, but
constructs `HandlerRegistry()` without it. Consequently handler enablement,
priority, and trusted-object-array policy do not describe the direct store. A
reproduction using a config that disables every built-in handler still creates
an `array` and `object` registry and successfully pickles a dictionary. The
topology-created `UnifiedCache` masks this by replacing the registry, and the
new injected-store regression masks it by manually registering a custom
handler, so prior CR-02's identity test does not cover normal config-driven
injection.

**Fix:** Construct `HandlerRegistry(self.config)` inside `BlobStore` and let
`UnifiedCache` preserve that registry for injected stores. Add direct-store and
injected-cache tests proving disabled handlers remain unavailable and an
explicit priority order is retained.

### CR-04: The public cutover leaves fourteen current test modules uncollectable

**Classification:** BLOCKER

**File:** `tests/test_config_validation.py:13-26`

**Issue:** This directly modified regression module still imports removed
`CacheBlobConfig` and `create_cache_config`, then spends most of the file
asserting the removed flat/blob/factory compatibility surface. It therefore
cannot collect. Repository-wide `pytest --collect-only` reports fourteen
cutover-related collection errors (including this module, legacy `cacheness`
alias users, and removed top-level handler/config exports) plus three separate
missing-`pandas` errors. The latter are Phase 8 environment qualification; the
fourteen stale API errors are Phase 6's required current-suite cutover gap and
prevent retained integrity/serialization tests from running at all. Updating
two assertions for WR-03 did not make this changed test file executable.

**Fix:** Migrate the fourteen stale modules to the canonical nested
`CacheConfig`, explicit `StoreTopology`/`BlobStore`, `UnifiedCache`, typed result,
and explicit decorator APIs, deleting only assertions whose sole purpose was
compatibility. Do not restore removed names. At minimum, fully migrate
`tests/test_config_validation.py` in the same change that edits its policy
assertions, then require repository-wide collection to fail only for explicitly
qualified optional-package gates.

---

_Reviewed: 2026-09-09T05:29:03Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
