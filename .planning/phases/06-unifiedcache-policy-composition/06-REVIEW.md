---
phase: 06-unifiedcache-policy-composition
reviewed: 2026-09-09T04:55:46Z
depth: standard
files_reviewed: 24
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
findings:
  critical: 4
  warning: 3
  info: 0
  total: 7
status: issues_found
---

# Phase 6: Code Review Report

**Reviewed:** 2026-09-09T04:55:46Z
**Depth:** standard
**Files Reviewed:** 24
**Status:** issues_found

## Summary

The Phase 6 policy surface has four ship-blocking correctness defects. Bounded
predicate/global/function removal cannot resume after deleting its first page;
injected caller-owned stores have their handler registry overwritten; a close
after canonical commit can discard the caller-visible receipt; and JSON/YAML
configuration saves omit the entire cache-policy section. The fixed verifier
also prints a false-green projection status, and the submitted regression scope
contains six failures caused by stale flat-configuration calls. These are Phase
6 defects, not requests to restore the intentionally removed compatibility API
and not Phase 8 packaging, platform, or live-service gates.

All fixes below preserve ADR 0001: no new lock, queue, coordinator, source of
lifecycle truth, cross-resource transaction, or stronger progress guarantee is
recommended. Exact mutations remain exclusively in `BlobStore` and its shared
`AuthorityLifecycleEngine`.

## Narrative Findings (AI reviewer)

## Critical Issues

### CR-01: Removal continuations are invalidated by the removal they just performed

**File:** `src/cacheness/core.py:1017-1039`

**Issue:** `invalidate_where()` returns `page.cursor` after deleting entries from
that page. Every successful exact deletion advances the authoritative catalog
revision, so the revision-bound cursor is stale before it reaches the caller.
On resume, the method catches `CatalogStaleCursorError` and returns the same
stale cursor unchanged. Repeating the documented continuation therefore makes
zero progress forever. This affects multi-page predicate invalidation,
`clear_all()`, and decorator `cache_clear()`. The test at
`tests/test_phase6_removal_contract.py:209-236` institutionalizes the defect by
calling the operation “resumable” while asserting only a zero-work retryable
result after the first page.

**Fix:** Treat mutation-invalidated catalog cursors as restart evidence, not as
continuations. After each bounded exact-delete page, return a policy continuation
that directs the next call to begin a fresh bounded query from the start (already
removed entries are absent, and exact expectations protect replacements), or
return a typed restart cause and require an explicit fresh call. Never return
the known-stale authority cursor as the next continuation. Add a three-page test
that repeatedly follows the public retry protocol and proves convergence without
duplicate successful accounting.

### CR-02: Injecting a caller-owned BlobStore silently replaces its handler registry

**File:** `src/cacheness/core.py:142-161`

**Issue:** The constructor correctly labels an injected `BlobStore` as
caller-owned, then immediately assigns `self.handlers` into `store.handlers`.
The same mutation is repeated before puts and lookups at
`src/cacheness/core.py:828` and `src/cacheness/core.py:958`. This discards custom
handler registration and changes how the still-caller-owned store reads and
writes direct objects, including after `UnifiedCache.close()`. It can make
previously stored values unreadable or select a different serializer. Current
ownership tests only prove that initialize/close are not delegated; they do not
verify that the injected store remains behaviorally unchanged.

**Fix:** Do not replace state on an injected store. Use the injected
`BlobStore.handlers` as the storage handler registry and preflight any required
cache configuration compatibility before policy operations. For the
topology-created form, pass/configure the intended registry during store
construction. Add an injected custom-handler test that checks registry identity
and direct-store behavior before cache construction, during cache use, and after
facade close.

### CR-03: A close after commit hides the canonical receipt instead of returning committed-partial truth

**File:** `src/cacheness/core.py:829-837`

**Issue:** `_put_for_cache_key()` obtains a committed `BlobReceipt` and then calls
the close-guarded public `maintain_size()`. If the facade closes between those
steps, `maintain_size()` raises `CacheBlobStoreClosedError` and `put()` exits
without returning the receipt even though the generation is committed. The
regression at `tests/test_phase3_local_workflows.py:224-243` explicitly expects
this lost-acknowledgement behavior, contradicting D-17 and the public
`CachePutResult` contract that post-commit policy work cannot falsify storage
truth.

**Fix:** Once `put_entry()` returns, always preserve that receipt. Run the one
bounded maintenance step through an internal path that can translate a concurrent
facade close into an incomplete retryable `CacheMaintenanceResult` with the typed
cause, then return `CachePutResult(receipt, maintenance)`. Do not add close
coordination or another lifecycle sequencer. Update the close-after-commit test
to assert the returned receipt and typed maintenance outcome rather than an
exception that discards acknowledgement.

### CR-04: Config save/load silently resets all cache-policy settings

**File:** `src/cacheness/config.py:908-917`

**Issue:** Both `save_config_to_json()` and `save_config_to_yaml()` (the latter at
`src/cacheness/config.py:951-960`) omit `config.policy`, although
`load_config_from_dict()` accepts it. A round trip silently resets TTL, maximum
authoritative bytes, page size, work cap, and continuation-size bounds to
defaults. A reproduced configuration with `default_ttl_hours=None` and a 7-byte
limit loaded back as 24 hours and 2,097,152,000 bytes. This changes expiration
and eviction behavior without an error.

**Fix:** Include `"policy": asdict(config.policy)` in both serialized mappings
and add JSON and YAML round-trip tests using non-default values for every
`CachePolicyConfig` field.

## Warnings

### WR-01: The fixed verifier prints PASS when the strict projection node fails

**File:** `tools/verify_phase6_contracts.py:367-405`

**Issue:** The projection pytest run records errors with the label
`SC-06 strict incompatible projection rejection`, but `main()` searches for
`SC-06 strict projection rejection`. A failing strict projection test therefore
still prints `SC-06 strict projection rejection: PASS` even though the process
eventually exits nonzero. This is false-green human-readable evidence, and the
verifier self-tests do not exercise the status rendering path.

**Fix:** Define one label constant and use it for both `_run_pytest()` and the
status-prefix check. Add a self-test that injects a failing projection result and
asserts both nonzero exit and a non-PASS projection status.

### WR-02: Submitted decorator/key regression files still use the intentionally removed flat config API

**File:** `tests/test_decorators.py:10-20`

**Issue:** The helper calls `CacheConfig(cache_dir=tmp_path)`, which Phase 6
intentionally removed. `tests/test_cache_key_consistency.py:27-38`,
`tests/test_cache_key_consistency.py:234-260`, and
`tests/test_cache_key_consistency.py:262-280` repeat the stale form and also read
the removed `config.cache_dir` attribute. Running the complete submitted review
scope produces six failures. The fixed verifier does not include these changed
regression files, so its CACH-01 through CACH-06 results do not expose the stale
tests.

**Fix:** Migrate these tests to
`CacheConfig(storage=CacheStorageConfig(cache_dir=...))` and use
`config.storage.cache_dir`. Do not restore the removed flat constructor or
attribute. Add the migrated decorator/key nodes to the fixed retained regression
manifest so the verifier cannot pass while these changed tests fail.

### WR-03: Two obsolete size/TTL knobs remain in other config sections and are silently ignored by policy

**File:** `src/cacheness/config.py:36-71`

**Issue:** `CacheStorageConfig.max_cache_size_mb` and
`CacheMetadataConfig.default_ttl_hours` remain public and are serialized and
validated, but Phase 6 policy reads only
`CachePolicyConfig.max_authoritative_bytes` and
`CachePolicyConfig.default_ttl_hours`. Users can supply valid-looking nested
settings that have no effect, leaving two competing vocabularies despite the
documented single ownership-aligned policy section. This is distinct from the
approved removal of flat compatibility arguments: the misleading nested fields
are still accepted as current configuration.

**Fix:** Remove the dead pre-production fields (and other retired policy toggles
that no runtime path consumes), or reject them explicitly during config loading.
Keep TTL and authoritative size solely in `CachePolicyConfig`; update config
serialization/tests accordingly rather than adding a compatibility mapping.

---

_Reviewed: 2026-09-09T04:55:46Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
