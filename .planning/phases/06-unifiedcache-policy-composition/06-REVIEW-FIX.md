---
phase: 06
fixed_at: 2026-09-09T05:19:35Z
review_path: .planning/phases/06-unifiedcache-policy-composition/06-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 6: Code Review Fix Report

**Fixed at:** 2026-09-09T05:19:35Z  
**Source review:** `.planning/phases/06-unifiedcache-policy-composition/06-REVIEW.md`  
**Iteration:** 1

## Summary

- Findings in scope: 7
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: Stale removal cursor continuation can strand unfinished removal work

**Status:** Fixed — requires human verification  
**Files modified:** `src/cacheness/core.py`, `tests/test_phase6_removal_contract.py`  
**Commit:** `6be21e1`

The facade now returns an explicit restart continuation after a bounded removal
page or stale authority cursor. Resumption begins a fresh bounded scan rather
than attempting to drain a stale page cursor.

### CR-02: Injected BlobStore handler registry is overwritten by UnifiedCache

**Files modified:** `src/cacheness/core.py`, `tests/contracts/test_phase6_topology_policy.py`  
**Commit:** `bf2f5ee`

Injected stores retain their handler registry and `UnifiedCache` uses that
registry. Newly composed stores still receive the cache-owned registry.

### CR-03: Close after a durable commit hides the write receipt

**Status:** Fixed — requires human verification  
**Files modified:** `src/cacheness/core.py`, `tests/test_phase3_local_workflows.py`, `tests/test_phase6_policy_contract.py`  
**Commits:** `877784d`, `a21843d`

Post-commit maintenance uses the internal maintenance start path, so a close
race yields typed, retryable incomplete maintenance while preserving the durable
receipt. The policy contract test now targets that canonical internal seam.

### CR-04: Configuration serializers omit CachePolicy

**Files modified:** `src/cacheness/config.py`, `tests/test_phase6_public_api_contract.py`  
**Commit:** `29978fb`

JSON and YAML configuration serializers now persist `CachePolicyConfig`; both
formats round-trip policy values including an infinite TTL.

### WR-01: Strict projection verifier can falsely render a failing check as PASS

**Files modified:** `tools/verify_phase6_contracts.py`, `tests/test_phase6_contract_verifier.py`  
**Commit:** `335265e`

The strict projection check shares one label constant between execution and
rendering, and its self-test proves a failing result exits nonzero and is not
rendered as PASS.

### WR-02: Decorator/key regressions use removed flat configuration API

**Files modified:** `tests/test_decorators.py`, `tests/test_cache_key_consistency.py`, `tools/verify_phase6_contracts.py`, `tests/test_phase6_contract_verifier.py`  
**Commit:** `dc3e38f`

The regressions use nested `CacheStorageConfig` and their canonical paths. They
are included in the verifier's retained regression manifest.

### WR-03: Obsolete nested size and TTL configuration compete with policy

**Files modified:** `src/cacheness/config.py`, `tests/test_config_validation.py`, `tests/test_phase6_public_api_contract.py`  
**Commit:** `2b3e4f1`

Removed unconsumed storage/metadata policy toggles and their validation. TTL and
authoritative-size settings are accepted only through `CachePolicyConfig`, and
configuration round-trip coverage was updated accordingly.

## Verification

All verification ran in the **main checkout** at
`/Users/akriz/code/cacheness` (the parent workflow explicitly requested no
isolated worktree).

- Focused checks passed for every finding, including stale-removal restart,
  injected custom-handler retention, post-close receipt preservation, policy
  JSON/YAML round-trips, strict-projection rendering, and canonical
  decorator/key regressions.
- Consolidated Phase 6 and directly affected regressions passed:
  `tests/test_phase6_lookup_contract.py`, `tests/test_phase6_removal_contract.py`,
  `tests/test_phase6_statistics.py`, `tests/test_phase6_decorator_contract.py`,
  `tests/test_phase6_public_api_contract.py`, `tests/test_phase6_policy_contract.py`,
  `tests/contracts/test_phase6_topology_policy.py`,
  `tests/test_phase6_contract_verifier.py`, `tests/test_phase6_examples.py`,
  `tests/test_decorators.py`, `tests/test_cache_key_consistency.py`, and
  `tests/test_phase3_local_workflows.py`. One Pandas-dependent key-consistency
  test was skipped because Pandas is not installed.
- `uv run --frozen ruff check` over every touched source and test path passed.
- `uv run --frozen python tools/verify_phase6_contracts.py --repo-root .`
  passed CACH-01 through CACH-06, canonical decorator/key regressions, and
  strict projection rejection. It exited 1 only for CACH-07 because
  `tests/test_sql_cache.py` cannot collect without the unavailable optional
  Pandas dependency; this is unrelated to these seven findings.
- The legacy `tests/test_config_validation.py` module still has an unrelated
  pre-existing collection failure from importing removed `CacheBlobConfig`.
  The WR-03 behavior was verified instead through the canonical public API
  contract test and a focused runtime configuration check.

---

_Fixed: 2026-09-09T05:19:35Z_  
_Fixer: the agent (gsd-code-fixer)_  
_Iteration: 1_
