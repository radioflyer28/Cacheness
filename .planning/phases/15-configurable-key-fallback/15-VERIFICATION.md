---
status: passed
phase: 15-configurable-key-fallback
requirement_ids: [SEC-02]
verified: 2026-04-06
---

# Phase 15 Verification: Configurable Key Fallback Behavior

## Requirement Coverage

### SEC-02: Configurable key fallback behavior
**Status:** PASSED

| Criterion | Evidence | Status |
|-----------|----------|--------|
| `key_fallback_policy` field in SecurityConfig accepts "raise", "warn", "fallback" | config.py:411 `key_fallback_policy: str = "warn"` | ✅ |
| `raise` policy triggers CacheSecurityError on key write failure | test_raise_mode_raises_on_write_failure passes | ✅ |
| `warn` policy logs warning and falls back to in-memory key | test_warn_mode_logs_warning_on_write_failure passes, security.py logs WARNING | ✅ |
| `fallback` policy silently uses in-memory key | test_fallback_mode_silent_on_write_failure passes | ✅ |
| `raise` policy on corrupt key file | test_raise_mode_on_corrupt_key passes, ERROR logged | ✅ |
| `warn` policy on corrupt key regenerates key | test_warn_mode_on_corrupt_key passes, WARNING logged | ✅ |
| `fallback` policy on corrupt key silently regenerates | test_fallback_mode_on_corrupt_key passes | ✅ |
| Default policy is "warn" | test_default_policy_is_warn passes, config.py default = "warn" | ✅ |
| Invalid policy value raises ValueError | test_invalid_policy_raises_valueerror passes | ✅ |
| Backward-compatible deprecation shim for `raise_on_key_fallback` | test_raise_on_key_fallback_true_maps_to_raise passes, config.py:420 shim in __post_init__ | ✅ |
| New field takes precedence over deprecated field | test_new_field_takes_precedence passes | ✅ |
| Factory passes policy to CacheEntrySigner | test_create_cache_signer_passes_policy passes | ✅ |

## Must-Haves Verification

| Truth | Verified | Evidence |
|-------|----------|---------|
| Three distinct policies exist: raise, warn, fallback | ✅ | config.py:411, 3 dedicated test classes |
| `raise` policy prevents silent key fallback | ✅ | CacheSecurityError raised in test_raise_mode_raises_on_write_failure |
| `warn` logs warning before falling back | ✅ | WARNING logged in test_warn_mode_logs_warning_on_write_failure |
| `fallback` silently recovers | ✅ | test_fallback_mode_silent_on_write_failure — no warning, no error |
| Deprecation shim maps old field to new policy | ✅ | test_raise_on_key_fallback_true_maps_to_raise |
| Default behavior is "warn" (safe default) | ✅ | test_default_policy_is_warn |

## Artifact Verification

| Artifact | Contains | Verified |
|----------|----------|----------|
| src/cacheness/config.py | `key_fallback_policy: str = "warn"` | ✅ (line 411) |
| src/cacheness/config.py | `raise_on_key_fallback` deprecation shim in `__post_init__` | ✅ (line 420) |
| src/cacheness/security.py | Policy enforcement in CacheEntrySigner | ✅ |
| tests/test_key_fallback_policy.py | 13 tests across 4 classes | ✅ |

## Key-Link Verification

| From | To | Pattern | Verified |
|------|----|---------|----------|
| config.py | security.py | `key_fallback_policy` passed to CacheEntrySigner | ✅ |
| core.py | security.py | `create_cache_signer()` propagates policy | ✅ |

## Test Results

- **Targeted tests:** 13 passed, 0 failures (tests/test_key_fallback_policy.py)
- **Regressions:** None

## Score

**6/6 must-haves verified. SEC-02 requirement satisfied.**
