# Phase 15: Configurable Key Fallback — Research

**Researched:** 2026-04-03
**Domain:** Python config patterns, deprecation strategies, key management behavior

## Standard Stack

No external libraries needed. Uses only:
- `dataclasses` (already used for `SecurityConfig`)
- `warnings` module (stdlib, for `DeprecationWarning`)
- `logging` module (already used throughout)

## Architecture & Patterns

### Current Implementation Map

All references to `raise_on_key_fallback` in production code:

| File | Location | Role |
|------|----------|------|
| `config.py` L408 | `SecurityConfig.raise_on_key_fallback: bool = False` | Config field |
| `config.py` L413-417 | `__post_init__` debug log | Logging |
| `security.py` L85 | `CacheEntrySigner.__init__` param | Constructor param |
| `security.py` L99 | `self.raise_on_key_fallback = raise_on_key_fallback` | Instance attr |
| `security.py` L157 | `if self.raise_on_key_fallback:` in `_generate_new_key` | Decision point |
| `security.py` L451 | `create_cache_signer()` factory param | Factory param |
| `security.py` L467 | Passed to `CacheEntrySigner()` constructor | Passthrough |
| `core.py` L339 | `_init_entry_signer()` passes `config.security.raise_on_key_fallback` | Passthrough |
| `blob_store.py` L113 | `BlobStore.__init__` param | Constructor param |
| `blob_store.py` L204 | `_init_signer()` call passes it | Passthrough |
| `blob_store.py` L231 | `_init_signer()` method param | Method param |
| `blob_store.py` L241 | Passed to `create_cache_signer()` | Passthrough |

Test references:
| File | Location | Role |
|------|----------|------|
| `test_key_rotation.py` L118-143 | `test_raise_on_key_fallback` | Tests raise mode |
| `test_key_rotation.py` L145-158 | `test_no_raise_on_key_fallback_by_default` | Tests silent fallback |

**No references** in examples/, docs/SECURITY.md, or `__init__.py`.

### Key Decision Points

**`_generate_new_key()` (security.py L133-167):** The ONLY place where fallback behavior diverges. On `OSError` when writing key file:
- Current: `raise_on_key_fallback=True` → raise `CacheSecurityError`; `False` → log error + warning, return in-memory key

**`_load_or_generate_key()` (security.py L109-131):** Handles corrupt key (wrong length). Currently always logs warning and regenerates. This should also respect the policy per D-07.

**`_init_entry_signer()` (core.py L329-353):** Catches `(OSError, ValueError)` — does NOT catch `CacheSecurityError`. So in "raise" mode, `CacheSecurityError` from `_generate_new_key()` will propagate through `create_cache_signer()` → through `_init_entry_signer()` → into `UnifiedCache.__init__()` → to the caller. This is the correct behavior.

### BlobStore Integration (MISSED IN ORIGINAL PLAN)

`BlobStore` has its own `raise_on_key_fallback` parameter and `_init_signer()` method (blob_store.py L231-250). This is a **separate code path** from `UnifiedCache._init_entry_signer()`. It must also be updated to use `key_fallback_policy`.

The `BlobStore._init_signer()` catches `Exception` (intentionally broad) and sets `self.signer = None`. So even in "raise" mode, `CacheSecurityError` would be caught here. However, this is the BlobStore's standalone storage mode — it has different error-handling semantics from the decorator mode. The existing behavior (catch-all) should be preserved.

### Deprecation Strategy

Python dataclass fields don't track "was this explicitly set?". With `raise_on_key_fallback: bool = False`, there's no way in `__post_init__` to know if the user wrote `SecurityConfig(raise_on_key_fallback=False)` or just got the default.

**Practical approach (from CONTEXT.md D-02):** Only shim `raise_on_key_fallback=True` because it's a non-default value proving explicit user intent. The `False` case is ambiguous (could be default), so we upgrade those users to the new `"warn"` default — which is intentionally better than the old silent behavior.

For `BlobStore.__init__`, the same principle applies: `raise_on_key_fallback=True` users get mapped to `key_fallback_policy="raise"`.

## Don't Hand-Roll

- Use `warnings.warn()` with `DeprecationWarning` (standard Python pattern)
- Use runtime validation in `__post_init__` (not `Literal` type — less friction for dict-based config)
- Keep the deprecated field in `SecurityConfig` (don't remove — breaking change)

## Common Pitfalls

1. **Forgetting BlobStore:** It has its own `raise_on_key_fallback` param separate from `SecurityConfig` → must update `BlobStore.__init__`, `BlobStore._init_signer()`, and the factory call inside
2. **Deprecation detection:** Can't detect explicit `raise_on_key_fallback=False` in a dataclass. Only shim `True` → `"raise"`
3. **Error propagation in core.py:** `CacheSecurityError` is NOT caught by `_init_entry_signer()` — it propagates to the caller. This is correct for "raise" mode.
4. **Error swallowing in blob_store.py:** `_init_signer()` catches `Exception` (intentionally broad). In "raise" mode, `CacheSecurityError` is still caught here. This matches BlobStore's design as a best-effort storage layer.
5. **Default behavior change:** Old default was silent fallback. New default (`"warn"`) logs a WARNING. This is intentional but should be documented clearly.

## Validation Architecture

### Behavioral Tests
- Each mode (raise/warn/fallback) × each trigger (write failure, corrupt key)
- Deprecation shim: `raise_on_key_fallback=True` → `key_fallback_policy="raise"` with warning
- Invalid policy value → `ValueError`
- Default policy is `"warn"`

### Integration Tests
- `UnifiedCache` with `key_fallback_policy="raise"` + write failure → `CacheSecurityError` propagates
- `BlobStore` with `key_fallback_policy="raise"` + write failure → caught by `_init_signer` catch-all

---
*Research complete — proceed to planning*
