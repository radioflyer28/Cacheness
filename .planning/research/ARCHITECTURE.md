# Architecture Patterns — Cleanup & Hardening Integration

**Domain:** Python disk caching library refactoring
**Researched:** 2026-04-02
**Overall confidence:** HIGH (all analysis based on direct codebase inspection)

## Executive Summary

The Cacheness cleanup & hardening milestone involves six categories of change: three structural decompositions (core.py, metadata.py, handlers.py), security hardening, error handling narrowing, and test gap coverage. These changes interact with the existing layered architecture at well-defined integration points, and most can proceed independently after establishing the module split infrastructure.

The critical insight is that **the three decompositions are purely structural refactors** — they move code between files without changing any runtime behavior, data flow, or public API. This makes them low-risk individually but high-coordination as a group (git conflicts, import chain updates). The security and error handling changes are behavioral modifications that are safer to land *after* the structural refactors are complete, because narrower files are easier to audit and test.

The recommended build order is: handlers.py split (smallest, lowest risk) → metadata.py split (medium, well-bounded) → core.py decomposition (largest, highest coordination) → error handling narrowing (behavioral, needs stable code) → security hardening (behavioral, new features). Test gap coverage runs in parallel throughout.

## Current Architecture State

### Module Sizes (Lines of Code)

| Module | Lines | Classes | Methods | Role |
|--------|-------|---------|---------|------|
| `core.py` | 3,900 | 2 (`_PutCleanup`, `UnifiedCache`) | 85+ | Central coordinator |
| `metadata.py` | 2,562 | 7 | 100+ | Three backend implementations + ABC + cache decorator |
| `handlers.py` | 1,676 | 8 | 60+ | Seven handler implementations + registry |
| `security.py` | 420 | 1 (`CacheEntrySigner`) | 12 | HMAC signing |
| `error_handling.py` | ~80 | 6 exception classes | 2 | Error hierarchy |

### Import Chain (Critical for All Refactors)

```
Users import from:
  cacheness              → __init__.py → core.py, handlers.py, metadata.py, ...
  cacheness.core         → core.py
  cacheness.storage      → storage/__init__.py → re-exports from handlers.py, metadata.py, security.py
  cacheness.storage.handlers    → storage/handlers/__init__.py → re-exports from handlers.py
  cacheness.storage.backends    → storage/backends/__init__.py → re-exports from metadata.py
  cacheness.storage.backends.base → storage/backends/base.py → re-exports from metadata.py
```

**Key constraint:** The `storage/` sub-package is a re-export layer. It does NOT contain primary definitions — it imports from parent-level modules (`handlers.py`, `metadata.py`, `security.py`) and re-exports them. Any module split must preserve these re-export chains.

---

## Change Analysis

### 1. handlers.py → handlers/ Package Split

**What changes:** Convert `src/cacheness/handlers.py` (single file, 1,676 lines) into `src/cacheness/handlers/` package with one file per handler.

**Integration points:**
- `cacheness/__init__.py` imports: `ArrayHandler`, `HandlerRegistry`, `ObjectHandler`
- `cacheness/storage/handlers/__init__.py` re-exports: `ArrayHandler`, `BytesHandler`, `ObjectHandler`, `HandlerRegistry`
- `cacheness/storage/blob_store.py` imports `HandlerRegistry`
- `cacheness/core.py` imports `HandlerRegistry`

**New components:**

| File | Contents | Lines (est.) |
|------|----------|-------------|
| `handlers/__init__.py` | `HandlerRegistry`, re-exports of all handlers, `_lazy_import_tensorflow()` | ~350 |
| `handlers/polars_handler.py` | `PolarsDataFrameHandler`, `PolarsSeriesHandler` | ~310 |
| `handlers/pandas_handler.py` | `PandasDataFrameHandler`, `PandasSeriesHandler` | ~200 |
| `handlers/array_handler.py` | `ArrayHandler` (NumPy/blosc2) | ~200 |
| `handlers/tensorflow_handler.py` | `TensorFlowTensorHandler` | ~150 |
| `handlers/bytes_handler.py` | `BytesHandler` | ~100 |
| `handlers/object_handler.py` | `ObjectHandler` (pickle/dill) | ~380 |

**Modified components:**
- `cacheness/__init__.py` — no change needed (imports from `cacheness.handlers` which becomes `cacheness.handlers.__init__`)
- `cacheness/storage/handlers/__init__.py` — no change needed (imports `from cacheness.handlers import ...`)

**Data flow changes:** None. Pure file reorganization.

**Risk assessment:** LOW
- Python's package `__init__.py` import mechanism means `from cacheness.handlers import HandlerRegistry` works identically whether `handlers` is a file or a package.
- Each handler is self-contained — no handler imports from another handler.
- The `HandlerRegistry` stays in `__init__.py` and imports handler classes from sub-modules.
- Only risk: circular imports if handler sub-modules try to import `HandlerRegistry`. Prevention: handlers only import from `cacheness.interfaces` (leaf module).

**Verification:** Run `tests/test_handlers.py` — covers all handler type detection and serialization.

---

### 2. metadata.py → metadata/ Package Split

**What changes:** Convert `src/cacheness/metadata.py` (single file, 2,562 lines) into `src/cacheness/metadata/` package with per-backend files.

**Integration points:**
- `cacheness/__init__.py` imports: `JsonBackend`, `create_metadata_backend`, `NamespaceInfo`, `validate_namespace_id`, `DEFAULT_NAMESPACE`, `SqliteBackend`
- `cacheness/storage/__init__.py` imports: `MetadataBackend`, `JsonBackend`, `create_metadata_backend`
- `cacheness/storage/backends/__init__.py` imports from `cacheness.metadata` and re-exports
- `cacheness/storage/backends/base.py` imports: `MetadataBackend`, `NamespaceInfo`, `validate_namespace_id`, `DEFAULT_NAMESPACE`
- `cacheness/core.py` imports: `create_metadata_backend`, `MetadataBackend`, `DEFAULT_NAMESPACE`, `NamespaceInfo`, `validate_namespace_id`
- `cacheness/config.py` imports: `validate_namespace_id`
- `cacheness/storage/backends/postgresql_backend.py` imports: `MetadataBackend` (from `cacheness.metadata`)
- `cacheness/custom_metadata.py` likely references metadata types

**New components:**

| File | Contents | Lines (est.) |
|------|----------|-------------|
| `metadata/__init__.py` | Re-exports all public names, `create_metadata_backend()` factory, `DEFAULT_NAMESPACE`, `validate_namespace_id()`, `NamespaceInfo` | ~150 |
| `metadata/base.py` | `MetadataBackend` ABC, `CachedMetadataBackend` decorator, SQLAlchemy model mixins (`CacheEntryMixin`, `CacheStatsMixin`, `CacheEntry`, `CacheStats`, `CacheNamespace`, `_get_namespace_models`), `create_entry_cache()` | ~750 |
| `metadata/json_backend.py` | `JsonBackend` | ~720 (lines 1077-1790) |
| `metadata/sqlite_backend.py` | `SqliteBackend`, migration functions (`_sqlite_migrate_v1_to_v2`, `_sqlite_migrate_v2_to_v3`) | ~660 (lines 1792-2562) |

**Modified components:**
- `cacheness/__init__.py` — no change needed (`from .metadata import ...` resolves to package `__init__.py`)
- `cacheness/storage/backends/base.py` — no change needed (`from cacheness.metadata import ...`)
- `cacheness/storage/backends/__init__.py` — no change needed
- `cacheness/core.py` — no change needed
- `cacheness/config.py` — no change needed

**Data flow changes:** None. Pure file reorganization.

**Risk assessment:** MEDIUM
- The SQLAlchemy model classes (`CacheEntry`, `CacheStats`, `CacheNamespace`) and `_get_namespace_models()` are shared between `JsonBackend` and `SqliteBackend`. They must live in `base.py` and be importable by both backends.
- `CachedMetadataBackend` wraps any `MetadataBackend` — it must import the ABC but not concrete backends (no circular risk).
- SQLite migration functions reference `SqliteBackend` type annotations — they must live in `sqlite_backend.py` or use forward references.
- PostgreSQL backend in `storage/backends/postgresql_backend.py` imports `MetadataBackend` from `cacheness.metadata` — this resolves correctly through the package's `__init__.py`.
- **Watch out:** The `metadata.py` file has a `try/except ImportError` block (lines 148-370) that conditionally defines SQLAlchemy models or stub classes depending on whether SQLAlchemy is available. This conditional definition block must stay in `base.py` and the stubs must be importable by both backends.

**Verification:** Run `tests/test_metadata.py`, `tests/test_backend_parity.py`, `tests/test_sqlite_schema_versioning.py`, `tests/test_json_schema_versioning.py`.

---

### 3. core.py → Mixin Decomposition

**What changes:** Extract coherent method groups from `UnifiedCache` (3,900 lines, 85+ methods) into mixin classes, while preserving the single `UnifiedCache` class as the public API.

**Integration points:**
- `UnifiedCache` is the central coordinator — it touches every other layer
- `cacheness/__init__.py` exports `UnifiedCache as cacheness`
- `cacheness/decorators.py` creates `UnifiedCache` instances
- Tests use `from cacheness.core import UnifiedCache`
- `self._lock`, `self.metadata_backend`, `self._blob_store`, `self.config`, `self._entry_signer` are shared state accessed by all method groups

**Proposed mixin structure (based on method analysis):**

| Mixin | Methods (approx.) | Lines (est.) | Depends on |
|-------|-------------------|-------------|------------|
| `_CustomMetadataMixin` | `_store_custom_metadata`, `_get_custom_metadata`, `_get_registered_schemas`, `query_custom`, `query_custom_session`, `query_custom_metadata`, `get_custom_metadata_for_entry` | ~350 | `self.metadata_backend`, `self.config` |
| `_QueryMixin` | `query_meta`, `_query_meta_generic`, `_query_meta_postgres`, `_query_meta_sqlite`, `_meta_value_matches`, `_fmt_timestamp`, `query_with_meta`, `query_with_model` | ~400 | `self.metadata_backend` |
| `_VerificationMixin` | `_verify_entry`, `_calculate_file_hash`, `verify_integrity` | ~170 | `self._entry_signer`, `self._blob_store`, `self.metadata_backend` |
| `_StorageModeMixin` | `_storage_mode_put`, `_storage_mode_get`, `_storage_mode_get_with_metadata` | ~200 | `self._blob_store`, `self.metadata_backend`, `self.config` |
| `_FileOpsMixin` | `put_file`, `get_file`, `_resolve_original_filename` | ~210 | `self.put`, `self.get`, `self.metadata_backend` |
| `_BatchOpsMixin` | `get_batch`, `delete_batch`, `touch_batch`, `delete_where`, `delete_matching` | ~220 | `self.metadata_backend`, `self.invalidate` |
| `_ManagementMixin` | `exists`, `update_data`, `touch`, `list_entries`, `get_stats`, `clear_all`, `clear`, `clear_all_namespaces`, `cleanup_expired`, `_enforce_size_limit` | ~400 | `self.metadata_backend`, `self._lock` |
| `_MetaAPIMixin` | `put_with_meta`, `get_with_meta`, `put_with_model`, `get_with_model`, `get_metadata`, `get_with_metadata`, `_merge_on_and_kwargs`, `_extract_metadata_dict` | ~350 | `self.put`, `self.get` |
| Core `UnifiedCache` | `__init__`, `put`, `get`, `_resolve_cache_key`, `_create_cache_key`, `_get_cache_file_path`, `_is_expired`, `_sign_entry_if_enabled`, `_extract_signable_fields`, `_build_metadata_dict`, `_try_inline_blob`, `_read_inline_blob`, `_cleanup_stale_blob`, `_invoke_hook`, `_record_hit/miss`, `_init_*`, `close`, dunder methods, `for_api` | ~1,300 | Everything |

**New components:**

| File | Contents |
|------|----------|
| `core/_custom_metadata.py` | `_CustomMetadataMixin` |
| `core/_query.py` | `_QueryMixin` |
| `core/_verification.py` | `_VerificationMixin` |
| `core/_storage_mode.py` | `_StorageModeMixin` |
| `core/_file_ops.py` | `_FileOpsMixin` |
| `core/_batch_ops.py` | `_BatchOpsMixin` |
| `core/_management.py` | `_ManagementMixin` |
| `core/_meta_api.py` | `_MetaAPIMixin` |
| `core/__init__.py` | Re-exports `UnifiedCache`, `CacheConfig`, `get_cache` |
| `core/_core.py` | `UnifiedCache(mixins..., ...)` class with `__init__`, `put`, `get`, core methods |

**Modified components:**
- `cacheness/__init__.py` — change `from .core import ...` path (resolves automatically if core becomes a package)
- No other files need changes — `from cacheness.core import UnifiedCache` resolves through `core/__init__.py`

**Data flow changes:** None. Pure structural refactor. All mixins access the same `self` instance — they're just code organization.

**Risk assessment:** HIGH
- **State coupling:** All mixins access shared instance attributes (`self._lock`, `self.metadata_backend`, `self._blob_store`, `self.config`, `self._entry_signer`). This is inherent to the mixin pattern and acceptable, but requires care to avoid import cycles between mixin files.
- **Method cross-references:** Some methods call other methods that may end up in different mixins (e.g., `_storage_mode_put` calls `_build_metadata_dict`, `_sign_entry_if_enabled`, `_try_inline_blob`). These work correctly via `self.method()` because mixins compose into one class, but the dependency graph must be documented.
- **Type checking:** Mixin classes won't have type information about attributes from other mixins. Use `if TYPE_CHECKING: from ._core import UnifiedCache` for IDE support, or use `Protocol` for attribute contracts.
- **Testing:** `tests/test_core.py` imports `from cacheness.core import UnifiedCache` — must continue to work.
- **Git history:** Moving code across files destroys `git blame` continuity. Consider using `git log --follow` and documenting the move.

**Mitigation:**
- Keep mixins as underscored private classes (`_QueryMixin`) so users never import them directly.
- Each mixin file only imports from `cacheness.interfaces` (types), `cacheness.config` (config), and `cacheness.error_handling` (exceptions) — leaf modules with no circular risk.
- `_core.py` imports all mixins and composes them into `UnifiedCache`.
- `core/__init__.py` re-exports only `UnifiedCache`, `CacheConfig`, `get_cache`.

**Verification:** Full test suite — every test that touches caching exercises `UnifiedCache`.

---

### 4. Security Layer Changes

**What changes:** Three additions to the security layer:
1. Blob content hashing default-on (currently computed but verification is opt-in)
2. Per-namespace key derivation via HKDF
3. Configurable in-memory key fallback behavior
4. Windows key file permissions (ACLs or documented limitation)

**Integration points for each:**

#### 4a. Blob Content Hashing Default-On

**Current flow:**
1. `core.py` `put()` line ~1969: calls `hash_file_content()` from `file_hashing.py`, stores hash in metadata as `file_hash`
2. `core.py` `_verify_entry()` line ~1293: checks `file_hash` if present in metadata and if `verify_file_hash` config is True
3. `CacheEntrySigner.sign_entry()` includes `file_hash` in signed fields (signature covers the hash)

**Change needed:**
- `config.py` `SecurityConfig`: change `verify_file_hash` default from `False` to `True`
- `core.py` `_verify_entry()`: already supports verification — the config change makes it default
- No new components. One config default change + potential performance implication (re-hashing on every read verification).

**Risk:** LOW. The hash is already computed and stored on write. Only the read-side verification is newly default. Performance impact is one xxhash per `get()` (fast — xxhash is ~10GB/s).

#### 4b. Per-Namespace Key Derivation (HKDF)

**Current flow:**
- `security.py` `CacheEntrySigner.__init__()`: loads/generates one signing key per `key_file_path`
- `core.py` `_init_entry_signer()` line ~352: creates one signer per `UnifiedCache` instance
- All namespaces sharing a `cache_dir` share the same key file by default

**Change needed:**
- `security.py`: Add `derive_namespace_key(master_key: bytes, namespace_id: str) -> bytes` using `cryptography.hazmat.primitives.kdf.hkdf.HKDF` or `hmac`-based derivation
- `security.py` `CacheEntrySigner.__init__()`: Accept optional `namespace_id` parameter; if provided, derive a namespace-specific key from the master key
- `core.py` `_init_entry_signer()`: Pass `self.config.namespace` to the signer
- `config.py` `SecurityConfig`: Add `per_namespace_keys: bool = True` flag

**New components:** `derive_namespace_key()` function in `security.py` (~30 lines).
**Modified:** `CacheEntrySigner.__init__()`, `core.py` `_init_entry_signer()`, `config.py` `SecurityConfig`.

**Risk:** MEDIUM. Changing key derivation invalidates existing signatures. Need migration path — try derived key first, fall back to master key for existing entries, and re-sign on next write.

#### 4c. Configurable Key Fallback

**Current flow:** `security.py` `_generate_new_key()` lines ~155-160: on disk write failure, falls back to in-memory key with a warning log.

**Change needed:**
- `config.py` `SecurityConfig`: Add `allow_memory_key_fallback: bool = True`
- `security.py` `_generate_new_key()`: Raise `CacheSecurityError` if fallback disabled and disk write fails

**New components:** None (small conditional + config field).
**Risk:** LOW. Additive behavior with backward-compatible default.

#### 4d. Windows Key Permissions

**Current flow:** `security.py` line ~144: `os.chmod(str(self.key_file_path), 0o600)` wrapped in `try/except`.

**Change needed:** Either:
- Option A: Use `subprocess.run(["icacls", ...])` to set Windows ACLs — requires no extra dependencies but is platform-specific and brittle
- Option B: Use `pywin32` (`win32security`) for proper ACL manipulation — requires optional dependency
- Option C: Document as known limitation, recommend BitLocker or user-profile-scoped directories

**Risk:** LOW for Option C, MEDIUM for A/B. Recommend Option C for this milestone (documentation) with A as a stretch goal.

---

### 5. Error Handling — Exception Narrowing

**What changes:** Narrow ~36 `except Exception` catches in `core.py` (plus ~10 in `compress_pickle.py`, ~6 in `custom_metadata.py`) to specific exception types.

**Integration points:**
- `error_handling.py` defines the exception hierarchy: `CacheError` → `CacheConfigurationError`, `CacheStorageError`, `CacheSerializationError`, `CacheHandlerError`, etc.
- Every `except Exception` in `core.py` is a potential integration point

**Exception narrowing map for core.py (36 instances):**

| Location | Lines | Current catch | Likely narrowed types |
|----------|-------|---------------|----------------------|
| Module-level imports | 53, 80, 125 | `except Exception` | `ImportError` only |
| `_init_auto_backend` | 282 | `except Exception as e` | `ImportError`, `OSError`, `sqlalchemy.exc.SQLAlchemyError` |
| `_sign_current_namespace` | 372, 414 | `except Exception as e` | `CacheSecurityError`, `OSError` |
| `_store_custom_metadata` | 491, 549, 588 | `except Exception` | `sqlalchemy.exc.SQLAlchemyError`, `CacheStorageError` |
| `query_custom`/`query_meta` | 664, 818, 845, 896, 915, 979, 986 | `except Exception` | `sqlalchemy.exc.SQLAlchemyError`, `KeyError`, `ValueError` |
| `content_key` | 1100 | `except Exception` | `TypeError`, `CacheSerializationError` |
| `_invoke_hook` | 1278 | `except Exception as exc` | Keep — hooks are user code, `Exception` is correct |
| `_read_inline_blob` | 1574 | `except Exception as exc` | `pickle.UnpicklingError`, `CacheReadError` |
| `_sign_entry_if_enabled` | 1645 | `except Exception as e` | `CacheSecurityError`, `KeyError` |
| `_cleanup_stale_blob` | 1671 | `except Exception as exc` | `OSError`, `CacheStorageError` |
| `_storage_mode_*` | 1758, 1804, 1847 | `except Exception as e` | `CacheStorageError`, `CacheReadError` |
| `put()` inner | 1969 | `except Exception as e` | `CacheWriteError`, `OSError` |
| `put()` outer | 2029 | `except Exception as e` | `CacheError` (re-raise known), `Exception` → wrap |
| `get()` | 2125, 2266 | `except Exception as e` | `CacheReadError`, `CacheFormatError`, `OSError` |
| `_resolve_original_filename` | 2903 | `except Exception` | `KeyError`, `TypeError` |
| `update_data` inner | 3139, 3155, 3164 | `except Exception` | `CacheWriteError`, `CacheStorageError`, `OSError` |
| `touch` | 3246 | `except Exception as e` | `CacheStorageError`, `KeyError` |
| `delete_where` | 3312 | `except Exception as exc` | `CacheStorageError`, `OSError` |
| `__del__` | 3823 | `except Exception` | Keep — destructor must never raise |
| `for_api` | 3895 | `except Exception` | `ImportError` |

**Modified components:** `core.py` (36 sites), `compress_pickle.py` (~10 sites), `custom_metadata.py` (~6 sites). No new components.

**Risk assessment:** MEDIUM
- Narrowing exceptions can expose previously-swallowed errors to callers. Each change must be verified to not break existing behavior.
- Some catches are intentionally broad (hooks, destructors) — these should be preserved with explicit comments.
- The narrowing should happen **after** the core.py mixin decomposition, because:
  1. Smaller files are easier to audit exhaustively
  2. Each mixin can be narrowed independently with targeted tests
  3. Reduces merge conflicts (different concerns in different files)

**Verification:** Full test suite, plus specific fault injection tests (`tests/test_fault_injection.py`).

---

### 6. Test Gap Coverage

**What changes:** Add thread safety tests and key rotation tests.

**Integration points:**
- Thread safety tests exercise `UnifiedCache.put()` and `UnifiedCache.get()` concurrently
- Key rotation tests exercise `CacheEntrySigner` key file lifecycle

**New components:**

| File | Contents | Depends on |
|------|----------|------------|
| `tests/test_thread_safety.py` | Concurrent put/get with `threading.Thread`, verify no data races | `UnifiedCache`, all backends |
| `tests/test_key_rotation.py` | Key file deletion, restart, re-sign, verify old entries | `CacheEntrySigner`, `UnifiedCache` |

**Risk:** LOW. Additive — new test files, no source changes.

---

## Dependency Graph Between Changes

```
                    ┌─────────────────────────┐
                    │                         │
                    ▼                         │
            ┌──────────────┐                  │
            │ 1. handlers/ │ ─ ─ ─ ─ ─ ─ ─ ─ ┤ (no real dependency,
            │    split     │                  │  but good warm-up)
            └──────┬───────┘                  │
                   │                          │
                   ▼                          │
            ┌──────────────┐                  │
            │ 2. metadata/ │                  │
            │    split     │                  │
            └──────┬───────┘                  │
                   │                          │
                   ▼                          │
            ┌──────────────┐           ┌──────┴───────┐
            │ 3. core.py   │           │ 6. Test gaps │
            │    mixins    │           │ (parallel)   │
            └──────┬───────┘           └──────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
   ┌──────────────┐  ┌──────────────┐
   │ 4. Exception │  │ 5. Security  │
   │   narrowing  │  │  hardening   │
   └──────────────┘  └──────────────┘
```

**Dependency rationale:**
- **1 → 2 → 3 (sequential):** Each split changes import paths. Landing them sequentially avoids compounding merge conflicts. Handlers is simplest (self-contained handlers), metadata is medium (shared base class), core is hardest (cross-cutting mixins).
- **3 → 4 (depends):** Exception narrowing in core.py is dramatically easier when the code is split into 300-line mixin files vs one 3,900-line monolith. Smaller files → exhaustive auditing of each catch site.
- **3 → 5 (depends):** Security changes modify `core.py` methods (`_init_entry_signer`, `_verify_entry`). After mixin decomposition, these live in `_verification.py` and the init section — cleaner, smaller diffs.
- **6 (parallel):** Test additions are independent of source refactoring and can proceed at any time.

---

## Suggested Phase Order

### Phase 1: handlers.py → handlers/ Package

**Scope:** Convert `handlers.py` to `handlers/` package.
**Effort:** Small (~1 day)
**Risk:** Low
**Build steps:**
1. Create `handlers/` directory
2. Move each handler class to its own file
3. Create `handlers/__init__.py` with `HandlerRegistry` + re-exports
4. Delete old `handlers.py`
5. Verify: `storage/handlers/__init__.py` re-exports still work
6. Run `tests/test_handlers.py`

**Why first:** Smallest scope, lowest fan-out. Good validation that the package-split pattern works before tackling metadata or core.

### Phase 2: metadata.py → metadata/ Package

**Scope:** Convert `metadata.py` to `metadata/` package.
**Effort:** Medium (~1-2 days)
**Risk:** Medium (SQLAlchemy model sharing, conditional import blocks)
**Build steps:**
1. Create `metadata/` directory
2. Move `MetadataBackend` ABC, SQLAlchemy models, `CachedMetadataBackend` to `metadata/base.py`
3. Move `JsonBackend` to `metadata/json_backend.py`
4. Move `SqliteBackend` + migration functions to `metadata/sqlite_backend.py`
5. Create `metadata/__init__.py` with re-exports + `create_metadata_backend()` factory
6. Delete old `metadata.py`
7. Verify all re-export chains (`storage/backends/base.py`, `storage/backends/__init__.py`)
8. Run `tests/test_metadata.py`, `tests/test_backend_parity.py`, `tests/test_sqlite_schema_versioning.py`, `tests/test_json_schema_versioning.py`

**Why second:** More complex than handlers due to shared base class, but still a clean structural split. Validates the pattern for more complex cases.

### Phase 3: core.py → Mixin Decomposition

**Scope:** Extract coherent method groups into mixin classes.
**Effort:** Large (~2-3 days)
**Risk:** High (tight state coupling, cross-method references)
**Build steps:**
1. Create `core/` directory (or keep as single file with imported mixins in a `_mixins/` subdir)
2. Extract one mixin at a time, starting with the most self-contained:
   a. `_CustomMetadataMixin` (clear boundary — custom metadata is isolated)
   b. `_StorageModeMixin` (clear boundary — storage mode methods are self-contained)
   c. `_FileOpsMixin` (clear boundary — `put_file`/`get_file` wrap `put`/`get`)
   d. `_QueryMixin` (clear boundary — query methods only read metadata)
   e. `_BatchOpsMixin` (clear boundary — batch operations wrap single operations)
   f. `_MetaAPIMixin` (clear boundary — convenience wrappers around put/get)
   g. `_ManagementMixin` (moderate coupling — accesses `_lock`, touches eviction)
   h. `_VerificationMixin` (moderate coupling — accesses signer, blob store, metadata)
3. After each extraction, run `tests/test_core.py` to verify no breakage
4. Create `core/__init__.py` with re-exports

**Why third:** Largest scope, requires most coordination. Benefits from the experience gained in phases 1-2. The mixin pattern is validated by the handler/metadata splits.

**Alternative approach (lower risk):** Instead of a `core/` package, keep `core.py` but extract mixins into sibling files:
```
src/cacheness/
  core.py              # UnifiedCache(mixins...) + __init__, put, get
  _core_query.py       # _QueryMixin
  _core_storage.py     # _StorageModeMixin
  _core_custom_meta.py # _CustomMetadataMixin
  ...
```
This avoids the package conversion and its import chain implications. `core.py` imports from `_core_*.py` files. The `_` prefix signals private modules. Trade-off: more files at the package root, but simpler import resolution.

### Phase 4: Error Handling Narrowing

**Scope:** Narrow `except Exception` catches across all source files.
**Effort:** Medium (~1-2 days)
**Risk:** Medium (behavioral change — previously-swallowed errors may surface)
**Build steps:**
1. Audit each `except Exception` site in the (now-smaller) mixin files
2. Categorize: intentionally broad (keep), narrowable (change), unknown (investigate)
3. For each narrowable site, determine specific exceptions and update
4. Add `# intentionally broad` comments to kept sites
5. Run fault injection tests after each file's changes

**Why after phase 3:** Mixin decomposition produces 300-line files instead of one 3,900-line file. Exhaustive auditing becomes tractable.

### Phase 5: Security Hardening

**Scope:** Blob content hashing default-on, per-namespace key derivation, configurable key fallback, Windows key permissions.
**Effort:** Medium (~2 days)
**Risk:** Medium (key derivation changes invalidate existing signatures)
**Build steps:**
1. Change `verify_file_hash` default to `True` in `SecurityConfig` (smallest change, immediate value)
2. Add `allow_memory_key_fallback` config flag + conditional in `_generate_new_key()`
3. Implement `derive_namespace_key()` + integration in `CacheEntrySigner.__init__()` with fallback to master key
4. Document Windows key permission limitation (or implement `icacls` approach)
5. Run `tests/test_security.py` + new `tests/test_key_rotation.py`

**Sub-ordering rationale:** 5.1 is a config default flip. 5.2 is a small conditional. 5.3 is the largest change with migration implications. 5.4 is documentation or platform-specific code.

### Phase 6 (Parallel): Test Gap Coverage

**Scope:** Thread safety tests, key rotation tests.
**Effort:** Small (~1 day)
**Risk:** Low (additive)
**Can run in parallel with phases 1-3.** Key rotation tests specifically should land with or after phase 5.

---

## Cross-Cutting Concerns

### Import Path Preservation

Every refactor must preserve these import paths (tested by existing tests):

```python
from cacheness import cacheness                            # UnifiedCache alias
from cacheness import get_cache, cached, cache_if          # Factory + decorators
from cacheness import HandlerRegistry, ArrayHandler        # Handlers
from cacheness import JsonBackend, SqliteBackend           # Backends
from cacheness import CacheConfig                          # Configuration
from cacheness.core import UnifiedCache                    # Direct import
from cacheness.storage import BlobStore                    # Storage layer
from cacheness.storage.handlers import HandlerRegistry     # Handler re-export
from cacheness.storage.backends import MetadataBackend     # Backend re-export
from cacheness.storage.backends.base import MetadataBackend # Deep re-export
from cacheness.interfaces import CacheHandler              # Interface
```

### Thread Safety Implications

The mixin decomposition doesn't change thread safety characteristics. The shared `self._lock` RLock is accessed by the same methods regardless of which file they live in. However, the decomposition makes it **easier to audit** which methods acquire the lock and which don't.

### Performance Implications

None of the structural refactors (phases 1-3) affect runtime performance. They're purely organizational. The security changes (phase 5) add:
- One xxhash computation per `get()` when blob verification is default-on (~negligible, xxhash is extremely fast)
- One HKDF derivation per `CacheEntrySigner.__init__()` when per-namespace keys are enabled (~negligible, one-time cost)

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| handlers/ split | HIGH | Each handler is completely self-contained, no shared state between handlers |
| metadata/ split | HIGH | Clear class boundaries, well-defined shared base. The SQLAlchemy conditional import block is the only wrinkle |
| core.py mixins | MEDIUM | Method coupling analysis is solid, but the actual extraction will reveal edge cases in cross-method calls |
| Exception narrowing | MEDIUM | The 36 sites are identified but determining correct narrow types requires per-site analysis during implementation |
| Security hardening | HIGH | Integration points are clear, existing infrastructure supports the changes |
| Test gaps | HIGH | Straightforward additive work |

---

## Gaps to Address

- **core.py mixin type safety:** Need to decide between `Protocol`-based attribute contracts vs runtime `self` access for mixin ↔ core attribute sharing. Recommend starting with bare `self` access (simpler) and adding `Protocol` only if `ty check` complains.
- **Migration path for per-namespace keys:** Need to design the fallback logic for existing entries signed with the master key when per-namespace keys are enabled. Likely: try namespace-derived key first, fall back to master key for verification, re-sign with namespace key on next write.
- **`compress_pickle.py` exception narrowing:** Not analyzed in detail — the ~10 `except Exception` catches there need separate per-site analysis.
- **PostgreSQL backend location:** Currently in `storage/backends/postgresql_backend.py`, not in `metadata.py`. The metadata/ package split doesn't affect it, but a future decision about whether to move it under `metadata/` might be needed.

---

*Architecture research: 2026-04-02*
