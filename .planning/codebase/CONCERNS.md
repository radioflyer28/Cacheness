# Codebase Concerns

**Analysis Date:** 2026-04-07
**Codebase Version:** v0.11.0 (post Cross-Backend Hardening milestone)

## Resolved Since Last Audit (v0.7.0–v0.11.0)

The following concerns from the 2026-04-02 audit have been resolved:

- **Broad `except Exception` usage** — All 60+ instances annotated with `# intentionally broad` and justification (v0.7.0). New error types `CacheSecurityError`, `CacheBackendError` added.
- **`core.py` monolith (3,307 lines)** — Decomposed into 12 mixins + core (v0.7.0, Phase 3). Core is now 1,217 lines.
- **`metadata.py` monolith (2,562 lines)** — Split into `metadata/` package: `base.py`, `json_backend.py`, `sqlite_backend.py`, `_compat.py` (v0.7.0, Phase 2).
- **`handlers.py` monolith (1,425 lines)** — Split into `handlers/` package: 11 files (v0.7.0, Phase 1).
- **Thread safety (`_lock` inconsistency)** — Confirmed `RLock` is acquired in put/get and 22+ methods. Docs corrected (v0.8.0).
- **Crash safety** — `WriteIntentJournal` records intent before blob write, cleans stale intents on init (v0.8.0, Phase 8).
- **Signing key permissions on Windows** — `icacls` used on Windows, `chmod 0o600` on Unix (v0.9.0, Phase 14).
- **Shared signing key across namespaces** — HKDF-SHA256 key derivation per namespace is now default (`use_hkdf_derivation=True`) (v0.10.0).
- **Key fallback behavior** — Configurable via `key_fallback_policy` ("raise"/"warn"/"fallback") (v0.10.0, Phase 15).
- **Encryption at rest** — AES-256-GCM with random 12-byte IV per blob, HKDF-derived per-namespace keys (v0.10.0).
- **Encryption schema in metadata** — `encryption_algorithm`, `encryption_iv`, `cacheness_version` columns added to SQLite/PG (v0.11.0, Phase 23).
- **Cross-backend encryption parity** — Parametrized tests across JSON/SQLite/PostgreSQL backends (v0.11.0, Phase 24).
- **Config validation** — Bad combos (encryption without signing, encryption without key file) caught at construction time (v0.11.0, Phase 26).

---

## Tech Debt

### High Severity

**JSON backend O(n²) scaling:**
- Issue: `JsonBackend._save_to_disk()` re-serializes the entire metadata file on every write. `_load_from_disk()` loads the full file into memory on init.
- Files: `src/cacheness/metadata/json_backend.py` (lines 74–106)
- Impact: Write performance degrades quadratically with entry count. At 500+ entries, individual writes take seconds. Memory usage grows linearly.
- Fix approach: Known limitation — documented in `docs/TROUBLESHOOTING.md`. Recommended mitigation: switch to SQLite for >200 entries. No incremental write support planned for JSON.

**Deprecated API surface still present:**
- Issue: Three deprecated APIs remain in the codebase without removal timeline.
- Files:
  - `src/cacheness/config.py` (line 408): `raise_on_key_fallback` — deprecated in favor of `key_fallback_policy`
  - `src/cacheness/config.py` (line 661): `store_cache_key_params` — deprecated in favor of `store_full_metadata`
  - `src/cacheness/_custom_metadata_mixin.py` (line 342): `query_custom_metadata()` — deprecated in favor of `query_custom()`
- Impact: Increases API surface, confuses new users, complicates documentation.
- Fix approach: Add deprecation timeline (e.g., remove in v1.0). Emit `DeprecationWarning` consistently (already done for two of three).

### Medium Severity

**`postgresql_backend.py` is 1,410 lines:**
- Issue: Largest file in the codebase. Combines the PostgreSQL metadata backend, schema migrations, namespace management, query optimization, and connection pool handling.
- Files: `src/cacheness/storage/backends/postgresql_backend.py`
- Impact: Difficult to navigate and modify. Changes in migration logic risk breaking query paths.
- Fix approach: Could split into `pg_backend.py` + `pg_migrations.py` + `pg_namespace.py`, following the pattern used for `metadata/` split.

**`config.py` is 1,253 lines — large dataclass constellation:**
- Issue: Contains 7+ dataclasses (`CacheMetadataConfig`, `StorageConfig`, `CompressionConfig`, `SerializationConfig`, `HandlerConfig`, `SecurityConfig`, `CacheConfig`) plus factory functions and convenience builders.
- Files: `src/cacheness/config.py`
- Impact: Finding a specific config option requires scanning 1,200+ lines. Adding new config requires modifying a very large file.
- Fix approach: Split into `config/` package by area (storage, security, handlers, etc.) if it continues growing.

**`ast.literal_eval` for shape parsing in numpy handler:**
- Issue: `ast.literal_eval(shape_str)` is used to reconstruct NumPy array shapes from stored strings at two sites. While `literal_eval` is safe (only evaluates literals), the input comes from file bytes read from disk.
- Files: `src/cacheness/handlers/numpy_array.py` (lines 167, 282)
- Impact: Minimal security risk (`literal_eval` is safe), but if the binary format header is corrupted, the error message is unhelpful.
- Fix approach: Add explicit validation that the parsed result is a tuple of ints.

### Low Severity

**`type: ignore` annotations in metadata ORM layer:**
- Issue: 6 `type: ignore[no-redef]` and 1 `type: ignore[arg-type]` in `metadata/_compat.py` for fallback class definitions when SQLAlchemy is unavailable.
- Files: `src/cacheness/metadata/_compat.py` (lines 197, 312–327)
- Impact: Type checker noise. The pattern is intentional (conditional imports), but tools report these as issues.
- Fix approach: Consider `TYPE_CHECKING` guard or Protocol-based stubs to satisfy type checkers.

---

## Known Bugs / Limitations

**`get()` is destructive on errors (by design):**
- Issue: With `delete_on_error=True` (default), `get()` auto-deletes entries that fail deserialization. This prevents retry if the failure was transient (e.g., file locked).
- Files: `src/cacheness/core.py`
- Workaround: Set `CacheMetadataConfig(delete_on_error=False)` to preserve entries and return `None`.
- In `storage_mode=True`, entries are never deleted regardless.

**Orphaned blob files after hard crashes:**
- Issue: A hard crash (kill -9, power loss) between blob write and metadata commit leaves orphaned blobs. The `WriteIntentJournal` (v0.8.0) catches most cases, but intents are not fsynced — a kernel crash before intent flush can still leave orphans.
- Files: `src/cacheness/write_intent.py`, `src/cacheness/storage/blob_store.py`
- Workaround: Run `cache.verify_integrity(repair=True)` periodically.

**Cache key fragility with new `put()` parameters:**
- Issue: `_create_cache_key()` strips known control parameters (`prefix`, `description`, `custom_metadata`, `ttl_seconds`) before hashing. Adding a new named parameter to `put()` without adding it to the strip list silently changes all cache keys.
- Files: `src/cacheness/serialization.py`, `src/cacheness/core.py`
- Mitigation: Documented in `.github/copilot-instructions.md` as a critical bug pattern.

**TensorFlow handler disabled:**
- Issue: `TensorFlowTensorHandler` exists and is fully implemented, but is commented out in `HandlerRegistry` due to system compatibility issues (hangs on Windows, import side effects).
- Files: `src/cacheness/handlers/registry.py` (lines 79–84, 110–112), `src/cacheness/handlers/tensorflow_tensor.py`
- Impact: Users must install TensorFlow and manually enable via config — but even then the handler is not registered. Tests are in `tests/test_tensorflow_handler.py` (always ignored on Windows).
- Documented: `docs/TENSORFLOW_HANDLER_STATUS.md`

---

## Security Considerations

### Addressed

| Feature | Status | Files |
|---------|--------|-------|
| HMAC-SHA256 metadata signing | ✅ v1/v2/v3 signatures, versioned | `src/cacheness/security.py` |
| HKDF per-namespace key derivation | ✅ Default on (`use_hkdf_derivation=True`) | `src/cacheness/security.py` |
| AES-256-GCM blob encryption | ✅ Optional (`enable_content_encryption`) | `src/cacheness/encryption.py` |
| Key rotation | ✅ `rotate_signing_key()` re-signs all entries | `src/cacheness/core.py`, `src/cacheness/storage/blob_store.py` |
| Config validation | ✅ Bad combos caught at construction | `src/cacheness/config.py` |
| Cross-platform key permissions | ✅ Unix `chmod 0o600`, Windows `icacls` | `src/cacheness/security.py` |
| Deserialization security docs | ✅ 5 call sites annotated, SECURITY.md | `src/cacheness/compress_pickle.py`, `src/cacheness/handlers/object_handler.py` |

### Remaining Considerations

**Pickle/dill deserialization of untrusted data (inherent risk):**
- Risk: `pickle.loads()` (4 sites in `compress_pickle.py`, 1 in `object_handler.py`) and `dill.loads()` (2 sites) can execute arbitrary code.
- Files: `src/cacheness/compress_pickle.py` (lines 204, 895, 898, 901), `src/cacheness/handlers/object_handler.py` (lines 232, 260, 391, 392)
- Current mitigation: 3-layer integrity chain (xxhash → HMAC → signature verification) runs before deserialization. Security comments at all sites. Documented in `docs/SECURITY.md`.
- Residual risk: If an attacker can write to the cache directory AND knows/compromises the signing key, they can inject malicious pickled objects. This is inherent to any pickle-based cache.

**`subprocess.run` for Windows ACL:**
- Risk: `os.getlogin()` result is interpolated into an `icacls` command array. This is safe (array form, not shell string), but `os.getlogin()` can fail in non-interactive contexts (services, containers).
- Files: `src/cacheness/security.py` (lines 233–245)
- Impact: Key file permissions may not be set in headless Windows environments. Logged as warning, not fatal.

**No metadata encryption for SQLite:**
- Risk: SQLite metadata databases store cache keys, timestamps, data types, custom metadata, and encryption IVs in plaintext. Blob encryption protects blob content but metadata is still readable.
- Files: `src/cacheness/metadata/sqlite_backend.py`
- Mitigation: Use a separate encrypted filesystem, or wait for potential libSQL backend with built-in `encryption_key` support (see `docs/LIBSQL_BACKEND.md`).

---

## Performance Bottlenecks

**JSON backend full-file rewrite:**
- Problem: Every `put_entry()` triggers `_save_to_disk()` which serializes and writes the entire metadata dict.
- Files: `src/cacheness/metadata/json_backend.py` (lines 74–106)
- Cause: JSON format has no random-access update capability.
- Recommendation: Use SQLite backend for any cache with >200 entries.

**`_normalize_function_args` uses `inspect.signature` on every decorator call:**
- Problem: `inspect.signature(func)` is called on every cached function invocation to normalize positional/keyword arguments for consistent cache keys.
- Files: `src/cacheness/core.py` (lines 42–74)
- Impact: Measured overhead is small (~microseconds) but adds up in tight loops with many cached function calls.
- Improvement path: Could cache the signature per function (keyed by `id(func)`).

**Handler `can_handle()` chain:**
- Problem: `HandlerRegistry.get_handler()` iterates all registered handlers calling `can_handle()` until one matches. Some handlers import optional libraries in `can_handle()`.
- Files: `src/cacheness/handlers/registry.py`
- Impact: Negligible for typical use (5–8 handlers), but lazy TensorFlow import avoidance required special early-return checks in `can_handle()`.
- Mitigation: Handlers are priority-sorted. The common types (DataFrame, ndarray) match early.

---

## Fragile Areas

**Handler type detection ordering:**
- Files: `src/cacheness/handlers/registry.py`, individual handler `can_handle()` methods
- Why fragile: Priority-based handler selection means a new handler with a broad `can_handle()` can shadow specific handlers. TensorFlow tensor handler needed elaborate early returns to avoid matching numpy arrays.
- Safe modification: Always assign higher priority numbers (lower priority) to generic handlers. Test with all data types.
- Test coverage: `tests/test_handlers.py` covers priority and conflict detection.

**Mixin diamond inheritance in `UnifiedCache`:**
- Files: `src/cacheness/core.py` (lines 86–99)
- Why fragile: `UnifiedCache` inherits from 12 mixins. All mixins access `self.metadata_backend`, `self.blob_store`, `self._lock`, `self.signer`, and `self.config` via `self`. A mixin that accidentally shadows one of these attributes breaks all other mixins.
- Safe modification: Mixins should never define `__init__`. All shared state is initialized in `UnifiedCache.__init__`.
- Test coverage: Full test suite exercises all mixin methods through `UnifiedCache`.

**Signature version compatibility chain:**
- Files: `src/cacheness/security.py` (v1/v2/v3 + ns1/ns2 signatures)
- Why fragile: Verification must handle bare-hex (v1), versioned (v2), HKDF-derived (v3), and namespace (ns1/ns2) signatures. Adding v4 requires updating `parse_versioned_signature`, `SIGNED_FIELDS_BY_VERSION`, and verification key selection logic.
- Safe modification: Always add new versions, never modify existing version semantics. The version is embedded in the signature string.

---

## Missing Features

**Async/await support:**
- All operations are synchronous. No `AsyncUnifiedCache` class exists.
- Impact: Cannot efficiently use in async frameworks (FastAPI, aiohttp). Large blob I/O blocks the event loop.
- Files: No async code exists anywhere in `src/cacheness/`.
- Effort: High — requires async backends (asyncpg, aiosqlite, aioboto3), separate `AsyncUnifiedCache` class.
- Documented: `docs/FUTURE_IMPROVEMENTS.md` section 2.

**Advanced eviction policies:**
- Only TTL-based eviction exists. No LRU, LFU, or size-based eviction.
- Impact: Cannot bound cache size on disk, no access-pattern-aware eviction.
- Documented: `docs/FUTURE_IMPROVEMENTS.md` section 3.

**Tiered pull-through cache:**
- No composition of local + remote caches with automatic pull-through.
- Impact: Users must manually manage local/remote cache coordination.
- Status: Pending todo in `.planning/STATE.md`. Design documented in `docs/FUTURE_IMPROVEMENTS.md` section 9.

**CLI tool for cache inspection:**
- All cache operations require Python code. No `cacheness inspect`, `cacheness list`, `cacheness cleanup` commands.
- Impact: Debugging and maintenance require writing scripts.
- Documented: `docs/FUTURE_IMPROVEMENTS.md` section 4.

**Remaining management APIs:**
- `get_batch()`, `delete_batch()`, `touch()`, `get_metadata()`, `copy()`, `move()` not implemented.
- `put_batch()` and `delete_by_prefix()` shipped in v0.8.0–v0.9.0.
- Documented: `docs/FUTURE_IMPROVEMENTS.md` section 1.

---

## Platform-Specific Issues

**Windows:**
- TensorFlow handler tests hang and must always be ignored (`--ignore=tests/test_tensorflow_handler.py`).
- `os.getlogin()` in `security.py` can fail in Windows service/container contexts.
- `icacls` timeout is hardcoded to 10 seconds (`subprocess.run(..., timeout=10)`).
- Git commit messages with non-ASCII characters (em dashes, unicode) can cause terminal hangs.

**macOS:**
- No known issues.

**Linux:**
- No known issues.

---

## Test Coverage Gaps

**Property-based testing for cache key serialization:**
- What's not tested: Edge cases in `serialize_for_cache_key()` with complex nested objects, circular references, large collections.
- Files: `src/cacheness/serialization.py`
- Risk: Cache key collisions or instability with unusual data types.
- Priority: Medium — pending todo in `.planning/STATE.md`.
- Approach: Hypothesis-based property tests to verify determinism and collision resistance.

**TensorFlow handler integration:**
- What's not tested: Full integration on Windows (always skipped).
- Files: `tests/test_tensorflow_handler.py`
- Risk: Handler may be broken on Windows without detection.
- Priority: Low — handler is disabled by default.

**Concurrent encryption operations:**
- What's not tested: Multiple threads encrypting/decrypting simultaneously with key rotation in progress.
- Files: `src/cacheness/encryption.py`, `src/cacheness/core.py`
- Risk: Key rotation during concurrent encrypt operations could produce entries with mixed key states.
- Priority: Medium — thread safety tests exist for signing but not specifically for encryption + rotation.

---

*Concerns audit: 2026-04-07 (post v0.11.0)*
