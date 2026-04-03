# Codebase Concerns

**Analysis Date:** 2026-04-02

## Tech Debt

**No TODO/FIXME/HACK/XXX markers found in source code.**
A grep across `src/cacheness/**/*.py` (including ignored files) returned zero results for these comment markers, indicating either disciplined maintenance or that issues are tracked externally (beads issue tracker).

**JSON backend O(n²) scaling:**
- Issue: `JsonBackend._save_to_disk()` re-serializes the entire metadata file on every write operation.
- Files: `src/cacheness/metadata.py` (line ~1130)
- Impact: Write performance degrades quadratically with cache size. With 500+ entries, individual writes take seconds instead of milliseconds.
- Fix approach: Documented as a known limitation. Mitigation: switch to SQLite backend for >200 entries. No incremental write support planned for JSON.

**Broad `except Exception` usage:**
- Issue: Over 30 instances of `except Exception` across source files, particularly in `src/cacheness/core.py` (~15 instances), `src/cacheness/compress_pickle.py` (~10 instances), and `src/cacheness/custom_metadata.py` (~6 instances). Many swallow errors with only logging.
- Files: `src/cacheness/core.py`, `src/cacheness/compress_pickle.py`, `src/cacheness/custom_metadata.py`
- Impact: Can mask bugs, make debugging difficult, and silently degrade correctness. For example, `_init_auto_backend()` catches all exceptions when trying SQLite and silently falls back to JSON.
- Fix approach: Narrow exception types where possible; ensure swallowed exceptions are logged at DEBUG or WARNING level (most already are).

## Known Issues and Workarounds

**`get()` is destructive on errors (default behavior):**
- Issue: When `delete_on_error=True` (the default), `get()` auto-deletes entries that fail to load due to deserialization failures, corruption, or handler mismatches.
- Files: `src/cacheness/core.py` (lines ~2127, ~2268)
- Workaround: Set `CacheMetadataConfig(delete_on_error=False)` to preserve entries and return `None` instead. In `storage_mode=True`, entries are never deleted regardless of this setting.

**Orphaned blob files after crashes:**
- Issue: Hard crashes between blob write and metadata write leave orphaned blob files on disk. Cacheness writes blobs first, then metadata.
- Files: `src/cacheness/core.py` (`_PutCleanup` class), `src/cacheness/storage/blob_store.py`
- Workaround: Run `cache.verify_integrity(repair=True)` periodically to detect and clean orphaned blobs. The `_PutCleanup` class handles rollback during normal exceptions, but not process kills.

**Cache key collisions with `**kwargs`:**
- Issue: `_create_cache_key()` must carefully strip `prefix`, `description`, `custom_metadata`, `ttl_seconds` before hashing. Failure to strip new control parameters is a recurring bug pattern.
- Files: `src/cacheness/serialization.py`, `src/cacheness/core.py`
- Workaround: Critical bug pattern documented in copilot-instructions. All new parameters must be added to the strip list.

**`UnifiedCache._lock` consistently acquired (resolved in v0.9.0):**
- ~~Issue: `_lock` inconsistently acquired~~ — Investigation found `_lock` IS acquired in put/get and 22+ other methods. TROUBLESHOOTING.md and API_REFERENCE.md corrected.
- Files: `src/cacheness/core.py`, `docs/API_REFERENCE.md`, `docs/TROUBLESHOOTING.md`
- Status: **Resolved** — documentation was wrong, code was correct.

## Security Considerations

**Pickle/dill deserialization of untrusted data (documented in v0.9.0):**
- Risk: `pickle.loads()` and `dill.loads()` can execute arbitrary code during deserialization.
- Files: `src/cacheness/compress_pickle.py`, `src/cacheness/handlers/object_handler.py`
- Current mitigation: 3-layer defense: file_hash (xxhash) integrity check → HMAC-SHA256 metadata signing → signature verification before deserialization. Security comments added at all 5 deserialization sites.
- Status: **Documented** in `docs/SECURITY.md` "Deserialization Security" section with full threat model.

**Signing key file permissions — cross-platform (resolved in v0.9.0):**
- ~~Risk: `chmod(0o600)` is a no-op on Windows~~
- Files: `src/cacheness/security.py` (`_set_key_file_permissions` static method)
- Current mitigation: Cross-platform permission setting: Unix uses `chmod(0o600)`, Windows uses `icacls` to remove inheritance and grant only the current user `(R,W)`.
- Status: **Resolved** in v0.9.0.

**Shared signing key across namespaces:**
- Risk: All namespaces sharing the same `cache_dir` share the same signing key by default. No cryptographic isolation between tenants.
- Files: `src/cacheness/security.py`, config at `src/cacheness/config.py`
- Current mitigation: Documented in `docs/SECURITY.md`. Users can configure per-namespace `signing_key_file` names.
- Recommendations: Consider per-namespace key derivation (e.g., HKDF from master key + namespace ID) as a default.

**Fallback to in-memory key on disk errors:**
- Risk: If the signing key file cannot be written, `_generate_new_key()` silently falls back to an in-memory key. This means existing entries will fail verification, and new entries will be signed with a transient key.
- Files: `src/cacheness/security.py` (lines ~155-160)
- Current mitigation: Warning logged. The system continues to function.
- Recommendations: Consider raising an error instead of silently degrading, or at least make the fallback behavior configurable.

## Performance Concerns

**`core.py` is 3,307 lines — monolithic coordination module:**
- Problem: The main `UnifiedCache` class in `src/cacheness/core.py` is over 3,300 lines. It handles initialization, put, get, metadata management, blob storage delegation, entry signing, integrity verification, custom metadata, namespace management, statistics, and eviction.
- Files: `src/cacheness/core.py`
- Impact: Difficult to navigate, test, and modify. Changes in one area (e.g., eviction) risk breaking another (e.g., put/get). High merge-conflict risk.
- Improvement path: Extract coherent concerns into mixins or delegate classes (e.g., `_CacheVerifier`, `_CacheStatistics`).

**`metadata.py` is 2,562 lines — three backend implementations in one file:**
- Problem: JSON, SQLite, and PostgreSQL backend implementations coexist in a single file.
- Files: `src/cacheness/metadata.py`
- Impact: Lengthy file, hard to navigate. Backend-specific changes require working through unrelated code.
- Improvement path: Split into `metadata/json_backend.py`, `metadata/sqlite_backend.py`, `metadata/pg_backend.py` with shared base in `metadata/base.py`.

**JSON backend loads entire file into memory:**
- Problem: `_load_from_disk()` reads and parses the complete JSON metadata file on every backend initialization. Each write re-serializes the entire structure.
- Files: `src/cacheness/metadata.py` (lines ~1110-1130)
- Cause: JSON has no incremental update capability.
- Improvement path: Already mitigated by recommending SQLite for >200 entries. Not worth fixing in JSON backend.

## Fragile Areas

**Handler type detection ordering:**
- Files: `src/cacheness/handlers.py` (1,425 lines)
- Why fragile: `HandlerRegistry` iterates handlers in registration order and uses the first `can_handle()` match. Adding a new handler that matches broadly (e.g., anything with `.dtype`) can shadow existing handlers. The TensorFlow handler uses elaborate early-return checks to avoid accidentally matching numpy arrays.
- Safe modification: Always add specific handlers before generic ones. Test with all data types after adding/modifying handlers.
- Test coverage: `tests/test_handlers.py` provides good coverage, but edge cases with overlapping type detection are easy to miss.

**`_create_cache_key()` parameter stripping:**
- Files: `src/cacheness/serialization.py`, `src/cacheness/core.py`
- Why fragile: New named parameters added to `put()` must also be added to the strip list in `_create_cache_key()`. Missing a parameter causes it to be included in the cache key, breaking key stability.
- Safe modification: Add new parameter to the strip list AND add a test that verifies the key doesn't change when the parameter varies.
- Test coverage: Covered in `tests/test_core.py`, but regressions are easy if the checklist is missed.

**Atomic write pattern on Windows:**
- Files: `src/cacheness/metadata.py` (line ~1152, `shutil.move`), `src/cacheness/storage/backends/blob_backends.py` (line ~292)
- Why fragile: Atomic rename (`shutil.move`) is not truly atomic on Windows if source and destination are on different volumes. `NamedTemporaryFile` behavior differs between Windows and Unix. File handle cleanup requires explicit `gc.collect()` and sleep delays in tests.
- Safe modification: See `docs/WINDOWS_COMPATIBILITY.md` for patterns. Always use `delete=False` with `NamedTemporaryFile` on Windows.
- Test coverage: Platform-specific fixtures in `tests/conftest.py` handle cleanup ordering.

## Missing Features / Incomplete Implementations

**TensorFlow handler disabled:**
- What's missing: The `TensorFlowTensorHandler` is fully implemented but disabled due to system compatibility issues. TF tests hang on Windows and must always be ignored.
- Files: `src/cacheness/handlers.py` (line ~74, lazy import), `docs/TENSORFLOW_HANDLER_STATUS.md`
- Impact: TensorFlow tensor caching is not reliably available. Users must handle TF serialization manually.
- Priority: Low — TF is a heavy optional dependency.

**No async/await support:**
- What's missing: All cache operations are synchronous. No `AsyncUnifiedCache` exists.
- Files: Documented in `docs/FUTURE_IMPROVEMENTS.md`
- Impact: Cannot efficiently integrate with async web frameworks (FastAPI, aiohttp). Concurrent I/O operations cannot overlap.
- Priority: High — documented as a key future improvement.

**No advanced eviction policies:**
- What's missing: Only TTL-based eviction. No LRU, LFU, or size-based eviction.
- Files: `src/cacheness/core.py` (eviction logic scattered through `_cleanup_expired()`)
- Impact: Cannot bound cache size by disk usage or entry count. Users must manage eviction externally.
- Priority: Medium-High — documented in `docs/FUTURE_IMPROVEMENTS.md`.

**Management operations (resolved in v0.9.0):**
- ~~What's missing: Various management APIs~~ — All APIs now implemented: `put()`, `get()`, `update_data()`, `touch()`, `get_metadata()`, `put_batch()`, `get_batch()`, `delete_batch()`, `touch_batch()`, `delete_by_prefix()`, `delete_where()`, `delete_matching()`.
- Status: **Resolved** — `put_batch()` added in v0.9.0, all others already existed.

## Platform-Specific Issues

**Windows file locking:**
- Issue: Windows has stricter file locking semantics. SQLite database files cannot be deleted while connections are open. `NamedTemporaryFile` keeps file handles open by default.
- Files: `src/cacheness/core.py` (context manager / `close()`), `src/cacheness/metadata.py` (SQLite backend disposal)
- Impact: Test fixtures require explicit cleanup ordering with `gc.collect()` and sleep delays. Documented extensively in `docs/WINDOWS_COMPATIBILITY.md`.

**TensorFlow tests hang on Windows:**
- Issue: TF handler tests must always be ignored on Windows (`--ignore=tests/test_tensorflow_handler.py`). The tests hang indefinitely.
- Files: `tests/test_tensorflow_handler.py`
- Impact: TF handler cannot be validated on Windows CI.

**`chmod(0o600)` is a no-op on Windows (resolved in v0.9.0):**
- ~~Issue: Signing key file permissions are not actually restricted on Windows.~~
- Status: **Resolved** — Now uses `icacls` on Windows. See Security Considerations section.

## Dependency Risks

**TensorFlow (optional, heavy):**
- Risk: TensorFlow is a ~500MB+ dependency that can cause system-level issues (GPU driver conflicts, protobuf version incompatibilities). Import is extremely slow.
- Files: `src/cacheness/handlers.py` (lazy import at line ~79)
- Impact: Lazy import mitigates startup cost, but the handler is effectively disabled on Windows.
- Migration plan: Already isolated as optional dependency group. Consider removing from core handlers and making it a plugin.

**SQLAlchemy dependency for SQLite/PostgreSQL:**
- Risk: SQLAlchemy is a large ORM that adds complexity. Using raw `sqlite3` would be lighter for the SQLite backend.
- Files: `src/cacheness/metadata.py` (SQLite backend), `src/cacheness/storage/backends/postgresql_backend.py`
- Impact: Not a significant risk — SQLAlchemy is stable and well-maintained. Provides useful abstractions for schema migration.
- Migration plan: None needed. Core dependency for production backends.

**Blosc2 for compression:**
- Risk: `blosc2` has C extensions that can fail to build on some platforms. Binary wheels may not be available for all architectures.
- Files: `src/cacheness/handlers.py`, `src/cacheness/compress_pickle.py`
- Impact: Graceful degradation — code checks `BLOSC2_AVAILABLE` and falls back to pickle or lz4. NumPy array compression unavailable without blosc2.
- Migration plan: Already handled via optional dependency group and availability checks.

**`dill` serialization risks:**
- Risk: Dill extends pickle's attack surface. Cached objects may become incompatible across Python versions or if class definitions change (silent data corruption or crashes).
- Files: `src/cacheness/handlers.py` (line ~1226), `src/cacheness/compress_pickle.py` (line ~284)
- Impact: Documented extensively in `docs/DILL_INTEGRATION.md` with security warnings.
- Migration plan: Dill is optional. Users can disable it via handler configuration.

## Areas That Need Refactoring

**`core.py` decomposition (3,307 lines):**
- Files: `src/cacheness/core.py`
- Why: Single file contains initialization, CRUD operations, verification, statistics, eviction, namespace management, custom metadata support, and storage mode delegation. This is the most complex file in the codebase by a wide margin.
- Suggested approach: Extract `_StorageModeMixin` (lines ~1678+), `_VerificationMixin`, `_StatisticsMixin`, and `_CustomMetadataMixin` to reduce the class to ~1,500 lines.

**`metadata.py` backend separation (2,562 lines):**
- Files: `src/cacheness/metadata.py`
- Why: Three complete backend implementations (JSON, SQLite, PostgreSQL) in one file with a shared base class. Each backend is 500-800 lines.
- Suggested approach: Create `metadata/` package with `base.py`, `json_backend.py`, `sqlite_backend.py`, and re-export from `metadata/__init__.py`.

**`handlers.py` handler isolation (1,425 lines):**
- Files: `src/cacheness/handlers.py`
- Why: All handler implementations (Array, DataFrame, Series, Object, Dill, Inline, Raw, TensorFlow) in one file.
- Suggested approach: Create `handlers/` package with one file per handler. The `HandlerRegistry` stays in `handlers/__init__.py`.

## Test Coverage Gaps

**Thread safety under concurrent access:**
- What's not tested: No tests exercise concurrent `put()`/`get()` from multiple threads against `UnifiedCache` directly. Backend-level thread safety is implicitly tested.
- Files: No dedicated concurrency test file exists.
- Risk: Data races in `UnifiedCache` methods that don't acquire `_lock` (put, get).
- Priority: Medium — thread safety is documented as limited, but users may assume it works.

**Cross-platform atomic writes:**
- What's not tested: No tests verify that `shutil.move()` atomic rename actually prevents corruption on Windows when the temp file and destination are on different volumes.
- Files: `src/cacheness/metadata.py`, `src/cacheness/storage/backends/blob_backends.py`
- Risk: Low — temp files are created in the same directory as the destination.
- Priority: Low.

**Key rotation scenarios:**
- What's not tested: No test exercises key rotation (deleting key file, restarting, and verifying that entries signed with the old key are properly handled).
- Files: `src/cacheness/security.py`
- Risk: Users following the documented key rotation procedure may encounter unexpected behavior.
- Priority: Medium.

---

*Concerns audit: 2026-04-02*
