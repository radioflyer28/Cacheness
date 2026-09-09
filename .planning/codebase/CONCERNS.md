# Codebase Concerns

**Analysis Date:** 2026-08-29

**Independent Review:** 2026-08-29 — findings were re-validated with direct source tracing, runtime probes, pytest, Ruff, and coverage

## Tech Debt

**Base-package dependency metadata is internally inconsistent:**
- Issue: NumPy is listed only in optional groups, but is imported unconditionally by `src/cacheness/handlers.py` and `src/cacheness/compress_pickle.py`, both reached during `import cacheness`.
- Files: `pyproject.toml`, `src/cacheness/__init__.py`, `src/cacheness/handlers.py:9`, `src/cacheness/compress_pickle.py:50`
- Impact: A clean installation with only the two declared core dependencies can fail before any optional array feature is requested. Development succeeds because the lock/environment installs recommended dependencies, masking the packaging defect.
- Fix approach: Declare NumPy as a core dependency or isolate every NumPy-dependent import/export behind a lazy optional boundary; add a wheel-level minimal-install smoke test.

**Configured quality gates are not release gates:**
- Issue: The repository has pytest, coverage, and Ruff configuration but no CI workflow. The independent baseline is 2 failed tests, 137 repository-wide Ruff findings, and 66% statement coverage with no `fail_under` threshold.
- Files: `pyproject.toml`, `tests/`, `.planning/codebase/TESTING.md`
- Impact: Regressions and compatibility failures can ship unless a maintainer happens to run the relevant local commands and interpret the existing failures.
- Fix approach: Add a Python-version matrix that runs tests, a minimal-install import job, optional-backend jobs, Ruff, and an initially realistic coverage floor that increases over time.

**Split storage abstractions:**
- Issue: `CacheBlobConfig` and the filesystem/S3 blob backend registry exist, but `UnifiedCache` in `src/cacheness/core.py` writes directly through handlers and never constructs or uses a blob backend. `BlobStore` in `src/cacheness/storage/blob_store.py` has a separate storage path and does not accept a blob backend instance.
- Files: `src/cacheness/config.py`, `src/cacheness/core.py`, `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/backends/blob_backends.py`, `src/cacheness/storage/backends/s3_backend.py`
- Impact: Configuring `blob_backend="s3"` does not move normal cache payloads to S3; the compatibility validation gives users an apparently supported configuration that the main cache does not honor.
- Fix approach: Choose one storage abstraction, inject it into `UnifiedCache`/`BlobStore`, and add an integration test that proves writes, reads, invalidation, and cleanup use the selected backend.

**Configuration flags are not connected to runtime behavior:**
- Issue: `enable_metadata`, `auto_cleanup_expired`, `CacheStorageConfig.verify_cache_integrity`, `create_cache_dir`, `temp_dir`, `use_atomic_writes`, `create_subdirectories`, and `enable_parallel_compression` are defined or mapped but are not consistently consulted by the runtime. `UnifiedCache` always creates the directory and only checks `metadata.verify_cache_integrity`; initialization cleanup is controlled by `storage.cleanup_on_init` instead of `metadata.auto_cleanup_expired`.
- Files: `src/cacheness/config.py`, `src/cacheness/core.py`, `src/cacheness/handlers.py`
- Impact: Users can believe safety, persistence, atomicity, or performance settings are active when they are ignored.
- Fix approach: Remove unsupported options or make each option authoritative, then test every public configuration option through the actual write/read lifecycle.

**Configuration validation and serialization drift:**
- Issue: `CacheConfig` applies several legacy and factory overrides after sub-config `__post_init__` validation and does not revalidate them. Nested YAML loading also normalizes relative paths through `CacheStorageConfig`, while direct legacy construction preserves them.
- Files: `src/cacheness/config.py:404-540`, `src/cacheness/config.py:657-712`, `src/cacheness/config.py:1032-1071`, `tests/test_config_validation.py:622-658`
- Impact: Invalid overrides can fail only at runtime, and loading/saving a valid YAML config changes `./yaml_cache` into an absolute path. The current suite reports two failures in `TestYamlConfig`.
- Fix approach: Centralize path normalization and run strict validation after all overrides; preserve the chosen path representation across JSON/YAML round trips.

**Schema and model migration is absent:**
- Issue: SQLite and PostgreSQL tables are created with `create_all`, but there is no versioned migration mechanism for adding columns/indexes or reconciling databases created by older releases.
- Files: `src/cacheness/metadata.py:1034-1038`, `src/cacheness/storage/backends/postgresql_backend.py:236-240`, `pyproject.toml`
- Impact: Existing cache databases can fail when new metadata columns are inserted, or can retain an incompatible schema with no guided recovery.
- Fix approach: Add an explicit schema version and migrations (or document a safe rebuild/export-import procedure) for both SQL backends.

**Large, overlapping coordination modules:**
- Issue: Core responsibilities are concentrated in `src/cacheness/core.py` (1,238 lines), `src/cacheness/metadata.py` (1,514 lines), `src/cacheness/handlers.py` (1,349 lines), and `src/cacheness/sql_cache.py` (1,816 lines), with compatibility implementations duplicated across `src/cacheness/metadata.py` and `src/cacheness/storage/backends/`.
- Files: `src/cacheness/core.py`, `src/cacheness/metadata.py`, `src/cacheness/handlers.py`, `src/cacheness/sql_cache.py`, `src/cacheness/storage/backends/__init__.py`
- Impact: Backend behavior, cleanup ownership, and metadata shapes can diverge; changes require checking several parallel APIs.
- Fix approach: Separate metadata lifecycle, payload lifecycle, eviction, and compatibility adapters into small modules with one canonical entry shape.

## Known Bugs

**Injected and registered custom metadata backends are bypassed:**
- Symptoms: `UnifiedCache(metadata_backend=instance)` assigns the supplied backend and then overwrites it with JSON/memory/SQLite/PostgreSQL/auto selection. Likewise, a custom name registered through `register_metadata_backend()` is never resolved by `UnifiedCache`; an unrecognized config name falls into auto mode.
- Files: `src/cacheness/core.py:109-212`, `src/cacheness/storage/backends/__init__.py:127-266`, `src/cacheness/metadata.py:1421-1514`
- Trigger: Supply `InMemoryBackend()` with default config; the independent runtime probe returned `SqliteBackend False sqlite`, proving the active backend is not the supplied object.
- Workaround: There is no reliable high-level injection path. Select one of the literal built-in config names, or use backend APIs directly outside `UnifiedCache`.

**YAML capability reporting is a false positive:**
- Symptoms: `src/cacheness/__init__.py` sets `_has_yaml_config=True` by importing helper functions whose module does not import PyYAML until invocation. The helpers may therefore be exported even though calling them raises an install-time `ImportError`.
- Files: `src/cacheness/__init__.py:56-61`, `src/cacheness/config.py:1032-1144`, `pyproject.toml`
- Trigger: Install without PyYAML, import `cacheness`, then call `load_config_from_yaml()` or `save_config_to_yaml()`.
- Workaround: Install PyYAML explicitly; it is present in the current lock only transitively and is not a declared extra.

**Size-limit enforcement crashes once the limit is exceeded:**
- Symptoms: A `put()` that pushes the cache over `max_cache_size_mb` raises `AttributeError: '<backend>' object has no attribute 'cleanup_by_size'` after the payload and metadata have already been written.
- Files: `src/cacheness/core.py:1055-1071`, `src/cacheness/metadata.py:1421-1514`, `src/cacheness/storage/backends/base.py`
- Trigger: Use a small limit and store an incompressible array or other payload larger than that limit.
- Workaround: Leave the limit at a value that is never reached, or manually delete entries; no backend implements the method called by the coordinator.

**Metadata cleanup does not remove payload files:**
- Symptoms: Expired entries and invalidated entries disappear from metadata but their `.pkl`, `.npz`, `.b2nd`, or `.parquet` files remain. `UnifiedCache.invalidate()` only calls `metadata_backend.remove_entry()`, and `_cleanup_expired()` only calls the backend metadata cleanup.
- Files: `src/cacheness/core.py:803-806`, `src/cacheness/core.py:1055-1071`, `src/cacheness/metadata.py:789-794`, `src/cacheness/metadata.py:904-931`, `src/cacheness/metadata.py:1350-1363`
- Trigger: Call `invalidate()`, wait for TTL cleanup, or restart with `cleanup_on_init=True`.
- Workaround: Periodically scan the cache directory and reconcile files against metadata; `clear_all()` removes only known top-level payload extensions.

**`BlobStore` deletion and existence checks use the wrong metadata level:**
- Symptoms: JSON and SQLite backends store `actual_path` under `entry["metadata"]`, but `BlobStore.delete()` and `BlobStore.exists()` read only the top-level field. `delete()` reports success while leaving the payload, and `exists()` returns false for a live blob.
- Files: `src/cacheness/storage/blob_store.py:196-229`, `src/cacheness/storage/blob_store.py:285-326`
- Trigger: Use `BlobStore(..., backend="json")` or SQLite, put an object, then call `exists()` or `delete()`.
- Workaround: Inspect `get_metadata(key)["metadata"]["actual_path"]` and remove the file explicitly.

**`BlobStore.clear()` clears metadata but not payloads:**
- Symptoms: `clear()` delegates to `backend.clear_all()` and never deletes the files recorded in the entries.
- Files: `src/cacheness/storage/blob_store.py:367-375`, `src/cacheness/metadata.py:933-947`, `src/cacheness/metadata.py:1365-1385`
- Trigger: Store one or more blobs and call `clear()`.
- Workaround: Retain a separate payload inventory and delete files before clearing metadata.

**Decorator cache management is incomplete and `None` results are not cached:**
- Symptoms: `wrapper.cache_clear()` always returns `0` and does not remove entries. A function returning `None` is recomputed on every call because the wrapper treats `cache.get(...) is None` as a miss.
- Files: `src/cacheness/decorators.py:171-176`, `src/cacheness/decorators.py:220-227`, `src/cacheness/core.py:927-1040`
- Trigger: Use `@cached` on a function returning `None`, or call its attached `cache_clear()` method.
- Workaround: Use a non-`None` sentinel result and explicitly invalidate known keys.

**SQL pull-through cache can return incomplete data as a successful result:**
- Symptoms: Fetch failures are printed and swallowed, after which `get_data()` returns whatever rows are already cached. Internal gaps are explicitly treated as absent in `_find_internal_gaps()`.
- Files: `src/cacheness/sql_cache.py:502-529`, `src/cacheness/sql_cache.py:798-820`, `src/cacheness/sql_cache.py:1028-1055`, `src/cacheness/sql_cache.py:1148-1159`
- Trigger: Make the adapter fail for one missing range, or cache rows spanning a missing interval inside a requested time range.
- Workaround: Supply a strict custom `gap_detector` and validate returned coverage in the caller.

**PostgreSQL statistics and custom metadata paths are inconsistent:**
- Symptoms: `PostgresBackend.get_stats()` returns denormalized `PgCacheStats.total_entries` and `total_size_bytes`, but normal put/remove paths never update those columns. Custom metadata migration is also called with a `UnifiedCache` in the existing test and logs an `_run_ddl_visitor` error because the function expects an engine.
- Files: `src/cacheness/storage/backends/postgresql_backend.py:149-160`, `src/cacheness/storage/backends/postgresql_backend.py:307-442`, `src/cacheness/storage/backends/postgresql_backend.py:467-491`, `src/cacheness/custom_metadata.py:398-464`, `tests/test_custom_metadata.py:460-472`
- Trigger: Use PostgreSQL statistics or invoke `migrate_custom_metadata_tables(cache)` as the test/API path does.
- Workaround: Query `PgCacheEntry` directly and pass `cache.metadata_backend.engine` to migration helpers.

## Security Considerations

**Untrusted cache files can execute code during deserialization:**
- Risk: General objects use pickle or dill, arrays are loaded with `allow_pickle=True`, and the Blosc2 array header is converted with `eval()`. Any process that can alter cache files or metadata can achieve code execution when a cache entry is read.
- Files: `src/cacheness/handlers.py:562-577`, `src/cacheness/handlers.py:593-600`, `src/cacheness/handlers.py:982-1026`, `src/cacheness/compress_pickle.py:764-837`
- Current mitigation: HMAC metadata signing and XXH3 file hashes are enabled by default, but `allow_unsigned_entries=True` is the default and users can disable integrity checks.
- Recommendations: Replace `eval` with a safe shape parser, load NPZ files with `allow_pickle=False` unless explicitly required, document pickle as trusted-input-only, and require strict signed/integrity-checked entries for deployments exposed to other users.

**Signing configuration can silently degrade to unsigned operation:**
- Risk: Signer initialization catches every exception and sets `self.signer=None`; signing also allows unsigned entries by default. With an unreadable or replaced persistent key, the cache can continue without the intended metadata-authenticity guarantee.
- Files: `src/cacheness/core.py:258-284`, `src/cacheness/security.py:63-116`, `src/cacheness/config.py:298-314`
- Current mitigation: The key file is written with mode `0600`, and invalid signatures can be deleted.
- Recommendations: Fail closed when signing is required, default `allow_unsigned_entries=False` for new caches, and make key rotation/backup an explicit operational workflow.

**`query_meta()` interpolates user-controlled JSON-path keys into SQL:**
- Risk: Filter values are bound, but filter keys are directly embedded in the `JSON_EXTRACT` expression. A key containing quotes or SQL syntax can break the query and cause a denial of service; the current test covers malicious values, not malicious keys.
- Files: `src/cacheness/core.py:535-651`, `tests/test_query_meta.py:322-344`
- Current mitigation: Values use SQLAlchemy bind parameters and the method catches the resulting database error.
- Recommendations: Whitelist valid parameter names from stored schema, escape JSON paths, or use a structured SQLAlchemy JSON expression rather than interpolating keys.

**Filesystem blob path containment is not enforced at the backend boundary:**
- Risk: `FilesystemBlobBackend._get_blob_path()` rewrites `..` but allows absolute path components when called directly, and `read_blob()`/`delete_blob()` trust arbitrary paths returned by callers.
- Files: `src/cacheness/storage/backends/blob_backends.py:232-268`, `src/cacheness/storage/backends/blob_backends.py:306-324`
- Current mitigation: `BlobStore._sanitize_key()` strips user key characters before normal BlobStore writes.
- Recommendations: Resolve paths and enforce that they remain below `base_dir` for every filesystem operation, including direct backend use.

## Performance Bottlenecks

**SQLite statistics scan all metadata rows on every stats request and size check:**
- Problem: `SqliteBackend.get_stats()` materializes every `CacheEntry` to count types and sum sizes. `UnifiedCache.put()` calls it before every size enforcement decision.
- Files: `src/cacheness/metadata.py:1273-1314`, `src/cacheness/core.py:905`, `src/cacheness/core.py:1055-1068`
- Cause: Aggregate SQL queries are replaced with Python list materialization.
- Improvement path: Use `COUNT`, `SUM`, and grouped aggregate queries; make size enforcement use a dedicated indexed query and a working eviction API.

**JSON backend rewrites the complete metadata document for hot-path operations:**
- Problem: `put_entry()`, access-time updates, and hit/miss counters all call `_save_to_disk()`, serializing and atomically replacing the full JSON file.
- Files: `src/cacheness/metadata.py:697-727`, `src/cacheness/metadata.py:758-787`, `src/cacheness/metadata.py:884-902`
- Cause: The nominal batching fields are present, but the non-batch branch still saves immediately and `_pending_writes` is never populated.
- Improvement path: Prefer SQLite for production, or implement real append/batch semantics with bounded flushes and a lock strategy for multiple processes.

**Key generation can perform expensive full-content work synchronously:**
- Problem: Path parameters hash entire files/directories, and DataFrame parameters materialize values for hashing; this occurs inline before every cache lookup.
- Files: `src/cacheness/serialization.py:8-61`, `src/cacheness/serialization.py:63-119`, `src/cacheness/file_hashing.py:51-117`
- Cause: Content-based identity is the default and directory scans can spawn up to eight processes.
- Improvement path: Offer explicit caller-provided version keys, cache content fingerprints, and make expensive hashing opt-in for hot request paths.

## Fragile Areas

**Same-key concurrent writes are not coordinated at the payload layer:**
- Files: `src/cacheness/core.py:82-102`, `src/cacheness/core.py:822-905`, `src/cacheness/handlers.py:448-553`, `src/cacheness/storage/backends/blob_backends.py:232-245`
- Why fragile: `UnifiedCache._lock` is created but not used to guard `put()`/`get()`. Handlers write directly to the final filename, while filesystem blob writes use one fixed `.tmp` name per blob. Concurrent writers can interleave file and metadata updates.
- Safe modification: Use unique temporary files, atomic replace only after a complete write/hash, and serialize or compare-and-swap updates for a single cache key.
- Test coverage: Concurrency tests use distinct keys and do not assert same-key payload/metadata consistency; see `tests/test_sqlite_concurrency.py`.

**Resource finalization emits interpreter-shutdown errors:**
- Files: `src/cacheness/metadata.py:1397-1409`, `.venv/lib/python3.13/site-packages/sqlalchemy/engine/base.py`, `.venv/lib/python3.13/site-packages/sqlalchemy/pool/impl.py`
- Why fragile: The full test run and direct probes emit `Exception ignored in ... SqliteBackend.__del__` with `TypeError: catching classes that do not inherit from BaseException` while SQLAlchemy disposes during Python 3.13 teardown.
- Safe modification: Make `__del__` minimal and guarded, rely on explicit context-manager/`close()` ownership, and pin/test compatible SQLAlchemy/Python versions.
- Test coverage: No assertion checks clean finalization under Python 3.13; the warning appears after otherwise successful tests.

**Custom metadata relies on global registration and swallowed errors:**
- Files: `src/cacheness/custom_metadata.py:94-104`, `src/cacheness/custom_metadata.py:204-272`, `src/cacheness/core.py:284-422`, `src/cacheness/custom_metadata.py:462-517`
- Why fragile: Registry state is process-global, duplicate schema registration overwrites the previous class, and storage/query/migration helpers catch broad exceptions and return empty results.
- Safe modification: Scope registries to an application/cache instance where possible, reject duplicate schema names, and surface migration/storage failures as typed errors.
- Test coverage: Most tests assert graceful return values rather than persistence failure behavior; PostgreSQL custom metadata is not exercised.

**SQLCache error reporting is inconsistent:**
- Files: `src/cacheness/sql_cache.py:502-529`, `src/cacheness/sql_cache.py:620-626`, `src/cacheness/sql_cache.py:736-743`
- Why fragile: Library operations use `print()` for fetch, upsert, and gap-detector failures while other modules use logging and typed exceptions; callers cannot reliably distinguish partial success from a complete result.
- Safe modification: Use structured logger calls, preserve the original exception context, and expose a strict/partial-result policy.
- Test coverage: Tests mostly exercise successful adapters and do not assert the caller-visible behavior of partial fetch failure.

## Scaling Limits

**64-bit cache identifiers and single-file JSON metadata:**
- Current capacity: Cache keys are truncated to 16 hexadecimal characters in `src/cacheness/serialization.py:140-160`, and BlobStore content-addressable keys are truncated similarly in `src/cacheness/storage/blob_store.py:389-397`.
- Limit: Collision probability grows with large key populations, while JSON metadata requires whole-document reads/writes and becomes increasingly expensive as entries accumulate.
- Scaling path: Use a longer collision-resistant identifier (or collision verification), SQLite/PostgreSQL metadata, and indexed/partitioned cleanup for large caches.

**Local SQLite is not a distributed payload store:**
- Current capacity: `SqliteBackend` uses a local file and `FilesystemBlobBackend` uses local paths.
- Limit: Multiple hosts cannot safely share payload paths, and PostgreSQL metadata does not make filesystem blobs available on other machines.
- Scaling path: Complete the remote blob integration and store backend-neutral blob URLs/checksums in metadata.

## Dependencies at Risk

**Python 3.13 / SQLAlchemy finalization compatibility:**
- Risk: The supported runtime floor is `>=3.11` with no upper bound, while the current Python 3.13 environment produces teardown errors from SQLAlchemy disposal.
- Impact: Resource warnings can hide real cleanup failures and indicate an unsupported dependency combination.
- Migration plan: Add a supported-version matrix, pin compatible SQLAlchemy releases in CI, and harden backend finalizers.
- Files: `pyproject.toml`, `src/cacheness/metadata.py:1397-1409`

**Optional backend coverage is incomplete:**
- Risk: PostgreSQL tests are skipped when `psycopg` is unavailable and TensorFlow tests are intentionally skipped because imports can freeze the system.
- Impact: Production-only backends and the lazy TensorFlow path can regress without CI detection.
- Migration plan: Add isolated service/container jobs for PostgreSQL and a safe TensorFlow compatibility job, or explicitly remove unsupported backends from the supported matrix.
- Files: `pyproject.toml`, `tests/test_postgresql_backend.py`, `tests/test_tensorflow_handler.py`

## Missing Critical Features

**Reliable eviction and reconciliation:**
- Problem: The public `max_cache_size_mb` feature has no backend eviction implementation, and TTL/invalid-signature cleanup removes only metadata.
- Blocks: Predictable disk usage, long-running cache processes, and safe recovery from orphaned files.
- Files: `src/cacheness/core.py:803-806`, `src/cacheness/core.py:1055-1071`, `src/cacheness/metadata.py`, `src/cacheness/storage/blob_store.py`

**Backend-neutral payload lifecycle:**
- Problem: There is no complete implementation connecting metadata, payload writes, integrity checks, eviction, and deletion across filesystem, memory, and S3 backends.
- Blocks: Distributed deployment and safe use of the configured cloud backends.
- Files: `src/cacheness/config.py:103-161`, `src/cacheness/core.py`, `src/cacheness/storage/backends/blob_backends.py`, `src/cacheness/storage/backends/s3_backend.py`

## Test Coverage Gaps

**Minimal install and dependency extras:**
- What's not tested: Building/installing the package with only mandatory dependencies and importing the public package, plus exercising each extra in isolation.
- Files: `pyproject.toml`, `src/cacheness/__init__.py`, `src/cacheness/handlers.py`, `src/cacheness/compress_pickle.py`
- Risk: Missing mandatory dependencies and misleading optional-feature flags remain hidden by the fully populated development environment.
- Priority: High

**Metadata backend injection and custom registry selection:**
- What's not tested: Identity preservation for an injected backend and end-to-end selection of a registered custom backend name.
- Files: `src/cacheness/core.py:109-212`, `tests/test_metadata_backend_registry.py`
- Risk: A documented extensibility mechanism is unusable from the primary cache while registry unit tests still pass.
- Priority: High

**Eviction and orphan cleanup:**
- What's not tested: An over-limit write that must evict entries, and deletion of payload files during TTL cleanup, invalidation, and size cleanup.
- Files: `src/cacheness/core.py:1055-1071`, `tests/test_integration.py:196-218`
- Risk: A missing method currently fails only when the test payload actually exceeds the configured limit; orphaned files accumulate unnoticed.
- Priority: High

**BlobStore lifecycle semantics:**
- What's not tested: `BlobStore.exists()`, `delete()`, and `clear()` against JSON/SQLite nested `actual_path` metadata.
- Files: `src/cacheness/storage/blob_store.py:285-375`, `tests/test_blob_backend_registry.py`
- Risk: Consumers receive false negatives and leaked files while metadata appears healthy.
- Priority: High

**Security parser and query-key cases:**
- What's not tested: Malicious Blosc2 shape headers, NPZ object arrays with `allow_pickle=True`, and SQL-injection/syntax cases in filter keys.
- Files: `src/cacheness/handlers.py:555-600`, `src/cacheness/core.py:585-601`, `tests/test_query_meta.py:322-344`
- Risk: Crafted cache artifacts can execute code or make metadata queries fail.
- Priority: High

**Optional backend and portability paths:**
- What's not tested: PostgreSQL integration in the default environment, custom metadata on PostgreSQL, and clean teardown under the supported Python version range.
- Files: `tests/test_postgresql_backend.py`, `tests/test_custom_metadata.py:460-479`, `tests/test_tensorflow_handler.py`, `src/cacheness/metadata.py:1397-1409`
- Risk: Release regressions can ship in advertised optional integrations.
- Priority: Medium

---

*Concerns audit: 2026-08-29*
