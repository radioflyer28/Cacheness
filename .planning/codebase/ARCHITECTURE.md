# Architecture

**Analysis Date:** 2026-04-07

## Pattern Overview

**Overall:** Mixin-based Composition with Strategy Pattern and Pluggable Backends

**Key Characteristics:**
- `UnifiedCache` composes 11 mixins via multiple inheritance (MRO-safe, no diamond conflicts)
- Strategy pattern for type-aware serialization via `HandlerRegistry`
- Pluggable metadata backends (JSON, SQLite, PostgreSQL) behind `MetadataBackend` ABC
- Pluggable blob backends (filesystem, S3, in-memory) behind `BlobBackend` ABC
- Two-tier API: `UnifiedCache` (cache semantics) wraps `BlobStore` (raw storage)
- AES-256-GCM encryption pipeline between handler output and blob write

## Layers

**Public API (`__init__.py`, `core.py`, `decorators.py`):**
- Purpose: User-facing cache operations and function decorators
- Location: `src/cacheness/__init__.py`, `src/cacheness/core.py`, `src/cacheness/decorators.py`
- Contains: `UnifiedCache` class, `cached`/`cache_if` decorators, `get_cache()` factory
- Depends on: Mixins, config, handlers, metadata, storage, serialization
- Used by: Application code, tests

**Mixins (`_*_mixin.py`, `_put_cleanup.py`):**
- Purpose: Decompose `UnifiedCache` into single-responsibility concerns
- Location: `src/cacheness/_verification_mixin.py` (261 lines), `src/cacheness/_stats_mixin.py` (26 lines), `src/cacheness/_custom_metadata_mixin.py` (315 lines), `src/cacheness/_storage_mode_mixin.py` (170 lines), `src/cacheness/_query_mixin.py` (247 lines), `src/cacheness/_convenience_mixin.py` (328 lines), `src/cacheness/_batch_mixin.py` (264 lines), `src/cacheness/_file_ops_mixin.py` (182 lines), `src/cacheness/_get_variants_mixin.py` (169 lines), `src/cacheness/_update_mixin.py` (315 lines), `src/cacheness/_inline_blob_mixin.py` (232 lines), `src/cacheness/_put_cleanup.py` (58 lines)
- Contains: Verification/signing, statistics, custom metadata, storage-mode passthrough, queries, convenience helpers, batch ops, file ops, get variants, update/exists/touch, inline blob logic, rollback guard
- Depends on: `interfaces`, `config`, `error_handling`, `entry_list`
- Used by: `UnifiedCache` (via MRO)

**Handlers (`handlers/`):**
- Purpose: Type-aware serialization — detect data type, write optimized format, read back
- Location: `src/cacheness/handlers/`
- Contains: `HandlerRegistry` (priority-based handler selection), 7 type handlers:
  - `PandasDataFrameHandler` → Parquet via pyarrow
  - `PandasSeriesHandler` → Parquet (wraps in DataFrame)
  - `PolarsDataFrameHandler` → Parquet via polars native
  - `PolarsSeriesHandler` → Parquet (wraps in DataFrame)
  - `ArrayHandler` → blosc2 compressed `.npz`
  - `BytesHandler` → raw bytes (no serialization overhead)
  - `ObjectHandler` → pickle/dill with blosc2 compression
  - `TensorFlowTensorHandler` → blosc2 tensor format (disabled by default)
- Depends on: `interfaces.CacheHandler` ABC, `_compat.py` (shared imports, optional dep flags)
- Used by: `BlobStore._write_blob()`, `BlobStore._read_blob()`

**Metadata (`metadata/`):**
- Purpose: Pluggable metadata storage — entry CRUD, namespace isolation, schema migrations
- Location: `src/cacheness/metadata/`
- Contains:
  - `base.py` (578 lines): `MetadataBackend` ABC, `CachedMetadataBackend` wrapper
  - `json_backend.py` (632 lines): JSON file backend
  - `sqlite_backend.py` (1122 lines): SQLite + SQLAlchemy ORM backend
  - `_compat.py` (268 lines): ORM models (`CacheEntryMixin`, `CacheStatsMixin`, `CacheNamespace`), namespace utilities, migration definitions
  - `__init__.py` (148 lines): `create_metadata_backend()` factory
- Depends on: `interfaces.EntrySummary`, SQLAlchemy (optional), cachetools (optional)
- Used by: `UnifiedCache`, `BlobStore`, PostgreSQL backend

**Storage (`storage/`):**
- Purpose: Low-level blob I/O, backend abstraction, compression, path normalization
- Location: `src/cacheness/storage/`
- Contains:
  - `blob_store.py` (1199 lines): `BlobStore` — standalone blob storage API
  - `compression.py` (49 lines): Re-exports from `compress_pickle.py`
  - `paths.py` (41 lines): `resolve_actual_path()`, `to_relative_path()`
  - `security.py` (23 lines): Re-exports `CacheEntrySigner`
  - `handlers/__init__.py` (86 lines): Re-exports from `cacheness.handlers`
  - `backends/` — blob and metadata backend implementations
- Depends on: Handlers, metadata, config, encryption
- Used by: `UnifiedCache._init_blob_store()`, direct `BlobStore` usage

**Storage Backends (`storage/backends/`):**
- Purpose: Concrete blob and metadata backend implementations
- Location: `src/cacheness/storage/backends/`
- Contains:
  - `blob_backends.py` (532 lines): `BlobBackend` ABC, `FilesystemBlobBackend`, `InMemoryBlobBackend`, blob backend registry
  - `s3_backend.py` (542 lines): `S3BlobBackend` for S3/MinIO
  - `postgresql_backend.py` (1410 lines): `PostgresBackend` for PostgreSQL metadata
  - `base.py` (25 lines): Re-exports `MetadataBackend` from `cacheness.metadata`
  - `__init__.py` (240 lines): Backend registries, re-exports
- Depends on: `cacheness.metadata`, boto3 (optional), psycopg2 (optional)
- Used by: `BlobStore`, `create_metadata_backend()`

**Security & Encryption (`security.py`, `encryption.py`):**
- Purpose: HMAC-SHA256 entry signing, HKDF key derivation, AES-256-GCM blob encryption
- Location: `src/cacheness/security.py` (470 lines), `src/cacheness/encryption.py` (81 lines)
- Contains: `CacheEntrySigner` (versioned HMAC signing, v1/v2/v3 field lists), `_hkdf_sha256()`, `encrypt_blob()`, `decrypt_blob()`, `derive_encryption_key()`
- Depends on: `interfaces.SignableFields`, `cryptography` package (optional for encryption)
- Used by: `UnifiedCache._init_entry_signer()`, `BlobStore._write_blob()`, `BlobStore._read_blob()`

**Configuration (`config.py`):**
- Purpose: Dataclass-based configuration with validation and file loading
- Location: `src/cacheness/config.py` (1253 lines)
- Contains: `CacheConfig` (top-level), `CacheStorageConfig`, `CacheMetadataConfig`, `CompressionConfig`, `SerializationConfig`, `HandlerConfig`, `HooksConfig`, `SecurityConfig`, `CacheBlobConfig`, `ConfigValidationError`, `validate_config()`, `validate_config_strict()`, JSON/YAML loaders
- Depends on: `metadata.validate_namespace_id`, `size_utils`
- Used by: Everything — config flows through all layers

**Utilities:**
- `serialization.py` (361 lines): Cache key generation (`create_unified_cache_key`)
- `compress_pickle.py` (760 lines): Blosc2/pickle serialization and compression
- `file_hashing.py` (192 lines): xxhash-based file/directory hashing
- `size_utils.py` (260 lines): Human-readable size/duration parsing
- `json_utils.py` (67 lines): orjson wrapper with stdlib fallback
- `error_handling.py` (302 lines): Exception hierarchy, `@cache_operation_context` decorator
- `entry_list.py` (148 lines): `EntryList` — rich list wrapper for query results
- `write_intent.py` (89 lines): `WriteIntentJournal` for crash-safe writes
- `custom_metadata.py` (547 lines): SQLAlchemy-based custom metadata models

**Interfaces (`interfaces.py`):**
- Purpose: Typed contracts for all cross-layer communication
- Location: `src/cacheness/interfaces.py` (505 lines)
- Contains: `CacheHandler` ABC, `HandlerResult` dataclass, `WriteBlobResult`, `IntegrityReport`, `RotationResult`, `BlobReadContext` TypedDict, `EntryData` TypedDict, `EntrySummary` TypedDict, `SignableFields` TypedDict
- Depends on: Nothing (leaf module)
- Used by: Every layer

## Data Flow

**put() — Cache Write Path:**

1. `UnifiedCache.put()` acquires `_lock` (RLock)
2. Resolves cache key from `cache_key` / `on` / `**kwargs` via `_create_cache_key()`
3. If `storage_mode=True`: delegates to `_storage_mode_put()` (skip TTL/eviction/stats)
4. Gets `_PutCleanup` rollback guard
5. Checks if overwriting — saves old blob path for stale cleanup
6. `HandlerRegistry.get_handler(data)` selects handler by type priority
7. Tries `_try_direct_inline()` — zero-disk inline for tiny objects (pickle+hash in memory)
8. If not inline: calls `BlobStore._write_blob(data, path, compute_hash)`
   - Handler's `put()` writes blob file (parquet/npz/pkl)
   - If encryption enabled: `encrypt_blob()` encrypts file content, rewrites file
   - If hash requested: `xxhash.xxh3_64` on (encrypted) bytes
   - Returns `WriteBlobResult(handler, HandlerResult, file_hash)`
9. Records write intent via `WriteIntentJournal`
10. Builds metadata dict from `HandlerResult` + file hash
11. If `store_full_metadata`: serializes kwargs as JSON metadata_dict
12. Tries disk-based inline: `_try_inline_blob()` reads back small blobs into metadata
13. Signs entry via `CacheEntrySigner.sign_entry()` if signing enabled
14. `metadata_backend.put_entry(cache_key, entry_data)` persists metadata
15. Clears write intent, cleans stale blob, stores custom metadata
16. `_enforce_size_limit()` — LRU eviction if over max_cache_size
17. `cleanup.commit()` — disarms rollback guard

**get() — Cache Read Path:**

1. `UnifiedCache.get()` acquires `_lock`
2. Resolves cache key, resolves TTL from `ttl`/`ttl_seconds`
3. If `storage_mode=True`: delegates to `_storage_mode_get()`
4. `metadata_backend.get_entry(cache_key)` loads metadata
5. Checks expiration via `_is_expired()` (compares `created_at + ttl` vs now)
6. Resolves `actual_path` from metadata to filesystem path
7. `_verify_entry()` — file hash check + HMAC signature verification
8. If inline: `_read_inline_blob()` deserializes from `blob_data` bytes in metadata
9. Else: `BlobStore._read_blob(file_path, data_type, metadata)`
   - If encryption enabled: `decrypt_blob()` decrypts file content
   - `HandlerRegistry.get_handler_by_type(data_type)` finds handler
   - Handler's `get()` deserializes blob file
10. Updates access time, records hit
11. On `FileNotFoundError`: removes metadata entry (destructive on missing blob)
12. On other deserialization errors: removes entry if `delete_on_error=True`

**State Management:**
- Thread safety: `threading.RLock` shared between `UnifiedCache` and its `BlobStore`
- Metadata: persisted via backend (JSON file, SQLite DB, or PostgreSQL)
- Blob files: stored on filesystem under `{cache_dir}/{namespace}/`
- Inline blobs: stored as `blob_data` bytes in metadata row (no filesystem file)
- Configuration: immutable `CacheConfig` dataclasses set at construction time

## Key Abstractions

**UnifiedCache (`core.py`):**
- Purpose: Primary user-facing cache with full lifecycle management
- Pattern: Mixin-based composition (11 mixins + `_PutCleanup`)
- Owns: config, metadata backend, handler registry, signer, blob store, write journal, lock

**BlobStore (`storage/blob_store.py`):**
- Purpose: Standalone blob storage without cache semantics (no TTL, no eviction)
- Pattern: Direct composition — owns its own config+backend+handlers or shares from UnifiedCache
- Use cases: ML model versioning, artifact storage, data pipeline checkpoints

**MetadataBackend (`metadata/base.py`):**
- Purpose: Abstract CRUD for cache entry metadata with namespace isolation
- Implementations: `JsonBackend`, `SqliteBackend`, `PostgresBackend`
- Pattern: ABC with optional `CachedMetadataBackend` wrapper (TTLCache)
- Schema versioning: migration chain v1→v2→v3→v4 (encryption columns added in v4)

**CacheHandler (`interfaces.py`):**
- Purpose: Type-aware serialization contract (can_handle, put, get, data_type)
- Pattern: ABC with focused sub-interfaces (CacheWriter, CacheReader, FormatProvider)
- Returns: `HandlerResult` dataclass with storage_format, file_size, actual_path, extras

**HandlerRegistry (`handlers/registry.py`):**
- Purpose: Priority-ordered handler selection by data type
- Pattern: Registry with configurable priority, custom handler registration/unregistration
- Selection: First handler where `can_handle(data)` returns True

**CacheEntrySigner (`security.py`):**
- Purpose: HMAC-SHA256 signing of metadata fields to detect tampering
- Pattern: Versioned field lists (v1, v2, v3) for safe schema evolution
- Features: HKDF-SHA256 per-namespace key derivation, key rotation support

## Entry Points

**Library Entry (`__init__.py`):**
- Location: `src/cacheness/__init__.py`
- Triggers: `import cacheness` or `from cacheness import cacheness`
- Responsibilities: Re-exports `UnifiedCache` as `cacheness`, decorators, config classes, interfaces

**Cache Construction (`core.py`):**
- Location: `src/cacheness/core.py` — `UnifiedCache.__init__()`
- Triggers: `cacheness(config)` or `get_cache(cache_dir)`
- Responsibilities: Creates config, metadata backend, handler registry, signer, blob store, write journal; runs cleanup

**Decorator Entry (`decorators.py`):**
- Location: `src/cacheness/decorators.py`
- Triggers: `@cached(cache_dir=...)` or `@cache_if(condition, ...)`
- Responsibilities: Creates per-decorator `UnifiedCache`, generates cache keys from function args

**Direct Storage (`storage/blob_store.py`):**
- Location: `src/cacheness/storage/blob_store.py` — `BlobStore.__init__()`
- Triggers: `from cacheness.storage import BlobStore; BlobStore(...)`
- Responsibilities: Standalone blob store without cache semantics

## Error Handling

**Strategy:** Exception hierarchy rooted in `CacheError`, intentionally-broad catches annotated, destructive error recovery configurable

**Exception Hierarchy (`error_handling.py`):**
- `CacheError` → base
  - `CacheConfigurationError` → invalid config
  - `CacheStorageError` → I/O failures
  - `CacheSerializationError` → key serialization failures
  - `CacheHandlerError` → handler put/get failures
  - `CacheIntegrityError` → hash/signature mismatch, decryption failure
  - `CacheSecurityError` → key file issues, signing failures
  - `CacheBackendError` → metadata backend failures

**Patterns:**
- All `except Exception` blocks annotated with `# intentionally broad` comment
- `get()` auto-deletes corrupt entries (configurable via `delete_on_error`)
- I/O errors in `get()` are non-destructive (entry retained for retry)
- `_PutCleanup` rollback guard deletes orphaned blobs on write failure
- `WriteIntentJournal` recovers from crashes between blob write and metadata commit
- Lifecycle hooks (`on_evict`) are swallowed-exception safe

## Cross-Cutting Concerns

**Logging:** Python `logging` module, module-level loggers, emoji-prefixed info messages (e.g. `"✅ Unified cache initialized"`)

**Validation:** `config.py` — `validate_config()` and `validate_config_strict()` check bad config combinations at construction time (encryption without signing, inline blobs without signing key, etc.)

**Authentication/Authorization:** Not applicable (local library). Security is about data integrity (HMAC signing) and confidentiality (AES-256-GCM encryption).

**Thread Safety:** `threading.RLock` shared between `UnifiedCache` and `BlobStore`; allows re-entrant calls (e.g. `delete_where` → `invalidate`). SQLite backend uses `check_same_thread=False` with internal locking.

**Namespace Isolation:** Multi-tenant namespace support — each namespace gets its own metadata table (SQLite/PostgreSQL) or JSON section, blob subdirectory, and HKDF-derived signing/encryption keys.

**Encryption Pipeline:** Handler output → compress → encrypt (AES-256-GCM, random 12-byte IV) → write to disk → xxhash on ciphertext → HMAC sign metadata. Decrypt path reverses: read → decrypt → decompress → handler deserialize. Per-namespace key derivation via HKDF-SHA256 with domain-separated info strings.

---

*Architecture analysis: 2026-04-07*
