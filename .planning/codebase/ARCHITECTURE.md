<!-- refreshed: 2026-08-29 -->
# Architecture

**Analysis Date:** 2026-08-29

**Independent Review:** 2026-08-29 — call paths and lifecycle boundaries were re-traced from source and runtime probes

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    Public package API                       │
│ `src/cacheness/__init__.py`, `src/cacheness/decorators.py`,  │
│ `src/cacheness/sql_cache.py`                                │
└───────────────┬──────────────────────┬──────────────────────┘
                │                      │
                ▼                      ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│ UnifiedCache coordinator  │  │ SqlCache pull-through layer   │
│ `src/cacheness/core.py`   │  │ `src/cacheness/sql_cache.py`  │
└───────────┬──────────────┘  └───────────────┬──────────────┘
            │                                  │
            ▼                                  ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│ HandlerRegistry           │  │ SQLAlchemy engine/session     │
│ `src/cacheness/handlers.py`│ │ + adapter + gap detection    │
└───────────┬──────────────┘  └───────────────┬──────────────┘
            │                                  │
            ▼                                  ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│ Format handlers           │  │ User data fetcher             │
│ parquet / b2nd / npz /    │  │ returns pandas DataFrame      │
│ compressed pickle files   │  │                               │
└───────────┬──────────────┘  └──────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│ Metadata backends: SQLite / JSON / memory / PostgreSQL       │
│ `src/cacheness/metadata.py`, `storage/backends/`              │
└─────────────────────────────────────────────────────────────┘
```

`UnifiedCache` is the primary key/value cache. It coordinates key generation, type-specific serialization, metadata, TTL, integrity checks, signing, and eviction. `BlobStore` in `src/cacheness/storage/blob_store.py` is a separate lower-level object store that reuses handlers and metadata but deliberately omits TTL and eviction. `SqlCache` in `src/cacheness/sql_cache.py` is a separate row/table cache for DataFrame results and does not use `UnifiedCache`.

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Public API and optional exports | Re-export cache classes, configuration, handlers, registries, and optional integrations | `src/cacheness/__init__.py` |
| Unified cache coordinator | Own cache lifecycle, key/value operations, TTL, metadata, integrity, signing, custom metadata, and size enforcement | `src/cacheness/core.py` |
| Configuration model | Compose storage, metadata, blob, compression, serialization, handler, and security settings; validate combinations | `src/cacheness/config.py` |
| Function cache facade | Normalize function arguments, add function identity to keys, and call `UnifiedCache` through `@cached` | `src/cacheness/decorators.py` |
| Handler registry | Select a handler by `can_handle`, resolve persisted `data_type`, and manage priorities/registration | `src/cacheness/handlers.py` |
| Handler contracts | Define cacheability, write/read, format, and specialized handler interfaces | `src/cacheness/interfaces.py` |
| Format handlers | Persist/reconstruct pandas/polars objects, NumPy arrays, TensorFlow tensors, and arbitrary objects | `src/cacheness/handlers.py` |
| Cache-key serializer | Deterministically serialize parameters and hash them with XXH3_64 | `src/cacheness/serialization.py` |
| Metadata abstraction | Persist entry records and hit/miss statistics through interchangeable backends | `src/cacheness/metadata.py` |
| Backend registries | Register and construct metadata and blob backend classes | `src/cacheness/storage/backends/__init__.py`, `src/cacheness/storage/backends/blob_backends.py` |
| BlobStore | Store arbitrary handler-backed objects with explicit keys/content hashes and metadata | `src/cacheness/storage/blob_store.py` |
| SQL pull-through cache | Query cached rows, detect missing ranges, fetch gaps, upsert, and return DataFrames | `src/cacheness/sql_cache.py` |
| Custom metadata | Register SQLAlchemy metadata models and link them to cache entries | `src/cacheness/custom_metadata.py` |
| Cross-cutting utilities | Compression, file hashing, HMAC signing, JSON compatibility, and error wrappers | `src/cacheness/compress_pickle.py`, `src/cacheness/file_hashing.py`, `src/cacheness/security.py`, `src/cacheness/json_utils.py`, `src/cacheness/error_handling.py` |

## Pattern Overview

**Overall:** Strategy/registry architecture with a coordinator facade and adapter-based SQL subsystem.

**Key Characteristics:**
- `UnifiedCache` delegates data-format decisions to ordered `CacheHandler` strategies rather than branching on every type in the coordinator (`src/cacheness/handlers.py`).
- Metadata and blob backends are selected through factories/registries, while the primary `UnifiedCache` path writes handler-produced files and records their paths in metadata (`src/cacheness/core.py`).
- Optional dependencies are imported conditionally; handlers are enabled only when their libraries are available (`src/cacheness/handlers.py`, `src/cacheness/__init__.py`).
- SQL caching uses an adapter contract for schema, query parsing, and external fetches, allowing builder methods to generate simple adapters (`src/cacheness/sql_cache.py`).
- Compatibility re-exports preserve older import paths through `src/cacheness/storage/handlers/__init__.py` and `src/cacheness/storage/backends/__init__.py`.

## Lifecycle Boundaries

The repository exposes three adjacent products rather than one fully unified storage stack:

```text
UnifiedCache                 BlobStore                    SqlCache
`core.py`                    `storage/blob_store.py`      `sql_cache.py`
    │                              │                           │
    ├─ HandlerRegistry            ├─ HandlerRegistry          ├─ SQLAlchemy Table
    ├─ direct payload files       ├─ direct payload files     ├─ caller adapter
    └─ metadata backend           └─ metadata backend          └─ database rows

BlobBackend registry (`filesystem`, `memory`, optional `s3`)
    └─ currently independent of both handler-backed payload paths
```

**Transaction boundary:**
- `UnifiedCache.put()` writes a handler payload first, computes metadata/signature information, and then writes the metadata entry (`src/cacheness/core.py:811-922`). There is no rollback if metadata persistence fails, so payload and metadata are not one atomic transaction.
- Reads perform the inverse lookup through metadata and then the recorded `actual_path`; integrity/signature failures can remove metadata without consistently removing the payload (`src/cacheness/core.py:924-1053`).
- `BlobStore` repeats a similar two-step payload/metadata lifecycle independently (`src/cacheness/storage/blob_store.py:128-241`). It merges nested metadata during reads, but deletion/existence do not use the same normalization.

**Dependency-injection boundary:**
- Handler instances are genuinely selected through `HandlerRegistry`.
- Metadata backend classes can be constructed through a registry, but `UnifiedCache` does not consult that registry. Even an explicitly injected backend instance is overwritten by the subsequent config-selection branch (`src/cacheness/core.py:109-212`).
- Blob backends have a registry and implementations, but no high-level coordinator injects them into handler persistence. Treat these as incomplete composition seams, not interchangeable production strategies.

## Layers

**Public API and facades:**
- Purpose: Expose the stable user-facing constructors, decorators, factories, and registries.
- Location: `src/cacheness/__init__.py`, `src/cacheness/decorators.py`.
- Contains: `cacheness` alias, `cached`, `get_cache`, configuration helpers, and optional integrations.
- Depends on: `core.py`, configuration, handlers, metadata, and optional storage/SQL modules.
- Used by: Applications, examples, and tests.

**Unified cache coordination:**
- Purpose: Implement key/value caching semantics and orchestrate handlers plus metadata.
- Location: `src/cacheness/core.py`.
- Contains: `UnifiedCache.put`, `UnifiedCache.get`, invalidation, cleanup, stats, custom metadata, and global-cache factories.
- Depends on: `CacheConfig`, `HandlerRegistry`, key serialization, metadata factory, file hashing, and signing.
- Used by: Public API and decorators.

**Type-aware serialization:**
- Purpose: Map Python values to storage formats and reconstruct them.
- Location: `src/cacheness/handlers.py`, `src/cacheness/interfaces.py`, `src/cacheness/compress_pickle.py`.
- Contains: Parquet handlers, NumPy Blosc2/NPZ handling, pickle/dill object handling, and format metadata.
- Depends on: Optional NumPy, pandas, polars, TensorFlow, Blosc2, dill, and compression helpers.
- Used by: `UnifiedCache` and `BlobStore`.

**Metadata persistence:**
- Purpose: Store cache-entry descriptors, timestamps, statistics, and backend-specific fields.
- Location: `src/cacheness/metadata.py`, `src/cacheness/storage/backends/`.
- Contains: JSON, SQLite, in-memory, PostgreSQL, and optional memory-cache wrapper implementations.
- Depends on: JSON utilities; SQLAlchemy for relational backends; psycopg for PostgreSQL.
- Used by: `UnifiedCache`, `BlobStore`, and custom metadata support.

**Low-level blob storage:**
- Purpose: Provide reusable object/blob storage independent of cache TTL and eviction semantics.
- Location: `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/backends/blob_backends.py`, `src/cacheness/storage/backends/s3_backend.py`.
- Contains: `BlobStore`, filesystem/in-memory blob contracts, and optional S3 implementation.
- Depends on: Handler registry, metadata backend, compression, and boto3 for S3.
- Used by: Consumers importing the storage API directly. `UnifiedCache` does not instantiate `BlobStore`.

**SQL pull-through caching:**
- Purpose: Cache tabular query results in a SQL table and fetch only missing data.
- Location: `src/cacheness/sql_cache.py`.
- Contains: `SqlCache`, `SqlCacheAdapter`, backend factories, query condition builders, TTL columns, gap detection, and upserts.
- Depends on: SQLAlchemy, pandas, and a caller-provided fetch adapter/function.
- Used by: Applications needing range-aware or analytical tabular caching.

## Data Flow

### Primary Request Path

1. A caller creates `UnifiedCache`/`cacheness`, optionally passing `CacheConfig` or a metadata backend (`src/cacheness/core.py:68`).
2. Initialization creates the cache directory, ordered handler registry, metadata backend, optional custom metadata support, and optional HMAC signer (`src/cacheness/core.py:68-285`).
3. `put(data, **kwargs)` selects the first handler whose `can_handle` succeeds and creates a 16-character XXH3_64 key from sorted serialized parameters (`src/cacheness/core.py:811`, `src/cacheness/handlers.py:1192`, `src/cacheness/serialization.py:381`).
4. The handler writes a format-specific file and returns `storage_format`, `file_size`, `actual_path`, and handler metadata (`src/cacheness/handlers.py:426-1055`).
5. The coordinator optionally hashes the file, builds the metadata entry, signs selected fields, persists it through `metadata_backend.put_entry`, stores custom metadata, and enforces the size limit (`src/cacheness/core.py:811-923`).
6. `get(...)` resolves the key, reads metadata, checks TTL, verifies file hash/signature when enabled, resolves the handler by persisted `data_type`, and calls `handler.get` on `actual_path` (`src/cacheness/core.py:924-1054`).
7. Successful reads update access time and hit statistics; missing, expired, corrupt, or unreadable entries increment misses and are removed from metadata (`src/cacheness/core.py:924-1054`).

### Decorated Function Flow

1. `@cached` normalizes positional/keyword arguments using `inspect.signature` and adds module-qualified function identity (`src/cacheness/decorators.py:39-78`).
2. The decorator asks the same unified serializer for a cache key, then calls `UnifiedCache.get`/`put` using `__decorator_cache_key` as a synthetic key parameter (`src/cacheness/decorators.py:145-220`).
3. Cache errors can be suppressed according to `ignore_errors`; decorator-created cache instances are weakly tracked and closed with an `atexit` hook (`src/cacheness/decorators.py:22-36`, `src/cacheness/decorators.py:145-220`).

### SQL Pull-Through Flow

1. A builder such as `SqlCache.for_timeseries`, `for_lookup_table`, or `for_analytics_table` creates a SQLAlchemy table and a simple `SqlCacheAdapter` around the caller’s fetcher (`src/cacheness/sql_cache.py:1450-1775`).
2. `get_data(**query_params)` parses parameters, opens a session, queries cached rows including TTL conditions, and analyzes missing ranges (`src/cacheness/sql_cache.py:486-554`, `src/cacheness/sql_cache.py:710-1323`).
3. Each missing range is passed to `data_adapter.fetch_data`; non-empty DataFrames are upserted and the session commits (`src/cacheness/sql_cache.py:486-530`, `src/cacheness/sql_cache.py:600-709`).
4. The complete matching dataset is queried back as a pandas DataFrame. Invalidation, expiration cleanup, statistics, and engine disposal remain within `SqlCache` (`src/cacheness/sql_cache.py:1339-1455`).

**State Management:**
- `UnifiedCache` holds a per-instance `HandlerRegistry`, metadata backend, optional signer, and a `threading.Lock`; the module also exposes a lazily initialized global cache (`src/cacheness/core.py:68-108`, `src/cacheness/core.py:1204-1238`).
- Persistent metadata is authoritative for cache entries; cache files contain payloads and metadata stores `actual_path`, type, format, timestamps, and size.
- `CachedMetadataBackend` can add a `cachetools` in-memory entry layer over JSON/SQLite metadata (`src/cacheness/metadata.py:253-457`).
- SQL cache state lives in caller-defined tables with `cached_at` and optional `expires_at` columns (`src/cacheness/sql_cache.py:301-383`).

## Key Abstractions

**`CacheHandler`:**
- Purpose: Contract for data detection, write/read operations, file format, and persisted type identifier.
- Examples: `ArrayHandler`, `PandasDataFrameHandler`, `PolarsDataFrameHandler`, `ObjectHandler` in `src/cacheness/handlers.py`.
- Pattern: Strategy objects selected in registry order; register new handlers through `HandlerRegistry.register_handler` or the module-level API.

**`MetadataBackend`:**
- Purpose: Abstract entry CRUD, statistics, TTL cleanup, and lifecycle operations.
- Examples: `JsonBackend`, `SqliteBackend`, `InMemoryBackend` in `src/cacheness/metadata.py`; `PostgresBackend` in `src/cacheness/storage/backends/postgresql_backend.py`.
- Pattern: Factory/registry selection through `create_metadata_backend` and `get_metadata_backend`; custom backends implement the backend contract.

**`BlobBackend`:**
- Purpose: Abstract binary blob read/write/delete/stream operations independently of metadata semantics.
- Examples: `FilesystemBlobBackend`, `InMemoryBlobBackend`, `S3BlobBackend` in `src/cacheness/storage/backends/`.
- Pattern: Registry lookup via `get_blob_backend`; S3 is optional and must be available/registered before use.

**`SqlCacheAdapter`:**
- Purpose: Bind a SQL table schema and external data source to pull-through caching.
- Examples: User-defined adapters or generated `SimpleTimeseriesAdapter`, `SimpleLookupAdapter`, and `SimpleAnalyticsAdapter` in `src/cacheness/sql_cache.py`.
- Pattern: Adapter plus builder/factory methods; fetchers return pandas DataFrames matching the table columns.

**`CacheConfig`:**
- Purpose: Central configuration object with focused sub-configurations and backward-compatible flat arguments.
- Examples: `CacheStorageConfig`, `CacheMetadataConfig`, `CacheBlobConfig`, `CompressionConfig`, `SerializationConfig`, `HandlerConfig`, and `SecurityConfig` in `src/cacheness/config.py`.
- Pattern: Dataclass sub-configurations composed by an imperative compatibility-aware constructor and factory helpers.

## Entry Points

**Package import:**
- Location: `src/cacheness/__init__.py`.
- Triggers: `import cacheness`.
- Responsibilities: Expose public aliases, optional dependency guards, registration APIs, and version metadata.

**Unified cache:**
- Location: `src/cacheness/core.py:60` (`UnifiedCache`), `src/cacheness/core.py:1219` (`get_cache`).
- Triggers: Direct construction, `cacheness(...)`, `get_cache()`, or `cached()`.
- Responsibilities: Key/value cache lifecycle and persistence coordination.

**Decorator API:**
- Location: `src/cacheness/decorators.py:80` (`cached`).
- Triggers: `@cached`, `@memoize`, `cache_function`, or `CacheContext`.
- Responsibilities: Function-aware keying and transparent fallback to original function execution.

**BlobStore:**
- Location: `src/cacheness/storage/blob_store.py:56`.
- Triggers: `from cacheness.storage import BlobStore`.
- Responsibilities: Explicit-key/content-addressable object storage with metadata and handler-backed formats.

**SQL cache:**
- Location: `src/cacheness/sql_cache.py:142` (`SqlCache`).
- Triggers: `SqlCache.with_*` or `SqlCache.for_*` builders.
- Responsibilities: Query-aware tabular cache and external fetch orchestration.

## Architectural Constraints

- **Threading:** `UnifiedCache` creates `self._lock` but does not acquire it in `put()`, `get()`, invalidation, or cleanup. JSON/in-memory metadata backends use their own locks, and SQLite/PostgreSQL rely on SQLAlchemy sessions/pools, so metadata operations have some local coordination while the payload-plus-metadata lifecycle is not serialized (`src/cacheness/core.py:82-103`, `src/cacheness/metadata.py`, `src/cacheness/storage/backends/postgresql_backend.py`).
- **Global state:** `_global_cache` in `src/cacheness/core.py`, the decorator weak-reference list in `src/cacheness/decorators.py`, and handler/backend registries in `src/cacheness/handlers.py` and `src/cacheness/storage/backends/` are module-level mutable state.
- **Circular imports:** Compatibility modules intentionally re-export parent implementations: `src/cacheness/storage/handlers/__init__.py` imports `cacheness.handlers`, while `src/cacheness/storage/backends/__init__.py` imports implementations from `cacheness.metadata`.
- **Optional dependencies:** pandas, polars, SQLAlchemy, Blosc2, dill, TensorFlow, boto3, and database drivers are declared as optional and mostly guarded at runtime. NumPy is the exception: it is declared only in optional groups but imported eagerly by package-import paths, making it an effective undeclared base requirement (`pyproject.toml`, `src/cacheness/handlers.py`, `src/cacheness/compress_pickle.py`, `src/cacheness/__init__.py`).
- **Filesystem payloads:** The primary `UnifiedCache` handler path writes directly under `CacheStorageConfig.cache_dir`; persisted metadata must retain `actual_path` because handler extensions can be dynamic (`src/cacheness/core.py`, `src/cacheness/handlers.py`).
- **SQL schema ownership:** `SqlCache` mutates the supplied SQLAlchemy `Table` by appending cache columns before creating it (`src/cacheness/sql_cache.py:459-483`); callers must provide compatible primary keys and column definitions.

## Anti-Patterns

### Bypassing the handler and metadata contracts

**What happens:** Cache payloads are written or read directly without `HandlerRegistry` and without persisting matching metadata.
**Why it's wrong:** `UnifiedCache.get` resolves the handler and payload path from metadata, so direct files become undiscoverable or unreadable (`src/cacheness/core.py:924-1054`).
**Do this instead:** Implement/register a `CacheHandler` and let `UnifiedCache.put/get` own the file and metadata transaction (`src/cacheness/interfaces.py`, `src/cacheness/handlers.py`).

### Treating `BlobStore` as a TTL cache

**What happens:** `BlobStore` is used with expectations of automatic expiration, hit/miss accounting, or size-based eviction.
**Why it's wrong:** `BlobStore` explicitly provides low-level storage and its `put/get` path does not implement `UnifiedCache` TTL or eviction semantics (`src/cacheness/storage/blob_store.py:56-226`).
**Do this instead:** Use `UnifiedCache` for cache semantics, or implement lifecycle policy around `BlobStore` when artifact persistence is intended (`src/cacheness/core.py`, `src/cacheness/storage/blob_store.py`).

### Mixing SQL pull-through and key/value APIs

**What happens:** A `SqlCache` table is treated as a `UnifiedCache` metadata store, or a `UnifiedCache` file entry is expected to support range-gap detection.
**Why it's wrong:** They have separate storage models, query paths, and adapter contracts (`src/cacheness/sql_cache.py`, `src/cacheness/metadata.py`).
**Do this instead:** Choose `SqlCache` for DataFrame/table range queries and `UnifiedCache` for object/array/function key/value entries.

### Importing implementation modules through unstable compatibility paths

**What happens:** New code reaches through compatibility re-export modules and duplicates registry or handler implementations.
**Why it's wrong:** Canonical implementations live in top-level modules while `src/cacheness/storage/handlers/__init__.py` and `src/cacheness/storage/backends/__init__.py` primarily re-export them, increasing ambiguity.
**Do this instead:** Use the public package API or canonical implementation modules (`src/cacheness/__init__.py`, `src/cacheness/handlers.py`, `src/cacheness/metadata.py`) and add compatibility exports only when required.

## Error Handling

**Strategy:** Validate configuration and dependencies at construction, log operational failures, and either propagate or convert errors depending on the API boundary.

**Patterns:**
- `UnifiedCache.put` propagates I/O and handler failures after logging; `UnifiedCache.get` treats missing/corrupt/unreadable entries as misses, removes metadata, and returns `None` (`src/cacheness/core.py:811-1054`).
- Handler-level domain exceptions are defined in `src/cacheness/interfaces.py`; broader cache error classes and decorators live in `src/cacheness/error_handling.py`.
- `@cached` optionally suppresses key/retrieval/storage errors and executes the wrapped function according to `ignore_errors` (`src/cacheness/decorators.py:145-220`).
- `SqlCache.get_data` wraps failures in `SQLCacheError`; failures fetching one missing range are printed as warnings while other ranges continue (`src/cacheness/sql_cache.py:486-530`).
- Optional feature imports raise focused `ImportError`/`MissingDependencyError` messages when a requested backend or format is unavailable (`src/cacheness/core.py`, `src/cacheness/sql_cache.py`).

## Cross-Cutting Concerns

**Logging:** Module-level `logging.getLogger(__name__)` is used across coordinators, handlers, metadata backends, and utilities. Operational events use info/warning/error; detailed cache paths and performance details use debug (`src/cacheness/core.py`, `src/cacheness/error_handling.py`).

**Validation:** Sub-configurations validate local values in `__post_init__`; `CacheConfig` validates metadata/blob backend compatibility; handler registries validate required methods/properties; SQL adapters validate through their abstract contract (`src/cacheness/config.py`, `src/cacheness/handlers.py`, `src/cacheness/sql_cache.py`).

**Authentication:** `CacheEntrySigner` provides optional HMAC-SHA256 signatures for metadata fields; `UnifiedCache` verifies signatures and may remove invalid/unsigned entries according to `SecurityConfig` (`src/cacheness/security.py`, `src/cacheness/core.py:734-778`, `src/cacheness/config.py:287-336`).

**Integrity:** Optional XXH3_64 file hashes are recorded in metadata and checked on reads; atomic temporary-file writes are used by JSON metadata and filesystem blob writes (`src/cacheness/core.py:779-801`, `src/cacheness/metadata.py:697-735`, `src/cacheness/storage/backends/blob_backends.py:232-249`).

**Serialization and compression:** Cache keys use sorted parameter names plus configurable recursive serialization; payload formats are selected by handler and compression settings (`src/cacheness/serialization.py`, `src/cacheness/config.py`, `src/cacheness/handlers.py`, `src/cacheness/compress_pickle.py`).

---

*Architecture analysis: 2026-08-29*
