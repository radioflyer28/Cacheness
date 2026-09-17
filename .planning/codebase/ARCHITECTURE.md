<!-- refreshed: 2026-09-17 -->
# Architecture

**Analysis Date:** 2026-09-17

**Independent Review:** 2026-09-17 — Phase 9 public paths and lifecycle composition were re-traced from current source

## System Overview

```text
Application -> UnifiedCache policy (optional) -> BlobStore lifecycle
                                              -> HandlerRegistry / FormatHandler
                                              -> one LifecycleAuthority
                                              -> one immutable payload participant

Application ---------------------------------> BlobStore lifecycle

SqlCache remains a separate pull-through subsystem pending its planned removal.
```

`BlobStore` is the single storage lifecycle engine. It owns handler staging,
immutable payload publication, authoritative catalog promotion, reads, exact
deletion, cleanup debt, and recovery. `UnifiedCache` adds key derivation, TTL,
outcomes, invalidation, and bounded maintenance policy over one caller-selected
or privately owned `BlobStore`. A store instance is configured for persistence
or used under cache policy; it does not need to serve both roles simultaneously.

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Public API and optional exports | Re-export cache classes, configuration, handlers, registries, and optional integrations | `src/cacheness/__init__.py` |
| Unified cache policy | Own cache keys, TTL, typed outcomes, invalidation, and bounded maintenance over one `BlobStore` | `src/cacheness/core.py` |
| Configuration model | Compose storage, metadata, blob, compression, serialization, handler, and security settings; validate combinations | `src/cacheness/config.py` |
| Function cache facade | Normalize function arguments, add function identity to keys, and call `UnifiedCache` through `@cached` | `src/cacheness/decorators.py` |
| Handler registry | Select a handler by `can_handle`, resolve persisted `data_type`, and manage priorities/registration | `src/cacheness/handlers.py` |
| Handler contracts | Define cacheability, write/read, format, and specialized handler interfaces | `src/cacheness/interfaces.py` |
| Format handlers | Persist/reconstruct pandas/polars objects, NumPy arrays, TensorFlow tensors, and arbitrary objects | `src/cacheness/handlers.py` |
| Cache-key serializer | Deterministically serialize parameters and hash them with XXH3_64 | `src/cacheness/serialization.py` |
| Metadata abstraction | Persist entry records and hit/miss statistics through interchangeable backends | `src/cacheness/metadata.py` |
| Role-aware composition | Validate and resolve one topology's authority, payload participant, and optional projections | `src/cacheness/storage/composition.py` |
| BlobStore | Own the backend-neutral object lifecycle and canonical catalog | `src/cacheness/storage/blob_store.py` |
| SQL pull-through cache | Query cached rows, detect missing ranges, fetch gaps, upsert, and return DataFrames | `src/cacheness/sql_cache.py` |
| Custom metadata | Register SQLAlchemy metadata models and link them to cache entries | `src/cacheness/custom_metadata.py` |
| Cross-cutting utilities | Compression, file hashing, HMAC signing, JSON compatibility, and error wrappers | `src/cacheness/compress_pickle.py`, `src/cacheness/file_hashing.py`, `src/cacheness/security.py`, `src/cacheness/json_utils.py`, `src/cacheness/error_handling.py` |

## Pattern Overview

**Overall:** Strategy/registry architecture with a coordinator facade and adapter-based SQL subsystem.

**Key Characteristics:**
- `BlobStore` delegates native-format staging and reading to ordered `FormatHandler` strategies while retaining publication and lifecycle authority (`src/cacheness/handlers.py`, `src/cacheness/storage/blob_store.py`).
- `StoreTopology` selects a declared authority/payload pair through the role registry; `UnifiedCache` either accepts that topology or a caller-owned `BlobStore` (`src/cacheness/storage/composition.py`, `src/cacheness/core.py`).
- Optional dependencies are imported conditionally; handlers are enabled only when their libraries are available (`src/cacheness/handlers.py`, `src/cacheness/__init__.py`).
- SQL caching uses an adapter contract for schema, query parsing, and external fetches, allowing builder methods to generate simple adapters (`src/cacheness/sql_cache.py`).
- Compatibility re-exports preserve older import paths through `src/cacheness/storage/handlers/__init__.py` and `src/cacheness/storage/backends/__init__.py`.

## Lifecycle Boundaries

Storage and cache policy now have one directional composition boundary:

```text
UnifiedCache policy ──uses──> BlobStore lifecycle
                                ├─ HandlerRegistry / guarded staging
                                ├─ LifecycleAuthority (memory/SQLite/PostgreSQL)
                                └─ immutable payload participant (memory/filesystem/S3)
```

**Transaction boundary:**
- `UnifiedCache.put()` delegates the storage mutation to `BlobStore.put_entry()` and applies cache maintenance only after a committed receipt exists.
- `BlobStore` uses one authority to select the visible immutable generation. Payload creation and cleanup remain outside the authority transaction and converge through attributable intent/cleanup debt; ADR 0001 does not claim cross-resource ACID.
- Reads consult authoritative descriptors, verify canonical SHA-256 plus size, and only then give a private snapshot path to the selected handler.

**Dependency-injection boundary:**
- `HandlerRegistry` is store-local and selected by persisted format identity.
- `StoreTopology` and the role registry are the only supported participant-selection surface.
- An injected `BlobStore` remains caller-owned; a topology passed to `UnifiedCache` creates one private owned store.

## Layers

**Public API and facades:**
- Purpose: Expose the stable user-facing constructors, decorators, factories, and registries.
- Location: `src/cacheness/__init__.py`, `src/cacheness/decorators.py`.
- Contains: `cacheness` alias, `cached`, `get_cache`, configuration helpers, and optional integrations.
- Depends on: `core.py`, configuration, handlers, metadata, and optional storage/SQL modules.
- Used by: Applications, examples, and tests.

**Unified cache policy:**
- Purpose: Implement key/value cache semantics above the storage lifecycle.
- Location: `src/cacheness/core.py`.
- Contains: `UnifiedCache.put`, `UnifiedCache.get`, invalidation, cleanup, stats, custom metadata, and global-cache factories.
- Depends on: `CacheConfig`, key serialization, catalog queries, and one `BlobStore`.
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

**Blob storage lifecycle:**
- Purpose: Provide reusable object storage independent of cache TTL and eviction semantics.
- Location: `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/composition.py`, and lifecycle/payload participant modules under `src/cacheness/storage/`.
- Contains: `BlobStore`, lifecycle authorities, obstore-backed immutable payload I/O, catalog/projection contracts, and recovery.
- Depends on: Store-local handlers, one declared lifecycle authority, and one payload participant.
- Used by: Direct storage consumers and every `UnifiedCache` instance.

**SQL pull-through caching:**
- Purpose: Cache tabular query results in a SQL table and fetch only missing data.
- Location: `src/cacheness/sql_cache.py`.
- Contains: `SqlCache`, `SqlCacheAdapter`, backend factories, query condition builders, TTL columns, gap detection, and upserts.
- Depends on: SQLAlchemy, pandas, and a caller-provided fetch adapter/function.
- Used by: Applications needing range-aware or analytical tabular caching.

## Data Flow

### Primary Request Path

1. A caller constructs `UnifiedCache(CacheConfig(...), store=...)` with either a qualified `StoreTopology` or a caller-owned `BlobStore`.
2. `initialize()` initializes only a cache-owned store; an injected store retains its caller-owned lifecycle boundary.
3. `put(...)` derives the policy key and catalog fields, then calls `BlobStore.put_entry()`.
4. `BlobStore` chooses a `FormatHandler`, validates its private staged artifact, publishes an immutable payload generation, and promotes the authenticated descriptor through the sole lifecycle authority.
5. The committed `BlobReceipt` is returned inside `CachePutResult`; size maintenance is a later bounded policy step and cannot invalidate that storage commit.
6. `lookup(...)` reads through `BlobStore`, classifies storage failures into typed cache outcomes, applies TTL policy, and returns `CacheLookupResult` without making paths or payload presence authoritative.

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
- `UnifiedCache` holds one store reference, policy configuration, typed outcome counters, and bounded signed maintenance continuations.
- The selected lifecycle authority is canonical for committed membership and descriptors; payload listings, filesystem paths, and projections are not authority.
- `BlobStore` owns exact mutation/recovery state, while derived JSON projections consume bounded authenticated catalog pages.
- SQL cache state lives in caller-defined tables with `cached_at` and optional `expires_at` columns (`src/cacheness/sql_cache.py:301-383`).

## Key Abstractions

**`FormatHandler`:**
- Purpose: Contract for data detection, write/read operations, file format, and persisted type identifier.
- Examples: `ArrayHandler`, `PandasDataFrameHandler`, `PolarsDataFrameHandler`, `ObjectHandler` in `src/cacheness/handlers.py`.
- Pattern: Strategy objects selected in store-local registry order; register through `store.handlers.register_handler(...)`.

**`LifecycleAuthority` and payload participant:**
- Purpose: Separate canonical visibility/descriptor transitions from immutable payload byte I/O without inventing cross-resource ACID.
- Examples: memory/SQLite/PostgreSQL authorities and obstore memory/filesystem/S3 payload participants.
- Pattern: `StoreTopology` resolves one qualified pair; `BlobStore` is the only coordinator.

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
- Triggers: Direct construction or `cached(cache=...)`.
- Responsibilities: Key/value cache policy over one explicit store.

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

- **Concurrency:** The selected authority serializes canonical lifecycle transitions for its declared topology. Safe typed conflicts/timeouts are valid progress outcomes; process-local locks do not extend authority across processes or resources.
- **Global state:** Decorator-created cache instances are weakly tracked for close-at-exit; store handlers and participant composition remain instance-local.
- **Circular imports:** Compatibility modules intentionally re-export parent implementations: `src/cacheness/storage/handlers/__init__.py` imports `cacheness.handlers`, while `src/cacheness/storage/backends/__init__.py` imports implementations from `cacheness.metadata`.
- **Optional dependencies:** pandas, polars, SQLAlchemy, Blosc2, dill, TensorFlow, boto3, and database drivers are declared as optional and mostly guarded at runtime. NumPy is the exception: it is declared only in optional groups but imported eagerly by package-import paths, making it an effective undeclared base requirement (`pyproject.toml`, `src/cacheness/handlers.py`, `src/cacheness/compress_pickle.py`, `src/cacheness/__init__.py`).
- **Payload staging:** A format handler sees only a private contained staging/snapshot path. Managed locators and obstore participants remain behind `BlobStore`.
- **SQL schema ownership:** `SqlCache` mutates the supplied SQLAlchemy `Table` by appending cache columns before creating it (`src/cacheness/sql_cache.py:459-483`); callers must provide compatible primary keys and column definitions.

## Anti-Patterns

### Bypassing BlobStore lifecycle authority

**What happens:** Policy or format code treats a path, object listing, or projection as canonical lifecycle state.
**Why it's wrong:** Only the selected authority can establish committed membership and authorize exact mutation/recovery.
**Do this instead:** Implement/register a `FormatHandler` for serialization and route every storage mutation through `BlobStore`.

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
