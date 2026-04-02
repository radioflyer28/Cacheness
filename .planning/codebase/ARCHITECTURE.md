# Architecture

**Analysis Date:** 2026-04-02

## Pattern Overview

**Overall:** Layered architecture with Strategy pattern (pluggable handlers) and Abstract Factory pattern (pluggable backends)

**Key Characteristics:**
- Two-tier design: high-level `UnifiedCache` (caching semantics) delegates storage I/O to low-level `BlobStore`
- Handler-based type dispatch using the Strategy pattern — each data type has a specialized handler implementing `CacheHandler`
- Pluggable metadata backends (JSON, SQLite, PostgreSQL) behind an abstract `MetadataBackend` interface
- Pluggable blob storage backends (Filesystem, S3, In-memory) behind an abstract `BlobBackend` interface
- Dataclass-based hierarchical configuration (`CacheConfig` → sub-configs for storage, metadata, blob, compression, serialization, security, hooks, handlers)
- Thread-safe operations via `threading.RLock` shared between `UnifiedCache` and its internal `BlobStore`
- Optional cryptographic entry signing via HMAC-SHA256 (`CacheEntrySigner`)

## Layers

**Public API Layer:**
- Purpose: User-facing interfaces — decorator API, direct put/get API, factory functions
- Location: `src/cacheness/__init__.py`, `src/cacheness/core.py`, `src/cacheness/decorators.py`
- Contains: `UnifiedCache` class (exported as `cacheness`), `cached` / `cache_if` decorators, `get_cache()` factory
- Depends on: Core Layer, Configuration Layer
- Used by: Application code

**Core Layer (Cache Orchestration):**
- Purpose: Coordinates caching semantics — TTL enforcement, eviction, stats tracking, key generation, metadata integrity, namespace isolation
- Location: `src/cacheness/core.py`
- Contains: `UnifiedCache` class (~3900 lines) — the central coordinator
- Depends on: Storage Layer, Handler Layer, Metadata Layer, Security Layer, Configuration Layer
- Used by: Public API Layer, Decorator Layer

**Storage Layer (Blob I/O):**
- Purpose: Low-level blob read/write, handler dispatch, integrity verification, path management
- Location: `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/paths.py`
- Contains: `BlobStore` class — pure storage without caching semantics (no TTL, no eviction)
- Depends on: Handler Layer, Metadata Layer, Blob Backend Layer, Compression Layer
- Used by: Core Layer. Also usable standalone for non-caching storage (ML model versioning, artifact storage)

**Handler Layer (Type-Aware Serialization):**
- Purpose: Detect data types and serialize/deserialize using optimal formats
- Location: `src/cacheness/handlers.py`, `src/cacheness/storage/handlers/__init__.py`
- Contains: `HandlerRegistry`, `PolarsDataFrameHandler`, `PandasDataFrameHandler`, `PandasSeriesHandler`, `ArrayHandler`, `BytesHandler`, `ObjectHandler`, `TensorFlowTensorHandler`
- Depends on: Interfaces Layer, Compression Layer
- Used by: Storage Layer (via `HandlerRegistry.get_handler()`)

**Metadata Backend Layer:**
- Purpose: Store and query cache entry metadata (keys, timestamps, sizes, custom fields)
- Location: `src/cacheness/metadata.py`, `src/cacheness/storage/backends/`
- Contains: `MetadataBackend` ABC, `JsonBackend`, `SqliteBackend`, `PostgresBackend`, `create_metadata_backend()` factory
- Depends on: Interfaces Layer (`EntrySummary`, `EntryData`)
- Used by: Core Layer, Storage Layer

**Blob Backend Layer:**
- Purpose: Store and retrieve raw binary blob data independently of metadata
- Location: `src/cacheness/storage/backends/blob_backends.py`, `src/cacheness/storage/backends/s3_backend.py`
- Contains: `BlobBackend` ABC, `FilesystemBlobBackend`, `InMemoryBlobBackend`, `S3BlobBackend`, blob backend registry
- Depends on: None (pure I/O abstraction)
- Used by: Storage Layer (`BlobStore`)

**Configuration Layer:**
- Purpose: Hierarchical, validated configuration with human-readable size/duration parsing
- Location: `src/cacheness/config.py`
- Contains: `CacheConfig`, `CacheStorageConfig`, `CacheMetadataConfig`, `CacheBlobConfig`, `CompressionConfig`, `SerializationConfig`, `HandlerConfig`, `SecurityConfig`, `HooksConfig`
- Depends on: `size_utils.py` (parsing), `metadata.py` (namespace validation)
- Used by: All layers

**Security Layer:**
- Purpose: HMAC-SHA256 signing and verification of cache entry metadata, namespace signing
- Location: `src/cacheness/security.py`, `src/cacheness/storage/security.py`
- Contains: `CacheEntrySigner` with versioned signed-field lists, key generation/rotation
- Depends on: Interfaces Layer (`SignableFields`)
- Used by: Core Layer, Storage Layer

**Interfaces Layer:**
- Purpose: Typed contracts and ABCs shared across layers — prevents circular imports
- Location: `src/cacheness/interfaces.py`
- Contains: `CacheHandler` ABC, `HandlerResult`, `WriteBlobResult`, `IntegrityReport`, `EntryData`, `EntrySummary`, `BlobReadContext`, `SignableFields` (TypedDicts and dataclasses)
- Depends on: Nothing (leaf module)
- Used by: All layers

**Utilities Layer:**
- Purpose: Cross-cutting utilities — serialization, hashing, size/duration parsing, JSON, error handling
- Location: `src/cacheness/serialization.py`, `src/cacheness/file_hashing.py`, `src/cacheness/size_utils.py`, `src/cacheness/json_utils.py`, `src/cacheness/compress_pickle.py`, `src/cacheness/error_handling.py`
- Contains: `create_unified_cache_key()`, `hash_file_content()`, `hash_directory_parallel()`, `parse_size()`, `parse_duration()`, orjson-backed JSON helpers, blosc2/pickle compression, error hierarchy
- Depends on: xxhash, numpy, orjson (optional), blosc2 (optional), dill (optional)
- Used by: All layers

## Data Flow

**Cache Write (`put()`):**

1. User calls `cache.put(data, ...)` on `UnifiedCache` (`src/cacheness/core.py`)
2. `_resolve_cache_key()` generates a deterministic hash key via `create_unified_cache_key()` (`src/cacheness/serialization.py`) using xxhash
3. `_get_cache_file_path()` computes the base file path within the namespace subdirectory
4. If `storage_mode` is True, bypasses TTL/eviction and calls `_storage_mode_put()`
5. `HandlerRegistry.get_handler(data)` iterates registered handlers; first `can_handle(data) == True` wins
6. The selected handler's `put(data, file_path, config)` serializes data to disk and returns a `HandlerResult` (format, size, path, codec, extras)
7. If blob backend is non-filesystem (e.g. S3), blob is uploaded via `BlobBackend.write_blob()`
8. File hash is computed via xxhash for integrity verification
9. Entry signer (if enabled) computes HMAC-SHA256 signature over metadata fields
10. Metadata is written to the backend via `metadata_backend.save_entry(cache_key, entry_data)`
11. Custom metadata (if provided) is stored in linked SQLAlchemy tables
12. `_enforce_cache_size_limit()` evicts oldest entries if cache exceeds configured max size
13. `_PutCleanup` rollback guard ensures orphaned blobs are cleaned up on any failure

**Cache Read (`get()`):**

1. User calls `cache.get(...)` on `UnifiedCache`
2. Cache key is resolved (same as write)
3. `metadata_backend.get_entry(cache_key)` retrieves entry metadata
4. `_is_expired()` checks per-entry TTL and config default TTL
5. If expired: entry is auto-deleted and `None` returned (TTL enforcement)
6. `_verify_entry()` checks file integrity (xxhash) and signature (HMAC) if enabled
7. If inline blob: `_read_inline_blob()` deserializes from bytes stored in metadata row
8. Otherwise: `_blob_store._read_blob(file_path, data_type, metadata)` dispatches to the correct handler's `get()` method
9. `metadata_backend.update_access_time(cache_key)` records the access
10. Cache hit/miss stats are updated

**State Management:**
- Metadata state is persisted in the chosen backend (JSON file, SQLite DB, or PostgreSQL)
- Blob data is persisted via the blob backend (filesystem files, S3 objects, or inline in metadata)
- Thread safety via a shared `threading.RLock` between `UnifiedCache` and `BlobStore`
- Optional in-memory metadata cache layer (`cachetools.TTLCache`) for read-heavy workloads

## Key Abstractions

**UnifiedCache (exported as `cacheness`):**
- Purpose: The main cache coordinator — one instance per cache directory/namespace
- Examples: `src/cacheness/core.py`
- Pattern: Facade/Coordinator — delegates storage to `BlobStore`, serialization to handlers, metadata to backends

**CacheHandler (ABC):**
- Purpose: Type-aware serialization — each handler knows how to serialize/deserialize one data type
- Examples: `PolarsDataFrameHandler`, `PandasDataFrameHandler`, `ArrayHandler`, `ObjectHandler`, `BytesHandler` in `src/cacheness/handlers.py`
- Pattern: Strategy pattern — handlers are registered in priority order; first match wins

**HandlerRegistry:**
- Purpose: Ordered collection of handlers; dispatches `get_handler(data)` to find the right strategy
- Examples: `src/cacheness/handlers.py` (bottom of file)
- Pattern: Registry/Chain of Responsibility — iterates handlers until one claims the data

**MetadataBackend (ABC):**
- Purpose: Abstract interface for cache metadata CRUD — entries, stats, namespaces, schema versioning
- Examples: `JsonBackend`, `SqliteBackend` in `src/cacheness/metadata.py`; `PostgresBackend` in `src/cacheness/storage/backends/postgresql_backend.py`
- Pattern: Abstract Factory — `create_metadata_backend(name, **kwargs)` instantiates the right backend

**BlobBackend (ABC):**
- Purpose: Abstract interface for raw binary blob storage
- Examples: `FilesystemBlobBackend`, `InMemoryBlobBackend` in `src/cacheness/storage/backends/blob_backends.py`; `S3BlobBackend` in `src/cacheness/storage/backends/s3_backend.py`
- Pattern: Abstract Factory — `get_blob_backend(name, **kwargs)` instantiates via registry

**BlobStore:**
- Purpose: Low-level storage engine without caching semantics — reusable for artifact stores, ML model versioning
- Examples: `src/cacheness/storage/blob_store.py`
- Pattern: Service — provides put/get/delete/verify_integrity for arbitrary Python objects

**CacheConfig (dataclass tree):**
- Purpose: Hierarchical validated configuration with backwards-compatible flat parameter constructors
- Examples: `src/cacheness/config.py`
- Pattern: Composite configuration — sub-dataclasses for each concern (storage, metadata, blob, compression, serialization, handlers, security, hooks)

**EntryList:**
- Purpose: Rich result wrapper for cache entry listings — extends `list` with `.to_dataframe()`, `.filter()`, `.sort_by()`, `.keys()`
- Examples: `src/cacheness/entry_list.py`
- Pattern: Decorator pattern on `list` — fully backward compatible

## Entry Points

**Decorator API (`@cached`, `@cache_if`):**
- Location: `src/cacheness/decorators.py`
- Triggers: Function calls decorated with `@cached()` or `@cache_if(condition)`
- Responsibilities: Generates cache keys from function signatures, creates/shares `UnifiedCache` instances, handles TTL, atexit cleanup

**Direct API (`cache.put()` / `cache.get()`):**
- Location: `src/cacheness/core.py` — `UnifiedCache.put()`, `UnifiedCache.get()`
- Triggers: Explicit user calls
- Responsibilities: Full cache lifecycle — write, read, delete, list, query, verify integrity, statistics

**Factory Function (`get_cache()`):**
- Location: `src/cacheness/core.py` (line ~3878)
- Triggers: `from cacheness import get_cache; cache = get_cache(cache_dir="./my_cache")`
- Responsibilities: Convenience constructor that builds `CacheConfig` from flat kwargs and returns a `UnifiedCache` instance

**Storage Mode (standalone key-value store):**
- Location: `src/cacheness/core.py` — `_storage_mode_put()`, `_storage_mode_get()`
- Triggers: `CacheConfig(storage_mode=True)` — disables TTL, eviction, stats, auto-delete
- Responsibilities: Pure persistent key-value storage using the same handler/backend infrastructure

**BlobStore standalone API:**
- Location: `src/cacheness/storage/blob_store.py`
- Triggers: `from cacheness.storage import BlobStore; store = BlobStore(...)`
- Responsibilities: Low-level blob storage without caching semantics — for ML models, artifacts, pipeline checkpoints

**Package import:**
- Location: `src/cacheness/__init__.py`
- Exports: `cacheness` (alias for `UnifiedCache`), `cached`, `cache_if`, `get_cache`, handler classes, backend classes, config classes, interface types

## Module Dependency Graph

```
__init__.py ─────────────────────────────┐
  ├── core.py (UnifiedCache)             │
  │     ├── config.py (CacheConfig)      │
  │     ├── handlers.py (HandlerRegistry)│
  │     ├── metadata.py (backends)       │
  │     ├── serialization.py (key gen)   │
  │     ├── security.py (signing)        │
  │     ├── entry_list.py (EntryList)    │
  │     ├── size_utils.py (parsing)      │
  │     ├── custom_metadata.py           │
  │     └── storage/                     │
  │           ├── blob_store.py          │
  │           ├── paths.py               │
  │           ├── compression.py ────────┤── compress_pickle.py
  │           ├── security.py ───────────┤── security.py (re-export)
  │           ├── handlers/ ─────────────┤── handlers.py (re-export)
  │           └── backends/              │
  │                 ├── base.py ─────────┤── metadata.py (re-export)
  │                 ├── blob_backends.py │
  │                 ├── s3_backend.py    │
  │                 └── postgresql_backend.py
  ├── decorators.py (cached, cache_if)   │
  └── interfaces.py (ABCs, TypedDicts)   │ ◄── leaf module, no internal deps
```

**Circular import prevention:** `interfaces.py` is a leaf module with zero internal imports. All ABCs (`CacheHandler`, `CacheWriter`, `CacheReader`) and typed contracts (`HandlerResult`, `EntryData`, `EntrySummary`, `BlobReadContext`) live here. The `storage/` sub-package re-exports from parent modules (`handlers.py`, `metadata.py`, `security.py`) to provide a clean `from cacheness.storage import ...` API without duplicating definitions.

## Extension Points

**Custom Handlers:**
- Implement `CacheHandler` ABC from `src/cacheness/interfaces.py` (methods: `can_handle()`, `put()`, `get()`)
- Register via `HandlerRegistry` — handlers are checked in priority order

**Custom Metadata Backends:**
- Implement `MetadataBackend` ABC from `src/cacheness/metadata.py`
- Register via `register_metadata_backend(name, cls)` in `src/cacheness/storage/backends/__init__.py`

**Custom Blob Backends:**
- Implement `BlobBackend` ABC from `src/cacheness/storage/backends/blob_backends.py`
- Register via `register_blob_backend(name, cls)` in `src/cacheness/storage/backends/blob_backends.py`

**Custom Metadata Models:**
- Use `@custom_metadata_model(schema_name)` decorator from `src/cacheness/custom_metadata.py`
- Define SQLAlchemy columns; linked to cache entries via foreign key with cascade delete

**Lifecycle Hooks:**
- `HooksConfig` in `src/cacheness/config.py` — `on_put`, `on_get`, `on_delete`, `on_evict` callbacks

**Configuration-driven handler toggling:**
- `HandlerConfig` in `src/cacheness/config.py` — enable/disable individual handler types (pandas, polars, numpy, tensorflow, dill fallback) without code changes

## Error Handling

**Strategy:** Hierarchical exception classes rooted at `CacheError`, with context dicts for structured logging. Handler errors are specialized (`CacheWriteError`, `CacheReadError`, `CacheFormatError`).

**Patterns:**
- `_PutCleanup` rollback guard in `core.py` ensures blob cleanup on write failures
- `cache_operation_context()` context manager in `error_handling.py` for structured error logging
- `get()` auto-deletes corrupt entries (configurable via `delete_on_error`) — destructive on errors except transient I/O
- Storage mode `get()` preserves entries on error and returns `None`
- Lazy imports for optional dependencies (TensorFlow, dill, blosc2) with graceful degradation

## Cross-Cutting Concerns

**Logging:** Python `logging` module — structured log messages with emoji prefixes for quick visual parsing (✅, 🗄️, 🔒, 📊, ⚠️)
**Validation:** Dataclass `__post_init__()` validators in config; `validate_namespace_id()` regex for namespace IDs; size/duration parsing with `ValueError` on invalid input
**Thread Safety:** Shared `threading.RLock` across `UnifiedCache` and `BlobStore`; re-entrant to support nested operations (e.g. `delete_where` → `invalidate`)
**Namespace Isolation:** Namespace-scoped tables (SQLite/PostgreSQL), subdirectories (filesystem), and metadata — configured once at init, immutable thereafter

---

*Architecture analysis: 2026-04-02*
