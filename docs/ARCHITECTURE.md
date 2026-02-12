# Architecture & Separation of Concerns

This document describes Cacheness's internal architecture, the separation of responsibilities between components, and the design principles that guide development.

## Design Principles

1. **Each layer owns exactly one concern** — metadata, serialization, or blob lifecycle
2. **Handlers own format-specific I/O** — user-extensible handlers choose how data is serialized (parquet, npz, pkl, etc.)
3. **Metadata backends are pure indexes** — they never touch blob files
4. **Blob backends manage file lifecycle** — existence checks, deletion, and future replication
5. **UnifiedCache composes BlobStore** — cache concerns (TTL, eviction, stats) wrap a shared BlobStore for storage delegation

## Component Overview

```
┌──────────────────────────────────────────────────────────┐
│                    User-Facing APIs                      │
│          UnifiedCache / @cached / @cache_if              │
│              BlobStore (low-level)                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  UnifiedCache  ──composes──►  BlobStore                  │
│  (TTL, eviction, stats,       (blob I/O, integrity,     │
│   signing enrichment)          handler dispatch)         │
│                                                          │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │    Handlers      │  │   Metadata   │  │    Blob    │  │
│  │  (serialization) │  │   Backends   │  │  Backends  │  │
│  │                  │  │              │  │            │  │
│  │ DataFrameHandler │  │ JsonBackend  │  │ Filesystem │  │
│  │ ArrayHandler     │  │ SqliteBackend│  │ InMemory   │  │
│  │ ObjectHandler    │  │ PostgresBack.│  │ (S3, etc.) │  │
│  │ TensorHandler    │  │              │  │            │  │
│  │ (user-defined)   │  │              │  │            │  │
│  └────────┬─────────┘  └──────┬───────┘  └─────┬──────┘  │
│           │                   │                │         │
│     Serialize &          Index keys,      Delete/exists  │
│     write to disk       timestamps,       file lifecycle │
│     (format-specific)    metadata                        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Composition Model

`UnifiedCache` creates an internal `BlobStore` instance (`self._blob_store`) that shares the same metadata backend, handler registry, lock, and signer. Storage operations are delegated to BlobStore via its composition API:

- **`_write_blob()`** — handler dispatch + serialization + file hash computation
- **`_read_blob()`** — handler dispatch + deserialization from a file path
- **`_clear_blob_files()`** — delete all blob files by pattern
- **`_calculate_file_hash()`** — xxhash-based integrity hash
- **`verify_integrity()`** — full integrity verification of stored entries
- **`delete()` / `clear()`** — remove individual or all blob files + metadata

UnifiedCache adds cache concerns on top: TTL checking, size-based eviction, hit/miss statistics, signing enrichment, and auto-delete on errors.

When `storage_mode=True`, UnifiedCache early-returns to dedicated passthrough methods (`_storage_mode_put`, `_storage_mode_get`, `_storage_mode_get_with_metadata`) that delegate to BlobStore without any cache overhead.

## The Three Backend Types

### 1. Metadata Backends (`MetadataBackend`)

**Concern:** Index of cache entries — keys, timestamps, data types, custom metadata.

**Never touches blob files.** Metadata backends store references (paths) to blob files but never read, write, or delete them.

All metadata backends implement a single unified `MetadataBackend` ABC (`src/cacheness/storage/backends/base.py`). This ABC defines the complete interface: CRUD operations, TTL cleanup, statistics, iteration, and custom metadata. Both `UnifiedCache` and `BlobStore` share the same backend instance.

| Backend | Use Case |
|---------|----------|
| `JsonBackend` | Dev/small caches (<200 entries), human-readable, single-process only |
| `SqliteBackend` | Production, multi-process safe, scales to millions |
| `PostgresBackend` | Distributed deployments |
| `SqliteMemoryBackend` | Maximum performance for temporary data |

See [Backend Selection Guide](BACKEND_SELECTION.md) for detailed comparison.

### 2. Handlers (`CacheHandler`)

**Concern:** Format-specific serialization and deserialization **including file I/O**.

Handlers take a `file_path: Path` argument and manage both serialization and writing to disk. This is **by design** — user-defined handlers need control over I/O operations because different formats require different write strategies (e.g., `pd.to_parquet()`, `np.savez()`, `blosc2.save_array()`).

| Handler | Format | I/O Method |
|---------|--------|------------|
| `DataFrameHandler` | `.parquet` | `pd.to_parquet()` / `pd.read_parquet()` |
| `PolarsHandler` | `.parquet` | `pl.write_parquet()` / `pl.read_parquet()` |
| `ArrayHandler` | `.b2nd` | `blosc2.save_array()` / `blosc2.open()` |
| `TensorFlowHandler` | `.npz` | `np.savez()` / `np.load()` |
| `ObjectHandler` | `.pkl` | `compress_pickle.write_file()` / `read_file()` |
| `DillHandler` | `.pkl` | `dill.dump()` / `dill.load()` |

**Key design decision:** Handlers own I/O to local file paths. This means handlers write to the local filesystem. For remote storage scenarios (S3, GCS), the handler still writes locally and a separate sync/replication mechanism would move files to remote storage.

See [Plugin Development Guide](PLUGIN_DEVELOPMENT.md) for creating custom handlers.

### 3. Blob Backends (`BlobBackend`)

**Concern:** File lifecycle management — checking existence, deleting files, and (in future) managing replication.

Blob backends **do not** handle serialization — that's the handler's job. They manage the lifecycle of files that handlers have already written.

| Backend | Storage |
|---------|---------|
| `FilesystemBlobBackend` | Local filesystem (default) |
| `InMemoryBlobBackend` | In-memory dict (testing) |

**Current scope:** `BlobStore` provides the full storage data path including handler dispatch (`_write_blob`/`_read_blob`), xxhash-based integrity verification, cryptographic signing, and file lifecycle via `FilesystemBlobBackend`. `UnifiedCache` composes an internal `BlobStore` instance and delegates all storage I/O through it.

## Data Flow

### PUT (store data)

```
User calls cache.put(data, key="my-key")
  │
  ├── 1. Generate cache key (xxhash)
  ├── 2. [storage_mode?] → early-return to _storage_mode_put() (no TTL/eviction/stats)
  ├── 3. Delegate to BlobStore._write_blob():
  │      ├── Detect data type → select handler
  │      └── Handler serializes & writes to file_path   ← Handler owns I/O
  ├── 4. Enrich metadata (signing, cache_key_params)
  ├── 5. Metadata backend records entry                  ← Metadata only
  ├── 6. Enforce size limit (eviction)
  └── 7. Return cache key
```

### GET (retrieve data)

```
User calls cache.get(key="my-key")
  │
  ├── 1. [storage_mode?] → early-return to _storage_mode_get() (no TTL/stats/auto-delete)
  ├── 2. Metadata backend looks up entry               ← Metadata only
  ├── 3. Check TTL expiration
  ├── 4. Verify integrity (file hash via BlobStore)
  ├── 5. Verify signature (if signing enabled)
  ├── 6. Delegate to BlobStore._read_blob():
  │      └── Handler reads & deserializes from file_path ← Handler owns I/O
  ├── 7. Update access time, record hit
  └── 8. Return data
```

### DELETE (remove cached data)

```
User calls cache.invalidate(key="my-key")
  │
  ├── 1. Metadata backend looks up entry               ← Metadata only
  ├── 2. Blob backend deletes the file                  ← Lifecycle mgmt
  └── 3. Metadata backend removes entry                ← Metadata only
```

## Module Responsibilities

| Module | Responsibility | Should NOT contain |
|--------|---------------|--------------------|
| `core.py` | UnifiedCache API, key generation, TTL, eviction, composes BlobStore | Direct file I/O, serialization |
| `handlers.py` | Type detection, serialization, format-specific I/O | Metadata operations, key generation |
| `compress_pickle.py` | Blosc2 compression utilities + ObjectHandler I/O | Metadata, key generation, handler selection |
| `metadata.py` | JSON metadata backend | File I/O for blobs, serialization |
| `storage/blob_store.py` | Low-level BlobStore API, composition API for UnifiedCache | Caching semantics (TTL, eviction) |
| `storage/backends/` | Metadata + blob backend implementations | Handler logic, serialization |
| `storage/backends/base.py` | Unified MetadataBackend ABC | Implementation details |
| `config.py` | Configuration dataclasses | Business logic, I/O |
| `decorators.py` | @cached, @cache_if decorators | Direct cache operations |

## `compress_pickle.py` Organization

This module contains two categories of functionality:

1. **Serialization Utilities** (pure functions, no I/O):
   - `is_pickleable()`, `verify_pickleable()` — Pickle compatibility checks
   - `is_dill_serializable()`, `verify_dill_serializable()` — Dill compatibility checks
   - `optimize_compression_params()` — Data-aware blosc2 parameter tuning
   - `get_recommended_settings()` — Preset compression configurations

2. **File I/O Operations** (serialize + compress + write to disk):
   - `write_file()` / `read_file()` — Core blosc2 serialization
   - `write_file_with_metadata()` / `read_file_with_metadata()` — With metadata wrappers
   - `get_compression_info()` — File compression statistics
   - `benchmark_codecs()` — Codec performance comparison

The file I/O operations are consumed by `ObjectHandler` and `DillHandler`. Other handlers use their own format-specific I/O.

## Anti-Patterns to Avoid

When contributing to Cacheness, avoid these violations of separation of concerns:

1. **Metadata backends touching blob files** — metadata backends should never `open()`, `unlink()`, or `stat()` blob files
2. **Handlers managing metadata** — handlers should not call `backend.put_entry()` or know about cache keys
3. **Core module doing serialization** — `core.py` should delegate to handlers, never import serialization libraries directly
4. **Blob backends doing serialization** — blob backends manage file lifecycle (delete, exists), not data formats
5. **Direct `Path.unlink()` / `Path.exists()` in BlobStore** — use `self.blob_backend` methods for file lifecycle operations

## Related Documentation

- [Backend Selection Guide](BACKEND_SELECTION.md) — Choosing metadata backends
- [Plugin Development Guide](PLUGIN_DEVELOPMENT.md) — Creating custom handlers and backends
- [BlobStore Guide](BLOB_STORE.md) — Low-level storage API
- [API Reference](API_REFERENCE.md) — Complete API documentation
