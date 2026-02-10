# Architecture & Separation of Concerns

This document describes Cacheness's internal architecture, the separation of responsibilities between components, and the design principles that guide development.

## Design Principles

1. **Each layer owns exactly one concern** — metadata, serialization, or blob lifecycle
2. **Handlers own format-specific I/O** — user-extensible handlers choose how data is serialized (parquet, npz, pkl, etc.)
3. **Metadata backends are pure indexes** — they never touch blob files
4. **Blob backends manage file lifecycle** — existence checks, deletion, and future replication

## Component Overview

```
┌──────────────────────────────────────────────────────────┐
│                    User-Facing APIs                      │
│          UnifiedCache / @cached / @cache_if              │
│              BlobStore (low-level)                        │
├──────────────────────────────────────────────────────────┤
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

## The Three Backend Types

### 1. Metadata Backends (`MetadataBackend`)

**Concern:** Index of cache entries — keys, timestamps, data types, custom metadata.

**Never touches blob files.** Metadata backends store references (paths) to blob files but never read, write, or delete them.

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

**Current scope:** `BlobStore` uses `FilesystemBlobBackend` for `delete()`, `exists()`, and file validation in `get()`. The `put()`/`get()` serialization path still routes through handlers directly.

**Important limitation:** The blob backend registry (`register_blob_backend`, `get_blob_backend`) is exported as public API but is **not yet wired** into the main data path. See [CACHE-69y] for the roadmap to resolve this.

## Data Flow

### PUT (store data)

```
User calls cache.put(data, key="my-key")
  │
  ├── 1. Generate cache key (xxhash)
  ├── 2. Detect data type → select handler
  ├── 3. Handler serializes & writes to file_path     ← Handler owns I/O
  ├── 4. Metadata backend records entry                ← Metadata only
  └── 5. Return cache key
```

### GET (retrieve data)

```
User calls cache.get(key="my-key")
  │
  ├── 1. Metadata backend looks up entry               ← Metadata only
  ├── 2. Blob backend verifies file exists              ← Lifecycle check
  ├── 3. Handler reads & deserializes from file_path   ← Handler owns I/O
  └── 4. Return data
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
| `core.py` | UnifiedCache API, key generation, TTL, eviction | Direct file I/O, serialization |
| `handlers.py` | Type detection, serialization, format-specific I/O | Metadata operations, key generation |
| `compress_pickle.py` | Blosc2 compression utilities + ObjectHandler I/O | Metadata, key generation, handler selection |
| `metadata.py` | JSON metadata backend | File I/O for blobs, serialization |
| `storage/blob_store.py` | Low-level BlobStore API | Caching semantics (TTL, eviction) |
| `storage/backends/` | Metadata + blob backend implementations | Handler logic, serialization |
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
