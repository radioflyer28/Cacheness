# Cacheness Benchmark Suite

Performance benchmarks for identifying regressions and optimization opportunities
across cacheness metadata backends, handlers, compression, security, and the decorator API.

## Valid Backends

| Backend | Config Value | Description |
|---------|-------------|-------------|
| JSON | `"json"` | File-per-entry, good for small caches (<500 entries) |
| SQLite | `"sqlite"` | Dedicated-column schema, best for large caches |
| SQLite In-Memory | `"sqlite_memory"` | Ephemeral `:memory:` DB, good for tests/CI |
| PostgreSQL | `"postgresql"` | Production-grade, requires external server |

> **Note:** The `"memory"` (InMemoryBackend) metadata backend was removed. Use `"sqlite_memory"` for ephemeral caching.

## Benchmark Files

### `comprehensive_backend_benchmark.py` — Primary Backend Benchmark
Raw backend performance, memory cache layer impact, realistic workload patterns,
and scaling characteristics across JSON, SQLite, and SQLite in-memory.

```bash
uv run python benchmarks/comprehensive_backend_benchmark.py
```

### `management_ops_benchmark.py` — Management Operations Benchmark
Benchmarks all management operations: `update_data()`, `touch()`, `touch_batch()`,
`delete_where()`, `delete_matching()`, `get_batch()`, `delete_batch()`,
`invalidate()`, and `clear_all()`.
Measures ops/sec at varying cache sizes across all backends.

```bash
uv run python benchmarks/management_ops_benchmark.py
```

### `decorator_benchmark.py` — Decorator Overhead Benchmark
Benchmarks the `@cached` / `@cacheness_it` decorator — the primary user-facing API.
Measures cache key generation overhead, hit vs miss latency, raw function vs
decorated function overhead, memory cache layer impact, and TTL expiration cost.

```bash
uv run python benchmarks/decorator_benchmark.py
```

### `handler_benchmark.py` — Data Handler Benchmark
Benchmarks put/get performance per data-type handler: ObjectHandler (dicts),
ArrayHandler (NumPy), PandasDataFrameHandler, PandasSeriesHandler,
PolarsDataFrameHandler, PolarsSeriesHandler. Tests small/medium data sizes
and scaling characteristics.

```bash
uv run python benchmarks/handler_benchmark.py
```

### `compression_benchmark.py` — Compression Codec Benchmark
Benchmarks compression codec impact (lz4, zstd, gzip) on put/get performance
and file size. Includes raw codec micro-benchmark via `benchmark_codecs()`,
end-to-end put/get with each codec, and comparison table.

```bash
uv run python benchmarks/compression_benchmark.py
```

### `security_benchmark.py` — Security Signing Overhead Benchmark
Benchmarks HMAC-SHA256 entry signing/verification overhead. Measures raw
sign/verify micro-cost, field count impact, end-to-end put/get with signing
enabled vs disabled, and throughput impact at scale.

```bash
uv run python benchmarks/security_benchmark.py
```

### `sqlite_metadata_analysis.py` — SQLite Schema Analysis
Analyzes the dedicated-column SQLite schema: column utilization, storage
overhead per entry, column query performance, and JSON vs SQLite list comparison.

```bash
uv run python benchmarks/sqlite_metadata_analysis.py
```

### `serialization_benchmark.py` — Serialization Key Performance
Tests `serialize_for_cache_key()` ordering performance with different key
structures and sizes.

```bash
uv run python benchmarks/serialization_benchmark.py
```

### `threshold_benchmark.py` — Directory Hashing Thresholds
Tests parallel vs sequential directory hashing performance to find the
optimal switchover threshold for `hash_directory_parallel()`.

```bash
uv run python benchmarks/threshold_benchmark.py
```

## Running All Benchmarks

```bash
# Run one at a time (recommended)
uv run python benchmarks/comprehensive_backend_benchmark.py
uv run python benchmarks/management_ops_benchmark.py
uv run python benchmarks/decorator_benchmark.py
uv run python benchmarks/handler_benchmark.py
uv run python benchmarks/compression_benchmark.py
uv run python benchmarks/security_benchmark.py
uv run python benchmarks/sqlite_metadata_analysis.py
uv run python benchmarks/serialization_benchmark.py
uv run python benchmarks/threshold_benchmark.py
```

## Architecture Notes

### SQLite Schema
The SQLite backend uses **dedicated columns** (not a JSON blob):

```
cache_key, description, data_type, prefix, created_at, accessed_at,
file_size, file_hash, entry_signature, s3_etag, object_type,
storage_format, serializer, compression_codec, actual_path,
cache_key_params (JSON Text), metadata_dict (JSON Text)
```

Benefits:
- No JSON parsing overhead on `list_entries()` / `get_stats()`
- SQL aggregates for statistics instead of loading all rows
- Column-level queries faster than JSON blob scanning

### Memory Cache Layer
- `CachedMetadataBackend`: Optional cachetools LRU wrapper around any backend
- Enabled via `enable_memory_cache=True`
- Provides 15-25% improvement for SQLite, 5-10% for JSON

### Benchmark Coverage Map

| Code Path | Benchmark File |
|-----------|---------------|
| `put()` / `get()` | comprehensive_backend_benchmark, handler_benchmark |
| `list_entries()` | comprehensive_backend_benchmark, sqlite_metadata_analysis |
| `get_stats()` | comprehensive_backend_benchmark, sqlite_metadata_analysis |
| `update_data()` | management_ops_benchmark |
| `touch()` / `touch_batch()` | management_ops_benchmark |
| `delete_where()` / `delete_matching()` | management_ops_benchmark |
| `get_batch()` / `delete_batch()` | management_ops_benchmark |
| `invalidate()` / `clear_all()` | management_ops_benchmark |
| `@cached` / `@cacheness_it` decorator | decorator_benchmark |
| Handler dispatch per data type | handler_benchmark |
| Compression codecs (lz4/zstd/gzip) | compression_benchmark |
| Entry signing / verification | security_benchmark |
| `serialize_for_cache_key()` | serialization_benchmark |
| `hash_directory_parallel()` | threshold_benchmark |
| Memory cache layer | comprehensive_backend_benchmark, decorator_benchmark |
| SQLite schema / column queries | sqlite_metadata_analysis |

## Regression Thresholds

| Operation | Expected | Warning | Critical |
|-----------|----------|---------|----------|
| `put()` | <20ms | >50ms | >100ms |
| `get()` (hit) | <10ms | >30ms | >50ms |
| `update_data()` | <20ms | >50ms | >100ms |
| `touch()` | <5ms | >10ms | >30ms |
| `invalidate()` | <10ms | >20ms | >50ms |
| `clear_all()` (100) | <100ms | >300ms | >500ms |
| `list_entries()` (200) | <50ms | >200ms | >500ms |
| `get_batch()` (10) | <100ms | >300ms | >500ms |
| Decorator overhead (hit) | <5ms | >10ms | >30ms |
| `sign_entry()` | <50μs | >200μs | >500μs |
| Signing total overhead | <5% | >15% | >30% |
