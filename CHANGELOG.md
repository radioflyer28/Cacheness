# Changelog

All notable changes to Cacheness will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.0] - Unreleased

### Changed

- Cache key fallback serialization is now stable across processes for large tuples and default object fallback paths. Existing cached entries created via the old unstable `hash()` or memory-address `repr` fallback may become unreachable; this is intentional because those keys could already change across Python processes or `PYTHONHASHSEED` values.

## [0.6.0] - 2026-02-22

### Added

- **File storage API**: `put_file()`/`get_file()` on both `UnifiedCache` and `BlobStore` with automatic metadata (filename, MIME type, size) (CACHE-zsw)
  - `put_file(move=True)`: move-in semantics — deletes source file after caching (CACHE-2fb)
  - `get_file(dest=..., move=True)`: move-out semantics — deletes cache entry after writing to disk (CACHE-2fb)
  - `get_file(overwrite=False)`: raises `FileExistsError` if destination already exists (CACHE-2fb)
- **Convenience metadata helpers** on `UnifiedCache` — eliminates the common pattern of passing the same values twice for key derivation and metadata storage (CACHE-dmr, CACHE-a2t):
  - `put_with_meta(data, **kwargs)` — stores kwargs as both cache key and `metadata_dict`
  - `get_with_meta(**kwargs)` → `(data, metadata_dict)` — retrieves data + stored kwargs
  - `put_with_model(data, ModelClass, **kwargs)` — stores kwargs as both cache key and ORM instance
  - `get_with_model(ModelClass, **kwargs)` → `(data, orm_instance)` — retrieves data + ORM row
  - `query_with_meta(**filters)` — lazily yields `(data, metadata_dict)` for all matching entries
  - All helpers accept `on=dict` — extra key-only parameters that affect the cache key but are not stored in metadata (creates distinct entries with identical metadata)
- **Inline blob storage**: small entries stored directly in the metadata row, eliminating a round-trip to the blob backend (CACHE-gf0)
  - Configurable via `CacheConfig(blob=CacheBlobConfig(max_inline_size=4096))` — disabled (`0`) by default
  - `put_bytes`/`get_bytes` protocol on the handler ABC — allows handlers to skip disk I/O entirely for inline blobs, with temp-file as universal fallback for handlers that don't implement it (CACHE-728)
- **`BytesHandler`**: stores raw `bytes`, `bytearray`, and `memoryview` values natively without file I/O (CACHE-slh)
- **Schema Phase 1**: new columns on all metadata backends (CACHE-cq5):
  - `access_count` — incremented read counter per entry
  - `ttl_seconds` — TTL stored at write time for introspection
  - `expires_at` — pre-computed expiration timestamp with partial index for fast cleanup scans

### Performance

- **PostgreSQL**: `metadata_dict` and `cache_key_params` columns upgraded from `Text` (JSON string) to native `JSONB`, enabling server-side containment queries and GIN indexing (CACHE-jcm)
- **SQLite**: partial index on `metadata_dict IS NOT NULL` — speeds up `query_meta()` scans when most entries carry no metadata dict (CACHE-aei)

### Internal

- **Tier 1–3 refactors** (CACHE-8yw, CACHE-22z, CACHE-aqq): major extraction pass on `core.py` — helper methods like `_resolve_cache_key()`, `_verify_entry()`, `_sign_entry_if_enabled()`, `_cleanup_stale_blob()`, `BlobReadContext`, `EntryData` TypedDict, normalized Parquet handler error handling, and `HooksConfig` lifecycle callbacks
- `core.py` reduced from ~3100 to ~2940 lines across all three tiers

### Fixed

- Flaky parallel tests: `test_memoize_basic`, `test_path_object_handling`, `test_cache_consistency` migrated from shared `cache/default/` directory to `tmp_path` isolation (CACHE-aa0)

### Tests

- Test suite: 1603 passed, 102 skipped, 0 failures (~50s parallel)

---

## [0.5.2] - 2026-02-19

### Added

- `EntryList` result wrapper for `list_entries()`/`query_meta()` with `.keys()`, `.filter()`, `.sort_by()`, `.to_dataframe()` (CACHE-5d3)
- `HandlerResult` dataclass replacing untyped dicts from handler `put()` (CACHE-98f)
- `WriteBlobResult` and `IntegrityReport` typed contracts for `_write_blob()` and `verify_integrity()` (CACHE-d5x)
- `EntryData` TypedDict — canonical contract for `get_entry()`/`put_entry()` across all metadata backends (CACHE-aqq)
- `HooksConfig` lifecycle callbacks — `on_evict(cache_key, reason)` and `on_integrity_failure(cache_key, failure_type, detail)` with flat-kwarg convenience in `CacheConfig.__init__` (CACHE-aqq)
- `delete_on_error` config option (`CacheMetadataConfig`) — when `False`, `get()` returns `None` on errors but preserves cache entries instead of auto-deleting (CACHE-ajv)
- `__len__`, `__contains__`, `__iter__` dunder methods on `UnifiedCache`
- Human-readable duration strings for TTL (`"1h"`, `"7d"`, `"30d"`)
- `ttl` parameter alias on `put()` (alongside `ttl_seconds`)
- Namespace registry signing — HMAC-SHA256 signatures on namespace registry rows prevent tampering with namespace metadata (CACHE-0xc)
- SQLite ORM-to-Core column select optimization for `list_entries()` and `get_entry()` — reduces overhead for large caches
- `pytest-xdist` parallel test execution (`-n auto --dist loadgroup`)

### Fixed

- `put()` metadata columns (`storage_format`, `compression_codec`, `serializer`, `object_type`) now populated for all 7 handlers (CACHE-198)
- Transaction ordering: metadata-first delete, `get()` blob cleanup, `put()` overwrite cleanup
- File size units standardized to bytes internally (`max_cache_size` accepts `"2gb"` strings)
- Pre-commit hook re-stages files after `ruff format`
- All 4 Parquet handlers now consistently raise `CacheWriteError`/`CacheReadError` with `cache_operation_context` (previously only `PolarsDataFrameHandler` had error wrapping) (CACHE-aqq)

### Internal

- **Tier 1 refactors** (CACHE-8yw): `_resolve_hash_key_alias()`, `_resolve_cache_key()`, `_enrich_entry()`, `_merge_entry_metadata()` extracted from core.py
- **Tier 2 refactors** (CACHE-22z): `_verify_entry()`, `_build_metadata_dict()`, `_sign_entry_if_enabled()`, `_cleanup_stale_blob()`, `BlobReadContext` TypedDict
- **Tier 3 refactors** (CACHE-aqq): `EntryData` TypedDict, normalized Parquet handler error handling, `HooksConfig` lifecycle callbacks with `_invoke_hook()` dispatch
- core.py reduced from ~3100 to ~2940 lines across all three tiers

### Tests

- Test suite: 1424 passed, 65 skipped, 0 failures (~37s parallel)

## [0.5.1] - 2026-02-16

### Added

- S3 ETag-based integrity verification: `verify_integrity(verify_hashes=True)` uses cheap HEAD check instead of downloading entire S3 blobs (CACHE-6x5)
- `list_blobs()` method on all BlobBackend implementations
- S3 ETag verification and integrity error handling in `read_blob()`
- Per-namespace migration runner

### Fixed

- `update_data()` write-then-swap to prevent data loss on crash (CACHE-c1e)
- `query_meta()` SQLite fast path now respects namespace isolation (CACHE-2o9)
- Custom metadata ORM is namespace-aware with per-namespace tables (CACHE-oow)
- `verify_integrity()` uses blob_backend uniformly, fixing sharded blob discovery
- All blob writes routed through blob_backend (CACHE-ee3)
- S3 blob orphan cleanup on metadata write failure (CACHE-yxf)
- SQLAlchemy global metadata pollution in custom metadata tests (CACHE-qg3)
- Normalized `actual_path` to relative forward-slash format (CACHE-6o0)
- Consolidated `_resolve_actual_path` into shared paths module (CACHE-74o)

### Infrastructure

- Replaced MinIO with Garage for S3 testing
- Widened `entry_signature` column to VARCHAR(100)
- Metadata column cleanup (CACHE-a1d epic)

### Tests

- Test suite: 1202 passed, 65 skipped, 0 failures

## [0.5.0] - 2026-02-15

### Breaking Changes

- Removed `prefix` parameter from all public API methods (`put`, `get`, `exists`, `invalidate`, `get_with_metadata`)
- Removed `prefix` column from all metadata backends (JSON, SQLite, PostgreSQL)
- Schema baseline reset to v1 (no migration path from v0)

### Added

- Full namespace support: per-namespace metadata isolation across all backends
- Schema versioning infrastructure for all backends (JSON, SQLite, PostgreSQL)
- Filesystem blob namespace isolation
- Namespace-scoped eviction and cleanup
- Signing key coordination documentation
- `cleanup_by_size()` on all metadata backends with LRU eviction

### Fixed

- Replaced `eval()` with `ast.literal_eval()` in array handler (security)
- Fixed `invalidate()`/`delete()` to clean up blob files
- Fixed storage-mode signing gap (`_storage_mode_put` now signs entries)
- Fixed `JsonBackend` deadlock from non-reentrant lock
- Fixed cache key inconsistency for positional vs keyword args

### Tests

- Test suite: 1092 passed, 97 skipped, 0 failures

## [0.4.0] - 2026-02-07

- Initial tracked release
