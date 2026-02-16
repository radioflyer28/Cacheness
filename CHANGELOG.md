# Changelog

All notable changes to Cacheness will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
