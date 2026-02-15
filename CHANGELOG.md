# Changelog

All notable changes to Cacheness will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
