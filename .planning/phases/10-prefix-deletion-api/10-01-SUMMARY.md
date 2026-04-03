---
phase: 10-prefix-deletion-api
plan: 01
status: complete
---

# Summary: Prefix Deletion API

## What Was Built

`delete_by_prefix()` public API on UnifiedCache for bulk deletion of cache entries by key prefix. Backend-optimized: SQLite uses SQL `LIKE` for prefix matching, JSON backend uses Python-side `startswith` filtering.

## Files Created/Modified

| File | Change |
|------|--------|
| `src/cacheness/metadata/base.py` | Added `keys_by_prefix()` to MetadataBackend ABC (default: Python filter) + MetadataBackendWrapper delegation |
| `src/cacheness/metadata/sqlite_backend.py` | Overrode `keys_by_prefix()` with SQL `LIKE` query on namespace-specific table |
| `src/cacheness/metadata/json_backend.py` | Overrode `keys_by_prefix()` with explicit Python dict keys filter |
| `src/cacheness/core.py` | Added `delete_by_prefix(prefix)` -> int method in Bulk & Batch Operations section |
| `tests/test_core.py` | 5 new tests in `TestDeleteByPrefix` class |

## Key Decisions

- **SQL LIKE with parameterized queries**: Safe against injection, uses namespace-specific table name
- **Explicit cache keys**: Prefix deletion operates on explicit cache keys (set via `cache_key=` param), not on hashed kwarg-derived keys
- **Both metadata + blob deletion**: Uses `_blob_store.delete()` which handles both

## Test Results

- 5 new tests: basic prefix delete, blob removal verification, no-match returns 0, empty prefix matches all, SQLite backend path
- Full suite: 1641 passed, 101 skipped, 0 failures
