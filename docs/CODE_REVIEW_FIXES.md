# Code Review Fixes Tracker

## Review Date: February 5, 2026

Systematic fixes for issues identified during the code quality, separation of concerns, and documentation accuracy review.

---

## 🔴 Critical — Separation of Concerns / Bugs

### Fix 1: Add `update_blob_data()` to MetadataBackend ABC
- **File:** `src/cacheness/metadata.py` — `MetadataBackend` class (line ~175)
- **Issue:** `update_blob_data()` is implemented in all 4 concrete backends (InMemory, Json, Sqlite, Postgres) but is NOT declared as an `@abstractmethod` in the base class. New backend implementations will silently lack this method, failing only at runtime.
- **Fix:** Add `@abstractmethod def update_blob_data(self, cache_key, new_data, handler, config) -> bool` to `MetadataBackend`.
- **Status:** [ ]

### Fix 2: Add `update_blob_data()` delegation to `CachedMetadataBackend`
- **File:** `src/cacheness/metadata.py` — `CachedMetadataBackend` class (line ~263)
- **Issue:** This wrapper delegates every `MetadataBackend` method EXCEPT `update_blob_data()`. If `enable_memory_cache=True`, calling `cache.update_data()` raises `AttributeError`. **This is a live bug.**
- **Fix:** Add delegation method that forwards to `self.backend.update_blob_data(...)` and invalidates the memory cache entry.
- **Status:** [ ]

### Fix 3: Fix `invalidate()` dead branch — `remove_entry()` returns None
- **File:** `src/cacheness/core.py` — `invalidate()` method (line ~1507)
- **Also:** `src/cacheness/metadata.py` — all `remove_entry()` implementations
- **Issue:** `invalidate()` does `if self.metadata_backend.remove_entry(cache_key):` but `remove_entry()` returns `None` in all backends (no return statement). The success log never fires; only the "not found" debug line runs even on successful deletions.
- **Fix:** Make `remove_entry()` return `bool` in all backends (True if entry existed and was removed, False otherwise). Update ABC signature to `-> bool`. Alternatively, fix `invalidate()` to not depend on return value.
- **Status:** [ ]

### Fix 4: Fix `touch()` ignoring `ttl_seconds` parameter
- **File:** `src/cacheness/core.py` — `touch()` method (line ~1224)
- **Issue:** `touch()` accepts `ttl_seconds: Optional[float] = None` but never uses it. Implementation just sets `created_at = now`. Users calling `cache.touch(key, ttl_seconds=172800)` get no TTL effect.
- **Fix:** Either implement TTL adjustment logic or remove the parameter and document the actual behavior (reset created_at to now, effectively extending TTL by the full default amount).
- **Status:** [ ]

---

## 🟡 Medium — Code Quality

### Fix 5: Field name inconsistency between `get_entry()` and `list_entries()`
- **File:** `src/cacheness/metadata.py` — all backend `list_entries()` implementations
- **Issue:** `get_entry()` returns `created_at`, `accessed_at`, `file_size`. `list_entries()` returns `created`, `last_accessed`, `size_mb`. Consumers must know which method produced the dict. Filter functions in `delete_where()` operate on `list_entries()` shapes which differ from `get_entry()`.
- **Fix:** Standardize field names. Keep `list_entries()` shape as-is (it's the user-facing format with human-readable names) but document the contract clearly. Alternatively, unify to one shape. Must check downstream consumers before changing.
- **Status:** [ ]

### Fix 6: `SqliteBackend.get_stats()` loads all rows into memory
- **File:** `src/cacheness/metadata.py` — `SqliteBackend.get_stats()` (line ~1493)
- **Issue:** `session.execute(select(CacheEntry)).scalars().all()` loads every CacheEntry row into Python just to count and sum. Should use SQL aggregates.
- **Fix:** Replace with `SELECT COUNT(*), SUM(file_size), COUNT(CASE WHEN data_type='dataframe' THEN 1 END), COUNT(CASE WHEN data_type='array' THEN 1 END) FROM cache_entries`.
- **Status:** [ ]

### Fix 7: `create_entry_cache()` ignores `cache_type` parameter
- **File:** `src/cacheness/metadata.py` — `create_entry_cache()` (line ~247)
- **Issue:** All 4 branches (lru, lfu, fifo, rr) create identical `TTLCache(maxsize, ttl)`. Config option is misleading.
- **Fix:** Either implement actual LFU/FIFO/RR using cachetools' `LFUCache`/`FIFOCache`/`RRCache` (they don't have built-in TTL, so need TTL wrapper), or simplify to only offer TTLCache/LRUCache and document it honestly. Recommend the latter — remove false options.
- **Status:** [ ]

### Fix 8: `JsonBackend` dead batching code
- **File:** `src/cacheness/metadata.py` — `JsonBackend` class
- **Issue:** Has `_batch_size = 10` and `_write_count` counter but every mutation calls `_save_to_disk()` unconditionally. Batch tracking is dead code.
- **Fix:** Either implement real write batching (defer `_save_to_disk()` until `_write_count >= _batch_size`) or remove the dead batch tracking fields.
- **Status:** [ ]

### Fix 9: Refactor `_init_metadata_backend()` if/elif chain
- **File:** `src/cacheness/core.py` — `_init_metadata_backend()` (line ~110)
- **Issue:** ~100-line if/elif chain with duplicated `SQLALCHEMY_AVAILABLE` checks and error messaging.
- **Fix:** Refactor to dispatch dict or factory pattern. Lower priority — functional but not clean.
- **Status:** [ ]

---

## 🟠 Documentation Accuracy

### Fix 10: Update `MISSING_MANAGEMENT_API.md` implementation status
- **File:** `docs/MISSING_MANAGEMENT_API.md` (bottom section, lines ~960-1022)
- **Issue:** All items (`update_blob_data`, `delete_by_prefix`, `touch`, `batch_operations`) still shown as `[ ]` unchecked despite Sprint 1-3 being complete with 47 passing tests.
- **Fix:** Check off completed items, add implementation notes with actual method names.
- **Status:** [ ]

### Fix 11: Fix API name mismatches in `MISSING_MANAGEMENT_API.md`
- **File:** `docs/MISSING_MANAGEMENT_API.md`
- **Issue:** Doc proposes `cache.replace()`, `backend.delete_by_prefix()`, `cache.update()` with `if_exists`. Actual implementation uses `cache.update_data()`, `cache.delete_where()`, `cache.delete_matching()`. Doc has duplicate sections (operations 2-8 appear twice with different numbering).
- **Fix:** Add a reconciliation section mapping proposed → actual names. Mark the doc as a historical design doc (not current API reference). Remove or consolidate duplicate sections.
- **Status:** [ ]

### Fix 12: ~~Update `DEVELOPMENT_PLANNING.md` Phase 3 status~~
- **Status:** N/A — `DEVELOPMENT_PLANNING.md` has been deleted (work tracked in beads-mcp now)

### Fix 13: Fix `API_REFERENCE.md` `delete_where` example data_type
- **File:** `docs/API_REFERENCE.md` (Management Operations section)
- **Issue:** Example likely uses `"pandas_dataframe"` but actual handler registers `data_type = "dataframe"`. Users following the example match zero entries.
- **Fix:** Verify the actual data_type string used by PandasDataFrameHandler and fix the example.
- **Status:** [ ]

---

## 🔵 Deferred — Larger Refactors

### Deferred A: Move blob I/O out of metadata backends
- **Files:** All `update_blob_data()` implementations in metadata.py, base.py, postgresql_backend.py
- **Issue:** Metadata backends called `handler.put(new_data, base_file_path, config)` — doing file I/O. Metadata layer should only track metadata; blob writes belong in `core.py`.
- **Rationale for deferral:** This is a significant architectural refactor that changes the method signature contract across all backends. Should be done as a dedicated sprint after the simpler fixes land.
- **Fix applied:** 
  - Renamed `update_blob_data(cache_key, new_data, handler, config)` → `update_entry_metadata(cache_key, updates)` across all backends (InMemory, Json, Sqlite, PostgreSQL) + both ABCs (`MetadataBackend`, storage `MetadataBackendBase`) + `CachedMetadataBackend` delegation
  - Moved blob I/O (`handler.put()`) into `core.py`'s `update_data()` method, mirroring how `put()` already works
  - Metadata backends now receive a plain `Dict[str, Any]` of field updates — no handler, no config, no file I/O
  - Updated all tests: `test_update_operations.py` (backend tests no longer need mock handlers/configs), `test_metadata_backend_registry.py` (mock backends updated)
- **Status:** [x] (Completed — blob I/O separated from metadata layer)

### Deferred B: Deprecated `ttl_hours` parameter cleanup
- **File:** `src/cacheness/core.py` — `_is_expired()` and `get()`
- **Issue:** `ttl_hours` parameter was DEPRECATED and has now been fully removed. The codebase consistently uses `ttl_seconds`.
- **Rationale for deferral:** Removing a deprecated parameter is a breaking change. Needs deprecation warning period and version bump.
- **Status:** [x] (Completed — `ttl_hours` removed, `ttl_seconds` is now the standard)

---

## Execution Order

1. **Fix 1 + Fix 2** — ABC and CachedMetadataBackend (critical bug, 5 min)
2. **Fix 3** — remove_entry return values (bug, 10 min)
3. **Fix 4** — touch() ttl_seconds parameter (bug, 15 min)
4. **Fix 5** — Field name audit and decision (design, 30 min)
5. **Fix 6** — SqliteBackend.get_stats SQL aggregates (perf, 15 min)
6. **Fix 7** — create_entry_cache honesty (quality, 20 min)
7. **Fix 8** — JsonBackend dead batching (quality, 10 min)
8. **Fix 9** — _init_metadata_backend refactor (quality, 30 min)
9. **Fix 10-13** — Documentation updates (docs, 30 min)
10. **Run full test suite** — Verify no regressions
