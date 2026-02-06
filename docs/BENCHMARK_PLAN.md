# Benchmark Review & Optimization Plan

## Review Date: February 6, 2026

After removing InMemoryBackend, SqlCache, `ttl_hours`, and refactoring `update_blob_data` → `update_entry_metadata`, the benchmark suite needs updating to reflect the current codebase and to detect performance regressions from these changes.

---

## Current State Audit

### Files in `benchmarks/`

| File | Status | Issues |
|---|---|---|
| `comprehensive_backend_benchmark.py` | 🔴 **Broken** | Uses `metadata_backend="memory"` (removed). References InMemoryBackend throughout. |
| `optimization_results.py` | 🔴 **Broken** | Uses `metadata_backend="memory"` ("InMemory Backend (unified)"). Outputs reference InMemory backend in summary. |
| `list_performance_analysis.py` | 🔴 **Broken** | Uses `metadata_backend="memory"` in all functions. Key insights reference InMemoryBackend. |
| `sqlite_metadata_analysis.py` | 🔴 **Broken** | Queries `metadata_json` column which no longer exists in SQLite schema. |
| `test_memory_backend.py` | 🔴 **Dead** | Entirely tests the removed `metadata_backend="memory"`. 100% obsolete. |
| `test_performance_comparison.py` | 🟡 **Functional** | Tests `serialize_for_cache_key` which still exists. No broken references. |
| `threshold_benchmark.py` | 🟢 **Functional** | Tests `hash_directory_parallel` / `_hash_directory_sequential` which still exist. No broken references. |
| `README.md` | 🔴 **Stale** | Documents deprecated benchmarks, references "Memory backend", mentions `metadata_json` optimizations. |

### Root Causes

1. **InMemoryBackend removal**: The `"memory"` metadata backend was removed. Valid backends are now: `"auto"`, `"json"`, `"sqlite"`, `"sqlite_memory"`, `"postgresql"`.
2. **`metadata_json` column removal**: SQLite schema was refactored to use dedicated columns instead of a JSON blob. Any benchmark querying `metadata_json` directly will crash.
3. **`update_blob_data` → `update_entry_metadata`**: Metadata backends no longer do blob I/O. No benchmarks currently test management operations (update_data, delete_where, delete_matching, touch, batch ops).
4. **`ttl_hours` removal**: Only `ttl_seconds` exists now. No benchmarks reference `ttl_hours` (not an issue).

---

## Action Plan

### Phase 1: Remove Dead Files

| # | Action | Rationale |
|---|---|---|
| 1.1 | **Delete `test_memory_backend.py`** | 100% tests the removed `metadata_backend="memory"`. Every line is broken. |

### Phase 2: Fix Broken Benchmarks

| # | File | Changes Required |
|---|---|---|
| 2.1 | `comprehensive_backend_benchmark.py` | Replace all `"memory"` → `"sqlite_memory"`. Update backend lists from `["memory", "json", "sqlite"]` → `["json", "sqlite", "sqlite_memory"]`. Update summary text. |
| 2.2 | `optimization_results.py` | Replace `"memory"` → `"sqlite_memory"` in backend list. Update label `"InMemory Backend (unified)"` → `"SQLite In-Memory"`. Remove InMemory references from summary output. |
| 2.3 | `list_performance_analysis.py` | Replace all `"memory"` → `"sqlite_memory"` in all 3 functions. Update key insights text from "InMemoryBackend" → "SQLite In-Memory". |
| 2.4 | `sqlite_metadata_analysis.py` | Remove `metadata_json` column queries — the column no longer exists. Rewrite `analyze_sqlite_metadata_overhead()` to inspect the new dedicated columns. Update `estimate_optimization_impact()` to measure current schema efficiency instead of analyzing the removed JSON blob. |

### Phase 3: Add Missing Benchmark Coverage

New operations were added during Phase 3 Management Operations that have no benchmark coverage:

| # | Operation | Why Benchmark |
|---|---|---|
| 3.1 | `update_data()` | New blob I/O was moved from metadata layer to core. Need to verify the refactored path doesn't regress. |
| 3.2 | `delete_where(filter_fn)` | Iterates all entries, applies filter, deletes matches. O(n) scan — could be slow at scale. |
| 3.3 | `delete_matching(**kwargs)` | Finds entries by kwargs, deletes matches. Query + delete path. |
| 3.4 | `touch()` / `touch_matching()` | Metadata-only update. Should be fast but worth baselining. |
| 3.5 | `get_batch(cache_keys)` | Sequential gets. Could expose per-get overhead at scale. |
| 3.6 | `delete_batch(cache_keys)` | Sequential deletes. Similar concern to get_batch. |

These should be consolidated into a single new benchmark file: `management_ops_benchmark.py`.

### Phase 4: Update Documentation

| # | Action |
|---|---|
| 4.1 | Rewrite `benchmarks/README.md` to reflect current file inventory, valid backends, and removal of InMemoryBackend. |

---

## Benchmark-to-Code Mapping

Which benchmarks cover which code paths:

| Code Path | Benchmark File | Notes |
|---|---|---|
| `cache.put()` | `comprehensive_backend_benchmark.py` | PUT ops/sec across backends |
| `cache.get()` | `comprehensive_backend_benchmark.py` | GET ops/sec, cache hit patterns |
| `cache.list_entries()` | `comprehensive_backend_benchmark.py`, `list_performance_analysis.py` | Scaling characteristics |
| `cache.get_stats()` | `optimization_results.py` | Memory cache layer stats |
| `cache.update_data()` | **None** ← gap | Needs new benchmark |
| `cache.delete_where()` | **None** ← gap | Needs new benchmark |
| `cache.delete_matching()` | **None** ← gap | Needs new benchmark |
| `cache.touch()` | **None** ← gap | Needs new benchmark |
| `cache.get_batch()` | **None** ← gap | Needs new benchmark |
| `cache.delete_batch()` | **None** ← gap | Needs new benchmark |
| `cache.touch_matching()` | **None** ← gap | Needs new benchmark |
| `serialize_for_cache_key()` | `test_performance_comparison.py` | Serialization ordering |
| `hash_directory_parallel()` | `threshold_benchmark.py` | Parallel vs sequential thresholds |
| `CachedMetadataBackend` (memory cache layer) | `comprehensive_backend_benchmark.py`, `optimization_results.py` | LRU cache effectiveness |
| `SqliteBackend.get_stats()` (SQL aggregates) | `optimization_results.py` | Was Fix 6 — verify no regression |

---

## Success Criteria

1. **All benchmarks run without errors**: `uv run python benchmarks/<file>.py` exits cleanly for every file.
2. **No references to removed APIs**: No `"memory"` backend, no `metadata_json`, no `InMemoryBackend`, no `update_blob_data`.
3. **Management operations baselined**: New `management_ops_benchmark.py` covers all Phase 3 operations with ops/sec metrics.
4. **README accurate**: Documents current file inventory and valid backend options.

---

## Execution Order

1. Delete `test_memory_backend.py`
2. Fix `comprehensive_backend_benchmark.py` (highest value — primary benchmark)
3. Fix `optimization_results.py`
4. Fix `list_performance_analysis.py`
5. Fix `sqlite_metadata_analysis.py`
6. Create `management_ops_benchmark.py`
7. Update `benchmarks/README.md`
8. Run all benchmarks end-to-end to verify
