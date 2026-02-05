# Cacheness Benchmark Suite

## Current Benchmarks

### `comprehensive_backend_benchmark.py` 🏆
**Primary benchmark** - Consolidates multiple backend tests into a single comprehensive suite.

**Coverage:**
- Raw backend performance comparison (JSON, SQLite, SQLite in-memory)
- Memory cache layer impact analysis (enabled vs disabled)
- Realistic workload patterns (Sequential, Random, Hot-Spot)
- Backend scaling characteristics across cache sizes
- Detailed memory cache effectiveness analysis

**Run with:** `uv run python comprehensive_backend_benchmark.py`

**Key Results (Updated with Schema Optimizations):**
- JSON backend: Good for small caches (<500 entries), simple row-based storage
- SQLite backend: Best for large caches (>500 entries), optimized dedicated columns
- SQLite in-memory (`sqlite_memory`): Fast ephemeral caching for tests/CI
- All backends: Consistent entry schema with backend-optimized storage patterns
- Memory cache layer provides 15-25% improvement for SQLite, 5-10% for JSON

### `test_performance_comparison.py` 
Cross-language and cross-system performance comparisons.

### `threshold_benchmark.py`
Cache size threshold analysis and optimization recommendations.

## Deprecated Benchmarks

The following benchmarks have been consolidated into `comprehensive_backend_benchmark.py` and moved to `deprecated_benchmarks/`:

- `backend_comparison_benchmark.py` - Raw backend comparison
- `test_entry_caching.py` - Memory cache layer testing  
- `test_realistic_caching.py` - Realistic workload patterns
- `quick_backend_demo.py` - Quick backend demonstration
- `list_performance_analysis.py` - List operation performance analysis

## Architecture Notes

### Memory Cache Layer
- **CachedMetadataBackend**: Optional cachetools LRU wrapper for JSON/SQLite backends
- **Memory Cache Layer**: Configurable via `enable_memory_cache=True/False`
- For ephemeral/in-memory caching, use `metadata_backend="sqlite_memory"`

### Performance Characteristics (Post Schema Optimization)
- **JSON Backend**: Simple {cache_key: entry} storage, eliminated columnar overhead
- **SQLite Backend**: Dedicated columns for backend metadata, eliminated metadata_json parsing
- **SQLite In-Memory**: Same as SQLite but uses `:memory:` for ephemeral caching

### Schema Consistency
- **All Backends**: Return identical entry structures via get_entry(), list_entries()
- **Storage Optimization**: Each backend uses optimal storage pattern for its type
- **API Consistency**: Unified schema at the API level, not storage level

### Cache Hit Patterns
- **Sequential access**: Minimal caching benefit
- **Random access**: Moderate caching benefit  
- **Hot-spot access (80/20 rule)**: Maximum caching benefit

## Benchmark History

### Schema Optimization (August 2025)
Major performance improvement through schema alignment:

1. **Unified Entry Structure**: All backends now return identical entry formats
2. **SQLite Backend**: Eliminated metadata_json column, uses dedicated columns for backend metadata
3. **JSON Backend**: Simplified to {cache_key: entry} storage, removed columnar complexity  
4. **API Consistency**: Schema alignment at entry level, not storage structure level

Performance improvements:
- JSON Backend: 35% improvement in PUT operations (4,678 vs 3,472 ops/sec at 50 entries)
- SQLite Backend: Reduced metadata parsing overhead by eliminating JSON column
- All Backends: Maintained optimal storage patterns while providing consistent APIs

### Previous Optimizations
This consolidation addressed the following issues discovered during performance analysis:

1. **Clarified Architecture**: Distinguished memory cache layer from backend types
2. **Corrected Benchmarks**: Ensured `enable_memory_cache=False` for fair raw backend comparison
3. **Identified Performance**: SQLite overhead due to database operations plus metadata processing

The comprehensive benchmark now provides accurate, reproducible performance metrics across all backend types with consistent schema alignment.