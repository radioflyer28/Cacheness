# Backend Selection Guide

This guide helps you choose between JSON and SQLite metadata backends based on your specific use case.

## Quick Decision Matrix

| Your Scenario | Recommended Backend | Reason |
|---------------|--------------------|---------| 
| Development/Testing | JSON | Faster for small data, human-readable |
| Small cache (< 200 entries) | JSON | Better performance until ~200 entries |
| Large cache (200+ entries) | SQLite | Scales much better, constant performance |
| Multiple processes | SQLite | JSON **unsafe** for concurrency |
| Production deployment | SQLite | Robust, ACID compliance, data integrity |
| Need complex queries | SQLite | Full SQL support vs basic key lookup |
| Temporary high-performance cache | SQLite (memory) | Best performance without persistence |

## Performance Characteristics

### JSON Backend

**Strengths:**
- ✅ 2-3x faster operations for small caches
- ✅ 10-30x faster metadata operations (< 200 entries)
- ✅ Human-readable metadata files
- ✅ Zero dependencies (built into Python)
- ✅ Simple setup and debugging

**Limitations:**
- ❌ **NOT SAFE** for multiple processes
- ❌ Performance degrades with cache size
- ❌ No transaction support
- ❌ No complex query capabilities
- ❌ Risk of metadata corruption

**Performance Profile:**
```
Cache Size:    50     200     500    1000
PUT ops/sec:   3472   3962    2016   1500
GET ops/sec:   4329   3497    1511    969
LIST time:     0.2ms  0.7ms   1.5ms  3.0ms
```

### SQLite Backend

**Strengths:**
- ✅ **Fully concurrent** - safe for multiple processes
- ✅ ACID transactions prevent corruption
- ✅ Scales to millions of entries
- ✅ Complex SQL queries supported
- ✅ Optimized with aggressive pragmas
- ✅ Production-ready and robust

**Limitations:**
- ⚠️ Slower for very small caches (< 100 entries)
- ⚠️ Requires SQLite library
- ⚠️ Binary metadata files (not human-readable)

**Performance Profile:**
```
Cache Size:    50     200     500    1000
PUT ops/sec:   1768   1264    640     394
GET ops/sec:   1766   1802    1466   1454
LIST time:     5.2ms  19ms    50ms   98ms
```

## Concurrency Comparison

### JSON Backend Concurrency Issues

```python
# ❌ DANGEROUS - Multiple processes accessing JSON backend
import multiprocessing
from cacheness import CacheConfig, cacheness

def worker_process(worker_id):
    config = CacheConfig(
        cache_dir="./shared_cache",
        metadata_backend="json"  # NOT SAFE!
    )
    cache = cacheness(config)
    
    # This can corrupt metadata.json if multiple workers run
    cache.put(f"data from worker {worker_id}", worker=worker_id)

# This will likely corrupt your cache metadata
with multiprocessing.Pool(4) as pool:
    pool.map(worker_process, range(4))
```

**What happens:**
- Race conditions during file reads/writes
- Corrupted JSON metadata files
- Lost cache entries
- Unpredictable failures

### SQLite Backend Concurrency Safety

```python
# ✅ SAFE - Multiple processes with SQLite backend
import multiprocessing
from cacheness import CacheConfig, cacheness

def worker_process(worker_id):
    config = CacheConfig(
        cache_dir="./shared_cache",
        metadata_backend="sqlite"  # SAFE!
    )
    cache = cacheness(config)
    
    # SQLite handles concurrent access safely
    cache.put(f"data from worker {worker_id}", worker=worker_id)

# This works perfectly with SQLite
with multiprocessing.Pool(4) as pool:
    pool.map(worker_process, range(4))
```

**SQLite safety features:**
- WAL (Write-Ahead Logging) mode for concurrency
- ACID transactions prevent corruption
- Built-in file locking
- Automatic retry on busy conditions

## Scaling Analysis

### When Performance Crossover Occurs

Based on benchmark data, here's when SQLite becomes advantageous:

| Operation Type | Crossover Point | Why |
|----------------|-----------------|-----|
| Write operations | ~500 entries | JSON file parsing overhead grows |
| Read operations | ~300 entries | SQLite indexing becomes beneficial |
| Metadata operations | ~200 entries | SQLite optimized queries vs JSON parsing |
| Overall recommendation | **200 entries** | Balance of all operations |

### Memory Usage Patterns

```python
# JSON Backend - loads entire metadata into memory
json_memory_usage = cache_entries * avg_metadata_size
# Linear growth - can become problematic

# SQLite Backend - uses efficient caching
sqlite_memory_usage = constant_overhead + (active_queries * query_cache)
# Constant base usage with controlled caching
```

## Configuration Examples

### Development Configuration

```python
# Fast setup for development and testing
from cacheness import CacheConfig

dev_config = CacheConfig(
    cache_dir="./dev_cache",
    metadata_backend="json",       # Fast for small caches
    max_cache_size_mb=100,         # Keep it small
    default_ttl_hours=1            # Short TTL for rapid iteration
)
```

### Production Configuration

```python
# Robust production setup
from cacheness import CacheConfig

prod_config = CacheConfig(
    cache_dir="/var/cache/myapp",
    metadata_backend="sqlite",     # Production-ready
    max_cache_size_mb=10000,      # Large cache
    default_ttl_hours=24,         # Longer TTL
    verify_cache_integrity=True   # Extra safety
)
```

### High-Performance Configuration

```python
# Maximum performance for temporary data
from cacheness import CacheConfig

perf_config = CacheConfig(
    cache_dir="./temp_cache",
    metadata_backend="sqlite_memory",  # In-memory for speed
    max_cache_size_mb=5000,           # Large memory cache
    default_ttl_hours=6               # Medium TTL
)
```

### Multi-Process Web Application

```python
# Safe for web apps with multiple workers
from cacheness import CacheConfig

web_config = CacheConfig(
    cache_dir="/shared/cache",
    metadata_backend="sqlite",     # Concurrency-safe
    max_cache_size_mb=2000,
    default_ttl_hours=12,
    verify_cache_integrity=True   # Important for web apps
)

# Use with Flask/Django/FastAPI workers
app = Flask(__name__)
cache = cacheness(web_config)

@app.route('/api/data')
def get_data():
    # Multiple workers can safely access this cache
    result = cache.get(request.path)
    if result is None:
        result = expensive_computation()
        cache.put(result, endpoint=request.path)
    return result
```

## Migration Between Backends

### From JSON to SQLite

```python
from cacheness import CacheConfig, cacheness
import shutil

# Step 1: Backup your existing cache
shutil.copytree("./old_cache", "./backup_cache")

# Step 2: Create new SQLite config
new_config = CacheConfig(
    cache_dir="./old_cache",
    metadata_backend="sqlite"
)

# Step 3: Initialize - this will read existing data and convert
cache = cacheness(new_config)

# Your cache entries are preserved, metadata is converted to SQLite
print(f"Migrated {len(cache.list_entries())} entries to SQLite")
```

### From SQLite to JSON (not recommended for large caches)

```python
# Only recommended for small caches or testing
new_config = CacheConfig(
    cache_dir="./sqlite_cache",
    metadata_backend="json"
)

# This will export SQLite metadata to JSON format
cache = cacheness(new_config)
```

## Troubleshooting

### JSON Backend Issues

**Problem**: Corrupted metadata.json file
```bash
# Solution: Remove metadata.json and let it rebuild
rm ./cache/metadata.json
# Cache entries preserved, metadata rebuilt from file system
```

**Problem**: Slow performance with many entries
```python
# Solution: Switch to SQLite backend
config = CacheConfig(metadata_backend="sqlite")
```

**Problem**: Multiple process errors or corruption
```python
# Solution: Use SQLite backend - JSON is not multi-process safe
config = CacheConfig(metadata_backend="sqlite")  # Thread and process-safe
```

**Problem**: Memory usage growing with cache size
```python
# JSON loads entire metadata into memory
# Solution: Use SQLite for large caches
if cache_size_entries > 200:
    config = CacheConfig(metadata_backend="sqlite")
```

### SQLite Backend Issues

**Problem**: Database locked errors
```python
# Solution: Our setup automatically enables WAL mode for concurrency
# Check for long-running transactions or file permissions
```

**Problem**: Large database file size
```python
# Solution: Regular VACUUM operations (automatic in our setup)
# Monitor with: SELECT page_size * page_count as size FROM pragma_page_count(), pragma_page_size();
```

**Problem**: Slower than expected performance
```python
# Check if you're on a slow disk - move to SSD if possible
config = CacheConfig(
    cache_dir="/fast_ssd/cache",  # SSD storage
    metadata_backend="sqlite"
)
```

### Performance Troubleshooting

**Issue**: Lower than expected throughput
```python
# 1. Check disk I/O - use SSD storage
# 2. Verify not running on network storage
# 3. Consider in-memory SQLite for temporary data
config = CacheConfig(metadata_backend="sqlite_memory")
```

**Issue**: High latency spikes
```python
# 1. JSON: Caused by large metadata file parsing
# 2. SQLite: Usually due to WAL checkpoint operations (normal)
# 3. Consider tuning checkpoint frequency if problematic
```

**Issue**: Memory consumption concerns
```python
# JSON: Loads all metadata into memory (grows with cache size)
# SQLite: Constant memory usage with configurable cache size
# Our SQLite config uses 20MB cache - adjust if needed
```

## Best Practices

### JSON Backend

1. **Use only for single-process applications**
2. **Keep cache small** (< 200 entries)
3. **Monitor metadata.json size** - switch to SQLite if it grows large
4. **Backup metadata.json regularly** for important data
5. **Use for development and testing** where human-readable metadata helps

### SQLite Backend

1. **Use for all production deployments**
2. **Enable WAL mode** (automatically done)
3. **Regular VACUUM operations** (automatically done)
4. **Monitor database file size** and plan for growth
5. **Use prepared statements** for custom queries (advanced usage)

### General

1. **Start with JSON for prototyping**, migrate to SQLite for production
2. **Benchmark your specific workload** - patterns may vary
3. **Consider cache size growth** over time
4. **Plan for concurrency needs** early in development
5. **Use appropriate TTLs** to prevent unbounded growth

### Advanced Performance Tuning

#### SQLite Backend Optimization

```python
# Our SQLite backend is pre-optimized with these settings:
# - WAL mode for concurrency (PRAGMA journal_mode=WAL)
# - 20MB cache size (PRAGMA cache_size=20000) 
# - 512MB memory-mapped I/O (PRAGMA mmap_size=536870912)
# - 32KB page size for better I/O (PRAGMA page_size=32768)
# - Query planner optimizations enabled

# For extreme performance, consider in-memory SQLite:
config = CacheConfig(
    metadata_backend="sqlite_memory",  # Fastest option
    cache_dir="./temp_cache"
)
```

#### JSON Backend Optimization

```python
# JSON backend performance tips:
# 1. Keep metadata files small (< 200 entries)
# 2. Use faster JSON library (orjson) - automatically used if available  
# 3. Minimize metadata complexity
# 4. Consider disabling parameter storage for better performance

config = CacheConfig(
    metadata_backend="json",
    store_cache_key_params=False,  # Reduces JSON file size
    verify_cache_integrity=False   # Skip hash verification for speed
)
```

#### Storage-Level Optimizations

```python
# Use fast storage for best performance
config = CacheConfig(
    cache_dir="/nvme_ssd/cache",      # NVMe SSD storage
    metadata_backend="sqlite",        # Production backend
    max_cache_size_mb=20000          # Large cache on fast storage
)

# For network storage (slower), prefer smaller frequent writes
config = CacheConfig(
    cache_dir="/network_storage/cache",
    metadata_backend="json",          # Fewer write operations
    max_cache_size_mb=1000           # Smaller cache
)
```

## Performance Testing

### What Our Benchmarks Measure

Our benchmarks provide **end-to-end latency measurements** that reflect real-world usage:

**PUT Operations (Insert Latency):**
- Data serialization and compression
- File system write operations
- Metadata backend insertion (JSON file update vs SQL INSERT)
- Transaction commit and durability guarantees

**GET Operations (Query Latency):**
- Metadata backend lookup (JSON parse vs SQL SELECT)
- File system read operations  
- Data deserialization and decompression
- Index utilization (SQLite) vs linear search (JSON)

**Metadata Operations:**
- `list_entries()`: Full metadata scan and formatting
- Backend-specific optimizations (SQL queries vs JSON parsing)

This gives you realistic performance expectations for your actual workload.

### Running Benchmarks

```bash
# Quick comparison (recommended)
python benchmarks/quick_backend_demo.py

# Comprehensive analysis (takes longer)  
python benchmarks/backend_comparison_benchmark.py

# Custom testing for your specific scenario
python -c "
from cacheness import CacheConfig, cacheness
import time

# Test your specific scenario
config = CacheConfig(metadata_backend='json')  # or 'sqlite'
cache = cacheness(config)

start = time.time()
for i in range(100):
    cache.put({'test': f'data_{i}'}, test_id=i)
print(f'PUT: {100 / (time.time() - start):.0f} ops/sec')
"
```

### Interpreting Results

**Operations per second** reflects throughput under sustained load:
- Higher values = better performance for bulk operations
- Consider both PUT and GET performance for your use case

**Metadata operation times** show scaling characteristics:
- JSON: Linear growth with cache size
- SQLite: Logarithmic growth (indexed queries)

**Crossover points** help you choose the right backend:
- Small caches: JSON advantage due to lower overhead
- Large caches: SQLite advantage due to indexing and optimization

This will help you make an informed decision based on your actual hardware and usage patterns.

## Memory Cache Layer Optimization

For both JSON and SQLite backends, you can add a memory cache layer to avoid repeated disk I/O operations:

### Memory Cache Architecture

```
Application → Memory Cache Layer → Disk Backend (JSON/SQLite)
```

### Performance Benefits

The memory cache layer provides significant speedups for workloads with repeated metadata access:

| Scenario | Without Memory Cache | With Memory Cache | Speedup |
|----------|---------------------|------------------|---------|
| Cold access | Disk I/O speed | Disk I/O speed | 1.0x |
| Warm access | Disk I/O speed | Memory speed | 1.5-3.0x |
| Hot access | Disk I/O speed | Memory speed | 2.0-5.0x |

### Configuration

```python
from cacheness import CacheConfig, CacheMetadataConfig

# Enable memory cache layer for any disk backend
config = CacheConfig(
    cache_dir="./my_cache",
    metadata_backend="sqlite",  # or "json"
    enable_memory_cache=True,
    memory_cache_type="lru",
    memory_cache_maxsize=1000,
    memory_cache_ttl_seconds=300,  # 5 minutes
    memory_cache_stats=True
)
```

### When to Use Memory Cache Layer

**Recommended for:**
- ✅ Workloads with repeated metadata access
- ✅ Large caches where disk I/O becomes bottleneck
- ✅ Applications that frequently query the same cache entries
- ✅ Production environments with predictable access patterns

**Not needed for:**
- ❌ Pure in-memory backend (already in memory)
- ❌ Applications with completely random access patterns
- ❌ Memory-constrained environments
- ❌ Very small caches (< 100 entries)

### Memory Cache vs In-Memory Backend

| Feature | Memory Cache Layer | In-Memory Backend |
|---------|-------------------|------------------|
| **Purpose** | Cache disk metadata | Store all data in memory |
| **Data Location** | Disk (cached in memory) | Memory only |
| **Persistence** | Survives restarts | Lost on restart |
| **Memory Usage** | Low (metadata only) | High (all data) |
| **Best For** | Large persistent caches | Temporary fast caches |

The memory cache layer is complementary to disk backends and provides the best of both worlds: persistence with performance.