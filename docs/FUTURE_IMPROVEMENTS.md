# Future Improvements & Feature Roadmap

## Overview

This document outlines potential improvements to cacheness, prioritized by impact and effort. It also addresses frequently requested features and explains architectural decisions (particularly regarding REST API).

**Last Updated:** February 5, 2026

---

## Quick Navigation

- [High Priority Improvements](#high-priority---core-functionality-gaps) - Start here for next features
- [Medium Priority Improvements](#medium-priority---developer-experience) - UX enhancements
- [Lower Priority Features](#lower-priority---advanced-features) - Nice-to-haves
- [REST API Analysis](#the-rest-api-question) - Why we're not building it
- [Recommended Roadmap](#recommended-implementation-roadmap) - What to build next

---

## High Priority - Core Functionality Gaps

### 1. **Complete Missing Management Operations** ✅ Highest Priority

**Status:** Analyzed in [MISSING_MANAGEMENT_API.md](MISSING_MANAGEMENT_API.md)

**Missing Operations:**
- `update_blob_data(cache_key, new_data)` - Update data at existing key without changing key
- `delete_by_prefix(**kwargs)` - Bulk delete matching entries
- `touch(**kwargs, ttl_seconds)` - Refresh TTL without reloading data
- `get_metadata(**kwargs)` - Expose backend metadata access in cache layer
- Batch operations: `get_batch()`, `delete_batch()`, `update_batch()`
- Copy/move operations: `copy(source, dest)`, `move(source, dest)` (convenience wrappers)

**Why High Priority:**
- Frequently requested by users
- Common operations in real-world usage
- Well-defined, fits existing architecture cleanly

**Impact:** High - These are essential for production use cases

**Effort:** Medium - APIs designed, just need implementation

**Implementation Plan:**
1. Storage backend layer: Add `update_blob_data()`, `delete_by_prefix()`, batch operations
2. Cache layer: Add `touch()`, expose `get_metadata()`, add convenience wrappers
3. Test across all backends (SQLite, PostgreSQL, JSON, Memory)
4. Document in API reference

---

### 2. **Async/Await Support** ✅ High Priority

**Current State:** All operations are synchronous

**Proposed API:**

```python
import asyncio
from cacheness import AsyncUnifiedCache

async def main():
    # Async cache operations
    cache = AsyncUnifiedCache(
        metadata_backend=PostgreSQLBackend("postgresql://..."),
        blob_store=S3BlobStore("s3://...")
    )
    
    # All operations are async
    await cache.put(data, experiment="exp_001")
    result = await cache.get(experiment="exp_001")
    
    # Batch operations benefit most
    results = await cache.get_batch([
        {"experiment": "exp_001"},
        {"experiment": "exp_002"},
        {"experiment": "exp_003"}
    ])
    
    # Async context manager
    async with cache:
        data = await cache.get(experiment="exp_001")

if __name__ == "__main__":
    asyncio.run(main())
```

**Why High Priority:**
- Large blob I/O is naturally async (network, disk)
- Database queries can be async (asyncpg for PostgreSQL)
- Modern Python standard (Python 3.7+)
- Enables web service integration (FastAPI, aiohttp)
- Better concurrency without threads

**Benefits:**

```python
# Synchronous (current)
data1 = cache.get(key1)  # Wait
data2 = cache.get(key2)  # Wait
data3 = cache.get(key3)  # Wait
# Total: 300ms if each takes 100ms

# Asynchronous (proposed)
data1, data2, data3 = await asyncio.gather(
    cache.get(key1),
    cache.get(key2),
    cache.get(key3)
)
# Total: ~100ms (concurrent)
```

**Implementation Approach:**

1. **Separate `AsyncUnifiedCache` class** (don't complicate sync API)
2. **Async backends:**
   - `AsyncPostgreSQLBackend` using asyncpg
   - `AsyncS3BlobStore` using aioboto3
   - SQLite: aiosqlite (though less beneficial - file-based)
3. **Shared serialization logic** (handlers remain sync, run in executor)
4. **Coexist with sync API** (users choose based on needs)

**Trade-offs:**
- ✅ Better concurrency and throughput
- ✅ Modern Python standard
- ❌ Increases API surface (now have sync + async)
- ❌ More complex testing (need async test infrastructure)

**Impact:** High - Enables modern async Python workflows

**Effort:** High - Requires async versions of backends, thorough testing

---

### 3. **Advanced Eviction Policies** ⚠️ Medium-High Priority

**Current State:** TTL-based eviction only

**Proposed:**

```python
from cacheness import UnifiedCache
from cacheness.eviction import LRUPolicy, LFUPolicy, SizeBasedPolicy

# Least Recently Used (LRU)
cache = UnifiedCache(
    eviction_policy=LRUPolicy(max_entries=10000),
    # Automatically evicts least recently accessed
)

# Least Frequently Used (LFU)
cache = UnifiedCache(
    eviction_policy=LFUPolicy(max_entries=10000),
    # Evicts least frequently accessed
)

# Size-Based Eviction
cache = UnifiedCache(
    eviction_policy=SizeBasedPolicy(max_size_gb=100),
    # Evicts when total cache size exceeds limit
)

# Composite Policy (combine multiple)
cache = UnifiedCache(
    eviction_policy=CompositePolicy([
        TTLPolicy(ttl_seconds=days(7)),      # Expire after 7 days
        SizeBasedPolicy(max_size_gb=100),    # Or if cache > 100GB
        LRUPolicy(max_entries=50000)         # Or if > 50k entries
    ])
)
```

**Why Needed:**
- TTL alone isn't always appropriate
- Some data rarely accessed but shouldn't expire
- Some data hot but takes up too much space
- Need to bound cache size on disk

**Use Cases:**

```python
# ML model cache - size matters more than time
cache = UnifiedCache(
    eviction_policy=SizeBasedPolicy(max_size_gb=500),
    # Keep last 500GB of models regardless of age
)

# API response cache - freshness matters
cache = UnifiedCache(
    eviction_policy=TTLPolicy(ttl_seconds=minutes(15))
    # Expire responses after 15 minutes
)

# Hot data cache - access patterns matter
cache = UnifiedCache(
    eviction_policy=LRUPolicy(max_entries=10000)
    # Keep 10k most recently used entries
)
```

**Implementation Considerations:**
- Needs access tracking (last_accessed timestamp)
- Needs size tracking (already have file_size in metadata)
- Needs frequency tracking for LFU (new metadata field)
- Background eviction task (don't block operations)

**Impact:** Medium-High - Enables new use cases, better resource management

**Effort:** Medium-High - Requires metadata schema changes, background tasks

---

## Medium Priority - Developer Experience

### 4. **CLI Tool for Cache Inspection** ✅ Recommended

**Current State:** All operations require Python code

**Proposed CLI:**

```bash
# Inspect cache contents
cacheness inspect ./cache --backend sqlite
# Output:
# Cache: ./cache
# Backend: SQLite (cache.db)
# Entries: 1,234
# Total Size: 45.6 GB
# Oldest Entry: 2026-01-15 14:23:01
# Newest Entry: 2026-02-05 09:45:12

# List entries with details
cacheness list ./cache --format table --limit 10
# Output:
# | cache_key           | size      | created_at          | data_type |
# |---------------------|-----------|---------------------|-----------|
# | exp_001_7a8b9c     | 1.2 GB    | 2026-02-05 09:00:00 | DataFrame |
# | model_v2_3f4e5d    | 3.4 GB    | 2026-02-04 15:30:00 | ndarray   |

# Query metadata with SQL
cacheness query "./cache" \
  --sql "SELECT cache_key, file_size_bytes, created_at FROM metadata WHERE file_size_bytes > 1000000000" \
  --format json

# Clean expired entries
cacheness cleanup ./cache --ttl 86400
# Output:
# Removed 45 expired entries
# Freed 12.3 GB

# Statistics
cacheness stats ./cache
# Output:
# Total Entries: 1,234
# Total Size: 45.6 GB
# By Type:
#   DataFrame: 456 entries (23.4 GB)
#   ndarray: 321 entries (18.9 GB)
#   Tensor: 89 entries (2.1 GB)
#   pickle: 368 entries (1.2 GB)

# Migrate between backends
cacheness migrate \
  --from sqlite://./cache/cache.db \
  --to postgresql://prod-db/cache \
  --dry-run

# Verify cache integrity
cacheness verify ./cache
# Output:
# Checking 1,234 entries...
# ✓ All metadata entries have corresponding blobs
# ✓ All blob signatures valid
# ! Warning: 3 orphaned blobs found (no metadata)
```

**Why Needed:**
- Debugging cache issues
- Monitoring cache size and growth
- Maintenance operations (cleanup, migration)
- Quick inspection without writing Python code

**Impact:** High - Significantly improves developer experience

**Effort:** Low-Medium - Wrapper around existing APIs

**Implementation:**
- Use Click or Typer for CLI framework
- Leverage existing cache APIs
- Add output formatting (table, JSON, CSV)
- Package as `cacheness` command

---

### 5. **Connection Pooling for Database Backends** ⚠️ Medium Priority

**Current State:** One connection per operation (inefficient for high concurrency)

**Proposed:**

```python
from cacheness import UnifiedCache
from cacheness.backends import PostgreSQLBackend

# Connection pool configuration
backend = PostgreSQLBackend(
    "postgresql://user:pass@host/db",
    pool_size=10,           # Normal pool size
    max_overflow=20,        # Additional connections under load
    pool_timeout=30,        # Wait up to 30s for connection
    pool_recycle=3600       # Recycle connections after 1 hour
)

cache = UnifiedCache(metadata_backend=backend)

# Multiple concurrent operations share pool
# No connection exhaustion
```

**Why Needed:**
- High-concurrency workloads (web servers, APIs)
- Avoid connection exhaustion
- Reduce connection overhead
- Better resource utilization

**Current Workaround:**
```python
# Users must manage connections manually
from sqlalchemy import create_engine, pool

engine = create_engine(
    "postgresql://...",
    poolclass=pool.QueuePool,
    pool_size=10
)
backend = PostgreSQLBackend(engine=engine)
```

**Implementation:**
- Use SQLAlchemy pooling (already dependency)
- Default to sensible pool sizes
- Make configurable
- Document for high-concurrency scenarios

**Impact:** Medium - Important for web service use cases

**Effort:** Low - SQLAlchemy provides this, just needs API exposure

---

### 6. **Better Type Hints & Generic Support** ⚠️ Low-Medium Priority

**Current State:** Type hints exist but not generic

**Proposed:**

```python
from typing import TypeVar, Generic, Optional
from cacheness import UnifiedCache
import pandas as pd

T = TypeVar('T')

class UnifiedCache(Generic[T]):
    """Generic cache with type hints."""
    
    def get(self, cache_key: Optional[str] = None, **kwargs) -> Optional[T]:
        """Get cached data with type preserved."""
        ...
    
    def put(self, data: T, cache_key: Optional[str] = None, **kwargs) -> None:
        """Put data with type checking."""
        ...

# Usage with type checking
cache: UnifiedCache[pd.DataFrame] = UnifiedCache()

df = cache.get(experiment="exp_001")
# IDE knows: df is Optional[pd.DataFrame]

cache.put(df, experiment="exp_002")
# Type checker validates df is DataFrame

# Type error caught at design time!
cache.put("not a dataframe", experiment="exp_003")
# mypy error: Expected DataFrame, got str
```

**Why Useful:**
- Better IDE autocomplete and hints
- Catch type errors at design time (mypy, pyright)
- Self-documenting code
- Improved maintainability

**Challenges:**
- Cache stores heterogeneous types (DataFrames, arrays, objects)
- Generic type may be too restrictive
- Could use `Union[DataFrame, ndarray, ...]` but gets verbose

**Possible Approach:**

```python
# Option 1: Generic but flexible
cache = UnifiedCache()  # Type: UnifiedCache[Any]
df: pd.DataFrame = cache.get(...)  # User provides type hint

# Option 2: Type-specific caches
df_cache: UnifiedCache[pd.DataFrame] = UnifiedCache()
array_cache: UnifiedCache[np.ndarray] = UnifiedCache()

# Option 3: Return type based on handler
@overload
def get(self, **kwargs) -> pd.DataFrame: ...  # If handler is PandasHandler
@overload
def get(self, **kwargs) -> np.ndarray: ...    # If handler is NumpyHandler
```

**Impact:** Low-Medium - Nice for users with mypy/pyright

**Effort:** Medium - Requires careful design to not break flexibility

---

## Lower Priority - Advanced Features

### 7. **Content Deduplication** ⚠️ Low Priority

**Concept:** If two cache entries have identical content, store blob once

**Proposed:**

```python
cache = UnifiedCache(
    deduplication=True,
    dedup_method="content_hash"  # or "content_compare"
)

# Stores blob once, metadata twice
cache.put(large_data, experiment="exp_001")  # Stores blob
cache.put(large_data, experiment="exp_002")  # References same blob
cache.put(large_data, experiment="exp_003")  # References same blob

# Saves: 2 * size(large_data) on disk
```

**Implementation:**
- Compute content hash (SHA256 of serialized data)
- Store blobs by content hash
- Metadata references content hash
- Reference counting for deletion

**Trade-offs:**
- ✅ Saves disk space when same data cached multiple times
- ✅ Faster put (no write if already exists)
- ❌ Slower put (must compute hash)
- ❌ More complex deletion (reference counting)
- ❌ Metadata can't be deleted until all references gone

**When Useful:**
- Caching results of pure functions with multiple keys
- ML experiments with same base model
- Repeated API responses

**When Not Useful:**
- All cached data is unique
- Data changes frequently
- Hash computation overhead > storage savings

**Impact:** Low - Niche benefit, most caches have unique data

**Effort:** High - Significant complexity for reference counting, cleanup

**Recommendation:** Don't implement unless users specifically request

---

### 8. **Incremental Updates / Delta Compression** ⚠️ Low Priority

**Concept:** Store only changes from previous version

**Proposed:**

```python
# Store initial version
cache.put(large_df_v1, key="data_v1")

# Store delta from v1 (much smaller)
cache.put(
    large_df_v2, 
    key="data_v2",
    delta_from="data_v1"  # Store only difference
)

# Retrieval automatically reconstructs
df_v2 = cache.get(key="data_v2")
# Behind scenes: load data_v1, apply delta → data_v2
```

**Use Cases:**
- Versioned datasets with small changes
- Time-series data (daily snapshots)
- ML model checkpoints

**Trade-offs:**
- ✅ Saves disk space for versioned data
- ❌ Slower reads (must apply delta)
- ❌ Complicated deletion (can't delete v1 if v2 depends on it)
- ❌ Complex implementation (delta computation, dependency tracking)

**Alternatives:**
- Use content deduplication (simpler)
- Store compressed full versions (simpler)
- Use version control system (DVC, Git LFS)

**Impact:** Low - Very specific use case

**Effort:** Very High - Complex dependency management

**Recommendation:** Don't implement - use external version control tools instead

---

### 9. **Tiered Storage (Hot/Cold)** ⚠️ Low Priority

**Concept:** Automatically move old/cold data to cheaper storage

**Proposed:**

```python
cache = UnifiedCache(
    hot_storage=MemoryBlobStore(),           # Fast, expensive
    warm_storage=FilesystemBlobStore(...),   # Medium
    cold_storage=S3BlobStore(...),           # Slow, cheap
    
    # Aging policy
    hot_to_warm_days=1,   # Memory → Disk after 1 day
    warm_to_cold_days=7   # Disk → S3 after 7 days
)

# Recent data in memory (fast)
cache.put(data, key="recent")
result = cache.get(key="recent")  # ~1ms (memory)

# Older data on disk (medium)
# ... 2 days later ...
result = cache.get(key="recent")  # ~10ms (disk)

# Old data in S3 (slow but cheap)
# ... 10 days later ...
result = cache.get(key="recent")  # ~100ms (S3)
```

**Use Cases:**
- Large caches with access skew (hot/cold data)
- Cost optimization (memory expensive, S3 cheap)
- Performance tiers for different data ages

**Trade-offs:**
- ✅ Optimizes cost vs performance
- ✅ Keeps hot data fast
- ❌ Very complex implementation (background migration, tier tracking)
- ❌ Unpredictable latency (don't know which tier)
- ❌ Hard to reason about

**Alternatives:**
- Use separate caches for hot/cold data
- Let users manage tiers explicitly
- Use cloud provider features (S3 lifecycle policies)

**Impact:** Low - Very niche, enterprise-scale problem

**Effort:** Very High - Complex background jobs, state management

**Recommendation:** Don't implement - too complex for benefit

---

## The REST API Question

### Should Cacheness Provide a REST API?

**Short Answer: No** ❌

### Arguments AGAINST REST API (Recommended Position)

#### 1. **Contradicts Core Philosophy**

Cacheness's primary value proposition is **simplicity** and **zero infrastructure**:

```python
# Current (simple)
pip install cacheness
cache = UnifiedCache()  # Works immediately

# With REST API (complex)
pip install cacheness
cacheness server start --config server.yaml  # Start server
# Configure authentication, networking, reverse proxy, TLS...
# Deploy with Docker/Kubernetes
# Monitor with Prometheus
# Scale with load balancer
```

**This contradicts the "pip install and go" philosophy that makes cacheness attractive.**

#### 2. **Network Overhead Destroys Performance**

```python
# Direct Python (current): ~10ms for large DataFrame
df = cache.get(experiment="exp_001")

# Over REST API: ~100-500ms
response = requests.get("http://cache-api/cache/exp_001")
df = pd.read_json(response.content)
```

**Multi-gigabyte blobs over HTTP are inherently slow.** The whole point of cacheness is fast local access to large data. Adding network layer defeats this.

#### 3. **Serialization Double-Penalty**

One of cacheness's key innovations is the handler system with optimized formats:

```python
# Direct (optimal): DataFrame → Parquet + LZ4
cache.put(df, key="data")
df = cache.get(key="data")
# Stored: data.parquet (columnar, compressed, fast)

# REST API (terrible):
# Write: DataFrame → JSON → HTTP → Server → Parquet
# Read: Parquet → Server → JSON → HTTP → DataFrame
# Loses ALL handler optimization benefits!
```

**REST API forces JSON serialization, losing all type-aware storage optimizations.**

#### 4. **Primary Use Cases Don't Need Network Access**

**Who uses cacheness:**
- Data scientists in Jupyter notebooks (local)
- ML training scripts on single machine (local)
- Data pipelines in Python (local or same cluster)
- Research computing (local or HPC node)

**None of these need REST API - they run where the cache is.**

#### 5. **Multi-Language? Use Shared Backend Instead**

If you need multi-language access, don't use REST - use shared storage:

```python
# Python service
py_cache = UnifiedCache(
    metadata_backend=PostgreSQLBackend("postgresql://shared-db"),
    blob_store=S3BlobStore("s3://shared-bucket")
)

# Node.js service (thin client)
const pg = require('pg')
const AWS = require('aws-sdk')

// Query metadata directly
const metadata = await pg.query(
    "SELECT blob_path, data_type FROM metadata WHERE cache_key = $1",
    [cacheKey]
)

// Fetch blob directly from S3
const s3 = new AWS.S3()
const blob = await s3.getObject({
    Bucket: "shared-bucket",
    Key: metadata.blob_path
}).promise()
```

**Better than REST API:**
- No centralized bottleneck
- No serialization overhead
- Scales horizontally (all services access storage directly)
- No server to manage

---

### Arguments FOR REST API (Devil's Advocate)

#### 1. **Remote Caching for Distributed Teams**

```python
# Developer on laptop (slow CPU)
cache = RemoteCacheClient("https://team-cache-server.com")

# Uses powerful server's cached computation
result = cache.get(expensive_computation_key)
# Don't recompute on weak laptop
```

**Counter-argument:** Just use shared PostgreSQL + S3 backend:

```python
# All team members access same cache
cache = UnifiedCache(
    metadata_backend=PostgreSQLBackend("postgresql://team-db"),
    blob_store=S3BlobStore("s3://team-cache")
)
# No REST server needed!
```

#### 2. **Language-Agnostic Access**

```javascript
// JavaScript service wants to use cache
const response = await fetch("http://cache-api/cache/exp_001")
const data = await response.json()
```

**Counter-argument:** Implement thin clients that access storage directly:

```javascript
// Thin JavaScript client
const CachenessClient = require('cacheness-js-client')

const client = new CachenessClient({
    postgres: "postgresql://shared-db",
    s3Bucket: "shared-cache"
})

// Queries PostgreSQL + S3 directly (no REST server)
const data = await client.get({experiment: "exp_001"})
```

**More efficient, no centralized bottleneck.**

#### 3. **Cache-as-a-Service Business Model**

```python
# Offer managed caching to customers
# They hit your API, you manage infrastructure
```

**Counter-argument:** This is a **different product**. Cacheness is a library, not a SaaS platform. If you want to build a service, use cacheness as a component, but that's not the core project.

---

### Alternative: Thin Client Pattern

If multi-language access is required, provide **thin client libraries** instead of REST API:

#### Python (Full Cacheness)

```python
from cacheness import UnifiedCache
from cacheness.backends import PostgreSQLBackend
from cacheness.blob_stores import S3BlobStore

cache = UnifiedCache(
    metadata_backend=PostgreSQLBackend("postgresql://shared"),
    blob_store=S3BlobStore("s3://shared")
)
```

#### JavaScript (Thin Client)

```javascript
// cacheness-js-client package
const CachenessClient = require('cacheness-js-client')

const cache = new CachenessClient({
    metadataDb: "postgresql://shared",
    blobStore: "s3://shared"
})

// Client queries metadata from PostgreSQL
const metadata = await cache.getMetadata({experiment: "exp_001"})

// Client fetches blob from S3
const blob = await cache.getBlob(metadata.blob_path)

// Client deserializes using metadata.data_type
const data = deserialize(blob, metadata.data_type)
```

#### Ruby (Thin Client)

```ruby
# cacheness-ruby-client gem
require 'cacheness/client'

cache = Cacheness::Client.new(
  metadata_db: "postgresql://shared",
  blob_store: "s3://shared"
)

data = cache.get(experiment: "exp_001")
```

**Benefits:**
- ✅ No centralized server (no bottleneck, no SPOF)
- ✅ No REST overhead (direct storage access)
- ✅ Scales horizontally (N clients, one storage)
- ✅ Maintains format optimization (Parquet, not JSON)
- ✅ Language-agnostic
- ✅ Simpler deployment

**Challenges:**
- Each language needs client implementation
- Clients need to understand metadata schema
- Serialization formats need language support (Parquet, etc.)

**Verdict:** Still better than REST API if multi-language is critical.

---

## Recommended Implementation Roadmap

### Phase 1: Essential Management Operations (3-6 months)

**Goal:** Complete missing CRUD operations

1. **Storage Backend Layer**
   - [ ] `update_blob_data(cache_key, new_data)` - Replace data at key
   - [ ] `delete_by_prefix(prefix)` - Bulk delete
   - [ ] `delete_where(filter_fn)` - Conditional bulk delete
   - [ ] `get_entries_batch(cache_keys)` - Batch get metadata
   - [ ] `delete_entries_batch(cache_keys)` - Batch delete
   - [ ] `copy_entry(source, dest)` / `move_entry(source, dest)` - Convenience wrappers

2. **Cache Layer Wrappers**
   - [ ] `cache.update_data(data, **kwargs)` - Update wrapper
   - [ ] `cache.touch(**kwargs, ttl_seconds)` - Refresh TTL
   - [ ] `cache.get_metadata(**kwargs)` - Expose metadata access
   - [ ] `cache.delete_by_prefix(**kwargs)` - Convenience wrapper
   - [ ] `cache.get_batch([kwargs_list])` - Batch get wrapper
   - [ ] `cache.copy(source, dest)` / `cache.move(source, dest)` - Convenience wrappers

3. **Testing & Documentation**
   - [ ] Test all operations across all backends
   - [ ] Update API reference
   - [ ] Add usage examples

**Priority:** ✅✅✅ Highest - Most requested, well-defined

---

### Phase 2: Async Support (6-9 months)

**Goal:** Enable modern async Python workflows

1. **Async Backends**
   - [ ] `AsyncPostgreSQLBackend` (using asyncpg)
   - [ ] `AsyncS3BlobStore` (using aioboto3)
   - [ ] `AsyncAzureBlobStore` (using aioazure)
   - [ ] `AsyncSQLiteBackend` (using aiosqlite) - optional

2. **Async Cache Layer**
   - [ ] `AsyncUnifiedCache` class
   - [ ] All operations async (`await cache.get()`, etc.)
   - [ ] Async context manager (`async with cache:`)
   - [ ] Async batch operations (concurrent by default)

3. **Handler Integration**
   - [ ] Run sync handlers in thread executor
   - [ ] Don't block async event loop

4. **Documentation & Examples**
   - [ ] Async API reference
   - [ ] FastAPI integration example
   - [ ] Performance benchmarks (sync vs async)

**Priority:** ✅✅ High - Modern Python standard, enables web services

---

### Phase 3: CLI Tool (2-3 months)

**Goal:** Improve developer experience for debugging/maintenance

1. **Core Commands**
   - [ ] `cacheness inspect <path>` - Show cache overview
   - [ ] `cacheness list <path>` - List entries with details
   - [ ] `cacheness query <path> --sql <query>` - SQL queries
   - [ ] `cacheness stats <path>` - Statistics by type, size, age
   - [ ] `cacheness cleanup <path> --ttl <seconds>` - Remove expired
   - [ ] `cacheness verify <path>` - Check integrity

2. **Advanced Commands**
   - [ ] `cacheness migrate --from <source> --to <dest>` - Migrate backends
   - [ ] `cacheness export <path> --output <file>` - Export cache
   - [ ] `cacheness import <file> --into <path>` - Import cache

3. **Output Formatting**
   - [ ] Table format (default)
   - [ ] JSON format (`--format json`)
   - [ ] CSV format (`--format csv`)

**Priority:** ✅ Medium-High - High impact on DX, relatively easy

---

### Phase 4: Eviction Policies (Optional - 6-12 months)

**Goal:** Better resource management

1. **Policy Implementations**
   - [ ] `LRUPolicy` - Least Recently Used
   - [ ] `LFUPolicy` - Least Frequently Used
   - [ ] `SizeBasedPolicy` - Total cache size limit
   - [ ] `CompositePolicy` - Combine multiple policies

2. **Metadata Schema Updates**
   - [ ] Add `last_accessed` timestamp
   - [ ] Add `access_count` for LFU
   - [ ] Migration for existing caches

3. **Background Eviction**
   - [ ] Background task for policy enforcement
   - [ ] Don't block cache operations
   - [ ] Configurable check frequency

**Priority:** ⚠️ Medium - Nice-to-have, enables new use cases

---

### Not Recommended (Don't Build)

- ❌ **REST API** - Contradicts philosophy, adds overhead, alternatives better
- ❌ **Content Deduplication** - Complex, niche benefit
- ❌ **Delta Compression** - Very complex, use external version control
- ❌ **Tiered Storage** - Extremely complex, narrow use case
- ❌ **Built-in Distributed Lock** - Use external lock manager (Redis, etcd)
- ❌ **Built-in Metrics/Monitoring** - Use standard observability tools

---

## Summary: What to Build Next

### Top 3 Priorities

1. **✅ Complete Management Operations** (3-6 months)
   - Missing CRUD operations, batch operations, convenience wrappers
   - Highest user impact, well-defined scope
   - **Start here**

2. **✅ Async/Await Support** (6-9 months)
   - Modern Python standard, enables web services
   - High impact, aligns with ecosystem trends
   - **Do this second**

3. **✅ CLI Tool** (2-3 months)
   - Debugging, maintenance, migration
   - High DX impact, relatively easy
   - **Do this third**

### Consider Later

- ⚠️ **LRU/LFU Eviction** - Useful but not essential
- ⚠️ **Connection Pooling** - Easy win for high concurrency
- ⚠️ **Better Type Hints** - Nice DX improvement

### Don't Build

- ❌ **REST API** - Wrong abstraction, contradicts philosophy
- ❌ **Deduplication** - Too complex for benefit
- ❌ **Tiered Storage** - Too complex for benefit

---

## Conclusion

**Keep cacheness focused on what it does best:**
- Fast, local, Python-native caching
- Type-aware storage with optimal formats
- Pluggable architecture for flexibility
- Zero infrastructure to enterprise scale

**Don't try to be everything:**
- Not a distributed cache (use Redis for that)
- Not a REST service (use shared storage instead)
- Not a version control system (use DVC/Git LFS for that)
- Not a monitoring platform (use standard observability tools)

**The roadmap prioritizes:**
1. Completing core functionality (management operations)
2. Modernizing for async Python (async/await)
3. Improving developer experience (CLI tool)

**This keeps cacheness true to its mission while addressing real user needs.**

---

## References

- [Missing Management API Analysis](MISSING_MANAGEMENT_API.md)
- [Comparison to Existing Solutions](COMPARISON_TO_EXISTING_SOLUTIONS.md)
- [API Reference](API_REFERENCE.md)
