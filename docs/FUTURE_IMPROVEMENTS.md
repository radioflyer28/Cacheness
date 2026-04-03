# Future Improvements & Feature Roadmap

## Overview

This document outlines potential improvements to cacheness, prioritized by impact and effort. It also addresses frequently requested features and explains architectural decisions (particularly regarding REST API).

**Last Updated:** April 3, 2026

---

## Quick Navigation

- [High Priority Improvements](#high-priority---core-functionality-gaps) - Start here for next features
- [Medium Priority Improvements](#medium-priority---developer-experience) - UX enhancements
- [Lower Priority Features](#lower-priority---advanced-features) - Nice-to-haves
- [REST API Analysis](#the-rest-api-question) - Why we're not building it
- [Recommended Roadmap](#recommended-implementation-roadmap) - What to build next

---

## High Priority - Core Functionality Gaps

### 1. **Complete Missing Management Operations** ✅ Partially Shipped

**Status:** Analyzed in [MISSING_MANAGEMENT_API.md](MISSING_MANAGEMENT_API.md)

**Shipped (v0.8.0–v0.9.0):**
- ✅ `delete_by_prefix(**kwargs)` — bulk delete matching entries (v0.8.0, Phase 10)
- ✅ `put_batch(items)` — batch put with partial-success semantics (v0.9.0, Phase 11)

**Remaining Operations:**
- `update_blob_data(cache_key, new_data)` - Update data at existing key without changing key
- `touch(**kwargs, ttl_seconds)` - Refresh TTL without reloading data
- `get_metadata(**kwargs)` - Expose backend metadata access in cache layer
- Batch operations: `get_batch()`, `delete_batch()`
- Copy/move operations: `copy(source, dest)`, `move(source, dest)` (convenience wrappers)

**Impact:** High - These are essential for production use cases

**Effort:** Medium - APIs designed, just need implementation

**Implementation Plan:**
1. Storage backend layer: Add `update_blob_data()`, remaining batch operations
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
from cacheness.core import UnifiedCache
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

**Current State:** SQLAlchemy's `QueuePool` is already used by the PostgreSQL backend. SQLite uses a single-connection model with `RLock` for thread safety (appropriate for file-based databases).

**What's needed:**
- Expose pool configuration parameters (`pool_size`, `max_overflow`, `pool_timeout`) in `CacheConfig`
- Document pool tuning for high-concurrency PostgreSQL deployments
- Consider connection pool metrics for observability

**Current Workaround:**
```python
# Users can pass a custom engine with pool settings
from sqlalchemy import create_engine, pool

engine = create_engine(
    "postgresql://...",
    poolclass=pool.QueuePool,
    pool_size=10
)
backend = PostgreSQLBackend(engine=engine)
```

**Impact:** Medium - Important for web service use cases

**Effort:** Low - SQLAlchemy provides this, just needs API exposure

---

### 6. **Better Type Hints & Generic Support** ⚠️ Low-Medium Priority

**Current State:** Type hints exist but not generic

**Proposed:**

```python
from typing import TypeVar, Generic, Optional
from cacheness.core import UnifiedCache
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

### 9. **Tiered Pull-Through Cache** ✅ Medium Priority

**Concept:** Compose two `UnifiedCache` instances — a fast local cache backed by a remote shared store — with automatic pull-through on miss.

**Proposed:**

```python
from cacheness.tiered import TieredCache
from cacheness.core import UnifiedCache

# Local tier: SQLite + filesystem (microsecond reads)
local = UnifiedCache(cache_dir="./local_cache", metadata_backend="sqlite")

# Remote tier: PostgreSQL/libSQL + S3 (shared, durable)
remote = UnifiedCache(
    metadata_backend=PostgreSQLBackend("postgresql://team-db"),
    blob_store=S3BlobStore("s3://team-cache")
)

cache = TieredCache(local=local, remote=remote, max_local_size_gb=10)

# First access: miss locally → fetch from remote → cache locally
result = cache.get(key="experiment_001")  # ~100ms (S3 fetch)

# Second access: hit local cache
result = cache.get(key="experiment_001")  # ~1ms (local disk)

# Writes go to remote first (source of truth), then local
cache.put(data, key="experiment_002")  # Remote + local
```

**Why This Design:**
- **Composes existing primitives** — no new backends needed, just orchestration (~200 LOC)
- **Caches blobs locally** — unlike libSQL embedded replicas which only sync metadata, this caches actual data files on local disk
- **Independent policies per tier** — local: 10GB cap, LRU, no signing. Remote: unlimited, signed, encrypted
- **Works with any backend combination** — SQLite+filesystem → PostgreSQL+S3, JSON+filesystem → libSQL+S3, etc.

**Invalidation strategies:**
1. **TTL-based** — local entries expire after N seconds. Simple, slightly stale. Fine for most caching use cases.
2. **Metadata-version check** — on local hit, compare local entry's timestamp/hash against remote metadata. If stale, re-fetch blob. Especially cheap if remote metadata uses libSQL embedded replicas (local microsecond read for staleness check).
3. **No invalidation** — for single-user or read-heavy workloads, staleness isn't a problem.

**Storage mode value:** For storage mode users, the tiered cache becomes a **local workspace pattern** — work against local Cacheness, persist to a remote store. Think of it like git's local/remote model. The `migrate()` API (see section 10 below) would work at each tier independently.

**Impact:** High - Fills the gap between local-only and fully-remote caching

**Effort:** Low-Medium - Thin orchestration layer over existing `UnifiedCache` instances

---

### 10. **Version-in-Metadata for Migration Support** ✅ Medium-High Priority

**Concept:** Store the Cacheness version (or serialization format version) per entry in metadata, enabling targeted migration when Cacheness is upgraded.

**Why Not Version-in-Cache-Key:**
Version-in-cache-key forces full invalidation on every upgrade — every cached entry becomes a miss even if the serialization format didn't change. Version-in-metadata enables per-entry migration decisions.

**Why It Matters:**
- **Decorator mode:** A format-incompatible entry causes a cache miss and re-execution — acceptable but wasteful when migration could preserve it.
- **Storage mode (critical):** There is no function to re-execute. If an upgrade changes serialization and old entries can't be read, that's **data loss**, not a performance hit.

**Proposed:**

```python
# On put(): store version with every entry
entry.cacheness_version = "0.10.0"
entry.serialization_format_version = 2  # bumps only when format changes

# On get(): check compatibility
if entry.serialization_format_version < CURRENT_FORMAT_VERSION:
    data = deserialize_with_compat(entry)  # compat path
    # Optionally re-serialize in new format (lazy migration)

# Bulk migration API
cache.migrate()  # re-serializes all stale entries
cache.migrate(dry_run=True)  # report what would change
```

**Design considerations:**
- **Version granularity:** `cacheness.__version__` vs a separate `serialization_format_version` that only bumps when format actually changes (reduces unnecessary migrations)
- **Handler-level versioning:** Each handler (parquet for DataFrames, blosc2 for NumPy, pickle for objects) may evolve independently — consider per-handler format versions
- **Backward compatibility window:** Define how many prior format versions must be readable without explicit migration
- **Lazy migration:** Entries re-written in new format on first access rather than requiring big-bang migration

**Impact:** High - Essential for storage mode users, valuable for all users

**Effort:** Medium - Schema migration for all backends + compat read paths

---

### 11. **Encryption at Rest** ✅ Medium Priority

**Concept:** Protect cached data confidentiality — not just integrity (which signing already provides). Client-side encryption before data reaches storage, so untrusted or compromisable servers never see plaintext.

**Primary threat model:** Cached data stored on remote servers (S3 blobs, PostgreSQL metadata, libSQL with cloud sync) that could be compromised. The encryption boundary must be the client — data is encrypted before it leaves the local process, and the server only ever stores ciphertext. This is especially important for the [tiered cache pattern](#9-tiered-pull-through-cache--medium-priority) where blobs are synced to shared S3 buckets or metadata is replicated via libSQL embedded replicas.

**Current state:** Cacheness provides **integrity** via HMAC signing (v2/v3 signatures, HKDF-derived per-namespace keys, key rotation). Blobs are stored as plaintext files on disk. Anyone with filesystem or server access can read cached data.

#### Metadata Encryption (via libSQL)

libSQL's built-in `encryption_key` parameter encrypts the SQLite database at rest:

```python
# libSQL backend with metadata encryption
cache = UnifiedCache(
    metadata_backend=LibsqlBackend(
        db_file="cache_metadata.db",
        encryption_key=os.environ["CACHE_ENC_KEY"],
    )
)
# Metadata DB is AES-encrypted on disk — cache keys, timestamps, paths all protected
```

This covers metadata confidentiality (cache keys, data types, timestamps, custom metadata) but **not blob files**. See [LIBSQL_BACKEND.md](LIBSQL_BACKEND.md) for full libSQL research.

#### Blob Encryption

For full confidentiality, blobs need encryption in the write path:

```python
# Proposed: blob encryption config
config = CacheConfig(
    security=SecurityConfig(
        enable_entry_signing=True,       # integrity (existing)
        enable_blob_encryption=True,     # confidentiality (new)
        blob_encryption_key=os.environ["BLOB_ENC_KEY"],
    )
)
```

**Design considerations:**
- **Algorithm:** AES-256-GCM (authenticated encryption — confidentiality + integrity in one pass)
- **Client-side encryption boundary:** Encryption happens in `BlobStore.put()` after handler serialization, before the blob backend writes to storage. The remote backend (S3, filesystem, future GCS) only ever receives ciphertext. Decryption happens in `BlobStore.get()` after blob backend reads, before handler deserialization.
- **Envelope encryption:** Generate a unique DEK (data encryption key) per blob, encrypt the DEK with the master key, store encrypted DEK alongside the blob. This limits the blast radius of a single compromised DEK.
- **Key management:** Leverage existing `SecurityConfig` and HKDF infrastructure. Derive blob encryption keys per namespace using HKDF with a different info string (`cacheness-aes-gcm-v1:{namespace_id}`) — same master key, cryptographically isolated from signing keys.
- **Performance:** AES-256-GCM is hardware-accelerated on modern CPUs (AES-NI). Overhead is proportional to blob size, not metadata complexity.
- **Interaction with signing:** Signing covers metadata fields including `file_hash`. With blob encryption, `file_hash` should be computed on the **ciphertext** (not plaintext), so integrity verification doesn't require decryption.
- **Remote storage model:** Encrypted blobs can be stored on untrusted S3 buckets, shared PostgreSQL databases, or libSQL cloud replicas. A server compromise exposes only ciphertext — useless without the client-held master key.

**Orthogonality with signing:**

| Feature | Protects | Against |
|---------|----------|---------|
| Entry signing (existing) | Metadata integrity | Tampering with cache keys, timestamps, file hashes |
| Blob hash verification (existing) | Blob integrity | Tampering with cached data files |
| Metadata encryption (libSQL) | Metadata confidentiality | Reading cache keys, data types, paths |
| Blob encryption (proposed) | Blob confidentiality | Reading cached data files |

**Impact:** Medium-High - Completes the security story for sensitive data use cases

**Effort:** Medium - Encryption primitives are straightforward; key management is the hard part (but HKDF infra exists)

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
from cacheness.core import UnifiedCache
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

### Recently Shipped (v0.7.0–v0.10.0)

The following capabilities from earlier roadmap versions have been implemented:

| Feature | Milestone | What shipped |
|---------|-----------|-------------|
| Handler package split | v0.7.0 | `handlers/` package with 11 files, `HandlerRegistry` with priority-based selection |
| Metadata package split | v0.7.0 | `metadata/` package with ABC, JSON, SQLite backends |
| Core mixin decomposition | v0.7.0 | 4 mixins extracted from `core.py` (verification, stats, custom metadata, storage mode) |
| Narrow exception handling | v0.7.0 | Zero unannotated `except Exception` in `src/cacheness/` |
| Concurrency foundation | v0.8.0 | Thread-safe `RLock` + SQLite WAL pragmas, crash-safe write-intent logging |
| Blob integrity validation | v0.8.0 | `verify_integrity(verify_signatures=True)`, `IntegrityReport.signature_failures` |
| `delete_by_prefix()` | v0.8.0 | Bulk prefix deletion with backend-optimized queries (SQL LIKE for SQLite) |
| `put_batch()` | v0.9.0 | Batch put with partial-success semantics |
| Deserialization security docs | v0.9.0 | Layered defense model documented, code-level comments |
| Windows key file permissions | v0.9.0 | `icacls`-based permission management for key files |
| Configurable key fallback | v0.10.0 | `key_fallback_policy` ("warn"/"raise"/"fallback") replacing boolean flag |
| HKDF key derivation | v0.10.0 | Per-namespace derived keys via HKDF-SHA256 (RFC 5869) |
| Key rotation API | v0.10.0 | `rotate_key()` with `RotationResult`, crash-safe mixed v2/v3 state |

---

### Phase 1: Remaining Management Operations

**Goal:** Complete missing CRUD operations (partially shipped)

1. **Storage Backend Layer**
   - [ ] `update_blob_data(cache_key, new_data)` - Replace data at key
   - [ ] `get_entries_batch(cache_keys)` - Batch get metadata
   - [ ] `delete_entries_batch(cache_keys)` - Batch delete
   - [ ] `copy_entry(source, dest)` / `move_entry(source, dest)` - Convenience wrappers

2. **Cache Layer Wrappers**
   - [ ] `cache.update_data(data, **kwargs)` - Update wrapper
   - [ ] `cache.touch(**kwargs, ttl_seconds)` - Refresh TTL
   - [ ] `cache.get_metadata(**kwargs)` - Expose metadata access
   - [ ] `cache.get_batch([kwargs_list])` - Batch get wrapper
   - [ ] `cache.copy(source, dest)` / `cache.move(source, dest)` - Convenience wrappers

3. **Testing & Documentation**
   - [ ] Test all operations across all backends
   - [ ] Update API reference
   - [ ] Add usage examples

**Priority:** ✅✅✅ Highest - Most requested, well-defined

---

### Phase 2: Version-in-Metadata & Migration Support

**Goal:** Enable safe upgrades and format migration (see [Section 10](#10-version-in-metadata-for-migration-support--medium-high-priority))

1. **Schema changes** — add `cacheness_version` and `serialization_format_version` fields to all metadata backends
2. **Compat read paths** — per-handler version dispatch for backward-compatible deserialization
3. **`cache.migrate()` API** — bulk re-serialization with `dry_run=True` support
4. **Critical for storage mode** — no function to re-execute on format change

**Priority:** ✅✅✅ High - Essential for storage mode, valuable for all users

---

### Phase 3: Encryption at Rest

**Goal:** Protect cached data confidentiality, not just integrity (see [Section 11](#11-encryption-at-rest--medium-priority))

1. **Metadata encryption** — via libSQL backend (`encryption_key` parameter)
2. **Blob encryption** — AES-256-GCM envelope encryption in the blob write path
3. **Key management** — leverage existing `SecurityConfig` and HKDF infrastructure

**Priority:** ✅✅ Medium-High - Completes the security story (signing = integrity, encryption = confidentiality)

---

### Phase 4: Tiered Pull-Through Cache

**Goal:** Compose local + remote caches with automatic pull-through (see [Section 9](#9-tiered-pull-through-cache--medium-priority))

1. **`TieredCache` orchestrator** — thin wrapper composing two `UnifiedCache` instances (~200 LOC)
2. **Invalidation strategies** — TTL-based, metadata-version check, or no invalidation
3. **Local size cap** — evict from local tier when size exceeds limit

**Priority:** ✅ Medium - High value for teams sharing caches

---

### Phase 5: Async Support

**Goal:** Enable modern async Python workflows

1. **Async Backends** — `AsyncPostgreSQLBackend` (asyncpg), `AsyncS3BlobStore` (aioboto3)
2. **`AsyncUnifiedCache`** — separate class, sync handlers run in executor
3. **Documentation** — FastAPI integration example, benchmarks

**Priority:** ✅ Medium - Modern Python standard, enables web services

---

### Phase 6: CLI Tool

**Goal:** Improve developer experience for debugging/maintenance

1. **Core Commands** — `cacheness inspect`, `list`, `stats`, `cleanup`, `verify`
2. **Advanced Commands** — `cacheness migrate`, `export`, `import`
3. **Output Formatting** — table, JSON, CSV

**Priority:** ⚠️ Medium - High DX impact, relatively easy

---

### Phase 7: Eviction Policies (Optional)

**Goal:** Better resource management

1. **Policies** — LRU, LFU, SizeBasedPolicy, CompositePolicy
2. **Schema changes** — `last_accessed` timestamp, `access_count`
3. **Background eviction** — non-blocking policy enforcement

**Priority:** ⚠️ Low-Medium - Nice-to-have

---

### Not Recommended (Don't Build)

- ❌ **REST API** - Contradicts philosophy, adds overhead, alternatives better
- ❌ **Content Deduplication** - Complex, niche benefit
- ❌ **Delta Compression** - Very complex, use external version control
- ❌ **Built-in Distributed Lock** - Use external lock manager (Redis, etcd)
- ❌ **Built-in Metrics/Monitoring** - Use standard observability tools

---

## Summary: What to Build Next

### Top 3 Priorities

1. **✅ Remaining Management Operations**
   - `update_data`, `touch`, `get_batch`, copy/move
   - Highest user impact, well-defined scope
   - **Start here — partially shipped**

2. **✅ Version-in-Metadata for Migration Support**
   - Critical for storage mode (data loss prevention)
   - Per-entry format versioning + `migrate()` API
   - **Do this second**

3. **✅ Encryption at Rest**
   - Metadata encryption via libSQL, blob encryption via AES-256-GCM
   - Completes signing (integrity) + encryption (confidentiality)
   - **Do this third**

### Consider Later

- ⚠️ **Tiered Pull-Through Cache** - High value for teams, low implementation effort
- ⚠️ **Async Support** - Important for web services, high effort
- ⚠️ **CLI Tool** - Good DX, moderate effort
- ⚠️ **LRU/LFU Eviction** - Useful but not essential
- ⚠️ **Connection Pool Config** - Easy win for PostgreSQL users
- ⚠️ **Better Type Hints** - Nice DX improvement

### Don't Build

- ❌ **REST API** - Wrong abstraction, contradicts philosophy
- ❌ **Deduplication** - Too complex for benefit

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
