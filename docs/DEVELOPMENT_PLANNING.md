# Development Planning

This document tracks potential bugs, issues, and design improvements identified during code review. Items are prioritized by severity and impact.

---

## 🏗️ Architectural Evolution: Separation of Concerns

### Background

This library began as a simple persistent cache solution. Over time, it evolved to include:
- Multiple storage backends (JSON, SQLite, In-Memory)
- Blob management with compression (blosc2, lz4, zstd, gzip)
- Type-aware serialization handlers (DataFrames, arrays, objects)
- Security features (HMAC signing, integrity verification)
- SQL pull-through caching with gap detection

**Key Insight:** Most of the complexity lives in the **storage infrastructure**, not the caching logic itself. The actual "caching" semantics (TTL, eviction, key generation) are a thin layer on top of a sophisticated blob storage and metadata management system.

### Current Architecture Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER-FACING APIs                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  cacheness  │  │  @cached    │  │  SqlCache               │  │
│  │  (core.py)  │  │ (decorator) │  │  (pull-through cache)   │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CACHING SEMANTICS LAYER                        │
│  • TTL management          • Key generation                      │
│  • Eviction policies       • Cache statistics                    │
│  • Hit/miss tracking       • Expiration cleanup                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BLOB STORAGE LAYER                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  HandlerRegistry │  │  Compression    │  │  Security       │  │
│  │  (type handlers) │  │  (blosc2/lz4)   │  │  (signing)      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Metadata Backends                         ││
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐     ││
│  │  │  JSON    │  │   SQLite     │  │   In-Memory        │     ││
│  │  │  Backend │  │   Backend    │  │   Backend          │     ││
│  │  └──────────┘  └──────────────┘  └────────────────────┘     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Code Distribution Analysis

| Layer | Files | LOC (approx) | Complexity |
|-------|-------|--------------|------------|
| **Caching Semantics** | core.py (partial), decorators.py | ~400 | Low |
| **Blob Storage** | handlers.py, compress_pickle.py, metadata.py | ~2,500 | High |
| **SQL Pull-Through** | sql_cache.py | ~1,800 | High |
| **Configuration** | config.py | ~550 | Medium |
| **Security** | security.py | ~240 | Medium |
| **Utilities** | serialization.py, file_hashing.py, etc. | ~800 | Medium |

**Observation:** ~75% of the codebase is blob storage infrastructure, ~25% is caching logic.

---

### Recommendation: Two-Layer Architecture

#### Option A: Monorepo with Clear Module Boundaries (Recommended for Now)

Keep as a single library but reorganize into distinct sub-packages with clear boundaries:

```
cacheness/
├── storage/                    # Low-level blob storage (reusable)
│   ├── __init__.py
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract backend interface
│   │   ├── json_backend.py
│   │   ├── sqlite_backend.py
│   │   └── memory_backend.py
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── base.py            # Handler interfaces
│   │   ├── dataframe.py       # Pandas/Polars handlers
│   │   ├── array.py           # NumPy/blosc2 handlers
│   │   └── object.py          # Pickle/dill handlers
│   ├── compression.py
│   ├── security.py
│   └── blob_store.py          # Main BlobStore class
│
├── cache/                      # Caching semantics (uses storage)
│   ├── __init__.py
│   ├── unified_cache.py       # Current UnifiedCache
│   ├── decorators.py
│   ├── ttl.py                 # TTL/expiration logic
│   └── eviction.py            # Eviction policies
│
├── sql/                        # SQL pull-through cache
│   ├── __init__.py
│   ├── sql_cache.py
│   ├── adapters.py
│   └── gap_detection.py
│
└── __init__.py                 # Public API exports
```

**Pros:**
- Clear separation without breaking changes
- Storage layer becomes independently testable
- Easier to reason about responsibilities
- Natural migration path to full separation

**Cons:**
- Still a single package to version
- Can't use storage layer without installing full library

---

#### Option B: Separate Packages (Future State)

Split into two separate PyPI packages:

```
# Package 1: blobcache-storage (or similar name)
blobcache_storage/
├── backends/
├── handlers/
├── compression.py
├── security.py
└── blob_store.py

# Package 2: cacheness (depends on blobcache-storage)
cacheness/
├── unified_cache.py
├── decorators.py
├── sql_cache.py
└── __init__.py
```

**Pros:**
- True separation of concerns
- Storage layer usable for non-caching use cases (e.g., ML model versioning, artifact storage)
- Independent versioning and release cycles
- Smaller install size for users who only need one layer

**Cons:**
- Breaking change for existing users
- Two packages to maintain
- Dependency coordination complexity

---

#### Option C: Service-Oriented Architecture (Future Consideration)

For enterprise/team use cases, the blob storage could become a service:

```
┌─────────────────────┐     ┌─────────────────────────────┐
│  cacheness (client) │────▶│  Blob Storage Service       │
│  - TTL logic        │ HTTP│  - REST/gRPC API            │
│  - Key generation   │     │  - Distributed storage      │
│  - Local fallback   │     │  - Central metadata DB      │
└─────────────────────┘     └─────────────────────────────┘
```

**Pros:**
- Centralized cache for teams
- Shared storage across services
- Better observability

**Cons:**
- Significant complexity increase
- Network latency
- Deployment overhead
- Overkill for most use cases

---

### Recommended Migration Path

#### Phase 1: Internal Reorganization (Low Risk)
1. Create `storage/` sub-package with clean interfaces
2. Move handlers, backends, compression into storage
3. Update imports (maintain backward compatibility in `__init__.py`)
4. Add `BlobStore` class as primary storage API

```python
# New low-level API
from cacheness.storage import BlobStore

store = BlobStore(backend="sqlite", compression="lz4")
blob_id = store.put(data, metadata={"type": "model", "version": "1.0"})
data = store.get(blob_id)

# Existing high-level API (unchanged)
from cacheness import cacheness

cache = cacheness()
cache.put(data, model="xgboost", version="1.0")
```

#### Phase 2: Stabilize Storage API (Medium Risk)
1. Document storage layer as semi-public API
2. Gather feedback on interface design
3. Add storage-specific tests
4. Consider use cases beyond caching:
   - ML model versioning
   - Artifact storage
   - Data pipeline checkpoints

#### Phase 3: Evaluate Full Separation (Future Decision)
1. Assess community adoption and feedback
2. Determine if separate packages provide value
3. If splitting, use namespace packages for smooth migration:
   ```python
   # Before split
   from cacheness.storage import BlobStore
   
   # After split (same import!)
   from cacheness.storage import BlobStore  # Now from cacheness-storage package
   ```

---

### Design Principles for Storage Layer

If we extract the storage layer, it should follow these principles:

1. **Content-Addressable Option**: Support content-based addressing (hash of content as key) for deduplication

2. **Pluggable Backends**: Easy to add new backends (Redis, S3, etc.)

3. **Streaming Support**: Handle large blobs without loading entirely into memory

4. **Metadata First-Class**: Rich, queryable metadata separate from blob content

5. **Atomic Operations**: Ensure put/get/delete are atomic

6. **No Caching Semantics**: TTL, eviction, etc. belong in the caching layer

**Proposed BlobStore Interface:**
```python
class BlobStore(ABC):
    """Low-level blob storage interface."""
    
    def put(self, data: Any, key: Optional[str] = None, 
            metadata: Optional[Dict] = None) -> str:
        """Store blob, return key."""
        
    def get(self, key: str) -> Optional[Any]:
        """Retrieve blob by key."""
        
    def delete(self, key: str) -> bool:
        """Delete blob by key."""
        
    def exists(self, key: str) -> bool:
        """Check if blob exists."""
        
    def list(self, prefix: Optional[str] = None, 
             metadata_filter: Optional[Dict] = None) -> List[str]:
        """List blob keys with optional filtering."""
        
    def get_metadata(self, key: str) -> Optional[Dict]:
        """Get blob metadata without loading content."""
        
    def update_metadata(self, key: str, metadata: Dict) -> bool:
        """Update blob metadata."""
```

---

### Questions to Resolve

1. **Naming**: If split, what should the storage package be called?
   - `blobcache-storage`
   - `pyblobstore`
   - `cacheness-core`
   - Other?

2. **Scope**: Should storage layer include SQL pull-through cache, or is that a separate concern?

3. **Backends**: Which backends should be in core vs. extras?
   - Core: SQLite, JSON, Memory
   - Extras: S3, Redis, PostgreSQL?

4. **Versioning**: How to handle API versioning if split?

---

## 🔴 High Severity Issues

### 1. Query Method Returns Unclosed Session ✅ FIXED

**Location:** [src/cacheness/core.py#L407-411](../src/cacheness/core.py#L407-L411)

**Description:** The `query_custom()` method previously created a SQLAlchemy session but returned a query object without ensuring the session was closed. This leaked database connections.

**Risk:** 
- SQLite connection pool exhaustion
- Resource leaks in long-running applications
- Potential database locking issues on Windows

**Fix Applied:**
1. `query_custom()` now returns a list directly with automatic session cleanup
2. Added optional `filters` parameter for simple equality filtering
3. Added new `query_custom_session()` context manager for advanced queries

```python
# Simple query - session is automatically cleaned up
results = cache.query_custom("ml_experiments")

# With filters
results = cache.query_custom("ml_experiments", {"model_type": "xgboost"})

# Advanced filtering with context manager (guaranteed cleanup)
with cache.query_custom_session("ml_experiments") as query:
    high_accuracy = query.filter(MLExperimentMetadata.accuracy >= 0.9).all()
```

**Status:** ✅ Fixed

---

## 🟠 Medium Severity Issues

### 2. JSON Metadata File Corruption Risk ✅ FIXED

**Location:** [src/cacheness/metadata.py#L697-731](../src/cacheness/metadata.py#L697-L731)

**Description:** The JSON backend's `_save_to_disk()` previously wrote directly to the metadata file without atomic write protection.

**Risk:**
- Process crash during write could corrupt metadata
- All cache entries could be lost

**Fix Applied:** Implemented atomic write pattern using temp file + rename.

**Status:** ✅ Fixed in commit

---

### 3. Global Cache Memory Leak on Reset ✅ FIXED

**Location:** [src/cacheness/core.py#L1079-1087](../src/cacheness/core.py#L1079-L1087)

**Description:** The `reset_cache()` function previously created a new cache without closing the old one, leaking SQLite connections.

**Previous Code:**
```python
def reset_cache(config=None, metadata_backend=None):
    global _global_cache
    _global_cache = UnifiedCache(config, metadata_backend)  # Old cache not closed!
```

**Fix Applied:** Now properly closes the previous cache before creating a new one.

**Status:** ✅ Fixed in commit

---

### 4. Signature Verification Field Mismatch Risk

**Location:** [src/cacheness/core.py#L698-727](../src/cacheness/core.py#L698-L727)

**Description:** The signature fields used during `put()` must exactly match those used during `get()`. If the stored metadata structure differs (e.g., missing `cache_key_params` during retrieval), verification may fail incorrectly.

**Risk:**
- Valid cache entries could be rejected as tampered
- Silent data loss if `delete_invalid_signatures=True`

**Recommended Fix:**
```python
# Ensure consistent field extraction in both put() and get()
def _extract_signable_fields(self, entry_data: Dict, cache_key: str) -> Dict:
    """Extract fields for signing in a consistent manner."""
    return {
        field: entry_data.get(field) 
        for field in self.signer.signed_fields
        if field in entry_data or field in self.signer.required_fields
    }
```

**Status:** ✅ Fixed - Added `_extract_signable_fields()` helper method for consistent field extraction in both `put()` and `get()`

---

### 5. Decorator Cache Instance Never Closed

**Location:** [src/cacheness/decorators.py#L95-98](../src/cacheness/decorators.py#L95-L98)

**Description:** The `@cached` decorator creates its own `UnifiedCache` instance if none is provided, but this instance is never explicitly closed.

**Current Code:**
```python
class cached:
    def __init__(self, ...):
        if self.cache_instance is None:
            self.cache_instance = UnifiedCache()  # Never closed!
```

**Risk:**
- Connection leaks when decorated functions go out of scope
- Resource accumulation in long-running processes

**Recommended Fix:**
```python
import atexit

class cached:
    _instances = []  # Track for cleanup
    
    def __init__(self, ...):
        if self.cache_instance is None:
            self.cache_instance = UnifiedCache()
            cached._instances.append(self.cache_instance)
    
    @classmethod
    def _cleanup_all(cls):
        for instance in cls._instances:
            try:
                instance.close()
            except Exception:
                pass
        cls._instances.clear()

# Register cleanup
atexit.register(cached._cleanup_all)
```

**Status:** ✅ Fixed - Added atexit cleanup, weakref tracking for decorator-created cache instances, and `close()` method

---

## 🟡 Low Severity Issues

### 6. Missing Validation for Custom Signed Fields

**Location:** [src/cacheness/config.py#L223-225](../src/cacheness/config.py#L223-L225)

**Description:** `SecurityConfig.custom_signed_fields` accepts any list of field names without validating against available/valid fields.

**Current Code:**
```python
@dataclass
class SecurityConfig:
    custom_signed_fields: Optional[List[str]] = None  # No validation
```

**Risk:**
- User could specify non-existent fields, causing silent failures
- Typos in field names would go undetected

**Recommended Fix:**
```python
VALID_SIGNED_FIELDS = {
    "cache_key", "file_hash", "data_type", "file_size", 
    "created_at", "prefix", "description", "actual_path",
    "object_type", "storage_format", "serializer", "compression_codec"
}

def __post_init__(self):
    if self.custom_signed_fields:
        invalid = set(self.custom_signed_fields) - VALID_SIGNED_FIELDS
        if invalid:
            raise ValueError(f"Invalid signed fields: {invalid}. Valid: {VALID_SIGNED_FIELDS}")
```

**Status:** ✅ Fixed - Added `VALID_SIGNED_FIELDS` set and validation in `SecurityConfig.__post_init__`

---

### 7. Inconsistent Exception Handling

**Location:** Multiple modules

**Description:** The codebase has a well-defined `CacheError` hierarchy in [error_handling.py](../src/cacheness/error_handling.py), but many places catch bare `Exception` and only log without re-raising or converting to typed exceptions.

**Examples:**
```python
# Good (in error_handling.py)
class CacheStorageError(CacheError): ...
class CacheSerializationError(CacheError): ...

# Inconsistent (scattered throughout)
except Exception as e:
    logger.error(f"Failed to X: {e}")  # Silent failure
    return None
```

**Risk:**
- Difficult to handle specific error types in calling code
- Silent failures can mask bugs

**Recommended Fix:**
- Audit all `except Exception` blocks
- Convert to typed exceptions where appropriate
- Document which methods can raise which exceptions

**Status:** ✅ Fixed - Improved error handling in `core.py` `put()` and `get()` methods with specific exception types (FileNotFoundError, OSError, IOError)

---

### 8. Test Cleanup Writes to Deleted Directory

**Location:** Test fixtures

**Description:** During test teardown, the temp directory is deleted before the cache's `close()` method tries to save metadata, causing benign error logs.

**Log Output:**
```
ERROR: Failed to save JSON metadata: [Errno 2] No such file or directory
```

**Risk:** Low - only affects test output cleanliness

**Fix Applied:** JSON backend now gracefully handles deleted directories.

**Status:** ✅ Fixed in commit

---

## 🔧 Design Improvements

### 1. Add Session Context Manager API

**Description:** Provide a proper context manager for database queries to ensure cleanup.

```python
# Proposed API
with cache.session() as session:
    results = session.query(MyMetadata).filter(...).all()
```

**Benefits:**
- Guaranteed session cleanup
- Familiar pattern for SQLAlchemy users
- Prevents connection leaks

---

### 2. Implement Connection Pooling Monitoring

**Description:** Add optional metrics/logging for connection pool health.

```python
def get_pool_stats(self) -> Dict[str, int]:
    """Get connection pool statistics."""
    return {
        "pool_size": self.engine.pool.size(),
        "checked_out": self.engine.pool.checkedout(),
        "overflow": self.engine.pool.overflow(),
    }
```

**Benefits:**
- Early warning for connection leaks
- Performance debugging
- Capacity planning

---

### 3. Add Graceful Degradation for Signing

**Description:** When signature verification fails, provide option to return data with warning flag instead of returning `None`.

```python
# Current behavior
if not self.signer.verify_entry(...):
    return None  # Data lost

# Proposed behavior
result = cache.get(..., strict_verification=False)
if result.signature_valid == False:
    logger.warning("Signature invalid, data may be tampered")
```

**Benefits:**
- Allows recovery from legitimate signature mismatches
- Better debugging of signature issues
- Gradual migration path for security upgrades

---

### 4. Standardize Logging Levels

**Description:** Create logging guidelines and audit current usage.

| Level | Current Usage | Recommended Usage |
|-------|--------------|-------------------|
| DEBUG | Verbose operations | Internal details, cache key generation |
| INFO | Cache hits, initialization | Significant state changes only |
| WARNING | Non-critical failures | Degraded performance, fallbacks |
| ERROR | Critical failures | Data loss, unrecoverable errors |

---

## Priority Matrix

| Issue | Severity | Effort | Status |
|-------|----------|--------|--------|
| Query Session Leak | High | Medium | ✅ Fixed |
| JSON Atomic Writes | Medium | Low | ✅ Fixed |
| Global Cache Leak on Reset | Medium | Low | ✅ Fixed |
| Signature Field Mismatch | Medium | Low | ✅ Fixed |
| Decorator Cache Cleanup | Medium | Low | ✅ Fixed |
| Custom Fields Validation | Low | Low | ✅ Fixed |
| Exception Handling Audit | Low | Medium | ✅ Fixed |
| Test Cleanup Errors | Low | Low | ✅ Fixed |

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-01-23 | Initial document created from code review | Code Review |
| 2026-01-23 | Fixed: JSON atomic writes, global cache leak | Code Review |
| 2026-01-23 | Fixed: Query session leak (High #1) - Added `query_custom_session()` context manager | Code Review |
| 2026-01-23 | Fixed: Signature field mismatch (Med #4) - Added `_extract_signable_fields()` helper | Code Review |
| 2026-01-23 | Fixed: Decorator cache never closed (Med #5) - Added atexit cleanup | Code Review |
| 2026-01-23 | Fixed: Custom signed fields validation (Low #6) - Added field validation | Code Review |
| 2026-01-23 | Fixed: Inconsistent exception handling (Low #7) - Improved error types | Code Review |

