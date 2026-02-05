# Missing Cache Management Operations

## Analysis Date: February 5, 2026

This document identifies missing management operations and clarifies the architectural separation between:
- **Storage Backend Operations** - CRUD, bulk operations, metadata (storage layer)
- **Cache Layer Operations** - TTL, expiration, cache semantics (caching layer)

---

## Architectural Clarification

### Storage Backend Responsibilities
- Basic CRUD (create, read, update, delete)
- Bulk operations (delete many, batch operations)
- Metadata management (get/update metadata without loading data)
- Pattern-based operations (delete by prefix, list with filters)
- **NO TTL awareness** - storage is "dumb" about expiration
- **Enforce cache_key immutability** - keys cannot be modified after creation

### Cache Layer Responsibilities  
- TTL policy and enforcement
- Cache hit/miss logic with expiration checking
- Touch/refresh operations (extending expiration)
- Cleanup of expired entries (delegates timestamp deletion to backend)
- Wraps storage backend with caching semantics

**Current Issue:** Some operations are missing from the appropriate layer.

---

## Current API Coverage

### ✅ Currently Available Operations

**UnifiedCache (`cacheness`):**
- `put()` - Store data
- `get()` - Retrieve data  
- `get_with_metadata()` - Retrieve with metadata
- `exists()` - Check if entry exists (via kwargs, not direct)
- `invalidate()` - Delete by kwargs or cache_key
- `clear_all()` - Delete all entries
- `list_entries()` - List all entries with metadata
- `get_stats()` - Get cache statistics
- `query_meta()` - Query metadata with filters
- `query_custom()` - Query custom metadata tables
- `query_custom_session()` - Get session for custom queries

**BlobStore:**
- `put()` - Store blob
- `get()` - Retrieve blob
- `get_metadata()` - Get metadata without loading data
- `update_metadata()` - Update blob metadata
- `delete()` - Delete blob by key
- `exists()` - Check if blob exists
- `list()` - List blobs with prefix/metadata filters
- `clear()` - Delete all blobs

---

## ❌ Missing Operations (By Layer)

### Storage Backend Layer (MetadataBackend, BlobStore)

These operations manipulate stored data and should be implemented in the storage backends, NOT the cache layer.

#### 1. **Update Blob Data** (High Priority)

**Current State:** Must delete then re-insert to update data at an existing cache_key.

**Storage Backend Operation:** Replace blob data at a fixed cache_key.

**Important:** The cache_key is computed from input parameters and is **immutable**. This operation replaces the blob data stored at that key, not the key itself.

**Proposed API:**

```python
# Storage Backend (MetadataBackend) - Replace blob data at existing key
backend.update_blob_data(cache_key, new_data)  # Replace data, update derived metadata

# BlobStore (already has update_metadata for metadata-only updates)
store.update_metadata(key, metadata)  # ✅ Already exists - update metadata only
store.update_data(key, new_data)  # ❌ Missing - replace blob data at key

# Cache Layer (delegates to backend)
cache.update_data(new_data, experiment="exp_001")  # Replace data at this cache_key
```

**Implementation:** Add to `MetadataBackend` and `BlobStore` base classes. Updates the blob data and derived metadata (file_size, content_hash, created_at) but cache_key remains unchanged.

**Note:** There is no "replace cache_key" operation - the cache_key is immutable and derived from input parameters. Changing the input parameters creates a NEW cache entry.

---

#### 2. **Bulk Delete by Pattern** (High Priority)

**Current State:** Must loop over list results and delete individually.

**Storage Backend Operation:** Delete multiple entries matching criteria.

**Proposed API:**

```python
# Storage Backend (MetadataBackend)
count = backend.delete_where(lambda entry: entry.get("project") == "ml_models")
count = backend.delete_by_prefix(cache_key_prefix="exp_")

# BlobStore (already has prefix support via list())
deleted = store.delete_by_prefix(prefix="old_models/")  # ❌ Missing

# Cache Layer (delegates to backend)
count = cache.delete_by_prefix(project="ml_models")  # Convenience wrapper
```

**Implementation:** Add to `MetadataBackend` base class. Can be implemented using existing `list_entries()` + `remove_entry()` loop, but backends can optimize.

---

#### 3. **Get Metadata Without Loading Data** (Medium Priority)

**Current State:** 
- `BlobStore.get_metadata()` ✅ Already exists
- `MetadataBackend.get_entry()` ✅ Already exists
- `UnifiedCache.get_metadata()` ❌ Not exposed

**Storage Backend Operation:** Already supported in backends!

**Proposed API:**

```python
# Storage Backend - Already exists
entry = backend.get_entry(cache_key)  # ✅ Returns metadata
meta = store.get_metadata(key)  # ✅ Returns metadata

# Cache Layer - Missing exposure
meta = cache.get_metadata(experiment="exp_001")  # ❌ Should delegate to backend
```

**Implementation:** Trivial - just expose `backend.get_entry()` in `UnifiedCache`.

---

#### 4. **Batch Operations** (Medium Priority)

**Current State:** Must make N individual calls for N operations.

**Storage Backend Operation:** Optimize multiple operations in single transaction/batch.

**Proposed API:**

```python
# Storage Backend (MetadataBackend)
entries = backend.get_entries_batch([key1, key2, key3])  # Get multiple
backend.delete_entries_batch([key1, key2, key3])  # Delete multiple
backend.update_entries_batch({key1: meta1, key2: meta2})  # Update multiple

# Cache Layer (delegates to backend)
results = cache.get_batch([
    {"experiment": "exp_001"},
    {"experiment": "exp_002"},
])
```

**Implementation:** Add to `MetadataBackend`. SQLite/PostgreSQL can use transactions, JSON/Memory can iterate efficiently.

---

#### 5. **Copy/Move Entries** (Low Priority - Convenience Operations)

**Current State:** Users must manually get then put to duplicate entries.

**Storage Backend Operation:** Convenience methods that compose CRUD primitives.

**Why Provide These:** While users can always compose CRUD operations themselves, copy/move provide:
- **Atomicity** - Backend can guarantee atomic move (important for cache_key immutability)
- **Efficiency** - Server-side copy avoids transferring large blobs through client
- **Convenience** - Common operations packaged for ease of use

**Implementation Strategy:**
```python
# Internally implemented using CRUD primitives
def copy_entry(self, source_key, dest_key):
    """Convenience: Get source + Put to dest."""
    entry = self.get_entry(source_key)  # CRUD: Read
    if entry:
        self.put_entry(dest_key, entry)  # CRUD: Create
    return entry is not None

def move_entry(self, source_key, dest_key):
    """Convenience: Atomic copy + delete."""
    # Backend can wrap in transaction for atomicity
    if self.copy_entry(source_key, dest_key):
        self.remove_entry(source_key)  # CRUD: Delete
        return True
    return False
```

**Proposed API:**

```python
# Storage Backend (MetadataBackend) - Optional convenience methods
backend.copy_entry(source_key, dest_key)  # Implemented as: get + put
backend.move_entry(source_key, dest_key)  # Implemented as: get + put + delete (atomic)

# BlobStore - Optional convenience methods
store.copy(source_key, dest_key)  # Server-side copy (avoids data transfer)
store.move(source_key, dest_key)  # Atomic rename

# Cache Layer - Wraps backend convenience methods
cache.copy(
    source={"experiment": "exp_001"},
    dest={"experiment": "exp_001_backup"}
)  # Delegates to backend.copy_entry()

cache.move(
    source={"experiment": "exp_001"},
    dest={"experiment": "exp_002"}
)  # Delegates to backend.move_entry()

# Users can always use CRUD directly instead
data = cache.get(experiment="exp_001")
cache.put(data, experiment="exp_001_backup")  # Equivalent to copy
```

**Priority Rationale:** Low priority because they're convenience operations - users can always compose CRUD operations. Provide value for atomicity and efficiency, but not required for core functionality.

---

### Cache Layer Operations

These operations involve caching semantics (TTL, expiration) and should be implemented in the cache layer.

#### 6. **Touch/Refresh TTL** (High Priority)

**Current State:** No way to extend expiration without reloading data.

**Cache Layer Operation:** Update entry timestamp to reset TTL.

**Proposed API:**

```python
# Cache Layer ONLY - storage backends don't know about TTL
cache.touch(experiment="exp_001")  # Reset TTL to default
cache.touch(experiment="exp_001", ttl_seconds=hours(48))  # Custom TTL
```

**Implementation:** Update `created_at` timestamp in backend entry. This is a cache-layer concern because:
- Storage backend is TTL-agnostic (stores timestamps, doesn't interpret them)
- Cache layer defines TTL policy and checks expiration
- `touch()` is semantic sugar for "reset cache expiration"

**Backend Support Needed:** `backend.update_entry_timestamp(cache_key, new_timestamp)`

---

#### 7. **Cleanup Expired Entries** (Already Implemented)

**Current State:** ✅ `backend.cleanup_expired(ttl_seconds)` exists

**Note:** This is currently in the backend, which is correct - the backend deletes entries by timestamp. The cache layer decides WHAT TTL to use, then calls `backend.cleanup_expired(ttl_seconds)`.

---

### Cross-Layer Operations

Some operations span both layers.

#### 8. **Export/Import Cache** (Low Priority)

**Span:** Storage backend provides data, cache layer provides TTL/policy context.

**Proposed API:**

```python
# Export complete cache state (data + metadata + TTL info)
cache.export_to_file("backup.tar.gz", compress=True)
cache.export_to_dict()

# Import (validates TTL, restores cache state)
cache.import_from_file("backup.tar.gz")
```

---

#### 9. **Verify and Repair Cache** (Low Priority)

**Span:** Storage backend verifies data integrity, cache layer verifies cache coherence.

**Proposed API:**

```python
# Storage validation: files exist, signatures valid, etc.
issues = cache.verify_integrity()

# Cache validation: TTL consistency, orphaned entries, etc.
issues = cache.verify_cache_coherence()

# Repair
cache.repair(dry_run=True)
```

---

## Priority Recommendations (Revised by Layer)

**Use Case:** 
- Keep frequently accessed data alive longer
- Reset TTL for active sessions
- Prevent expiration of long-running computations

**Proposed API:**

```python
# For cacheness
cache.touch(experiment="exp_001")  # Reset TTL to default
cache.touch(experiment="exp_001", ttl_seconds=hours(48))  # Custom TTL

# Alternative names: refresh(), extend_ttl(), renew()

# For BlobStore  
store.touch(key="model_v1")  # Update access time (already tracked)
store.extend_ttl(key="model_v1", ttl_seconds=days(7))  # If TTL supported
```

**Implementation:**
```python
def touch(self, cache_key: Optional[str] = None, ttl_seconds: Optional[float] = None, **kwargs):
    """
    Update entry timestamp to extend TTL without reloading data.
    
    Args:
        cache_key: Direct cache key
        ttl_seconds: New TTL in seconds (uses default if None)
        **kwargs: Cache key parameters
    
    Returns:
        bool: True if entry exists and was touched
    """
    if cache_key is None:
        cache_key = self._create_cache_key(kwargs)
    
    entry = self.metadata_backend.get_entry(cache_key)
    if not entry:
        return False
    
    # Update timestamp
    entry["created_at"] = datetime.now(timezone.utc).isoformat()
    self.metadata_backend.put_entry(cache_key, entry)
    return True
```

---

### 2. **Replace/Update Data** (Medium Priority)

**Problem:** No explicit API to update cached data - must invalidate then put.

**Use Case:**
- Update stale data in-place
- Modify cached results
- Atomic updates with version checking

**Proposed API:**

```python
# For cacheness
cache.replace(new_data, experiment="exp_001")  # Replace if exists
cache.update(new_data, experiment="exp_001", if_exists=True)  # Only if exists

# Conditional update
cache.update(new_data, experiment="exp_001", 
             version_check="v1.0")  # Only if version matches

# For BlobStore
store.replace(key="model_v1", data=new_model)  # Replace blob data
```

**Implementation:**
```python
def replace(self, data: Any, cache_key: Optional[str] = None, **kwargs):
    """
    Replace existing cache entry with new data.
    
    Args:
        data: New data to store
        cache_key: Direct cache key
        **kwargs: Cache key parameters
    
    Returns:
        bool: True if entry existed and was replaced, False otherwise
    """
    if cache_key is None:
        cache_key = self._create_cache_key(kwargs)
    
    # Check if exists
    if not self.metadata_backend.get_entry(cache_key):
        return False
    
    # Replace using existing put logic
    self.put(data, cache_key=cache_key, **kwargs)
    return True
```

---

### 3. **Bulk Delete by Pattern** (Medium Priority)

**Problem:** `invalidate()` deletes single entries. No bulk delete by pattern/prefix.

**Use Case:**
- Delete all experiments from a project
- Clear specific data types
- Remove old versions

**Proposed API:**

```python
# For cacheness
cache.delete_by_prefix(project="ml_models")  # All entries with project=ml_models
cache.delete_by_pattern(**{"experiment_*": "exp_"})  # Wildcard matching
cache.delete_matching(lambda meta: meta.get("version", "").startswith("v1"))

# For BlobStore (already has prefix support in list())
deleted = store.delete_by_prefix(prefix="old_models/")  # Returns count
```

**Implementation:**
```python
def delete_by_prefix(self, **prefix_kwargs) -> int:
    """
    Delete all entries matching prefix parameters.
    
    Args:
        **prefix_kwargs: Cache key parameters to match
    
    Returns:
        int: Number of entries deleted
    """
    # Query matching entries
    entries = self.query_meta(**prefix_kwargs)
    
    deleted_count = 0
    for entry in entries:
        cache_key = entry.get("cache_key")
        if cache_key and self.metadata_backend.remove_entry(cache_key):
            deleted_count += 1
    
    return deleted_count
```

---

### 4. **Get Metadata Without Loading Data** (Low Priority)

**Problem:** `cacheness` doesn't expose `get_metadata()` like `BlobStore` does.

**Use Case:**
- Check metadata before deciding to load
- Inspect TTL/expiration without loading
- Query file size before download

**Proposed API:**

```python
# For cacheness
meta = cache.get_metadata(experiment="exp_001")
# Returns: {
#     "cache_key": "...",
#     "created_at": "...",
#     "file_size": 1024,
#     "data_type": "pandas_dataframe",
#     "expires_at": "...",
# }

# Already available in BlobStore
meta = store.get_metadata(key="model_v1")
```

**Implementation:**
```python
def get_metadata(self, cache_key: Optional[str] = None, **kwargs) -> Optional[Dict[str, Any]]:
    """
    Get entry metadata without loading data.
    
    Args:
        cache_key: Direct cache key
        **kwargs: Cache key parameters
    
    Returns:
        Metadata dictionary or None if not found
    """
    if cache_key is None:
        cache_key = self._create_cache_key(kwargs)
    
    return self.metadata_backend.get_entry(cache_key)
```

---

### 5. **Export/Import Cache** (Low Priority)

**Problem:** No way to backup/restore cache or migrate between environments.

**Use Case:**
- Backup cache state
- Share cache between team members
- Migrate from dev to prod

**Proposed API:**

```python
# Export
cache.export_to_file("cache_backup.tar.gz", compress=True)
cache.export_to_dict()  # Returns serializable dict

# Import
cache.import_from_file("cache_backup.tar.gz")
cache.import_from_dict(data)

# Selective export
cache.export_to_file("ml_models.tar.gz", 
                     filter_fn=lambda meta: meta.get("project") == "ml")
```

---

### 6. **Verify and Repair Cache** (Low Priority)

**Problem:** No tools to detect corrupted entries or missing files.

**Use Case:**
- Detect broken cache entries
- Clean up orphaned files
- Verify integrity after crashes

**Proposed API:**

```python
# Verify cache integrity
issues = cache.verify_integrity()
# Returns: [
#     {"cache_key": "...", "issue": "missing_file"},
#     {"cache_key": "...", "issue": "corrupted_signature"},
# ]

# Repair cache
cache.repair(dry_run=True)  # Show what would be fixed
cache.repair()  # Actually fix issues

# Detect orphaned files
orphaned = cache.find_orphaned_files()  # Files without metadata entries
cache.cleanup_orphaned_files()
```

---

### 7. **Batch Operations** (Low Priority)

**Problem:** No efficient way to operate on multiple entries at once.

**Use Case:**
- Load multiple related entries
- Update metadata for many entries
- Bulk touch/refresh

**Proposed API:**

```python
# Batch get
results = cache.get_batch([
    {"experiment": "exp_001"},
    {"experiment": "exp_002"},
    {"experiment": "exp_003"},
])
# Returns: {
#     "exp_001": data1,
#     "exp_002": data2,
#     "exp_003": None,  # Not found
# }

# Batch touch
cache.touch_batch(project="ml_models")  # All matching entries

# Batch update metadata
cache.update_metadata_batch(
    filters={"project": "ml_models"},
    metadata={"reviewed": True}
)
```

---

### 8. **Copy/Clone Entries** (Very Low Priority)

**Problem:** No way to duplicate cache entries with different keys.

**Use Case:**
- Create entry variants
- Backup before modification
- Fork experiments

**Proposed API:**

```python
# Copy entry
cache.copy(
    source={"experiment": "exp_001"},
    dest={"experiment": "exp_001_backup"}
)

# Clone with modifications
cache.clone(
    source={"experiment": "exp_001"},
    dest={"experiment": "exp_002"},
    update_metadata={"cloned_from": "exp_001"}
)
```

---

### 2. **Replace/Update Data** (Medium Priority)

**Problem:** No explicit API to update cached data - must invalidate then put.

**Use Case:**
- Update stale data in-place
- Modify cached results
- Atomic updates with version checking

**Proposed API:**

```python
# For cacheness
cache.replace(new_data, experiment="exp_001")  # Replace if exists
cache.update(new_data, experiment="exp_001", if_exists=True)  # Only if exists

# Conditional update
cache.update(new_data, experiment="exp_001", 
             version_check="v1.0")  # Only if version matches

# For BlobStore
store.replace(key="model_v1", data=new_model)  # Replace blob data
```

**Implementation:**
```python
def replace(self, data: Any, cache_key: Optional[str] = None, **kwargs):
    """
    Replace existing cache entry with new data.
    
    Args:
        data: New data to store
        cache_key: Direct cache key
        **kwargs: Cache key parameters
    
    Returns:
        bool: True if entry existed and was replaced, False otherwise
    """
    if cache_key is None:
        cache_key = self._create_cache_key(kwargs)
    
    # Check if exists
    if not self.metadata_backend.get_entry(cache_key):
        return False
    
    # Replace using existing put logic
    self.put(data, cache_key=cache_key, **kwargs)
    return True
```

---

### 3. **Bulk Delete by Pattern** (Medium Priority)

**Problem:** `invalidate()` deletes single entries. No bulk delete by pattern/prefix.

**Use Case:**
- Delete all experiments from a project
- Clear specific data types
- Remove old versions

**Proposed API:**

```python
# For cacheness
cache.delete_by_prefix(project="ml_models")  # All entries with project=ml_models
cache.delete_by_pattern(**{"experiment_*": "exp_"})  # Wildcard matching
cache.delete_matching(lambda meta: meta.get("version", "").startswith("v1"))

# For BlobStore (already has prefix support in list())
deleted = store.delete_by_prefix(prefix="old_models/")  # Returns count
```

**Implementation:**
```python
def delete_by_prefix(self, **prefix_kwargs) -> int:
    """
    Delete all entries matching prefix parameters.
    
    Args:
        **prefix_kwargs: Cache key parameters to match
    
    Returns:
        int: Number of entries deleted
    """
    # Query matching entries
    entries = self.query_meta(**prefix_kwargs)
    
    deleted_count = 0
    for entry in entries:
        cache_key = entry.get("cache_key")
        if cache_key and self.metadata_backend.remove_entry(cache_key):
            deleted_count += 1
    
    return deleted_count
```

---

### 4. **Get Metadata Without Loading Data** (Low Priority)

**Problem:** `cacheness` doesn't expose `get_metadata()` like `BlobStore` does.

**Use Case:**
- Check metadata before deciding to load
- Inspect TTL/expiration without loading
- Query file size before download

**Proposed API:**

```python
# For cacheness
meta = cache.get_metadata(experiment="exp_001")
# Returns: {
#     "cache_key": "...",
#     "created_at": "...",
#     "file_size": 1024,
#     "data_type": "pandas_dataframe",
#     "expires_at": "...",
# }

# Already available in BlobStore
meta = store.get_metadata(key="model_v1")
```

**Implementation:**
```python
def get_metadata(self, cache_key: Optional[str] = None, **kwargs) -> Optional[Dict[str, Any]]:
    """
    Get entry metadata without loading data.
    
    Args:
        cache_key: Direct cache key
        **kwargs: Cache key parameters
    
    Returns:
        Metadata dictionary or None if not found
    """
    if cache_key is None:
        cache_key = self._create_cache_key(kwargs)
    
    return self.metadata_backend.get_entry(cache_key)
```

---

### 5. **Export/Import Cache** (Low Priority)

**Problem:** No way to backup/restore cache or migrate between environments.

**Use Case:**
- Backup cache state
- Share cache between team members
- Migrate from dev to prod

**Proposed API:**

```python
# Export
cache.export_to_file("cache_backup.tar.gz", compress=True)
cache.export_to_dict()  # Returns serializable dict

# Import
cache.import_from_file("cache_backup.tar.gz")
cache.import_from_dict(data)

# Selective export
cache.export_to_file("ml_models.tar.gz", 
                     filter_fn=lambda meta: meta.get("project") == "ml")
```

---

### 6. **Verify and Repair Cache** (Low Priority)

**Problem:** No tools to detect corrupted entries or missing files.

**Use Case:**
- Detect broken cache entries
- Clean up orphaned files
- Verify integrity after crashes

**Proposed API:**

```python
# Verify cache integrity
issues = cache.verify_integrity()
# Returns: [
#     {"cache_key": "...", "issue": "missing_file"},
#     {"cache_key": "...", "issue": "corrupted_signature"},
# ]

# Repair cache
cache.repair(dry_run=True)  # Show what would be fixed
cache.repair()  # Actually fix issues

# Detect orphaned files
orphaned = cache.find_orphaned_files()  # Files without metadata entries
cache.cleanup_orphaned_files()
```

---

### 7. **Batch Operations** (Low Priority)

**Problem:** No efficient way to operate on multiple entries at once.

**Use Case:**
- Load multiple related entries
- Update metadata for many entries
- Bulk touch/refresh

**Proposed API:**

```python
# Batch get
results = cache.get_batch([
    {"experiment": "exp_001"},
    {"experiment": "exp_002"},
    {"experiment": "exp_003"},
])
# Returns: {
#     "exp_001": data1,
#     "exp_002": data2,
#     "exp_003": None,  # Not found
# }

# Batch touch
cache.touch_batch(project="ml_models")  # All matching entries

# Batch update metadata
cache.update_metadata_batch(
    filters={"project": "ml_models"},
    metadata={"reviewed": True}
)
```

---

### 8. **Copy/Clone Entries** (Very Low Priority)

**Problem:** No way to duplicate cache entries with different keys.

**Use Case:**
- Create entry variants
- Backup before modification
- Fork experiments

**Proposed API:**

```python
# Copy entry
cache.copy(
    source={"experiment": "exp_001"},
    dest={"experiment": "exp_001_backup"}
)

# Clone with modifications
cache.clone(
    source={"experiment": "exp_001"},
    dest={"experiment": "exp_002"},
    update_metadata={"cloned_from": "exp_001"}
)
```

---

---

## Priority Recommendations (Revised by Layer)

### **Storage Backend Layer - High Priority**
1. ✅ **Update Blob Data** - Replace data at existing cache_key (core CRUD operation)
2. ✅ **Bulk Delete by Pattern** - Essential for cleanup operations
3. ✅ **Expose Get Metadata** - Already in backend, just need to expose in cache

### **Cache Layer - High Priority**
4. ✅ **Touch/Refresh TTL** - Core cache operation for extending expiration

### **Storage Backend Layer - Medium Priority**
5. **Batch Operations** - Performance optimization

### **Storage Backend Layer - Low Priority (Convenience)**
6. **Copy/Move Entries** - Convenience wrappers around CRUD (users can compose directly)

### **Cross-Layer - Low Priority**
7. Export/Import Cache
8. Verify and Repair

---

## Architectural Summary

### Correct Separation of Concerns:

**Storage Backends Should Implement:**
- ✅ CRUD operations (create, read, update, delete) - *update blob data missing*
- ✅ Bulk operations (delete many, batch get/delete) - *bulk delete missing*
- ✅ Metadata access (get without loading data) - *exists but not exposed*
- ✅ Pattern-based operations (prefix matching)
- ❌ **NO TTL logic** - just store/retrieve timestamps
- ⚠️ **Cache keys are immutable** - computed from input params, not content

**Cache Layer Should Implement:**
- ✅ TTL policy and expiration checking - *already done*
- ✅ Touch/refresh operations - *missing*
- ✅ Cache hit/miss with TTL validation - *already done*
- ✅ Cleanup coordination (calls backend.cleanup_expired with TTL) - *already done*
- ✅ Convenience wrappers for backend operations

### Why This Separation Matters:

1. **Storage backends** can be used without caching (e.g., `BlobStore` for model versioning)
2. **Storage backends** don't make policy decisions (like "what TTL to use")
3. **Cache layer** provides caching semantics on top of storage
4. **Reusability** - storage operations useful in non-cache contexts

---

## Implementation Status (Revised)

### Storage Backend (MetadataBackend, BlobStore)
- [ ] Add `update_blob_data(cache_key, new_data)` - replace data at existing key
- [ ] Add `delete_by_prefix()` / `delete_where()`
- [x] Expose `get_entry()` in cache layer (already exists in backend)
- [ ] Add `get_entries_batch()`
- [ ] Add `delete_entries_batch()`
- [ ] Add `copy_entry()` / `move_entry()` - **convenience methods** (built on CRUD: get + put [+ delete])

### Cache Layer (UnifiedCache)
- [ ] Add `touch()` method with TTL parameter
- [ ] Add `get_metadata()` method (delegates to backend)
- [ ] Add convenience wrappers for bulk operations
- [ ] Add `update_data()` method (delegates to backend.update_blob_data())

### Cross-Layer
- [ ] Export/Import functionality
- [ ] Verify and Repair functionality

---

## Notes

**Correct Architectural Pattern:**

```python
# Storage Backend - Dumb storage, no TTL awareness
class MetadataBackend(ABC):
    @abstractmethod
    def get_entry(cache_key) -> Optional[Dict]:
        """Retrieve entry - returns timestamp, but doesn't interpret it."""
        pass
    
    @abstractmethod
    def update_blob_data(cache_key, new_data):
        """Replace blob data at existing cache_key (key stays same)."""
        pass
    
    @abstractmethod
    def update_metadata(cache_key, metadata_updates):
        """Update metadata fields (not key-affecting params)."""
        pass
    
    @abstractmethod
    def delete_by_prefix(prefix) -> int:
        """Delete entries matching prefix."""
        pass

# Cache Layer - Smart caching with TTL policy
class UnifiedCache:
    def get(self, **kwargs):
        entry = self.backend.get_entry(cache_key)
        if self._is_expired(entry):  # Cache decides expiration
            return None
        return self._load_data(entry)
    
    def touch(self, **kwargs):
        """Cache operation - reset expiration."""
        entry = self.backend.get_entry(cache_key)
        entry["created_at"] = now()  # Update timestamp
        self.backend.update_metadata(cache_key, entry)  # Update metadata only
    
    def cleanup_expired(self):
        """Cache operation - remove expired entries."""
        ttl = self.config.metadata.default_ttl_seconds  # Cache policy
        self.backend.cleanup_expired(ttl)  # Backend does deletion
```

**Why `cleanup_expired()` is in Backend:**
- Backend must delete by timestamp (knows how to delete entries)
- Cache layer provides the TTL policy (what timestamp cutoff to use)
- Separation: backend has deletion mechanism, cache has expiration policy

**Key Insights:** 
1. Storage backends store timestamps but don't interpret them. Cache layer interprets timestamps according to TTL policy.
2. **Cache keys are immutable and ENFORCED** - they're computed from input parameters, not content. You can update the data AT a cache_key, but you can't change the cache_key itself.
   - **Why enforce?** Prevents data corruption, matches industry standards (Redis, S3, DynamoDB), prevents user errors.
   - **How to "change" a key?** Use CRUD operations (`get + put + delete`) or convenience wrappers (`copy()`/`move()`).
3. `update_blob_data(cache_key, new_data)` replaces data at a fixed key - it updates derived metadata (file_size, content_hash) but the cache_key stays the same.
4. **Copy/Move are convenience operations** - built internally using CRUD primitives (get + put + delete). Users can always compose these operations directly, but convenience methods provide atomicity and efficiency benefits.

