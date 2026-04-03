# API Reference

Complete reference for all cacheness classes, methods, and configuration options.

## Storage Layer

### `BlobStore`

Low-level blob storage API without caching semantics (TTL, eviction). Useful for
ML model versioning, artifact storage, and data pipeline checkpoints.

`BlobStore` also serves as the internal storage engine for `UnifiedCache` — all
file I/O, handler dispatch, integrity verification, and signing are delegated to
it via a shared composition model (same metadata backend, handlers, lock, and
signer).

```python
from cacheness.storage import BlobStore

class BlobStore:
    def __init__(
        self,
        cache_dir: str = ".blobstore",
        backend: Optional[Union[str, MetadataBackend]] = None,
        compression: str = "lz4",
        compression_level: int = 3,
        content_addressable: bool = False
    )
```

**Parameters:**
- `cache_dir` (str): Directory for storing blobs and metadata
- `backend` (str|MetadataBackend): Backend type ("json", "sqlite") or instance
- `compression` (str): Compression codec (lz4, zstd, gzip, blosclz)
- `compression_level` (int): Compression level (1-9)
- `content_addressable` (bool): If True, use content hash as blob key

#### Methods

##### `put(data, key=None, metadata=None) -> str`
Store a blob with optional metadata.

**Parameters:**
- `data` (Any): The data to store
- `key` (Optional[str]): Optional key for the blob (auto-generated if None)
- `metadata` (Optional[Dict]): Custom metadata to store with the blob

**Returns:**
- `str`: The blob key

**Example:**
```python
store = BlobStore(cache_dir="./models")
key = store.put(model, key="xgboost_v1", metadata={"accuracy": 0.95})
```

##### `get(key: str) -> Optional[Any]`
Retrieve a blob by key.

##### `get_metadata(key: str) -> Optional[Dict]`
Get blob metadata without loading content.

##### `update_metadata(key: str, metadata: Dict) -> bool`
Update metadata for an existing blob.

##### `delete(key: str) -> bool`
Delete a blob and its metadata.

##### `exists(key: str) -> bool`
Check if a blob exists.

##### `list(prefix=None, metadata_filter=None) -> List[str]`
List blob keys with optional filtering.

**Example:**
```python
# List all blobs
all_keys = store.list()

# Filter by prefix
model_keys = store.list(prefix="model_")

# Filter by metadata
v1_models = store.list(metadata_filter={"version": "1.0"})
```

##### `put_file(file_path, *, key=None, metadata=None, move=False) -> str`
Store an arbitrary file as a blob. Reads the file into bytes and delegates to `put()`. File metadata (original filename, MIME type, original size) is automatically merged into the metadata dict.

**Parameters:**
- `file_path` (str|Path): Path to the source file
- `key` (Optional[str]): Explicit blob key (auto-generated if None)
- `metadata` (Optional[Dict]): Custom metadata to merge with file metadata
- `move` (bool): If `True`, delete the source file after a successful store (move-in semantics). Defaults to `False` (copy-in).

**Returns:**
- `str`: The blob key

**Example:**
```python
store = BlobStore(cache_dir="./artifacts")

# Copy file into store (default)
key = store.put_file("models/xgboost.onnx", key="model-v2")

# Move file into store (source deleted after store)
key = store.put_file("data/output.csv", metadata={"project": "alpha"}, move=True)
```

##### `get_file(key, *, dest=None, move=False, overwrite=True) -> bytes | Path | None`
Retrieve a cached file blob, optionally writing it to disk.

**Parameters:**
- `key` (str): The blob key
- `dest` (Optional[str|Path]): Destination path. If a directory, the original filename from metadata is used (falls back to `<key>.bin`). Parent directories are created automatically.
- `move` (bool): If `True`, delete the cache entry after a successful write to `dest` (move-out semantics). Requires `dest` to be set. Defaults to `False` (copy-out).
- `overwrite` (bool): If `False`, raise `FileExistsError` when `dest` already exists on disk. Defaults to `True` (silently overwrite).

**Returns:**
- `bytes` when `dest` is `None` and entry exists
- `Path` when `dest` is given and entry exists
- `None` on miss

**Raises:**
- `ValueError`: If `move` is `True` but `dest` is `None`
- `FileExistsError`: If `overwrite` is `False` and `dest` already exists

**Example:**
```python
# Get raw bytes
data = store.get_file("model-v2")

# Copy out to a specific path
path = store.get_file("model-v2", dest="./restored/model.onnx")

# Move out (write + delete entry)
path = store.get_file("model-v2", dest="./restored/model.onnx", move=True)

# Safe write (no overwrite)
path = store.get_file("model-v2", dest="./restored/model.onnx", overwrite=False)
```

##### `verify_integrity(key: str) -> Dict`
Verify blob integrity by checking file hash against stored metadata.

**Returns:** Dict with `valid` (bool), `expected_hash`, `actual_hash` keys.

##### `clear() -> int`
Remove all blobs and their data files.

##### `close()`
Close the blob store and release resources.

**Example:**
```python
# Context manager usage (recommended)
with BlobStore(cache_dir="./artifacts") as store:
    store.put(data, key="artifact_1")
    result = store.get("artifact_1")
```

---

## Core Classes

### `cacheness` Class

The main cache interface providing key-value storage with intelligent compression and serialization.

```python
class cacheness:
    def __init__(self, config: Optional[CacheConfig] = None)
```

#### Methods

##### `put(data, **cache_key_params) -> str`
Store data in the cache with optional metadata.

**Parameters:**
- `data` (Any): The data to cache
- `**cache_key_params`: Key-value pairs used to generate the cache key
- `metadata` (Optional[dict]): Custom metadata to store with the entry
- `ttl` (Optional[str|int|float]): Per-entry TTL — duration string (`"1h"`, `"7d"`) or seconds. Overrides `default_ttl` for this entry.
- `ttl_seconds` (Optional[float]): Legacy alias for `ttl`
- `description` (Optional[str]): Human-readable description for the entry

**Returns:**
- `str`: The generated cache key

**Example:**
```python
cache = cacheness()
key = cache.put(
    {"results": [1, 2, 3]}, 
    experiment="exp_001", 
    model="xgboost",
    metadata={"accuracy": 0.95}
)
```

##### `get(**cache_key_params) -> Optional[Any]`
Retrieve data from the cache.

**Parameters:**
- `**cache_key_params`: Key-value pairs used to generate the cache key

**Returns:**
- The cached data if found and not expired, otherwise `None`

**Example:**
```python
result = cache.get(experiment="exp_001", model="xgboost")
if result is not None:
    print("Cache hit!")
```

##### `get_with_metadata(**cache_key_params) -> Optional[Tuple[Any, dict]]`
Retrieve data along with its metadata.

**Parameters:**
- `**cache_key_params`: Key-value pairs used to generate the cache key

**Returns:**
- Tuple of (data, metadata) if found, otherwise `None`

**Example:**
```python
result = cache.get_with_metadata(experiment="exp_001", model="xgboost")
if result:
    data, metadata = result
    print(f"Accuracy: {metadata.get('accuracy')}")
```

##### `exists(**cache_key_params) -> bool`
Check if an entry exists in the cache.

**Parameters:**
- `**cache_key_params`: Key-value pairs used to generate the cache key

**Returns:**
- `bool`: True if the entry exists and is not expired

##### `invalidate(**cache_key_params) -> bool`
Remove an entry from the cache.

**Parameters:**
- `**cache_key_params`: Key-value pairs used to generate the cache key

**Returns:**
- `bool`: True if the entry was found and removed

##### `invalidate_by_cache_key(cache_key: str) -> bool`
Remove an entry by its cache key.

**Parameters:**
- `cache_key` (str): The cache key to remove

**Returns:**
- `bool`: True if the entry was found and removed

##### `list_entries(include_metadata: bool = False) -> EntryList`
List all cache entries.

**Parameters:**
- `include_metadata` (bool): Whether to include custom metadata in results

**Returns:**
- `EntryList` — a `list` subclass with convenience methods:
  - `.keys()` — list of cache keys
  - `.first()` / `.last()` — first/last entry (or `None`)
  - `.sort_by(field, reverse=False)` — return sorted `EntryList`
  - `.filter(predicate)` — return filtered `EntryList`
  - `.to_json(path=None)` — serialize to JSON string or file
  - `.to_dataframe()` — convert to pandas DataFrame (requires pandas)

**Example:**
```python
entries = cache.list_entries(include_metadata=True)

# EntryList convenience methods
keys = entries.keys()              # ["key1", "key2", ...]
first = entries.first()            # First entry dict or None
sorted_ = entries.sort_by("file_size", reverse=True)
recent = entries.filter(lambda e: e.get("data_type") == "dataframe")

# Serialization
entries.to_json("entries.json")     # Save to file
df = entries.to_dataframe()         # pandas DataFrame
```

##### `get_stats() -> dict`
Get cache statistics.

**Returns:**
- Dictionary with cache statistics including total entries, size, hit rate, etc.

##### `query_custom(table_name: str, filters: Optional[Dict] = None) -> List[Any]`
Query custom metadata entries from SQLite backend. Returns results directly with automatic session cleanup.

**Parameters:**
- `table_name` (str): Name of the custom metadata table
- `filters` (Optional[Dict]): Simple equality filters to apply

**Returns:**
- `List`: List of matching entries

**Example:**
```python
# Simple query - session automatically cleaned up
results = cache.query_custom("ml_experiments")

# With filters
results = cache.query_custom("ml_experiments", {"model_type": "xgboost"})
```

##### `query_custom_session(table_name: str) -> ContextManager`
Context manager for advanced custom metadata queries with SQLAlchemy. Guarantees session cleanup.

**Parameters:**
- `table_name` (str): Name of the custom metadata table

**Returns:**
- Context manager yielding a SQLAlchemy Query object

**Example:**
```python
# Advanced filtering with context manager
with cache.query_custom_session("ml_experiments") as query:
    high_accuracy = query.filter(MLExperimentMetadata.accuracy >= 0.9).all()
    sorted_results = query.order_by(MLExperimentMetadata.accuracy.desc()).limit(10).all()
```

##### `close()`
Close the cache and release resources. Called automatically if using context manager.

##### `verify_integrity(repair=False, verify_hashes=False) -> IntegrityReport`
Verify cache integrity by cross-checking blob files and metadata entries.

**Parameters:**
- `repair` (bool): If `True`, delete orphaned blobs and remove dangling metadata entries
- `verify_hashes` (bool): If `True`, verify file hashes (slower but catches corruption). For S3 blobs, uses cheap HEAD/ETag check.

**Returns:**
- `IntegrityReport` with attributes:
  - `.orphaned_blobs` — blob files with no metadata entry
  - `.dangling_entries` — metadata entries pointing to missing blobs
  - `.size_mismatches` — entries where `file_size` != actual size
  - `.hash_mismatches` — entries where `file_hash` != actual hash (only when `verify_hashes=True`)
  - `.repaired` — dict with `orphans_deleted` and `dangling_removed` counts (only when `repair=True`)

Supports dict-style access for backward compatibility: `report["orphaned_blobs"]`.

**Example:**
```python
# Detect issues
report = cache.verify_integrity()
print(f"Orphaned: {len(report.orphaned_blobs)}, Dangling: {len(report.dangling_entries)}")

# Detect and repair
report = cache.verify_integrity(repair=True)
print(f"Cleaned {report.repaired['orphans_deleted']} orphans")

# Full check including hash verification
report = cache.verify_integrity(repair=True, verify_hashes=True)
```

**Example:**
```python
# Manual close
cache = cacheness()
try:
    cache.put(data, key="example")
finally:
    cache.close()

# Or use context manager (recommended)
with cacheness() as cache:
    cache.put(data, key="example")
```

#### Factory Methods

##### `cacheness.for_api(cache_dir=None, ttl="6h", **kwargs)`
Create a cache instance optimized for API requests.

**Parameters:**
- `cache_dir` (Optional[str]): Cache directory (default: "./cache")
- `ttl` (Optional[str]): TTL as a duration string (e.g., `"6h"`, `"30m"`) — default: `"6h"`
- `ttl_seconds` (Optional[float]): TTL in seconds (mutually exclusive with `ttl`)
- `**kwargs`: Additional configuration options

**Returns:**
- `cacheness`: Configured cache instance with LZ4 compression for fast JSON/text caching

**Example:**
```python
api_cache = cacheness.for_api(cache_dir="./api_cache", ttl="4h")
api_cache.put({"users": [...]}, endpoint="users", version="v1")
```

##### `cleanup_expired() -> int`
Remove all expired entries from the cache.

**Returns:**
- `int`: Number of entries removed

##### `clear() -> int`
Remove all entries from the cache.

**Returns:**
- `int`: Number of entries removed

#### Management Operations

These methods provide fine-grained control over cache entries — inspecting metadata, updating stored data in-place, refreshing TTLs, and performing bulk/batch operations.

##### `get_metadata(cache_key=None, check_expiration=True, **kwargs) -> Optional[Dict]`
Get entry metadata without loading blob data.

Useful for inspecting cache entries (TTL, file size, data type) before deciding whether to load the actual data.

**Parameters:**
- `cache_key` (Optional[str]): Direct cache key (if provided, `**kwargs` are ignored)
- `check_expiration` (bool): If True, returns None for expired entries (default: True)
- `**kwargs`: Parameters identifying the cached data (used if `cache_key` is None)

**Returns:**
- Metadata dictionary (including `cache_key`, `data_type`, `file_size`, `created_at`, etc.) or `None` if not found/expired

**Example:**
```python
# Check metadata before loading large file
meta = cache.get_metadata(experiment="exp_001")
if meta and meta.get("file_size", 0) > 1e9:
    print("Large file — loading may take time")
    data = cache.get(experiment="exp_001")
```

##### `update_data(data, cache_key=None, **kwargs) -> bool`
Replace blob data at an existing cache entry without changing the cache key.

Updates derived metadata (file_size, content_hash, timestamp) and re-signs the entry if signing is enabled. The cache key remains unchanged to maintain referential integrity.

**Parameters:**
- `data` (Any): New data to store (must be serializable by handler)
- `cache_key` (Optional[str]): Direct cache key (if provided, `**kwargs` are ignored)
- `**kwargs`: Parameters identifying the cached data (used if `cache_key` is None)

**Returns:**
- `bool`: True if entry was updated, False if entry doesn't exist

**Example:**
```python
# Update cached DataFrame with new data
success = cache.update_data(
    new_df,
    experiment="exp_001",
    run_id=42
)

if not success:
    print("Entry not found — use put() to create new entry")
```

> **Note:** Cache keys are immutable and derived from input params, not content. Use `update_data()` to refresh data at the same logical location. Use `put()` to create new entries.

##### `touch(cache_key=None, **kwargs) -> bool`
Update entry timestamp to extend TTL without reloading data.

Resets the creation timestamp to now, effectively giving the entry a full config-TTL extension. Useful for keeping frequently accessed data alive or preventing expiration of long-running computations.

**Parameters:**
- `cache_key` (Optional[str]): Direct cache key (if provided, `**kwargs` are ignored)
- `**kwargs`: Parameters identifying the cached data (used if `cache_key` is None)

**Returns:**
- `bool`: True if entry exists and was touched, False if entry doesn't exist

**Example:**
```python
# Reset TTL to default
cache.touch(experiment="exp_001")

# Keep long-running computation alive
for i in range(100):
    process_chunk(i)
    if i % 10 == 0:
        cache.touch(job_id="long_job")  # Prevent expiration
```

##### `delete_where(filter_fn) -> int`
Delete all cache entries matching a filter function.

Iterates over every entry and deletes those for which `filter_fn` returns `True`. Works with all backends.

**Parameters:**
- `filter_fn` (Callable[[Dict], bool]): Receives an entry dict and returns True if the entry should be deleted. Each dict contains at least `cache_key`, `data_type`, `description`, `metadata`, `created`, `last_accessed`, and `size_mb`.

**Returns:**
- `int`: Number of entries deleted

**Example:**
```python
from datetime import datetime, timezone, timedelta

# Delete all entries older than 7 days
cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
deleted = cache.delete_where(
    lambda e: (e.get("created") or "") < cutoff
)

# Delete all DataFrames
deleted = cache.delete_where(
    lambda e: e.get("data_type") == "dataframe"
)
```

##### `delete_matching(**kwargs) -> int`
Delete all entries whose metadata contains the given key/value pairs.

Convenience wrapper around `delete_where()`. For SQLite with `store_full_metadata=True`, uses indexed `query_meta()` for fast lookups.

**Parameters:**
- `**kwargs`: Key-value pairs to match. An entry is deleted when **all** pairs match.

**Returns:**
- `int`: Number of entries deleted

**Example:**
```python
# Delete all entries for a specific project
deleted = cache.delete_matching(project="ml_models")

# Delete entries matching multiple criteria
deleted = cache.delete_matching(
    experiment="exp_001",
    model_type="xgboost"
)
```

##### `get_batch(kwargs_list) -> Dict[str, Any]`
Get multiple cache entries in one call.

**Parameters:**
- `kwargs_list` (List[Dict]): List of kwarg dicts, each identifying one entry (same parameters you would pass to `get()`).

**Returns:**
- `dict`: Mapping of cache_key → data (or `None` if not found/expired)

**Example:**
```python
results = cache.get_batch([
    {"experiment": "exp_001"},
    {"experiment": "exp_002"},
    {"experiment": "exp_003"},
])
for key, data in results.items():
    if data is not None:
        print(f"{key}: loaded")
```

##### `delete_batch(kwargs_list) -> int`
Delete multiple cache entries in one call.

**Parameters:**
- `kwargs_list` (List[Dict]): List of kwarg dicts, each identifying one entry (same parameters you would pass to `invalidate()`).

**Returns:**
- `int`: Number of entries that were actually deleted (existed)

**Example:**
```python
deleted = cache.delete_batch([
    {"experiment": "exp_001"},
    {"experiment": "exp_002"},
])
print(f"Removed {deleted} entries")
```

##### `put_file(file_path, *, cache_key=None, on=None, description="", custom_metadata=None, move=False, **kwargs) -> str`
Store an arbitrary file in the cache. Reads the file into bytes and delegates to `put()`. File metadata (original filename, MIME type, file size) is automatically recorded in `metadata_dict` when `store_full_metadata=True`.

File metadata does **not** participate in cache key derivation — only `on` and `**kwargs` determine the key.

**Parameters:**
- `file_path` (str|Path): Path to the source file
- `cache_key` (Optional[str]): Explicit cache key (if provided, `on` and `**kwargs` are ignored for key derivation)
- `on` (Optional[Dict]): Dictionary of key parameters for cache key derivation
- `description` (str): Human-readable description
- `custom_metadata`: Custom metadata for the cache entry. Supports single ORM objects, lists/tuples of ORM objects, or dicts. Passed through to `put()` unchanged.
- `move` (bool): If `True`, delete the source file after a successful store (move-in semantics). Defaults to `False` (copy-in).
- `**kwargs`: Extra key-value pairs for key derivation and/or `metadata_dict`

**Returns:**
- `str`: 16-character hex cache key

**Example:**
```python
# Copy file into cache (default)
key = cache.put_file("data/model.onnx", description="ONNX model v2")

# Move file into cache (source deleted after store)
key = cache.put_file("output.csv", on={"run": "exp_01"}, move=True)

# Store with explicit key
key = cache.put_file("report.pdf", cache_key="monthly_report_jan")

# Query by auto-populated metadata
results = cache.query_meta(mime_type="application/pdf")
```

##### `get_file(cache_key=None, *, dest=None, on=None, ttl=None, ttl_seconds=None, move=False, overwrite=True, **kwargs) -> bytes | Path | None`
Retrieve cached file data, optionally writing it to disk. Read counterpart of `put_file()`.

**Parameters:**
- `cache_key` (Optional[str]): Explicit cache key
- `dest` (Optional[str|Path]): Destination path. If a directory, the original filename from metadata is used (falls back to `<cache_key>.bin`). Parent directories are created automatically.
- `on` (Optional[Dict]): Dictionary of key parameters for key lookup
- `ttl` (Optional[str]): TTL as duration string (e.g. `"6h"`)
- `ttl_seconds` (Optional[float]): TTL in seconds
- `move` (bool): If `True`, delete the cache entry after a successful write to `dest` (move-out semantics). Requires `dest` to be set. Defaults to `False` (copy-out).
- `overwrite` (bool): If `False`, raise `FileExistsError` when `dest` already exists on disk. Defaults to `True` (silently overwrite).
- `**kwargs`: Key parameters for key lookup (must match `put_file()` call)

**Returns:**
- `bytes` when `dest` is `None` and entry exists
- `Path` when `dest` is given and entry exists
- `None` on cache miss

**Raises:**
- `TypeError`: If the cached entry is not bytes (e.g. stored via `put()` with a dict)
- `ValueError`: If `move` is `True` but `dest` is `None`
- `FileExistsError`: If `overwrite` is `False` and `dest` already exists

**Example:**
```python
# Get raw bytes
data = cache.get_file(cache_key=key)

# Copy out to a specific file
path = cache.get_file(cache_key=key, dest="output/model.onnx")

# Move out (write + delete cache entry)
path = cache.get_file(cache_key=key, dest="output/model.onnx", move=True)

# Safe write (no overwrite)
path = cache.get_file(cache_key=key, dest="output/", overwrite=False)

# Retrieve using on= key params
data = cache.get_file(on={"run": "exp_01"})
```

##### `touch_batch(**filter_kwargs) -> int`
Touch (refresh TTL of) all cache entries whose metadata matches the given key/value pairs.

**Parameters:**
- `**filter_kwargs`: Key-value pairs to match against entry metadata.

**Returns:**
- `int`: Number of entries touched

**Example:**
```python
# Extend TTL for all entries in a project
touched = cache.touch_batch(project="ml_models")
```

#### Convenience Metadata Helpers

These helpers eliminate the common pattern of passing the same values twice — once for key derivation and once for metadata storage. All accept an optional `on=` keyword for key-only discriminators that participate in cache key derivation but are **not** stored as metadata.

> **Requirement:** `put_with_meta` and `get_with_meta` require `store_full_metadata=True` in `CacheConfig`.  `put_with_model` and `get_with_model` require a SQLite or PostgreSQL metadata backend.

##### `put_with_meta(data, *, on=None, description="", **kwargs) -> str`
Store data using `**kwargs` as both the cache key **and** `metadata_dict`.

**Parameters:**
- `data`: Data to cache.
- `on`: Extra key-only parameters that affect the cache key but are not stored in metadata.
- `description`: Human-readable description (not part of the cache key).
- `**kwargs`: Key-value pairs used for both key derivation and `metadata_dict` storage.

**Returns:** 16-character hex cache key.

**Example:**
```python
cache.put_with_meta(df, experiment="exp_001", model="xgboost", accuracy=0.95)

# on= creates distinct entries with identical metadata
cache.put_with_meta(df, on={"epoch": 5}, model="xgboost", lr=0.01)
```

##### `get_with_meta(*, on=None, ttl=None, ttl_seconds=None, **kwargs) -> Optional[tuple[Any, dict]]`
Retrieve data and its `metadata_dict` by the same kwargs used at store time.

**Returns:** `(data, metadata_dict)` on a hit, `None` on a miss. `metadata_dict` is the plain dict of the originally stored kwargs.

**Example:**
```python
result = cache.get_with_meta(experiment="exp_001", model="xgboost")
if result:
    data, meta = result
    print(meta["accuracy"])  # 0.95
```

##### `put_with_model(data, model_class, *, on=None, description="", **kwargs) -> str`
Store data using `**kwargs` as both cache key and ORM custom metadata instance.

**Parameters:**
- `data`: Data to cache.
- `model_class`: A custom metadata model class decorated with `@custom_metadata_model`.
- `on`: Extra key-only discriminator parameters.
- `**kwargs`: Values passed to `model_class(...)` and used for cache key derivation.

**Example:**
```python
cache.put_with_model(df, ExperimentMetadata,
                     experiment_id="exp_001",
                     model_type="xgboost",
                     accuracy=0.95)

# With key discriminator for distinct entries sharing the same ORM metadata:
cache.put_with_model(df, ExperimentMetadata,
                     on={"run_id": "run_42"},
                     experiment_id="exp_001",
                     model_type="xgboost",
                     accuracy=0.95)
```

##### `get_with_model(model_class, *, on=None, ttl=None, ttl_seconds=None, **kwargs) -> Optional[tuple[Any, Any]]`
Retrieve data and its ORM metadata instance.

**Returns:** `(data, orm_instance)` on a hit, `None` on a miss or if no ORM row exists.

**Example:**
```python
result = cache.get_with_model(ExperimentMetadata,
                               experiment_id="exp_001",
                               model_type="xgboost")
if result:
    data, exp = result
    print(exp.accuracy)
```

##### `query_with_meta(**kwargs) -> Iterator[tuple[Any, dict]]`
Lazily yield `(data, metadata_dict)` tuples for all entries whose `metadata_dict` matches the given filters.

**Example:**
```python
for data, meta in cache.query_with_meta(model="xgboost"):
    print(meta["experiment_id"], data.shape)
```

---

#### Dunder Methods

`UnifiedCache` supports Python dunder methods for idiomatic cache interaction.

##### `len(cache)` — `__len__`
Return the number of entries in the cache.

```python
cache = cacheness()
cache.put("a", key="x")
cache.put("b", key="y")
print(len(cache))  # 2
```

##### `key in cache` — `__contains__`
Check whether a cache key exists (not expired).

```python
if "my_key" in cache:
    data = cache.get(cache_key="my_key")
```

##### `list(cache)` / `for key in cache` — `__iter__`
Iterate over all non-expired cache keys.

```python
for key in cache:
    print(key)

all_keys = list(cache)  # ["key1", "key2", ...]
```

---

## Typed Contracts

Cacheness uses typed dataclasses for return values across internal and public APIs. These provide IDE autocompletion, named field access, and dict-compatible accessors for backward compatibility.

### `HandlerResult`

Returned by handler `put()` methods. Replaces the legacy untyped `Dict[str, Any]`.

```python
from cacheness import HandlerResult

@dataclass
class HandlerResult:
    storage_format: str              # e.g. "parquet", "blosc2", "pickle"
    file_size: int                   # Size in bytes
    actual_path: str                 # Relative path to the blob file
    compression_codec: Optional[str] # e.g. "lz4", "zstd"
    serializer: Optional[str]       # e.g. "pickle", "dill"
    object_type: Optional[str]      # e.g. "pandas.DataFrame"
    extra: Dict[str, Any]           # Handler-specific metadata (shape, dtypes, etc.)
```

Provides dict-compatible accessors (`result["storage_format"]`, `"actual_path" in result`) for transitional use.

### `EntryList`

Returned by `list_entries()` and `query_meta()`. A `list` subclass with convenience methods.

```python
from cacheness import EntryList

entries: EntryList = cache.list_entries()
entries.keys()                          # ["key1", "key2", ...]
entries.sort_by("file_size", reverse=True)  # Sorted EntryList
entries.filter(lambda e: e["data_type"] == "dataframe")  # Filtered EntryList
entries.to_dataframe()                  # pandas DataFrame
entries.to_json("entries.json")         # Save to file
entries.first()                         # First entry or None
```

### `IntegrityReport`

Returned by `verify_integrity()`. See [verify_integrity](#verify_integrityrepairfalse-verify_hashesfalse---integrityreport) above.

```python
from cacheness import IntegrityReport

report: IntegrityReport = cache.verify_integrity(repair=True, verify_hashes=True)
report.orphaned_blobs      # List[str]
report.dangling_entries     # List[Dict]
report.size_mismatches      # List[Dict]
report.hash_mismatches      # Optional[List[Dict]] (when verify_hashes=True)
report.repaired             # Optional[Dict] (when repair=True)
```

### `EntryData` (TypedDict)

Canonical shape for the dict returned by `MetadataBackend.get_entry()` and accepted by `put_entry()`. All metadata backends (JSON, SQLite, PostgreSQL) produce and consume dicts conforming to this contract.

```python
from cacheness import EntryData

class EntryData(TypedDict, total=False):
    # Always present from all backends
    description: str
    data_type: str
    created_at: Any       # ISO str or float timestamp depending on backend
    accessed_at: Any
    file_size: int
    metadata: Dict[str, Any]  # Nested handler/storage metadata

    # Present in some backends
    cache_key: str        # PostgreSQL includes this; JSON/SQLite do not
```

The nested `metadata` dict contains handler-written fields (`actual_path`, `storage_format`, `file_hash`, `entry_signature`, handler extras). Exact keys vary by handler type.

Primarily relevant for custom metadata backend authors implementing `get_entry()`/`put_entry()`.

### `HooksConfig` (dataclass)

Lifecycle callback configuration for cache events. See [Configuration Guide — Hooks](CONFIGURATION.md#hooks-configuration-hooksconfig) for full details.

```python
from cacheness import HooksConfig

@dataclass
class HooksConfig:
    on_evict: Optional[Callable] = None
    on_integrity_failure: Optional[Callable] = None
```

### `WriteBlobResult`

Internal typed contract returned by `BlobStore._write_blob()`. Replaces the unnamed `tuple[handler, result, file_hash]`.

```python
from cacheness import WriteBlobResult

@dataclass
class WriteBlobResult:
    handler: CacheHandler        # The handler that serialized the data
    result: HandlerResult        # Serialization metadata
    file_hash: Optional[str]     # xxhash of the blob file (if computed)
```

Primarily relevant for plugin/handler developers and internal composition.

---

## Custom Metadata

Custom metadata allows you to define typed, queryable SQLAlchemy ORM models that link to cache entries. This provides structured metadata storage with advanced querying capabilities using SQLAlchemy ORM.

### `@custom_metadata_model` Decorator

Decorator for registering custom SQLAlchemy metadata models.

```python
from cacheness.custom_metadata import custom_metadata_model, CustomMetadataBase
from cacheness.metadata import Base
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean

@custom_metadata_model("experiments")
class ExperimentMetadata(Base, CustomMetadataBase):
    """Custom metadata for ML experiments."""
    
    __tablename__ = "custom_ml_experiments"
    
    # cache_key FK inherited from CustomMetadataBase
    experiment_id = Column(String(100), nullable=False, unique=True, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    accuracy = Column(Float, nullable=False, index=True)
    epochs = Column(Integer, nullable=False, index=True)
    is_production = Column(Boolean, default=False)
    created_at = Column(DateTime)
```

**Parameters:**
- `schema_name` (str): Unique name for this metadata schema (used in queries)

**Requirements:**
- Must inherit from both `Base` and `CustomMetadataBase`
- Table name should start with `custom_` prefix (validated with warning)
- Add indexes to frequently queried columns for performance

### `CustomMetadataBase` Mixin

Base class that provides the `cache_key` foreign key for custom metadata models.

```python
class CustomMetadataBase:
    """
    Mixin that adds cache_key foreign key to custom metadata tables.
    
    Automatically adds:
    - cache_key: String(16), ForeignKey("cache_entries.cache_key", ondelete="CASCADE")
    - Indexed for fast lookups
    - Cascade delete ensures metadata is removed when cache entry deleted
    """
```

**Inherited Attributes:**
- `cache_key` (str): Foreign key to `cache_entries.cache_key` with cascade delete

### Storing Custom Metadata

Use the `custom_metadata` parameter when calling `cache.put()`:

```python
# Store with single custom metadata
experiment = ExperimentMetadata(
    experiment_id="exp_001",
    model_type="xgboost",
    accuracy=0.95,
    epochs=100
)

cache.put(
    trained_model,
    experiment="exp_001",
    custom_metadata=experiment
)

# Store with multiple custom metadata types
performance = PerformanceMetadata(
    run_id="run_001",
    training_time_seconds=120.5,
    memory_usage_mb=2048.0
)

cache.put(
    trained_model,
    experiment="exp_001",
    custom_metadata=[experiment, performance]  # List of metadata objects
)
```

### Querying Custom Metadata

#### `query_custom_session(schema_name) -> contextmanager`

Get a SQLAlchemy query session for a specific custom metadata schema.

**Parameters:**
- `schema_name` (str): The registered schema name from `@custom_metadata_model`

**Returns:**
- Context manager yielding SQLAlchemy Query object

**Example:**
```python
# Query with filters
with cache.query_custom_session("experiments") as query:
    high_accuracy = query.filter(
        ExperimentMetadata.accuracy >= 0.95
    ).all()
    
    for exp in high_accuracy:
        print(f"{exp.experiment_id}: {exp.accuracy}")

# Query with sorting
with cache.query_custom_session("experiments") as query:
    best_models = query.filter(
        ExperimentMetadata.is_production == True
    ).order_by(
        ExperimentMetadata.accuracy.desc()
    ).limit(10).all()

# Query with aggregation
from sqlalchemy import func

with cache.query_custom_session("experiments") as query:
    avg_accuracy = query.with_entities(
        func.avg(ExperimentMetadata.accuracy)
    ).scalar()
    print(f"Average accuracy: {avg_accuracy:.3f}")
```

#### `get_custom_metadata_for_entry(**kwargs) -> Dict[str, Any]`

Retrieve all custom metadata for a specific cache entry.

**Parameters:**
- `**kwargs`: Cache key parameters (same as used in `put()`)

**Returns:**
- `Dict[str, Any]`: Dictionary mapping schema names to metadata objects

**Example:**
```python
# Get all custom metadata for an entry
metadata = cache.get_custom_metadata_for_entry(experiment="exp_001")

if "experiments" in metadata:
    exp = metadata["experiments"]
    print(f"Model: {exp.model_type}, Accuracy: {exp.accuracy}")

if "performance" in metadata:
    perf = metadata["performance"]
    print(f"Training time: {perf.training_time_seconds}s")

# Returns empty dict if entry doesn't exist
meta = cache.get_custom_metadata_for_entry(nonexistent="key")
print(meta)  # {}
```

### Correlating Across Custom Tables

Use the `cache_key` foreign key to correlate data across multiple custom metadata tables:

```python
# Find high-accuracy experiments
with cache.query_custom_session("experiments") as query:
    high_acc_exps = query.filter(
        ExperimentMetadata.accuracy >= 0.95
    ).all()
    cache_keys = {exp.cache_key for exp in high_acc_exps}

# Get corresponding performance metrics
with cache.query_custom_session("performance") as query:
    perf_metrics = query.filter(
        PerformanceMetadata.cache_key.in_(cache_keys)
    ).all()
    
    for perf in perf_metrics:
        print(f"{perf.run_id}: {perf.training_time_seconds}s")
```

### Migrating Custom Metadata Tables

#### `migrate_custom_metadata_tables(engine=None)`

Create custom metadata tables in the database.

**Parameters:**
- `engine` (Optional[Engine]): SQLAlchemy engine. If None, uses backend's engine.

**Example:**
```python
from cacheness.custom_metadata import migrate_custom_metadata_tables

# With cache instance
cache = cacheness(config=config)
if hasattr(cache.metadata_backend, 'engine'):
    migrate_custom_metadata_tables(engine=cache.metadata_backend.engine)

# With explicit engine
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pass@localhost/db")
migrate_custom_metadata_tables(engine=engine)
```

### Cascade Deletion Behavior

Custom metadata records are **automatically cascade-deleted** when the parent cache entry is removed:

```python
# Store with custom metadata
cache.put(model, experiment="exp_001", custom_metadata=experiment)

# Delete cache entry
cache.invalidate(experiment="exp_001")

# Custom metadata automatically deleted (cascade)
meta = cache.get_custom_metadata_for_entry(experiment="exp_001")
print(meta)  # {} (empty)
```

**Benefits:**
- No orphaned metadata records
- Clear ownership semantics (metadata belongs to cache entry)
- Automatic cleanup - no manual maintenance needed
- Better cache isolation

### Backend Support

Custom metadata works with:
- **SQLite** (default) - single-file database, good for local development
- **PostgreSQL** - production-grade, supports high concurrency

Both backends work identically with custom metadata models.

**Example with PostgreSQL:**
```python
from cacheness import cacheness, CacheConfig
from cacheness.config import CacheMetadataConfig

config = CacheConfig(
    metadata=CacheMetadataConfig(
        metadata_backend="postgresql",
        metadata_backend_options={
            "connection_url": "postgresql://user:pass@localhost/cacheness_db"
        }
    )
)

cache = cacheness(config=config)
# Custom metadata works the same way
cache.put(model, custom_metadata=experiment)
```

### Best Practices

1. **Table Naming**: Use `custom_` prefix for table names
   ```python
   __tablename__ = "custom_ml_experiments"  # ✅ Good
   ```

2. **Add Indexes**: Index frequently queried columns
   ```python
   accuracy = Column(Float, index=True)  # ✅ Indexed
   ```

3. **Use Appropriate Types**: Choose correct SQLAlchemy column types
   ```python
   accuracy = Column(Float, index=True)        # Not String!
   epochs = Column(Integer, index=True)        # Not String!
   is_production = Column(Boolean)             # Not String!
   ```

4. **Separate Concerns**: Use multiple models for different metadata types
   ```python
   @custom_metadata_model("experiments")
   class ExperimentMetadata(Base, CustomMetadataBase):
       ...

   @custom_metadata_model("performance")
   class PerformanceMetadata(Base, CustomMetadataBase):
       ...
   ```

5. **Query Efficiently**: Use SQLAlchemy's filtering, ordering, and aggregation
   ```python
   with cache.query_custom_session("experiments") as query:
       results = query.filter(
           ExperimentMetadata.accuracy >= 0.9,
           ExperimentMetadata.created_by == "alice"
       ).order_by(ExperimentMetadata.accuracy.desc()).all()
   ```

---

## Decorators

### `@cached`

Decorator for caching function results with intelligent TTL management.

```python
def cached(
    cache_instance: Optional[cacheness] = None,
    ttl: Optional[str] = None,
    ttl_seconds: Optional[float] = None,
    cache_key_prefix: Optional[str] = None,
    include_defaults: bool = True,
    metadata: Optional[dict] = None
) -> Callable
```

**Parameters:**
- `cache_instance` (Optional[cacheness]): Cache instance to use (uses global if None)
- `ttl` (Optional[str]): TTL as a duration string (e.g., `"24h"`, `"30m"`)
- `ttl_seconds` (Optional[float]): TTL in seconds (mutually exclusive with `ttl`)
- `cache_key_prefix` (Optional[str]): Prefix for cache keys
- `include_defaults` (bool): Whether to include default parameter values in cache key
- `metadata` (Optional[dict]): Custom metadata to store with cached results

**Example:**
```python
@cached(ttl="24h", cache_key_prefix="weather")
def get_weather(city: str, units: str = "metric"):
    return fetch_weather_api(city, units)

# Function calls are automatically cached
weather = get_weather("London")  # Cache miss - calls API
weather = get_weather("London")  # Cache hit - returns cached result
```

### Factory Decorators

#### `@cached.for_api()`

Decorator optimized for API requests with error handling.

```python
@cached.for_api(ttl="6h", ignore_errors=True)
def fetch_user_data(user_id):
    response = requests.get(f"/api/users/{user_id}")
    return response.json()
```

**Parameters:**
- `ttl` (Optional[str]): TTL as a duration string (e.g., `"6h"`) — default: `"6h"`
- `ttl_seconds` (Optional[float]): TTL in seconds (mutually exclusive with `ttl`; default: 21600)
- `ignore_errors` (bool): Continue on cache errors (default: True)
- Uses LZ4 compression optimized for JSON/text data

### `@cache_if`

Conditional caching decorator that only caches when a condition is met.

```python
def cache_if(
    condition: Callable[[Any], bool],
    cache_instance: Optional[cacheness] = None,
    ttl: Optional[str] = None,
    ttl_seconds: Optional[float] = None,
    **kwargs
) -> Callable
```

**Parameters:**
- `condition` (Callable): Function that returns True if result should be cached
- Other parameters same as `@cached`

**Example:**
```python
@cache_if(lambda result: result['status'] == 'success', ttl="1h")
def api_call(endpoint):
    response = requests.get(endpoint)
    return response.json()
```

### `@cache_async`

Async version of the cached decorator.

**Example:**
```python
@cache_async(ttl="2h")
async def fetch_data(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

## Configuration Classes

### `CacheConfig`

Main configuration class for cacheness.

```python
@dataclass
class CacheConfig:
    storage: CacheStorageConfig = field(default_factory=CacheStorageConfig)
    metadata: CacheMetadataConfig = field(default_factory=CacheMetadataConfig)
    blob: CacheBlobConfig = field(default_factory=CacheBlobConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    serialization: SerializationConfig = field(default_factory=SerializationConfig)
    handlers: HandlerConfig = field(default_factory=HandlerConfig)
    default_ttl: Optional[Union[str, int, float]] = None  # "24h", "2d", or seconds
    default_ttl_seconds: Optional[float] = None  # Legacy — use default_ttl instead
```

### `CacheStorageConfig`

Configuration for cache storage options.

```python
@dataclass
class CacheStorageConfig:
    cache_dir: str = "./cache"
    max_cache_size: Optional[Union[str, int]] = None  # "2gb", "500mb"
    max_cache_size_mb: Optional[int] = 2000  # Legacy — use max_cache_size instead
    cleanup_on_init: bool = False
```

**Fields:**
- `cache_dir` (str): Directory for cache files
- `max_cache_size` (str|int): Size string (`"2gb"`) or bytes int. Overrides `max_cache_size_mb`
- `max_cache_size_mb` (int): Legacy: Maximum cache size in MB (use `max_cache_size` instead)
- `cleanup_on_init` (bool): Whether to clean expired entries on initialization

### `CacheBlobConfig`

Configuration for the blob storage layer, including optional inline blob storage.

```python
@dataclass
class CacheBlobConfig:
    backend: str = "filesystem"
    max_inline_size: int = 0  # 0 = disabled
```

**Fields:**
- `backend` (str): Blob backend to use (`"filesystem"`, `"s3"`, `"memory"`)
- `max_inline_size` (int): Maximum blob size in bytes to store inline in the metadata row instead of writing a separate file. `0` disables inline storage (default). Recommended values: `512`–`8192` bytes for small objects like short strings, scalars, or tiny dicts.

**Example:**
```python
from cacheness import cacheness, CacheConfig
from cacheness.config import CacheBlobConfig

# Inline blobs up to 4 KB avoids blob file I/O for small cache entries
cache = cacheness(CacheConfig(
    blob=CacheBlobConfig(max_inline_size=4096)
))
```

### `CacheMetadataConfig`

Configuration for metadata storage backend and memory cache layer.

```python
@dataclass
class CacheMetadataConfig:
    backend: Literal["json", "sqlite"] = "sqlite"
    database_url: Optional[str] = None
    store_full_metadata: bool = False  # Preferred (replaces store_cache_key_params)
    verify_cache_integrity: bool = True
    # Memory cache layer for disk-persistent backends
    enable_memory_cache: bool = False
    memory_cache_type: str = "lru"
    memory_cache_maxsize: int = 1000
    memory_cache_ttl_seconds: float = 300.0
    memory_cache_stats: bool = False
```

**Core Fields:**
- `backend` (str): Metadata backend ("json" or "sqlite")
- `database_url` (Optional[str]): SQLite database path (auto-generated if None)
- `store_full_metadata` (bool): Whether to store cache key parameters (replaces `store_cache_key_params`)
- `verify_cache_integrity` (bool): Whether to verify file integrity

**Memory Cache Layer Fields:**
- `enable_memory_cache` (bool): Enable memory cache layer for disk backends (SQLite/JSON)
- `memory_cache_type` (str): Cache type ("lru", "lfu", "fifo", "rr")
- `memory_cache_maxsize` (int): Maximum number of cached metadata entries
- `memory_cache_ttl_seconds` (float): Time-to-live for cached entries in seconds
- `memory_cache_stats` (bool): Enable cache hit/miss statistics tracking

**Memory Cache Architecture:**
```
Application → Memory Cache Layer → Disk Backend (JSON/SQLite)
```

The memory cache layer provides a fast in-memory cache on top of disk-persistent metadata backends to avoid repeated disk I/O operations.

### `CompressionConfig`

Configuration for compression settings.

```python
@dataclass
class CompressionConfig:
    pickle_compression_codec: str = "zstd"
    pickle_compression_level: int = 5
    use_blosc2_arrays: bool = True
    blosc2_array_codec: str = "lz4"
    blosc2_array_compression_level: int = 5
    parquet_compression: str = "lz4"
```

**Fields:**
- `pickle_compression_codec` (str): Compression codec for pickle files ("zstd", "lz4", "gzip")
- `pickle_compression_level` (int): Compression level (1-9)
- `use_blosc2_arrays` (bool): Whether to use Blosc2 for NumPy arrays
- `blosc2_array_codec` (str): Blosc2 codec for arrays
- `blosc2_array_compression_level` (int): Blosc2 compression level
- `parquet_compression` (str): Compression for Parquet files

### `SerializationConfig`

Configuration for cache key serialization.

```python
@dataclass
class SerializationConfig:
    enable_collections: bool = True
    enable_object_introspection: bool = True
    max_tuple_recursive_length: int = 10
    max_collection_depth: int = 10
```

**Fields:**
- `enable_collections` (bool): Whether to serialize collections intelligently
- `enable_object_introspection` (bool): Whether to inspect object attributes
- `max_tuple_recursive_length` (int): Maximum depth for tuple serialization
- `max_collection_depth` (int): Maximum depth for collection serialization

### `HandlerConfig`

Configuration for storage handlers.

```python
@dataclass
class HandlerConfig:
    handler_priority: List[str] = field(default_factory=lambda: [
        "numpy_arrays", "pandas_dataframes", "polars_dataframes", 
        "pandas_series", "object_pickle"
    ])
    enable_pandas_dataframes: bool = True
    enable_polars_dataframes: bool = True
    enable_numpy_arrays: bool = True
    enable_pandas_series: bool = True
```

**Fields:**
- `handler_priority` (List[str]): Order to try handlers
- `enable_*` (bool): Whether to enable specific handlers

## Error Classes

### `CacheError`

Base exception class for cache-related errors.

### `CacheStorageError`

Raised when there are issues with cache storage operations.

### `CacheSerializationError`

Raised when there are issues with serializing cache keys or data.

### `CacheCompressionError`

Raised when there are issues with compression/decompression.

### `CacheMetadataError`

Raised when there are issues with metadata operations.

## Utility Functions

### `hash_content(content: Any) -> str`
Generate a hash for arbitrary content.

**Parameters:**
- `content` (Any): Content to hash

**Returns:**
- `str`: Hexadecimal hash string

### `hash_file_path(file_path: Path) -> str`
Generate a hash for a file or directory path.

**Parameters:**
- `file_path` (Path): Path to hash

**Returns:**
- `str`: Hexadecimal hash string representing the path contents

### `get_size_mb(file_path: Path) -> float`
Get the size of a file in megabytes.

**Parameters:**
- `file_path` (Path): Path to the file

**Returns:**
- `float`: File size in MB

## Global Configuration

### Setting Default Cache Instance

```python
from cacheness import set_default_cache, CacheConfig

# Configure default cache for all @cached decorators
config = CacheConfig(
    storage=CacheStorageConfig(cache_dir="./project_cache"),
    default_ttl="24h"
)

set_default_cache(cacheness(config))

# Now all @cached decorators use this configuration
@cached(ttl="2h")
def my_function():
    return expensive_computation()
```

### Environment Variables

The following environment variables can configure cacheness:

- `CACHENESS_DIR`: Default cache directory
- `CACHENESS_MAX_SIZE_MB`: Default maximum cache size in MB
- `CACHENESS_DEFAULT_TTL_SECONDS`: Default TTL in seconds
- `CACHENESS_BACKEND`: Default metadata backend ("json" or "sqlite")
- `CACHENESS_COMPRESSION_CODEC`: Default compression codec
- `CACHENESS_COMPRESSION_LEVEL`: Default compression level

**Example:**
```bash
export CACHENESS_DIR="/fast_ssd/cache"
export CACHENESS_MAX_SIZE_MB="20000"
export CACHENESS_DEFAULT_TTL_SECONDS="604800"  # 1 week
export CACHENESS_BACKEND="sqlite"
```

---

## Extensibility API

Cacheness provides three extension points: handlers, metadata backends, and blob backends.

> **For comprehensive examples**, see the [Plugin Development Guide](./PLUGIN_DEVELOPMENT.md).

### Handler Registration

Register custom handlers to support new data types.

```python
from cacheness import (
    register_handler,
    unregister_handler,
    list_handlers,
    CacheHandler,
)
```

#### `register_handler(handler, priority=None, name=None)`

Register a custom handler with the default registry.

**Parameters:**
- `handler` (CacheHandler): Handler instance implementing the CacheHandler interface
- `priority` (int | None): Position in handler list (0 = highest priority, None = append)
- `name` (str | None): Optional name for the handler (defaults to `handler.data_type`)

**Example:**
```python
from cacheness import register_handler, CacheHandler

class ParquetHandler(CacheHandler):
    @property
    def data_type(self):
        return "parquet"
    
    def can_handle(self, data):
        import pandas as pd
        return isinstance(data, pd.DataFrame)
    
    def put(self, data, file_path, config):
        output = file_path.with_suffix(".parquet")
        data.to_parquet(output)
        return {"storage_format": "parquet", "file_path": str(output)}
    
    def get(self, file_path, metadata):
        import pandas as pd
        return pd.read_parquet(file_path)

# Register with highest priority
register_handler(ParquetHandler(), priority=0)
```

#### `unregister_handler(handler_name) -> bool`

Remove a handler from the registry.

**Parameters:**
- `handler_name` (str): The `data_type` of the handler to remove

**Returns:** `True` if removed, `False` if not found

#### `list_handlers() -> list`

List all registered handlers.

**Returns:** List of dicts with keys: `name`, `priority`, `class`, `is_builtin`

```python
for h in list_handlers():
    print(f"{h['name']}: priority={h['priority']}, builtin={h['is_builtin']}")
```

---

### Metadata Backend Registration

Register custom metadata backends for specialized storage.

```python
from cacheness import (
    register_metadata_backend,
    unregister_metadata_backend,
    get_metadata_backend,
    list_metadata_backends,
    MetadataBackend,  # Base class
)
```

#### `register_metadata_backend(name, backend_class, description=None, required_packages=None)`

Register a custom metadata backend.

**Parameters:**
- `name` (str): Unique identifier for the backend
- `backend_class` (type): Class implementing `MetadataBackend` interface
- `description` (str | None): Human-readable description
- `required_packages` (list | None): List of required package names

**Example:**
```python
from cacheness import register_metadata_backend, MetadataBackend

class RedisBackend(MetadataBackend):
    def __init__(self, connection_url: str, **kwargs):
        import redis
        self._client = redis.from_url(connection_url)
    
    # ... implement MetadataBackend methods

register_metadata_backend(
    name="redis",
    backend_class=RedisBackend,
    description="Redis-based metadata storage",
    required_packages=["redis"],
)
```

#### `get_metadata_backend(name, **kwargs) -> MetadataBackend`

Get a backend instance by name.

**Parameters:**
- `name` (str): Backend name
- `**kwargs`: Arguments passed to backend constructor

#### `list_metadata_backends() -> list`

List all registered metadata backends.

**Returns:** List of dicts with keys: `name`, `description`, `required_packages`, `is_builtin`

**Built-in Backends:**
- `json` - JSON file storage (default)
- `sqlite` - SQLite database
- `sqlite_memory` - In-memory SQLite (no persistence)
- `postgresql` - PostgreSQL database

---

### Blob Backend Registration

Register custom blob backends for data storage.

```python
from cacheness import (
    register_blob_backend,
    unregister_blob_backend,
    get_blob_backend,
    list_blob_backends,
    BlobBackend,  # Base class
    FilesystemBlobBackend,
    InMemoryBlobBackend,
)
```

#### `register_blob_backend(name, backend_class, description=None, required_packages=None)`

Register a custom blob backend.

**Example:**
```python
from cacheness import register_blob_backend, BlobBackend

class S3BlobBackend(BlobBackend):
    def __init__(self, bucket: str, prefix: str = "", **kwargs):
        import boto3
        self._s3 = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix
    
    # ... implement BlobBackend methods

register_blob_backend(
    name="s3",
    backend_class=S3BlobBackend,
    description="Amazon S3 blob storage",
    required_packages=["boto3"],
)
```

#### `get_blob_backend(name, **kwargs) -> BlobBackend`

Get a blob backend instance by name.

#### `list_blob_backends() -> list`

List all registered blob backends.

**Built-in Backends:**
- `filesystem` - Local file storage (default)
- `memory` - In-memory storage

---

### Using Custom Backends in Configuration

```python
from cacheness import cacheness, CacheConfig, CacheMetadataConfig, CacheBlobConfig

# Use registered backends by name
config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="redis",
        connection_url="redis://localhost:6379/0",
    ),
    blob=CacheBlobConfig(
        backend="s3",
        bucket="my-cache-bucket",
    )
)

cache = cacheness(config=config)
```

---

## Handler Interface (Legacy)

> **Note:** The `StorageHandler` interface below is the legacy API. For new implementations, use `CacheHandler` as shown in the Extensibility API section above.

```python
from cacheness.interfaces import StorageHandler

class CustomHandler(StorageHandler):
    def can_handle(self, data: Any) -> bool:
        """Return True if this handler can process the data type."""
        return isinstance(data, MyCustomType)
    
    def save(self, data: Any, file_path: Path, config: CompressionConfig) -> None:
        """Save data to file_path."""
        pass
    
    def load(self, file_path: Path, config: CompressionConfig) -> Any:
        """Load data from file_path."""
        pass
    
    def get_file_extension(self) -> str:
        """Return the file extension for this handler."""
        return ".custom"

# Register the handler
from cacheness.handlers import register_handler
register_handler("custom_type", CustomHandler())
```

## Thread Safety

Cacheness is thread-safe for all operations. A reentrant lock (`threading.RLock`) protects all cache operations at both the core and backend levels.

### Concurrency Model

**Core layer (`UnifiedCache`):**
- All public methods (`put()`, `get()`, `invalidate()`, `update_data()`, `touch()`, batch operations, etc.) acquire `self._lock` (an `RLock`) for their entire duration.
- The blob store shares the same lock instance — no separate synchronization needed.
- `RLock` allows composed operations (e.g., `delete_where()` calling `invalidate()`) without deadlocking.

**Metadata backends:**
- **SQLite:** Uses its own `RLock` plus SQLAlchemy sessions. WAL mode enables concurrent readers. All metadata operations are atomic.
- **JSON:** Uses its own `RLock` protecting the in-memory dict and disk I/O. Each write re-serializes the entire file.
- **PostgreSQL:** Uses SQLAlchemy session management with connection pooling.

**What's safe:**
- Multiple threads sharing a single `UnifiedCache` instance — fully safe
- Concurrent `put()`, `get()`, `invalidate()`, batch operations — all serialized via `RLock`
- Mixed read/write workloads from multiple threads — safe

**What's NOT covered:**
- **Multi-process access to the same cache directory** — use SQLite backend (WAL mode) or PostgreSQL for process-level concurrency. JSON backend is NOT safe for multi-process access.
- **Cross-machine access** — use PostgreSQL backend with S3 blob storage.

```python
import threading
from cacheness import cacheness

cache = cacheness()

def worker(worker_id):
    for i in range(100):
        # Thread-safe operations
        cache.put(f"data_{i}", worker_id=worker_id, iteration=i)
        result = cache.get(worker_id=worker_id, iteration=i)

# Multiple threads can safely use the same cache instance
threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## Performance Monitoring

### Built-in Metrics

```python
# Get detailed statistics
stats = cache.get_stats()
print(f"""
Cache Statistics:
- Total entries: {stats['total_entries']}
- Total size: {stats['total_size_mb']:.2f} MB
- Hit rate: {stats.get('hit_rate', 0):.1%}
- Backend: {stats['backend_type']}
- Average entry size: {stats['avg_entry_size_mb']:.3f} MB
- Oldest entry: {stats.get('oldest_entry_date', 'N/A')}
- Newest entry: {stats.get('newest_entry_date', 'N/A')}
""")
```

### Entry Metadata

Each cache entry includes these built-in metadata fields:

- `cache_key` (str): The generated cache key
- `created_at` (str): ISO timestamp when entry was created
- `accessed_at` (str): ISO timestamp when last accessed
- `expires_at` (Optional[str]): ISO timestamp when entry expires
- `size_mb` (float): Entry size in megabytes
- `handler_used` (str): Storage handler that processed the data
- `compression_info` (dict): Compression statistics
- `cache_key_params` (dict): Original parameters used to generate key (if enabled)

#### Backend-Specific Metadata Fields

**S3 Blob Backend:**
- `s3_etag` (Optional[str]): S3 ETag for blob integrity verification
  - Stored when using S3 or S3-compatible blob storage backends
  - Different from `file_hash` (cacheness content hash)
  - ETag is S3's server-side hash (MD5 for single uploads, composite for multipart)
  - Used for S3 consistency checks and conditional operations
  - Example: `"d41d8cd98f00b204e9800998ecf8427e"`

**File/Hash Fields:**
- `file_hash` (Optional[str]): Cacheness content hash (XXH3_64)
  - Client-side hash computed before upload
  - Consistent across all blob backends
  - Used for deduplication and cache key generation
- `entry_signature` (Optional[str]): HMAC signature for metadata integrity
  - Present when signing is enabled in configuration
  - Protects against metadata tampering

**Example:**
```python
# Query entry metadata
entry = cache.get_entry("my_cache_key")
metadata = entry.get("metadata", {})

# Check if using S3 backend
if "s3_etag" in metadata:
    print(f"S3 ETag: {metadata['s3_etag']}")
    print(f"Content Hash: {metadata.get('file_hash', 'N/A')}")
    
# List all S3-backed entries
s3_entries = [
    e for e in cache.list_entries()
    if e.get("metadata", {}).get("s3_etag")
]
```

## Migration and Compatibility

### Version Compatibility

Cacheness maintains backward compatibility for cache entries across minor versions. Major version upgrades may require cache rebuilding.

### Migration Utilities

```python
from cacheness.utils import migrate_cache

# Migrate from older cache format
migrate_cache(
    old_cache_dir="./old_cache",
    new_cache_dir="./new_cache",
    old_config=old_config,
    new_config=new_config
)
```

This completes the comprehensive API reference for cacheness. For more examples and use cases, see the [Examples](../examples/) directory and [Configuration Guide](./CONFIGURATION.md).
