# API Reference

Complete reference for all cacheness classes, methods, and configuration options.

## SQL Cache Classes

### `SqlCache`

SQL pull-through cache for intelligent data fetching and caching.

```python
class SqlCache:
    def __init__(
        self,
        db_url: str,
        table: Table,
        data_adapter: SqlCacheAdapter,
        ttl_hours: int = 24,
        echo: bool = False,
        engine_kwargs: Optional[dict] = None,
        time_increment: Optional[Union[str, timedelta, int]] = None,
        ordered_increment: Optional[Union[int, float]] = None,
        gap_detector: Optional[Callable] = None
    )
```

**Parameters:**
- `db_url` (str): Database URL (e.g., "duckdb:///data.db", "sqlite:///cache.db")
- `table` (Table): SQLAlchemy Table definition
- `data_adapter` (SqlCacheAdapter): Adapter for fetching external data
- `ttl_hours` (int): Time-to-live in hours for cached entries
- `echo` (bool): Whether to echo SQL statements
- `engine_kwargs` (Optional[dict]): Additional SQLAlchemy engine parameters
- `time_increment` (Optional[Union[str, timedelta, int]]): Expected time increment for data
  - `timedelta` objects: `timedelta(minutes=5)`
  - String formats: `"30sec"`, `"5min"`, `"2hour"`
  - Numeric seconds: `300` (5 minutes)
- `ordered_increment` (Optional[Union[int, float]]): Expected increment for ordered data (e.g., order IDs)
- `gap_detector` (Optional[Callable]): Custom gap detection function with signature:
  `gap_detector(query_params, cached_data, cache_instance) -> List[Dict]`

#### Factory Methods

##### Classic Factory Methods

##### `SqlCache.with_sqlite(db_path, table, data_adapter, **kwargs)`
Create cache with SQLite backend.

```python
cache = SqlCache.with_sqlite(
    db_path="cache.db",
    table=table_definition,
    data_adapter=adapter,
    ttl_hours=24,
    time_increment=timedelta(minutes=5),
    gap_detector=custom_detector
)
```

##### `SqlCache.with_duckdb(db_path, table, data_adapter, **kwargs)`
Create cache with DuckDB backend for analytical workloads.

```python
cache = SqlCache.with_duckdb(
    db_path="analytics.db",
    table=table_definition,
    data_adapter=adapter,
    time_increment="15sec",
    ordered_increment=1
)
```

##### `SqlCache.with_postgresql(db_url, table, data_adapter, **kwargs)`
Create cache with PostgreSQL backend for production environments.

```python
cache = SqlCache.with_postgresql(
    db_url="postgresql://user:pass@host/db",
    table=table_definition,
    data_adapter=adapter,
    gap_detector=lambda q, c, i: []  # Simple custom detector
)
```

##### New Builder Methods (Recommended)

##### `SqlCache.for_lookup_table(db_path, data_fetcher, **columns)`
Create cache for individual record lookups (uses SQLite).

```python
cache = SqlCache.for_lookup_table(
    "users.db",
    primary_keys=["user_id"],
    data_fetcher=fetch_user_data,
    ttl_hours=12,
    user_id=Integer,
    name=String(100),
    email=String(255)
)
```

##### `SqlCache.for_analytics_table(db_path, data_fetcher, **columns)`
Create cache for bulk analytical queries (uses DuckDB).

```python
cache = SqlCache.for_analytics_table(
    "analytics.db", 
    primary_keys=["department", "month"],
    data_fetcher=fetch_sales_data,
    ttl_hours=24,
    department=String(50),
    revenue=Float,
    headcount=Integer
)
```

##### `SqlCache.for_timeseries(db_path, data_fetcher, **columns)`
Create cache for historical timeseries analysis (uses DuckDB).

```python
cache = SqlCache.for_timeseries(
    "historical.db",
    data_fetcher=fetch_historical_prices,
    ttl_hours=48,
    price=Float,
    volume=Integer,
    market_cap=Float
)
```

##### `SqlCache.for_realtime_timeseries(db_path, data_fetcher, **columns)`
Create cache for real-time timeseries data (uses SQLite).

```python
cache = SqlCache.for_realtime_timeseries(
    "realtime.db",
    data_fetcher=fetch_live_prices,
    ttl_hours=1,
    price=Float,
    bid=Float,
    ask=Float
)
```

#### Methods

##### `get_data(**query_params) -> pd.DataFrame`
Main pull-through cache method.

**Parameters:**
- `**query_params`: Query parameters passed to data adapter

**Returns:**
- `pd.DataFrame`: Complete dataset from cache and external source

##### `get_cache_stats() -> dict`
Get cache statistics.

**Returns:**
- Dictionary with cache statistics

##### `cleanup_expired() -> int`
Remove expired entries.

**Returns:**
- Number of entries removed

##### `clear_cache() -> int`
Remove all entries.

**Returns:**
- Number of entries removed

### `SqlCacheAdapter`

Abstract base class for data adapters.

```python
class SqlCacheAdapter(ABC):
    @abstractmethod
    def get_table_definition(self) -> Table:
        """Return SQLAlchemy Table definition."""
        pass
    
    @abstractmethod
    def parse_query_params(self, **kwargs) -> Dict[str, Any]:
        """Parse query parameters for gap detection."""
        pass
    
    @abstractmethod
    def fetch_data(self, **kwargs) -> pd.DataFrame:
        """Fetch data from external source."""
        pass
```

#### Methods

##### `get_table_definition() -> Table`
Return SQLAlchemy Table definition for the cached data.

##### `parse_query_params(**kwargs) -> Dict[str, Any]`
Parse incoming query parameters into format expected by gap detection.

**Example:**
```python
def parse_query_params(self, **kwargs):
    return {
        'symbol': kwargs['symbol'],
        'date': {
            'start': kwargs['start_date'],
            'end': kwargs['end_date']
        }
    }
```

##### `fetch_data(**kwargs) -> pd.DataFrame`
Fetch data from external source (API, database, etc.).

**Parameters:**
- `**kwargs`: Parameters for data fetching

**Returns:**
- `pd.DataFrame`: Fetched data

### Custom Gap Detection

Custom gap detectors provide flexible control over when to fetch missing data.

#### Function Signature

```python
def custom_gap_detector(
    query_params: Dict[str, Any],
    cached_data: pd.DataFrame, 
    cache_instance: SqlCache
) -> List[Dict[str, Any]]:
    """
    Custom gap detection function.
    
    Args:
        query_params: Parsed query parameters from adapter
        cached_data: Currently cached data matching the query
        cache_instance: SqlCache instance providing access to:
            - cache_instance.time_increment: User-specified time increment
            - cache_instance.ordered_increment: User-specified ordered increment
            - cache_instance._detect_granularity(): Built-in time detection
            - cache_instance._detect_ordered_granularity(): Built-in ordered detection
            - cache_instance._convert_query_to_fetch_params(): Parameter conversion
    
    Returns:
        List of fetch parameter dictionaries. Empty list means no fetch needed.
    """
    pass
```

#### Example Custom Detectors

```python
# Always fetch (useful for testing)
def always_fetch(query_params, cached_data, cache_instance):
    return [cache_instance._convert_query_to_fetch_params(query_params)]

# Conservative detector (only fetch if no data)
def conservative_detector(query_params, cached_data, cache_instance):
    if cached_data.empty:
        return [cache_instance._convert_query_to_fetch_params(query_params)]
    return []

# Increment-aware detector
def increment_detector(query_params, cached_data, cache_instance):
    # Access increment settings
    time_inc = cache_instance.time_increment
    order_inc = cache_instance.ordered_increment
    
    # Use settings for custom logic
    if time_inc and time_inc < timedelta(minutes=1):
        # High-frequency logic
        pass
    
    return []  # Your logic here
```

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

##### `list_entries(include_metadata: bool = False) -> List[dict]`
List all cache entries.

**Parameters:**
- `include_metadata` (bool): Whether to include custom metadata in results

**Returns:**
- List of dictionaries containing entry information

**Example:**
```python
entries = cache.list_entries(include_metadata=True)
for entry in entries:
    print(f"Key: {entry['cache_key']}, Size: {entry['size_mb']:.2f} MB")
```

##### `get_stats() -> dict`
Get cache statistics.

**Returns:**
- Dictionary with cache statistics including total entries, size, hit rate, etc.

#### Factory Methods

##### `cacheness.for_api(cache_dir=None, ttl_hours=6, **kwargs)`
Create a cache instance optimized for API requests.

**Parameters:**
- `cache_dir` (Optional[str]): Cache directory (default: "./cache")
- `ttl_hours` (int): Default TTL in hours (default: 6)
- `**kwargs`: Additional configuration options

**Returns:**
- `cacheness`: Configured cache instance with LZ4 compression for fast JSON/text caching

**Example:**
```python
api_cache = cacheness.for_api(cache_dir="./api_cache", ttl_hours=4)
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

## Decorators

### `@cached`

Decorator for caching function results with intelligent TTL management.

```python
def cached(
    cache_instance: Optional[cacheness] = None,
    ttl_hours: Optional[float] = None,
    ttl_minutes: Optional[float] = None,
    ttl_seconds: Optional[float] = None,
    cache_key_prefix: Optional[str] = None,
    include_defaults: bool = True,
    metadata: Optional[dict] = None
) -> Callable
```

**Parameters:**
- `cache_instance` (Optional[cacheness]): Cache instance to use (uses global if None)
- `ttl_hours` (Optional[float]): Time-to-live in hours
- `ttl_minutes` (Optional[float]): Time-to-live in minutes  
- `ttl_seconds` (Optional[float]): Time-to-live in seconds
- `cache_key_prefix` (Optional[str]): Prefix for cache keys
- `include_defaults` (bool): Whether to include default parameter values in cache key
- `metadata` (Optional[dict]): Custom metadata to store with cached results

**Example:**
```python
@cached(ttl_hours=24, cache_key_prefix="weather")
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
@cached.for_api(ttl_hours=6, ignore_errors=True)
def fetch_user_data(user_id):
    response = requests.get(f"/api/users/{user_id}")
    return response.json()
```

**Parameters:**
- `ttl_hours` (int): Time-to-live in hours (default: 6)
- `ignore_errors` (bool): Continue on cache errors (default: True)
- Uses LZ4 compression optimized for JSON/text data

### `@cache_if`

Conditional caching decorator that only caches when a condition is met.

```python
def cache_if(
    condition: Callable[[Any], bool],
    cache_instance: Optional[cacheness] = None,
    ttl_hours: Optional[float] = None,
    **kwargs
) -> Callable
```

**Parameters:**
- `condition` (Callable): Function that returns True if result should be cached
- Other parameters same as `@cached`

**Example:**
```python
@cache_if(lambda result: result['status'] == 'success', ttl_hours=1)
def api_call(endpoint):
    response = requests.get(endpoint)
    return response.json()
```

### `@cache_async`

Async version of the cached decorator.

**Example:**
```python
@cache_async(ttl_hours=2)
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
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    serialization: SerializationConfig = field(default_factory=SerializationConfig)
    handlers: HandlerConfig = field(default_factory=HandlerConfig)
    default_ttl_hours: Optional[float] = None
```

### `CacheStorageConfig`

Configuration for cache storage options.

```python
@dataclass
class CacheStorageConfig:
    cache_dir: str = "./cache"
    max_cache_size_mb: int = 10000
    cleanup_on_init: bool = False
```

**Fields:**
- `cache_dir` (str): Directory for cache files
- `max_cache_size_mb` (int): Maximum cache size in MB
- `cleanup_on_init` (bool): Whether to clean expired entries on initialization

### `CacheMetadataConfig`

Configuration for metadata storage backend and memory cache layer.

```python
@dataclass
class CacheMetadataConfig:
    backend: Literal["json", "sqlite"] = "sqlite"
    database_url: Optional[str] = None
    store_cache_key_params: bool = True
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
- `store_cache_key_params` (bool): Whether to store cache key parameters
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

The memory cache layer provides a fast in-memory cache on top of disk-persistent metadata backends to avoid repeated disk I/O operations. This is separate from and complementary to the pure in-memory backend.

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
    default_ttl_hours=24
)

set_default_cache(cacheness(config))

# Now all @cached decorators use this configuration
@cached(ttl_hours=2)
def my_function():
    return expensive_computation()
```

### Environment Variables

The following environment variables can configure cacheness:

- `CACHENESS_DIR`: Default cache directory
- `CACHENESS_MAX_SIZE_MB`: Default maximum cache size in MB
- `CACHENESS_DEFAULT_TTL_HOURS`: Default TTL in hours
- `CACHENESS_BACKEND`: Default metadata backend ("json" or "sqlite")
- `CACHENESS_COMPRESSION_CODEC`: Default compression codec
- `CACHENESS_COMPRESSION_LEVEL`: Default compression level

**Example:**
```bash
export CACHENESS_DIR="/fast_ssd/cache"
export CACHENESS_MAX_SIZE_MB="20000"
export CACHENESS_DEFAULT_TTL_HOURS="168"  # 1 week
export CACHENESS_BACKEND="sqlite"
```

## Handler Interface

For implementing custom storage handlers:

```python
from cacheness.interfaces import StorageHandler

class CustomHandler(StorageHandler):
    def can_handle(self, data: Any) -> bool:
        """Return True if this handler can process the data type."""
        return isinstance(data, MyCustomType)
    
    def save(self, data: Any, file_path: Path, config: CompressionConfig) -> None:
        """Save data to file_path."""
        # Implementation here
        pass
    
    def load(self, file_path: Path, config: CompressionConfig) -> Any:
        """Load data from file_path."""
        # Implementation here
        pass
    
    def get_file_extension(self) -> str:
        """Return the file extension for this handler."""
        return ".custom"

# Register the handler
from cacheness.handlers import register_handler
register_handler("custom_type", CustomHandler())
```

## Thread Safety

Cacheness is thread-safe for all operations:

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
