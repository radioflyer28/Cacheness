# Plugin Development Guide

This guide covers how to extend cacheness with custom handlers and backends. Cacheness uses a **manual registration** approach where you explicitly register your extensions.

## Overview

Cacheness has two supported extension points:

| Extension Point | Purpose | Base Class | Registration Function |
|-----------------|---------|------------|----------------------|
| **Handlers** | Serialize/deserialize custom data types | `CacheHandler` | `store.handlers.register_handler()` |
| **Catalog projections** | Derive read models from authority-owned entries | projection contract | `StoreTopology(..., projections=...)` |

## Quick Start

```python
from cacheness import CacheHandler
from cacheness.storage import BackendRef, BlobStore, StoreTopology
```

---

## Custom Handlers

Handlers control how specific data types are serialized and deserialized. Use custom handlers when you have a proprietary data format or want optimized storage for a specific type.

### Handler Interface

```python
from cacheness import CacheHandler
from pathlib import Path
from typing import Any, Dict

class MyCustomHandler(CacheHandler):
    """Handler for MyCustomType objects."""
    
    @property
    def data_type(self) -> str:
        """Unique identifier for this handler."""
        return "my_custom_type"
    
    def can_handle(self, data: Any) -> bool:
        """Return True if this handler can serialize the data."""
        return isinstance(data, MyCustomType)
    
    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        """
        Serialize data to file.
        
        Args:
            data: The object to serialize
            file_path: Base path (add appropriate extension)
            config: Cache configuration object
            
        Returns:
            Dict with at least 'storage_format' and 'actual_path' used
        """
        output_path = file_path.with_suffix(".mycustom")
        
        # Your serialization logic here
        with open(output_path, "wb") as f:
            f.write(data.to_bytes())
        
        return {
            "storage_format": "my_custom_format",
            "actual_path": str(output_path),
            "custom_field": "any additional metadata",
        }
    
    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        """
        Deserialize data from file.
        
        Args:
            file_path: Path to the cached file
            metadata: Metadata dict returned by put()
            
        Returns:
            The deserialized object
        """
        with open(file_path, "rb") as f:
            return MyCustomType.from_bytes(f.read())
```

### Registering Handlers

```python
from cacheness import register_handler, list_handlers

# Create handler instance
handler = MyCustomHandler()

# Register with default priority (appended to handler list)
register_handler(handler)

# Register with high priority (checked first)
register_handler(handler, priority=0)

# Register with a custom name
register_handler(handler, name="my_custom")

# List all registered handlers
for h in list_handlers():
    print(f"{h['name']}: priority={h['priority']}, builtin={h['is_builtin']}")
```

### Handler Priority

Handlers are checked in order when caching data. The first handler where `can_handle()` returns `True` is used.

```python
# Priority 0 = checked first
register_handler(HighPriorityHandler(), priority=0)

# Priority None = appended to end
register_handler(FallbackHandler())
```

### Unregistering Handlers

```python
from cacheness import unregister_handler

# Remove by handler name (data_type)
success = unregister_handler("my_custom_type")

# Built-in handlers can also be removed if needed
unregister_handler("numpy_array")  # Remove NumPy handler
```

### Complete Example: Parquet Handler

```python
from cacheness import CacheHandler, register_handler
from pathlib import Path
from typing import Any, Dict
import pyarrow.parquet as pq

class ParquetHandler(CacheHandler):
    """Handler for pandas DataFrames using Parquet format."""
    
    @property
    def data_type(self) -> str:
        return "parquet_dataframe"
    
    def can_handle(self, data: Any) -> bool:
        try:
            import pandas as pd
            return isinstance(data, pd.DataFrame)
        except ImportError:
            return False
    
    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        output_path = file_path.with_suffix(".parquet")
        
        # Write with compression
        data.to_parquet(output_path, compression="snappy")
        
        return {
            "storage_format": "parquet",
            "actual_path": str(output_path),
            "compression": "snappy",
            "row_count": len(data),
            "columns": list(data.columns),
        }
    
    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        import pandas as pd
        return pd.read_parquet(file_path)

# Register with high priority to use instead of default DataFrame handler
register_handler(ParquetHandler(), priority=0)
```

---

## Custom Metadata Backends

Metadata backends store cache metadata (keys, timestamps, sizes, access patterns). Use custom backends for specialized storage like Redis, DynamoDB, or custom databases.

### Metadata Backend Interface

```python
from cacheness import MetadataBackend
from typing import Any, Dict, List, Optional
from datetime import datetime

class MyMetadataBackend(MetadataBackend):
    """Custom metadata backend using Redis."""
    
    def __init__(self, redis_url: str, **kwargs):
        self.redis_url = redis_url
        # Initialize your backend
        self._client = redis.Redis.from_url(redis_url)
    
    def put_entry(self, cache_key: str, metadata: Dict[str, Any]) -> None:
        """Store metadata for a cache entry."""
        self._client.hset(f"cache:{cache_key}", mapping=metadata)
    
    def get_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a cache entry."""
        data = self._client.hgetall(f"cache:{cache_key}")
        return dict(data) if data else None
    
    def remove_entry(self, cache_key: str) -> bool:
        """Remove a cache entry's metadata."""
        return self._client.delete(f"cache:{cache_key}") > 0
    
    def list_entries(
        self, 
        prefix: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """List all cache entries, optionally filtered by prefix."""
        pattern = f"cache:{prefix}*" if prefix else "cache:*"
        keys = self._client.keys(pattern)
        
        entries = []
        for key in keys[:limit] if limit else keys:
            entry = self.get_entry(key.decode().replace("cache:", ""))
            if entry:
                entries.append(entry)
        return entries
    
    def update_access_time(self, cache_key: str) -> None:
        """Update last access timestamp."""
        self._client.hset(
            f"cache:{cache_key}", 
            "accessed_at", 
            datetime.now().isoformat()
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Return backend statistics."""
        keys = self._client.keys("cache:*")
        return {
            "total_entries": len(keys),
            "backend_type": "redis",
        }
    
    def clear(self) -> int:
        """Remove all entries."""
        keys = self._client.keys("cache:*")
        if keys:
            return self._client.delete(*keys)
        return 0
    
    def close(self) -> None:
        """Close backend connections."""
        self._client.close()
```

### Registering Metadata Backends

```python
from cacheness import register_metadata_backend, list_metadata_backends

# Register a backend class (instantiated when needed)
register_metadata_backend(
    name="redis",
    backend_class=RedisMetadataBackend,
    description="Redis-based metadata storage",
    required_packages=["redis"],
)

# List available backends
for backend in list_metadata_backends():
    print(f"{backend['name']}: {backend['description']}")
    # Output: redis: Redis-based metadata storage
```

### Using Custom Backends

```python
from cacheness import cacheness, CacheConfig, CacheMetadataConfig

# Use by name in configuration
config = CacheConfig(
    metadata=CacheMetadataConfig(
        backend="redis",
        connection_url="redis://localhost:6379/0",
    )
)

cache = cacheness(config=config)
```

### Built-in Metadata Backends

| Name | Description | Best For |
|------|-------------|----------|
| `json` | JSON file storage | Development, small caches |
| `sqlite` | SQLite database | Production, concurrent access |
| `postgresql` | PostgreSQL database | Distributed, production |
| `memory` | In-memory storage | Testing, temporary caches |

---

## Store-local payload participation

Payload implementations are deliberately not a plugin registry. Built-in
`memory`, `filesystem`, and Amazon S3 payload roles materialize one guarded
obstore participant behind `StoreTopology`; `BlobStore` remains the lifecycle
owner. Do not add a second transport, fallback client, dual read/write path, or
payload selector.

For Amazon S3, use the built-in S3 payload role with an explicit bucket and region,
native AWS credentials, and no production custom endpoint. D-16 does
not support owner-pinning headers: use stable bucket ownership plus narrowly
scoped IAM and bucket policy instead.

```python
topology = StoreTopology(
    payload=BackendRef(name="memory"),
    authority=BackendRef(name="memory"),
)
store = BlobStore(topology)
store.handlers.register_handler(MyCustomHandler(), priority=0)
```

Handlers retain `put(data, Path, config)` and `get(Path, metadata)`. They
receive only private suffix-preserving staging or snapshot paths. Cacheness
validates descriptor identity, containment, and one regular file before object
publication or handler reads.

S3 direct conditional create has a configurable **128 MiB initial default**.
Oversized inputs fail rather than activating multipart or temporary-object
publication. Signed canonical SHA-256 plus byte size is verified before
deserialization. ETag and optional version are signed opaque, generation-bound
transport observations; their comparison is read-only and noncanonical.

Real AWS, compatible-service, native-platform, full package-matrix, RSS,
performance, and SHA-256-versus-XXH3 qualification remain Phase 8 work.

## Best Practices

### 1. Validate Interfaces

Use the interface classes to ensure your implementations are complete:

```python
from cacheness import CacheHandler

# Type hints will catch missing methods
class MyHandler(CacheHandler):
    ...  # IDE will warn about missing abstract methods
```

### 2. Handle Errors Gracefully

```python
class RobustHandler(CacheHandler):
    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        try:
            return self._load(file_path)
        except FileNotFoundError:
            return None  # Return None for missing files
        except CorruptedDataError as e:
            logger.error(f"Corrupted cache entry: {file_path}")
            raise  # Re-raise for cache to handle
```

### 3. Add Logging

```python
import logging

logger = logging.getLogger(__name__)

class LoggingHandler(CacheHandler):
    def put(self, data, file_path, config):
        logger.debug(f"Storing {type(data).__name__} to {file_path}")
        result = self._do_put(data, file_path, config)
        logger.info(f"Stored {result['storage_format']} ({result.get('size_bytes', 0)} bytes)")
        return result
```

---

## Testing Your Extensions

### Test Handlers

```python
import pytest
from pathlib import Path
import tempfile

def test_my_handler_roundtrip():
    handler = MyCustomHandler()
    
    # Test can_handle
    assert handler.can_handle(MyCustomType())
    assert not handler.can_handle("string")
    
    # Test put/get roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        data = MyCustomType(value=42)
        file_path = Path(tmpdir) / "test"
        
        metadata = handler.put(data, file_path, config=None)
        assert metadata["storage_format"] == "my_custom_format"
        
        loaded = handler.get(Path(metadata["actual_path"]), metadata)
        assert loaded.value == 42
```

### Test Backends

```python
def test_my_backend_operations():
    backend = MyMetadataBackend(connection_url="...")
    
    try:
        # Test put/get
        backend.put_entry("key1", {"data": "value"})
        entry = backend.get_entry("key1")
        assert entry["data"] == "value"
        
        # Test list
        entries = backend.list_entries()
        assert len(entries) == 1
        
        # Test remove
        assert backend.remove_entry("key1")
        assert backend.get_entry("key1") is None
        
    finally:
        backend.close()
```

---

## See Also

- [API Reference](./API_REFERENCE.md) - Complete API documentation
- [Configuration Guide](./CONFIGURATION.md) - Configuration options
- [Backend Selection Guide](./BACKEND_SELECTION.md) - Choosing backends
- [Custom Metadata Guide](./CUSTOM_METADATA.md) - SQLAlchemy custom metadata models
