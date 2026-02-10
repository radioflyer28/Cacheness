# BlobStore - Low-Level Storage API

`BlobStore` provides a simple key-value storage API for Python objects with metadata support. Unlike the high-level `cacheness()` API, BlobStore does **not** implement caching semantics (TTL, eviction) - it's pure storage.

## When to Use BlobStore

Use `BlobStore` when you need:

- **Pure storage** without caching semantics (no TTL, no eviction)
- **Explicit key control** - you manage storage keys directly
- **Content deduplication** - identical objects share storage automatically
- **Metadata-rich storage** - attach queryable metadata to stored objects

Use `cacheness()` or `@cached` when you need:

- **Function memoization** - automatic caching of function results
- **TTL-based expiration** - cached data expires after a time period
- **Eviction policies** - automatic cleanup based on size or count

## Quick Start

```python
from cacheness.storage import BlobStore

# Create a blob store
store = BlobStore(
    cache_dir="./my_storage",
    backend="sqlite",      # "json" or "sqlite"
    compression="lz4",     # Compression codec
)

# Store data with metadata
key = store.put(
    my_model,
    key="model_v1",
    metadata={"accuracy": 0.95, "author": "alice"}
)

# Retrieve data
model = store.get("model_v1")

# Get metadata only (fast - no deserialization)
meta = store.get_metadata("model_v1")

# List keys
all_keys = store.list()
model_keys = store.list(prefix="model_")

# Delete
store.delete("model_v1")
```

## Blob Backend Configuration

BlobStore supports pluggable blob backends for file lifecycle operations (delete, exists checks). By default, it uses the filesystem backend.

### Using Built-in Backends

`python
from cacheness.storage import BlobStore

# Default: filesystem backend
store = BlobStore(cache_dir="./data")

# Explicit filesystem backend
store = BlobStore(cache_dir="./data", blob_backend="filesystem")

# In-memory backend (for testing)
store = BlobStore(cache_dir="./temp", blob_backend="memory")
`

### Custom Blob Backend Instance

`python
from cacheness.storage import BlobStore
from cacheness.storage.backends import FilesystemBlobBackend

# Custom configuration
custom_backend = FilesystemBlobBackend(
    base_dir="./cache",
    shard_chars=3  # Use 3-char sharding (abc/abc123...)
)

store = BlobStore(
    cache_dir="./cache",
    blob_backend=custom_backend
)
`

> **Note:** Handlers still manage format-specific serialization to local paths. Blob backends manage file lifecycle (delete, exists). See [Architecture Guide](ARCHITECTURE.md) for details on the separation of concerns.

## API Reference

### Constructor

```python
BlobStore(
    cache_dir: str | Path = ".blobstore",
    backend: str | MetadataBackend = None,
    compression: str = "lz4",
    compression_level: int = 3,
    content_addressable: bool = False,
    blob_backend: str | BlobBackend | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_dir` | `str \| Path` | `.blobstore` | Directory for storing blobs and metadata |
| `backend` | `str \| MetadataBackend` | `None` (json) | Metadata backend: `"json"`, `"sqlite"`, or custom instance |
| `compression` | `str` | `"lz4"` | Compression codec (lz4, zstd, gzip, blosclz) |
| `compression_level` | `int` | `3` | Compression level (1-9) |
| `content_addressable` | `bool` | `False` | Use content hash as key (enables deduplication) |
| `blob_backend` | `BlobBackend \| None` | `None` | Blob storage backend for file lifecycle operations. Defaults to `FilesystemBlobBackend`. |

> **Architecture note:** The `backend` parameter controls _metadata_ storage (key index, timestamps). The `blob_backend` parameter controls _file lifecycle_ operations (delete, exists). Serialization is handled by handlers. See the [Architecture Guide](ARCHITECTURE.md) for details.

### Methods

#### `put(data, key=None, metadata=None) -> str`

Store a blob with optional metadata.

```python
# Store with explicit key
key = store.put(model, key="fraud_detector_v1")

# Store with auto-generated key
key = store.put(model)  # Returns generated key like "a1b2c3d4e5f6g7h8"

# Store with metadata
key = store.put(
    model,
    key="fraud_detector_v1",
    metadata={
        "accuracy": 0.95,
        "training_date": "2024-01-15",
        "author": "alice"
    }
)
```

#### `get(key) -> Any | None`

Retrieve a blob by key. Returns `None` if not found.

```python
model = store.get("fraud_detector_v1")
if model is None:
    print("Not found")
```

#### `get_metadata(key) -> dict | None`

Get metadata without loading the blob content. Fast for checking properties.

```python
meta = store.get_metadata("fraud_detector_v1")
if meta:
    print(f"Accuracy: {meta['metadata']['accuracy']}")
    print(f"Size: {meta['file_size']} bytes")
```

#### `update_metadata(key, metadata) -> bool`

Update metadata for an existing blob without re-storing the data.

```python
store.update_metadata("fraud_detector_v1", {
    "deployed": True,
    "deployment_date": "2024-01-20"
})
```

#### `delete(key) -> bool`

Delete a blob and its metadata. Returns `True` if deleted.

```python
if store.delete("old_model"):
    print("Deleted")
```

#### `exists(key) -> bool`

Check if a blob exists (both metadata and file).

```python
if store.exists("fraud_detector_v1"):
    model = store.get("fraud_detector_v1")
```

#### `list(prefix=None, metadata_filter=None) -> List[str]`

List blob keys with optional filtering.

```python
# All keys
all_keys = store.list()

# Keys starting with prefix
model_keys = store.list(prefix="model_")

# Filter by metadata (exact match)
deployed = store.list(metadata_filter={"deployed": True})
```

#### `clear() -> int`

Remove all blobs. Returns count of removed blobs.

```python
count = store.clear()
print(f"Removed {count} blobs")
```

#### `close()`

Close the store and release resources.

```python
store.close()

# Or use context manager
with BlobStore("./store") as store:
    store.put(data, key="key")
# Automatically closed
```

## Content-Addressable Storage

When `content_addressable=True`, the blob key is computed from the content hash. This enables automatic deduplication - storing the same data twice returns the same key.

```python
store = BlobStore(
    cache_dir="./models",
    content_addressable=True,
)

# Store model
key1 = store.put(model_a)  # Returns "8f14e45f..."

# Store identical model - same key, no duplicate storage
key2 = store.put(model_a)  # Returns "8f14e45f..." (same key)

assert key1 == key2  # True - content is deduplicated
```

This is useful for:
- **Model versioning**: Identical models share storage
- **Artifact deduplication**: Pipeline outputs are deduplicated automatically
- **Immutable storage**: Keys are deterministic from content

## Metadata Backends

### JSON Backend (Default)

Simple file-based storage. Good for small datasets.

```python
store = BlobStore(cache_dir="./store", backend="json")
```

Creates `cache_metadata.json` in the cache directory.

### SQLite Backend

Database-backed storage. Better for larger datasets and concurrent access.

```python
store = BlobStore(cache_dir="./store", backend="sqlite")
```

Creates `cache_metadata.db` in the cache directory.

## Compression Options

BlobStore supports multiple compression codecs:

| Codec | Speed | Ratio | Notes |
|-------|-------|-------|-------|
| `lz4` | ⚡ Fastest | Good | Default, best for most cases |
| `zstd` | Fast | Better | Good balance of speed/ratio |
| `gzip` | Slow | Good | Wide compatibility |
| `blosclz` | Fast | Good | Good for numeric data |

```python
# Fast compression (default)
store = BlobStore(compression="lz4")

# Better compression ratio
store = BlobStore(compression="zstd", compression_level=6)
```

## Type-Aware Serialization

BlobStore automatically selects the best serialization format based on data type:

| Data Type | Format | Extension |
|-----------|--------|-----------|
| NumPy arrays | Blosc2 | `.b2nd` |
| Pandas DataFrames | Parquet | `.parquet` |
| General objects | Pickle | `.pkl` |

```python
import numpy as np
import pandas as pd

# NumPy arrays use efficient Blosc2 format
store.put(np.random.randn(1000, 1000), key="array")

# DataFrames use Parquet
store.put(pd.DataFrame({"a": [1, 2, 3]}), key="df")

# Other objects use pickle
store.put({"config": "value"}, key="config")
```

## Example: ML Model Storage

```python
from cacheness.storage import BlobStore
from datetime import datetime

# Create model store
model_store = BlobStore(
    cache_dir="./models",
    backend="sqlite",
    content_addressable=True,  # Deduplicate identical models
)

# Store model with rich metadata
model_store.put(
    trained_model,
    key="fraud_detector_v2.1",
    metadata={
        "model_name": "fraud_detector",
        "version": "2.1",
        "accuracy": 0.95,
        "f1_score": 0.92,
        "author": "alice",
        "training_date": datetime.now().isoformat(),
        "hyperparameters": {
            "learning_rate": 0.001,
            "epochs": 100,
        }
    }
)

# List all fraud detector versions
versions = model_store.list(prefix="fraud_detector")
for key in versions:
    meta = model_store.get_metadata(key)
    nested = meta.get("metadata", {})
    print(f"{key}: accuracy={nested.get('accuracy')}")

# Mark model as deployed
model_store.update_metadata("fraud_detector_v2.1", {
    "deployed": True,
    "deployment_env": "production",
})
```

## Example: Pipeline Artifacts

```python
from cacheness.storage import BlobStore
from datetime import datetime

# Create artifact store
artifact_store = BlobStore(
    cache_dir="./pipeline_artifacts",
    backend="sqlite",
    compression="zstd",
)

run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Stage 1: Extract
raw_data = extract_from_source()
artifact_store.put(
    raw_data,
    key=f"{run_id}/raw_data",
    metadata={
        "stage": "extract",
        "row_count": len(raw_data),
        "depends_on": None,
    }
)

# Stage 2: Transform
transformed = transform(raw_data)
artifact_store.put(
    transformed,
    key=f"{run_id}/transformed",
    metadata={
        "stage": "transform",
        "row_count": len(transformed),
        "depends_on": f"{run_id}/raw_data",  # Track lineage
    }
)

# List artifacts for this run
for key in artifact_store.list(prefix=run_id):
    meta = artifact_store.get_metadata(key)
    print(f"{key}: {meta['metadata']['stage']}")
```

## Example: Training Checkpoints

```python
from cacheness.storage import BlobStore

checkpoint_store = BlobStore(
    cache_dir="./checkpoints",
    compression="lz4",  # Fast for frequent saves
)

# Training loop with periodic checkpoints
for epoch in range(100):
    loss = train_epoch(model)
    
    if epoch % 10 == 0:
        checkpoint_store.put(
            model.state_dict(),
            key=f"checkpoint_epoch_{epoch:04d}",
            metadata={
                "epoch": epoch,
                "loss": loss,
                "learning_rate": current_lr,
            }
        )

# Find best checkpoint
best_loss = float("inf")
best_key = None
for key in checkpoint_store.list(prefix="checkpoint_"):
    meta = checkpoint_store.get_metadata(key)
    loss = meta["metadata"]["loss"]
    if loss < best_loss:
        best_loss = loss
        best_key = key

# Resume from best checkpoint
state = checkpoint_store.get(best_key)
model.load_state_dict(state)
```

## Comparison: BlobStore vs cacheness()

| Feature | `BlobStore` | `cacheness()` |
|---------|-------------|---------------|
| **Use case** | Pure storage | Function caching |
| **TTL support** | ❌ No | ✅ Yes |
| **Eviction policies** | ❌ No | ✅ Yes (LRU, size-based) |
| **Function memoization** | ❌ No | ✅ Yes |
| **Custom metadata** | ✅ Yes | ✅ Yes |
| **Content-addressable** | ✅ Yes | ❌ No |
| **Key-value API** | ✅ Direct | Abstracted |

## Related Documentation

- [Configuration](CONFIGURATION.md) - Compression and serialization options
- [Custom Metadata](CUSTOM_METADATA.md) - SQLAlchemy-based queryable metadata
- [Backend Selection](BACKEND_SELECTION.md) - Choosing metadata backends
