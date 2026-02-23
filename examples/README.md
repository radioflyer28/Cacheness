# Cacheness Examples

Practical, runnable examples organised by complexity.
Every file is a **minimum viable example** — short, self-contained, and copy-pasteable.

Run any example with:
```bash
uv run python examples/<file>.py
```

---

## Getting Started

| File | What it shows |
|------|---------------|
| [`simple_function_caching.py`](simple_function_caching.py) | `@cached` decorator — the simplest possible usage |
| [`simple_api_caching.py`](simple_api_caching.py) | `@cached.for_api()` for HTTP responses |
| [`simple_object_caching.py`](simple_object_caching.py) | Caching dataclasses, dicts, nested structures |
| [`simple_config_demo.py`](simple_config_demo.py) | Custom dirs, TTL strategies, `CacheConfig` |
| [`intelligent_storage_demo.py`](intelligent_storage_demo.py) | Auto-optimised formats (Parquet / Blosc / LZ4 / Pickle) |
| [`file_operations_demo.py`](file_operations_demo.py) | `put_file`/`get_file` — copy-in, move-in, copy-out, move-out, overwrite guard |

## Intermediate

| File | What it shows |
|------|---------------|
| [`simple_ml_pipeline.py`](simple_ml_pipeline.py) | Cache ML steps — DataFrames, sklearn models, evaluations |
| [`management_operations_demo.py`](management_operations_demo.py) | `get_metadata`, `update_data`, `touch`, batch ops, dunder methods |
| [`custom_metadata_demo.py`](custom_metadata_demo.py) | SQLAlchemy-backed queryable metadata on cache entries |
| [`api_request_caching.py`](api_request_caching.py) | Class-based API client with per-endpoint TTLs |
| [`convenience_metadata_helpers.py`](convenience_metadata_helpers.py) | `put_with_meta`/`get_with_meta`, `on=` discriminator, ORM model helpers |

## Advanced

| File | What it shows |
|------|---------------|
| [`configurable_serialization_demo.py`](configurable_serialization_demo.py) | Fine-tune serialisation configs and handler selection |
| [`dill_class_caching_demo.py`](dill_class_caching_demo.py) | Cache objects with closures/lambdas via dill fallback |
| [`s3_caching.py`](s3_caching.py) | S3 file downloads with a local disk cache |
| [`custom_metadata_postgresql.py`](custom_metadata_postgresql.py) | Custom metadata with PostgreSQL backend |

## BlobStore (Key-Value Storage)

These examples use `BlobStore` for persistent storage **without cache eviction**.

| File | What it shows |
|------|---------------|
| [`checkpoint_storage.py`](checkpoint_storage.py) | ML training checkpoints — save, list, resume |
| [`pipeline_artifact_storage.py`](pipeline_artifact_storage.py) | Data pipeline artifacts with lineage tracking |
| [`ml_model_versioning.py`](ml_model_versioning.py) | Model versioning with rich metadata |

---

## Quick Reference

```python
from cacheness import cached, cacheness, CacheConfig

# Decorator caching (simplest)
@cached(ttl_seconds="1h")
def expensive(x):
    ...

# API-optimised decorator
@cached.for_api(ttl_seconds="6h")
def call_api(endpoint):
    ...

# Explicit cache instance with config
config = CacheConfig(
    cache_dir="./my_cache",
    metadata_backend="sqlite",
    default_ttl="1d",
    max_cache_size="500mb",
    store_full_metadata=True,   # required for put_with_meta / get_with_meta
)
cache = cacheness(config)

@cached(cache_instance=cache, ttl_seconds="12h")
def with_instance(x):
    ...

# Convenience metadata helpers
cache.put_with_meta(value, experiment="run1", epoch=10, accuracy=0.94)
result = cache.get_with_meta(experiment="run1", epoch=10, accuracy=0.94)

# on= discriminator — store multiple values under the same kwargs key
cache.put_with_meta(value_v1, on="v1", model="resnet")
cache.put_with_meta(value_v2, on="v2", model="resnet")

# File storage (copy-in / copy-out / move-in / move-out)
cache.put_file("./report.pdf", name="report", version=1)
cache.get_file(name="report", version=1, dest="./output/report.pdf")
cache.put_file("./data.csv", name="data", move=True)    # move-in (deletes source)
cache.get_file(name="data", dest="./out/", move=True)   # move-out (removes from cache)

# Duration strings: "30s", "5m", "1h", "2d", "1w", "1mo", "1y"
# Size strings:     "100kb", "50mb", "2gb"
```