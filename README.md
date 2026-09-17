# Cacheness

Cacheness is a Python object-storage library. `BlobStore` owns storage lifecycle;
`UnifiedCache` adds cache policy over a caller-selected store.

> **Status — local-ready development version.** Use this checked-out revision
> for the qualified local workflows below. It is `NOT_PUBLISHED` and may change
> before the first supported release. See the
> [release qualification guide](docs/RELEASE_QUALIFICATION.md) for the detailed
> evidence matrix and current limits.

## Install from a checkout

Clone the repository, then create the minimal local environment:

```bash
uv sync --frozen --no-default-groups
```

This is the primary installation path. Task guides introduce an optional
capability only when that task needs it.

### Local wheel (secondary)

To consume a locally built artifact instead, run `uv build` from the checkout,
then install the emitted wheel into the target environment with `uv pip install`.
The checkout workflow above remains the current local-ready path.

## Quick starts

### Quick start: store an object

This same-process memory topology is useful for a direct storage round trip:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from cacheness.storage import BackendRef, BlobStore, StoreTopology

topology = StoreTopology(
    payload=BackendRef(name="memory"),
    authority=BackendRef(name="memory"),
)

with TemporaryDirectory() as temporary:
    store = BlobStore(topology, cache_dir=Path(temporary))
    try:
        store.initialize()
        receipt = store.put_entry({"owner": "Ada"}, key="profile-ada")
        assert store.get(receipt.key) == {"owner": "Ada"}
    finally:
        store.close()
```

Run [the exact memory BlobStore example](examples/memory_blob_store.py), or
follow the [store objects guide](docs/BLOB_STORE.md) for a durable catalog.

### Quick start: cache a function result

Cache policy is explicit and uses the same store topology:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from cacheness import CacheConfig, UnifiedCache, cached
from cacheness.config import CacheStorageConfig
from cacheness.storage import BackendRef, StoreTopology

topology = StoreTopology(
    payload=BackendRef(name="memory"),
    authority=BackendRef(name="memory"),
)

with TemporaryDirectory() as temporary:
    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=Path(temporary))),
        store=topology,
    )
    try:
        cache.initialize()

        @cached(cache=cache)
        def double(value: int) -> int:
            return value * 2

        assert double(21) == 42
        assert double(21) == 42
    finally:
        cache.close()
```

Run [the exact UnifiedCache example](examples/unified_cache.py), or follow the
[cache function results guide](docs/CACHE_POLICY.md) for typed outcomes and
invalidation.

## Next tasks

- [Store objects and catalog metadata](docs/BLOB_STORE.md)
- [Cache function results](docs/CACHE_POLICY.md)
- [Add a file format](docs/PLUGIN_DEVELOPMENT.md)
- [Operate or migrate a store](docs/STORAGE_INITIALIZATION.md)

## Security and Integrity

Application payloads are trusted input only: pickle and dill are never safe to
deserialize from a hostile source. Integrity signatures detect modification but
do not sandbox executable serializers. Read the
[Security Guide](docs/SECURITY.md) for the canonical boundary.

The local Phase 8 boundary retains a 128 MiB payload limit; opaque transport
evidence corroborates but never replaces the canonical digest. Its documented
projection uses `{"actual_path": str(...)}` rather than a payload locator;
see the Phase 8 material in the qualification guide for that bounded evidence.

## Documentation

The [task-first documentation index](docs/README.md) links the current store,
cache, format, and maintenance journeys. Component reference and qualification
material live there rather than duplicating lifecycle or release claims here.
