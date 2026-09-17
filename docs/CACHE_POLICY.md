# Cache function results with `UnifiedCache`

`BlobStore` owns storage lifecycle. `UnifiedCache` is the policy layer above
one caller-selected store: it derives cache keys, applies TTL and bounded
removal policy, and reports typed results. It does not create a second payload
or catalog authority.

## Compose, initialize, and close explicitly

Use a `StoreTopology` when the cache should construct and own its private
store. The memory topology below is same-process and ephemeral:

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
        # Use cache below.
    finally:
        cache.close()
```

`UnifiedCache(config, store=...)` also accepts an already constructed
`BlobStore`. In that form the application owns the direct store's initialization
and close boundary; the cache adds policy without taking lifecycle ownership.

## Inspect typed cache results

`cache.put(...)` returns a `CachePutResult`. Its receipt identifies a committed
storage generation, while its maintenance result describes separate bounded
policy work. `cache.lookup(...)` returns a `CacheLookupResult`:

```python
written = cache.put({"status": "ready"}, request_id="status")
lookup = cache.lookup(cache_key=written.receipt.key)
assert lookup.outcome.value == "hit"
assert lookup.value == {"status": "ready"}
```

The outcome is explicit even when the cached value is `None`:

| Outcome | Meaning |
| --- | --- |
| `hit` | An entry is present; value can legitimately be `None`. |
| `absent` | No canonical entry was observed. |
| `expired` | The entry exceeded policy TTL; cleanup is reported separately. |
| `corrupt` | Integrity or format validation failed closed. |
| `conflict` | Exact-generation removal met a concurrent change. |
| `backend_error` | A storage operation failed; inspect the typed cause. |

## Cache a function with one explicit policy

The decorator never discovers or constructs a global cache. Give it the cache
instance you created:

```python
@cached(cache=cache)
def load_optional_record(identifier: str):
    return None

assert load_optional_record("missing") is None
assert load_optional_record("missing") is None
```

The second call is a `hit`, even though the value is `None`. By default the
decorator recomputes only `absent` and `expired`; it preserves the typed cause
for `corrupt`, `conflict`, and `backend_error` unless the caller explicitly
changes `recompute_on`. The wrapped function exposes its most recent lookup as
`cache_last_lookup`.

## Invalidate and clear truthfully

`cache.invalidate(cache_key=...)`, predicate invalidation, and
`cache.clear_all()` select bounded catalog pages and request exact-generation
deletion from `BlobStore`. They return a `CacheRemovalReport`, not unconditional
success:

```python
report = cache.clear_all()
if not report.complete:
    # Resume with report.continuation when application policy allows it.
    pass
```

An incomplete, conflicted, or retryable report does not change the truth of an
already committed receipt. Cache policy uses the store's authority-selected
generation rather than treating a path or payload listing as a second source
of truth.

For initialization and offline maintenance discipline, read
[initialization](STORAGE_INITIALIZATION.md) and
[migration and rebuild](STORAGE_MIGRATION.md). The
[UnifiedCache example](../examples/unified_cache.py) is the executable version
of this local journey.
