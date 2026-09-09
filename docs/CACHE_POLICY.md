# Cache policy with `UnifiedCache`

`BlobStore` is Cacheness's storage-lifecycle engine. `UnifiedCache` is the
policy layer above exactly one selected store: it decides cache keys, TTL,
bounded removal, and statistics; it never becomes a second authority for
payloads or metadata. `SqlCache` is a separate subsystem.

This is the pre-production cutover API. Import `UnifiedCache`, `CacheConfig`,
`CacheOutcome`, and `cached` from `cacheness`; use `StoreTopology` (and
`BackendRef` from `cacheness.storage`) to select a supported store topology.
There are no aliases, hidden global caches, or compatibility adapters in the
canonical route.

## Compose, initialize, and close explicitly

Construct one `UnifiedCache` with either one `StoreTopology` or one already
initialized `BlobStore`:

```python
from cacheness import CacheConfig, StoreTopology, UnifiedCache
from cacheness.storage import BackendRef

topology = StoreTopology(
    payload=BackendRef(name="memory"),
    authority=BackendRef(name="memory"),
)
cache = UnifiedCache(CacheConfig(), store=topology)
try:
    cache.initialize()  # required before sharing a cache-owned durable store
    # cache policy operations
finally:
    cache.close()  # closes only the store constructed from this topology
```

The memory/memory profile is one-process and ephemeral. A SQLite authority
with filesystem blobs is durable only within its declared local topology. A
PostgreSQL authority with filesystem or S3 blobs coordinates the authority but
does not make external blob effects part of its transaction. JSON with
filesystem blobs is explicitly limited to its documented local lock/recovery
profile. None of these statements promises exact LRU, global-oldest ordering,
cross-resource ACID, or a successful result for every concurrent contender.
Contention may instead return the topology's typed retryable outcome.

When the application injects a `BlobStore`, the application initializes and
closes that caller-owned store; `UnifiedCache.initialize()` deliberately does
not initialize it and `UnifiedCache.close()` does not close it. Choose one role
per store instance: direct durable object storage and cache policy should use
separate stores or separate namespaces rather than one mixed ownership role.

`CacheConfig` is nested: `storage`, `metadata`, `policy`, `compression`,
`serialization`, `handlers`, and `security` describe separate concerns. Put
finite size policy in `CachePolicyConfig`; do not infer it from a backend name.

## Read and write results

`cache.put(...)` returns `CachePutResult`. Its `receipt` records the committed
`BlobStore` generation. Its `maintenance` is separate policy truth: an
incomplete or retryable maintenance step never revokes an already committed
receipt.

`cache.lookup(...)` returns one `CacheLookupResult` from one entry observation.
Its `outcome` is always one of these six `CacheOutcome` values:

| Outcome | Meaning |
| --- | --- |
| `hit` | An entry is present; `value` can legitimately be `None`. |
| `absent` | No canonical entry was observed. |
| `expired` | Policy TTL expired from the entry snapshot; cleanup is reported separately. |
| `corrupt` | A fail-closed integrity/format observation; it is not silently relabeled as absence. |
| `conflict` | Exact-generation removal or lifecycle work met a concurrent change. |
| `backend_error` | A backend operation failed; inspect the typed `cause`. |

Presence is the primary lookup outcome. Put receipts tell committed storage
truth. `CacheRemovalReport` and `CacheMaintenanceResult` tell bounded
removal/maintenance truth. `CacheStatistics` is an immutable derived snapshot
with separate counters for hits and every non-hit outcome; it is neither a
backend dictionary nor lifecycle authority.

## Bounded removal and maintenance

`invalidate`, predicate invalidation, function clearing, and global clear use
bounded catalog selection plus exact-generation `BlobStore` removal. A
`CacheRemovalReport` names `attempted`, `removed`, `conflicted`, `retryable`,
`failed`, and `complete`; `complete=False` carries an opaque continuation. It
does not claim that all matching entries were removed in one unbounded pass.

`maintain_size()` performs one finite policy step. A `CacheMaintenanceResult`
with `complete=False` either has a sealed `CacheMaintenanceState` to pass to
`resume_maintenance(state)`, or a typed restart cause. Callers decide when to
resume; no hidden background maintenance is scheduled. A finite quiescent
catalog returns `complete=True`. Conflict, churn, or bounded work can instead
report an incomplete/retryable result without changing the committed put truth.

## Explicit decorator policy

Use only an explicit cache instance:

```python
from cacheness import cached

@cached(cache=cache)
def load_optional_record(identifier: str):
    return None
```

The decorator returns a cached `None` on a `hit`; default recomputation occurs
only for `absent` and `expired`. It preserves typed `corrupt`, `conflict`, and
`backend_error` outcomes. An application may explicitly choose a policy such
as `recompute_on=frozenset({CacheOutcome.BACKEND_ERROR})`; the wrapper retains
the original result at `cache_last_lookup` for inspection. `cache_clear()` is
function-scoped and returns the actual `CacheRemovalReport`, not unconditional
success. The application owns the cache's initialization and close boundary.

## Pre-production cutover

| Removed development path | Canonical replacement |
| --- | --- |
| Global cache factories and process-global cache ownership | Construct `UnifiedCache(config, store=...)` explicitly. |
| Alternate cache constructors and aliases | Use `UnifiedCache` with nested `CacheConfig`. |
| Alternate decorator names or implicit decorator caches | Use `cached(cache=cache)`. |
| Flat configuration and independent backend selectors | Select one `StoreTopology` and nested `CacheConfig`. |

Stored formats and schemas remain explicitly identified. An unsupported stored
version fails explicitly and without mutation. Offline migration or rebuild
tooling is a Phase 7 responsibility; opening a store never upgrades it
implicitly.
