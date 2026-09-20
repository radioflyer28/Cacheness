<!-- refreshed: 2026-09-17 -->
# Architecture

**Analysis date:** 2026-09-17

## System overview

```text
Application ───────────────────────────────────> BlobStore
                                                    │
UnifiedCache policy ──uses one BlobStore───────────┤
                                                    ├─ HandlerRegistry
                                                    ├─ LifecycleAuthority
                                                    └─ immutable obstore payload participant
```

`BlobStore` is the single storage-lifecycle engine. It owns native-format
staging, immutable generation publication, authoritative descriptor promotion,
snapshot reads, exact deletion, cleanup debt, and reconciliation. `UnifiedCache`
is a one-way policy layer over a `BlobStore`; it adds cache keys, TTL, outcomes,
statistics, invalidation, and bounded maintenance without owning another storage
catalog.

## Component responsibilities

| Component | Responsibility | Primary files |
|---|---|---|
| Public API | Exposes cache, storage, configuration, handler, and topology entry points | `src/cacheness/__init__.py`, `src/cacheness/storage/__init__.py` |
| Cache policy | Cache keys, TTL, outcomes, invalidation, bounded maintenance | `src/cacheness/core.py`, `src/cacheness/cache_policy.py`, `src/cacheness/decorators.py` |
| Storage lifecycle | Authoritative descriptors, immutable generations, recovery, exact cleanup | `src/cacheness/storage/blob_store.py`, `src/cacheness/storage/lifecycle.py`, `src/cacheness/storage/reconciliation.py` |
| Topology composition | Validates the one-authority/one-participant boundary and capability profile | `src/cacheness/storage/composition.py`, `src/cacheness/storage/lifecycle_authority.py` |
| Payload participant | Guarded local, memory, and S3 object mechanics beneath lifecycle ownership | `src/cacheness/storage/obstore_generation_io.py`, `src/cacheness/storage/guarded_handler_io.py` |
| Catalog and projections | Validated application metadata, signed descriptors, optional projections | `src/cacheness/storage/catalog.py`, `src/cacheness/storage/projections.py`, `src/cacheness/metadata.py` |
| Format handling | Store-local handler selection, staged serialization, reconstruction | `src/cacheness/handlers.py`, `src/cacheness/interfaces.py` |
| Authorities | Memory, SQLite, and optional PostgreSQL transactional catalog transitions | `src/cacheness/storage/memory_lifecycle_authority.py`, `src/cacheness/storage/sqlite_lifecycle_authority.py`, `src/cacheness/storage/backends/postgresql_lifecycle_authority.py` |
| Security/utilities | Manifest integrity, signing, paths, key hashing, migrations, errors | `src/cacheness/storage/integrity.py`, `src/cacheness/storage/path_security.py`, `src/cacheness/security.py`, `src/cacheness/serialization.py` |

## Lifecycle boundaries

```text
UnifiedCache ──policy calls──> BlobStore
                                │
                                ├─ private handler staging / snapshot
                                ├─ immutable payload create or exact delete
                                └─ sole authority transition
```

- The selected authority establishes canonical committed membership and the
  descriptor. Payload listings, paths, and optional projections are derived
  observations, never commit authority.
- A handler can stage and reconstruct an artifact but cannot choose a visible
  generation. Custom formats use the store-local registry and contained paths.
- Payload effects occur outside an authority transaction. Attributable intent,
  cleanup debt, and bounded reconciliation handle the gap; this is not a
  cross-resource ACID promise.
- An injected `BlobStore` remains caller-owned. A topology passed to
  `UnifiedCache` creates one private owned store. Cache maintenance follows a
  committed receipt and cannot revoke it.

## Request paths

### Direct persistence

1. A caller constructs `BlobStore` from a validated topology.
2. `put_entry()` selects a handler, validates private staged output, publishes
   one immutable generation, and promotes the authenticated descriptor through
   the authority.
3. `get_entry()` resolves the descriptor, verifies digest and size, gives the
   selected handler a private snapshot, and reconstructs the value.
4. Deletion is exact-generation deletion with recovery/cleanup ownership in the
   same lifecycle engine.

### Cache policy

1. `UnifiedCache` derives a policy key and delegates the value mutation to its
   single store.
2. A committed `BlobReceipt` precedes TTL/statistics/size maintenance.
3. Reads use `BlobStore` snapshots, then apply cache-policy freshness and
   typed-outcome rules. No cache path treats payload presence as authoritative.

### Decorated functions

`@cached` normalizes call arguments and delegates to the same cache policy.
Decorator-created caches are weakly tracked for close-at-exit; this tracking
does not extend lifecycle authority beyond the underlying store.

## Retained data and integration roles

- SQLAlchemy supports local metadata/projection facilities and the optional
  PostgreSQL lifecycle authority; it is not a separate cache product.
- `psycopg` provides the optional PostgreSQL authority connection path.
- pandas, PyArrow, Polars, NumPy, Blosc2, dill, and orjson are retained handler
  integrations. Dataframes persist through handler-owned formats such as
  Parquet; their support does not create a table-oriented cache lifecycle.
- Built-in local, memory, and S3 payload mechanics are unified behind
  `ObstoreGenerationIO`; `BlobStore` remains the only lifecycle coordinator.

## Architectural constraints

- Follow [ADR 0001](../../docs/adr/0001-topology-specific-storage-guarantees.md)
  before changing lifecycle, concurrency, recovery, timeouts, or topology.
- Same-key safety is a topology-specific authority contract. Typed conflicts
  and retryable outcomes are valid; process-local coordination does not create
  distributed authority.
- Constructors validate rather than perform implicit migration. Migration and
  rebuild require explicit stopped-worker maintenance workflows.
- Projections and explicitly requested external metadata retain their bounded
  partial-outcome policy and never become canonical lifecycle authority.
- Current local readiness does not qualify live PostgreSQL/Amazon-S3 service
  behavior or controlled-Linux performance; those require their declared future
  evidence.

## Anti-patterns

- **Bypassing `BlobStore`:** do not publish handlers directly to a managed
  locator or treat a filesystem path/object listing as canonical.
- **Making `BlobStore` a cache policy:** direct persistence has no implicit TTL,
  hit/miss accounting, or eviction policy.
- **Adding coordination to chase stronger guarantees:** do not introduce a
  second catalog, queue, bootstrap protocol, or projection gate. Stop at ADR
  0001's boundary instead.
- **Using stale compatibility imports:** prefer public APIs or canonical modules
  rather than re-export paths when implementing a new feature.

## Error handling

Storage/handler boundaries translate narrow operational failures into the
domain error hierarchy and preserve causes. `UnifiedCache` classifies selected
read failures as cache outcomes while mutation failures remain visible.
Integrity, path, descriptor, and lifecycle-control failures fail closed.

---

*Current architecture map refreshed for the post-cut product boundary on 2026-09-19.*
