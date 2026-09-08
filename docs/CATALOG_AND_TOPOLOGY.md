# Catalog and topology contracts

`BlobStore` is the sole owner of payload lifecycle and committed catalog
membership. The selected `LifecycleAuthority` is the only component that can
promote a generation, establish canonical catalog membership, or authorize a
complete catalog query. A payload file, JSON export, ORM row, or PostgreSQL
projection is never lifecycle authority.

## Composition roles

`StoreTopology` resolves every participant through one role-aware composition
path:

- **payload** stores immutable generation bytes;
- **authority** owns the local transactional lifecycle and canonical catalog;
- **projection** consumes committed catalog pages as a derived view.

Role and capability validation happen before a named participant is created or
used. An injected participant retains its identity and is caller-owned unless
ownership is explicitly transferred. A projection has no promote, cleanup,
delete, reconciliation, or canonical-query-completeness permission.

## Canonical catalog and projections

Portable catalog queries enumerate authenticated, committed authority
descriptors in bounded keyset pages. The cursor binds store identity, store
epoch, schema identity/fingerprint, query fingerprint, authority revision, and
the last examined identity. It is not an offset and does not rely on a derived
index.

`ProjectionController` copies these pages into a `ProjectionSink`. The first
page fixes the source/checkpoint identity and later pages must remain at that
revision. A sink applies an idempotently identified page before it advances its
checkpoint, so an interruption can safely replay the same page. Projection
state is derived work: it is not consulted for BlobStore reads, cleanup, or
lifecycle recovery.

The Phase 4 built-in projection inventory contains only **JSON**. Its
`JsonProjection` is derived-only and supports caller-invoked bounded
refresh/checkpoint delivery through `ProjectionSink`; it does not advertise
isolated rebuild publication. PostgreSQL remains classified as a derived
projection family, but has no constructible registration until Phase 5
qualifies an actual sink. Neither JSON nor PostgreSQL can authorize canonical
membership, reads, deletes, cleanup, repair, or query completeness.

## Declared catalog values

`BlobStore` stores application catalog values in the authenticated descriptor.
Applications may keep opaque catalog values without a schema, but only fields
declared by `CatalogSchema` have portable query semantics. A declared write
validates before handler selection, authority preflight, payload staging, or
projection delivery; declared defaults are materialized into a new descriptor.

```python
from cacheness.storage.catalog import CatalogField, CatalogQuery, CatalogPredicate, CatalogSchema

schema = CatalogSchema(
    fields=(
        CatalogField("rank", "integer", default=0, queryable=True),
        CatalogField("note", "string", nullable=True),
    ),
    schema_id="application-objects",
)
receipt = store.put_entry(
    payload,
    key="object-1",
    catalog_schema=schema,
    catalog_values={"note": None, "vendor": {"source": "import"}},
)
updated = store.update_catalog(
    receipt.key,
    catalog_schema=schema,
    catalog_values={"rank": 3},
    expected=receipt.expectation,
)
page = store.query_catalog(
    CatalogQuery((CatalogPredicate("rank", "eq", 3),)),
    schema=schema,
)
```

`update_catalog()` is an exact-record catalog patch: it keeps the committed
payload generation and integrity fields, promotes a new authenticated catalog
revision through the existing authority transaction, and returns a new
`BlobReceipt`. A stale `expected` receipt raises the typed lifecycle conflict.
Stored absence, an explicit `None`, and a materialized default remain distinct.
Use `replace=True` only when the supplied mapping is the complete new stored
mapping; ordinary calls patch existing stored fields.

Schema identity and revision are part of the signed descriptor. Normal opens
and writes never migrate schemas or historical layouts. An unsupported schema
or store format fails with migration/rebuild-required evidence; offline
migration or confirmed rebuild remains the explicit future-release path.

## Failure and maintenance boundaries

Authority commit precedes any projection attempt. A projection failure leaves
the committed blob readable and exposes derived work as a named status on its
receipt; an explicitly requested refresh reports a committed-partial outcome
instead of fabricating rollback. Refresh is a caller-invoked bounded pull, not
a background queue or retry worker.

Rebuild creates an isolated projection destination and publishes it only after
the copy completes under an explicitly advertised online/offline capability.
SQLite projection publication is offline/stopped-worker maintenance. Normal
opens and writes never run schema or historical-layout migration; unsupported
layouts report migration/rebuild-required evidence without conversion.

## Phase boundaries

- **Phase 4** defines the native catalog, topology roles, bounded projection
  pulls, and explicit derived refresh/rebuild contracts.
- **Phase 5** qualifies additional backend pairs. It does not retroactively
  claim live PostgreSQL or S3 behavior from a projection declaration.
- **Phase 6** owns `UnifiedCache` policy delegation and cache-facing outcomes.
- **Phase 7** owns offline migration and rebuild execution for released
  formats; Phase 4 intentionally does not execute historical migrations.
