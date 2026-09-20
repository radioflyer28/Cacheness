# Catalog and topology contracts

`BlobStore` is the sole owner of payload lifecycle and committed catalog
membership. The selected `LifecycleAuthority` alone promotes a generation,
establishes canonical catalog membership, and authorizes a complete catalog
query. A payload file, JSON export, ORM row, or PostgreSQL projection is never
lifecycle authority.

This reference declares composition requirements, not observed release status.
For the single current matrix of local, platform, payload, performance,
remote-service, and publication claims, see [Release qualification](RELEASE_QUALIFICATION.md).
Constructibility, local tests, fakes, mocks, and compatible endpoints do not
substitute for a named topology's required evidence.

## Composition roles

`StoreTopology` resolves every participant through one role-aware composition
path:

- **payload** stores immutable generation bytes;
- **authority** owns the transactional lifecycle and canonical catalog;
- **projection** consumes committed catalog pages as a derived view.

Role and capability validation happen before a named participant is created or
used. An injected participant retains its identity and is caller-owned unless
ownership is explicitly transferred. A projection has no promotion, cleanup,
delete, reconciliation, or canonical-query-completeness permission.

## Declared topology profiles

Exactly one authority/payload pair must match a row before `BlobStore` performs
payload or authority I/O. A row is an immutable contract declaration, not a
mutable observation. JSON is available only as a derived projection in every
profile; it cannot become authority or a fallback authority.

<!-- phase5-topology-matrix:start -->
| profile | authority | payload | coordination | durability / atomicity boundary | progress outcomes | required service configuration | derived JSON projection | evidence requirement ID | evidence schema ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ephemeral process | memory | memory | one_process | atomic only within one process; no crash durability | conflict, success | no external service | true | local-memory-contract | phase5-local-contract-v1 |
| initialized local host | sqlite | filesystem | one_host_multiple_processes | SQLite transaction is authoritative; immutable filesystem generations reconcile outside cross-resource ACID | conflict, retryable_timeout, success | explicit initialization before shared workers; writable local filesystem | true | local-sqlite-filesystem-contract | phase5-local-contract-v1 |
| configured remote deployment | postgresql | s3 | multiple_hosts | PostgreSQL transaction is authoritative; immutable Amazon S3 objects reconcile outside cross-resource ACID | conflict, retryable_connection_timeout, retryable_deadlock, retryable_lock_timeout, retryable_serialization, retryable_statement_timeout, success | explicit PostgreSQL initialization before shared workers; real PostgreSQL service; real Amazon S3 bucket and test-owned prefix; shared external manifest signing key | true | live-postgresql-amazon-s3 | phase5-live-service-evidence-v1 |
<!-- phase5-topology-matrix:end -->

Every pair outside the matrix is rejected before staging, including
memory/filesystem, memory/S3, SQLite/memory, SQLite/S3, PostgreSQL/memory, and
PostgreSQL/filesystem. The absence of a pair is intentional: this is not a
Cartesian backend-parity declaration.

For durable rows, authority promotion is visibility. Immutable payload creation
and destructive cleanup sit outside the database transaction; durable intent
and cleanup debt make interruption attributable and reconcilable. This does
not claim cross-resource ACID, universal contender success, or a
filesystem/S3 listing as catalog authority. Declared retryable outcomes are
safe progress results while integrity and recovery remain mandatory.

## Canonical catalog and projections

Portable catalog queries enumerate authenticated, committed authority
descriptors in bounded keyset pages. The cursor binds store identity, store
epoch, schema identity/fingerprint, query fingerprint, authority revision, and
the last examined identity. It is not an offset and does not rely on a derived
index.

`ProjectionController` copies these pages into a `ProjectionSink`. The first
page fixes the source/checkpoint identity and later pages must remain at that
revision. A sink applies an idempotently identified page before advancing its
checkpoint, so interruption can safely replay the same page. Projection state
is derived work; it is not consulted for BlobStore reads, cleanup, or lifecycle
recovery.

The built-in projection inventory contains JSON only. `JsonProjection` is
derived-only and supports caller-invoked bounded refresh/checkpoint delivery
through `ProjectionSink`; it does not advertise isolated rebuild publication.
Neither JSON nor any future projection can authorize canonical membership,
reads, deletes, cleanup, repair, or query completeness.

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

`update_catalog()` is an exact-record catalog patch: it keeps committed payload
integrity fields, promotes a new authenticated catalog revision through the
existing authority transaction, and returns a new `BlobReceipt`. A stale
`expected` receipt raises the typed lifecycle conflict. Stored absence, an
explicit `None`, and a materialized default remain distinct. Use `replace=True`
only when the supplied mapping is the complete new stored mapping; ordinary
calls patch existing stored fields.

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
