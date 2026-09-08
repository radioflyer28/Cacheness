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

## Declared topology profiles and qualification

Construction or registration of a participant proves only that it can be
constructed. Eligibility is the narrower contract in the matrix below: exactly
one authority/payload pair must match one row before `BlobStore` performs
payload or authority I/O. A row is an immutable contract declaration, not a
mutable observation that its external evidence has passed. JSON is available
only as a derived projection in every profile; it cannot become authority or a
fallback authority.

<!-- phase5-topology-matrix:start -->
| profile | authority | payload | coordination | durability / atomicity boundary | progress outcomes | required service configuration | derived JSON projection | evidence requirement ID | evidence schema ID |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ephemeral process | memory | memory | one_process | atomic only within one process; no crash durability | conflict, success | no external service | true | local-memory-contract | phase5-local-contract-v1 |
| initialized local host | sqlite | filesystem | one_host_multiple_processes | SQLite transaction is authoritative; immutable filesystem generations reconcile outside cross-resource ACID | conflict, retryable_timeout, success | explicit initialization before shared workers; writable local filesystem | true | local-sqlite-filesystem-contract | phase5-local-contract-v1 |
| configured remote deployment | postgresql | s3 | multiple_hosts | PostgreSQL transaction is authoritative; immutable Amazon S3 objects reconcile outside cross-resource ACID | conflict, retryable_connection_timeout, retryable_deadlock, retryable_lock_timeout, retryable_serialization, retryable_statement_timeout, success | explicit PostgreSQL initialization before shared workers; real PostgreSQL service; real Amazon S3 bucket and test-owned prefix; shared external manifest signing key | true | live-postgresql-amazon-s3 | phase5-live-service-evidence-v1 |
<!-- phase5-topology-matrix:end -->

The memory/memory and SQLite/filesystem rows are Phase 5-qualified. The remote
row is a declared, constructible candidate and is **not release-qualified**
until Phase 8 satisfies its immutable `live-postgresql-amazon-s3` requirement.
The sanitized release-evidence artifact is the exclusive record of a service
run; the runtime catalog and this guide deliberately contain requirements
rather than a mutable observation. Local tests, fakes, mocks, and compatible
endpoints prove adapter contracts only. They do not substitute for the exact
real PostgreSQL and Amazon S3 evidence requirement. Amazon S3-compatible
services need their own named qualification before they can be added as a
release-supported profile.

Every pair outside the matrix is rejected before staging, including
memory/filesystem, memory/S3, SQLite/memory, SQLite/S3, PostgreSQL/memory, and
PostgreSQL/filesystem. The absence of a pair is intentional: this is not a
Cartesian backend-parity declaration.

For all durable rows, authority promotion is visibility. External immutable
payload creation and destructive cleanup are outside the database transaction;
durable intent and cleanup debt make interruption attributable and
reconcilable. This does not claim cross-resource ACID, universal contender
success, or a filesystem/S3 listing as catalog authority. Declared retryable
outcomes are safe progress results, while integrity and recovery remain
mandatory. Performance evidence is a separate measured distribution; Phase 8
owns final budget and platform-matrix acceptance.

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
