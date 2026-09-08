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
