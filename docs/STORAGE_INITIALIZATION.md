# Local storage: initialization and failure boundaries

This guide implements [ADR 0001](adr/0001-topology-specific-storage-guarantees.md).
The currently supported target is SQLite lifecycle authority plus local filesystem
payloads on one host. Memory authority is single-process and not crash-durable.
PostgreSQL/S3 lifecycle composition and native Windows qualification remain later
work; an existing backend name does not imply that those combinations are ready.

## Initialize before sharing

```python
from cacheness.storage import BackendRef, BlobStore, StoreTopology


def local_sqlite_topology(root: str) -> StoreTopology:
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )

# Run once, before threads or independent worker processes start.
with BlobStore(local_sqlite_topology("./objects"), cache_dir="./objects") as store:
    store.initialize()

# Each worker opens the same initialized root using an ordinary constructor.
with BlobStore(local_sqlite_topology("./objects"), cache_dir="./objects") as store:
    store.put({"answer": 42}, key="result", metadata={"experiment": "alpha"})
```

`UnifiedCache.initialize()` delegates to the same engine. Initialize cache
instances before sharing them too. Single-process first-write convenience is
retained, but concurrent first creation is not a supported availability guarantee.
Inspection of an absent store does not initialize it. Repeated initialization of
a valid current format-2 store is validation-only; it is not an online schema
migration operation. The current authority file is
`.cacheness/lifecycle-authority-v2.sqlite3` with application ID `0x43414348`
and SQLite `user_version = 7`. This database version is intentionally separate
from the public store-format and payload-format versions.

An incomplete, foreign, or obsolete catalog fails closed with
`CacheBlobMigrationRequiredError`; it is not adopted, overwritten, or upgraded
automatically. Stop workers and preserve the original root, including signing
material and SQLite sidecars, before maintenance. For a disposable cache, explicitly
select a **new empty root** and rebuild from source. For non-disposable stored data,
retain the original and use a compatible reader/export path where available;
otherwise wait for the Phase 7 migration tooling. There is no general migration
command in this change. Never delete an unexplained empty SQLite leaf to make an
error go away: its ownership or interruption history may be unknown.

## One storage engine, separate policy

Use separate roots for a non-expiring object store and a cache. Cache instances
still use BlobStore internally; they add TTL and eviction policy. A root need not
serve both roles simultaneously. Application mapping metadata is stored in the
authenticated catalog and supports `get_metadata` and `update_metadata`.
`BlobStore.list()` accepts a key prefix only; portable metadata queries use the
declared catalog schema and `query_catalog`, never an unbounded dictionary
filter. Richer catalog schema customization is a Phase 4 contract.

For policy that needs metadata and payload from one generation:

```python
with BlobStore(local_sqlite_topology("./objects"), cache_dir="./objects") as store:
    receipt = store.put_entry(None, key="optional", metadata={"label": "example"})
    with store.open_entry("optional") as entry:
        assert entry is not None       # Presence, even when the value is None.
        assert entry.read() is None    # Uses the already verified private snapshot.
        assert entry.expectation == receipt.expectation
    store.delete("optional", expected=receipt.expectation)
```

`BlobReceipt` is the immutable result of a committed generation. `BlobEntry` is
an authenticated snapshot: `get_entry_info` inspects its metadata without a
payload reader, while `open_entry` verifies the manifest, generation, payload
digest, and size before yielding a reader valid only inside its context. No
database transaction is held across that context. Treat `expectation` as opaque:
a later replacement makes an old conditional deletion conflict rather than
deleting the replacement.

## Commit and derived data are distinct

SQLite promotion is the visibility point. Immutable payload creation precedes
it; old-payload reclamation follows it. They are not one cross-resource ACID
transaction. Interrupted cleanup remains recorded as debt for reconciliation.

| Outcome | Caller interpretation |
| --- | --- |
| Successful `put` / `put_entry` | The generation committed; another writer may subsequently replace it. |
| Exact lifecycle conflict or retryable contention timeout | The requested conditional operation could not complete as requested; inspect the typed context and retry deliberately. |
| Operational SQLite error | A typed backend failure with the original cause, not an automatic demand to migrate. |
| Recoverable post-commit cleanup error | The new generation committed; the error includes key, generation, expectation, and `committed=True`. Reconcile recorded debt; do not assume rollback. |
| Derived projection refresh failure | Storage remains committed; the receipt records a dirty projection outcome, while an explicit refresh reports committed-partial work. Canonical reads do not need a derived row. |

Concurrent cache close after the engine operation completes can cause the same
derived-data outcomes. BlobStore still drains its own admitted operations and
snapshot contexts; it does not extend admission across a second catalog merely
to guarantee optional export completion.

Canonical cache reads use authenticated catalog metadata, not mutable compatibility
rows. A damaged projection cannot revoke a valid blob. Direct projection point,
list, and query observations still reject malformed structured fields with typed
corruption errors. Exact-generation queries can omit stale or missing derived
rows. Reads do not silently repair them. Canonical integrity failures preserve
evidence and return a cache miss without automatic deletion; unsafe authoritative
paths remain typed errors. BlobStore manifest authentication and payload digest
verification are independent of cache policy.

There is no durable queue for failed projections and no automatic index-repair
protocol in this change. Canonical metadata remains available for a future explicit
projection rebuild. A projection is not part of the blob transaction.
