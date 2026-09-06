# Local storage: initialization and failure boundaries

This guide implements [ADR 0001](adr/0001-topology-specific-storage-guarantees.md).
The currently supported target is SQLite lifecycle authority plus local filesystem
payloads on one host. Memory authority is single-process and not crash-durable.
PostgreSQL/S3 lifecycle composition and native Windows qualification remain later
work; an existing backend name does not imply that those combinations are ready.

## Initialize before sharing

```python
from cacheness.storage import BlobStore

# Run once, before threads or independent worker processes start.
with BlobStore("./objects") as store:
    store.initialize()

# Each worker opens the same initialized root using an ordinary constructor.
with BlobStore("./objects") as store:
    store.put({"answer": 42}, key="result", metadata={"experiment": "alpha"})
```

`UnifiedCache.initialize()` delegates to the same engine. Initialize cache
instances before sharing them too. Single-process first-write convenience is
retained, but concurrent first creation is not a supported availability guarantee.
Inspection of an absent store does not initialize it. Repeated initialization of
a valid current store is safe; it is not an online schema migration operation.

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
authenticated catalog and supports `get_metadata`, `list(metadata_filter=...)`,
and `update_metadata`. Richer catalog schema customization is Phase 4 work.

For policy that needs metadata and payload from one generation:

```python
with BlobStore("./objects") as store:
    receipt = store.put_entry(None, key="optional", metadata={"label": "example"})
    with store.open_entry("optional") as entry:
        assert entry is not None       # Presence, even when the value is None.
        assert entry.read() is None    # Uses the already verified private snapshot.
        assert entry.expectation == receipt.expectation
    store.delete("optional", expected=receipt.expectation)
```

`BlobEntryInfo` is an immutable authenticated receipt. `get_entry_info` retrieves
metadata without opening/deserializing the payload. `open_entry` verifies the
manifest, generation, payload digest, and size before yielding; its snapshot and
reader are valid only inside the context. No database transaction is held across
that context. Treat `expectation` as opaque: a later replacement makes an old
conditional deletion conflict rather than deleting the replacement.

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
| Optional cache projection/export failure | Storage remains committed; `put` returns its key and logs a structured warning. Canonical reads do not need that derived row. |
| Explicitly requested ORM custom-metadata failure | `CacheMetadataError` reports `committed=True`, key, generation, and the original cause. The blob remains committed; the separate link transaction was not acknowledged. |

Concurrent cache close after the engine operation completes can cause the same
derived-data outcomes. BlobStore still drains its own admitted operations and
snapshot contexts; it does not extend admission across a second catalog merely
to guarantee optional export completion.

Canonical cache reads use authenticated catalog metadata, not mutable compatibility
rows. A damaged projection cannot revoke a valid blob. Direct projection point,
list, and query observations still reject malformed structured fields with typed
corruption errors; SQL filtering may exclude unrelated stale rows before decoding.
Exact-generation queries can omit stale/missing derived rows. Reads do not silently
repair them. Canonical integrity failures preserve evidence and return a cache miss
without automatic deletion; unsafe authoritative paths remain typed errors.
`delete_invalid_signatures` does not authorize canonical evidence deletion; legacy
compatibility handling remains separate. Disabling the legacy cache hash option
does not disable the engine's canonical SHA-256 verification.

There is no new durable queue for failed cache exports and no automatic index
repair protocol in this change. Canonical metadata remains available for a future
explicit index rebuild. Applications needing a guaranteed external custom-link
commit must handle the reported partial outcome rather than retrying the entire
blob write blindly. A custom link is not part of the blob transaction.
