# API reference

This page is a focused inventory of the current public Cacheness surface. It
names imports from the two supported barrels, then points to the task guides
for complete workflows. `BlobStore` owns storage lifecycle; `UnifiedCache`
adds cache policy above a store selected by the application.

## Current public imports

Cache policy and its typed results are available from `cacheness`:

```python
from cacheness import CacheConfig, UnifiedCache, cached
from cacheness import BlobStore, CacheLookupResult, CacheOutcome, CachePutResult
from cacheness import RoleRegistry, StoreTopology
```

Direct-store composition, catalog, migration, format, and storage error types
are available from `cacheness.storage`:

```python
from cacheness.storage import (
    BackendRef,
    BlobEntry,
    BlobReceipt,
    BlobStore,
    CacheBlobPayloadTamperedError,
    CacheMigrationOrRebuildRequiredError,
    CatalogQuery,
    CatalogSchema,
    FormatHandler,
    FormatHandlerError,
    HandlerRegistry,
    MigrationPlan,
    OfflineMigrationService,
    StoreTopology,
)
```

Those imports are the compatibility boundary for this checkout. Do not infer
support for an older constructor, a global extension registry, or a payload
transport selected separately from its `StoreTopology`.

## Direct object storage

Create one topology, construct one store, initialize it before use, and close
it when the application is done. `BackendRef` names each role; `StoreTopology`
binds payload and authority roles into one lifecycle composition.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from cacheness.storage import BackendRef, BlobStore, StoreTopology

with TemporaryDirectory(prefix="cacheness-reference-") as temporary:
    root = Path(temporary)
    topology = StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )
    store = BlobStore(topology, cache_dir=root)
    try:
        store.initialize()
        receipt = store.put_entry({"answer": 42}, key="reference-object")
        assert store.get(receipt.key) == {"answer": 42}
    finally:
        store.close()
```

`put_entry(...) -> BlobReceipt` is the direct write operation when callers
need the committed key and generation receipt. `put(...) -> str` is the
convenience form that returns the key. `get(key)` returns the decoded value or
`None` when the key is absent. `get_entry_info(key) -> BlobEntry | None` and
`get_metadata(key)` provide authority-backed facts; `delete(key, expected=...)`
performs an exact-generation deletion when the caller supplies an expectation.

### Catalog metadata

Catalog fields are declared through `CatalogSchema`; applications write
portable field values with `put_entry(..., catalog_schema=...,
catalog_values=...)`, revise them with `update_catalog(...)`, and retrieve
bounded pages with `query_catalog(CatalogQuery(...), schema=...)`. See
[Direct BlobStore and catalog metadata](BLOB_STORE.md) for the full write,
query, update, and reopen journey. Catalog metadata is not a second payload
or lifecycle authority.

### Explicit offline maintenance

`OfflineMigrationService` and `MigrationPlan` support stopped-worker
inspection, planning, copy/verification, switch, and explicit rebuild work.
Ordinary opens validate rather than upgrading implicitly. See [Offline
migration and rebuild](STORAGE_MIGRATION.md) for the maintenance workflow.
`CacheMigrationOrRebuildRequiredError` means the caller must choose that
workflow instead of treating an unsupported layout as a normal open.

## Cache policy

`UnifiedCache` consumes an explicit `CacheConfig` and a selected store
topology. `put(...) -> CachePutResult` exposes the storage receipt, and
`lookup(...) -> CacheLookupResult` carries both a value and a `CacheOutcome`.
The outcome distinguishes a hit from an absent, expired, corrupt, conflict, or
backend-error result without treating a cached `None` as a miss.

Use `cached` as `@cached(cache=cache)` when a function result should use that
same policy. The complete constructor, decorator, TTL, maintenance, and
removal contract lives in [UnifiedCache policy](CACHE_POLICY.md).

## File-format extensions

`HandlerRegistry` is owned by each `BlobStore`; `FormatHandler` describes a
native payload representation. Register a handler on the store that will use
it, with an explicit priority when it must run before a built-in handler:

```python
store.handlers.register_handler(my_handler, priority=0)
```

The durable contract is the handler's `data_type`, `payload_format`, and
`payload_format_version`, not its Python protocol name. `FormatHandlerError`
is the focused extension error base. Read the [format tutorial](PLUGIN_DEVELOPMENT.md)
before implementing an application format.

## Integrity and qualification boundaries

Direct reads fail closed on invalid payload evidence. For example,
`CacheBlobPayloadTamperedError` signals failed canonical payload verification
before the format handler deserializes the value. Error handling does not turn
the store into an availability guarantee or a second lifecycle coordinator.

Evidence scope, topology limits, payload-size limits, platform boundaries, and
release state are owned by [Release qualification](RELEASE_QUALIFICATION.md).
This reference intentionally does not duplicate those claims.
