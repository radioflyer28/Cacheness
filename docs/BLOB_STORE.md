# Store objects with `BlobStore`

`BlobStore` is Cacheness's storage foundation. It owns the authoritative
catalog and the payload lifecycle; it is useful whether or not an application
also layers cache policy above a separate store.

Choose a topology explicitly, initialize it before workers share it, and close
it when the application is done. The four canonical examples are executable
reference implementations, including the [memory round trip](../examples/memory_blob_store.py)
and the [durable catalog journey](../examples/durable_catalog_store.py).

## Store, query, update, and reopen a local catalog

The local durable topology uses a filesystem payload participant and a SQLite
catalog authority. Define catalog fields up front, then use receipts as the
exact identity for later updates or deletion:

```python
from pathlib import Path

from cacheness.storage import (
    BackendRef,
    BlobStore,
    CatalogField,
    CatalogPredicate,
    CatalogQuery,
    CatalogSchema,
    StoreTopology,
)

root = Path("./local-artifacts")
topology = StoreTopology(
    payload=BackendRef(name="filesystem", options={"base_dir": root}),
    authority=BackendRef(name="sqlite", options={"root": root}),
)
catalog = CatalogSchema(
    schema_id="artifacts",
    fields=(
        CatalogField("kind", "string", queryable=True),
        CatalogField("reviewed", "boolean", default=False, queryable=True),
    ),
)

store = BlobStore(topology, cache_dir=root)
try:
    store.initialize()
    receipt = store.put_entry(
        {"dataset": "measurements", "rows": 3},
        key="measurement-set-001",
        metadata={"owner": "research"},
        catalog_schema=catalog,
        catalog_values={"kind": "measurement"},
    )

    page = store.query_catalog(
        CatalogQuery(predicates=(CatalogPredicate("kind", "eq", "measurement"),)),
        schema=catalog,
    )
    assert [entry.key for entry in page.entries] == [receipt.key]

    updated = store.update_catalog(
        receipt.key,
        catalog_schema=catalog,
        catalog_values={"reviewed": True},
        expected=receipt.expectation,
    )
    assert updated is not None
    assert store.get(receipt.key) == {"dataset": "measurements", "rows": 3}
finally:
    store.close()
```

Catalog updates authenticate and revise catalog data without rewriting the
payload generation. `BlobReceipt.expectation` is a compare-and-swap precondition:
if another operation has already changed that entry, `update_catalog` returns
`None` instead of overwriting the newer record.

After a close, an ordinary reopen validates the same store rather than creating
another catalog or silently changing its layout:

```python
reopened = BlobStore(topology, cache_dir=root)
try:
    reopened.initialize()
    assert reopened.get("measurement-set-001") == {
        "dataset": "measurements",
        "rows": 3,
    }
finally:
    reopened.close()
```

## Read, inspect, and delete

`store.get(key)` returns the deserialized value or `None` when no canonical
entry exists. `store.get_entry_info(key)` exposes authenticated entry metadata
without reading the payload. For a deletion that must target the exact version
you observed, retain the receipt expectation:

```python
deleted = store.delete(receipt.key, expected=receipt.expectation)
assert deleted
```

Deletion and bounded catalog work are lifecycle operations. They act on an
authority-selected generation; filesystem presence is not a second source of
truth.

## Add a native format

The store chooses a built-in format handler from the value being written. To
add an application format, register a `FormatHandler` on the store that will
use it:

```python
store.handlers.register_handler(my_format_handler)
```

The [FormatHandler tutorial](PLUGIN_DEVELOPMENT.md) covers the path-based
extension contract, stable data and payload identities, one safe suffix, and a
custom MCAP-style round trip. Format handlers receive private staging or
snapshot paths, never a managed payload locator.

## Local durability boundary

For this filesystem-plus-SQLite topology, the SQLite transaction is the
catalog authority. Immutable payload generations and reconciliation provide
crash consistency across the external filesystem boundary; they are not one
cross-resource ACID transaction. See [initialization](STORAGE_INITIALIZATION.md),
[migration and rebuild](STORAGE_MIGRATION.md), and the detailed
[qualification guide](RELEASE_QUALIFICATION.md) for the declared limits.
