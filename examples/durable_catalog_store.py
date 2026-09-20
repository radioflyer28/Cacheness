#!/usr/bin/env python3
"""Use the durable local filesystem-plus-SQLite BlobStore catalog directly."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from cacheness.storage import (
    BackendRef,
    BlobStore,
    CatalogField,
    CatalogPredicate,
    CatalogQuery,
    CatalogSchema,
    StoreTopology,
)


def local_topology(root: Path) -> StoreTopology:
    """Build the explicit single-host filesystem payload and SQLite authority."""

    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


CATALOG = CatalogSchema(
    schema_id="local-artifacts",
    fields=(
        CatalogField("kind", "string", queryable=True),
        CatalogField("reviewed", "boolean", default=False, queryable=True),
    ),
)


def main() -> None:
    """Persist a cataloged payload, close it, and prove it survives reopening."""

    with TemporaryDirectory(prefix="cacheness-durable-catalog-") as temporary:
        root = Path(temporary)
        payload = {"dataset": "measurements", "rows": 3}

        store = BlobStore(local_topology(root), cache_dir=root)
        try:
            store.initialize()
            receipt = store.put_entry(
                payload,
                key="measurement-set-001",
                metadata={"owner": "research"},
                catalog_schema=CATALOG,
                catalog_values={"kind": "measurement"},
            )
            page = store.query_catalog(
                CatalogQuery(
                    predicates=(CatalogPredicate("kind", "eq", "measurement"),)
                ),
                schema=CATALOG,
            )
            assert [entry.key for entry in page.entries] == [receipt.key]

            updated = store.update_catalog(
                receipt.key,
                catalog_schema=CATALOG,
                catalog_values={"reviewed": True},
                expected=receipt.expectation,
            )
            assert updated is not None
            assert store.get(receipt.key) == payload
        finally:
            store.close()

        reopened = BlobStore(local_topology(root), cache_dir=root)
        try:
            reopened.initialize()
            assert reopened.get("measurement-set-001") == payload
            page = reopened.query_catalog(
                CatalogQuery(
                    predicates=(CatalogPredicate("reviewed", "eq", True),)
                ),
                schema=CATALOG,
            )
            assert [entry.key for entry in page.entries] == ["measurement-set-001"]
        finally:
            reopened.close()

    print("DURABLE_CATALOG_STORE_EXAMPLE_OK")


if __name__ == "__main__":
    main()
