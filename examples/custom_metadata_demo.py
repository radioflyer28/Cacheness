#!/usr/bin/env python3
"""Catalog metadata with the direct BlobStore public API.

The catalog is a schema-validated descriptor attached to a blob generation.
It can be queried and updated without an ORM model, a second metadata
authority, or a cache-policy layer.
"""

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
    """Build the explicit local filesystem/SQLite topology for this example."""
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


def main() -> None:
    """Store, query, and update schema-declared catalog values."""
    schema = CatalogSchema(
        schema_id="experiments",
        fields=(
            CatalogField("experiment", "string", queryable=True),
            CatalogField("accuracy", "integer", queryable=True),
            CatalogField("published", "boolean", default=False, queryable=True),
        ),
    )

    with TemporaryDirectory(prefix="cacheness-catalog-") as temporary:
        root = Path(temporary)
        with BlobStore(local_topology(root), cache_dir=root) as store:
            receipt = store.put_entry(
                {"weights": [1, 2, 3]},
                key="experiment-001",
                metadata={"owner": "research"},
                catalog_schema=schema,
                catalog_values={"experiment": "baseline", "accuracy": 95},
            )

            page = store.query_catalog(
                CatalogQuery(
                    predicates=(
                        CatalogPredicate("experiment", "eq", "baseline"),
                    )
                ),
                schema=schema,
            )
            print("Catalog matches:", [(entry.key, dict(entry.values)) for entry in page.entries])

            updated = store.update_catalog(
                receipt.key,
                catalog_schema=schema,
                catalog_values={"published": True},
                expected=receipt.expectation,
            )
            assert updated is not None
            print("Payload:", store.get(receipt.key))
            print("Updated catalog revision:", updated.catalog_revision)


if __name__ == "__main__":
    main()
