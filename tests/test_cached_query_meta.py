"""Public catalog-query regressions for the BlobStore cutover.

These checks deliberately observe only the direct storage contract. They do
not inspect a database session or a metadata-backend implementation: canonical
catalog descriptors are the query source of truth.
"""

from __future__ import annotations

import pytest

from cacheness.storage import (
    BackendRef,
    BlobStore,
    CacheBlobStoreClosedError,
    CatalogField,
    CatalogPredicate,
    CatalogQuery,
    CatalogSchema,
    StoreTopology,
)


def _memory_topology() -> StoreTopology:
    """Return the explicit same-process topology used by direct-store tests."""
    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def _schema() -> CatalogSchema:
    """Declare the catalog values exercised by these public queries."""
    return CatalogSchema(
        schema_id="query-regression",
        fields=(
            CatalogField("cohort", "string", queryable=True),
            CatalogField("rank", "integer", queryable=True),
            CatalogField("active", "boolean", default=True, queryable=True),
        ),
    )


def _matching_entries(store: BlobStore, schema: CatalogSchema, rank: int):
    """Run a portable exact query through the public descriptor surface."""
    return store.query_catalog(
        CatalogQuery(
            predicates=(CatalogPredicate("rank", "eq", rank),),
            page_size=10,
        ),
        schema=schema,
    ).entries


def test_catalog_query_returns_current_descriptor_values_without_sql_inspection(
    tmp_path,
) -> None:
    """Declared values remain queryable through one public canonical scan."""
    schema = _schema()
    with BlobStore(_memory_topology(), cache_dir=tmp_path) as store:
        receipt = store.put_entry(
            {"payload": "one"},
            key="one",
            catalog_schema=schema,
            catalog_values={"cohort": "alpha", "rank": 3},
        )
        entries = _matching_entries(store, schema, 3)

    assert [(entry.key, entry.generation, dict(entry.values)) for entry in entries] == [
        (
            receipt.key,
            receipt.generation,
            {"cohort": "alpha", "rank": 3, "active": True},
        )
    ]


def test_catalog_query_excludes_replaced_generation_from_current_visibility(tmp_path) -> None:
    """A replacement is visible once at its new generation, never as a stale row."""
    schema = _schema()
    with BlobStore(_memory_topology(), cache_dir=tmp_path) as store:
        first = store.put_entry(
            "first",
            key="same-key",
            catalog_schema=schema,
            catalog_values={"cohort": "alpha", "rank": 1},
        )
        replacement = store.put_entry(
            "replacement",
            key="same-key",
            catalog_schema=schema,
            catalog_values={"cohort": "alpha", "rank": 2},
        )

        assert _matching_entries(store, schema, 1) == ()
        current = _matching_entries(store, schema, 2)
        assert [(entry.key, entry.generation) for entry in current] == [
            (replacement.key, replacement.generation)
        ]
        assert current[0].generation != first.generation
        assert store.get("same-key") == "replacement"


def test_catalog_update_is_observable_by_query_without_changing_payload_value(tmp_path) -> None:
    """The public catalog patch promotes descriptor metadata, not another authority."""
    schema = _schema()
    with BlobStore(_memory_topology(), cache_dir=tmp_path) as store:
        receipt = store.put_entry(
            {"payload": "stable"},
            key="catalog-key",
            catalog_schema=schema,
            catalog_values={"cohort": "beta", "rank": 3},
        )
        updated = store.update_catalog(
            receipt.key,
            catalog_schema=schema,
            catalog_values={"rank": 4},
            expected=receipt.expectation,
        )

        assert updated is not None
        assert _matching_entries(store, schema, 3) == ()
        assert [
            (entry.key, dict(entry.values))
            for entry in _matching_entries(store, schema, 4)
        ] == [(receipt.key, {"cohort": "beta", "rank": 4, "active": True})]
        assert store.get(receipt.key) == {"payload": "stable"}


def test_close_releases_the_store_surface_without_backend_type_introspection(tmp_path) -> None:
    """A closed store no longer exposes catalog or payload operations."""
    store = BlobStore(_memory_topology(), cache_dir=tmp_path)
    store.put_entry("value", key="entry")
    store.close()

    with pytest.raises(CacheBlobStoreClosedError):
        store.get("entry")
    with pytest.raises(CacheBlobStoreClosedError):
        store.query_catalog(CatalogQuery(), schema=_schema())
