"""Public catalog-query regressions for the BlobStore cutover.

The catalog is the signed BlobStore descriptor surface. These tests deliberately
do not select a metadata backend or inspect a derived metadata row.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest

from cacheness.error_handling import CacheBlobBackendError
from cacheness.storage import (
    BackendRef,
    BlobStore,
    CatalogField,
    CatalogPredicate,
    CatalogQuery,
    CatalogQueryValidationError,
    CatalogSchema,
    StoreTopology,
)
from cacheness.storage.catalog import MAX_SIGNED_64, MIN_SIGNED_64


def _memory_topology() -> StoreTopology:
    """Return the qualified same-process topology for catalog regressions."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def _schema() -> CatalogSchema:
    """Declare the exact scalar fields exercised through public queries."""

    return CatalogSchema(
        schema_id="query-meta-regression",
        fields=(
            CatalogField("experiment", "string", queryable=True),
            CatalogField("model_type", "string", queryable=True),
            CatalogField("version", "string", queryable=True),
            CatalogField("score", "integer", queryable=True),
            CatalogField("active", "boolean", queryable=True),
            CatalogField("note", "string", nullable=True, queryable=True),
        ),
    )


@pytest.fixture
def catalog_store(tmp_path) -> Iterator[BlobStore]:
    """Yield one explicitly initialized caller-owned canonical store."""

    store = BlobStore(_memory_topology(), cache_dir=tmp_path)
    store.initialize()
    try:
        yield store
    finally:
        store.close()


def _put(store: BlobStore, key: str, values: dict[str, object]):
    """Commit one catalog-described value without a cache-policy wrapper."""

    return store.put_entry(
        {"key": key},
        key=key,
        catalog_schema=_schema(),
        catalog_values=values,
    )


def _query(
    store: BlobStore,
    *predicates: CatalogPredicate,
    cursor: str | None = None,
    limit: int = 10,
    work_cap: int = 20,
):
    """Run one finite, explicitly bounded catalog page."""

    query = CatalogQuery(
        predicates=predicates,
        page_size=limit,
        cursor=cursor,
    )
    return store.query_catalog(
        query,
        schema=_schema(),
        cursor=cursor,
        limit=limit,
        work_cap=work_cap,
    )


def test_catalog_query_returns_committed_descriptor_values(catalog_store: BlobStore) -> None:
    """An unfiltered finite page returns only direct committed catalog values."""

    _put(
        catalog_store,
        "model-1",
        {"experiment": "exp-001", "model_type": "xgboost", "score": 95},
    )
    _put(
        catalog_store,
        "model-2",
        {"experiment": "exp-002", "model_type": "cnn", "score": 88},
    )

    page = _query(catalog_store, limit=10, work_cap=10)

    assert [(entry.key, dict(entry.values)) for entry in page.entries] == [
        (
            "model-1",
            {"experiment": "exp-001", "model_type": "xgboost", "score": 95},
        ),
        (
            "model-2",
            {"experiment": "exp-002", "model_type": "cnn", "score": 88},
        ),
    ]
    assert page.exhausted is True
    assert page.cursor is None


def test_catalog_query_preserves_exact_scalar_and_threshold_semantics(
    catalog_store: BlobStore,
) -> None:
    """Strings are exact while declared signed integers use explicit operators."""

    _put(
        catalog_store,
        "one",
        {"model_type": "xgboost", "version": "v1", "score": 50},
    )
    _put(
        catalog_store,
        "two",
        {"model_type": "cnn", "version": "v1", "score": 75},
    )
    _put(
        catalog_store,
        "three",
        {"model_type": "xgboost", "version": "v2", "score": 100},
    )

    exact = _query(
        catalog_store,
        CatalogPredicate("model_type", "eq", "xgboost"),
    )
    threshold = _query(
        catalog_store,
        CatalogPredicate("score", "gte", 75),
    )

    assert [entry.key for entry in exact.entries] == ["one", "three"]
    assert [entry.key for entry in threshold.entries] == ["three", "two"]


@pytest.mark.parametrize("endpoint", (MIN_SIGNED_64, MAX_SIGNED_64))
def test_catalog_query_accepts_signed_64_bit_endpoints(
    catalog_store: BlobStore, endpoint: int
) -> None:
    """Both portable integer endpoints remain valid finite catalog values."""

    _put(catalog_store, "endpoint", {"score": endpoint})

    page = _query(catalog_store, CatalogPredicate("score", "eq", endpoint))

    assert [(entry.key, entry.values["score"]) for entry in page.entries] == [
        ("endpoint", endpoint)
    ]


def test_catalog_query_and_composes_mixed_predicates(catalog_store: BlobStore) -> None:
    """Multiple declared predicates keep portable AND semantics."""

    _put(catalog_store, "matching", {"model_type": "xgboost", "active": True})
    _put(catalog_store, "wrong-active", {"model_type": "xgboost", "active": False})
    _put(catalog_store, "wrong-model", {"model_type": "cnn", "active": True})

    page = _query(
        catalog_store,
        CatalogPredicate("model_type", "eq", "xgboost"),
        CatalogPredicate("active", "eq", True),
    )

    assert [entry.key for entry in page.entries] == ["matching"]


def test_catalog_query_distinguishes_no_match_null_and_absence(
    catalog_store: BlobStore,
) -> None:
    """Presence remains distinct from an explicit null in signed descriptors."""

    _put(catalog_store, "present-null", {"note": None})
    _put(catalog_store, "absent-note", {"experiment": "exp-001"})

    assert _query(
        catalog_store,
        CatalogPredicate("experiment", "eq", "does-not-exist"),
    ).entries == ()
    assert [entry.key for entry in _query(
        catalog_store,
        CatalogPredicate("note", "eq", None),
    ).entries] == ["present-null"]
    assert [entry.key for entry in _query(
        catalog_store,
        CatalogPredicate("note", "exists", False),
    ).entries] == ["absent-note"]


def test_catalog_query_handles_unicode_without_string_interpretation(
    catalog_store: BlobStore,
) -> None:
    """Unicode is a typed catalog scalar rather than executable query text."""

    _put(catalog_store, "unicode", {"experiment": "München 🧪"})

    page = _query(
        catalog_store,
        CatalogPredicate("experiment", "eq", "München 🧪"),
    )

    assert [entry.key for entry in page.entries] == ["unicode"]


def test_catalog_query_is_bounded_and_resumes_with_authority_cursor(
    catalog_store: BlobStore,
) -> None:
    """Pages remain finite and the opaque continuation is the only resume state."""

    for key in ("a", "b", "c"):
        _put(catalog_store, key, {"experiment": "paged"})

    first = _query(catalog_store, limit=1, work_cap=1)
    second = _query(
        catalog_store,
        cursor=first.cursor,
        limit=1,
        work_cap=1,
    )
    third = _query(
        catalog_store,
        cursor=second.cursor,
        limit=1,
        work_cap=1,
    )

    assert [entry.key for entry in first.entries + second.entries + third.entries] == [
        "a",
        "b",
        "c",
    ]
    assert first.cursor is not None
    assert second.cursor is not None
    assert third.exhausted is True
    assert third.cursor is None


def test_catalog_query_excludes_deleted_entry_from_current_membership(
    catalog_store: BlobStore,
) -> None:
    """Exact lifecycle removal leaves no separate metadata-query membership."""

    receipt = _put(catalog_store, "removed", {"experiment": "gone"})

    assert catalog_store.delete(receipt.key, expected=receipt.expectation) is True
    assert _query(
        catalog_store,
        CatalogPredicate("experiment", "eq", "gone"),
    ).entries == ()


def test_catalog_query_reopens_on_supported_sqlite_filesystem_topology(tmp_path) -> None:
    """Catalog descriptors survive a direct-store reopen without a cache facade."""

    root = tmp_path / "reopen"
    topology = StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )

    class SharedManifestKey:
        key = b"q" * 32

        def get_key(self) -> bytes:
            return self.key

        def get_or_initialize_new_store(self) -> bytes:
            return self.key

        def initialize_new_store(self) -> bytes:
            return self.key

    first = BlobStore(topology, cache_dir=root, manifest_key_provider=SharedManifestKey())
    try:
        first.initialize()
        _put(first, "persisted", {"experiment": "reopen"})
    finally:
        first.close()

    reopened = BlobStore(topology, cache_dir=root, manifest_key_provider=SharedManifestKey())
    try:
        page = _query(
            reopened,
            CatalogPredicate("experiment", "eq", "reopen"),
        )
    finally:
        reopened.close()

    assert [entry.key for entry in page.entries] == ["persisted"]


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf, 1.5))
def test_catalog_query_rejects_non_integer_numeric_values_before_dispatch(
    catalog_store: BlobStore, value: float
) -> None:
    """The integer schema rejects non-finite and non-exact scalar thresholds."""

    with pytest.raises(CatalogQueryValidationError):
        _query(catalog_store, CatalogPredicate("score", "gte", value))


def test_catalog_query_preserves_typed_backend_failure(
    catalog_store: BlobStore, monkeypatch
) -> None:
    """Direct backend failures remain direct typed failures, never a false miss."""

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise CacheBlobBackendError("catalog unavailable")

    monkeypatch.setattr(catalog_store.lifecycle_authority, "catalog_page", unavailable)

    with pytest.raises(CacheBlobBackendError, match="catalog unavailable"):
        _query(catalog_store)
