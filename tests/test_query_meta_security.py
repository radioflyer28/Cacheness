"""Security contracts for typed, bounded BlobStore catalog queries."""

from __future__ import annotations

import math

import pytest

from cacheness.storage.blob_store import BlobStore
from cacheness.storage.catalog import (
    MAX_CURSOR_ENCODED_BYTES,
    MAX_SIGNED_64,
    MIN_SIGNED_64,
    CatalogCursorError,
    CatalogField,
    CatalogPredicate,
    CatalogQuery,
    CatalogQueryValidationError,
    CatalogSchema,
    validate_catalog_query,
)
from cacheness.storage.composition import BackendRef, StoreTopology


def _schema() -> CatalogSchema:
    """Return the declared scalar boundary used by query validation tests."""
    return CatalogSchema(
        fields=(
            CatalogField("first_safe", "string", queryable=True),
            CatalogField("middle_safe", "string", queryable=True),
            CatalogField("last_safe", "string", queryable=True),
            CatalogField("score", "integer", queryable=True),
            CatalogField("experiment", "string", queryable=True),
        ),
        schema_id="query-security",
    )


@pytest.fixture
def catalog_store(tmp_path: pytest.TempPathFactory) -> BlobStore:
    """Create one explicit caller-owned memory topology for boundary probes."""
    store = BlobStore(
        StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        ),
        cache_dir=tmp_path,
    )
    try:
        yield store
    finally:
        store.close()


def _blocked_boundary(calls: list[str], name: str):
    """Return a spy that fails if rejected input crosses a storage boundary."""
    def blocked(*_args: object, **_kwargs: object) -> None:
        calls.append(name)
        raise AssertionError(f"rejected catalog query reached {name}")

    return blocked


@pytest.mark.parametrize("position", ("first", "middle", "last"))
def test_hostile_fields_fail_before_authority_manifest_or_handler_io(
    catalog_store: BlobStore,
    monkeypatch: pytest.MonkeyPatch,
    position: str,
) -> None:
    """Every predicate position is fully validated before any lifecycle access."""
    calls: list[str] = []
    monkeypatch.setattr(
        catalog_store.lifecycle_authority,
        "catalog_page",
        _blocked_boundary(calls, "authority"),
    )
    monkeypatch.setattr(
        catalog_store,
        "_authenticated_authority_manifest",
        _blocked_boundary(calls, "manifest"),
    )
    monkeypatch.setattr(
        catalog_store.handlers,
        "get_handler",
        _blocked_boundary(calls, "handler"),
    )

    predicates = [
        CatalogPredicate("first_safe", "eq", "first"),
        CatalogPredicate("last_safe", "eq", "last"),
    ]
    predicates.insert(
        {"first": 0, "middle": 1, "last": 2}[position],
        CatalogPredicate("unsafe[0]", "eq", "blocked"),
    )

    with pytest.raises(CatalogQueryValidationError):
        catalog_store.query_catalog(
            CatalogQuery(predicates=tuple(predicates)),
            schema=_schema(),
            limit=1,
            work_cap=1,
        )

    assert calls == []


def test_empty_query_is_distinct_from_an_empty_caller_field() -> None:
    """Query-all stays explicit while an empty predicate field fails closed."""
    schema = _schema()
    assert validate_catalog_query(CatalogQuery(), schema=schema) == CatalogQuery()

    with pytest.raises(CatalogQueryValidationError):
        validate_catalog_query(
            CatalogQuery(predicates=(CatalogPredicate("", "eq", "value"),)),
            schema=schema,
        )


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_nonfinite_numeric_values_fail_before_authority_dispatch(value: float) -> None:
    """Non-finite values cannot cross the exact scalar validation boundary."""
    with pytest.raises(CatalogQueryValidationError):
        validate_catalog_query(
            CatalogQuery(predicates=(CatalogPredicate("score", "eq", value),)),
            schema=_schema(),
        )


@pytest.mark.parametrize("endpoint", (MIN_SIGNED_64, MAX_SIGNED_64))
def test_signed_64_bit_numeric_endpoints_are_valid(endpoint: int) -> None:
    """The declared integer domain includes both finite signed-64 endpoints."""
    query = CatalogQuery(predicates=(CatalogPredicate("score", "eq", endpoint),))
    assert validate_catalog_query(query, schema=_schema()) is query


@pytest.mark.parametrize(
    "value", (MIN_SIGNED_64 - 1, MAX_SIGNED_64 + 1, -(10**100), 10**100)
)
@pytest.mark.parametrize("position", ("first", "middle", "last"))
def test_out_of_domain_integers_fail_in_every_predicate_position(
    value: int, position: str
) -> None:
    """Out-of-range integers fail during preflight without predicate reordering."""
    predicates = [
        CatalogPredicate("first_safe", "eq", "before"),
        CatalogPredicate("last_safe", "eq", "after"),
    ]
    predicates.insert(
        {"first": 0, "middle": 1, "last": 2}[position],
        CatalogPredicate("score", "eq", value),
    )

    with pytest.raises(CatalogQueryValidationError):
        validate_catalog_query(CatalogQuery(predicates=tuple(predicates)), schema=_schema())


def test_multiple_predicates_retain_order_and_native_bound_values() -> None:
    """Validation receives immutable typed predicates, not caller-built SQL text."""
    query = CatalogQuery(
        predicates=(
            CatalogPredicate("experiment", "eq", "bound-value"),
            CatalogPredicate("score", "gte", 3),
        )
    )

    assert validate_catalog_query(query, schema=_schema()) is query
    assert query.predicates == (
        CatalogPredicate("experiment", "eq", "bound-value"),
        CatalogPredicate("score", "gte", 3),
    )


def test_oversized_cursor_fails_before_decode_authority_manifest_or_handler_io(
    catalog_store: BlobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cursor envelopes reject excess input before any lifecycle or payload work."""
    from cacheness.storage import catalog

    calls: list[str] = []
    monkeypatch.setattr(
        catalog,
        "base64",
        type("Base64Spy", (), {"b64decode": _blocked_boundary(calls, "decode")})(),
    )
    monkeypatch.setattr(
        catalog_store.lifecycle_authority,
        "catalog_page",
        _blocked_boundary(calls, "authority"),
    )
    monkeypatch.setattr(
        catalog_store,
        "_authenticated_authority_manifest",
        _blocked_boundary(calls, "manifest"),
    )
    monkeypatch.setattr(
        catalog_store.handlers,
        "get_handler",
        _blocked_boundary(calls, "handler"),
    )

    with pytest.raises(CatalogCursorError):
        catalog_store.query_catalog(
            CatalogQuery(),
            schema=_schema(),
            cursor="A" * (MAX_CURSOR_ENCODED_BYTES + 1),
            limit=1,
            work_cap=1,
        )

    assert calls == []
