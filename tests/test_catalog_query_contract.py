"""Red contracts for bounded, revision-complete catalog queries."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import pytest


CATALOG_MODULE = "cacheness.storage.catalog"


def _catalog():
    assert find_spec(CATALOG_MODULE) is not None, (
        "Phase 4 must expose native typed catalog predicates and cursors"
    )
    return import_module(CATALOG_MODULE)


def _queryable_schema():
    catalog = _catalog()
    return catalog.CatalogSchema(
        fields=(
            catalog.CatalogField("rank", "integer", queryable=True),
            catalog.CatalogField("name", "string", queryable=True),
            catalog.CatalogField("published", "boolean", queryable=True),
            catalog.CatalogField("private", "string", queryable=False),
        )
    )


class _AccessSpy:
    def __init__(self) -> None:
        self.calls = 0

    def query_catalog(self, _query: object) -> None:
        self.calls += 1


@pytest.mark.parametrize(
    "operator",
    ["eq", "lt", "lte", "gt", "gte", "in", "exists"],
)
def test_portable_predicate_operators_are_explicit_and_finite(operator: str) -> None:
    catalog = _catalog()

    predicate = catalog.CatalogPredicate(field="rank", operator=operator, value=3)

    assert predicate.operator == operator


@pytest.mark.parametrize(
    ("operator", "value"),
    [
        ("lt", True),
        ("gte", "3"),
        ("in", []),
        ("exists", "yes"),
        ("contains", 3),
    ],
)
def test_invalid_operator_type_and_bounds_fail_before_authority_dispatch(
    operator: str, value: object
) -> None:
    catalog = _catalog()
    authority = _AccessSpy()

    with pytest.raises(catalog.CatalogQueryValidationError):
        catalog.validate_catalog_query(
            catalog.CatalogQuery(predicates=(catalog.CatalogPredicate("rank", operator, value),)),
            schema=_queryable_schema(),
            authority=authority,
        )

    assert authority.calls == 0


def test_predicates_are_and_composed_over_declared_queryable_fields() -> None:
    catalog = _catalog()
    query = catalog.CatalogQuery(
        predicates=(
            catalog.CatalogPredicate("rank", "gte", 2),
            catalog.CatalogPredicate("published", "eq", True),
        )
    )

    matches = catalog.evaluate_predicates(
        query.predicates,
        {"rank": 3, "published": True},
        schema=_queryable_schema(),
    )

    assert matches is True
    assert (
        catalog.evaluate_predicates(query.predicates, {"rank": 3, "published": False}, schema=_queryable_schema())
        is False
    )


def test_absent_stored_field_never_equals_a_declared_default() -> None:
    catalog = _catalog()
    schema = catalog.CatalogSchema(
        fields=(catalog.CatalogField("rank", "integer", default=0, queryable=True),)
    )

    matched = catalog.evaluate_predicates(
        (catalog.CatalogPredicate("rank", "eq", 0),), {}, schema=schema
    )

    assert matched is False


def test_non_queryable_unknown_and_boolean_as_integer_predicates_are_rejected() -> None:
    catalog = _catalog()
    schema = _queryable_schema()
    for predicate in (
        catalog.CatalogPredicate("private", "eq", "x"),
        catalog.CatalogPredicate("unknown", "exists", True),
        catalog.CatalogPredicate("rank", "eq", True),
    ):
        with pytest.raises(catalog.CatalogQueryValidationError):
            catalog.validate_catalog_query(catalog.CatalogQuery(predicates=(predicate,)), schema=schema)


def test_page_is_bounded_and_ordered_by_key_then_generation() -> None:
    catalog = _catalog()
    page = catalog.CatalogPage(
        entries=(
            catalog.CatalogEntry(key="a", generation="02", values={}),
            catalog.CatalogEntry(key="a", generation="10", values={}),
            catalog.CatalogEntry(key="b", generation="01", values={}),
        ),
        revision=4,
        cursor=None,
        exhausted=True,
    )

    assert [(entry.key, entry.generation) for entry in page.entries] == [
        ("a", "02"),
        ("a", "10"),
        ("b", "01"),
    ]


@pytest.mark.parametrize("mismatch", ["store", "format", "schema", "query", "revision", "signature"])
def test_cursor_is_authenticated_and_bound_to_every_snapshot_context(mismatch: str) -> None:
    catalog = _catalog()
    cursor = catalog.CatalogCursor.create(
        store_id="store-a",
        format_version=2,
        schema_id="schema-a",
        query_fingerprint="query-a",
        revision=3,
        last_identity=("key", "generation"),
        signing_key=b"x" * 32,
    )
    context = {
        "store_id": "store-a",
        "format_version": 2,
        "schema_id": "schema-a",
        "query_fingerprint": "query-a",
        "revision": 3,
        "signing_key": b"x" * 32,
    }
    if mismatch == "store":
        context["store_id"] = "store-b"
    elif mismatch == "format":
        context["format_version"] = 3
    elif mismatch == "schema":
        context["schema_id"] = "schema-b"
    elif mismatch == "query":
        context["query_fingerprint"] = "query-b"
    elif mismatch == "revision":
        context["revision"] = 4
    else:
        cursor = cursor[:-1] + ("A" if cursor[-1] != "A" else "B")

    with pytest.raises(catalog.CatalogCursorError):
        catalog.CatalogCursor.parse(cursor, **context)


def test_stale_revision_is_retryable_not_a_partial_success() -> None:
    catalog = _catalog()

    with pytest.raises(catalog.CatalogStaleCursorError) as raised:
        catalog.require_current_revision(cursor_revision=2, authority_revision=3)

    assert raised.value.retryable is True


def test_sparse_work_capped_scan_advances_to_last_examined_identity_without_match() -> None:
    catalog = _catalog()
    page = catalog.CatalogPage(
        entries=(),
        revision=9,
        cursor="opaque-last-examined",
        exhausted=False,
        examined_identity=("unmatched", "generation-2"),
    )

    assert page.entries == ()
    assert page.exhausted is False
    assert page.cursor == "opaque-last-examined"
    assert page.examined_identity == ("unmatched", "generation-2")
