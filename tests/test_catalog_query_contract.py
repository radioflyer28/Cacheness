"""Red contracts for bounded, revision-complete catalog queries."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
import hashlib
from pathlib import Path

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


@pytest.mark.parametrize("authority_name", ["memory", "sqlite"])
def test_authenticated_canonical_scan_is_dense_sparse_and_revision_bound(
    tmp_path: Path, authority_name: str
) -> None:
    """Both authorities page authenticated descriptors without a catalog mirror."""
    from cacheness.storage.blob_store import BlobStore
    from cacheness.storage.composition import BackendRef, StoreTopology
    from cacheness.storage.lifecycle_authority import MutationSpec, VerificationProof
    from cacheness.storage.manifest import BlobManifest, StoreVersionDimensions, sign_current_manifest

    catalog = _catalog()
    root = tmp_path / authority_name
    if authority_name == "memory":
        topology = StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        )
    else:
        topology = StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        )
    store = BlobStore(topology, cache_dir=root)
    store.initialize()
    schema = _queryable_schema()
    authority = store.lifecycle_authority
    signing_key = store._authority_manifest_key()

    def promote(key: str, generation: str, rank: int) -> None:
        manifest = BlobManifest(
            versions=StoreVersionDimensions(),
            key=key,
            generation=generation,
            locator=f"generations/{key}/{generation}",
            handler_type="object",
            payload_format="pickle",
            digest="a" * 64,
            byte_size=0,
            created_at="2026-09-08T00:00:00+00:00",
            catalog_schema_id=schema.schema_id,
            catalog_schema_revision=schema.revision,
            catalog_schema_fingerprint=schema.fingerprint,
            catalog_values={"rank": rank},
            catalog_presence=("rank",),
            user_metadata={},
            handler_metadata={},
        )
        raw = sign_current_manifest(manifest, signing_key).canonical_bytes()
        prepared = authority.prepare_mutation(
            MutationSpec(
                operation_id=f"operation-{key}",
                key=key,
                generation=generation,
                candidate_locator=f"generations/{key}/{generation}",
                expected=authority.read_expectation(key),
                manifest=raw,
            )
        )
        authority.record_verification(
            prepared,
            VerificationProof(
                digest=hashlib.sha256(raw).hexdigest(), byte_size=0, manifest=raw
            ),
        )
        authority.promote_mutation(prepared)

    try:
        promote("a", "generation-a", 0)
        promote("b", "generation-b", 1)
        promote("c", "generation-c", 2)

        sparse = catalog.CatalogQuery(
            predicates=(catalog.CatalogPredicate("rank", "gte", 99),)
        )
        first_sparse = store.query_catalog(sparse, schema=schema, work_cap=2)
        assert first_sparse.entries == ()
        assert first_sparse.exhausted is False
        assert first_sparse.examined_identity == ("b", "generation-b")
        final_sparse = store.query_catalog(
            sparse, schema=schema, cursor=first_sparse.cursor, work_cap=2
        )
        assert final_sparse.entries == ()
        assert final_sparse.exhausted is True
        assert final_sparse.examined_identity == ("c", "generation-c")

        dense = catalog.CatalogQuery(
            predicates=(catalog.CatalogPredicate("rank", "gte", 1),)
        )
        first_dense = store.query_catalog(dense, schema=schema, limit=1, work_cap=3)
        assert [(entry.key, entry.values["rank"]) for entry in first_dense.entries] == [
            ("b", 1)
        ]
        assert first_dense.exhausted is False
        assert first_dense.examined_identity == ("b", "generation-b")
        final_dense = store.query_catalog(
            dense, schema=schema, cursor=first_dense.cursor, limit=1, work_cap=3
        )
        assert [(entry.key, entry.values["rank"]) for entry in final_dense.entries] == [
            ("c", 2)
        ]
        assert final_dense.exhausted is True
        assert store.capabilities.portable_query is True
        assert store.capabilities.index_acceleration is False

        stale = store.query_catalog(dense, schema=schema, limit=1, work_cap=1)
        promote("d", "generation-d", 3)
        with pytest.raises(catalog.CatalogStaleCursorError):
            store.query_catalog(
                dense, schema=schema, cursor=stale.cursor, limit=1, work_cap=1
            )
    finally:
        store.close()
