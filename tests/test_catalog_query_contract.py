"""Red contracts for bounded, revision-complete catalog queries."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobLifecycleConflictError


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


def _cursor_record(catalog, *, field_bytes: int) -> dict[str, object]:
    value = "\x00" * field_bytes
    return {
        "format_version": catalog.STORE_FORMAT_VERSION,
        "last_identity": [value, value],
        "query_fingerprint": value,
        "revision": catalog.MAX_SIGNED_64,
        "schema_fingerprint": value,
        "schema_id": value,
        "store_epoch": catalog.MAX_SIGNED_64,
        "store_id": value,
    }


def _signed_cursor(record: dict[str, object], signing_key: bytes = b"x" * 32) -> str:
    unsigned = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    signed = dict(record)
    signed["signature"] = hmac.new(signing_key, unsigned, hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(
        json.dumps(
            signed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).decode("ascii").rstrip("=")


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


def test_cursor_encoded_and_decoded_limits_accept_the_closed_record_boundary() -> None:
    catalog = _catalog()
    cursor = _signed_cursor(
        _cursor_record(catalog, field_bytes=catalog.MAX_CURSOR_FIELD_BYTES)
    )
    raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))

    assert len(cursor.encode("utf-8")) == catalog.MAX_CURSOR_ENCODED_BYTES
    assert len(raw) == catalog.MAX_CURSOR_DECODED_BYTES
    assert catalog.CatalogCursor.inspect(cursor, signing_key=b"x" * 32)["store_id"] == (
        "\x00" * catalog.MAX_CURSOR_FIELD_BYTES
    )

    below_boundary = _signed_cursor(
        _cursor_record(catalog, field_bytes=catalog.MAX_CURSOR_FIELD_BYTES - 1)
    )
    assert len(below_boundary.encode("utf-8")) < catalog.MAX_CURSOR_ENCODED_BYTES
    assert catalog.CatalogCursor.inspect(below_boundary, signing_key=b"x" * 32)


def test_cursor_creation_never_emits_a_token_the_inspector_rejects() -> None:
    catalog = _catalog()
    value = "\x00" * catalog.MAX_CURSOR_FIELD_BYTES
    cursor = catalog.CatalogCursor.create(
        store_id=value,
        format_version=catalog.STORE_FORMAT_VERSION,
        schema_id=value,
        schema_fingerprint=value,
        query_fingerprint=value,
        revision=catalog.MAX_SIGNED_64,
        last_identity=(value, value),
        signing_key=b"x" * 32,
        store_epoch=catalog.MAX_SIGNED_64,
    )

    assert catalog.CatalogCursor.inspect(cursor, signing_key=b"x" * 32)
    with pytest.raises(catalog.CatalogCursorError):
        catalog.CatalogCursor.create(
            store_id="x" * (catalog.MAX_CURSOR_FIELD_BYTES + 1),
            format_version=catalog.STORE_FORMAT_VERSION,
            schema_id="schema",
            query_fingerprint="query",
            revision=1,
            last_identity=("key", "generation"),
            signing_key=b"x" * 32,
        )


def test_oversized_cursor_is_rejected_before_base64_or_json_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog()
    calls: list[str] = []

    def reached(*_args: object, **_kwargs: object) -> object:
        calls.append("decode")
        raise AssertionError("oversized cursor reached decoder")

    monkeypatch.setattr(catalog.base64, "b64decode", reached)
    oversized = "A" * (catalog.MAX_CURSOR_ENCODED_BYTES + 1)

    with pytest.raises(catalog.CatalogCursorError):
        catalog.CatalogCursor.inspect(oversized, signing_key=b"x" * 32)

    assert calls == []


def test_oversized_decoded_cursor_is_rejected_before_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog()
    calls: list[str] = []

    def reached(*_args: object, **_kwargs: object) -> object:
        calls.append("json")
        raise AssertionError("oversized decoded cursor reached JSON")

    monkeypatch.setattr(catalog.json, "loads", reached)
    cursor = base64.urlsafe_b64encode(
        b"x" * (catalog.MAX_CURSOR_DECODED_BYTES + 1)
    ).decode("ascii").rstrip("=")

    with pytest.raises(catalog.CatalogCursorError):
        catalog.CatalogCursor.inspect(cursor, signing_key=b"x" * 32)

    assert calls == []


def test_non_urlsafe_base64_is_rejected_before_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog()
    calls: list[str] = []

    def reached(*_args: object, **_kwargs: object) -> object:
        calls.append("json")
        raise AssertionError("malformed base64 reached JSON")

    monkeypatch.setattr(catalog.json, "loads", reached)
    with pytest.raises(catalog.CatalogCursorError):
        catalog.CatalogCursor.inspect("!!!!", signing_key=b"x" * 32)
    assert calls == []


@pytest.mark.parametrize(
    "field",
    (
        "store_id",
        "schema_id",
        "schema_fingerprint",
        "query_fingerprint",
        "last_identity_key",
        "last_identity_generation",
    ),
)
def test_cursor_fields_are_bounded_before_hmac_comparison(
    field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = _catalog()
    record = _cursor_record(catalog, field_bytes=1)
    oversized = "x" * (catalog.MAX_CURSOR_FIELD_BYTES + 1)
    if field == "last_identity_key":
        record["last_identity"] = [oversized, "x"]
    elif field == "last_identity_generation":
        record["last_identity"] = ["x", oversized]
    else:
        record[field] = oversized
    cursor = _signed_cursor(record)

    def hmac_reached(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("oversized cursor field reached HMAC comparison")

    monkeypatch.setattr(catalog.hmac, "compare_digest", hmac_reached)
    with pytest.raises(catalog.CatalogCursorError):
        catalog.CatalogCursor.inspect(cursor, signing_key=b"x" * 32)


def test_malformed_fixed_signature_is_rejected_before_hmac_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _catalog()
    record = _cursor_record(catalog, field_bytes=1)
    record["signature"] = "f" * (catalog.MAX_CURSOR_SIGNATURE_BYTES + 1)
    cursor = base64.urlsafe_b64encode(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")

    def hmac_reached(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("malformed signature reached HMAC comparison")

    monkeypatch.setattr(catalog.hmac, "compare_digest", hmac_reached)
    with pytest.raises(catalog.CatalogCursorError):
        catalog.CatalogCursor.inspect(cursor, signing_key=b"x" * 32)


@pytest.mark.parametrize("cursor", ("!!!!", base64.urlsafe_b64encode(b"\xff").decode("ascii"), ""))
def test_malformed_cursor_syntax_is_rejected_before_hmac(cursor: str) -> None:
    catalog = _catalog()

    with pytest.raises(catalog.CatalogCursorError):
        catalog.CatalogCursor.inspect(cursor, signing_key=b"x" * 32)


def test_rejected_cursor_never_reaches_authority_or_manifest_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cacheness.storage.blob_store import BlobStore
    from cacheness.storage.composition import BackendRef, StoreTopology

    catalog = _catalog()
    store = BlobStore(
        StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        ),
        cache_dir=tmp_path,
    )
    calls: list[str] = []

    def reached(name: str):
        def fail(*_args: object, **_kwargs: object) -> None:
            calls.append(name)
            raise AssertionError(f"rejected cursor reached {name}")

        return fail

    monkeypatch.setattr(store.lifecycle_authority, "catalog_page", reached("authority"))
    monkeypatch.setattr(store, "_authenticated_authority_manifest", reached("manifest"))
    try:
        oversized_field = _cursor_record(catalog, field_bytes=1)
        oversized_field["last_identity"] = [
            "x" * (catalog.MAX_CURSOR_FIELD_BYTES + 1),
            "generation",
        ]
        for cursor in (
            "A" * (catalog.MAX_CURSOR_ENCODED_BYTES + 1),
            "A" * 2_000_000,
            _signed_cursor(oversized_field),
        ):
            with pytest.raises(catalog.CatalogCursorError):
                store.query_catalog(
                    catalog.CatalogQuery(),
                    schema=_queryable_schema(),
                    cursor=cursor,
                )
        assert calls == []
    finally:
        store.close()


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
def test_public_catalog_put_update_query_and_reopen(
    tmp_path: Path, authority_name: str
) -> None:
    """Public BlobStore calls own catalog validation and authority promotion."""
    from cacheness.storage.blob_store import BlobStore
    from cacheness.storage.composition import BackendRef, StoreTopology
    from cacheness.storage.backends.blob_backends import InMemoryBlobBackend
    from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority

    catalog = _catalog()
    root = tmp_path / authority_name
    resources: tuple[object, ...] = ()
    if authority_name == "memory":
        payload = InMemoryBlobBackend()
        authority = InMemoryLifecycleAuthority()
        topology = StoreTopology(
            payload=BackendRef(instance=payload), authority=BackendRef(instance=authority)
        )
        resources = (payload, authority)
    else:
        topology = StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        )
    class _SharedKey:
        def __init__(self) -> None:
            self.key = b"k" * 32

        def get_key(self) -> bytes:
            return self.key

        def get_or_initialize_new_store(self) -> bytes:
            return self.key

        def initialize_new_store(self) -> bytes:
            return self.key

    signing = _SharedKey()
    schema = catalog.CatalogSchema(
        fields=(
            catalog.CatalogField("rank", "integer", default=0, queryable=True),
            catalog.CatalogField("note", "string", nullable=True),
        ),
        schema_id="public-catalog",
    )
    store = BlobStore(topology, cache_dir=root, manifest_key_provider=signing)
    try:
        receipt = store.put_entry(
            {"payload": "one"},
            key="public-entry",
            catalog_schema=schema,
            catalog_values={"note": None, "vendor": {"opaque": "kept"}},
        )
        initial = store.get_entry_info(receipt.key)
        assert initial is not None
        assert initial.metadata["catalog"]["values"] == {
            "rank": 0,
            "note": None,
            "vendor": {"opaque": "kept"},
        }
        assert initial.metadata["catalog"]["presence"] == ("note", "rank", "vendor")

        updated = store.update_catalog(
            receipt.key,
            catalog_schema=schema,
            catalog_values={"rank": 3},
            expected=receipt.expectation,
        )
        assert updated is not None
        assert updated.generation == receipt.generation
        assert updated.locator == receipt.locator
        assert updated.expectation != receipt.expectation
        assert store.get(receipt.key) == {"payload": "one"}

        query = catalog.CatalogQuery(
            predicates=(catalog.CatalogPredicate("rank", "eq", 3),)
        )
        page = store.query_catalog(query, schema=schema)
        assert [(entry.key, entry.values["rank"]) for entry in page.entries] == [
            ("public-entry", 3)
        ]

        with pytest.raises(CacheBlobLifecycleConflictError):
            store.update_catalog(
                receipt.key,
                catalog_schema=schema,
                catalog_values={"rank": 4},
                expected=receipt.expectation,
            )
    finally:
        store.close()
    try:
        reopened = BlobStore(topology, cache_dir=root, manifest_key_provider=signing)
        try:
            page = reopened.query_catalog(query, schema=schema)
            assert [(entry.key, entry.values["rank"]) for entry in page.entries] == [
                ("public-entry", 3)
            ]
        finally:
            reopened.close()
    finally:
        for resource in resources:
            resource.close()


def test_invalid_public_catalog_values_do_not_reach_handler_or_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema rejection is a pre-staging boundary, not a failed lifecycle write."""
    from cacheness.storage.blob_store import BlobStore
    from cacheness.storage.composition import BackendRef, StoreTopology

    catalog = _catalog()
    schema = catalog.CatalogSchema(
        fields=(catalog.CatalogField("rank", "integer", required=True),),
        schema_id="validated-before-io",
    )
    store = BlobStore(
        StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        ),
        cache_dir=tmp_path,
    )
    calls: list[str] = []

    def reached(name: str):
        def fail(*_args: object, **_kwargs: object) -> None:
            calls.append(name)
            raise AssertionError(f"invalid catalog values reached {name}")

        return fail

    monkeypatch.setattr(store.handlers, "get_handler", reached("handler"))
    monkeypatch.setattr(
        store.lifecycle_authority, "preflight_mutation", reached("preflight")
    )
    monkeypatch.setattr(
        store, "_materialize_authority_store", reached("payload staging")
    )

    try:
        with pytest.raises(catalog.CatalogValidationError):
            store.put_entry(
                {"payload": "ignored"},
                key="invalid-catalog",
                catalog_schema=schema,
                catalog_values={"rank": True},
            )
        assert calls == []
    finally:
        store.close()


def test_public_opaque_catalog_values_round_trip_without_schema(tmp_path: Path) -> None:
    """Schema-free values remain authenticated storage metadata, not query fields."""
    from cacheness.storage.blob_store import BlobStore
    from cacheness.storage.composition import BackendRef, StoreTopology

    store = BlobStore(
        StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        ),
        cache_dir=tmp_path,
    )
    try:
        receipt = store.put_entry(
            {"payload": "opaque"},
            key="opaque-catalog",
            catalog_values={"vendor": {"source": "import"}},
        )
        entry = store.get_entry_info(receipt.key)
        assert entry is not None
        assert entry.metadata["catalog"]["values"] == {
            "vendor": {"source": "import"}
        }
    finally:
        store.close()
