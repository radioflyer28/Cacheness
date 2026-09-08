"""Red contracts for the native Phase 4 catalog schema and format boundary."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
import sqlite3
from types import MappingProxyType

import pytest


CATALOG_MODULE = "cacheness.storage.catalog"


def _catalog():
    """Load the clean catalog surface only after an assertion-level gate."""
    assert find_spec(CATALOG_MODULE) is not None, (
        "Phase 4 must provide the native cacheness.storage.catalog surface"
    )
    return import_module(CATALOG_MODULE)


def _schema():
    catalog = _catalog()
    fields = (
        catalog.CatalogField("title", "string", required=True, queryable=True),
        catalog.CatalogField("count", "integer", default=0, queryable=True),
        catalog.CatalogField("enabled", "boolean", default=False, queryable=True),
        catalog.CatalogField("note", "string", nullable=True),
    )
    return catalog.CatalogSchema(fields=fields)


def test_store_format_two_is_distinguishable_from_development_format_one() -> None:
    catalog = _catalog()

    assert catalog.STORE_FORMAT_VERSION == 2
    assert catalog.STORE_FORMAT_VERSION != 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "one"),
        ("count", -(2**63)),
        ("count", 2**63 - 1),
        ("enabled", True),
        ("note", None),
    ],
)
def test_schema_accepts_only_declared_exact_scalar_types(field: str, value: object) -> None:
    schema = _schema()

    validated = schema.validate_mapping({"title": "present", field: value})

    assert validated[field] is value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", 1),
        ("count", True),
        ("count", 2**63),
        ("enabled", 1),
        ("enabled", "true"),
        ("note", 1),
    ],
)
def test_schema_rejects_type_mismatches_without_coercion(field: str, value: object) -> None:
    catalog = _catalog()

    with pytest.raises(catalog.CatalogValidationError):
        _schema().validate_mapping({"title": "present", field: value})


def test_mapping_metadata_needs_no_schema_and_preserves_opaque_values() -> None:
    catalog = _catalog()
    opaque = {"external": {"shape": ["kept", 1]}, "title": "not declared"}

    validated = catalog.validate_catalog_mapping(opaque, schema=None)

    assert validated == opaque


def test_declared_schema_preserves_bounded_undeclared_values() -> None:
    opaque = {"vendor": {"version": "v1"}}

    validated = _schema().validate_mapping({"title": "present", **opaque})

    assert validated["vendor"] == opaque["vendor"]


def test_new_writes_materialize_defaults_but_reads_keep_missing_distinct() -> None:
    schema = _schema()

    write_values = schema.validate_mapping({"title": "present"}, materialize_defaults=True)
    stored_values = schema.validate_mapping({"title": "present"}, materialize_defaults=False)

    assert write_values["count"] == 0
    assert write_values["enabled"] is False
    assert "count" not in stored_values
    assert "enabled" not in stored_values


def test_missing_required_field_and_explicit_null_have_distinct_outcomes() -> None:
    catalog = _catalog()
    schema = _schema()

    with pytest.raises(catalog.CatalogValidationError):
        schema.validate_mapping({"count": 1})
    with pytest.raises(catalog.CatalogValidationError):
        schema.validate_mapping({"title": None})
    assert schema.validate_mapping({"title": "present", "note": None})["note"] is None


def test_additive_schema_evolution_keeps_old_records_readable_without_rewrite() -> None:
    catalog = _catalog()
    old_schema = catalog.CatalogSchema(
        fields=(catalog.CatalogField("title", "string", required=True),)
    )
    evolved_schema = catalog.CatalogSchema(
        fields=(
            catalog.CatalogField("title", "string", required=True),
            catalog.CatalogField("count", "integer", default=0),
        )
    )
    stored = old_schema.validate_mapping({"title": "old"}, materialize_defaults=True)

    read_values = evolved_schema.read_mapping(stored)

    assert read_values["title"] == "old"
    assert read_values["count"] == 0
    assert stored == {"title": "old"}


def test_unsupported_development_layout_is_rejected_read_only(tmp_path: Path) -> None:
    catalog = _catalog()
    root = tmp_path / "development-v1"
    root.mkdir()
    marker = root / "manifest.json"
    marker.write_text('{"schema_version":1}', encoding="utf-8")
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    with pytest.raises(catalog.CatalogMigrationRequiredError):
        catalog.inspect_store_layout(root)

    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert after == before


def test_future_and_foreign_layouts_require_migration_or_rebuild_without_mutation(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    root = tmp_path / "future"
    root.mkdir()
    marker = root / "store-format.json"
    marker.write_text('{"store_format_version":99}', encoding="utf-8")

    with pytest.raises(catalog.CatalogMigrationRequiredError):
        catalog.inspect_store_layout(root)

    assert marker.read_text(encoding="utf-8") == '{"store_format_version":99}'


def test_sqlite_format_two_initialization_is_explicit_and_idempotent(tmp_path: Path) -> None:
    """Current authority identity is created once without a payload write."""
    from cacheness.storage.sqlite_lifecycle_authority import (
        AUTHORITY_RELATIVE_PATH,
        SQLITE_APPLICATION_ID,
        SQLITE_USER_VERSION,
        SqliteLifecycleAuthority,
    )

    authority = SqliteLifecycleAuthority.for_root(tmp_path)
    try:
        authority.initialize()
        database = tmp_path / AUTHORITY_RELATIVE_PATH
        assert database.is_file()
        with sqlite3.connect(database) as connection:
            before_version = connection.execute("PRAGMA data_version").fetchone()[0]
            assert connection.execute("PRAGMA application_id").fetchone()[0] == SQLITE_APPLICATION_ID
            assert connection.execute("PRAGMA user_version").fetchone()[0] == SQLITE_USER_VERSION
            assert connection.execute("SELECT count(*) FROM entries").fetchone()[0] == 0
        before = database.read_bytes()

        authority.initialize()

        with sqlite3.connect(database) as connection:
            after_version = connection.execute("PRAGMA data_version").fetchone()[0]
        assert database.read_bytes() == before
        assert after_version == before_version
    finally:
        authority.close()


def test_current_manifest_keeps_version_dimensions_independent_and_authenticated() -> None:
    manifest = import_module("cacheness.storage.manifest")
    versions = manifest.StoreVersionDimensions(
        store_epoch=11,
        manifest_schema_version=3,
        sqlite_user_version=7,
        payload_format_version=19,
        store_format_version=2,
    )
    current = manifest.BlobManifest(
        versions=versions,
        key="key",
        generation="generation",
        locator="generations/key/generation",
        handler_type="object",
        payload_format="pickle",
        digest="a" * 64,
        byte_size=12,
        created_at="2026-09-08T00:00:00+00:00",
        catalog_schema_id="example",
        catalog_schema_revision=5,
        catalog_schema_fingerprint="b" * 64,
        catalog_values={"count": 0},
        catalog_presence=("count",),
        user_metadata={"opaque": {"kept": True}},
        handler_metadata={"codec": "native"},
    )

    signed = manifest.sign_current_manifest(current, b"x" * 32)
    restored = manifest.BlobManifest.from_canonical_bytes(signed.canonical_bytes())
    manifest.verify_current_manifest(restored, b"x" * 32)

    assert restored.versions.store_epoch == 11
    assert restored.versions.manifest_schema_version == 3
    assert restored.versions.sqlite_user_version == 7
    assert restored.payload_format_version == 19
    assert isinstance(restored.catalog_values, MappingProxyType)


def test_blob_receipt_is_frozen_and_not_the_legacy_entry_info_alias() -> None:
    storage = import_module("cacheness.storage")
    from cacheness.storage.lifecycle_authority import EntryExpectation

    receipt = storage.BlobReceipt(
        operation_id="operation",
        key="key",
        generation="generation",
        locator="generations/key/generation",
        expectation=EntryExpectation.absent(),
        catalog_revision=4,
        projections={"json": {"state": "pending"}},
    )

    assert receipt.generation == "generation"
    assert "BlobEntryInfo" not in storage.__all__
    with pytest.raises((AttributeError, TypeError)):
        receipt.projections["json"] = {"state": "complete"}
