"""Red contracts for the native Phase 4 catalog schema and format boundary."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

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
