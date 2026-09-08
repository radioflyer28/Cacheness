"""Stored-format boundaries for the current direct BlobStore contract.

Pre-production layouts are intentionally not reopened through adapters. The
tests preserve corruption and migration-required evidence while proving the
current format can be reopened without a hidden runtime upgrade.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.storage import (
    BackendRef,
    BlobStore,
    CacheBlobMigrationRequiredError,
    CatalogField,
    CatalogPredicate,
    CatalogQuery,
    CatalogSchema,
    STORE_FORMAT_VERSION,
    StoreTopology,
)
from cacheness.storage.catalog import CatalogMigrationRequiredError, inspect_store_layout
from cacheness.storage.sqlite_lifecycle_authority import AUTHORITY_RELATIVE_PATH


def _local_topology(root: Path) -> StoreTopology:
    """Create the format-2 filesystem and SQLite topology used by direct stores."""
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


def _schema() -> CatalogSchema:
    return CatalogSchema(
        schema_id="stored-format-regression",
        fields=(CatalogField("label", "string", queryable=True),),
    )


def _tree_bytes(root: Path) -> dict[Path, bytes]:
    """Return all file bytes to prove a rejected layout was not changed."""
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_current_format_two_store_reopens_with_catalog_values_and_payload(tmp_path: Path) -> None:
    """A current format store survives close/reopen without migration work."""
    root = tmp_path / "current"
    schema = _schema()
    with BlobStore(_local_topology(root), cache_dir=root) as store:
        store.initialize()
        receipt = store.put_entry(
            {"answer": 42},
            key="current-entry",
            catalog_schema=schema,
            catalog_values={"label": "current"},
        )

    # Version dimensions remain explicit and independent of a normal reopen.
    # The format-inspection seam classifies foreign/old roots before a future
    # offline migration; the SQLite authority owns the current local layout.
    assert STORE_FORMAT_VERSION == 2

    with BlobStore(_local_topology(root), cache_dir=root) as reopened:
        assert reopened.get(receipt.key) == {"answer": 42}
        page = reopened.query_catalog(
            CatalogQuery(
                predicates=(CatalogPredicate("label", "eq", "current"),)
            ),
            schema=schema,
        )

    assert [(entry.key, dict(entry.values)) for entry in page.entries] == [
        (receipt.key, {"label": "current"})
    ]


@pytest.mark.parametrize(
    ("marker_name", "marker_bytes"),
    (
        ("manifest.json", b'{"schema_version":1}'),
        ("store-format.json", b'{"store_format_version":99}'),
    ),
)
def test_development_and_foreign_markers_require_offline_migration_without_mutation(
    tmp_path: Path, marker_name: str, marker_bytes: bytes
) -> None:
    """Recognized non-current formats fail before a normal open can rewrite them."""
    root = tmp_path / "unsupported"
    root.mkdir()
    marker = root / marker_name
    marker.write_bytes(marker_bytes)
    before = _tree_bytes(root)

    with pytest.raises(CatalogMigrationRequiredError):
        inspect_store_layout(root)

    assert _tree_bytes(root) == before


def test_corrupt_authority_evidence_requires_rebuild_without_implicit_upgrade(
    tmp_path: Path,
) -> None:
    """A corrupt authority is migration-required and remains byte-for-byte intact."""
    root = tmp_path / "corrupt-authority"
    database = root / AUTHORITY_RELATIVE_PATH
    database.parent.mkdir(parents=True)
    database.write_bytes(b"not a sqlite database")
    before = _tree_bytes(root)

    store = BlobStore(_local_topology(root), cache_dir=root)
    try:
        with pytest.raises(CacheBlobMigrationRequiredError):
            store.initialize()
    finally:
        store.close()

    assert _tree_bytes(root) == before


def test_explicit_legacy_inspection_remains_separate_from_normal_open(tmp_path: Path) -> None:
    """Future migration tooling can inspect evidence but normal open never adopts it."""
    root = tmp_path / "legacy-evidence"
    root.mkdir()
    (root / "manifest.json").write_text('{"schema_version":1}', encoding="utf-8")
    before = _tree_bytes(root)

    with pytest.raises(CatalogMigrationRequiredError):
        inspect_store_layout(root)
    store = BlobStore(_local_topology(root), cache_dir=root)
    try:
        with pytest.raises(CacheBlobMigrationRequiredError):
            store.initialize()
    finally:
        store.close()

    assert _tree_bytes(root) == before
