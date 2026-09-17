"""Stored-format boundaries for the current direct BlobStore contract.

Pre-production layouts are intentionally not reopened through adapters. The
tests preserve corruption and migration-required evidence while proving the
current format can be reopened without a hidden runtime upgrade.
"""

from __future__ import annotations

from pathlib import Path
import stat

import pytest

from cacheness.config import CacheConfig, CacheStorageConfig
from cacheness.core import UnifiedCache
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
from cacheness.storage.migration import inspect_migration_store
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


def _tree_snapshot(root: Path) -> dict[Path, tuple[int, bytes | str | None]]:
    """Capture every disposable fixture node without following symlinks."""
    snapshot: dict[Path, tuple[int, bytes | str | None]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            content: bytes | str | None = path.read_bytes()
        elif stat.S_ISLNK(mode):
            content = path.readlink().as_posix()
        else:
            content = None
        snapshot[relative] = (stat.S_IFMT(mode), content)
    return snapshot


def _memory_topology() -> StoreTopology:
    """Create the explicitly same-process topology used for memory validation."""
    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def _open_unsupported_entry(store: BlobStore) -> None:
    """Exercise the ordinary entry-open boundary for an unsupported store."""
    with store.open_entry("ordinary-read"):
        pass


def _compose_unsupported_cache(root: Path, store: BlobStore) -> None:
    """Exercise the policy facade without giving it migration authority."""
    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=root)),
        store=store,
    )
    try:
        cache.initialize()
        result = cache.lookup("ordinary-read")
        assert result.cause is not None
        assert isinstance(result.cause, CacheBlobMigrationRequiredError)
    finally:
        cache.close()


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


def test_current_store_reopen_preserves_handler_and_version_identity(tmp_path: Path) -> None:
    """A protocol rename cannot rewrite a committed local store's identity."""
    root = tmp_path / "identity-reopen"
    value = {"answer": 42, "labels": ["persisted", True]}

    with BlobStore(_local_topology(root), cache_dir=root) as store:
        store.initialize()
        receipt = store.put_entry(value, key="identity-entry")
        entry = store.get_entry_info(receipt.key)
        assert entry is not None
        before = (
            entry.manifest.handler_type,
            entry.manifest.payload_format,
            entry.manifest.payload_format_version,
            entry.manifest.versions.to_mapping(),
        )
        before_tree = _tree_snapshot(root)

    with BlobStore(_local_topology(root), cache_dir=root) as reopened:
        reopened.initialize()
        entry = reopened.get_entry_info(receipt.key)
        assert entry is not None
        after = (
            entry.manifest.handler_type,
            entry.manifest.payload_format,
            entry.manifest.payload_format_version,
            entry.manifest.versions.to_mapping(),
        )
        assert reopened.get(receipt.key) == value
        assert after == before
        assert _tree_snapshot(root) == before_tree


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


@pytest.mark.parametrize(
    ("fixture_name", "relative_path", "contents"),
    (
        ("historical", "manifest.json", b'{"schema_version":1}'),
        ("future", "store-format.json", b'{"store_format_version":99}'),
        ("retired-scheduler", ".cacheness-clear-journal-v1.json", b"retired"),
        ("corrupt-authority", AUTHORITY_RELATIVE_PATH, b"not a sqlite database"),
        ("foreign-root", "foreign-payload.bin", b"foreign"),
    ),
)
def test_ordinary_entry_points_never_adopt_or_modify_unsupported_roots(
    tmp_path: Path,
    fixture_name: str,
    relative_path: Path | str,
    contents: bytes,
) -> None:
    """Unsupported evidence stays byte-identical across all ordinary boundaries."""
    root = tmp_path / fixture_name
    path = root / relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(contents)
    before = _tree_snapshot(root)

    try:
        store = BlobStore(_local_topology(root), cache_dir=root)
    except CacheBlobMigrationRequiredError:
        assert _tree_snapshot(root) == before
        return

    try:
        assert _tree_snapshot(root) == before
        for ordinary_operation in (
            lambda: store.initialize(),
            lambda: _open_unsupported_entry(store),
            lambda: store.get("ordinary-read"),
            lambda: store.reconcile(),
        ):
            with pytest.raises(CacheBlobMigrationRequiredError):
                ordinary_operation()
            assert _tree_snapshot(root) == before
        _compose_unsupported_cache(root, store)
        assert _tree_snapshot(root) == before
    finally:
        store.close()


def test_initialized_current_memory_and_sqlite_stores_repeat_initialize_by_validation(
    tmp_path: Path,
) -> None:
    """Current stores preserve canonical entries and bytes through repeat initialization."""
    memory_store = BlobStore(_memory_topology(), cache_dir=tmp_path / "memory")
    try:
        memory_store.initialize()
        memory_receipt = memory_store.put_entry({"answer": 42}, key="memory-entry")
        memory_store.initialize()
        assert memory_store.get(memory_receipt.key) == {"answer": 42}
    finally:
        memory_store.close()

    root = tmp_path / "sqlite"
    with BlobStore(_local_topology(root), cache_dir=root) as store:
        store.initialize()
        receipt = store.put_entry({"answer": 42}, key="sqlite-entry")
        before = _tree_snapshot(root)
        inspection = inspect_migration_store(store)
        store.initialize()
        assert store.get(receipt.key) == {"answer": 42}
        assert inspection.source_identity.authority_kind == "sqlite"
        assert inspection.totals.total_entries == 1
        assert _tree_snapshot(root) == before

    before_reopen = _tree_snapshot(root)
    with BlobStore(_local_topology(root), cache_dir=root) as reopened:
        reopened.initialize()
        assert reopened.get(receipt.key) == {"answer": 42}
        assert _tree_snapshot(root) == before_reopen
