"""Non-mutating inspection coverage for the bounded migration planner."""

from __future__ import annotations

from pathlib import Path

from cacheness.storage import BackendRef, BlobStore, StoreTopology
from cacheness.storage.migration import (
    MigrationPlanKind,
    MigrationPlanState,
    inspect_migration_store,
    inspect_store_path,
)


def _local_topology(root: Path) -> StoreTopology:
    """Build the direct filesystem/SQLite topology used by local inspection."""
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


def _tree_bytes(root: Path) -> dict[Path, bytes]:
    """Capture all fixture bytes before and after a read-only inspection."""
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_initialized_sqlite_store_is_current_without_a_format_marker(tmp_path: Path) -> None:
    """Authority-owned current identity is sufficient for non-mutating inspection."""
    root = tmp_path / "current"
    with BlobStore(_local_topology(root), cache_dir=root) as store:
        store.initialize()
        store.put_entry({"answer": 42}, key="current-entry")
        before = _tree_bytes(root)

        plan = inspect_migration_store(store)

        assert not (root / "store-format.json").exists()
        assert plan.plan_kind is MigrationPlanKind.MIGRATION
        assert plan.state is MigrationPlanState.PLANNED
        assert plan.source_identity.authority_kind == "sqlite"
        assert plan.totals.total_entries == 1
        assert _tree_bytes(root) == before


def test_historical_path_is_rebuild_only_without_writing(tmp_path: Path) -> None:
    """A transitional marker remains inspectable but cannot gain a migration override."""
    root = tmp_path / "historical"
    root.mkdir()
    (root / "manifest.json").write_bytes(b'{"schema_version":1}')
    before = _tree_bytes(root)

    plan = inspect_store_path(root)

    assert plan.plan_kind is MigrationPlanKind.REBUILD
    assert plan.state is MigrationPlanState.REFUSED
    assert plan.intended_actions == ("rebuild",)
    assert plan.stopped_worker_acknowledgement_required is True
    assert _tree_bytes(root) == before
