"""Non-mutating inspection coverage for the bounded migration planner."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobLifecycleConflictError
from cacheness.storage.lifecycle_authority import EntryExpectation, MutationSpec, VerificationProof
from cacheness.storage.memory_lifecycle_authority import InMemoryLifecycleAuthority
from cacheness.storage import BackendRef, BlobStore, StoreTopology
from cacheness.storage.migration import (
    MigrationPlanKind,
    MigrationPlanState,
    inspect_migration_store,
    inspect_store_path,
)
from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority


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


def _commit_opaque_entry(authority: object, key: str, generation: str) -> None:
    """Create an authority row whose manifest remains opaque to inventory reads."""
    prepared = authority.prepare_mutation(
        MutationSpec.create(
            operation_id=f"inventory-{key}",
            key=key,
            generation=generation,
            candidate_locator=f"generations/{generation}.native",
            expected=EntryExpectation.absent(),
            manifest=b"opaque-schema-mismatched-manifest",
        )
    )
    authority.record_verification(
        prepared,
        VerificationProof(digest="a" * 64, byte_size=len(key)),
    )
    authority.promote_mutation(prepared)


@pytest.mark.parametrize("authority_kind", ("memory", "sqlite"))
def test_raw_inventory_pages_every_canonical_entry_without_manifest_filtering(
    tmp_path: Path, authority_kind: str
) -> None:
    """Maintenance sees opaque raw rows that catalog-schema queries must not omit."""
    authority = (
        InMemoryLifecycleAuthority()
        if authority_kind == "memory"
        else SqliteLifecycleAuthority.for_root(tmp_path / "inventory-store")
    )
    _commit_opaque_entry(authority, "entry-a", "generation-a")
    _commit_opaque_entry(authority, "entry-b", "generation-b")

    before = _tree_bytes(tmp_path) if authority_kind == "sqlite" else None
    first = authority.inventory_page(limit=1, work_cap=1024)
    second = authority.inventory_page(cursor=first.next_cursor, limit=1, work_cap=1024)

    assert first.identity.authority_kind == authority_kind
    assert first.identity.capability
    assert first.identity.schema_version >= 0
    assert tuple(entry.key for entry in first.entries + second.entries) == (
        "entry-a",
        "entry-b",
    )
    assert first.entries[0].manifest == b"opaque-schema-mismatched-manifest"
    assert first.exhausted is False
    assert second.exhausted is True
    assert second.next_cursor is None
    if before is not None:
        assert _tree_bytes(tmp_path) == before
    authority.close()


@pytest.mark.parametrize("authority_kind", ("memory", "sqlite"))
def test_inventory_continuation_rejects_revision_drift_without_refreshing(
    tmp_path: Path, authority_kind: str
) -> None:
    """A changed source requires reinspection instead of a silently refreshed page."""
    authority = (
        InMemoryLifecycleAuthority()
        if authority_kind == "memory"
        else SqliteLifecycleAuthority.for_root(tmp_path / "stale-inventory-store")
    )
    _commit_opaque_entry(authority, "entry-a", "generation-a")
    _commit_opaque_entry(authority, "entry-b", "generation-b")
    page = authority.inventory_page(limit=1, work_cap=1024)
    _commit_opaque_entry(authority, "entry-c", "generation-c")

    with pytest.raises(CacheBlobLifecycleConflictError, match="reinspection"):
        authority.inventory_page(cursor=page.next_cursor, limit=1, work_cap=1024)
    authority.close()


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


def test_corrupt_authority_path_is_refused_without_writing(tmp_path: Path) -> None:
    """Unreadable authority evidence is not upgraded or treated as rebuild-ready."""
    root = tmp_path / "corrupt"
    authority = root / ".cacheness" / "lifecycle-authority-v2.sqlite3"
    authority.parent.mkdir(parents=True)
    authority.write_bytes(b"not sqlite")
    before = _tree_bytes(root)

    plan = inspect_store_path(root)

    assert plan.plan_kind is MigrationPlanKind.REFUSED
    assert plan.state is MigrationPlanState.REFUSED
    assert plan.intended_actions == ()
    assert _tree_bytes(root) == before
