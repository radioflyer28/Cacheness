"""LifecycleAuthority clear contracts for canonical BlobStore stores."""

from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobMigrationRequiredError, CacheReason
from cacheness.storage import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology


def _store(root: Path) -> BlobStore:
    """Create the qualified local filesystem/SQLite topology used by this suite."""
    return BlobStore(
        StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        ),
        cache_dir=root,
    )


def test_authority_clear_removes_only_committed_entries(tmp_path):
    """Clear revokes every committed authority entry and its payload."""
    store = _store(tmp_path / "authority-clear")
    try:
        first = store.put("first", key="first")
        second = store.put("second", key="second")

        assert store.clear() == 2
        assert store.get(first) is None
        assert store.get(second) is None
        assert store.get_metadata(first) is None
        assert store.list() == []
        assert store.lifecycle_authority.read_entry(first) is None
        assert store.lifecycle_authority.read_entry(second) is None
    finally:
        store.close()


@pytest.mark.parametrize(
    ("relative_path", "content"),
    (
        (".cacheness-clear-journal-v1.json", b'{"owner":"forged"}'),
        (".cacheness-inventory-v2", b'{"state":"stale"}'),
    ),
)
def test_retired_control_requires_rebuild_without_authority_mutation(
    tmp_path: Path, relative_path: str, content: bytes
) -> None:
    """Known predecessor controls are classified without parsing or repair."""
    root = tmp_path / "retired-control"
    root.mkdir()
    control_path = root / relative_path
    control_path.write_bytes(content)
    before = control_path.read_bytes()

    with pytest.raises(CacheBlobMigrationRequiredError) as raised:
        _store(root)

    assert raised.value.context["reason"] == CacheReason.BLOB_MIGRATION_REQUIRED.value
    assert control_path.read_bytes() == before
    assert not (root / ".cacheness" / "lifecycle-authority-v1.sqlite3").exists()
