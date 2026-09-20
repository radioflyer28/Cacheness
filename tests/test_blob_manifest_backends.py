"""Topology composition tests for canonical manifest authority ownership."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.storage import BlobReceipt, BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.manifest import BlobManifest, verify_current_manifest


def _sqlite_topology(root: Path) -> StoreTopology:
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


def test_store_topology_keeps_one_signed_manifest_in_its_authority(tmp_path: Path) -> None:
    """A composed local store has no alternate metadata authority."""
    root = tmp_path / "sqlite-authority"
    store = BlobStore(_sqlite_topology(root), cache_dir=root)
    try:
        receipt = store.put_entry("authority payload", key="authority-key")
        entry = store.lifecycle_authority.read_entry(receipt.key)

        assert isinstance(receipt, BlobReceipt)
        assert entry is not None
        descriptor = BlobManifest.from_canonical_bytes(entry.manifest)
        verify_current_manifest(descriptor, store._authority_manifest_key())
        assert descriptor.key == receipt.key
        assert descriptor.generation == receipt.generation
        assert store.get(receipt.key) == "authority payload"
    finally:
        store.close()


def test_legacy_backend_selector_is_rejected_before_any_topology_io(tmp_path: Path) -> None:
    """The retired selector cannot be reintroduced as a test-only adapter."""
    root = tmp_path / "unsupported"

    with pytest.raises(TypeError):
        BlobStore(root, backend="json")  # type: ignore[call-arg]

    assert not root.exists()
