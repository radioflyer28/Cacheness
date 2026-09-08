"""Current direct-read contracts for the composed BlobStore topology."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobStoreClosedError
from cacheness.storage import BlobReceipt, BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.legacy_manifest import LegacyManifestRecognitionError
from cacheness.storage.manifest import BlobManifest, verify_current_manifest


def _topology(root: Path) -> StoreTopology:
    """Build the supported local filesystem/SQLite authority pairing."""
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


def _store(root: Path) -> BlobStore:
    return BlobStore(_topology(root), cache_dir=root)


def test_composed_store_reopens_one_authenticated_canonical_generation(tmp_path: Path) -> None:
    """A current descriptor, rather than a metadata backend, determines a read."""
    root = tmp_path / "reopen"
    first = _store(root)
    try:
        receipt = first.put_entry(
            {"answer": 42}, key="read-key", metadata={"label": "answer"}
        )
        assert isinstance(receipt, BlobReceipt)
    finally:
        first.close()

    reopened = _store(root)
    try:
        committed = reopened.lifecycle_authority.read_entry(receipt.key)
        assert committed is not None
        descriptor = BlobManifest.from_canonical_bytes(committed.manifest)
        verify_current_manifest(descriptor, reopened._authority_manifest_key())
        assert descriptor.generation == receipt.generation
        assert reopened.get(receipt.key) == {"answer": 42}
    finally:
        reopened.close()


def test_absence_is_distinct_from_a_present_none_payload(tmp_path: Path) -> None:
    """The direct store still distinguishes absence before later cache policy."""
    store = _store(tmp_path / "absence")
    try:
        receipt = store.put_entry(None, key="present-none")

        assert store.get("missing") is None
        assert store.exists("missing") is False
        assert store.get(receipt.key) is None
        assert store.exists(receipt.key) is True
    finally:
        store.close()


def test_metadata_update_reauthenticates_a_new_current_generation(tmp_path: Path) -> None:
    """A metadata patch promotes an immutable descriptor without replacing payload bytes."""
    store = _store(tmp_path / "metadata")
    try:
        receipt = store.put_entry("value", key="metadata-key", metadata={"phase": 1})
        assert store.update_metadata(receipt.key, {"phase": 2, "label": "current"})

        committed = store.lifecycle_authority.read_entry(receipt.key)
        assert committed is not None
        descriptor = BlobManifest.from_canonical_bytes(committed.manifest)
        verify_current_manifest(descriptor, store._authority_manifest_key())
        assert descriptor.generation != receipt.generation
        assert descriptor.user_metadata == {"phase": 2, "label": "current"}
        assert store.get(receipt.key) == "value"
    finally:
        store.close()


def test_unsupported_legacy_layout_is_rejected_without_mutation(tmp_path: Path) -> None:
    """Layout inspection never opens, upgrades, or repairs a legacy root."""
    root = tmp_path / "legacy-layout"
    root.mkdir()
    legacy = root / "provenance.json"
    legacy.write_text('{"source":"development-layout"}', encoding="utf-8")
    before = legacy.read_bytes()

    with pytest.raises(LegacyManifestRecognitionError):
        _store(root)

    assert legacy.read_bytes() == before
    assert not (root / ".cacheness" / "lifecycle-authority-v1.sqlite3").exists()


def test_close_prevents_later_direct_reads_without_reopening_authority(tmp_path: Path) -> None:
    """Close remains an admission boundary for the composed store."""
    store = _store(tmp_path / "closed")
    store.close()

    with pytest.raises(CacheBlobStoreClosedError):
        store.get("after-close")
