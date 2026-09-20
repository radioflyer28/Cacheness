"""Same-generation conditional authority regressions for composed BlobStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobLifecycleConflictError
from cacheness.storage import BlobReceipt, BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.manifest import BlobManifest, verify_current_manifest


def _topology(root: Path) -> StoreTopology:
    """Return the qualified persistent topology for CAS observations."""
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


def _store(root: Path) -> BlobStore:
    return BlobStore(_topology(root), cache_dir=root)


def test_receipt_binds_one_authenticated_committed_authority_generation(
    tmp_path: Path,
) -> None:
    """A receipt names the exact signed descriptor committed by the authority."""
    root = tmp_path / "projection-store"
    store = _store(root)
    try:
        receipt = store.put_entry(
            "authority payload", key="projection-key", metadata={"tag": "v1"}
        )
        committed = store.lifecycle_authority.read_entry(receipt.key)

        assert isinstance(receipt, BlobReceipt)
        assert committed is not None
        descriptor = BlobManifest.from_canonical_bytes(committed.manifest)
        verify_current_manifest(descriptor, store._authority_manifest_key())
        assert (descriptor.key, descriptor.generation, descriptor.locator) == (
            receipt.key,
            receipt.generation,
            receipt.locator,
        )
    finally:
        store.close()


def test_stale_receipt_cannot_delete_a_newer_committed_generation(
    tmp_path: Path,
) -> None:
    """Exact authority expectations protect a later committed payload."""
    root = tmp_path / "stale-receipt"
    store = _store(root)
    try:
        first = store.put_entry("first", key="cas-key")
        winner = store.put_entry("winner", key="cas-key")

        with pytest.raises(CacheBlobLifecycleConflictError):
            store.delete("cas-key", expected=first.expectation)

        assert store.get("cas-key") == "winner"
        assert winner.generation != first.generation
    finally:
        store.close()


def test_independent_composed_store_observes_the_authoritative_winner(
    tmp_path: Path,
) -> None:
    """Reopen derives canonical state from SQLite, not a repository shape."""
    root = tmp_path / "independent-store"
    store = _store(root)
    try:
        receipt = store.put_entry("winner", key="shared-key")
    finally:
        store.close()

    reopened = _store(root)
    try:
        assert reopened.get("shared-key") == "winner"
        committed = reopened.lifecycle_authority.read_entry(receipt.key)
        assert committed is not None
        assert BlobManifest.from_canonical_bytes(committed.manifest).generation == receipt.generation
    finally:
        reopened.close()
