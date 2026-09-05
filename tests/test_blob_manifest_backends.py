"""Compatibility backend tests after authority repository retirement."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobBackendError, CacheReason
from cacheness.metadata import InMemoryBackend
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.manifest import BlobManifestV1


@pytest.mark.parametrize("backend", ("json", "sqlite", InMemoryBackend()))
def test_supported_projection_choices_keep_manifests_in_lifecycle_authority(
    tmp_path: Path, backend
) -> None:
    """Compatibility metadata never becomes a second manifest repository."""
    store = BlobStore(tmp_path / str(backend), backend=backend)
    try:
        key = store.put("authority payload", key="authority-key")
        entry = store.lifecycle_authority.read_entry(key)
        assert entry is not None
        assert BlobManifestV1.from_canonical_bytes(entry.manifest).key == key
        assert store.get(key) == "authority payload"
        assert not hasattr(store, "manifest_repository")
    finally:
        store.close()


def test_blob_store_rejects_custom_backend_before_payload_staging(tmp_path: Path) -> None:
    """Capability-shaped projection objects cannot inherit authority guarantees."""

    class CustomMemoryBackend(InMemoryBackend):
        """A deliberately unsupported compatibility projection identity."""

    root = tmp_path / "unsupported"
    with pytest.raises(CacheBlobBackendError) as error:
        BlobStore(root, backend=CustomMemoryBackend())

    assert error.value.context["reason"] == CacheReason.BLOB_BACKEND_FAILURE.value
    assert not list(root.glob("*candidate-*"))
