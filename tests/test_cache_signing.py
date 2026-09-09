"""Manifest-authenticity regressions for the canonical BlobStore lifecycle."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from cacheness.error_handling import (
    CacheBlobManifestUnauthenticatedError,
    CacheReason,
)
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.manifest import BlobManifest


def _signed_store(root: Path) -> BlobStore:
    """Build a direct store whose authority authenticates its manifests."""

    return BlobStore(
        StoreTopology(
            payload=BackendRef(
                name="filesystem", options={"base_dir": root / "payloads"}
            ),
            authority=BackendRef(name="sqlite", options={"root": root / "authority"}),
        ),
        cache_dir=root / "cache",
    )


def _tampered_authority_entry(store: BlobStore, key: str) -> object:
    """Return an authority record with a digest-consistent, invalid HMAC."""

    entry = store.lifecycle_authority.read_entry(key)
    assert entry is not None
    manifest = BlobManifest.from_canonical_bytes(entry.manifest)
    raw_manifest = manifest.with_signature("0" * 64).canonical_bytes()
    return replace(
        entry,
        manifest=raw_manifest,
        expectation=replace(
            entry.expectation,
            manifest_digest=hashlib.sha256(raw_manifest).hexdigest(),
        ),
    )


def test_tampered_manifest_is_typed_and_never_removes_authority_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid HMAC fails before reads while retained authority evidence is intact."""
    store = _signed_store(tmp_path / "manifest-authenticity")
    try:
        value = {"message": "authoritative", "values": [1, 2, 3]}
        key = store.put(value, key="signed-record")
        assert store.get(key) == value

        original_entry = store.lifecycle_authority.read_entry(key)
        assert original_entry is not None
        manifest_key_path = store.cache_dir / "blob_manifest_hmac_key.bin"
        original_key = manifest_key_path.read_bytes()
        tampered_entry = _tampered_authority_entry(store, key)
        original_read_entry = store.lifecycle_authority.read_entry

        monkeypatch.setattr(
            store.lifecycle_authority,
            "read_entry",
            lambda requested_key: (
                tampered_entry
                if requested_key == key
                else original_read_entry(requested_key)
            ),
        )

        def forbidden(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("unauthenticated manifests must not read payloads")

        monkeypatch.setattr(store.guarded_handler_io, "open_snapshot", forbidden)
        monkeypatch.setattr(store.handlers, "resolve_payload_contract", forbidden)

        with pytest.raises(CacheBlobManifestUnauthenticatedError) as error:
            store.get(key)

        assert error.value.context["reason"] == (
            CacheReason.BLOB_MANIFEST_UNAUTHENTICATED.value
        )
        assert original_read_entry(key) == original_entry
        assert manifest_key_path.read_bytes() == original_key
    finally:
        store.close()
