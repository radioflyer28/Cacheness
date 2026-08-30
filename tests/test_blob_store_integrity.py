"""Adversarial integrity contracts for canonical BlobStore records."""

from __future__ import annotations

import os

import pytest

from cacheness.error_handling import CacheManifestIntegrityError, CacheReason
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.integrity import (
    ManifestKeyError,
    ManifestKeyProvider,
    sign_hmac_sha256,
    verify_hmac_sha256,
)


_KEY = b"canonical-manifest-key-material-32"


def test_strict_provider_requires_explicit_initialization(tmp_path):
    """An absent file key cannot turn a read/reopen into a new trust root."""
    key_path = tmp_path / "blob_manifest_hmac_key.bin"
    provider = ManifestKeyProvider(key_path)

    with pytest.raises(ManifestKeyError) as error:
        provider.get_key()

    assert error.value.context["reason"] == CacheReason.MANIFEST_SIGNING_KEY_INVALID.value
    assert not key_path.exists()

    provider.initialize_new_store()
    assert provider.get_key() == key_path.read_bytes()
    assert len(key_path.read_bytes()) == 32


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and mode contract")
def test_strict_provider_rejects_symlink_and_unsafe_permissions(tmp_path):
    """A file-backed key is trusted only after no-follow regular-file attestation."""
    target = tmp_path / "target-key.bin"
    target.write_bytes(_KEY)
    target.chmod(0o600)
    key_path = tmp_path / "blob_manifest_hmac_key.bin"
    key_path.symlink_to(target)

    with pytest.raises(ManifestKeyError) as symlink_error:
        ManifestKeyProvider(key_path).get_key()

    assert symlink_error.value.context["reason"] == (
        CacheReason.MANIFEST_SIGNING_KEY_INVALID.value
    )

    key_path.unlink()
    key_path.write_bytes(_KEY)
    key_path.chmod(0o644)

    with pytest.raises(ManifestKeyError):
        ManifestKeyProvider(key_path).get_key()


def test_canonical_hmac_requires_exact_material_and_rejects_bad_signature(tmp_path):
    """Canonical signing uses exact supplied bytes and a fixed HMAC-SHA256 check."""
    provider = ManifestKeyProvider(tmp_path / "unused-key.bin", key=_KEY)
    payload = b'{"canonical":"manifest"}'
    signature = sign_hmac_sha256(payload, provider.get_key())

    assert verify_hmac_sha256(payload, signature, _KEY)
    assert not verify_hmac_sha256(payload, "0" * 64, _KEY)

    for invalid_key in (b"", b"too-short", b"x" * 33):
        with pytest.raises(ManifestKeyError) as error:
            ManifestKeyProvider(tmp_path / "invalid.bin", key=invalid_key)
        assert error.value.context["reason"] == (
            CacheReason.MANIFEST_SIGNING_KEY_INVALID.value
        )


def test_reopen_with_missing_key_never_creates_a_replacement_key(tmp_path):
    """A signed store with a missing key fails closed without evidence mutation."""
    store = BlobStore(tmp_path)
    store.put({"value": "signed"}, key="signed")
    key_path = tmp_path / "blob_manifest_hmac_key.bin"
    raw_manifest = store.manifest_repository.get_raw("signed")
    assert raw_manifest is not None
    key_path.unlink()
    store.close()

    reopened = BlobStore(tmp_path)
    try:
        with pytest.raises(CacheManifestIntegrityError) as error:
            reopened.get("signed")
    finally:
        reopened.close()

    assert error.value.context["reason"] == CacheReason.MANIFEST_SIGNING_KEY_INVALID.value
    assert not key_path.exists()
    assert raw_manifest == BlobStore(tmp_path).manifest_repository.get_raw("signed")
