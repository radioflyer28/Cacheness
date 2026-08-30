"""Adversarial integrity contracts for canonical BlobStore records."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from cacheness.error_handling import (
    CacheBlobManifestUnauthenticatedError,
    CacheBlobPayloadMissingError,
    CacheBlobPayloadTamperedError,
    CacheManifestIntegrityError,
    CacheReason,
)
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.integrity import (
    ManifestKeyError,
    ManifestKeyProvider,
    sign_hmac_sha256,
    verify_hmac_sha256,
)
from cacheness.storage.manifest import BlobManifestV1


_KEY = b"0123456789abcdef0123456789abcdef"


def test_strict_key_provider_requires_explicit_initialization(tmp_path):
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
def test_strict_key_provider_rejects_symlink_and_unsafe_permissions(tmp_path):
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
    final_store = BlobStore(tmp_path)
    try:
        assert raw_manifest == final_store.manifest_repository.get_raw("signed")
    finally:
        final_store.close()


def _manifest_for(store: BlobStore, key: str) -> BlobManifestV1:
    """Load the test record through the canonical raw manifest boundary."""
    raw_manifest = store.manifest_repository.get_raw(key)
    assert raw_manifest is not None
    return BlobManifestV1.from_canonical_bytes(raw_manifest)


def _replace_signed_manifest(
    store: BlobStore,
    key: str,
    **overrides: object,
) -> BlobManifestV1:
    """Replace one test record with a deliberately altered authenticated manifest."""
    current = _manifest_for(store, key)
    values = current.to_mapping(include_signature=False)
    values.update(overrides)
    altered = BlobManifestV1(**values)
    signed = altered.with_signature(
        sign_hmac_sha256(altered.signing_bytes(), store._manifest_key())
    )
    store.manifest_repository.put_raw(key, signed.canonical_bytes())
    return signed


@pytest.fixture
def signed_store(tmp_path):
    """Create a direct canonical record without retaining any test-global state."""
    store = BlobStore(tmp_path)
    store.put({"value": "verified"}, key="integrity-key")
    try:
        yield store
    finally:
        store.close()


def test_read_authenticates_validates_snapshots_hashes_then_deserializes(
    signed_store: BlobStore,
    monkeypatch: pytest.MonkeyPatch,
):
    """Only an authenticated, declared contract can reach one private snapshot."""
    import cacheness.storage.blob_store as blob_store_module

    store = signed_store
    manifest = _manifest_for(store, "integrity-key")
    handler = store.handlers.get_handler_by_type(manifest.handler_type)
    events: list[str] = []

    original_verify = blob_store_module.verify_hmac_sha256
    original_resolve = store.handlers.resolve_payload_contract
    original_snapshot = store.guarded_handler_io.open_snapshot
    original_digest = blob_store_module.sha256_and_size
    original_get = handler.get

    def verify_spy(*args, **kwargs):
        events.append("authenticate")
        return original_verify(*args, **kwargs)

    def resolve_spy(*args, **kwargs):
        events.append("validate")
        return original_resolve(*args, **kwargs)

    @contextmanager
    def snapshot_spy(*args, **kwargs):
        events.append("snapshot")
        with original_snapshot(*args, **kwargs) as snapshot:
            yield snapshot

    def digest_spy(*args, **kwargs):
        events.append("digest")
        return original_digest(*args, **kwargs)

    def get_spy(*args, **kwargs):
        events.append("handler")
        return original_get(*args, **kwargs)

    monkeypatch.setattr(blob_store_module, "verify_hmac_sha256", verify_spy)
    monkeypatch.setattr(store.handlers, "resolve_payload_contract", resolve_spy)
    monkeypatch.setattr(store.guarded_handler_io, "open_snapshot", snapshot_spy)
    monkeypatch.setattr(blob_store_module, "sha256_and_size", digest_spy)
    monkeypatch.setattr(handler, "get", get_spy)

    assert store.get("integrity-key") == {"value": "verified"}
    assert events == ["authenticate", "validate", "snapshot", "digest", "handler"]
    assert events.count("snapshot") == 1


def test_unauthenticated_manifest_fails_before_snapshot_or_handler(
    signed_store: BlobStore,
    monkeypatch: pytest.MonkeyPatch,
):
    """An absent or wrong HMAC never authorizes locator or handler use."""
    store = signed_store
    manifest = _manifest_for(store, "integrity-key")
    tampered = manifest.with_signature("0" * 64)
    store.manifest_repository.put_raw("integrity-key", tampered.canonical_bytes())
    raw_before = store.manifest_repository.get_raw("integrity-key")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unauthenticated records must not snapshot or deserialize")

    monkeypatch.setattr(store.guarded_handler_io, "open_snapshot", forbidden)
    monkeypatch.setattr(store.handlers, "resolve_payload_contract", forbidden)

    with pytest.raises(CacheBlobManifestUnauthenticatedError) as error:
        store.get("integrity-key")

    assert error.value.context["reason"] == (
        CacheReason.BLOB_MANIFEST_UNAUTHENTICATED.value
    )
    assert store.manifest_repository.get_raw("integrity-key") == raw_before


def test_unsupported_handler_contract_fails_before_snapshot(
    signed_store: BlobStore,
    monkeypatch: pytest.MonkeyPatch,
):
    """Authenticated handler identity is checked without opening payload bytes."""
    store = signed_store
    _replace_signed_manifest(store, "integrity-key", handler_type="unknown-handler")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unsupported handler identity must not snapshot payload")

    monkeypatch.setattr(store.guarded_handler_io, "open_snapshot", forbidden)

    with pytest.raises(CacheManifestIntegrityError):
        store.get("integrity-key")


def test_missing_or_tampered_payload_is_typed_and_does_not_rewrite_evidence(
    signed_store: BlobStore,
    monkeypatch: pytest.MonkeyPatch,
):
    """Payload absence and tampering remain distinct from cache misses and cleanup."""
    store = signed_store
    manifest = _manifest_for(store, "integrity-key")
    payload_path = Path(manifest.locator)
    original_payload = payload_path.read_bytes()
    modified_payload = bytes([original_payload[0] ^ 1]) + original_payload[1:]
    payload_path.write_bytes(modified_payload)
    raw_before = store.manifest_repository.get_raw("integrity-key")
    key_before = (store.cache_dir / "blob_manifest_hmac_key.bin").read_bytes()
    mtime_before = payload_path.stat().st_mtime_ns

    handler = store.handlers.get_handler_by_type(manifest.handler_type)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("tampered payload must not reach the handler")

    monkeypatch.setattr(handler, "get", forbidden)

    with pytest.raises(CacheBlobPayloadTamperedError) as tampered_error:
        store.get("integrity-key")

    assert tampered_error.value.context["reason"] == CacheReason.BLOB_PAYLOAD_TAMPERED.value
    assert payload_path.read_bytes() == modified_payload
    assert payload_path.stat().st_mtime_ns == mtime_before
    assert store.manifest_repository.get_raw("integrity-key") == raw_before
    assert (store.cache_dir / "blob_manifest_hmac_key.bin").read_bytes() == key_before

    payload_path.unlink()
    with pytest.raises(CacheBlobPayloadMissingError) as missing_error:
        store.get("integrity-key")

    assert missing_error.value.context["reason"] == CacheReason.BLOB_PAYLOAD_MISSING.value
    assert not payload_path.exists()
    assert store.manifest_repository.get_raw("integrity-key") == raw_before
