"""Tests for AES-256-GCM encryption at rest (Phase 18 / SEC-03)."""

import secrets
import tempfile
import pytest

cryptography = pytest.importorskip("cryptography")

from cacheness.config import CacheConfig, CacheMetadataConfig, CacheStorageConfig, CompressionConfig, SecurityConfig  # noqa: E402
from cacheness.core import UnifiedCache as cacheness  # noqa: E402
from cacheness.encryption import decrypt_blob, derive_encryption_key, encrypt_blob  # noqa: E402
from cacheness.error_handling import CacheIntegrityError  # noqa: E402
from cacheness.interfaces import RotationResult  # noqa: E402
from cacheness.storage.blob_store import BlobStore  # noqa: E402


def _generate_key_file(path):
    """Write 32 random bytes to a file and return the path."""
    path.write_bytes(secrets.token_bytes(32))
    return path


def _make_encrypted_cache(tmp_path, **security_overrides):
    """Create a UnifiedCache with encryption and signing enabled."""
    key_file = tmp_path / "cache_signing_key.bin"
    if not key_file.exists():
        _generate_key_file(key_file)

    defaults = {
        "enable_entry_signing": True,
        "enable_content_encryption": True,
        "encryption_key_file": "cache_signing_key.bin",
        "allow_unsigned_entries": True,
        "delete_invalid_signatures": False,
    }
    defaults.update(security_overrides)
    security = SecurityConfig(**defaults)
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_path)),
        metadata=CacheMetadataConfig(metadata_backend="json"),
        compression=CompressionConfig(use_blosc2_arrays=False),
        security=security,
    )
    return cacheness(config)


def _make_encrypted_blobstore(tmp_path, **overrides):
    """Create a BlobStore with encryption enabled."""
    key_file = tmp_path / "cache_signing_key.bin"
    if not key_file.exists():
        _generate_key_file(key_file)

    security = SecurityConfig(
        enable_entry_signing=True,
        enable_content_encryption=True,
        encryption_key_file="cache_signing_key.bin",
        allow_unsigned_entries=True,
        **overrides,
    )
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_path)),
        compression=CompressionConfig(use_blosc2_arrays=False),
        security=security,
    )
    return BlobStore(
        cache_dir=tmp_path,
        backend="json",
        enable_signing=True,
        config=config,
        namespace="default",
    )


# ── Unit tests for encrypt/decrypt module ──────────────────────────


class TestEncryptionModule:
    """Unit tests for the encryption primitives."""

    def test_encrypt_decrypt_roundtrip(self):
        """encrypt then decrypt returns original bytes."""
        key = secrets.token_bytes(32)
        plaintext = b"hello cacheness encryption!"
        ciphertext, iv, algo = encrypt_blob(plaintext, key)
        result = decrypt_blob(ciphertext, key, iv)
        assert result == plaintext
        assert algo == b"aes-256-gcm"

    def test_encrypt_produces_different_ciphertext(self):
        """Same plaintext encrypted twice produces different ciphertext (random IV)."""
        key = secrets.token_bytes(32)
        plaintext = b"deterministic? no."
        ct1, iv1, _ = encrypt_blob(plaintext, key)
        ct2, iv2, _ = encrypt_blob(plaintext, key)
        assert ct1 != ct2
        assert iv1 != iv2

    def test_decrypt_wrong_key_raises(self):
        """Decrypt with wrong key raises CacheIntegrityError."""
        key1 = secrets.token_bytes(32)
        key2 = secrets.token_bytes(32)
        plaintext = b"secret data"
        ciphertext, iv, _ = encrypt_blob(plaintext, key1)
        with pytest.raises(CacheIntegrityError, match="tampered with or wrong key"):
            decrypt_blob(ciphertext, key2, iv)

    def test_decrypt_tampered_ciphertext_raises(self):
        """Modified ciphertext raises CacheIntegrityError."""
        key = secrets.token_bytes(32)
        plaintext = b"important data"
        ciphertext, iv, _ = encrypt_blob(plaintext, key)
        tampered = bytearray(ciphertext)
        tampered[0] ^= 0xFF
        with pytest.raises(CacheIntegrityError, match="tampered with or wrong key"):
            decrypt_blob(bytes(tampered), key, iv)

    def test_derive_encryption_key_different_namespaces(self):
        """Different namespace_ids produce different keys."""
        master = secrets.token_bytes(32)
        key_a = derive_encryption_key(master, "ns-alpha")
        key_b = derive_encryption_key(master, "ns-beta")
        assert key_a != key_b
        assert len(key_a) == 32
        assert len(key_b) == 32

    def test_derive_encryption_key_deterministic(self):
        """Same inputs produce same key."""
        master = secrets.token_bytes(32)
        key1 = derive_encryption_key(master, "test-ns")
        key2 = derive_encryption_key(master, "test-ns")
        assert key1 == key2


# ── BlobStore integration tests ────────────────────────────────────


class TestBlobStoreEncryption:
    """Tests for BlobStore with encryption enabled."""

    def test_encrypted_put_get_roundtrip(self, tmp_path):
        """put with encryption, get returns original data."""
        store = _make_encrypted_blobstore(tmp_path)
        blob_key = store.put({"msg": "encrypted"}, key="test-data")
        result = store.get(blob_key)
        assert result == {"msg": "encrypted"}

    def test_encrypted_entry_metadata_has_encryption_fields(self, tmp_path):
        """Encrypted entry metadata contains encryption_algorithm and encryption_iv."""
        store = _make_encrypted_blobstore(tmp_path)
        blob_key = store.put("hello", key="enc-meta-test")
        meta = store.get_metadata(blob_key)
        nested = meta.get("metadata", {})
        assert nested.get("encryption_algorithm") == "aes-256-gcm"
        assert "encryption_iv" in nested
        # IV should be a valid 24-char hex string (12 bytes)
        iv_hex = nested["encryption_iv"]
        assert len(bytes.fromhex(iv_hex)) == 12

    def test_unencrypted_entry_readable_with_encryption_enabled(self, tmp_path):
        """Store without encryption, enable encryption, old entry still readable (SC-5)."""
        # First: store without encryption
        config_no_enc = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            compression=CompressionConfig(use_blosc2_arrays=False),
            security=SecurityConfig(enable_entry_signing=False),
        )
        store_plain = BlobStore(
            cache_dir=tmp_path, backend="json", config=config_no_enc
        )
        blob_key = store_plain.put("pre-encryption data", key="legacy")

        # Second: create store with encryption enabled (same dir)
        store_enc = _make_encrypted_blobstore(tmp_path)
        result = store_enc.get(blob_key)
        assert result == "pre-encryption data"

    def test_encrypted_entry_without_key_returns_none(self, tmp_path):
        """Encrypted entry with no encryption key configured returns None."""
        # Store encrypted data
        store_enc = _make_encrypted_blobstore(tmp_path)
        blob_key = store_enc.put("secret", key="locked")

        # Create store WITHOUT encryption key
        config_no_enc = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            compression=CompressionConfig(use_blosc2_arrays=False),
            security=SecurityConfig(enable_entry_signing=False),
        )
        store_plain = BlobStore(
            cache_dir=tmp_path, backend="json", config=config_no_enc
        )
        result = store_plain.get(blob_key)
        assert result is None

    def test_encryption_disabled_by_default(self, tmp_path):
        """Default BlobStore does not encrypt, no encryption_algorithm in metadata."""
        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            compression=CompressionConfig(use_blosc2_arrays=False),
        )
        store = BlobStore(cache_dir=tmp_path, backend="json", config=config)
        blob_key = store.put("not encrypted", key="plain")
        meta = store.get_metadata(blob_key)
        nested = meta.get("metadata", {})
        assert "encryption_algorithm" not in nested

    def test_encrypted_object_get_uses_in_memory_bytes_path(
        self, tmp_path, monkeypatch
    ):
        """Encrypted object reads should not create plaintext temp files."""
        store = _make_encrypted_blobstore(tmp_path)
        blob_key = store.put({"secret": "payload"}, key="bytes-first")

        def fail_named_tempfile(*args, **kwargs):
            raise AssertionError("plaintext temp file should not be created")

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", fail_named_tempfile)

        assert store.get(blob_key) == {"secret": "payload"}


# ── UnifiedCache integration tests ─────────────────────────────────


class TestUnifiedCacheEncryption:
    """Tests for UnifiedCache with encryption enabled."""

    def test_cache_encrypted_put_get_roundtrip(self, tmp_path):
        """UnifiedCache with encryption, put/get works for string data."""
        cache = _make_encrypted_cache(tmp_path)
        cache.put("secure string", on={"prefix": "test"}, description="roundtrip")
        result = cache.get(on={"prefix": "test"})
        assert result == "secure string"

    def test_cache_encrypted_put_get_various_types(self, tmp_path):
        """Encryption works with dict, list, int, float."""
        cache = _make_encrypted_cache(tmp_path)

        cache.put({"key": "value"}, on={"kind": "dict"})
        cache.put([1, 2, 3], on={"kind": "list"})
        cache.put(42, on={"kind": "int"})
        cache.put(3.14, on={"kind": "float"})

        assert cache.get(on={"kind": "dict"}) == {"key": "value"}
        assert cache.get(on={"kind": "list"}) == [1, 2, 3]
        assert cache.get(on={"kind": "int"}) == 42
        assert cache.get(on={"kind": "float"}) == 3.14

    def test_cache_encryption_disabled_by_default(self, tmp_path):
        """Default UnifiedCache has no encryption."""
        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            metadata=CacheMetadataConfig(metadata_backend="json"),
            compression=CompressionConfig(use_blosc2_arrays=False),
        )
        cache = cacheness(config)
        cache.put("no encryption", on={"prefix": "plain"})
        entries = cache.list_entries()
        for e in entries:
            meta = e.get("metadata", {})
            assert "encryption_algorithm" not in meta

    def test_cache_mixed_encrypted_unencrypted(self, tmp_path):
        """Some entries encrypted, some not, all readable."""
        # Store unencrypted entry first
        config_plain = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            metadata=CacheMetadataConfig(metadata_backend="json"),
            compression=CompressionConfig(use_blosc2_arrays=False),
            security=SecurityConfig(enable_entry_signing=False),
        )
        cache_plain = cacheness(config_plain)
        cache_plain.put("old data", on={"mix": "unencrypted"})

        # Now create encrypted cache on same directory
        cache_enc = _make_encrypted_cache(tmp_path)
        cache_enc.put("new data", on={"mix": "encrypted"})

        # Both should be readable
        assert cache_enc.get(on={"mix": "unencrypted"}) == "old data"
        assert cache_enc.get(on={"mix": "encrypted"}) == "new data"

    def test_cache_init_without_cryptography_raises(self, tmp_path, monkeypatch):
        """When cryptography is not importable, enable_content_encryption raises."""
        import cacheness.config as config_mod

        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "cryptography":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        from cacheness.error_handling import CacheConfigurationError

        with pytest.raises(CacheConfigurationError, match="cacheness\\[encryption\\]"):
            SecurityConfig(enable_content_encryption=True)


# ── Key rotation with re-encryption ────────────────────────────────


class TestEncryptionKeyRotation:
    """Tests for key rotation with re-encryption."""

    def test_rotate_key_re_encrypts_entries(self, tmp_path):
        """rotate_key re-encrypts, RotationResult.re_encrypted > 0."""
        cache = _make_encrypted_cache(tmp_path)
        cache.put("secret1", on={"rot": "one"})
        cache.put("secret2", on={"rot": "two"})

        new_key = _generate_key_file(tmp_path / "new_key.bin")
        result = cache.rotate_key(new_key)

        assert isinstance(result, RotationResult)
        assert result.re_encrypted == 2

    def test_rotated_encrypted_entries_readable(self, tmp_path):
        """After rotation, encrypted entries still readable with new key."""
        cache = _make_encrypted_cache(tmp_path)
        cache.put("alpha", on={"rot2": "a"})
        cache.put("beta", on={"rot2": "b"})

        new_key = _generate_key_file(tmp_path / "new_key.bin")
        cache.rotate_key(new_key)

        assert cache.get(on={"rot2": "a"}) == "alpha"
        assert cache.get(on={"rot2": "b"}) == "beta"
