"""
Cross-Phase Security Integration Tests
========================================

Tests combining features from Phases 15-19 (v0.10.0 Security & Architecture):
- Phase 15: key_fallback_policy (raise/warn/fallback)
- Phase 16: HKDF per-namespace key derivation
- Phase 18: AES-256-GCM encryption at rest
- Phase 19: rotate_key() API

Verifies that these features work correctly when combined, covering
integration paths the individual phase tests don't exercise.
"""

import secrets
import tempfile
from pathlib import Path

import pytest

cryptography = pytest.importorskip("cryptography")

from cacheness.config import CacheConfig, CacheMetadataConfig, CacheStorageConfig, CompressionConfig, SecurityConfig  # noqa: E402
from cacheness.core import UnifiedCache  # noqa: E402


def _generate_key_file(path: Path) -> Path:
    """Write 32 random bytes to a file and return the path."""
    path.write_bytes(secrets.token_bytes(32))
    return path


def _make_integration_cache(tmp_path, **security_overrides):
    """Create a cache with ALL security features enabled (fallback + HKDF + encryption)."""
    key_file = tmp_path / "cache_signing_key.bin"
    if not key_file.exists():
        _generate_key_file(key_file)

    defaults = {
        "enable_entry_signing": True,
        "enable_content_encryption": True,
        "encryption_key_file": "cache_signing_key.bin",
        "key_fallback_policy": "fallback",
        "use_hkdf_derivation": True,
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
    return UnifiedCache(config)


class TestSecurityFeatureIntegration:
    """Integration tests combining key_fallback_policy, HKDF, and encryption."""

    def test_fallback_hkdf_encryption_combined_put_get(self, tmp_path):
        """Full pipeline: fallback policy + HKDF derivation + encryption, all data types round-trip."""
        cache = _make_integration_cache(tmp_path)

        cache.put("hello", key="str-entry")
        cache.put(42, key="int-entry")
        cache.put([1, 2, 3], key="list-entry")
        cache.put({"a": 1}, key="dict-entry")
        cache.put(3.14, key="float-entry")

        assert cache.get(key="str-entry") == "hello"
        assert cache.get(key="int-entry") == 42
        assert cache.get(key="list-entry") == [1, 2, 3]
        assert cache.get(key="dict-entry") == {"a": 1}
        assert cache.get(key="float-entry") == 3.14

    def test_fallback_policy_with_inaccessible_key_still_works(self, tmp_path):
        """key_fallback_policy='fallback' + encryption: works even with inaccessible key path."""
        # Use a deeply nested non-existent path as the key directory
        impossible_key_dir = tmp_path / "nonexistent" / "deep" / "path"
        impossible_key = str(impossible_key_dir / "key.bin")

        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            metadata=CacheMetadataConfig(metadata_backend="json"),
            compression=CompressionConfig(use_blosc2_arrays=False),
            security=SecurityConfig(
                enable_entry_signing=True,
                enable_content_encryption=True,
                encryption_key_file=impossible_key,
                key_fallback_policy="fallback",
                use_hkdf_derivation=True,
                allow_unsigned_entries=True,
                delete_invalid_signatures=False,
            ),
        )
        cache = UnifiedCache(config)

        cache.put("fallback-data", key="fb-test")
        result = cache.get(key="fb-test")
        assert result == "fallback-data"

    def test_rotate_key_with_encryption_enabled(self, tmp_path):
        """rotate_key() correctly re-signs and re-encrypts entries with all features enabled."""
        cache = _make_integration_cache(tmp_path)

        for i in range(10):
            cache.put(f"data-{i}", key=f"entry-{i}")

        new_key_path = tmp_path / "new_key.bin"
        _generate_key_file(new_key_path)

        result = cache.rotate_key(new_key_path)
        assert result.re_signed >= 10
        assert result.re_encrypted >= 10

        # All entries must decrypt correctly with new key
        for i in range(10):
            val = cache.get(key=f"entry-{i}")
            assert val == f"data-{i}", f"entry-{i}: expected 'data-{i}', got {val!r}"

    def test_different_namespaces_isolated_encryption(self, tmp_path):
        """HKDF namespace isolation works with encryption — each namespace has its own derived key."""
        key_file = tmp_path / "cache_signing_key.bin"
        _generate_key_file(key_file)

        def make_ns_cache(namespace):
            security = SecurityConfig(
                enable_entry_signing=True,
                enable_content_encryption=True,
                encryption_key_file="cache_signing_key.bin",
                key_fallback_policy="fallback",
                use_hkdf_derivation=True,
                allow_unsigned_entries=True,
                delete_invalid_signatures=False,
            )
            config = CacheConfig(
                storage=CacheStorageConfig(cache_dir=str(tmp_path)),
                metadata=CacheMetadataConfig(metadata_backend="json"),
                compression=CompressionConfig(use_blosc2_arrays=False),
                security=security,
                namespace=namespace,
            )
            return UnifiedCache(config)

        cache_alpha = make_ns_cache("alpha")
        cache_beta = make_ns_cache("beta")

        cache_alpha.put("ns1-data", key="shared-key")
        cache_beta.put("ns2-data", key="shared-key")

        # Each namespace sees only its own data
        assert cache_alpha.get(key="shared-key") == "ns1-data"
        assert cache_beta.get(key="shared-key") == "ns2-data"

        cache_alpha.close()
        cache_beta.close()

    def test_mixed_encrypted_unencrypted_migration(self, tmp_path):
        """Old unencrypted entries coexist with new encrypted entries (migration path)."""
        key_file = tmp_path / "cache_signing_key.bin"
        _generate_key_file(key_file)

        # Phase 1: create cache WITHOUT encryption, put 5 entries
        config_plain = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            metadata=CacheMetadataConfig(metadata_backend="json"),
            compression=CompressionConfig(use_blosc2_arrays=False),
            security=SecurityConfig(
                enable_entry_signing=False,
                enable_content_encryption=False,
            ),
        )
        cache_plain = UnifiedCache(config_plain)
        for i in range(5):
            cache_plain.put(f"old-{i}", key=f"plain-{i}")
        cache_plain.close()

        # Phase 2: reopen WITH encryption enabled (same directory)
        cache_enc = _make_integration_cache(tmp_path)

        # Old unencrypted entries must still be readable
        for i in range(5):
            val = cache_enc.get(key=f"plain-{i}")
            assert val == f"old-{i}", f"plain-{i}: expected 'old-{i}', got {val!r}"

        # Put 5 new encrypted entries
        for i in range(5):
            cache_enc.put(f"new-{i}", key=f"enc-{i}")

        # All 10 entries readable
        for i in range(5):
            assert cache_enc.get(key=f"plain-{i}") == f"old-{i}"
            assert cache_enc.get(key=f"enc-{i}") == f"new-{i}"

        cache_enc.close()
