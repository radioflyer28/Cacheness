"""Key rotation tests — verify graceful behavior when signing key is deleted and regenerated."""

import pytest

from cacheness.core import UnifiedCache as cacheness
from cacheness.config import (
    CacheConfig,
    CacheStorageConfig,
    CacheMetadataConfig,
    CompressionConfig,
    SecurityConfig,
)


def _make_signed_cache(tmp_path, **security_overrides):
    """Create a cache with entry signing enabled."""
    defaults = {
        "enable_entry_signing": True,
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


class TestKeyRotation:
    """Tests for signing key deletion and regeneration scenarios."""

    def test_delete_key_and_regenerate(self, tmp_path):
        """After deleting the key file, a new key is auto-generated."""
        cache = _make_signed_cache(tmp_path)
        cache.put("data_before", test_key="before_rotation")

        # Find and delete the key file
        key_file = tmp_path / "cache_signing_key.bin"
        assert key_file.exists()
        old_key = key_file.read_bytes()
        key_file.unlink()

        # Re-create cache — new key auto-generated
        cache2 = _make_signed_cache(tmp_path)
        new_key = (tmp_path / "cache_signing_key.bin").read_bytes()
        assert old_key != new_key, "New key should differ from deleted key"

        # Old entry still readable (allow_unsigned_entries=True,
        # delete_invalid_signatures=False)
        val = cache2.get(test_key="before_rotation")
        assert val == "data_before"

    def test_new_entries_signed_with_new_key(self, tmp_path):
        """After key rotation, new entries are signed with the new key."""
        cache = _make_signed_cache(tmp_path)
        cache.put("old_data", test_key="old_entry")

        # Rotate key
        key_file = tmp_path / "cache_signing_key.bin"
        key_file.unlink()

        cache2 = _make_signed_cache(tmp_path)
        cache2.put("new_data", test_key="new_entry")

        # New entry is readable
        assert cache2.get(test_key="new_entry") == "new_data"

    def test_old_entry_rejected_when_strict(self, tmp_path):
        """With delete_invalid_signatures=True and allow_unsigned_entries=False,
        old entries with invalid signatures are rejected after key rotation."""
        cache = _make_signed_cache(
            tmp_path,
            allow_unsigned_entries=False,
            delete_invalid_signatures=True,
        )
        cache.put("sensitive_data", test_key="sensitive")

        # Rotate key
        key_file = tmp_path / "cache_signing_key.bin"
        key_file.unlink()

        cache2 = _make_signed_cache(
            tmp_path,
            allow_unsigned_entries=False,
            delete_invalid_signatures=True,
        )

        # Old entry should be rejected (signature mismatch with new key)
        val = cache2.get(test_key="sensitive")
        assert val is None, "Entry with invalid signature should be rejected"

    def test_key_rotation_with_mixed_entries(self, tmp_path):
        """After rotation, cache works with mix of old and new entries."""
        cache = _make_signed_cache(tmp_path)
        for i in range(5):
            cache.put(f"old_{i}", test_key=f"entry_{i}")

        # Rotate
        key_file = tmp_path / "cache_signing_key.bin"
        key_file.unlink()

        cache2 = _make_signed_cache(tmp_path)
        for i in range(5, 10):
            cache2.put(f"new_{i}", test_key=f"entry_{i}")

        # Old entries readable (allow_unsigned=True, delete_invalid=False)
        for i in range(5):
            assert cache2.get(test_key=f"entry_{i}") == f"old_{i}"

        # New entries readable
        for i in range(5, 10):
            assert cache2.get(test_key=f"entry_{i}") == f"new_{i}"

    def test_raise_on_key_fallback(self, tmp_path):
        """key_fallback_policy='raise' raises CacheSecurityError on write failure."""
        from cacheness.error_handling import CacheSecurityError
        from cacheness.security import CacheEntrySigner
        from pathlib import Path
        import os

        # Create a read-only directory to force key write failure
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()

        key_path = readonly_dir / "key.bin"

        # On Windows, make dir read-only by removing write permission
        # Use a path that can't be created (nested under non-existent dir
        # inside a file) to force OSError
        blocker_file = readonly_dir / "blocker"
        blocker_file.write_text("block")
        impossible_key = blocker_file / "subdir" / "key.bin"

        with pytest.raises(CacheSecurityError, match="Failed to persist"):
            CacheEntrySigner(
                key_file_path=impossible_key,
                use_in_memory_key=False,
                key_fallback_policy="raise",
            )

    def test_no_raise_on_key_fallback_by_default(self, tmp_path):
        """By default (warn mode), key write failure falls back to in-memory key."""
        from cacheness.security import CacheEntrySigner

        blocker_file = tmp_path / "blocker"
        blocker_file.write_text("block")
        impossible_key = blocker_file / "subdir" / "key.bin"

        # Should NOT raise — falls back to in-memory key with warning
        signer = CacheEntrySigner(
            key_file_path=impossible_key,
            use_in_memory_key=False,
            key_fallback_policy="warn",
        )
        assert signer.secret_key is not None
