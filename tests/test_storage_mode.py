"""
Tests for storage_mode feature (CACHE-tkg epic).

Validates that UnifiedCache with storage_mode=True:
- Disables TTL, eviction, stats, and auto-cleanup
- Accepts hash_key alias on all public methods
- Supports content-addressable key generation
- Preserves entries on errors (no auto-delete)
"""

import time
import pytest
import numpy as np

from cacheness.config import CacheConfig, CacheMetadataConfig
from cacheness.core import UnifiedCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage_cache(tmp_path):
    """UnifiedCache in storage_mode."""
    config = CacheConfig(
        cache_dir=str(tmp_path / "store"),
        storage_mode=True,
    )
    return UnifiedCache(config=config)


@pytest.fixture
def normal_cache(tmp_path):
    """Normal UnifiedCache for comparison."""
    config = CacheConfig(
        cache_dir=str(tmp_path / "cache"),
        default_ttl_seconds=1,  # 1-second TTL for expiration tests
        max_cache_size_mb=2000,
        enable_cache_stats=True,
    )
    return UnifiedCache(config=config)


# ---------------------------------------------------------------------------
# tkg.1 — storage_mode config
# ---------------------------------------------------------------------------


class TestStorageModeConfig:
    """storage_mode=True correctly overrides cache-specific config."""

    def test_storage_mode_defaults(self, storage_cache):
        cfg = storage_cache.config
        assert cfg.storage_mode is True
        assert cfg.metadata.default_ttl_seconds is None
        assert cfg.storage.max_cache_size_mb is None
        assert cfg.storage.cleanup_on_init is False
        assert cfg.metadata.enable_cache_stats is False
        assert cfg.metadata.auto_cleanup_expired is False

    def test_normal_mode_defaults(self, normal_cache):
        cfg = normal_cache.config
        assert cfg.storage_mode is False
        assert cfg.metadata.default_ttl_seconds == 1
        assert cfg.storage.max_cache_size_mb == 2000
        assert cfg.metadata.enable_cache_stats is True

    def test_storage_mode_init_no_errors(self, tmp_path):
        """storage_mode=True can be created without errors."""
        config = CacheConfig(cache_dir=str(tmp_path / "s"), storage_mode=True)
        cache = UnifiedCache(config=config)
        assert cache is not None


# ---------------------------------------------------------------------------
# tkg.2 — None TTL validation
# ---------------------------------------------------------------------------


class TestNoneTTLValidation:
    """CacheMetadataConfig accepts None TTL without raising."""

    def test_none_ttl_accepted(self):
        meta = CacheMetadataConfig(default_ttl_seconds=None)
        assert meta.default_ttl_seconds is None

    def test_zero_ttl_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            CacheMetadataConfig(default_ttl_seconds=0)

    def test_negative_ttl_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            CacheMetadataConfig(default_ttl_seconds=-5)


# ---------------------------------------------------------------------------
# tkg.3 — No TTL expiration in storage mode
# ---------------------------------------------------------------------------


class TestNoTTLExpiration:
    """Entries in storage mode never expire."""

    def test_entries_never_expire(self, storage_cache):
        storage_cache.put("hello", cache_key="k1")
        # Should always retrieve even after "time passes"
        assert storage_cache.get(cache_key="k1") == "hello"

    def test_cleanup_expired_is_noop(self, storage_cache):
        """_cleanup_expired does nothing when TTL is None."""
        storage_cache.put("data", cache_key="k2")
        storage_cache._cleanup_expired()
        assert storage_cache.exists(cache_key="k2")

    def test_normal_cache_expires(self, normal_cache):
        """Sanity: normal cache with 1s TTL expires entries."""
        normal_cache.put("data", cache_key="expire-me")
        time.sleep(1.5)
        # get() default ttl_seconds=None means "never expire" when not
        # specified, so we pass the configured TTL explicitly to test
        # expiration behaviour.
        result = normal_cache.get(cache_key="expire-me", ttl_seconds=1)
        assert result is None


# ---------------------------------------------------------------------------
# tkg.4 — No size-based eviction in storage mode
# ---------------------------------------------------------------------------


class TestNoSizeEviction:
    """_enforce_size_limit is a no-op when max_cache_size_mb is None."""

    def test_enforce_size_limit_noop(self, storage_cache):
        storage_cache.put("data", cache_key="big")
        # Should not raise even though max_cache_size_mb is None
        storage_cache._enforce_size_limit()
        assert storage_cache.exists(cache_key="big")


# ---------------------------------------------------------------------------
# tkg.5 — Stats disabled in storage mode
# ---------------------------------------------------------------------------


class TestStatsDisabled:
    """Stats tracking is off in storage mode."""

    def test_no_stats_tracked(self, storage_cache):
        storage_cache.put("data", cache_key="s1")
        storage_cache.get(cache_key="s1")
        storage_cache.get(cache_key="nonexistent")

        stats = storage_cache.metadata_backend.get_stats()
        assert stats.get("cache_hits", 0) == 0
        assert stats.get("cache_misses", 0) == 0

    def test_stats_tracked_in_normal_mode(self, normal_cache):
        normal_cache.put("data", cache_key="s2")
        normal_cache.get(cache_key="s2")
        normal_cache.get(cache_key="nonexistent")

        stats = normal_cache.metadata_backend.get_stats()
        assert stats.get("cache_hits", 0) >= 1
        assert stats.get("cache_misses", 0) >= 1


# ---------------------------------------------------------------------------
# tkg.6 — No auto-delete in storage mode
# ---------------------------------------------------------------------------


class TestNoAutoDelete:
    """Storage mode preserves entries even on errors."""

    def test_integrity_failure_preserves_entry(self, storage_cache):
        """Corrupt file hash → entry preserved in storage mode."""
        storage_cache.put("data", cache_key="corrupt")

        # Tamper with stored hash to trigger integrity failure
        entry = storage_cache.metadata_backend.get_entry("corrupt")
        if entry:
            metadata = entry.get("metadata", {})
            if metadata.get("file_hash"):
                metadata["file_hash"] = "000000bad"
                storage_cache.metadata_backend.put_entry("corrupt", entry)

        # get() should return None (bad integrity) but NOT delete entry
        result = storage_cache.get(cache_key="corrupt")
        # Entry should still exist in metadata
        assert storage_cache.metadata_backend.get_entry("corrupt") is not None


# ---------------------------------------------------------------------------
# tkg.7 — hash_key alias
# ---------------------------------------------------------------------------


class TestHashKeyAlias:
    """hash_key works as alias for cache_key on all public methods."""

    def test_put_get_with_hash_key(self, storage_cache):
        storage_cache.put("hello world", hash_key="hk1")
        assert storage_cache.get(hash_key="hk1") == "hello world"

    def test_exists_with_hash_key(self, storage_cache):
        storage_cache.put([1, 2, 3], hash_key="hk2")
        assert storage_cache.exists(hash_key="hk2") is True
        assert storage_cache.exists(hash_key="nonexistent") is False

    def test_get_metadata_with_hash_key(self, storage_cache):
        storage_cache.put("meta-data", hash_key="hk3")
        meta = storage_cache.get_metadata(hash_key="hk3")
        assert meta is not None
        assert meta["cache_key"] == "hk3"

    def test_get_with_metadata_with_hash_key(self, storage_cache):
        storage_cache.put({"key": "value"}, hash_key="hk4")
        result = storage_cache.get_with_metadata(hash_key="hk4")
        assert result is not None
        data, meta = result
        assert data == {"key": "value"}

    def test_invalidate_with_hash_key(self, storage_cache):
        storage_cache.put("remove me", hash_key="hk5")
        assert storage_cache.exists(hash_key="hk5")
        storage_cache.invalidate(hash_key="hk5")
        assert not storage_cache.exists(hash_key="hk5")

    def test_touch_with_hash_key(self, storage_cache):
        storage_cache.put("touched", hash_key="hk6")
        assert storage_cache.touch(hash_key="hk6") is True
        assert storage_cache.touch(hash_key="nonexistent") is False

    def test_update_data_with_hash_key(self, storage_cache):
        storage_cache.put("original", hash_key="hk7")
        success = storage_cache.update_data("updated", hash_key="hk7")
        assert success is True
        assert storage_cache.get(hash_key="hk7") == "updated"

    def test_cache_key_and_hash_key_conflict(self, storage_cache):
        """Cannot specify both cache_key and hash_key."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            storage_cache.get(cache_key="a", hash_key="b")

    def test_hash_key_interop_with_cache_key(self, storage_cache):
        """hash_key and cache_key refer to the same entry."""
        storage_cache.put("shared", hash_key="shared-key")
        assert storage_cache.get(cache_key="shared-key") == "shared"


# ---------------------------------------------------------------------------
# tkg.8 — Content-addressable key generation
# ---------------------------------------------------------------------------


class TestContentKey:
    """UnifiedCache.content_key() produces deterministic content-based keys."""

    def test_same_data_same_key(self):
        data = {"experiment": "test", "values": [1, 2, 3]}
        key1 = UnifiedCache.content_key(data)
        key2 = UnifiedCache.content_key(data)
        assert key1 == key2

    def test_different_data_different_key(self):
        key1 = UnifiedCache.content_key([1, 2, 3])
        key2 = UnifiedCache.content_key([4, 5, 6])
        assert key1 != key2

    def test_key_length(self):
        key = UnifiedCache.content_key("hello")
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_content_key_with_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        key = UnifiedCache.content_key(arr)
        assert len(key) == 16

    def test_content_key_roundtrip(self, storage_cache):
        """Store and retrieve using content-derived key."""
        data = {"deduplicated": True, "value": 42}
        key = UnifiedCache.content_key(data)

        storage_cache.put(data, hash_key=key)
        retrieved = storage_cache.get(hash_key=key)
        assert retrieved == data

    def test_deduplication(self, storage_cache):
        """Storing same content twice uses the same key (dedup)."""
        data = [10, 20, 30]
        key1 = UnifiedCache.content_key(data)
        key2 = UnifiedCache.content_key(data)

        storage_cache.put(data, hash_key=key1)
        # Second put with same key is effectively a no-op overwrite
        storage_cache.put(data, hash_key=key2)

        assert storage_cache.get(hash_key=key1) == data
        assert key1 == key2


# ---------------------------------------------------------------------------
# Integration: storage mode end-to-end
# ---------------------------------------------------------------------------


class TestStorageModeEndToEnd:
    """Full workflow: create storage, store, retrieve, verify durability."""

    def test_store_and_retrieve_multiple_types(self, storage_cache):
        """Storage mode handles various data types."""
        storage_cache.put("text", hash_key="t-text")
        storage_cache.put(42, hash_key="t-int")
        storage_cache.put([1, 2, 3], hash_key="t-list")
        storage_cache.put({"a": 1}, hash_key="t-dict")
        storage_cache.put(np.array([1, 2]), hash_key="t-numpy")

        assert storage_cache.get(hash_key="t-text") == "text"
        assert storage_cache.get(hash_key="t-int") == 42
        assert storage_cache.get(hash_key="t-list") == [1, 2, 3]
        assert storage_cache.get(hash_key="t-dict") == {"a": 1}
        np.testing.assert_array_equal(
            storage_cache.get(hash_key="t-numpy"), np.array([1, 2])
        )

    def test_storage_mode_with_on_parameter(self, storage_cache):
        """on parameter still works for key derivation in storage mode."""
        storage_cache.put("result", on={"exp": "001", "run": 5})
        assert storage_cache.get(on={"exp": "001", "run": 5}) == "result"

    def test_entries_persist_after_cleanup_calls(self, storage_cache):
        """Calling cleanup methods doesn't remove entries in storage mode."""
        storage_cache.put("persistent", hash_key="persist")
        storage_cache._cleanup_expired()
        storage_cache._enforce_size_limit()
        assert storage_cache.get(hash_key="persist") == "persistent"


# ---------------------------------------------------------------------------
# Signing in storage mode (CACHE-32o)
# ---------------------------------------------------------------------------


class TestStorageModeSigning:
    """Storage-mode put() signs entries so get() accepts them."""

    @pytest.fixture
    def signed_storage_cache(self, tmp_path):
        """Storage-mode cache with signing enabled and unsigned entries rejected."""
        from cacheness.config import SecurityConfig

        config = CacheConfig(
            cache_dir=str(tmp_path / "signed_store"),
            storage_mode=True,
            security=SecurityConfig(
                enable_entry_signing=True,
                allow_unsigned_entries=False,
            ),
        )
        return UnifiedCache(config=config)

    def test_put_get_roundtrip_with_signing(self, signed_storage_cache):
        """put() signs the entry; get() verifies and returns data."""
        signed_storage_cache.put("hello", hash_key="sig-test")
        result = signed_storage_cache.get(hash_key="sig-test")
        assert result == "hello"

    def test_get_with_metadata_roundtrip_with_signing(self, signed_storage_cache):
        """get_with_metadata() also succeeds with signed storage-mode entries."""
        signed_storage_cache.put(42, hash_key="sig-meta")
        data, meta = signed_storage_cache.get_with_metadata(hash_key="sig-meta")
        assert data == 42
        assert "entry_signature" in meta.get("metadata", {})

    def test_signature_stored_in_metadata(self, signed_storage_cache):
        """entry_signature is present in the stored metadata."""
        signed_storage_cache.put([1, 2, 3], hash_key="sig-check")
        entry = signed_storage_cache.metadata_backend.get_entry("sig-check")
        assert entry is not None
        assert "entry_signature" in entry.get("metadata", {})

    def test_numpy_roundtrip_with_signing(self, signed_storage_cache):
        """NumPy arrays round-trip through signed storage mode."""
        arr = np.array([1.0, 2.0, 3.0])
        signed_storage_cache.put(arr, hash_key="sig-np")
        result = signed_storage_cache.get(hash_key="sig-np")
        np.testing.assert_array_equal(result, arr)
