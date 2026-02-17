"""Tests for UnifiedCache dunder methods: __len__, __contains__, __iter__."""

import tempfile
import time
from pathlib import Path

import pytest

from cacheness import CacheConfig, cacheness
from cacheness.config import (
    CacheMetadataConfig,
    CacheStorageConfig,
    CompressionConfig,
    HandlerConfig,
    SerializationConfig,
)


class TestDunderMethods:
    """Test __len__, __contains__, and __iter__ on UnifiedCache."""

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as td:
            yield Path(td) / "dunder_cache"

    @pytest.fixture
    def cache(self, temp_cache_dir):
        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(temp_cache_dir)),
            metadata=CacheMetadataConfig(metadata_backend="sqlite"),
            compression=CompressionConfig(use_blosc2_arrays=False),
            serialization=SerializationConfig(),
            handlers=HandlerConfig(),
        )
        c = cacheness(config)
        yield c
        c.close()

    # ── __len__ ──────────────────────────────────────────────

    def test_len_empty_cache(self, cache):
        """len(cache) returns 0 when no entries exist."""
        assert len(cache) == 0

    def test_len_after_puts(self, cache):
        """len(cache) reflects the number of stored entries."""
        cache.put("a", on={"k": "1"})
        assert len(cache) == 1
        cache.put("b", on={"k": "2"})
        assert len(cache) == 2
        cache.put("c", on={"k": "3"})
        assert len(cache) == 3

    def test_len_after_delete(self, cache):
        """len(cache) decrements after invalidation."""
        cache.put("x", on={"k": "del1"})
        cache.put("y", on={"k": "del2"})
        assert len(cache) == 2

        cache.invalidate(on={"k": "del1"})
        assert len(cache) == 1

    # ── __contains__ ─────────────────────────────────────────

    def test_contains_existing_key(self, cache):
        """'key in cache' returns True for an existing entry."""
        cache.put(42, on={"k": "exists"})
        cache_key = cache._create_cache_key({"k": "exists"})
        assert cache_key in cache

    def test_contains_missing_key(self, cache):
        """'key in cache' returns False for a non-existent entry."""
        assert "no_such_key" not in cache

    def test_contains_expired_key(self, temp_cache_dir):
        """'key in cache' returns False for an expired entry."""
        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(temp_cache_dir / "ttl")),
            metadata=CacheMetadataConfig(
                metadata_backend="json", default_ttl_seconds=0.1
            ),
            compression=CompressionConfig(use_blosc2_arrays=False),
            serialization=SerializationConfig(),
            handlers=HandlerConfig(),
        )
        c = cacheness(config)
        c.put("val", on={"k": "ttl"})
        cache_key = c._create_cache_key({"k": "ttl"})

        # Should exist before expiration
        assert cache_key in c

        time.sleep(0.3)

        # Should be gone after expiration
        assert cache_key not in c

    # ── __iter__ ─────────────────────────────────────────────

    def test_iter_empty(self, cache):
        """Iterating an empty cache yields nothing."""
        entries = list(cache)
        assert entries == []

    def test_iter_yields_all_entries(self, cache):
        """for entry in cache yields all stored entries."""
        cache.put("a", on={"k": "i1"})
        cache.put("b", on={"k": "i2"})
        cache.put("c", on={"k": "i3"})

        entries = list(cache)
        assert len(entries) == 3

        # Each entry should be a dict with cache_key
        keys = {e["cache_key"] for e in entries}
        expected_keys = {
            cache._create_cache_key({"k": v}) for v in ("i1", "i2", "i3")
        }
        assert keys == expected_keys

    def test_iter_entry_has_expected_fields(self, cache):
        """Each yielded entry contains at least cache_key."""
        cache.put({"data": 1}, on={"k": "fields"})
        entry = next(iter(cache))
        assert "cache_key" in entry

    # ── combined ─────────────────────────────────────────────

    def test_len_contains_iter_consistency(self, cache):
        """len, in, and iter are consistent with each other."""
        cache.put("val1", on={"k": "c1"})
        cache.put("val2", on={"k": "c2"})

        # len matches iter count
        assert len(cache) == len(list(cache))

        # all iterated keys are in cache
        for entry in cache:
            assert entry["cache_key"] in cache
