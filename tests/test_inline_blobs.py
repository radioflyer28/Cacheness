"""Tests for Phase 2 inline blob storage in metadata backend.

Validates that small blobs can be stored directly in the metadata row
instead of as separate files, controlled by ``max_inline_size``.
"""

import tempfile
from pathlib import Path

import pytest

from cacheness.core import UnifiedCache
from cacheness.config import CacheBlobConfig, CacheConfig


# ── Config validation ─────────────────────────────────────────────


class TestInlineBlobConfig:
    """Tests for max_inline_size configuration."""

    def test_default_disabled(self):
        """max_inline_size defaults to 0 (disabled)."""
        cfg = CacheBlobConfig()
        assert cfg.max_inline_size == 0

    def test_set_positive(self):
        """max_inline_size accepts positive values."""
        cfg = CacheBlobConfig(max_inline_size=4096)
        assert cfg.max_inline_size == 4096

    def test_negative_raises(self):
        """Negative max_inline_size raises ValueError."""
        with pytest.raises(ValueError, match="max_inline_size"):
            CacheBlobConfig(max_inline_size=-1)

    def test_zero_is_valid(self):
        """Zero explicitly disables inlining."""
        cfg = CacheBlobConfig(max_inline_size=0)
        assert cfg.max_inline_size == 0

    def test_flat_kwarg_passthrough(self, tmp_path):
        """max_inline_size can be passed as flat kwarg to CacheConfig."""
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            max_inline_size=2048,
        )
        cache = UnifiedCache(config=config)
        assert cache.config.blob.max_inline_size == 2048


# ── Helpers ────────────────────────────────────────────────────────


def _make_cache(tmp_path, name="cache", **kwargs):
    """Create a UnifiedCache with given kwargs via CacheConfig."""
    config = CacheConfig(cache_dir=str(tmp_path / name), **kwargs)
    return UnifiedCache(config=config)


# ── Core put/get with inlining ────────────────────────────────────


class TestInlineBlobPutGet:
    """Test that small objects are inlined and large ones are not."""

    @pytest.fixture
    def inline_cache(self, tmp_path):
        """Cache with inlining enabled (4KB threshold)."""
        return _make_cache(tmp_path, "cache", max_inline_size=4096)

    @pytest.fixture
    def normal_cache(self, tmp_path):
        """Cache with inlining disabled (default)."""
        return _make_cache(tmp_path, "cache_normal")

    def test_small_object_inlined(self, inline_cache):
        """A small dict (well under 4KB) should be stored inline."""
        data = {"key": "value", "numbers": [1, 2, 3]}
        cache_key = inline_cache.put(data, cache_key="small-obj")

        # Verify the entry is marked as inline
        entry = inline_cache.metadata_backend.get_entry(cache_key)
        assert entry is not None
        assert entry.get("is_inline") == 1
        assert entry.get("blob_data") is not None
        assert isinstance(entry["blob_data"], bytes)

        # Verify the metadata has no actual_path (no blob file)
        metadata = entry.get("metadata", {})
        assert metadata.get("actual_path") is None

    def test_small_object_roundtrip(self, inline_cache):
        """Inlined data should be retrievable."""
        data = {"hello": "world", "nested": {"a": 1}}
        cache_key = inline_cache.put(data, cache_key="roundtrip")
        result = inline_cache.get(cache_key="roundtrip")
        assert result == data

    def test_small_string_roundtrip(self, inline_cache):
        """Small string should inline and roundtrip."""
        data = "hello, inline world!"
        cache_key = inline_cache.put(data, cache_key="str-inline")
        result = inline_cache.get(cache_key="str-inline")
        assert result == data

    def test_small_list_roundtrip(self, inline_cache):
        """Small list should inline and roundtrip."""
        data = [1, 2, 3, "four", 5.0]
        inline_cache.put(data, cache_key="list-inline")
        result = inline_cache.get(cache_key="list-inline")
        assert result == data

    def test_large_object_not_inlined(self, inline_cache):
        """An object larger than max_inline_size should remain as a file."""
        # Create incompressible data larger than 4KB (random bytes don't compress)
        import os

        data = os.urandom(8000)
        cache_key = inline_cache.put(data, cache_key="large-obj")

        entry = inline_cache.metadata_backend.get_entry(cache_key)
        assert entry is not None
        assert entry.get("is_inline", 0) == 0
        assert entry.get("blob_data") is None

        metadata = entry.get("metadata", {})
        assert metadata.get("actual_path") is not None

    def test_large_object_roundtrip(self, inline_cache):
        """Large objects (not inlined) should still roundtrip normally."""
        import os

        data = os.urandom(8000)
        inline_cache.put(data, cache_key="large-rt")
        result = inline_cache.get(cache_key="large-rt")
        assert result == data

    def test_disabled_by_default(self, normal_cache):
        """With max_inline_size=0, nothing should be inlined."""
        data = {"tiny": True}
        cache_key = normal_cache.put(data, cache_key="no-inline")

        entry = normal_cache.metadata_backend.get_entry(cache_key)
        assert entry is not None
        assert entry.get("is_inline", 0) == 0
        assert entry.get("blob_data") is None

    def test_no_blob_file_for_inlined(self, inline_cache, tmp_path):
        """Inlined entries should not leave a blob file on disk."""
        data = {"small": True}
        cache_key = inline_cache.put(data, cache_key="no-file-check")

        # Check that no blob files exist in the cache directory (only metadata DB)
        cache_dir = tmp_path / "cache" / "default"
        blob_files = [
            f
            for f in cache_dir.rglob("*")
            if f.is_file()
            and not f.name.endswith(
                (".db", ".json", ".db-journal", ".db-wal", ".db-shm")
            )
        ]
        assert len(blob_files) == 0, f"Found unexpected blob files: {blob_files}"


# ── get_with_metadata ─────────────────────────────────────────────


class TestInlineBlobGetWithMetadata:
    """Test get_with_metadata returns inline entries correctly."""

    @pytest.fixture
    def cache(self, tmp_path):
        return _make_cache(tmp_path, "cache", max_inline_size=4096)

    def test_get_with_metadata_inline(self, cache):
        """get_with_metadata should work for inlined entries."""
        data = {"meta_test": 42}
        cache.put(data, cache_key="meta-inline")

        result = cache.get_with_metadata(cache_key="meta-inline")
        assert result is not None

        retrieved_data, metadata = result
        assert retrieved_data == data
        assert metadata.get("is_inline") == 1
        assert metadata.get("data_type") is not None


# ── Storage mode with inlining ────────────────────────────────────


class TestInlineBlobStorageMode:
    """Test inline blobs with storage_mode=True."""

    @pytest.fixture
    def store(self, tmp_path):
        return _make_cache(tmp_path, "store", storage_mode=True, max_inline_size=4096)

    def test_storage_mode_inline_roundtrip(self, store):
        """Storage mode should support inline blobs."""
        data = {"storage": "inline"}
        store.put(data, hash_key="store-inline")

        result = store.get(hash_key="store-inline")
        assert result == data

    def test_storage_mode_get_with_metadata_inline(self, store):
        """Storage mode get_with_metadata with inline blob."""
        data = {"store_meta": True}
        store.put(data, hash_key="store-meta-inline")

        result = store.get_with_metadata(hash_key="store-meta-inline")
        assert result is not None
        retrieved_data, meta = result
        assert retrieved_data == data
        assert meta.get("is_inline") == 1


# ── update_data with inlining ────────────────────────────────────


class TestInlineBlobUpdateData:
    """Test update_data correctly handles inline blobs."""

    @pytest.fixture
    def cache(self, tmp_path):
        return _make_cache(tmp_path, "cache", max_inline_size=4096)

    def test_update_small_to_small(self, cache):
        """Updating an inlined entry with small data keeps it inlined."""
        cache.put({"v": 1}, cache_key="update-s2s")
        success = cache.update_data({"v": 2}, cache_key="update-s2s")
        assert success is True

        result = cache.get(cache_key="update-s2s")
        assert result == {"v": 2}

        entry = cache.metadata_backend.get_entry("update-s2s")
        assert entry.get("is_inline") == 1

    def test_update_small_to_large(self, cache):
        """Updating an inlined entry with large data should de-inline it."""
        import os

        cache.put({"v": 1}, cache_key="update-s2l")
        success = cache.update_data(os.urandom(8000), cache_key="update-s2l")
        assert success is True

        entry = cache.metadata_backend.get_entry("update-s2l")
        assert entry.get("is_inline", 0) == 0
        assert entry.get("blob_data") is None


# ── Integrity verification ────────────────────────────────────────


class TestInlineBlobIntegrity:
    """Test that integrity verification works with inline blobs."""

    @pytest.fixture
    def cache(self, tmp_path):
        return _make_cache(
            tmp_path, "cache", max_inline_size=4096, verify_cache_integrity=True
        )

    def test_integrity_roundtrip(self, cache):
        """Integrity-verified inline blob should roundtrip."""
        data = {"verified": True}
        cache.put(data, cache_key="integrity-inline")

        result = cache.get(cache_key="integrity-inline")
        assert result == data

    def test_integrity_hash_stored(self, cache):
        """Inlined entry should have file_hash in metadata."""
        data = {"hash_test": 123}
        cache.put(data, cache_key="hash-inline")

        entry = cache.metadata_backend.get_entry("hash-inline")
        metadata = entry.get("metadata", {})
        assert metadata.get("file_hash") is not None


# ── iter_entry_summaries ──────────────────────────────────────────


class TestInlineBlobEntrySummaries:
    """Test that iter_entry_summaries includes is_inline flag."""

    @pytest.fixture
    def cache(self, tmp_path):
        return _make_cache(tmp_path, "cache", max_inline_size=4096)

    def test_summary_has_is_inline(self, cache):
        """Entry summaries should include is_inline field."""
        import os

        cache.put({"small": True}, cache_key="summary-inline")
        cache.put(os.urandom(8000), cache_key="summary-file")

        summaries = cache.metadata_backend.iter_entry_summaries()
        by_key = {s["cache_key"]: s for s in summaries}

        assert by_key["summary-inline"]["is_inline"] == 1
        assert by_key["summary-file"]["is_inline"] == 0


# ── Overwrite scenarios ──────────────────────────────────────────


class TestInlineBlobOverwrite:
    """Test overwriting entries between inline and file-backed."""

    @pytest.fixture
    def cache(self, tmp_path):
        return _make_cache(tmp_path, "cache", max_inline_size=4096)

    def test_overwrite_inlined_with_inlined(self, cache):
        """Overwriting an inlined entry with another small value."""
        cache.put({"v": 1}, cache_key="overwrite-i2i")
        cache.put({"v": 2}, cache_key="overwrite-i2i")

        result = cache.get(cache_key="overwrite-i2i")
        assert result == {"v": 2}

        entry = cache.metadata_backend.get_entry("overwrite-i2i")
        assert entry.get("is_inline") == 1

    def test_overwrite_file_with_inlined(self, cache):
        """Overwriting a large (file) entry with a small (inline) value."""
        import os

        cache.put(os.urandom(8000), cache_key="overwrite-f2i")
        cache.put({"small": True}, cache_key="overwrite-f2i")

        result = cache.get(cache_key="overwrite-f2i")
        assert result == {"small": True}

        entry = cache.metadata_backend.get_entry("overwrite-f2i")
        assert entry.get("is_inline") == 1

    def test_overwrite_inlined_with_file(self, cache):
        """Overwriting a small (inline) entry with a large (file) value."""
        import os

        big_data = os.urandom(8000)
        cache.put({"small": True}, cache_key="overwrite-i2f")
        cache.put(big_data, cache_key="overwrite-i2f")

        result = cache.get(cache_key="overwrite-i2f")
        assert result == big_data

        entry = cache.metadata_backend.get_entry("overwrite-i2f")
        assert entry.get("is_inline", 0) == 0


# ── Delete/eviction safety ───────────────────────────────────────


class TestInlineBlobDelete:
    """Test that delete operations handle inline entries safely."""

    @pytest.fixture
    def cache(self, tmp_path):
        return _make_cache(tmp_path, "cache", max_inline_size=4096)

    def test_delete_inline_entry(self, cache):
        """Deleting an inline entry should not crash (no blob file to delete)."""
        cache.put({"del": True}, cache_key="delete-inline")
        assert cache.get(cache_key="delete-inline") is not None

        cache.invalidate(cache_key="delete-inline")
        assert cache.get(cache_key="delete-inline") is None

    def test_clear_with_inline_entries(self, cache):
        """Clearing cache with mixed inline/file entries should work."""
        import os

        cache.put({"small": True}, cache_key="clear-inline")
        cache.put(os.urandom(8000), cache_key="clear-file")

        cache.clear()

        assert cache.get(cache_key="clear-inline") is None
        assert cache.get(cache_key="clear-file") is None
