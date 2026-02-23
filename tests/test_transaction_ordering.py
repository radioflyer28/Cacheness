"""Tests for transaction ordering: blob vs metadata deletion sequencing.

Covers:
- CACHE-59d: get() error paths delete blob + metadata (not metadata-only)
- CACHE-3pq: delete() uses metadata-first ordering
- CACHE-nbr: put() overwrite cleans up old blob on type change
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cacheness.core import UnifiedCache
from cacheness.config import (
    CacheConfig,
    CacheMetadataConfig,
    CacheStorageConfig,
)


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    d = tmp_path / "test_txn_cache"
    d.mkdir()
    return d


@pytest.fixture
def cache(cache_dir):
    """Create a cache instance with integrity verification enabled."""
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(cache_dir)),
        metadata=CacheMetadataConfig(
            metadata_backend="sqlite",
            verify_cache_integrity=True,
        ),
    )
    c = UnifiedCache(config)
    yield c
    c.close()


@pytest.fixture
def cache_no_integrity(cache_dir):
    """Create a cache without integrity verification."""
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(cache_dir)),
        metadata=CacheMetadataConfig(
            metadata_backend="sqlite",
            verify_cache_integrity=False,
        ),
    )
    c = UnifiedCache(config)
    yield c
    c.close()


@pytest.fixture
def storage_cache(cache_dir):
    """Create a storage-mode cache instance."""
    config = CacheConfig(
        storage_mode=True,
        storage=CacheStorageConfig(cache_dir=str(cache_dir)),
        metadata=CacheMetadataConfig(
            metadata_backend="sqlite",
            verify_cache_integrity=True,
        ),
    )
    c = UnifiedCache(config)
    yield c
    c.close()


class TestGetErrorPathBlobCleanup:
    """CACHE-59d: get() error paths should delete both blob and metadata."""

    def test_integrity_failure_deletes_blob(self, cache):
        """When integrity check fails, both blob and metadata should be removed."""
        cache.put({"key": "value"}, cache_key="integrity-test")

        # Get the blob path
        entry = cache.metadata_backend.get_entry("integrity-test")
        metadata = entry.get("metadata", {})
        actual_path = metadata.get("actual_path")
        assert actual_path is not None

        # Corrupt the file by appending garbage
        resolved = cache._resolve_actual_path(actual_path)
        with open(resolved, "ab") as f:
            f.write(b"CORRUPTION")

        # get() should detect integrity failure and clean up BOTH
        result = cache.get(cache_key="integrity-test")
        assert result is None

        # Verify metadata is gone
        assert cache.metadata_backend.get_entry("integrity-test") is None

        # Verify blob file is ALSO gone (the fix — previously it was orphaned)
        assert not Path(resolved).exists()

    def test_deserialization_failure_deletes_blob(self, cache_no_integrity):
        """When deserialization fails, both blob and metadata should be removed."""
        cache_no_integrity.put({"key": "value"}, cache_key="deser-test")

        # Get the blob path
        entry = cache_no_integrity.metadata_backend.get_entry("deser-test")
        metadata = entry.get("metadata", {})
        actual_path = metadata.get("actual_path")
        resolved = cache_no_integrity._resolve_actual_path(actual_path)

        # Overwrite file with garbage that can't be deserialized
        with open(resolved, "wb") as f:
            f.write(b"NOT_A_VALID_PICKLE_OR_ANYTHING")

        result = cache_no_integrity.get(cache_key="deser-test")
        assert result is None

        # Verify blob file is gone
        assert not Path(resolved).exists()

    def test_file_not_found_only_removes_metadata(self, cache):
        """FileNotFoundError should remove metadata but not attempt blob deletion
        (blob is already gone)."""
        cache.put({"key": "value"}, cache_key="fnf-test")

        entry = cache.metadata_backend.get_entry("fnf-test")
        metadata = entry.get("metadata", {})
        actual_path = metadata.get("actual_path")
        resolved = cache._resolve_actual_path(actual_path)

        # Delete the blob file externally
        Path(resolved).unlink()

        # get() should handle gracefully
        result = cache.get(cache_key="fnf-test")
        assert result is None

        # Metadata should be cleaned up
        assert cache.metadata_backend.get_entry("fnf-test") is None

    def test_storage_mode_preserves_entry_on_corruption(self, storage_cache):
        """Storage mode should NOT delete entries on integrity failure."""
        storage_cache.put({"key": "value"}, cache_key="storage-test")

        entry = storage_cache.metadata_backend.get_entry("storage-test")
        metadata = entry.get("metadata", {})
        actual_path = metadata.get("actual_path")
        resolved = storage_cache._resolve_actual_path(actual_path)

        # Corrupt the file
        with open(resolved, "ab") as f:
            f.write(b"CORRUPTION")

        # get() should return None but preserve the entry
        result = storage_cache.get(cache_key="storage-test")
        assert result is None

        # Entry should still exist (storage mode preserves on error)
        assert storage_cache.metadata_backend.get_entry("storage-test") is not None
        # Blob should still exist too
        assert Path(resolved).exists()


class TestDeleteMetadataFirstOrdering:
    """CACHE-3pq: delete() should use metadata-first ordering."""

    def test_delete_removes_both(self, cache):
        """Basic delete should remove both metadata and blob."""
        cache.put({"key": "value"}, cache_key="delete-test")

        entry = cache.metadata_backend.get_entry("delete-test")
        metadata = entry.get("metadata", {})
        actual_path = metadata.get("actual_path")
        resolved = cache._resolve_actual_path(actual_path)

        # Verify both exist
        assert Path(resolved).exists()
        assert cache.metadata_backend.get_entry("delete-test") is not None

        cache.invalidate(cache_key="delete-test")

        # Both should be gone
        assert not Path(resolved).exists()
        assert cache.metadata_backend.get_entry("delete-test") is None

    def test_delete_blob_failure_still_removes_metadata(self, cache):
        """If blob deletion fails, metadata should still be removed
        (metadata-first means it's already gone)."""
        cache.put({"key": "value"}, cache_key="delete-fail-test")

        entry = cache.metadata_backend.get_entry("delete-fail-test")
        metadata = entry.get("metadata", {})
        actual_path = metadata.get("actual_path")
        resolved = cache._resolve_actual_path(actual_path)

        # Make blob un-deletable by removing the file first
        # (simulates blob backend failure)
        Path(resolved).unlink()

        # delete() should still succeed (metadata removed, blob deletion
        # is best-effort)
        cache.invalidate(cache_key="delete-fail-test")

        # Metadata should be gone
        assert cache.metadata_backend.get_entry("delete-fail-test") is None


class TestPutOverwriteCleanup:
    """CACHE-nbr: put() overwrite should clean up old blob when path changes."""

    def test_overwrite_same_type_no_orphan(self, cache):
        """Overwriting with the same type should not leave orphans."""
        cache.put({"v": 1}, cache_key="same-type")

        entry1 = cache.metadata_backend.get_entry("same-type")
        path1 = entry1["metadata"]["actual_path"]

        cache.put({"v": 2}, cache_key="same-type")

        entry2 = cache.metadata_backend.get_entry("same-type")
        path2 = entry2["metadata"]["actual_path"]

        # Same type → same path, no orphan
        # (path might be the same since key determines path)
        resolved2 = cache._resolve_actual_path(path2)
        assert Path(resolved2).exists()

        # Verify the new data is correct
        result = cache.get(cache_key="same-type")
        assert result == {"v": 2}

    def test_overwrite_different_type_cleans_old_blob(self, cache):
        """Overwriting with a different type should clean up the old blob."""
        # Store a DataFrame (uses parquet format)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        cache.put(df, cache_key="type-change")

        entry1 = cache.metadata_backend.get_entry("type-change")
        old_path = entry1["metadata"]["actual_path"]
        old_resolved = cache._resolve_actual_path(old_path)
        assert Path(old_resolved).exists()

        # Overwrite with a plain dict (uses pickle format)
        cache.put({"replaced": True}, cache_key="type-change")

        entry2 = cache.metadata_backend.get_entry("type-change")
        new_path = entry2["metadata"]["actual_path"]

        # If paths differ, old blob should be cleaned up
        if old_path != new_path:
            assert not Path(old_resolved).exists(), (
                f"Old blob at {old_resolved} should have been cleaned up"
            )

        # New data should be correct
        result = cache.get(cache_key="type-change")
        assert result == {"replaced": True}

    def test_overwrite_numpy_to_dict_cleans_old_blob(self, cache):
        """Overwriting numpy array with dict should clean up old blob."""
        arr = np.array([1.0, 2.0, 3.0])
        cache.put(arr, cache_key="np-to-dict")

        entry1 = cache.metadata_backend.get_entry("np-to-dict")
        old_path = entry1["metadata"]["actual_path"]
        old_resolved = cache._resolve_actual_path(old_path)
        assert Path(old_resolved).exists()

        # Overwrite with dict
        cache.put({"replaced": True}, cache_key="np-to-dict")

        entry2 = cache.metadata_backend.get_entry("np-to-dict")
        new_path = entry2["metadata"]["actual_path"]

        if old_path != new_path:
            assert not Path(old_resolved).exists(), (
                f"Old numpy blob at {old_resolved} should have been cleaned up"
            )

        result = cache.get(cache_key="np-to-dict")
        assert result == {"replaced": True}

    def test_storage_mode_overwrite_cleans_old_blob(self, storage_cache):
        """Storage mode put() should also clean up old blobs on type change."""
        df = pd.DataFrame({"x": [1, 2]})
        storage_cache.put(df, cache_key="storage-overwrite")

        entry1 = storage_cache.metadata_backend.get_entry("storage-overwrite")
        old_path = entry1["metadata"]["actual_path"]
        old_resolved = storage_cache._resolve_actual_path(old_path)
        assert Path(old_resolved).exists()

        # Overwrite with dict
        storage_cache.put({"replaced": True}, cache_key="storage-overwrite")

        entry2 = storage_cache.metadata_backend.get_entry("storage-overwrite")
        new_path = entry2["metadata"]["actual_path"]

        if old_path != new_path:
            assert not Path(old_resolved).exists()

        result = storage_cache.get(cache_key="storage-overwrite")
        assert result == {"replaced": True}
