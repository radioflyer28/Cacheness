"""
Test Management Operations: update_data() and update_entry_metadata()
Phase 3: Sprint 1
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from cacheness import cacheness
from cacheness.config import CacheConfig
from cacheness.metadata import JsonBackend, SqliteBackend


@pytest.fixture
def cache_dir(tmp_path):
    """Temporary cache directory."""
    return tmp_path / "cache"


@pytest.fixture
def memory_cache(cache_dir):
    """UnifiedCache with in-memory SQLite backend (fast, ephemeral)."""
    config = CacheConfig(
        cache_dir=str(cache_dir),
        metadata_backend="sqlite_memory",
        cleanup_on_init=False
    )
    # Disable signing for testing to avoid signature issues
    config.security.enable_signing = False
    cache = cacheness(config)
    yield cache
    cache.clear_all()


@pytest.fixture
def json_cache(cache_dir):
    """UnifiedCache with JsonBackend."""
    config = CacheConfig(
        cache_dir=str(cache_dir),
        metadata_backend="json",
        cleanup_on_init=False
    )
    # Disable signing for testing
    config.security.enable_signing = False
    cache = cacheness(config)
    yield cache
    cache.clear_all()


@pytest.fixture
def sqlite_cache(cache_dir):
    """UnifiedCache with SqliteBackend."""
    config = CacheConfig(
        cache_dir=str(cache_dir),
        metadata_backend="sqlite",
        cleanup_on_init=False
    )
    # Disable signing for testing
    config.security.enable_signing = False
    cache = cacheness(config)
    yield cache
    cache.clear_all()


class TestUpdateData:
    """Test update_data() method in UnifiedCache."""
    
    def test_update_existing_entry(self, memory_cache):
        """Test updating data at existing cache key."""
        # Create initial entry
        original_data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        memory_cache.put(original_data, experiment="exp_001", run_id=1)
        
        # Update with new data
        new_data = pd.DataFrame({"a": [10, 20], "b": [30, 40]})
        success = memory_cache.update_data(new_data, experiment="exp_001", run_id=1)
        
        assert success is True
        
        # Verify updated data
        result = memory_cache.get(experiment="exp_001", run_id=1)
        pd.testing.assert_frame_equal(result, new_data)
    
    def test_update_nonexistent_entry(self, memory_cache):
        """Test updating entry that doesn't exist."""
        new_data = pd.DataFrame({"a": [1, 2]})
        success = memory_cache.update_data(new_data, experiment="nonexistent")
        
        assert success is False
    
    # Test removed - requires internal cache_key generation details
    # Direct cache_key updates are tested in backend tests
    
    def test_update_different_data_types(self, memory_cache):
        """Test updating with different data types."""
        # Original: DataFrame
        df = pd.DataFrame({"x": [1, 2, 3]})
        memory_cache.put(df, test="type_change")
        
        # Update to: dict (different type)
        new_data = {"message": "now a dict"}
        success = memory_cache.update_data(new_data, test="type_change")
        
        assert success is True
        result = memory_cache.get(test="type_change")
        assert result == new_data
    
    def test_update_updates_metadata(self, memory_cache):
        """Test that update_data() updates derived metadata."""
        # Create small entry
        small_data = pd.DataFrame({"a": [1]})
        memory_cache.put(small_data, test="metadata_update")
        
        # Get initial metadata
        meta_before = memory_cache.get_metadata(test="metadata_update")
        size_before = meta_before.get("file_size", 0)
        created_before = meta_before.get("created_at")
        
        # Update with larger data
        import time
        time.sleep(0.1)  # Ensure timestamp changes
        large_data = pd.DataFrame({"a": range(1000), "b": range(1000)})
        success = memory_cache.update_data(large_data, test="metadata_update")
        
        assert success is True
        
        # Get updated metadata
        meta_after = memory_cache.get_metadata(test="metadata_update")
        size_after = meta_after.get("file_size", 0)
        created_after = meta_after.get("created_at")
        
        # Verify metadata changed
        assert size_after > size_before
        assert created_after != created_before  # Timestamp should update


class TestUpdateEntryMetadataBackends:
    """Test update_entry_metadata() implementation in different backends."""
    
    def test_memory_backend_update(self, cache_dir):
        """Test SqliteBackend (in-memory) update_entry_metadata()."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        backend = SqliteBackend(db_file=":memory:")
        
        # Create entry
        backend.put_entry("test_key", {
            "description": "test",
            "data_type": "dict",
            "prefix": "",
            "file_size": 100,
            "metadata": {"content_hash": "abc123"}
        })
        
        # Update metadata (no handler/config — blob I/O is core.py's job)
        success = backend.update_entry_metadata(
            cache_key="test_key",
            updates={
                "file_size": 200,
                "content_hash": "xyz789",
                "actual_path": str(cache_dir / "test_key"),
                "storage_format": "pickle",
                "data_type": "dict",
                "serializer": "pickle",
            }
        )
        
        assert success is True
        
        # Verify entry updated
        entry = backend.get_entry("test_key")
        assert entry["file_size"] == 200
        # SqliteBackend maps content_hash → file_hash column
        assert entry["metadata"]["file_hash"] == "xyz789"
        
        backend.close()
    
    def test_json_backend_update(self, cache_dir):
        """Test JsonBackend.update_entry_metadata()."""
        metadata_file = cache_dir / "metadata.json"
        cache_dir.mkdir(parents=True)
        
        backend = JsonBackend(metadata_file)
        
        # Create entry
        backend.put_entry("test_key", {
            "description": "test",
            "data_type": "dict",
            "prefix": "",
            "file_size": 100,
            "metadata": {}
        })
        
        # Update metadata only
        success = backend.update_entry_metadata(
            cache_key="test_key",
            updates={
                "file_size": 300,
                "content_hash": "new_hash",
                "storage_format": "pickle",
                "data_type": "dict",
                "serializer": "pickle",
            }
        )
        
        assert success is True
        
        # Verify persisted to disk
        entry = backend.get_entry("test_key")
        assert entry["file_size"] == 300
    
    def test_sqlite_backend_update(self, cache_dir):
        """Test SqliteBackend.update_entry_metadata()."""
        db_file = cache_dir / "cache.db"
        cache_dir.mkdir(parents=True)
        
        backend = SqliteBackend(db_file=str(db_file))
        
        # Create entry
        backend.put_entry("test_key", {
            "description": "test",
            "data_type": "dict",
            "prefix": "",
            "file_size": 100,
            "metadata": {}
        })
        
        # Update metadata only
        success = backend.update_entry_metadata(
            cache_key="test_key",
            updates={
                "file_size": 400,
                "content_hash": "sqlite_hash",
                "storage_format": "pickle",
                "data_type": "dict",
                "serializer": "pickle",
            }
        )
        
        assert success is True
        
        # Verify updated in database
        entry = backend.get_entry("test_key")
        assert entry["file_size"] == 400
        
        backend.close()
    
    def test_update_nonexistent_key_returns_false(self, cache_dir):
        """Test updating nonexistent key returns False."""
        backend = SqliteBackend(db_file=":memory:")
        
        success = backend.update_entry_metadata(
            cache_key="nonexistent",
            updates={"file_size": 100}
        )
        
        assert success is False
        
        backend.close()


class TestCacheKeyImmutability:
    """Test that cache_key remains immutable during updates."""
    
    def test_cache_key_unchanged_after_update(self, memory_cache):
        """Verify cache_key doesn't change when updating data."""
        # Create entry
        original = {"value": 1}
        memory_cache.put(original, test="immutable")
        
        # Get metadata before updates
        meta_before = memory_cache.get_metadata(test="immutable")
        cache_key_before = meta_before["cache_key"]
        
        # Update data multiple times
        memory_cache.update_data({"value": 2}, test="immutable")
        memory_cache.update_data({"value": 3}, test="immutable")
        
        # Verify cache_key unchanged
        meta_after = memory_cache.get_metadata(test="immutable")
        cache_key_after = meta_after["cache_key"]
        assert cache_key_after == cache_key_before
        
        # Verify we can still retrieve with same params
        result = memory_cache.get(test="immutable")
        assert result["value"] == 3


class TestIntegrationAllBackends:
    """Integration tests across all backends."""
    
    @pytest.mark.parametrize("cache_fixture", ["memory_cache", "json_cache", "sqlite_cache"])
    def test_update_workflow(self, cache_fixture, request):
        """Test complete update workflow on all backends."""
        cache = request.getfixturevalue(cache_fixture)
        
        # Initial put
        data_v1 = np.array([1, 2, 3])
        cache.put(data_v1, version="v1")
        
        # Verify initial
        result = cache.get(version="v1")
        np.testing.assert_array_equal(result, data_v1)
        
        # Update
        data_v2 = np.array([10, 20, 30, 40])
        success = cache.update_data(data_v2, version="v1")
        assert success is True
        
        # Verify updated
        result = cache.get(version="v1")
        np.testing.assert_array_equal(result, data_v2)
        
        # Verify metadata updated
        meta = cache.get_metadata(version="v1")
        assert meta is not None
        assert "file_size" in meta


class TestTouchOperation:
    """Test touch() operation for TTL extension."""
    
    def test_touch_existing_entry(self, memory_cache):
        """Test touching existing entry extends TTL."""
        # Create entry
        data = {"value": 100}
        memory_cache.put(data, test="touch_test")
        
        # Get initial timestamp
        meta_before = memory_cache.get_metadata(test="touch_test")
        created_before = meta_before.get("created_at")
        
        # Wait a moment
        import time
        time.sleep(0.1)
        
        # Touch the entry
        success = memory_cache.touch(test="touch_test")
        assert success is True
        
        # Verify timestamp updated
        meta_after = memory_cache.get_metadata(test="touch_test")
        created_after = meta_after.get("created_at")
        
        assert created_after != created_before
        assert created_after > created_before
    
    def test_touch_nonexistent_entry(self, memory_cache):
        """Test touching nonexistent entry returns False."""
        success = memory_cache.touch(test="nonexistent")
        assert success is False
    
    def test_touch_with_cache_key(self, memory_cache):
        """Test touch using direct cache_key."""
        # Create entry
        data = {"value": 200}
        cache_key = memory_cache.put(data, key="direct_touch")
        
        # Touch using cache_key
        success = memory_cache.touch(cache_key=cache_key)
        assert success is True
        
        # Verify data still accessible
        result = memory_cache.get(key="direct_touch")
        assert result["value"] == 200
    
    def test_touch_preserves_data(self, memory_cache):
        """Test that touch doesn't modify data."""
        # Create entry with specific data
        original_data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        memory_cache.put(original_data, test="preserve")
        
        # Touch the entry
        memory_cache.touch(test="preserve")
        
        # Verify data unchanged
        result = memory_cache.get(test="preserve")
        pd.testing.assert_frame_equal(result, original_data)
    
    @pytest.mark.parametrize("cache_fixture", ["memory_cache", "json_cache", "sqlite_cache"])
    def test_touch_all_backends(self, cache_fixture, request):
        """Test touch works on all backends."""
        cache = request.getfixturevalue(cache_fixture)
        
        # Create entry
        cache.put(np.array([1, 2, 3]), version="touch_test")
        
        # Touch it
        success = cache.touch(version="touch_test")
        assert success is True
        
        # Verify still accessible
        result = cache.get(version="touch_test")
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))


# ── Bulk & Batch Operation Tests ──────────────────────────────────────

class TestDeleteWhere:
    """Test delete_where() with filter functions."""

    def test_delete_where_by_data_type(self, memory_cache):
        """Delete entries matching a data_type filter."""
        # Seed: 2 DataFrames + 1 dict
        memory_cache.put(pd.DataFrame({"a": [1]}), kind="df1")
        memory_cache.put(pd.DataFrame({"b": [2]}), kind="df2")
        memory_cache.put({"c": 3}, kind="dict1")

        deleted = memory_cache.delete_where(
            lambda e: e.get("data_type") == "pandas_dataframe"
        )
        assert deleted == 2

        # dict entry should survive
        assert memory_cache.get(kind="dict1") == {"c": 3}
        assert memory_cache.get(kind="df1") is None
        assert memory_cache.get(kind="df2") is None

    def test_delete_where_no_match(self, memory_cache):
        """Filter that matches nothing returns 0."""
        memory_cache.put({"x": 1}, tag="keep")
        deleted = memory_cache.delete_where(lambda e: False)
        assert deleted == 0
        assert memory_cache.get(tag="keep") == {"x": 1}

    def test_delete_where_all_match(self, memory_cache):
        """Filter that matches everything clears the cache."""
        memory_cache.put({"a": 1}, slot="s1")
        memory_cache.put({"b": 2}, slot="s2")
        deleted = memory_cache.delete_where(lambda _: True)
        assert deleted == 2

    def test_delete_where_filter_error_skips(self, memory_cache):
        """If filter_fn raises, that entry is skipped (not deleted)."""
        memory_cache.put({"val": 1}, item="safe")

        def bad_filter(entry):
            raise ValueError("boom")

        deleted = memory_cache.delete_where(bad_filter)
        assert deleted == 0
        # Entry still alive
        assert memory_cache.get(item="safe") == {"val": 1}

    @pytest.mark.parametrize("cache_fixture", ["memory_cache", "json_cache", "sqlite_cache"])
    def test_delete_where_all_backends(self, cache_fixture, request):
        """delete_where works on every backend."""
        cache = request.getfixturevalue(cache_fixture)
        cache.put(np.array([1, 2]), exp="a")
        cache.put(np.array([3, 4]), exp="b")
        cache.put(np.array([5, 6]), exp="c")

        deleted = cache.delete_where(
            lambda e: e.get("data_type") == "array"
        )
        assert deleted == 3


class TestDeleteMatching:
    """Test delete_matching() with keyword filters."""

    def test_delete_matching_by_data_type(self, memory_cache):
        """Delete all entries of a certain data_type."""
        memory_cache.put(pd.DataFrame({"a": [1]}), grp="x")
        memory_cache.put({"b": 2}, grp="y")

        deleted = memory_cache.delete_matching(data_type="pandas_dataframe")
        assert deleted == 1
        assert memory_cache.get(grp="y") == {"b": 2}

    def test_delete_matching_no_kwargs(self, memory_cache):
        """Calling with no kwargs returns 0 (safety guard)."""
        memory_cache.put({"a": 1}, z="1")
        assert memory_cache.delete_matching() == 0

    @pytest.mark.parametrize("cache_fixture", ["memory_cache", "json_cache", "sqlite_cache"])
    def test_delete_matching_all_backends(self, cache_fixture, request):
        """delete_matching works on every backend."""
        cache = request.getfixturevalue(cache_fixture)
        cache.put(np.array([1]), run="r1")
        cache.put(np.array([2]), run="r2")
        cache.put({"v": 3}, run="r3")

        deleted = cache.delete_matching(data_type="array")
        assert deleted == 2
        assert cache.get(run="r3") == {"v": 3}


class TestGetBatch:
    """Test get_batch() for retrieving multiple entries."""

    def test_get_batch_mixed(self, memory_cache):
        """Retrieve a mix of existing and missing entries."""
        memory_cache.put({"val": 1}, key="k1")
        memory_cache.put({"val": 2}, key="k2")

        results = memory_cache.get_batch([
            {"key": "k1"},
            {"key": "k2"},
            {"key": "missing"},
        ])

        assert len(results) == 3
        vals = list(results.values())
        assert {"val": 1} in vals
        assert {"val": 2} in vals
        assert None in vals

    def test_get_batch_empty_list(self, memory_cache):
        """Empty input gives empty result."""
        assert memory_cache.get_batch([]) == {}

    @pytest.mark.parametrize("cache_fixture", ["memory_cache", "json_cache", "sqlite_cache"])
    def test_get_batch_all_backends(self, cache_fixture, request):
        """get_batch works on every backend."""
        cache = request.getfixturevalue(cache_fixture)
        cache.put(np.array([10]), idx="i1")
        cache.put(np.array([20]), idx="i2")

        results = cache.get_batch([{"idx": "i1"}, {"idx": "i2"}])
        assert len(results) == 2
        for v in results.values():
            assert v is not None


class TestDeleteBatch:
    """Test delete_batch() for removing multiple entries."""

    def test_delete_batch_some_exist(self, memory_cache):
        """Delete a mix of existing and missing entries."""
        memory_cache.put({"a": 1}, d="d1")
        memory_cache.put({"b": 2}, d="d2")

        deleted = memory_cache.delete_batch([
            {"d": "d1"},
            {"d": "d2"},
            {"d": "d_missing"},
        ])
        assert deleted == 2
        assert memory_cache.get(d="d1") is None
        assert memory_cache.get(d="d2") is None

    def test_delete_batch_empty_list(self, memory_cache):
        """Empty input deletes nothing."""
        assert memory_cache.delete_batch([]) == 0

    @pytest.mark.parametrize("cache_fixture", ["memory_cache", "json_cache", "sqlite_cache"])
    def test_delete_batch_all_backends(self, cache_fixture, request):
        """delete_batch works on every backend."""
        cache = request.getfixturevalue(cache_fixture)
        cache.put(np.array([1]), b="b1")
        cache.put(np.array([2]), b="b2")
        cache.put(np.array([3]), b="b3")

        deleted = cache.delete_batch([{"b": "b1"}, {"b": "b3"}])
        assert deleted == 2
        assert cache.get(b="b2") is not None


class TestTouchBatch:
    """Test touch_batch() for refreshing TTL on multiple entries."""

    def test_touch_batch_matching(self, memory_cache):
        """Touch all entries matching a data_type filter."""
        memory_cache.put(np.array([1]), tb="t1")
        memory_cache.put(np.array([2]), tb="t2")
        memory_cache.put({"d": 3}, tb="t3")

        touched = memory_cache.touch_batch(data_type="array")
        assert touched == 2

    def test_touch_batch_no_kwargs(self, memory_cache):
        """Calling with no kwargs returns 0 (safety guard)."""
        memory_cache.put({"a": 1}, n="n1")
        assert memory_cache.touch_batch() == 0

    def test_touch_batch_preserves_data(self, memory_cache):
        """Touched entries still return their data."""
        memory_cache.put(np.array([10, 20]), p="p1")
        memory_cache.put(np.array([30, 40]), p="p2")

        memory_cache.touch_batch(data_type="array")

        np.testing.assert_array_equal(
            memory_cache.get(p="p1"), np.array([10, 20])
        )
        np.testing.assert_array_equal(
            memory_cache.get(p="p2"), np.array([30, 40])
        )

    @pytest.mark.parametrize("cache_fixture", ["memory_cache", "json_cache", "sqlite_cache"])
    def test_touch_batch_all_backends(self, cache_fixture, request):
        """touch_batch works on every backend."""
        cache = request.getfixturevalue(cache_fixture)
        cache.put(np.array([1]), tb2="x1")
        cache.put(np.array([2]), tb2="x2")

        touched = cache.touch_batch(data_type="array")
        assert touched == 2


# ── Blob Cleanup Tests ───────────────────────────────────────────────

class TestBlobCleanupOnDelete:
    """Verify that invalidate/delete operations remove blob files from disk."""

    # Extensions that are actual cache blob files
    _BLOB_EXTENSIONS = {".pkl", ".npz", ".b2nd", ".b2tr", ".parquet",
                        ".lz4", ".zstd", ".gzip", ".zst", ".gz", ".bz2", ".xz"}

    def _get_blob_files(self, cache_dir: Path) -> set:
        """Return the set of cache blob files currently on disk."""
        if not cache_dir.exists():
            return set()
        blobs = set()
        for f in cache_dir.iterdir():
            if not f.is_file():
                continue
            # Check primary extension and compound extensions (e.g. .pkl.lz4)
            suffixes = f.suffixes  # e.g. ['.pkl', '.lz4']
            if any(s in self._BLOB_EXTENSIONS for s in suffixes):
                blobs.add(f)
        return blobs

    def test_invalidate_deletes_blob(self, memory_cache, cache_dir):
        """invalidate() should delete the blob file, not just metadata."""
        memory_cache.put(np.array([1, 2, 3]), item="blob_test")
        blobs_before = self._get_blob_files(cache_dir)
        assert len(blobs_before) > 0, "Expected blob file after put()"

        memory_cache.invalidate(item="blob_test")
        blobs_after = self._get_blob_files(cache_dir)
        assert len(blobs_after) == 0, f"Blob files should be deleted: {blobs_after}"

    def test_invalidate_by_cache_key_deletes_blob(self, memory_cache, cache_dir):
        """invalidate(cache_key=...) should also delete the blob."""
        memory_cache.put({"data": "test"}, item="key_test")
        blobs_before = self._get_blob_files(cache_dir)
        assert len(blobs_before) > 0

        # Get the cache key and invalidate directly
        entries = memory_cache.list_entries()
        cache_key = entries[0]["cache_key"]
        memory_cache.invalidate(cache_key=cache_key)

        blobs_after = self._get_blob_files(cache_dir)
        assert len(blobs_after) == 0

    def test_delete_where_deletes_blobs(self, memory_cache, cache_dir):
        """delete_where() should delete blob files for all matched entries."""
        memory_cache.put(np.array([1, 2]), kind="dw1")
        memory_cache.put(np.array([3, 4]), kind="dw2")
        memory_cache.put({"keep": True}, kind="dw3")

        blobs_before = self._get_blob_files(cache_dir)
        assert len(blobs_before) == 3

        deleted = memory_cache.delete_where(
            lambda e: e.get("data_type") == "array"
        )
        assert deleted == 2

        blobs_after = self._get_blob_files(cache_dir)
        assert len(blobs_after) == 1, f"Expected 1 remaining blob, got {blobs_after}"

    def test_delete_matching_deletes_blobs(self, memory_cache, cache_dir):
        """delete_matching() should delete blob files for matched entries."""
        memory_cache.put(np.array([10]), dm="a")
        memory_cache.put({"v": 2}, dm="b")

        deleted = memory_cache.delete_matching(data_type="array")
        assert deleted == 1

        blobs_after = self._get_blob_files(cache_dir)
        assert len(blobs_after) == 1

    def test_delete_batch_deletes_blobs(self, memory_cache, cache_dir):
        """delete_batch() should delete blob files for each entry."""
        memory_cache.put(np.array([1]), batch_del="b1")
        memory_cache.put(np.array([2]), batch_del="b2")

        blobs_before = self._get_blob_files(cache_dir)
        assert len(blobs_before) == 2

        deleted = memory_cache.delete_batch([
            {"batch_del": "b1"},
            {"batch_del": "b2"},
        ])
        assert deleted == 2
        assert len(self._get_blob_files(cache_dir)) == 0

    def test_verify_integrity_clean_after_invalidate(self, memory_cache, cache_dir):
        """verify_integrity() should report 0 issues after invalidate()."""
        memory_cache.put(np.array([1, 2, 3]), item="vi1")
        memory_cache.put(np.array([4, 5, 6]), item="vi2")

        memory_cache.invalidate(item="vi1")
        memory_cache.invalidate(item="vi2")

        report = memory_cache.verify_integrity()
        assert len(report["orphaned_blobs"]) == 0, f"Orphaned blobs found: {report}"
        assert len(report["dangling_entries"]) == 0

    @pytest.mark.parametrize("cache_fixture", ["memory_cache", "json_cache", "sqlite_cache"])
    def test_invalidate_blob_cleanup_all_backends(self, cache_fixture, cache_dir, request):
        """Blob cleanup on invalidate works across all backends."""
        cache = request.getfixturevalue(cache_fixture)
        cache.put(np.array([1, 2, 3]), blob_backend_test="val")

        blobs_before = self._get_blob_files(cache_dir)
        assert len(blobs_before) > 0

        cache.invalidate(blob_backend_test="val")

        blobs_after = self._get_blob_files(cache_dir)
        assert len(blobs_after) == 0

    def test_invalidate_missing_blob_no_error(self, memory_cache, cache_dir):
        """invalidate() handles missing blob file gracefully."""
        memory_cache.put(np.array([1, 2, 3]), item="ghost")

        # Manually delete the blob to simulate corruption
        blobs = self._get_blob_files(cache_dir)
        for b in blobs:
            b.unlink()

        # Should not raise, just remove metadata
        memory_cache.invalidate(item="ghost")
        assert memory_cache.get(item="ghost") is None
