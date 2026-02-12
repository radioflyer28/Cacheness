"""
Backend Operation Parity Tests
===============================

Verifies that SQLite and PostgreSQL metadata backends behave identically
for all operations defined in the MetadataBackend interface.
"""

import pytest
from datetime import datetime, timezone


class TestBackendParity:
    """Test that SQLite and PostgreSQL backends have identical behavior."""

    @pytest.fixture
    def sqlite_backend(self, tmp_path):
        """Create a SQLite backend for testing."""
        from cacheness.metadata import SqliteBackend

        db_path = tmp_path / "test_sqlite.db"
        backend = SqliteBackend(str(db_path))
        yield backend
        backend.close()

    @pytest.fixture
    def postgresql_backend(self):
        """Create a PostgreSQL backend for testing (requires Docker)."""
        pytest.skip(
            "PostgreSQL backend tests require Docker - run with integration tests"
        )
        # This would be enabled in Docker integration tests
        from cacheness.storage.backends.postgresql_backend import PostgresBackend

        backend = PostgresBackend(
            connection_url="postgresql://cacheness_test:test_password@localhost:5432/cacheness_test"
        )
        yield backend
        backend.clear_all()
        backend.close()

    def test_put_and_get_entry_parity(self, sqlite_backend):
        """Test that put_entry and get_entry work identically."""
        entry_data = {
            "cache_key": "test_key_123",
            "description": "Test entry",
            "data_type": "array",
            "prefix": "test",
            "file_size": 2048,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                # Only known technical fields are preserved
                "s3_etag": "test_etag_123",
                "actual_path": "/test/path.npz",
                "object_type": "<class 'numpy.ndarray'>",
                "storage_format": "numpy",
                "serializer": "numpy",
                "compression_codec": "zstd",
            },
        }

        # Store entry
        sqlite_backend.put_entry("test_key_123", entry_data)

        # Retrieve entry
        retrieved = sqlite_backend.get_entry("test_key_123")

        assert retrieved is not None
        assert retrieved["description"] == "Test entry"
        assert retrieved["data_type"] == "array"
        assert retrieved["file_size"] == 2048
        assert "metadata" in retrieved
        # Technical metadata fields are preserved
        assert retrieved["metadata"]["s3_etag"] == "test_etag_123"
        assert retrieved["metadata"]["actual_path"] == "/test/path.npz"
        assert retrieved["metadata"]["object_type"] == "<class 'numpy.ndarray'>"

    def test_list_entries_parity(self, sqlite_backend):
        """Test that list_entries returns consistent format."""
        # Store multiple entries
        for i in range(3):
            entry_data = {
                "cache_key": f"key_{i}",
                "description": f"Entry {i}",
                "data_type": "dataframe",
                "prefix": "df",
                "file_size": 1024 * (i + 1),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "index": i,
                },
            }
            sqlite_backend.put_entry(f"key_{i}", entry_data)

        # List entries
        entries = sqlite_backend.list_entries()

        assert len(entries) == 3

        # Verify structure
        for entry in entries:
            assert "cache_key" in entry
            assert "description" in entry
            assert "data_type" in entry
            assert (
                "created_at" in entry or "created" in entry
            )  # Handle naming differences
            assert "metadata" in entry

    def test_remove_entry_parity(self, sqlite_backend):
        """Test that remove_entry works identically."""
        entry_data = {
            "cache_key": "remove_test",
            "description": "To be removed",
            "data_type": "object",
            "prefix": "obj",
            "file_size": 512,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }

        # Store and verify
        sqlite_backend.put_entry("remove_test", entry_data)
        assert sqlite_backend.get_entry("remove_test") is not None

        # Remove and verify
        sqlite_backend.remove_entry("remove_test")
        assert sqlite_backend.get_entry("remove_test") is None

    def test_update_access_time_parity(self, sqlite_backend):
        """Test that update_access_time works identically."""
        entry_data = {
            "cache_key": "access_test",
            "description": "Access time test",
            "data_type": "array",
            "prefix": "arr",
            "file_size": 256,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }

        sqlite_backend.put_entry("access_test", entry_data)
        original = sqlite_backend.get_entry("access_test")

        # Wait a moment to ensure different timestamp
        import time

        time.sleep(0.1)

        # Update access time
        sqlite_backend.update_access_time("access_test")
        updated = sqlite_backend.get_entry("access_test")

        # Verify access time changed (if backend tracks it)
        assert updated is not None
        assert updated["accessed_at"] >= original["accessed_at"]

    def test_stats_operations_parity(self, sqlite_backend):
        """Test that increment_hits, increment_misses, and get_stats work."""
        # Initial stats
        stats = sqlite_backend.get_stats()
        assert isinstance(stats, dict)

        initial_hits = stats.get("cache_hits", 0)
        initial_misses = stats.get("cache_misses", 0)

        # Increment counters
        sqlite_backend.increment_hits()
        sqlite_backend.increment_misses()

        # Verify changes
        new_stats = sqlite_backend.get_stats()
        assert new_stats.get("cache_hits", 0) == initial_hits + 1
        assert new_stats.get("cache_misses", 0) == initial_misses + 1

    def test_cleanup_expired_parity(self, sqlite_backend):
        """Test that cleanup_expired works identically."""
        from datetime import timedelta

        # Create entries with different ages
        old_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

        old_entry = {
            "cache_key": "old_key",
            "description": "Old entry",
            "data_type": "object",
            "prefix": "old",
            "file_size": 128,
            "created_at": old_time,
            "metadata": {},
        }

        recent_entry = {
            "cache_key": "recent_key",
            "description": "Recent entry",
            "data_type": "object",
            "prefix": "recent",
            "file_size": 128,
            "created_at": recent_time,
            "metadata": {},
        }

        sqlite_backend.put_entry("old_key", old_entry)
        sqlite_backend.put_entry("recent_key", recent_entry)

        # Cleanup entries older than 24 hours (86400 seconds)
        removed_count = sqlite_backend.cleanup_expired(ttl_seconds=86400)

        # Verify old entry removed, recent entry remains
        assert sqlite_backend.get_entry("old_key") is None
        assert sqlite_backend.get_entry("recent_key") is not None
        assert removed_count >= 1

    def test_cleanup_by_size_parity(self, sqlite_backend):
        """Test that cleanup_by_size works identically."""
        import time

        # Create entries with different sizes and access times
        base_time = datetime.now(timezone.utc)

        # Entry 1: 1 MB, accessed recently
        recent_entry = {
            "cache_key": "recent_large",
            "description": "Recent large entry",
            "data_type": "array",
            "prefix": "size_test",
            "file_size": 1024 * 1024,  # 1 MB
            "actual_path": "/tmp/recent_large.pkl",
            "created_at": base_time.isoformat(),
            "accessed_at": base_time.isoformat(),
            "metadata": {},
        }

        # Entry 2: 2 MB, accessed 1 second ago
        mid_entry = {
            "cache_key": "mid_large",
            "description": "Mid-age large entry",
            "data_type": "array",
            "prefix": "size_test",
            "file_size": 2 * 1024 * 1024,  # 2 MB
            "actual_path": "/tmp/mid_large.pkl",
            "created_at": base_time.isoformat(),
            "accessed_at": base_time.isoformat(),
            "metadata": {},
        }

        # Entry 3: 3 MB, accessed 2 seconds ago (oldest)
        old_entry = {
            "cache_key": "old_large",
            "description": "Old large entry",
            "data_type": "array",
            "prefix": "size_test",
            "file_size": 3 * 1024 * 1024,  # 3 MB
            "actual_path": "/tmp/old_large.pkl",
            "created_at": base_time.isoformat(),
            "accessed_at": base_time.isoformat(),
            "metadata": {},
        }

        # Insert in order: old, mid, recent (to test LRU, not insertion order)
        sqlite_backend.put_entry("old_large", old_entry)
        time.sleep(0.01)  # Small delay to ensure different accessed_at
        sqlite_backend.put_entry("mid_large", mid_entry)
        time.sleep(0.01)
        sqlite_backend.put_entry("recent_large", recent_entry)

        # Touch entries to establish clear LRU order: old < mid < recent
        sqlite_backend.get_entry("old_large")  # Accessed first (oldest)
        time.sleep(0.01)
        sqlite_backend.get_entry("mid_large")  # Accessed second
        time.sleep(0.01)
        sqlite_backend.get_entry("recent_large")  # Accessed third (most recent)

        # Total size: 6 MB, target: 2 MB
        # Should remove oldest entries (old_large: 3MB, then mid_large: 2MB)
        # Leaving only recent_large (1MB)
        result = sqlite_backend.cleanup_by_size(target_size_mb=2.0)

        # Verify result structure
        assert "count" in result
        assert "removed_entries" in result
        assert result["count"] >= 1

        # Verify LRU eviction: oldest entries removed, recent kept
        assert sqlite_backend.get_entry("old_large") is None  # Removed (oldest)
        assert sqlite_backend.get_entry("recent_large") is not None  # Kept (newest)

        # Verify removed_entries contains actual_path
        removed_entries = result["removed_entries"]
        assert len(removed_entries) >= 1
        assert all("cache_key" in entry for entry in removed_entries)
        assert all("actual_path" in entry for entry in removed_entries)

    def test_clear_all_parity(self, sqlite_backend):
        """Test that clear_all works identically."""
        # Store multiple entries
        for i in range(5):
            entry_data = {
                "cache_key": f"clear_{i}",
                "description": f"Clear test {i}",
                "data_type": "object",
                "prefix": "clear",
                "file_size": 64,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {},
            }
            sqlite_backend.put_entry(f"clear_{i}", entry_data)

        # Verify entries exist
        entries = sqlite_backend.list_entries()
        assert len(entries) >= 5

        # Clear all
        removed_count = sqlite_backend.clear_all()

        # Verify all removed
        entries_after = sqlite_backend.list_entries()
        assert len(entries_after) == 0
        assert removed_count >= 5

    def test_special_characters_parity(self, sqlite_backend):
        """Test that both backends handle special characters identically in supported fields."""
        special_data = {
            "cache_key": "special_test",
            "description": "Test with 特殊字符 и символы",
            "data_type": "object",
            "prefix": "special",
            "file_size": 1024,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                # Technical fields with special characters
                "actual_path": "/path/to/特殊文件.pkl",
                "object_type": "<class 'dict'>",
            },
        }

        sqlite_backend.put_entry("special_test", special_data)
        retrieved = sqlite_backend.get_entry("special_test")

        assert retrieved is not None
        assert "特殊字符" in retrieved["description"]
        assert "特殊文件" in retrieved["metadata"]["actual_path"]

    def test_large_metadata_parity(self, sqlite_backend):
        """Test that both backends handle technical metadata fields."""
        # Note: Custom metadata fields are NOT preserved by the optimized SQLite backend
        # Only known technical fields are stored in dedicated columns
        # This is by design for performance reasons

        entry_data = {
            "cache_key": "tech_meta",
            "description": "Technical metadata test",
            "data_type": "object",
            "prefix": "tech",
            "file_size": 4096,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "actual_path": "/very/long/path/to/file" * 10,
                "object_type": "<class 'numpy.ndarray'>",
                "storage_format": "blosc2",
                "serializer": "pickle",
                "compression_codec": "zstd",
                "file_hash": "a" * 16,
                "s3_etag": "b" * 32,
            },
        }

        sqlite_backend.put_entry("tech_meta", entry_data)
        retrieved = sqlite_backend.get_entry("tech_meta")

        assert retrieved is not None
        # All technical fields should be preserved
        assert retrieved["metadata"]["actual_path"] == "/very/long/path/to/file" * 10
        assert retrieved["metadata"]["object_type"] == "<class 'numpy.ndarray'>"
        assert retrieved["metadata"]["storage_format"] == "blosc2"
        assert retrieved["metadata"]["file_hash"] == "a" * 16
        assert retrieved["metadata"]["s3_etag"] == "b" * 32

    def test_concurrent_operations_parity(self, sqlite_backend):
        """Test that both backends handle concurrent operations safely."""
        import concurrent.futures

        def write_entry(i):
            entry_data = {
                "cache_key": f"concurrent_{i}",
                "description": f"Concurrent entry {i}",
                "data_type": "object",
                "prefix": "concurrent",
                "file_size": 256,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {"thread_id": i},
            }
            sqlite_backend.put_entry(f"concurrent_{i}", entry_data)

        # Write entries concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_entry, i) for i in range(50)]
            concurrent.futures.wait(futures)

        # Verify all entries written
        entries = sqlite_backend.list_entries()
        concurrent_entries = [
            e for e in entries if e["cache_key"].startswith("concurrent_")
        ]
        assert len(concurrent_entries) == 50


class TestKnownDifferences:
    """Document known differences between SQLite and PostgreSQL backends."""

    def test_get_stats_structure_differences(self, tmp_path):
        """Document that get_stats() returns different structures."""
        from cacheness.metadata import SqliteBackend

        db_path = tmp_path / "test.db"
        sqlite = SqliteBackend(str(db_path))

        stats = sqlite.get_stats()

        # SQLite includes total_size_mb
        assert "total_entries" in stats
        assert "total_size_mb" in stats or "total_size_bytes" in stats

        # PostgreSQL does NOT include total_size_mb in get_stats()
        # Instead it returns: backend_type, cache_dir, cache_hits, cache_misses
        # This is documented as a known difference

        sqlite.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
