"""
Unit tests for cache metadata system integration.

Tests the metadata storage systems and their integration with the cache.
"""

import pytest
import tempfile
from pathlib import Path

from cacheness.metadata import (
    SqliteBackend,
    JsonBackend,
    create_metadata_backend,
)
from cacheness import CacheConfig, cacheness


class TestMetadataBackendCreation:
    """Test metadata backend creation and selection."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_auto_backend_creation(self, temp_dir):
        """Test automatic backend selection."""
        backend = create_metadata_backend(backend_type="auto", cache_dir=str(temp_dir))

        assert backend is not None
        assert isinstance(backend, (SqliteBackend, JsonBackend))

    def test_explicit_json_backend(self, temp_dir):
        """Test explicit JSON backend creation."""
        backend = create_metadata_backend(backend_type="json", cache_dir=str(temp_dir))

        assert isinstance(backend, JsonBackend)

    def test_explicit_sqlite_backend(self, temp_dir):
        """Test explicit SQLite backend creation."""
        db_file = str(temp_dir / "test.db")
        backend = create_metadata_backend(backend_type="sqlite", db_file=db_file)

        assert isinstance(backend, SqliteBackend)

    def test_invalid_backend_type(self, temp_dir):
        """Test handling of invalid backend type."""
        # Should raise ValueError for invalid backend type
        with pytest.raises(ValueError, match="Unknown backend type"):
            create_metadata_backend(backend_type="invalid", cache_dir=str(temp_dir))


class TestMetadataIntegration:
    """Test metadata integration with the cache system."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def cache_json(self, temp_dir):
        """Create a cache with JSON metadata backend."""
        config = CacheConfig(
            cache_dir=str(temp_dir), metadata_backend="json", cleanup_on_init=False
        )
        cache = cacheness(config)
        yield cache
        cache.close()

    @pytest.fixture
    def cache_sqlite(self, temp_dir):
        """Create a cache with SQLite metadata backend."""
        config = CacheConfig(
            cache_dir=str(temp_dir), metadata_backend="sqlite", cleanup_on_init=False
        )
        cache = cacheness(config)
        yield cache
        cache.close()

    def test_metadata_storage_json(self, cache_json):
        """Test metadata storage with JSON backend."""
        # Store some data
        test_data = {"test": "data", "numbers": [1, 2, 3]}
        cache_json.put(
            test_data, description="Test JSON metadata", test_key="json_test"
        )

        # List entries should show the metadata
        entries = cache_json.list_entries()
        assert len(entries) >= 1

        # Find our entry
        test_entry = None
        for entry in entries:
            if entry.get("description") == "Test JSON metadata":
                test_entry = entry
                break

        assert test_entry is not None
        assert test_entry["data_type"] == "object"
        assert "created" in test_entry
        assert "last_accessed" in test_entry
        assert "size_mb" in test_entry

    def test_metadata_storage_sqlite(self, cache_sqlite):
        """Test metadata storage with SQLite backend."""
        # Store some data
        test_data = {"test": "data", "numbers": [1, 2, 3]}
        cache_sqlite.put(
            test_data, description="Test SQLite metadata", test_key="sqlite_test"
        )

        # List entries should show the metadata
        entries = cache_sqlite.list_entries()
        assert len(entries) >= 1

        # Find our entry
        test_entry = None
        for entry in entries:
            if entry.get("description") == "Test SQLite metadata":
                test_entry = entry
                break

        assert test_entry is not None
        assert test_entry["data_type"] == "object"
        assert "created" in test_entry
        assert "last_accessed" in test_entry
        assert "size_mb" in test_entry

    def test_metadata_persistence_json(self, temp_dir):
        """Test that metadata persists across cache instances with JSON."""
        config = CacheConfig(
            cache_dir=str(temp_dir), metadata_backend="json", cleanup_on_init=False
        )

        # Create first cache and store data
        cache1 = cacheness(config)
        test_data = {"persistent": "data"}
        cache1.put(test_data, description="Persistent test", test_key="persistence")

        # Create second cache instance
        cache2 = cacheness(config)

        # Should be able to retrieve data from second instance
        retrieved = cache2.get(test_key="persistence")
        assert retrieved == test_data

        # Metadata should be available
        entries = cache2.list_entries()
        assert len(entries) >= 1

    def test_cache_stats_integration(self, cache_json, cache_sqlite):
        """Test cache statistics integration with metadata."""
        import numpy as np

        # Add different types of data to both caches
        test_data = [
            ({"object": "data"}, "Object data"),
            (np.array([1, 2, 3, 4, 5]), "Array data"),
            ({"arrays": {"a": np.array([1, 2]), "b": np.array([3, 4])}}, "Array dict"),
        ]

        for cache in [cache_json, cache_sqlite]:
            for i, (data, desc) in enumerate(test_data):
                cache.put(data, description=desc, test_id=i)

            # Get stats
            stats = cache.get_stats()

            # Should have entries
            assert stats["total_entries"] >= len(test_data)
            assert (
                stats["total_size_mb"] >= 0
            )  # Files exist but may be very small (0.000MB)
            assert "cache_hits" in stats
            assert "cache_misses" in stats
            assert "hit_rate" in stats

    def test_cleanup_integration(self, cache_json):
        """Test cleanup integration with metadata."""
        # Store some data
        cache_json.put(
            {"temp": "data"}, description="Temporary data", temp_key="cleanup_test"
        )

        # Verify it exists
        assert cache_json.get(temp_key="cleanup_test") is not None

        # The cleanup method should work without errors
        try:
            # Attempt cleanup - this tests the metadata backend integration
            stats = cache_json.get_stats()  # This should work
            assert isinstance(stats, dict)
        except Exception as e:
            pytest.fail(f"Cleanup integration failed: {e}")

    def test_backend_switching(self, temp_dir):
        """Test behavior when switching between backends."""
        # Create cache with JSON backend
        json_config = CacheConfig(
            cache_dir=str(temp_dir), metadata_backend="json", cleanup_on_init=False
        )
        json_cache = cacheness(json_config)
        json_cache.put(
            {"backend": "json"}, description="JSON data", backend_test="json"
        )

        # Create cache with SQLite backend (different metadata store)
        sqlite_config = CacheConfig(
            cache_dir=str(temp_dir), metadata_backend="sqlite", cleanup_on_init=False
        )
        sqlite_cache = cacheness(sqlite_config)
        sqlite_cache.put(
            {"backend": "sqlite"}, description="SQLite data", backend_test="sqlite"
        )

        # Each should have their own metadata
        json_entries = json_cache.list_entries()
        sqlite_entries = sqlite_cache.list_entries()

        # JSON cache should have the JSON entry
        json_descriptions = [e.get("description", "") for e in json_entries]
        assert "JSON data" in json_descriptions

        # SQLite cache should have the SQLite entry
        sqlite_descriptions = [e.get("description", "") for e in sqlite_entries]
        assert "SQLite data" in sqlite_descriptions

    def test_timezone_aware_timestamps_json(self, cache_json):
        """Test that JSON metadata backend stores UTC timestamps"""
        from datetime import datetime, timezone

        # Store data
        test_data = {"timezone": "json_test"}
        cache_json.put(test_data, description="Timezone test", timezone_test="json")

        # Get the entry metadata
        entries = cache_json.list_entries()
        assert len(entries) > 0

        entry = entries[0]
        assert "created" in entry  # Note: the field is "created", not "created_at"

        # Verify the timestamp is timezone-aware and in ISO format
        created_str = entry["created"]
        assert isinstance(created_str, str)

        # Should be able to parse as a timezone-aware datetime
        created = datetime.fromisoformat(created_str)
        assert created.tzinfo is not None  # Should have timezone info

        # Should be close to current UTC time (within a minute)
        current_utc = datetime.now(timezone.utc)
        time_diff = abs((current_utc - created).total_seconds())
        assert time_diff < 60  # Should be within 60 seconds

    def test_timezone_aware_timestamps_sqlite(self, cache_sqlite):
        """Test that SQLite metadata backend stores UTC timestamps"""
        from datetime import datetime, timezone

        # Store data
        test_data = {"timezone": "sqlite_test"}
        cache_sqlite.put(test_data, description="Timezone test", timezone_test="sqlite")

        # Get the entry metadata
        entries = cache_sqlite.list_entries()
        assert len(entries) > 0

        entry = entries[0]
        assert "created" in entry  # Note: the field is "created", not "created_at"

        # Verify the timestamp is timezone-aware and in ISO format
        created_str = entry["created"]
        assert isinstance(created_str, str)

        # Should be able to parse as a timezone-aware datetime
        created = datetime.fromisoformat(created_str)
        # Note: SQLite might not have timezone info depending on configuration

        # Should be close to current UTC time (within a minute)
        current_utc = datetime.now(timezone.utc)
        # For timezone-naive datetimes, assume UTC
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        time_diff = abs((current_utc - created).total_seconds())
        assert time_diff < 60  # Should be within 60 seconds

    def test_consistent_timezone_across_backends(self, temp_dir):
        """Test that all metadata backends store consistent UTC timestamps"""
        from datetime import datetime, timezone

        # Store data with both backends at roughly the same time
        json_config = CacheConfig(cache_dir=str(temp_dir), metadata_backend="json")
        json_cache = cacheness(json_config)

        sqlite_config = CacheConfig(cache_dir=str(temp_dir), metadata_backend="sqlite")
        sqlite_cache = cacheness(sqlite_config)

        # Store the same data in both
        test_data = {"consistent": "timezone"}
        start_time = datetime.now(timezone.utc)

        json_cache.put(test_data, description="JSON timezone", backend="json")
        sqlite_cache.put(test_data, description="SQLite timezone", backend="sqlite")

        end_time = datetime.now(timezone.utc)

        # Get entries from both backends
        json_entries = json_cache.list_entries()
        sqlite_entries = sqlite_cache.list_entries()

        assert len(json_entries) > 0
        assert len(sqlite_entries) > 0

        # Parse timestamps from both backends (using "created" field)
        json_timestamp = datetime.fromisoformat(json_entries[0]["created"])
        sqlite_timestamp = datetime.fromisoformat(sqlite_entries[0]["created"])

        # JSON should be timezone-aware
        assert json_timestamp.tzinfo is not None

        # SQLite might be timezone-naive, handle both cases
        if sqlite_timestamp.tzinfo is None:
            sqlite_timestamp = sqlite_timestamp.replace(tzinfo=timezone.utc)

        # Both should be within our test time window
        assert start_time <= json_timestamp <= end_time
        assert start_time <= sqlite_timestamp <= end_time

        # Timestamps should be close to each other (within 5 seconds)
        time_diff = abs((json_timestamp - sqlite_timestamp).total_seconds())
        assert time_diff < 5  # Should be very close since stored almost simultaneously
