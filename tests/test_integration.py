"""
Integration tests for the unified cache system.

Tests end-to-end functionality including all components working together.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path

from cacheness import cacheness, CacheConfig


class TestCacheIntegration:
    """Test complete cache system integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def cache_config(self, temp_dir):
        """Create a cache configuration for testing."""
        return CacheConfig(
            cache_dir=str(temp_dir),
            metadata_backend="json",  # Use JSON for simplicity
            cleanup_on_init=False,
            max_cache_size_mb=100,
        )

    @pytest.fixture
    def cache(self, cache_config):
        """Create a cache instance for testing."""
        return cacheness(cache_config)

    def test_end_to_end_workflow(self, cache):
        """Test complete end-to-end cache workflow."""
        # 1. Store different types of data
        test_data = {
            "simple_object": {"key": "value", "number": 42},
            "numpy_array": np.random.rand(10, 5),
            "array_dict": {
                "features": np.random.rand(100, 10),
                "labels": np.random.randint(0, 2, 100),
                "metadata": np.array(["train"] * 80 + ["test"] * 20),
            },
        }

        # Store each data type
        for data_type, data in test_data.items():
            cache.put(
                data,
                description=f"Test {data_type}",
                data_type=data_type,
                experiment="integration_test",
            )

        # 2. Retrieve and verify data
        for data_type, original_data in test_data.items():
            retrieved = cache.get(data_type=data_type, experiment="integration_test")

            if isinstance(original_data, dict) and any(
                isinstance(v, np.ndarray) for v in original_data.values()
            ):
                # Array dictionary
                assert isinstance(retrieved, dict)
                assert set(retrieved.keys()) == set(original_data.keys())
                for key in original_data.keys():
                    assert np.array_equal(retrieved[key], original_data[key])
            elif isinstance(original_data, np.ndarray):
                # Numpy array
                assert isinstance(retrieved, np.ndarray)
                assert np.array_equal(retrieved, original_data)
            else:
                # Simple object
                assert retrieved == original_data

        # 3. Test cache statistics
        stats = cache.get_stats()
        assert stats["total_entries"] >= len(test_data)
        assert stats["total_size_mb"] > 0
        assert stats["cache_hits"] >= len(test_data)  # From retrievals above

        # 4. Test listing entries
        entries = cache.list_entries()
        assert len(entries) >= len(test_data)

        # Check that our entries are in the list
        descriptions = [entry.get("description", "") for entry in entries]
        for data_type in test_data.keys():
            assert f"Test {data_type}" in descriptions

    def test_cache_persistence(self, temp_dir):
        """Test that cache persists across instances."""
        config = CacheConfig(
            cache_dir=str(temp_dir), metadata_backend="json", cleanup_on_init=False
        )

        # Create first cache instance and store data
        cache1 = cacheness(config)
        test_data = {"persistent_key": "persistent_value", "numbers": [1, 2, 3, 4, 5]}
        cache1.put(test_data, description="Persistence test", persistent="data")

        # Verify data exists
        assert cache1.get(persistent="data") == test_data

        # Create second cache instance (simulate restart)
        cache2 = cacheness(config)

        # Data should still be available
        retrieved = cache2.get(persistent="data")
        assert retrieved == test_data

        # Metadata should be available
        entries = cache2.list_entries()
        descriptions = [entry.get("description", "") for entry in entries]
        assert "Persistence test" in descriptions

    def test_cache_size_management(self, temp_dir):
        """Test cache size management and cleanup."""
        config = CacheConfig(
            cache_dir=str(temp_dir),
            metadata_backend="json",
            max_cache_size_mb=1,  # Small limit for testing
            cleanup_on_init=False,
        )
        cache = cacheness(config)

        # Add data that might exceed the limit
        for i in range(10):
            large_data = {"data": list(range(1000)), "id": i}
            cache.put(large_data, description=f"Large data {i}", size_test=i)

        # Cache should still function
        stats = cache.get_stats()
        assert stats["total_entries"] > 0

        # Should be able to retrieve recent data
        recent_data = cache.get(size_test=9)
        assert recent_data is not None

    def test_metadata_backends_comparison(self, temp_dir):
        """Test that both metadata backends produce similar results."""
        # Test with JSON backend
        json_config = CacheConfig(
            cache_dir=str(temp_dir / "json"),
            metadata_backend="json",
            cleanup_on_init=False,
        )
        json_cache = cacheness(json_config)

        # Test with SQLite backend
        sqlite_config = CacheConfig(
            cache_dir=str(temp_dir / "sqlite"),
            metadata_backend="sqlite",
            cleanup_on_init=False,
        )
        sqlite_cache = cacheness(sqlite_config)

        # Store same data in both caches
        test_data = {"comparison": "test", "value": 123}

        json_cache.put(test_data, description="Backend comparison", backend="json")
        sqlite_cache.put(test_data, description="Backend comparison", backend="sqlite")

        # Both should retrieve the same data
        json_result = json_cache.get(backend="json")
        sqlite_result = sqlite_cache.get(backend="sqlite")

        assert json_result == test_data
        assert sqlite_result == test_data
        assert json_result == sqlite_result

        # Both should provide similar statistics structure
        json_stats = json_cache.get_stats()
        sqlite_stats = sqlite_cache.get_stats()

        # Same keys should be present
        assert set(json_stats.keys()) == set(sqlite_stats.keys())

        # Both should have the entry
        json_entries = json_cache.list_entries()
        sqlite_entries = sqlite_cache.list_entries()

        assert len(json_entries) >= 1
        assert len(sqlite_entries) >= 1


class TestCacheConfiguration:
    """Test cache configuration options."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_different_configurations(self, temp_dir):
        """Test cache with different configuration options."""
        configs = [
            # JSON backend with compression
            CacheConfig(
                cache_dir=str(temp_dir / "config1"),
                metadata_backend="json",
                npz_compression=True,
                parquet_compression="lz4",
            ),
            # SQLite backend without compression
            CacheConfig(
                cache_dir=str(temp_dir / "config2"),
                metadata_backend="sqlite",
                npz_compression=False,
                parquet_compression="none",
            ),
            # Auto backend selection
            CacheConfig(cache_dir=str(temp_dir / "config3"), metadata_backend="auto"),
        ]

        test_data = np.random.rand(10, 10)

        for i, config in enumerate(configs):
            cache = cacheness(config)

            # Store and retrieve data
            cache.put(test_data, description=f"Config test {i}", config_test=i)
            retrieved = cache.get(config_test=i)

            assert retrieved is not None
            assert np.array_equal(retrieved, test_data)

            # Check that cache is functional
            stats = cache.get_stats()
            assert stats["total_entries"] >= 1

    def test_cache_directory_creation(self, temp_dir):
        """Test that cache directories are created automatically."""
        non_existent_dir = temp_dir / "nested" / "cache" / "dir"

        config = CacheConfig(cache_dir=str(non_existent_dir), metadata_backend="json")

        # Cache should create the directory
        cache = cacheness(config)

        assert non_existent_dir.exists()

        # Should be able to store data
        cache.put({"test": "data"}, directory_test="creation")
        retrieved = cache.get(directory_test="creation")
        assert retrieved == {"test": "data"}
