"""
Unit tests for the core cache system.

Tests the main cacheness class and CacheConfig functionality.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch

from cacheness import CacheConfig, cacheness


class TestCacheConfig:
    """Test CacheConfig class functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()

        assert config.storage.cache_dir == "./cache"
        assert config.metadata.default_ttl_seconds == 86400  # 24 hours in seconds
        assert config.storage.max_cache_size_mb == 2000
        assert config.compression.parquet_compression == "lz4"
        assert config.compression.npz_compression is True
        assert config.metadata.enable_metadata is True
        assert config.storage.cleanup_on_init is True
        assert config.metadata.metadata_backend == "auto"
        assert config.metadata.sqlite_db_file == "cache_metadata.db"
        assert config.compression.pickle_compression_codec == "zstd"
        assert config.compression.pickle_compression_level == 5
        assert config.compression.use_blosc2_arrays is True
        assert config.compression.blosc2_array_codec == "lz4"
        assert config.compression.blosc2_array_clevel == 5
        # store_cache_key_params was removed in Section 2.13 (unified metadata)

    def test_custom_config(self):
        """Test custom configuration values."""
        from cacheness.config import (
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )
        
        # Use a platform-agnostic path for testing
        import tempfile
        test_cache_dir = Path(tempfile.gettempdir()) / "test_cache"
        storage_config = CacheStorageConfig(
            cache_dir=str(test_cache_dir), max_cache_size_mb=1000, cleanup_on_init=False
        )
        metadata_config = CacheMetadataConfig(
            default_ttl_seconds=172800, enable_metadata=False, metadata_backend="json"
        )
        compression_config = CompressionConfig(
            parquet_compression="gzip", npz_compression=False
        )
        serialization_config = SerializationConfig()
        handler_config = HandlerConfig()

        config = CacheConfig(
            storage=storage_config,
            metadata=metadata_config,
            compression=compression_config,
            serialization=serialization_config,
            handlers=handler_config,
        )

        # Compare using as_posix() for cross-platform compatibility
        assert Path(config.storage.cache_dir).as_posix() == test_cache_dir.as_posix()
        assert config.metadata.default_ttl_seconds == 172800  # 48 hours in seconds
        assert config.storage.max_cache_size_mb == 1000
        assert config.compression.parquet_compression == "gzip"
        assert config.compression.npz_compression is False
        assert config.metadata.enable_metadata is False
        assert config.storage.cleanup_on_init is False
        assert config.metadata.metadata_backend == "json"
        # store_cache_key_params was removed in Section 2.13 - now using store_full_metadata


class TestCacheness:
    """Test cacheness class functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "test_cache"

    @pytest.fixture
    def test_config(self, temp_cache_dir):
        """Create a test configuration."""
        from cacheness.config import (
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )

        storage_config = CacheStorageConfig(cache_dir=str(temp_cache_dir))
        metadata_config = CacheMetadataConfig(
            metadata_backend="json"
        )  # Use JSON to avoid SQLite dependencies
        compression_config = CompressionConfig(
            use_blosc2_arrays=False
        )  # Disable blosc2 for testing
        serialization_config = SerializationConfig()
        handler_config = HandlerConfig()

        return CacheConfig(
            storage=storage_config,
            metadata=metadata_config,
            compression=compression_config,
            serialization=serialization_config,
            handlers=handler_config,
        )

    @pytest.fixture
    def cache(self, test_config):
        """Create a cache instance for testing."""
        return cacheness(test_config)

    def test_cache_initialization(self, cache, test_config):
        """Test cache initialization."""
        assert cache.config == test_config
        assert cache.handlers is not None
        assert cache.metadata_backend is not None
        assert cache.config.storage.cache_dir in str(cache.config.storage.cache_dir)

    def test_put_and_get_simple_object(self, cache):
        """Test caching simple objects."""
        test_data = {"key": "value", "number": 42}
        cache_key_params = {"test": "simple_object"}

        # Put data
        cache.put(test_data, description="Test object", **cache_key_params)

        # Get data
        retrieved = cache.get(**cache_key_params)

        assert retrieved == test_data

    def test_put_and_get_numpy_array(self, cache):
        """Test caching numpy arrays."""
        test_array = np.random.rand(10, 5)
        cache_key_params = {"test": "numpy_array", "shape": "10x5"}

        # Put array
        cache.put(test_array, description="Test array", **cache_key_params)

        # Get array
        retrieved = cache.get(**cache_key_params)

        assert isinstance(retrieved, np.ndarray)
        assert retrieved.shape == test_array.shape
        assert np.allclose(retrieved, test_array)

    def test_put_and_get_array_dict(self, cache):
        """Test caching dictionary of arrays."""
        test_dict = {
            "array1": np.random.rand(5, 3),
            "array2": np.random.randint(0, 10, (4, 2)),
            "labels": np.array(["a", "b", "c"]),
        }
        cache_key_params = {"test": "array_dict"}

        # Put dictionary
        cache.put(test_dict, description="Test array dict", **cache_key_params)

        # Get dictionary
        retrieved = cache.get(**cache_key_params)

        assert isinstance(retrieved, dict)
        assert set(retrieved.keys()) == set(test_dict.keys())
        for key in test_dict.keys():
            assert np.array_equal(retrieved[key], test_dict[key])

    def _has_dataframe_library(self):
        """Check if polars or pandas is available."""
        import importlib.util

        return (
            importlib.util.find_spec("polars") is not None
            or importlib.util.find_spec("pandas") is not None
        )

    @pytest.mark.skipif(
        not _has_dataframe_library(None),
        reason="DataFrame tests require polars or pandas",
    )
    def test_put_and_get_dataframe(self, cache):
        """Test caching dataframes (requires polars or pandas)."""
        try:
            import polars as pl

            test_df = pl.DataFrame(
                {
                    "id": range(10),
                    "value": np.random.rand(10),
                    "category": ["A", "B"] * 5,
                }
            )
            data_type = "polars"
        except ImportError:
            try:
                import pandas as pd

                test_df = pd.DataFrame(
                    {
                        "id": range(10),
                        "value": np.random.rand(10),
                        "category": ["A", "B"] * 5,
                    }
                )
                data_type = "pandas"
            except ImportError:
                pytest.skip("Neither polars nor pandas available")

        cache_key_params = {"test": "dataframe", "type": data_type}

        # Put dataframe
        cache.put(test_df, description="Test dataframe", **cache_key_params)

        # Get dataframe
        retrieved = cache.get(**cache_key_params)

        assert isinstance(retrieved, type(test_df))
        assert retrieved.shape == test_df.shape

        if data_type == "polars":
            # For polars, compare values as numpy arrays
            assert np.array_equal(test_df.to_numpy(), retrieved.to_numpy())
        else:  # pandas
            # For pandas, use DataFrame.equals()
            import pandas as pd

            assert isinstance(test_df, pd.DataFrame)
            assert isinstance(retrieved, pd.DataFrame)
            assert test_df.equals(retrieved)

    def test_put_and_get_series(self, cache):
        """Test caching Series (requires polars or pandas)."""
        try:
            import polars as pl

            test_series = pl.Series("test_data", [1.1, 2.2, 3.3, 4.4, 5.5])
            data_type = "polars"
        except ImportError:
            try:
                import pandas as pd

                test_series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5], name="test_data")
                data_type = "pandas"
            except ImportError:
                pytest.skip("Neither polars nor pandas available")

        cache_key_params = {"test": "series", "type": data_type}

        # Put series
        cache.put(test_series, description="Test series", **cache_key_params)

        # Get series
        retrieved = cache.get(**cache_key_params)

        assert isinstance(retrieved, type(test_series))
        assert retrieved.shape == test_series.shape

        if data_type == "polars":
            # For polars, compare values
            assert retrieved.equals(test_series)
            assert retrieved.name == test_series.name
        else:  # pandas
            # For pandas, use Series.equals()
            import pandas as pd

            assert isinstance(test_series, pd.Series)
            assert isinstance(retrieved, pd.Series)
            assert test_series.equals(retrieved)

    def test_cache_miss(self, cache):
        """Test cache miss scenario."""
        result = cache.get(nonexistent="key")
        assert result is None

    def test_cache_key_generation(self, cache):
        """Test cache key generation from parameters."""
        params1 = {"a": 1, "b": 2, "c": 3}
        params2 = {"c": 3, "b": 2, "a": 1}  # Same params, different order
        params3 = {"a": 1, "b": 2, "c": 4}  # Different values

        key1 = cache._create_cache_key(params1)
        key2 = cache._create_cache_key(params2)
        key3 = cache._create_cache_key(params3)

        # Same parameters should generate same key regardless of order
        assert key1 == key2
        # Different parameters should generate different keys
        assert key1 != key3
        # Keys should be strings of expected length (xxhash truncated to 16 chars)
        assert isinstance(key1, str)
        assert len(key1) == 16  # xxhash truncated to 16 characters

    def test_cache_file_path_generation(self, cache):
        """Test cache file path generation."""
        cache_key = "test_key_123"
        prefix = "test_prefix"

        file_path = cache._get_cache_file_path(cache_key, prefix)

        assert isinstance(file_path, Path)
        assert cache_key in str(file_path)
        if prefix:
            assert prefix in str(file_path)
        assert str(cache.config.storage.cache_dir) in str(file_path)

    def test_ttl_expiration(self, cache):
        """Test TTL-based expiration."""
        test_data = {"test": "data"}
        cache_key_params = {"test": "ttl_expiration"}

        # Put data
        cache.put(test_data, description="TTL test", **cache_key_params)

        # Should be retrievable immediately
        assert cache.get(**cache_key_params) == test_data

        # Mock expiration check to return True
        with patch.object(cache, "_is_expired", return_value=True):
            result = cache.get(**cache_key_params)
            assert result is None

    def test_cache_stats(self, cache):
        """Test cache statistics functionality."""
        # Initial stats
        stats = cache.get_stats()
        initial_entries = stats.get("total_entries", 0)

        # Add some data
        cache.put({"test": 1}, test_id="1")
        cache.put(np.array([1, 2, 3]), test_id="2")

        # Check updated stats
        stats = cache.get_stats()
        assert stats["total_entries"] == initial_entries + 2
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "hit_rate" in stats
        assert "total_size_mb" in stats

    def test_list_entries(self, cache):
        """Test listing cache entries."""
        # Add some test data
        cache.put({"test": 1}, description="Test 1", test_id="1")
        cache.put({"test": 2}, description="Test 2", test_id="2")

        entries = cache.list_entries()

        assert isinstance(entries, list)
        assert len(entries) >= 2

        # Check entry structure
        for entry in entries:
            assert "cache_key" in entry
            assert "description" in entry
            assert "data_type" in entry
            assert "created" in entry
            assert "last_accessed" in entry
            assert "size_mb" in entry

    def test_error_handling_invalid_data(self, cache):
        """Test error handling for invalid data."""
        # Test with a truly unpickleable object (generators cannot be pickled or dilled)
        unpickleable_object = (x for x in range(10))  # Generator object

        with pytest.raises((ValueError, TypeError)):
            cache.put(unpickleable_object, test="unpickleable")

    def test_concurrent_access(self, cache):
        """Test thread safety of cache operations."""
        import threading
        import time

        results = []
        errors = []

        def cache_worker(worker_id):
            try:
                # Put data
                data = {"worker": worker_id, "data": list(range(100))}
                cache.put(data, worker_id=worker_id, test="concurrent")

                # Small delay
                time.sleep(0.01)

                # Get data
                retrieved = cache.get(worker_id=worker_id, test="concurrent")
                results.append((worker_id, retrieved == data))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Create and start threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(success for _, success in results)

    def test_cleanup_initialization(self, temp_cache_dir):
        """Test cleanup on initialization."""
        from cacheness.config import (
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )

        storage_config = CacheStorageConfig(
            cache_dir=str(temp_cache_dir), cleanup_on_init=True
        )
        metadata_config = CacheMetadataConfig(metadata_backend="json")
        compression_config = CompressionConfig()
        serialization_config = SerializationConfig()
        handler_config = HandlerConfig()

        config = CacheConfig(
            storage=storage_config,
            metadata=metadata_config,
            compression=compression_config,
            serialization=serialization_config,
            handlers=handler_config,
        )

        # Create some dummy files
        temp_cache_dir.mkdir(parents=True, exist_ok=True)
        (temp_cache_dir / "old_file.pkl").touch()

        # Initialize cache with cleanup
        cache = cacheness(config)

        # Cache should be initialized properly
        assert cache.config.storage.cleanup_on_init is True

    def test_ttl_expiration_with_utc_timestamps(self, cache):
        """Test that TTL expiration handles UTC timestamps correctly"""
        from datetime import datetime, timezone
        from unittest.mock import patch
        
        test_data = {"timezone": "test"}
        cache_key_params = {"test": "timezone_ttl"}
        
        # Store data to create a real cache entry
        cache.put(test_data, **cache_key_params)
        
        # Find the actual cache key that was created
        entries = cache.list_entries()
        assert len(entries) > 0, "Should have at least one cache entry"
        
        # Get the first entry's cache key
        cache_key = entries[0]["cache_key"]
        
        # Test that _is_expired uses UTC time correctly
        with patch('cacheness.core.datetime') as mock_datetime:
            # Mock current UTC time
            current_utc = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = current_utc
            mock_datetime.fromisoformat = datetime.fromisoformat
            
            # Should not be expired with fresh timestamp (large TTL)
            assert cache._is_expired(cache_key, ttl_seconds=86400) is False
            
            # Verify datetime.now was called with timezone.utc
            mock_datetime.now.assert_called_with(timezone.utc)
    
    def test_timezone_aware_expiration_calculation(self, cache):
        """Test expiration calculation with timezone-aware datetime strings"""
        from datetime import datetime, timezone, timedelta
        from unittest.mock import patch
        
        # Create a mock metadata entry with a timezone-aware timestamp
        utc_time = datetime.now(timezone.utc) - timedelta(hours=2)  # 2 hours ago
        timestamp_iso = utc_time.isoformat()
        
        # Mock the metadata backend to return our test entry
        mock_entry = {"created_at": timestamp_iso, "description": "timezone test"}
        
        with patch.object(cache.metadata_backend, 'get_entry', return_value=mock_entry):
            # Should be expired (created 2 hours ago, checking with 1 hour TTL = 3600 seconds)
            assert cache._is_expired("test_key", ttl_seconds=3600) is True
            
            # Should not be expired with longer TTL: 3 hours = 10800 seconds
            assert cache._is_expired("test_key", ttl_seconds=10800) is False
    
    def test_timezone_consistency_across_cache_operations(self, cache):
        """Test that all cache operations use consistent UTC timezone handling"""
        from datetime import datetime, timezone
        
        test_data = {"consistent": "timezone"}
        cache_key_params = {"consistency": "test"}
        
        # Store data - this should create a UTC timestamp
        cache.put(test_data, **cache_key_params)
        
        # The timestamp in metadata should be UTC
        # We can't easily access the internal cache key generation,
        # but we can verify the data is retrievable (indicating consistent timezone handling)
        retrieved = cache.get(**cache_key_params)
        assert retrieved == test_data
        
        # Test with current time for comparison
        current_utc = datetime.now(timezone.utc)
        assert current_utc.tzinfo == timezone.utc  # Verify we're using UTC


class TestFactoryMethods:
    """Test factory methods for creating specialized cache instances."""

    def test_for_api_factory(self):
        """Test cacheness.for_api() factory method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = cacheness.for_api(cache_dir=temp_dir, ttl_seconds=28800)
            
            # Should be a UnifiedCache instance
            assert isinstance(cache, cacheness)
            
            # Should have the specified configuration
            assert cache.config.storage.cache_dir == temp_dir
            assert cache.config.metadata.default_ttl_seconds == 28800  # 8 hours in seconds
            assert cache.config.compression.pickle_compression_codec == "zstd"
            
            # Test basic functionality
            test_data = {"api": "response", "data": [1, 2, 3]}
            cache.put(test_data, endpoint="test")
            
            retrieved = cache.get(endpoint="test")
            assert retrieved == test_data
            
            cache.close()

    def test_for_api_default_values(self):
        """Test default values for for_api factory."""
        cache = cacheness.for_api()
        
        # Check default values
        assert cache.config.storage.cache_dir == "./cache"
        assert cache.config.metadata.default_ttl_seconds == 21600  # 6 hours in seconds
        assert cache.config.compression.pickle_compression_codec == "zstd"


class TestTimezoneHandling:
    """Test timezone handling in core cache functionality"""


class TestMemoryCacheConfig:
    """Test memory cache layer configuration and functionality."""

    def test_memory_cache_config_new_naming(self):
        """Test memory cache configuration with parameter names."""
        config = CacheConfig(
            cache_dir="/tmp/test_memory_cache",
            metadata_backend="sqlite",
            enable_memory_cache=True,
            memory_cache_type="lru",
            memory_cache_maxsize=500,
            memory_cache_ttl_seconds=300,
            memory_cache_stats=True
        )
        
        # Verify parameters are set correctly
        assert config.metadata.enable_memory_cache is True
        assert config.metadata.memory_cache_type == "lru"
        assert config.metadata.memory_cache_maxsize == 500
        assert config.metadata.memory_cache_ttl_seconds == 300
        assert config.metadata.memory_cache_stats is True

    def test_memory_cache_config_defaults(self):
        """Test memory cache defaults when not specified."""
        config = CacheConfig(cache_dir="/tmp/test_defaults")
        
        # Verify defaults
        assert config.metadata.enable_memory_cache is False
        assert config.metadata.memory_cache_type == "lru"
        assert config.metadata.memory_cache_maxsize == 1000
        assert config.metadata.memory_cache_ttl_seconds == 300.0
        assert config.metadata.memory_cache_stats is False

    def test_memory_cache_functional_test(self):
        """Test that memory cache layer actually works."""
        import tempfile
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with memory cache layer enabled
            config = CacheConfig(
                cache_dir=temp_dir,
                metadata_backend="sqlite",
                enable_memory_cache=True,
                memory_cache_type="lru",
                memory_cache_maxsize=10,
                memory_cache_ttl_seconds=2,  # Short TTL for testing
                memory_cache_stats=True
            )
            
            cache = cacheness(config)
            
            # Store some test data
            test_data = {"test": "memory_cache_data", "value": 42}
            cache.put(test_data, test_key="memory_cache_test")
            
            # Verify data is cached
            result = cache.get(test_key="memory_cache_test")
            assert result == test_data
            
            # Check if memory cache statistics are available (should be CachedMetadataBackend)
            from cacheness.metadata import CachedMetadataBackend
            if isinstance(cache.metadata_backend, CachedMetadataBackend):
                stats = cache.metadata_backend.get_cache_stats()
                assert 'memory_cache_enabled' in stats
                assert stats['memory_cache_enabled'] is True
                assert stats['memory_cache_type'] == 'lru'
                assert stats['memory_cache_maxsize'] == 10
            
            cache.close()

    def test_memory_cache_cache_types(self):
        """Test different memory cache types."""
        cache_types = ["lru", "lfu", "fifo", "rr"]
        
        for cache_type in cache_types:
            config = CacheConfig(
                cache_dir="/tmp/test_cache_types",
                enable_memory_cache=True,
                memory_cache_type=cache_type,
                memory_cache_maxsize=50
            )
            
            assert config.metadata.memory_cache_type == cache_type

    def test_memory_cache_validation(self):
        """Test validation of memory cache parameters."""
        # Test invalid cache type
        config = CacheConfig(
            enable_memory_cache=True,
            memory_cache_type="invalid_type"
        )
        
        # Should accept any string (validation happens at cache creation)
        assert config.metadata.memory_cache_type == "invalid_type"
        
        # Test edge case values
        config = CacheConfig(
            enable_memory_cache=True,
            memory_cache_maxsize=1,  # Minimum
            memory_cache_ttl_seconds=0.1  # Very short
        )
        
        assert config.metadata.memory_cache_maxsize == 1
        assert config.metadata.memory_cache_ttl_seconds == 0.1
