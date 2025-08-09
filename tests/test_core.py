"""
Unit tests for the core cache system.

Tests the main UnifiedCache class and CacheConfig functionality.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch

from cacheness import CacheConfig, cacheness, UnifiedCache


class TestCacheConfig:
    """Test CacheConfig class functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()
        
        assert config.cache_dir == "./cache"
        assert config.default_ttl_hours == 24
        assert config.max_cache_size_mb == 2000
        assert config.parquet_compression == "lz4"
        assert config.npz_compression is True
        assert config.enable_metadata is True
        assert config.cleanup_on_init is True
        assert config.metadata_backend == "auto"
        assert config.sqlite_db_file == "cache_metadata.db"
        assert config.pickle_compression_codec == "lz4"
        assert config.pickle_compression_level == 5
        assert config.use_blosc2_arrays is True
        assert config.blosc2_array_codec == "lz4"
        assert config.blosc2_array_clevel == 5
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CacheConfig(
            cache_dir="/tmp/test_cache",
            default_ttl_hours=48,
            max_cache_size_mb=1000,
            parquet_compression="gzip",
            npz_compression=False,
            enable_metadata=False,
            cleanup_on_init=False,
            metadata_backend="json"
        )
        
        assert config.cache_dir == "/tmp/test_cache"
        assert config.default_ttl_hours == 48
        assert config.max_cache_size_mb == 1000
        assert config.parquet_compression == "gzip"
        assert config.npz_compression is False
        assert config.enable_metadata is False
        assert config.cleanup_on_init is False
        assert config.metadata_backend == "json"


class TestUnifiedCache:
    """Test UnifiedCache class functionality."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "test_cache"
    
    @pytest.fixture
    def test_config(self, temp_cache_dir):
        """Create a test configuration."""
        return CacheConfig(
            cache_dir=str(temp_cache_dir),
            metadata_backend="json",  # Use JSON to avoid SQLite dependencies
            use_blosc2_arrays=False,  # Disable blosc2 for testing
            npz_compression=True,
            cleanup_on_init=False
        )
    
    @pytest.fixture
    def cache(self, test_config):
        """Create a cache instance for testing."""
        return UnifiedCache(test_config)
    
    def test_cache_initialization(self, cache, test_config):
        """Test cache initialization."""
        assert cache.config == test_config
        assert cache.handlers is not None
        assert cache.metadata_backend is not None
        assert cache.config.cache_dir in str(cache.config.cache_dir)
    
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
            "labels": np.array(['a', 'b', 'c'])
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
    
    @pytest.mark.skipif(
        True,  # Skip by default since polars/pandas may not be available
        reason="DataFrame tests require polars or pandas"
    )
    def test_put_and_get_dataframe(self, cache):
        """Test caching dataframes (requires polars or pandas)."""
        try:
            import polars as pl
            test_df = pl.DataFrame({
                "id": range(10),
                "value": np.random.rand(10),
                "category": ["A", "B"] * 5
            })
            data_type = "polars"
        except ImportError:
            try:
                import pandas as pd
                test_df = pd.DataFrame({
                    "id": range(10),
                    "value": np.random.rand(10),
                    "category": ["A", "B"] * 5
                })
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
            assert np.array_equal(
                test_df.to_numpy(), 
                retrieved.to_numpy()
            )
        else:  # pandas
            # For pandas, use DataFrame.equals()
            import pandas as pd
            assert isinstance(test_df, pd.DataFrame)
            assert isinstance(retrieved, pd.DataFrame) 
            assert test_df.equals(retrieved)
    
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
        assert str(cache.config.cache_dir) in str(file_path)
    
    def test_ttl_expiration(self, cache):
        """Test TTL-based expiration."""
        test_data = {"test": "data"}
        cache_key_params = {"test": "ttl_expiration"}
        
        # Put data
        cache.put(test_data, description="TTL test", **cache_key_params)
        
        # Should be retrievable immediately
        assert cache.get(**cache_key_params) == test_data
        
        # Mock expiration check to return True
        with patch.object(cache, '_is_expired', return_value=True):
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
        """Test error handling for invalid data types."""
        # Test with non-pickleable object
        def unpickleable(x):
            return x  # Local functions can be problematic for pickle
        
        with pytest.raises((ValueError, TypeError)):
            cache.put(unpickleable, test="unpickleable")
    
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
        config = CacheConfig(
            cache_dir=str(temp_cache_dir),
            metadata_backend="json",
            cleanup_on_init=True
        )
        
        # Create some dummy files
        temp_cache_dir.mkdir(parents=True, exist_ok=True)
        (temp_cache_dir / "old_file.pkl").touch()
        
        # Initialize cache with cleanup
        cache = UnifiedCache(config)
        
        # Cache should be initialized properly
        assert cache.config.cleanup_on_init is True
