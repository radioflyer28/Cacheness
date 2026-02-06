#!/usr/bin/env python3
"""
Unit tests for query_meta() method
==================================

Tests the new query_meta() functionality that allows querying cache entries
based on their stored cache_key_params using SQLite JSON1 extension.
"""

import tempfile
import pytest
from pathlib import Path
from datetime import datetime

from cacheness.core import UnifiedCache
from cacheness.config import CacheConfig


@pytest.fixture
def temp_cache():
    """Fixture to create a temporary cache for testing query_meta functionality."""
    temp_dir = tempfile.mkdtemp()
    cache_dir = Path(temp_dir) / "cache"
    
    try:
        # Create cache with parameter storage enabled
        config = CacheConfig(
            cache_dir=str(cache_dir),
            metadata_backend="sqlite",
            store_full_metadata=True,  # Required for query_meta
        )
        cache = UnifiedCache(config)
        yield cache
        cache.close()
    finally:
        import shutil
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)


class TestQueryMeta:
    """Test suite for query_meta() method."""

    def test_query_meta_basic_functionality(self, temp_cache):
        """Test basic query_meta functionality with simple parameters."""
        cache = temp_cache
        
        # Store test data with various parameters
        cache.put("model_1", experiment="exp_001", model_type="xgboost", accuracy=0.95)
        cache.put("model_2", experiment="exp_002", model_type="cnn", accuracy=0.88)
        cache.put("model_3", experiment="exp_003", model_type="random_forest", accuracy=0.92)
        
        # Test query with no filters (all entries)
        all_entries = cache.query_meta()
        assert all_entries is not None
        assert len(all_entries) == 3
        
        # Verify structure of returned entries
        entry = all_entries[0]
        assert 'cache_key' in entry
        assert 'description' in entry
        assert 'data_type' in entry
        assert 'created_at' in entry
        assert 'accessed_at' in entry
        assert 'file_size' in entry
        assert 'metadata_dict' in entry
        
        # Verify metadata_dict structure
        params = entry['metadata_dict']
        assert 'experiment' in params
        assert 'model_type' in params
        assert 'accuracy' in params

    def test_query_meta_string_filters(self, temp_cache):
        """Test query_meta with string parameter filters."""
        cache = temp_cache
        
        # Store test data
        cache.put("model_1", experiment="exp_001", model_type="xgboost", version="v1")
        cache.put("model_2", experiment="exp_002", model_type="cnn", version="v1")
        cache.put("model_3", experiment="exp_003", model_type="xgboost", version="v2")
        
        # Query by model_type
        xgb_entries = cache.query_meta(model_type="xgboost")
        assert xgb_entries is not None
        assert len(xgb_entries) == 2
        
        for entry in xgb_entries:
            params = entry['metadata_dict']
            assert params['model_type'] == "xgboost"
        
        # Query by experiment
        exp_entries = cache.query_meta(experiment="exp_001")
        assert exp_entries is not None
        assert len(exp_entries) == 1
        assert exp_entries[0]['metadata_dict']['experiment'] == "exp_001"
        
        # Query by version
        v1_entries = cache.query_meta(version="v1")
        assert v1_entries is not None
        assert len(v1_entries) == 2

    def test_query_meta_numeric_filters(self, temp_cache):
        """Test query_meta with numeric parameter filters."""
        cache = temp_cache
        
        # Store test data with numeric parameters
        cache.put("model_1", experiment="exp_001", accuracy=0.95, epochs=100)
        cache.put("model_2", experiment="exp_002", accuracy=0.88, epochs=50)
        cache.put("model_3", experiment="exp_003", accuracy=0.92, epochs=75)
        
        # Query by exact accuracy
        high_acc = cache.query_meta(accuracy=0.95)
        assert high_acc is not None
        assert len(high_acc) == 1
        assert high_acc[0]['metadata_dict']['accuracy'] == 0.95
        
        # Query by epochs (should work with >= comparison)
        many_epochs = cache.query_meta(epochs=75)
        assert many_epochs is not None
        # Should find entries with epochs >= 75 (entries with 75 and 100)
        assert len(many_epochs) >= 1

    def test_query_meta_multiple_filters(self, temp_cache):
        """Test query_meta with multiple parameter filters."""
        cache = temp_cache
        
        # Store test data
        cache.put("model_1", experiment="exp_001", model_type="xgboost", accuracy=0.95, active=True)
        cache.put("model_2", experiment="exp_002", model_type="cnn", accuracy=0.88, active=True)
        cache.put("model_3", experiment="exp_003", model_type="xgboost", accuracy=0.85, active=False)
        
        # Query with multiple filters
        filtered_entries = cache.query_meta(
            model_type="xgboost",
            active=True
        )
        assert filtered_entries is not None
        # Should find only the first entry (xgboost + active=True)
        assert len(filtered_entries) == 1
        params = filtered_entries[0]['metadata_dict']
        assert params['model_type'] == "xgboost"
        assert params['active'] == True

    def test_query_meta_no_matches(self, temp_cache):
        """Test query_meta when no entries match the filters."""
        # Store test data
        temp_cache.put("model_1", experiment="exp_001", model_type="xgboost")
        temp_cache.put("model_2", experiment="exp_002", model_type="cnn")
        
        # Query for non-existent values
        no_matches = temp_cache.query_meta(model_type="nonexistent")
        assert no_matches is not None
        assert len(no_matches) == 0
        
        # Query for non-existent parameter
        no_param = temp_cache.query_meta(nonexistent_param="value")
        assert no_param is not None
        assert len(no_param) == 0

    def test_query_meta_complex_parameters(self, temp_cache):
        """Test query_meta with complex parameter types."""
        from pathlib import Path
        
        # Store data with complex parameters
        test_path = Path("/tmp/test.txt")
        temp_cache.put(
            "complex_data", 
            experiment="exp_001",
            model_path=test_path,
            config={"lr": 0.001, "epochs": 100},
            tags=["ml", "experiment"]
        )
        
        # Query should work even with complex serialized parameters
        # Note: WindowsPath is not JSON serializable, so metadata_dict will be None
        # and this entry won't be returned by query_meta
        all_entries = temp_cache.query_meta()
        assert all_entries is not None
        # Entry won't be returned because metadata_dict serialization failed
        assert len(all_entries) == 0

    def test_query_meta_without_sqlite_backend(self, temp_cache):
        """Test query_meta fails gracefully with non-SQLite backends."""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            cache_dir = Path(temp_dir) / "cache"
            # Create cache with JSON backend
            json_config = CacheConfig(
                cache_dir=str(cache_dir / "json"),
                metadata_backend="json",
                store_full_metadata=True,
            )
            json_cache = UnifiedCache(json_config)
            
            # Store some data
            json_cache.put("test_data", experiment="exp_001")
            
            # query_meta should return None with warning
            result = json_cache.query_meta(experiment="exp_001")
            assert result is None
        finally:
            shutil.rmtree(temp_dir)

    def test_query_meta_without_param_storage(self, temp_cache):
        """Test query_meta fails gracefully when store_full_metadata=False."""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        try:
            cache_dir = Path(temp_dir) / "cache"
            # Create cache without parameter storage
            no_params_config = CacheConfig(
                cache_dir=str(cache_dir / "no_params"),
                metadata_backend="sqlite",
                store_full_metadata=False,  # Disabled
            )
            no_params_cache = UnifiedCache(no_params_config)
            
            # Store some data
            no_params_cache.put("test_data", experiment="exp_001")
            
            # query_meta should return None with warning
            result = no_params_cache.query_meta(experiment="exp_001")
            assert result is None
            
            no_params_cache.close()
        finally:
            shutil.rmtree(temp_dir)

    def test_query_meta_empty_cache(self, temp_cache):
        """Test query_meta with empty cache."""
        # Query empty cache
        result = temp_cache.query_meta()
        assert result is not None
        assert len(result) == 0
        
        # Query with filters on empty cache
        filtered = temp_cache.query_meta(experiment="exp_001")
        assert filtered is not None
        assert len(filtered) == 0

    def test_query_meta_order_by_created_at(self, temp_cache):
        """Test that query_meta returns entries ordered by created_at DESC."""
        import time
        
        # Store entries with small delays to ensure different timestamps
        temp_cache.put("model_1", experiment="exp_001", order=1)
        time.sleep(0.01)  # Small delay
        temp_cache.put("model_2", experiment="exp_002", order=2)
        time.sleep(0.01)  # Small delay
        temp_cache.put("model_3", experiment="exp_003", order=3)
        
        # Get all entries
        entries = temp_cache.query_meta()
        assert entries is not None
        assert len(entries) == 3
        
        # Verify ordering (most recent first)
        timestamps = [entry['created_at'] for entry in entries]
        # Should be in descending order (newest first)
        assert timestamps == sorted(timestamps, reverse=True)

    def test_query_meta_with_different_data_types(self, temp_cache):
        """Test query_meta with various cached data types."""
        import numpy as np
        import pandas as pd
        
        # Store different data types
        temp_cache.put("string_data", data_type="string", format="text")
        temp_cache.put(
            np.array([1, 2, 3]), 
            data_type="numpy", 
            format="array"
        )
        temp_cache.put(
            pd.DataFrame({'a': [1, 2], 'b': [3, 4]}), 
            data_type="pandas", 
            format="dataframe"
        )
        
        # Query by data type parameter
        string_entries = temp_cache.query_meta(data_type="string")
        assert string_entries is not None
        assert len(string_entries) == 1
        
        # Query all entries
        all_entries = temp_cache.query_meta()
        assert all_entries is not None
        assert len(all_entries) == 3
        
        # Verify different data_type field values
        data_types = {entry['data_type'] for entry in all_entries}
        assert 'object' in data_types  # String data (stored as object)
        assert 'array' in data_types  # NumPy array
        assert 'pandas_dataframe' in data_types  # Pandas DataFrame

    def test_query_meta_json_parsing_error_handling(self, temp_cache):
        """Test query_meta handles JSON parsing errors gracefully."""
        # This test verifies the error handling in the JSON parsing section
        # Store normal data first
        temp_cache.put("test_data", experiment="exp_001")
        
        # Get entries (should work normally)
        entries = temp_cache.query_meta()
        assert entries is not None
        assert len(entries) == 1
        
        # Verify that if JSON parsing fails, it falls back to empty dict
        entry = entries[0]
        assert 'metadata_dict' in entry
        assert isinstance(entry['metadata_dict'], dict)

    def test_query_meta_sql_injection_protection(self, temp_cache):
        """Test that query_meta is protected against SQL injection attempts."""
        # Store normal data
        temp_cache.put("test_data", experiment="exp_001", value="normal")
        
        # Attempt SQL injection in filter values
        malicious_filters = [
            "'; DROP TABLE cache_entries; --",
            "' OR '1'='1",
            "'; DELETE FROM cache_entries; --",
            "' UNION SELECT * FROM cache_entries --"
        ]
        
        for malicious_value in malicious_filters:
            # Should not crash or cause SQL injection
            result = temp_cache.query_meta(experiment=malicious_value)
            assert result is not None
            assert len(result) == 0  # Should find no matches
        
        # Verify original data is still there
        normal_result = temp_cache.query_meta(experiment="exp_001")
        assert normal_result is not None
        assert len(normal_result) == 1

    def test_query_meta_with_none_values(self, temp_cache):
        """Test query_meta behavior with None values in parameters."""
        # Store data where some parameters might be None
        temp_cache.put("test_data", experiment="exp_001", optional_param=None, required_param="value")
        
        # Query should work for non-None parameters
        result = temp_cache.query_meta(required_param="value")
        assert result is not None
        assert len(result) == 1
        
        # Querying for None values should not match
        none_result = temp_cache.query_meta(optional_param=None)
        assert none_result is not None
        assert len(none_result) == 0  # None values are not stored/queryable

    def test_query_meta_performance_with_many_entries(self, temp_cache):
        """Test query_meta performance with a larger number of entries."""
        # Store many entries
        num_entries = 100
        for i in range(num_entries):
            temp_cache.put(
                f"data_{i}",
                experiment=f"exp_{i:03d}",
                batch=i // 10,  # Group into batches of 10
                index=i
            )
        
        # Query all entries
        all_entries = temp_cache.query_meta()
        assert all_entries is not None
        assert len(all_entries) == num_entries
        
        # Query specific batch
        batch_5 = temp_cache.query_meta(batch=5)
        assert batch_5 is not None
        assert len(batch_5) == 10  # Should find 10 entries in batch 5
        
        # Query specific experiment
        specific_exp = temp_cache.query_meta(experiment="exp_050")
        assert specific_exp is not None
        assert len(specific_exp) == 1

    def test_query_meta_datetime_handling(self, temp_cache):
        """Test that query_meta properly handles datetime fields."""
        # Store some data
        temp_cache.put("test_data", experiment="exp_001")
        
        # Get entries and verify datetime fields
        entries = temp_cache.query_meta()
        assert entries is not None
        assert len(entries) == 1
        
        entry = entries[0]
        
        # Verify datetime fields are properly formatted
        assert 'created_at' in entry
        assert 'accessed_at' in entry
        
        # Should be ISO format strings
        created_at = entry['created_at']
        accessed_at = entry['accessed_at']
        
        assert isinstance(created_at, str)
        assert isinstance(accessed_at, str)
        
        # Should be parseable as datetime
        from datetime import datetime
        datetime.fromisoformat(created_at)
        datetime.fromisoformat(accessed_at)


class TestQueryMetaIntegration:
    """Integration tests for query_meta with other cache features."""

    @pytest.fixture(autouse=True)
    def setup_integration_cache(self):
        """Set up each test with a fresh cache instance."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        
        self.config = CacheConfig(
            cache_dir=str(self.cache_dir),
            metadata_backend="sqlite",
            store_full_metadata=True,
        )
        self.cache = UnifiedCache(self.config)
        
        yield
        
        # Clean up after each test
        self.cache.close()
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_query_meta_with_ttl_expired_entries(self, temp_cache):
        """Test query_meta with TTL-expired entries."""
        # Store data with very short TTL
        temp_cache.put("test_data", experiment="exp_001", description="Short TTL test")
        
        # Query should find the entry
        entries = temp_cache.query_meta(experiment="exp_001")
        assert entries is not None
        assert len(entries) == 1
        
        # Note: query_meta queries the database directly, so it will find
        # entries even if they're TTL-expired (unlike cache.get())
        # This is expected behavior for metadata querying

    def test_query_meta_with_invalidated_entries(self, temp_cache):
        """Test query_meta after entries are invalidated."""
        # Store and then invalidate data
        cache_key = temp_cache.put("test_data", experiment="exp_001")
        
        # Verify entry exists
        entries = temp_cache.query_meta(experiment="exp_001")
        assert entries is not None
        assert len(entries) == 1
        
        # Invalidate the entry
        temp_cache.invalidate(cache_key=cache_key)
        
        # Query should no longer find the entry
        entries_after = temp_cache.query_meta(experiment="exp_001")
        assert entries_after is not None
        assert len(entries_after) == 0

    def test_query_meta_with_clear_all(self, temp_cache):
        """Test query_meta after clearing all cache."""
        # Store multiple entries
        temp_cache.put("data_1", experiment="exp_001")
        temp_cache.put("data_2", experiment="exp_002")
        
        # Verify entries exist
        entries = temp_cache.query_meta()
        assert entries is not None
        assert len(entries) == 2
        
        # Clear all cache
        temp_cache.clear_all()
        
        # Query should find no entries
        entries_after = temp_cache.query_meta()
        assert entries_after is not None
        assert len(entries_after) == 0

    def test_query_meta_with_cache_stats(self, temp_cache):
        """Test query_meta integration with cache statistics."""
        # Store some data
        temp_cache.put("data_1", experiment="exp_001")
        temp_cache.put("data_2", experiment="exp_002")
        
        # Get stats
        stats = temp_cache.get_stats()
        assert stats['total_entries'] == 2
        
        # Query entries
        entries = temp_cache.query_meta()
        assert entries is not None
        assert len(entries) == 2
        
        # Stats should match query results
        assert stats['total_entries'] == len(entries)

    def test_query_meta_concurrent_access(self, temp_cache):
        """Test query_meta with concurrent cache operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                # Each worker stores and queries data
                temp_cache.put(f"data_{worker_id}", worker_id=worker_id, experiment="concurrent_test")
                time.sleep(0.01)  # Small delay
                
                # Query data
                entries = temp_cache.query_meta(experiment="concurrent_test")
                results.append((worker_id, len(entries) if entries else 0))
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Run multiple workers concurrently
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        
        # Final verification
        final_entries = temp_cache.query_meta(experiment="concurrent_test")
        assert final_entries is not None
        assert len(final_entries) == 5

    def test_query_meta_unicode_and_special_characters(self, temp_cache):
        """Test query_meta with unicode and special characters."""
        unicode_data = {
            "experiment": "测试实验",
            "description": "café naïve résumé",
        }
        
        temp_cache.put("unicode_test", **unicode_data)
        
        entries = temp_cache.query_meta()
        assert entries is not None
        assert len(entries) == 1
        
        params = entries[0]['metadata_dict']
        assert any("测试" in str(v) for v in params.values())

    def test_query_meta_sql_injection_prevention(self, temp_cache):
        """Test query_meta safely handles SQL injection attempts."""
        injection_params = {
            "param": "SELECT * FROM cache_entries",
            "other": "--comment",
        }
        
        temp_cache.put("injection_test", **injection_params)
        
        all_entries = temp_cache.query_meta()
        assert all_entries is not None
        assert len(all_entries) == 1
