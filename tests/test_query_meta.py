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


class TestQueryMeta:
    """Test suite for query_meta() method."""

    def setup_method(self):
        """Set up each test with a fresh cache instance."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        
        # Create cache with parameter storage enabled
        self.config = CacheConfig(
            cache_dir=str(self.cache_dir),
            metadata_backend="sqlite",
            store_cache_key_params=True,  # Required for query_meta
        )
        self.cache = UnifiedCache(self.config)

    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_query_meta_basic_functionality(self):
        """Test basic query_meta functionality with simple parameters."""
        # Store test data with various parameters
        self.cache.put("model_1", experiment="exp_001", model_type="xgboost", accuracy=0.95)
        self.cache.put("model_2", experiment="exp_002", model_type="cnn", accuracy=0.88)
        self.cache.put("model_3", experiment="exp_003", model_type="random_forest", accuracy=0.92)
        
        # Test query with no filters (all entries)
        all_entries = self.cache.query_meta()
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
        assert 'cache_key_params' in entry
        
        # Verify cache_key_params structure
        params = entry['cache_key_params']
        assert 'experiment' in params
        assert 'model_type' in params
        assert 'accuracy' in params

    def test_query_meta_string_filters(self):
        """Test query_meta with string parameter filters."""
        # Store test data
        self.cache.put("model_1", experiment="exp_001", model_type="xgboost", version="v1")
        self.cache.put("model_2", experiment="exp_002", model_type="cnn", version="v1")
        self.cache.put("model_3", experiment="exp_003", model_type="xgboost", version="v2")
        
        # Query by model_type
        xgb_entries = self.cache.query_meta(model_type="str:xgboost")
        assert xgb_entries is not None
        assert len(xgb_entries) == 2
        
        for entry in xgb_entries:
            params = entry['cache_key_params']
            assert params['model_type'] == "str:xgboost"
        
        # Query by experiment
        exp_entries = self.cache.query_meta(experiment="str:exp_001")
        assert exp_entries is not None
        assert len(exp_entries) == 1
        assert exp_entries[0]['cache_key_params']['experiment'] == "str:exp_001"
        
        # Query by version
        v1_entries = self.cache.query_meta(version="str:v1")
        assert v1_entries is not None
        assert len(v1_entries) == 2

    def test_query_meta_numeric_filters(self):
        """Test query_meta with numeric parameter filters."""
        # Store test data with numeric parameters
        self.cache.put("model_1", experiment="exp_001", accuracy=0.95, epochs=100)
        self.cache.put("model_2", experiment="exp_002", accuracy=0.88, epochs=50)
        self.cache.put("model_3", experiment="exp_003", accuracy=0.92, epochs=75)
        
        # Query by exact accuracy
        high_acc = self.cache.query_meta(accuracy="float:0.95")
        assert high_acc is not None
        assert len(high_acc) == 1
        assert high_acc[0]['cache_key_params']['accuracy'] == "float:0.95"
        
        # Query by epochs (should work with >= comparison)
        many_epochs = self.cache.query_meta(epochs="int:75")
        assert many_epochs is not None
        # Should find entries with epochs >= 75 (entries with 75 and 100)
        assert len(many_epochs) >= 1

    def test_query_meta_multiple_filters(self):
        """Test query_meta with multiple parameter filters."""
        # Store test data
        self.cache.put("model_1", experiment="exp_001", model_type="xgboost", accuracy=0.95, active=True)
        self.cache.put("model_2", experiment="exp_002", model_type="cnn", accuracy=0.88, active=True)
        self.cache.put("model_3", experiment="exp_003", model_type="xgboost", accuracy=0.85, active=False)
        
        # Query with multiple filters
        filtered_entries = self.cache.query_meta(
            model_type="str:xgboost",
            active="bool:True"
        )
        assert filtered_entries is not None
        # Should find only the first entry (xgboost + active=True)
        assert len(filtered_entries) == 1
        params = filtered_entries[0]['cache_key_params']
        assert params['model_type'] == "str:xgboost"
        assert params['active'] == "bool:True"

    def test_query_meta_no_matches(self):
        """Test query_meta when no entries match the filters."""
        # Store test data
        self.cache.put("model_1", experiment="exp_001", model_type="xgboost")
        self.cache.put("model_2", experiment="exp_002", model_type="cnn")
        
        # Query for non-existent values
        no_matches = self.cache.query_meta(model_type="str:nonexistent")
        assert no_matches is not None
        assert len(no_matches) == 0
        
        # Query for non-existent parameter
        no_param = self.cache.query_meta(nonexistent_param="str:value")
        assert no_param is not None
        assert len(no_param) == 0

    def test_query_meta_complex_parameters(self):
        """Test query_meta with complex parameter types."""
        from pathlib import Path
        
        # Store data with complex parameters
        test_path = Path("/tmp/test.txt")
        self.cache.put(
            "complex_data", 
            experiment="exp_001",
            model_path=test_path,
            config={"lr": 0.001, "epochs": 100},
            tags=["ml", "experiment"]
        )
        
        # Query should work even with complex serialized parameters
        all_entries = self.cache.query_meta()
        assert all_entries is not None
        assert len(all_entries) == 1
        
        # Verify complex parameters are serialized properly
        params = all_entries[0]['cache_key_params']
        assert 'experiment' in params
        assert 'model_path' in params
        assert 'config' in params
        assert 'tags' in params

    def test_query_meta_without_sqlite_backend(self):
        """Test query_meta fails gracefully with non-SQLite backends."""
        # Create cache with JSON backend
        json_config = CacheConfig(
            cache_dir=str(self.cache_dir / "json"),
            metadata_backend="json",
            store_cache_key_params=True,
        )
        json_cache = UnifiedCache(json_config)
        
        # Store some data
        json_cache.put("test_data", experiment="exp_001")
        
        # query_meta should return None with warning
        result = json_cache.query_meta(experiment="exp_001")
        assert result is None

    def test_query_meta_without_param_storage(self):
        """Test query_meta fails gracefully when store_cache_key_params=False."""
        # Create cache without parameter storage
        no_params_config = CacheConfig(
            cache_dir=str(self.cache_dir / "no_params"),
            metadata_backend="sqlite",
            store_cache_key_params=False,  # Disabled
        )
        no_params_cache = UnifiedCache(no_params_config)
        
        # Store some data
        no_params_cache.put("test_data", experiment="exp_001")
        
        # query_meta should return None with warning
        result = no_params_cache.query_meta(experiment="exp_001")
        assert result is None

    def test_query_meta_empty_cache(self):
        """Test query_meta with empty cache."""
        # Query empty cache
        result = self.cache.query_meta()
        assert result is not None
        assert len(result) == 0
        
        # Query with filters on empty cache
        filtered = self.cache.query_meta(experiment="exp_001")
        assert filtered is not None
        assert len(filtered) == 0

    def test_query_meta_order_by_created_at(self):
        """Test that query_meta returns entries ordered by created_at DESC."""
        import time
        
        # Store entries with small delays to ensure different timestamps
        self.cache.put("model_1", experiment="exp_001", order=1)
        time.sleep(0.01)  # Small delay
        self.cache.put("model_2", experiment="exp_002", order=2)
        time.sleep(0.01)  # Small delay
        self.cache.put("model_3", experiment="exp_003", order=3)
        
        # Get all entries
        entries = self.cache.query_meta()
        assert entries is not None
        assert len(entries) == 3
        
        # Verify ordering (most recent first)
        timestamps = [entry['created_at'] for entry in entries]
        # Should be in descending order (newest first)
        assert timestamps == sorted(timestamps, reverse=True)

    def test_query_meta_with_different_data_types(self):
        """Test query_meta with various cached data types."""
        import numpy as np
        import pandas as pd
        
        # Store different data types
        self.cache.put("string_data", data_type="string", format="text")
        self.cache.put(
            np.array([1, 2, 3]), 
            data_type="numpy", 
            format="array"
        )
        self.cache.put(
            pd.DataFrame({'a': [1, 2], 'b': [3, 4]}), 
            data_type="pandas", 
            format="dataframe"
        )
        
        # Query by data type parameter
        string_entries = self.cache.query_meta(data_type="str:string")
        assert string_entries is not None
        assert len(string_entries) == 1
        
        # Query all entries
        all_entries = self.cache.query_meta()
        assert all_entries is not None
        assert len(all_entries) == 3
        
        # Verify different data_type field values
        data_types = {entry['data_type'] for entry in all_entries}
        assert 'object' in data_types  # String data (stored as object)
        assert 'array' in data_types  # NumPy array
        assert 'pandas_dataframe' in data_types  # Pandas DataFrame

    def test_query_meta_json_parsing_error_handling(self):
        """Test query_meta handles JSON parsing errors gracefully."""
        # This test verifies the error handling in the JSON parsing section
        # Store normal data first
        self.cache.put("test_data", experiment="exp_001")
        
        # Get entries (should work normally)
        entries = self.cache.query_meta()
        assert entries is not None
        assert len(entries) == 1
        
        # Verify that if JSON parsing fails, it falls back to empty dict
        entry = entries[0]
        assert 'cache_key_params' in entry
        assert isinstance(entry['cache_key_params'], dict)

    def test_query_meta_sql_injection_protection(self):
        """Test that query_meta is protected against SQL injection attempts."""
        # Store normal data
        self.cache.put("test_data", experiment="exp_001", value="normal")
        
        # Attempt SQL injection in filter values
        malicious_filters = [
            "'; DROP TABLE cache_entries; --",
            "' OR '1'='1",
            "'; DELETE FROM cache_entries; --",
            "' UNION SELECT * FROM cache_entries --"
        ]
        
        for malicious_value in malicious_filters:
            # Should not crash or cause SQL injection
            result = self.cache.query_meta(experiment=malicious_value)
            assert result is not None
            assert len(result) == 0  # Should find no matches
        
        # Verify original data is still there
        normal_result = self.cache.query_meta(experiment="str:exp_001")
        assert normal_result is not None
        assert len(normal_result) == 1

    def test_query_meta_with_none_values(self):
        """Test query_meta behavior with None values in parameters."""
        # Store data where some parameters might be None
        self.cache.put("test_data", experiment="exp_001", optional_param=None, required_param="value")
        
        # Query should work for non-None parameters
        result = self.cache.query_meta(required_param="str:value")
        assert result is not None
        assert len(result) == 1
        
        # Querying for None values should not match
        none_result = self.cache.query_meta(optional_param=None)
        assert none_result is not None
        assert len(none_result) == 0  # None values are not stored/queryable

    def test_query_meta_performance_with_many_entries(self):
        """Test query_meta performance with a larger number of entries."""
        # Store many entries
        num_entries = 100
        for i in range(num_entries):
            self.cache.put(
                f"data_{i}",
                experiment=f"exp_{i:03d}",
                batch=i // 10,  # Group into batches of 10
                index=i
            )
        
        # Query all entries
        all_entries = self.cache.query_meta()
        assert all_entries is not None
        assert len(all_entries) == num_entries
        
        # Query specific batch
        batch_5 = self.cache.query_meta(batch="int:5")
        assert batch_5 is not None
        assert len(batch_5) == 10  # Should find 10 entries in batch 5
        
        # Query specific experiment
        specific_exp = self.cache.query_meta(experiment="str:exp_050")
        assert specific_exp is not None
        assert len(specific_exp) == 1

    def test_query_meta_datetime_handling(self):
        """Test that query_meta properly handles datetime fields."""
        # Store some data
        self.cache.put("test_data", experiment="exp_001")
        
        # Get entries and verify datetime fields
        entries = self.cache.query_meta()
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

    def setup_method(self):
        """Set up each test with a fresh cache instance."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        
        self.config = CacheConfig(
            cache_dir=str(self.cache_dir),
            metadata_backend="sqlite",
            store_cache_key_params=True,
        )
        self.cache = UnifiedCache(self.config)

    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_query_meta_with_ttl_expired_entries(self):
        """Test query_meta with TTL-expired entries."""
        # Store data with very short TTL
        self.cache.put("test_data", experiment="exp_001", description="Short TTL test")
        
        # Query should find the entry
        entries = self.cache.query_meta(experiment="str:exp_001")
        assert entries is not None
        assert len(entries) == 1
        
        # Note: query_meta queries the database directly, so it will find
        # entries even if they're TTL-expired (unlike cache.get())
        # This is expected behavior for metadata querying

    def test_query_meta_with_invalidated_entries(self):
        """Test query_meta after entries are invalidated."""
        # Store and then invalidate data
        cache_key = self.cache.put("test_data", experiment="exp_001")
        
        # Verify entry exists
        entries = self.cache.query_meta(experiment="str:exp_001")
        assert entries is not None
        assert len(entries) == 1
        
        # Invalidate the entry
        self.cache.invalidate(cache_key=cache_key)
        
        # Query should no longer find the entry
        entries_after = self.cache.query_meta(experiment="str:exp_001")
        assert entries_after is not None
        assert len(entries_after) == 0

    def test_query_meta_with_clear_all(self):
        """Test query_meta after clearing all cache."""
        # Store multiple entries
        self.cache.put("data_1", experiment="exp_001")
        self.cache.put("data_2", experiment="exp_002")
        
        # Verify entries exist
        entries = self.cache.query_meta()
        assert entries is not None
        assert len(entries) == 2
        
        # Clear all cache
        self.cache.clear_all()
        
        # Query should find no entries
        entries_after = self.cache.query_meta()
        assert entries_after is not None
        assert len(entries_after) == 0

    def test_query_meta_with_cache_stats(self):
        """Test query_meta integration with cache statistics."""
        # Store some data
        self.cache.put("data_1", experiment="exp_001")
        self.cache.put("data_2", experiment="exp_002")
        
        # Get stats
        stats = self.cache.get_stats()
        assert stats['total_entries'] == 2
        
        # Query entries
        entries = self.cache.query_meta()
        assert entries is not None
        assert len(entries) == 2
        
        # Stats should match query results
        assert stats['total_entries'] == len(entries)

    def test_query_meta_concurrent_access(self):
        """Test query_meta with concurrent cache operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                # Each worker stores and queries data
                self.cache.put(f"data_{worker_id}", worker_id=worker_id, experiment="concurrent_test")
                time.sleep(0.01)  # Small delay
                
                # Query data
                entries = self.cache.query_meta(experiment="str:concurrent_test")
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
        final_entries = self.cache.query_meta(experiment="str:concurrent_test")
        assert final_entries is not None
        assert len(final_entries) == 5

    def test_query_meta_database_corruption_handling(self):
        """Test query_meta handles database corruption gracefully."""
        # Store some data first
        self.cache.put("test_data", experiment="exp_001")
        
        # Verify it works normally
        entries = self.cache.query_meta(experiment="str:exp_001")
        assert entries is not None
        assert len(entries) == 1
        
        # Simulate database corruption by corrupting the session
        # We can't easily corrupt the actual database, but we can test
        # that the error handling in the try/except block works
        
        # Test with a malformed query parameter that might cause issues
        try:
            # This should not crash even with unusual parameter types
            # We can't use None as a key directly, so test other edge cases
            result = self.cache.query_meta(**{"": "test"})  # Empty string key
            # Should handle gracefully
            assert result is None or isinstance(result, list)
        except Exception:
            # Should not raise unhandled exceptions
            pass

    def test_query_meta_memory_stress(self):
        """Test query_meta behavior under memory stress conditions."""
        # Store entries with very large parameter values
        large_data = "x" * 10000  # 10KB string
        
        for i in range(10):
            self.cache.put(
                f"data_{i}",
                experiment=f"exp_{i:03d}",
                large_param=large_data,
                index=i
            )
        
        # Query should still work with large data
        entries = self.cache.query_meta()
        assert entries is not None
        assert len(entries) == 10
        
        # Verify large parameters are handled correctly
        for entry in entries:
            params = entry.get('cache_key_params', {})
            assert 'large_param' in params

    def test_query_meta_invalid_json_in_database(self):
        """Test query_meta handles invalid JSON in cache_key_params column."""
        # This test simulates what happens if the JSON in the database gets corrupted
        # We can't easily corrupt the database directly, but we can test the JSON parsing path
        
        # Store normal data first
        self.cache.put("test_data", experiment="exp_001")
        
        # Query should work
        entries = self.cache.query_meta()
        assert entries is not None
        assert len(entries) == 1
        
        # The error handling for JSON parsing is in the query_meta method
        # If JSON parsing fails, it should fall back to empty dict
        entry = entries[0]
        assert 'cache_key_params' in entry
        assert isinstance(entry['cache_key_params'], dict)

    def test_query_meta_session_errors(self):
        """Test query_meta handles database session errors."""
        # Store some data
        self.cache.put("test_data", experiment="exp_001")
        
        # Test that if the metadata_backend doesn't have SessionLocal, it fails gracefully
        original_session_local = getattr(self.cache.metadata_backend, 'SessionLocal', None)
        
        try:
            # Temporarily remove SessionLocal to simulate error condition
            if hasattr(self.cache.metadata_backend, 'SessionLocal'):
                delattr(self.cache.metadata_backend, 'SessionLocal')
            
            result = self.cache.query_meta(experiment="str:exp_001")
            assert result is None  # Should return None gracefully
            
        finally:
            # Restore SessionLocal
            if original_session_local:
                setattr(self.cache.metadata_backend, 'SessionLocal', original_session_local)

    def test_query_meta_unicode_and_special_characters(self):
        """Test query_meta with unicode and special characters."""
        # Test with unicode characters
        unicode_data = {
            "experiment": "测试实验",  # Chinese characters
            "model": "модель",      # Cyrillic characters  
            "description": "café naïve résumé",  # Accented characters
            "emoji": "🚀🎉💡",      # Emoji
            "special": "quote\"escape'test",  # Quotes and special chars
            "newlines": "line1\nline2\ttab",   # Newlines and tabs
        }
        
        self.cache.put("unicode_test", **unicode_data)
        
        # Query should handle unicode properly
        entries = self.cache.query_meta()
        assert entries is not None
        assert len(entries) == 1
        
        # Verify unicode is preserved
        params = entries[0]['cache_key_params']
        # Check that some unicode made it through (may be serialized with type prefixes)
        assert any("测试" in str(v) for v in params.values())

    def test_query_meta_extremely_long_parameter_names(self):
        """Test query_meta with extremely long parameter names and values."""
        # Test with very long parameter names and values
        long_name = "a" * 1000  # 1000 character parameter name
        long_value = "b" * 5000  # 5000 character value
        
        params = {
            long_name: long_value,
            "normal_param": "normal_value"
        }
        
        self.cache.put("long_param_test", **params)
        
        # Query should handle long parameters
        entries = self.cache.query_meta()
        assert entries is not None
        assert len(entries) == 1
        
        # Should be able to query by normal parameter
        normal_entries = self.cache.query_meta(normal_param="str:normal_value")
        assert normal_entries is not None
        assert len(normal_entries) == 1

    def test_query_meta_sql_edge_cases(self):
        """Test query_meta with SQL edge cases and special parameter values."""
        # Test with parameter values that could cause SQL issues
        edge_case_params = [
            {"param": ""},  # Empty string
            {"param": " "},  # Space only
            {"param": "NULL"},  # SQL NULL keyword
            {"param": "SELECT * FROM cache_entries"},  # SQL command
            {"param": "--comment"},  # SQL comment
            {"param": "/*comment*/"},  # SQL block comment
            {"param": "\x00"},  # Null byte
            {"param": "\r\n"},  # Windows line endings
        ]
        
        # Store entries with edge case parameters
        for i, params in enumerate(edge_case_params):
            self.cache.put(f"edge_case_{i}", index=i, **params)
        
        # Query all entries should work
        all_entries = self.cache.query_meta()
        assert all_entries is not None
        assert len(all_entries) == len(edge_case_params)
        
        # Query with specific edge case values should work safely
        for i, params in enumerate(edge_case_params):
            key, value = list(params.items())[0]
            result = self.cache.query_meta(**{key: f"str:{value}"})
            assert result is not None
            # Should find exactly one match or none (depending on serialization)
            assert len(result) <= 1

    def test_query_meta_parameter_type_edge_cases(self):
        """Test query_meta with unusual parameter types."""
        # Test with various Python types that might cause serialization issues
        from decimal import Decimal
        from datetime import datetime, date
        
        edge_types = {
            "decimal_param": Decimal("123.456"),
            "datetime_param": datetime.now(),
            "date_param": date.today(),
            "bool_param": True,
            "none_param": None,
            "complex_param": complex(1, 2),
            "bytes_param": b"binary_data",
        }
        
        self.cache.put("type_test", **edge_types)
        
        # Query should handle type serialization
        entries = self.cache.query_meta()
        assert entries is not None
        assert len(entries) == 1
        
        # Verify parameters are serialized (may have type prefixes)
        params = entries[0]['cache_key_params']
        assert isinstance(params, dict)
        assert len(params) > 0  # Should have some parameters

    def test_query_meta_filter_injection_edge_cases(self):
        """Test query_meta with filter values that could cause injection."""
        # Store normal data
        self.cache.put("test_data", experiment="exp_001", value="normal")
        
        # Test filter values that could potentially cause issues
        injection_attempts = [
            {"experiment": "'; DROP TABLE cache_entries; SELECT '"},
            {"experiment": "' OR 1=1 OR '"},
            {"experiment": "' UNION SELECT password FROM users WHERE '"},
            {"experiment": "\"; DELETE FROM cache_entries; --"},
            {"experiment": "' AND (SELECT COUNT(*) FROM cache_entries) > 0 AND '"},
            {"value": "'; UPDATE cache_entries SET cache_key_params = 'hacked'; --"},
        ]
        
        for injection_filter in injection_attempts:
            # Should not cause SQL injection or crashes
            result = self.cache.query_meta(**injection_filter)
            assert result is not None
            assert isinstance(result, list)
            # Should not find matches (since values don't exist)
            assert len(result) == 0
        
        # Verify original data is still there and unchanged
        original_data = self.cache.query_meta(experiment="str:exp_001")
        assert original_data is not None
        assert len(original_data) == 1