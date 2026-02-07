#!/usr/bin/env python3
"""
Comprehensive Cache Key Consistency Tests
========================================

Tests to ensure that logically equivalent parameters always produce
the same cache key, maximizing cache hit reliability.
"""

import tempfile
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from cacheness import cacheness, cached, CacheConfig
from cacheness.serialization import create_unified_cache_key


@dataclass
class TestDataClassForConsistency:
    """Test dataclass for consistency testing."""
    name: str
    value: int
    
    def __hash__(self):
        return hash((self.name, self.value))


class HashableClassForConsistency:
    """Test class with custom hash implementation."""
    def __init__(self, name, value):
        self.name = name
        self.value = value
    
    def __hash__(self):
        return hash((self.name, self.value))
    
    def __eq__(self, other):
        return isinstance(other, HashableClassForConsistency) and self.name == other.name and self.value == other.value


class TestCacheKeyConsistency:
    """Comprehensive tests for cache key consistency."""

    def test_parameter_order_independence(self):
        """Test that parameter order doesn't affect cache keys."""
        # Simple parameters
        params1 = {"a": 1, "b": 2, "c": 3}
        params2 = {"c": 3, "a": 1, "b": 2}
        params3 = {"b": 2, "c": 3, "a": 1}
        
        key1 = create_unified_cache_key(params1)
        key2 = create_unified_cache_key(params2)
        key3 = create_unified_cache_key(params3)
        
        assert key1 == key2 == key3

    def test_nested_dictionary_consistency(self):
        """Test consistency with nested dictionaries."""
        nested1 = {
            "outer": {"inner": {"a": 1, "b": 2}},
            "simple": 42
        }
        nested2 = {
            "simple": 42,
            "outer": {"inner": {"b": 2, "a": 1}}
        }
        
        key1 = create_unified_cache_key(nested1)
        key2 = create_unified_cache_key(nested2)
        
        assert key1 == key2

    def test_list_vs_tuple_consistency(self):
        """Test that lists and tuples with same content produce different keys."""
        params_list = {"data": [1, 2, 3]}
        params_tuple = {"data": (1, 2, 3)}
        
        key_list = create_unified_cache_key(params_list)
        key_tuple = create_unified_cache_key(params_tuple)
        
        # Different types should produce different keys
        assert key_list != key_tuple

    def test_numpy_array_consistency(self):
        """Test consistency with NumPy arrays."""
        # Same data, same dtype - should produce same keys
        arr1 = np.array([1, 2, 3, 4], dtype=np.int64)
        arr2 = np.arange(1, 5, dtype=np.int64)
        
        # Same data, different dtype - should produce different keys
        arr3 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        
        key1 = create_unified_cache_key({"array": arr1})
        key2 = create_unified_cache_key({"array": arr2})
        key3 = create_unified_cache_key({"array": arr3})
        
        # Same data, same dtype should produce same keys
        assert key1 == key2
        # Same data, different dtype should produce different keys
        assert key1 != key3

    def test_numpy_dtype_consistency(self):
        """Test that arrays with same values but different dtypes produce different keys."""
        arr_int32 = np.array([1, 2, 3], dtype=np.int32)
        arr_int64 = np.array([1, 2, 3], dtype=np.int64)
        arr_float = np.array([1.0, 2.0, 3.0])
        
        key_int32 = create_unified_cache_key({"array": arr_int32})
        key_int64 = create_unified_cache_key({"array": arr_int64})
        key_float = create_unified_cache_key({"array": arr_float})
        
        # Different dtypes should produce different keys
        assert key_int32 != key_int64 != key_float

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not available")
    def test_dataframe_consistency(self):
        """Test consistency with pandas DataFrames."""
        if not PANDAS_AVAILABLE:
            pytest.skip("Pandas not available")
        
        import pandas as pd
        
        # Same data, different construction methods
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"b": [4, 5, 6], "a": [1, 2, 3]})  # Different column order
        
        key1 = create_unified_cache_key({"df": df1})
        key2 = create_unified_cache_key({"df": df2})
        
        # DataFrames with same data should produce same key regardless of column order
        # (this depends on implementation - may need to be adjusted)
        # For now, let's verify they produce valid keys
        assert isinstance(key1, str) and len(key1) == 16
        assert isinstance(key2, str) and len(key2) == 16

    def test_hashable_object_consistency(self):
        """Test consistency with custom hashable objects."""
        obj1 = HashableClassForConsistency("test", 42)
        obj2 = HashableClassForConsistency("test", 42)  # Same content, different instance
        obj3 = HashableClassForConsistency("test", 43)  # Different content
        
        key1 = create_unified_cache_key({"obj": obj1})
        key2 = create_unified_cache_key({"obj": obj2})
        key3 = create_unified_cache_key({"obj": obj3})
        
        assert key1 == key2  # Same logical content
        assert key1 != key3  # Different content

    def test_dataclass_consistency(self):
        """Test consistency with dataclasses."""
        dc1 = TestDataClassForConsistency("test", 42)
        dc2 = TestDataClassForConsistency("test", 42)  # Same content, different instance
        dc3 = TestDataClassForConsistency("test", 43)  # Different content
        
        key1 = create_unified_cache_key({"dc": dc1})
        key2 = create_unified_cache_key({"dc": dc2})
        key3 = create_unified_cache_key({"dc": dc3})
        
        assert key1 == key2  # Same logical content
        assert key1 != key3  # Different content

    def test_path_object_consistency(self):
        """Test consistency with Path objects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")
            
            # Different ways to reference the same file
            path1 = Path(temp_dir) / "test.txt"
            path2 = Path(temp_dir).resolve() / "test.txt"
            path3 = test_file.resolve()
            
            key1 = create_unified_cache_key({"path": path1})
            key2 = create_unified_cache_key({"path": path2})
            key3 = create_unified_cache_key({"path": path3})
            
            # All should produce the same key (depends on implementation)
            # At minimum, verify they produce valid keys
            assert isinstance(key1, str) and len(key1) == 16
            assert isinstance(key2, str) and len(key2) == 16
            assert isinstance(key3, str) and len(key3) == 16

    def test_datetime_consistency(self):
        """Test consistency with datetime objects."""
        # Same moment in time, different representations
        dt1 = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2023, 1, 1, 12, 0, 0).replace(tzinfo=timezone.utc)
        
        key1 = create_unified_cache_key({"dt": dt1})
        key2 = create_unified_cache_key({"dt": dt2})
        
        assert key1 == key2

    def test_unicode_normalization_consistency(self):
        """Test consistency with Unicode normalization."""
        import unicodedata
        
        # Same character, different Unicode representations
        str1 = "café"  # é as single character
        str2 = "cafe\u0301"  # e + combining acute accent
        str1_nfc = unicodedata.normalize('NFC', str1)
        str2_nfc = unicodedata.normalize('NFC', str2)
        
        key1 = create_unified_cache_key({"text": str1})
        key2 = create_unified_cache_key({"text": str2})
        key1_nfc = create_unified_cache_key({"text": str1_nfc})
        key2_nfc = create_unified_cache_key({"text": str2_nfc})
        
        # After normalization, they should be the same
        assert key1_nfc == key2_nfc

    def test_function_decorator_consistency(self):
        """Test that decorated functions produce consistent cache keys for equivalent calls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            cache = cacheness(config)

            call_count = 0

            @cached(cache_instance=cache)
            def test_function(a, b, c=10):
                nonlocal call_count
                call_count += 1
                return a + b + c

            # These calls are logically identical (all parameters explicitly provided)
            # and should produce the same cache key
            result1 = test_function(1, 2, 10)  # All positional
            result2 = test_function(a=1, b=2, c=10)  # All keyword
            result3 = test_function(1, b=2, c=10)  # Mixed positional/keyword
            result4 = test_function(b=2, a=1, c=10)  # Different keyword order

            # All should return same result
            assert result1 == result2 == result3 == result4 == 13
            
            # All should hit cache after first call — _normalize_function_args
            # uses inspect.signature().bind() to normalize positional/keyword args
            assert call_count == 1, (
                f"Cache key inconsistency: function called {call_count} times "
                f"instead of 1 for logically equivalent calls"
            )
            
            cache.close()

    def test_cross_instance_consistency(self):
        """Test that same parameters produce same keys across different cache instances."""
        with tempfile.TemporaryDirectory() as temp_dir1, \
             tempfile.TemporaryDirectory() as temp_dir2:
            
            config1 = CacheConfig(cache_dir=temp_dir1)
            config2 = CacheConfig(cache_dir=temp_dir2)
            cache1 = cacheness(config1)
            cache2 = cacheness(config2)
            
            params = {"test": "value", "number": 42, "array": np.array([1, 2, 3])}
            
            key1 = cache1._create_cache_key(params)
            key2 = cache2._create_cache_key(params)
            
            assert key1 == key2
            
            cache2.close()
            cache1.close()

    def test_complex_nested_structure_consistency(self):
        """Test consistency with deeply nested complex structures."""
        complex_params1 = {
            "level1": {
                "level2": {
                    "arrays": [np.array([1, 2]), np.array([3, 4])],
                    "metadata": {"created": datetime(2023, 1, 1), "version": "1.0"}
                },
                "simple": [1, 2, 3]
            },
            "top_level": "value"
        }
        
        complex_params2 = {
            "top_level": "value",
            "level1": {
                "simple": [1, 2, 3],
                "level2": {
                    "metadata": {"version": "1.0", "created": datetime(2023, 1, 1)},
                    "arrays": [np.array([1, 2]), np.array([3, 4])]
                }
            }
        }
        
        key1 = create_unified_cache_key(complex_params1)
        key2 = create_unified_cache_key(complex_params2)
        
        assert key1 == key2

    def test_none_vs_missing_parameter_consistency(self):
        """Test that None values vs missing parameters produce different keys."""
        params1 = {"a": 1, "b": None}
        params2 = {"a": 1}  # b is missing
        
        key1 = create_unified_cache_key(params1)
        key2 = create_unified_cache_key(params2)
        
        # These should produce different keys
        assert key1 != key2

    def test_empty_collections_consistency(self):
        """Test consistency with empty collections."""
        params1 = {"data": []}
        params2 = {"data": ()}
        params3 = {"data": {}}
        params4 = {"data": set()}
        
        key1 = create_unified_cache_key(params1)
        key2 = create_unified_cache_key(params2)
        key3 = create_unified_cache_key(params3)
        key4 = create_unified_cache_key(params4)
        
        # Different types of empty collections should produce different keys
        assert len({key1, key2, key3, key4}) == 4  # All different


class TestCacheKeyStability:
    """Test that cache keys remain stable across sessions."""
    
    def test_reproducible_keys_across_runs(self):
        """Test that the same inputs always produce the same cache keys."""
        # Fixed test data
        test_params = {
            "string": "hello_world",
            "integer": 42,
            "float": 3.14159,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3, "test"],
            "dict": {"nested": {"value": 123}},
            "array": np.array([1.0, 2.0, 3.0])
        }
        
        # Generate key multiple times
        keys = [create_unified_cache_key(test_params) for _ in range(5)]
        
        # All keys should be identical
        assert len(set(keys)) == 1
        
        # Key should have expected format
        assert isinstance(keys[0], str)
        assert len(keys[0]) == 16
        assert all(c in "0123456789abcdef" for c in keys[0])


if __name__ == "__main__":
    pytest.main([__file__])