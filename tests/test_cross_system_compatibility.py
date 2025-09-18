"""
Cross-System Cache Compatibility Tests
=====================================

Tests to ensure that UnifiedCache and @cached decorators are fully interoperable.
When the same logical data is cached with equivalent parameters, both systems
should produce identical cache keys and be able to retrieve each other's cached data.
"""

import tempfile
import pytest
from pathlib import Path
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from cacheness import cacheness, cached, CacheConfig
from cacheness.decorators import _generate_cache_key
from cacheness.core import _normalize_function_args


class TestCrossSystemCompatibility:
    """Test cache key consistency and data interoperability between UnifiedCache and decorators."""

    def test_same_cache_keys_for_equivalent_data(self):
        """Test that UnifiedCache and decorators generate identical keys for equivalent data."""
        
        def test_function(user_id, data_type, include_meta=True):
            return f"result_for_{user_id}_{data_type}_{include_meta}"
        
        # Test parameters
        args = (123, "profile")
        kwargs = {"include_meta": True}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            cache = cacheness(config)
            
            # Generate decorator cache key
            decorator_key = _generate_cache_key(test_function, args, kwargs, config=config)
            
            # Create equivalent parameters for UnifiedCache
            normalized_params = _normalize_function_args(test_function, args, kwargs)
            func_name = getattr(test_function, "__qualname__", getattr(test_function, "__name__", "unknown"))
            func_module = getattr(test_function, "__module__", "unknown")
            func_id = f"{func_module}.{func_name}"
            
            enhanced_params = {**normalized_params, "__function__": func_id}
            unified_key = cache._create_cache_key(enhanced_params)
            
            assert decorator_key == unified_key, f"Keys should match: decorator={decorator_key}, unified={unified_key}"

    def test_decorator_cache_unified_retrieve(self):
        """Test that data cached by decorator can be retrieved by UnifiedCache."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            cache_instance = cacheness(config)
            
            # Create decorated function
            @cached(cache_instance=cache_instance)
            def compute_result(x, y, operation="add"):
                if operation == "add":
                    return x + y
                elif operation == "multiply":
                    return x * y
                else:
                    return 0
            
            # Cache data using decorator
            result1 = compute_result(10, 20, operation="add")
            assert result1 == 30
            
            # Retrieve using UnifiedCache directly
            normalized_params = _normalize_function_args(compute_result, (10, 20), {"operation": "add"})
            func_id = f"{compute_result.__module__}.{compute_result.__qualname__}"
            enhanced_params = {**normalized_params, "__function__": func_id}
            
            # Use the same synthetic parameter approach as decorator
            cached_data = cache_instance.get(__decorator_cache_key=cache_instance._create_cache_key(enhanced_params))
            
            assert cached_data == 30, f"UnifiedCache should retrieve decorator's cached data: {cached_data}"

    def test_unified_cache_decorator_retrieve(self):
        """Test that data cached by UnifiedCache can be retrieved by decorator."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            cache_instance = cacheness(config)
            
            def process_data(items, sort=True, limit=None):
                result = list(items)
                if sort:
                    result.sort()
                if limit:
                    result = result[:limit]
                return result
            
            # Cache data using UnifiedCache directly
            test_items = [3, 1, 4, 1, 5, 9, 2, 6]
            expected_result = [1, 1, 2, 3, 4]
            
            # Simulate what decorator would do: normalize parameters and create key
            normalized_params = _normalize_function_args(process_data, (test_items,), {"sort": True, "limit": 5})
            func_id = f"{process_data.__module__}.{process_data.__qualname__}"
            enhanced_params = {**normalized_params, "__function__": func_id}
            cache_key = cache_instance._create_cache_key(enhanced_params)
            
            # Store using UnifiedCache with decorator's synthetic parameter approach
            cache_instance.put(expected_result, __decorator_cache_key=cache_key)
            
            # Create decorated version of function
            @cached(cache_instance=cache_instance)
            def process_data_decorated(items, sort=True, limit=None):
                result = list(items)
                if sort:
                    result.sort()
                if limit:
                    result = result[:limit]
                return result
            
            # This should hit the cache instead of executing
            cached_result = process_data_decorated(test_items, sort=True, limit=5)
            
            assert cached_result == expected_result, f"Decorator should retrieve UnifiedCache's data: {cached_result}"

    def test_cross_compatibility_with_complex_data(self):
        """Test cross-compatibility with complex data types (NumPy arrays, etc)."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            cache_instance = cacheness(config)
            
            # Test with NumPy array
            test_array = np.array([[1, 2, 3], [4, 5, 6]])
            
            def matrix_operation(matrix, operation="transpose", scale=1.0):
                if operation == "transpose":
                    return matrix.T * scale
                elif operation == "flatten":
                    return matrix.flatten() * scale
                else:
                    return matrix * scale
            
            # Cache using decorator
            @cached(cache_instance=cache_instance)
            def decorated_matrix_op(matrix, operation="transpose", scale=1.0):
                return matrix_operation(matrix, operation, scale)
            
            # Get result from decorator
            decorator_result = decorated_matrix_op(test_array, operation="transpose", scale=2.0)
            expected = test_array.T * 2.0
            np.testing.assert_array_equal(decorator_result, expected)
            
            # Retrieve the same data using UnifiedCache directly
            normalized_params = _normalize_function_args(decorated_matrix_op, (test_array,), {"operation": "transpose", "scale": 2.0})
            func_id = f"{decorated_matrix_op.__module__}.{decorated_matrix_op.__qualname__}"
            enhanced_params = {**normalized_params, "__function__": func_id}
            cache_key = cache_instance._create_cache_key(enhanced_params)
            
            unified_result = cache_instance.get(__decorator_cache_key=cache_key)
            
            np.testing.assert_array_equal(unified_result, expected, "UnifiedCache should retrieve decorator's NumPy data")

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_cross_compatibility_with_dataframes(self):
        """Test cross-compatibility with pandas DataFrames."""
        
        if not PANDAS_AVAILABLE:
            pytest.skip("pandas not available")
        
        import pandas as pd
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            cache_instance = cacheness(config)
            
            # Test DataFrame
            test_df = pd.DataFrame({
                'A': [1, 2, 3, 4],
                'B': ['w', 'x', 'y', 'z'],
                'C': [1.1, 2.2, 3.3, 4.4]
            })
            
            def process_dataframe(df, column='A', operation='sum'):
                if operation == 'sum':
                    return df[column].sum()
                elif operation == 'mean':
                    return df[column].mean()
                else:
                    return len(df)
            
            # Store using UnifiedCache
            normalized_params = _normalize_function_args(process_dataframe, (test_df,), {"column": "A", "operation": "sum"})
            func_id = f"{process_dataframe.__module__}.{process_dataframe.__qualname__}"
            enhanced_params = {**normalized_params, "__function__": func_id}
            cache_key = cache_instance._create_cache_key(enhanced_params)
            
            expected_result = 10  # sum of [1, 2, 3, 4]
            cache_instance.put(expected_result, __decorator_cache_key=cache_key)
            
            # Retrieve using decorator
            @cached(cache_instance=cache_instance)
            def decorated_df_process(df, column='A', operation='sum'):
                return process_dataframe(df, column, operation)
            
            cached_result = decorated_df_process(test_df, column="A", operation="sum")
            
            assert cached_result == expected_result, f"Decorator should retrieve DataFrame processing result: {cached_result}"

    def test_parameter_normalization_consistency(self):
        """Test that parameter normalization is consistent between systems."""
        
        def multi_param_function(a, b=10, c=None, *args, **kwargs):
            return f"a={a}, b={b}, c={c}, args={args}, kwargs={kwargs}"
        
        # Test various calling conventions
        test_cases = [
            ((1, 20), {"c": "test"}),                           # positional + keyword
            ((1,), {"b": 20, "c": "test"}),                     # mixed
            ((), {"a": 1, "b": 20, "c": "test"}),              # all keyword
            ((1, 20, "test"), {}),                              # all positional
            ((1, 20, "test", "extra"), {"extra_kw": "value"}), # with *args and **kwargs
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            
            for i, (args, kwargs) in enumerate(test_cases):
                # Generate key using decorator method
                decorator_key = _generate_cache_key(multi_param_function, args, kwargs, config=config)
                
                # Generate key using UnifiedCache method
                normalized_params = _normalize_function_args(multi_param_function, args, kwargs)
                func_id = f"{multi_param_function.__module__}.{multi_param_function.__qualname__}"
                enhanced_params = {**normalized_params, "__function__": func_id}
                
                cache_instance = cacheness(config)
                unified_key = cache_instance._create_cache_key(enhanced_params)
                
                assert decorator_key == unified_key, f"Test case {i}: Keys should match for args={args}, kwargs={kwargs}"

    def test_cache_key_stability_across_systems(self):
        """Test that cache keys remain stable when switching between systems."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            cache_instance = cacheness(config)
            
            def stable_function(value, multiplier=2):
                return value * multiplier
            
            # Test data
            test_value = 42
            expected_result = 84
            
            # Round 1: Cache with decorator
            @cached(cache_instance=cache_instance)
            def cached_stable_function(value, multiplier=2):
                return stable_function(value, multiplier)
            
            result1 = cached_stable_function(test_value)
            assert result1 == expected_result
            
            # Round 2: Retrieve with UnifiedCache using same logical parameters
            normalized_params = _normalize_function_args(cached_stable_function, (test_value,), {"multiplier": 2})
            func_id = f"{cached_stable_function.__module__}.{cached_stable_function.__qualname__}"
            enhanced_params = {**normalized_params, "__function__": func_id}
            cache_key = cache_instance._create_cache_key(enhanced_params)
            
            result2 = cache_instance.get(__decorator_cache_key=cache_key)
            assert result2 == expected_result, "UnifiedCache should retrieve decorator's cached result"
            
            # Round 3: Cache additional data with UnifiedCache
            cache_instance.put(999, __decorator_cache_key="custom_test_key")
            retrieved_custom = cache_instance.get(__decorator_cache_key="custom_test_key")
            assert retrieved_custom == 999
            
            # Round 4: Verify original cache still works
            result3 = cached_stable_function(test_value, multiplier=2)
            assert result3 == expected_result, "Original decorator cache should still work"

    def test_different_functions_different_keys(self):
        """Test that different functions with same parameters produce different cache keys."""
        
        def function_a(x, y=10):
            return x + y
        
        def function_b(x, y=10):
            return x * y
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            
            # Same parameters, different functions
            args = (5,)
            kwargs = {"y": 10}
            
            key_a = _generate_cache_key(function_a, args, kwargs, config=config)
            key_b = _generate_cache_key(function_b, args, kwargs, config=config)
            
            assert key_a != key_b, f"Different functions should have different keys: {key_a} vs {key_b}"
            
            # Test with UnifiedCache approach too
            cache_instance = cacheness(config)
            
            normalized_a = _normalize_function_args(function_a, args, kwargs)
            func_id_a = f"{function_a.__module__}.{function_a.__qualname__}"
            enhanced_a = {**normalized_a, "__function__": func_id_a}
            unified_key_a = cache_instance._create_cache_key(enhanced_a)
            
            normalized_b = _normalize_function_args(function_b, args, kwargs)
            func_id_b = f"{function_b.__module__}.{function_b.__qualname__}"
            enhanced_b = {**normalized_b, "__function__": func_id_b}
            unified_key_b = cache_instance._create_cache_key(enhanced_b)
            
            assert unified_key_a != unified_key_b, "UnifiedCache should also generate different keys for different functions"
            assert key_a == unified_key_a, "Decorator and UnifiedCache should generate same key for function_a"
            assert key_b == unified_key_b, "Decorator and UnifiedCache should generate same key for function_b"