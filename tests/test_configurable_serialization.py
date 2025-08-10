"""
Tests for Configurable Serialization and Handler Priority
========================================================

This module tests the new CacheConfig options for controlling serialization
methods and handler selection order.
"""

import pytest
import tempfile
import numpy as np

from cacheness.core import CacheConfig, UnifiedCache
from cacheness.decorators import cached
from cacheness.serialization import serialize_for_cache_key, create_unified_cache_key

# DataFrame libraries
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore
    POLARS_AVAILABLE = False


class TestSerializationConfiguration:
    """Test configurable serialization behavior."""

    def test_default_serialization_all_enabled(self):
        """Test that default config enables all serialization methods."""
        config = CacheConfig()
        
        # Test basic types
        assert serialize_for_cache_key("hello", config) == "str:hello"
        assert serialize_for_cache_key(42, config) == "int:42"
        assert serialize_for_cache_key(None, config) == "None"
        
        # Test collections
        assert "list:" in serialize_for_cache_key([1, 2, 3], config)
        assert "dict:" in serialize_for_cache_key({"a": 1}, config)
        
        # Test numpy array
        arr = np.array([1, 2, 3])
        result = serialize_for_cache_key(arr, config)
        assert result.startswith("array:")

    def test_disable_basic_types(self):
        """Test disabling basic type serialization."""
        config = CacheConfig(enable_basic_types=False)
        
        # Should fall back to hashable/string methods
        result_str = serialize_for_cache_key("hello", config)
        result_int = serialize_for_cache_key(42, config)
        
        # Should not use basic type serialization
        assert not result_str.startswith("str:")
        assert not result_int.startswith("int:")
        
        # Should use hashable fallback
        assert "hashed:" in result_str or "str:" in result_str

    def test_disable_collections(self):
        """Test disabling collection introspection."""
        config = CacheConfig(enable_collections=False)
        
        # Should fall back to string method for lists (not recursive introspection)
        result = serialize_for_cache_key([1, 2, 3], config)
        
        # Should start with "list:" but be the string representation, not recursive
        assert result.startswith("list:")
        # String fallback should be: "list:[1, 2, 3]" (literal string, not introspected)
        assert result == "list:[1, 2, 3]"
        
        # Compare with collections enabled to see the difference
        config_enabled = CacheConfig(enable_collections=True)
        result_enabled = serialize_for_cache_key([1, 2, 3], config_enabled)
        
        # With collections enabled, should be introspected: "list:[int:1,int:2,int:3]"
        assert result_enabled.startswith("list:")
        assert "int:1" in result_enabled  # Should contain introspected elements
        
        # Results should be different
        assert result != result_enabled

    def test_disable_special_cases(self):
        """Test disabling special case handling."""
        config = CacheConfig(enable_special_cases=False)
        
        # NumPy array should not use special handling
        arr = np.array([1, 2, 3])
        result = serialize_for_cache_key(arr, config)
        assert not result.startswith("array:")

    def test_tuple_recursive_length_threshold(self):
        """Test configurable tuple recursive serialization threshold."""
        # Small threshold - should use hash for small tuples
        config = CacheConfig(max_tuple_recursive_length=2)
        
        small_tuple = (1, 2)
        large_tuple = (1, 2, 3, 4)
        
        small_result = serialize_for_cache_key(small_tuple, config)
        large_result = serialize_for_cache_key(large_tuple, config)
        
        # Small tuple should use recursive
        assert small_result.startswith("tuple:")
        
        # Large tuple should use hash
        assert "hashed:" in large_result

    def test_collection_depth_limit(self):
        """Test maximum collection recursion depth."""
        config = CacheConfig(max_collection_depth=2)
        
        # Create deeply nested structure
        deep_dict = {"level1": {"level2": {"level3": {"level4": "deep"}}}}
        
        result = serialize_for_cache_key(deep_dict, config)
        # Should contain max_depth_exceeded for the deepest levels
        assert "max_depth_exceeded" in result

    def test_unified_cache_key_with_config(self):
        """Test that create_unified_cache_key respects config."""
        config = CacheConfig(enable_collections=False)
        
        params = {"data": [1, 2, 3], "name": "test"}
        
        # Generate key with config
        key_with_config = create_unified_cache_key(params, config)
        
        # Generate key without config (default behavior)
        key_default = create_unified_cache_key(params, None)
        
        # Should be different due to different serialization
        assert key_with_config != key_default


class TestHandlerConfiguration:
    """Test configurable handler priority and enabling/disabling."""

    def test_default_handler_registry(self):
        """Test default handler registry initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            cache = UnifiedCache(config)
            
            # Should have all default handlers
            handler_types = [type(h).__name__ for h in cache.handlers.handlers]
            
            # Check that handlers are present (depending on what's available)
            if POLARS_AVAILABLE:
                assert "PolarsSeriesHandler" in handler_types
                assert "PolarsDataFrameHandler" in handler_types
            if PANDAS_AVAILABLE:
                assert "PandasSeriesHandler" in handler_types
                assert "PandasDataFrameHandler" in handler_types
            
            assert "ArrayHandler" in handler_types
            assert "ObjectHandler" in handler_types

    def test_disable_specific_handlers(self):
        """Test disabling specific handlers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=temp_dir,
                enable_pandas_dataframes=False,
                enable_numpy_arrays=False
            )
            cache = UnifiedCache(config)
            
            handler_types = [type(h).__name__ for h in cache.handlers.handlers]
            
            # Disabled handlers should not be present
            assert "PandasDataFrameHandler" not in handler_types
            assert "ArrayHandler" not in handler_types
            
            # Other handlers should still be present
            assert "ObjectHandler" in handler_types

    def test_custom_handler_priority(self):
        """Test custom handler priority order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Specify custom priority - object handler first
            config = CacheConfig(
                cache_dir=temp_dir,
                handler_priority=["object_pickle", "numpy_arrays", "pandas_dataframes"]
            )
            cache = UnifiedCache(config)
            
            handler_types = [type(h).__name__ for h in cache.handlers.handlers]
            
            # Object handler should be first (if present)
            if "ObjectHandler" in handler_types:
                assert handler_types[0] == "ObjectHandler"


class TestDecoratorWithConfiguration:
    """Test decorators with configurable serialization."""

    def test_decorator_with_custom_serialization(self):
        """Test that decorators respect config serialization settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Config that disables collection introspection
            config = CacheConfig(
                cache_dir=temp_dir,
                enable_collections=False
            )
            cache_instance = UnifiedCache(config)
            
            call_count = 0
            
            @cached(cache_instance=cache_instance)
            def process_list(data_list):
                nonlocal call_count
                call_count += 1
                return sum(data_list)
            
            # First call
            result1 = process_list([1, 2, 3, 4])
            assert result1 == 10
            assert call_count == 1
            
            # Second call with same list - should hit cache
            result2 = process_list([1, 2, 3, 4])
            assert result2 == 10
            assert call_count == 1

    def test_decorator_with_different_configs(self):
        """Test that different configs produce different cache keys."""
        with tempfile.TemporaryDirectory() as temp_dir1:
            with tempfile.TemporaryDirectory() as temp_dir2:
                # Two different configs
                config1 = CacheConfig(cache_dir=temp_dir1, enable_collections=True)
                config2 = CacheConfig(cache_dir=temp_dir2, enable_collections=False)
                
                cache1 = UnifiedCache(config1)
                cache2 = UnifiedCache(config2)
                
                call_count = 0
                
                @cached(cache_instance=cache1)
                def func_with_config1(data):
                    nonlocal call_count
                    call_count += 1
                    return len(data)
                
                @cached(cache_instance=cache2)
                def func_with_config2(data):
                    nonlocal call_count
                    call_count += 1
                    return len(data)
                
                # Call both with same data
                data = [1, 2, 3]
                result1 = func_with_config1(data)
                result2 = func_with_config2(data)
                
                # Both should execute (different cache keys due to different configs)
                assert result1 == 3
                assert result2 == 3
                assert call_count == 2


class TestRealWorldUseCases:
    """Test real-world use cases for configuration."""

    def test_performance_optimized_config(self):
        """Test a config optimized for performance over precision."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Config favoring hash-based methods for speed
            config = CacheConfig(
                cache_dir=temp_dir,
                enable_collections=False,  # Skip expensive introspection
                enable_object_introspection=False,  # Skip __dict__ inspection
                max_tuple_recursive_length=2,  # Limit recursion
                max_collection_depth=3  # Limit depth
            )
            cache = UnifiedCache(config)
            
            # Test with complex data
            complex_data = {
                'arrays': [np.array([1, 2, 3]), np.array([4, 5, 6])],
                'nested': {'deep': {'very_deep': {'extremely_deep': 'value'}}},
                'large_tuple': tuple(range(100))
            }
            
            # Should still work but use simpler serialization
            cache_key = cache._create_cache_key({"data": complex_data})
            assert len(cache_key) == 16  # Standard key length

    def test_precision_optimized_config(self):
        """Test a config optimized for precision over performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Config favoring detailed introspection
            config = CacheConfig(
                cache_dir=temp_dir,
                enable_collections=True,
                enable_object_introspection=True,
                max_tuple_recursive_length=50,  # Allow deep recursion
                max_collection_depth=20  # Allow deep nesting
            )
            cache = UnifiedCache(config)
            
            # Test with detailed data
            class CustomObject:
                def __init__(self, value):
                    self.value = value
                    self.metadata = {"created": True}
            
            obj = CustomObject(42)
            
            # Should use detailed object introspection
            cache_key = cache._create_cache_key({"obj": obj})
            assert len(cache_key) == 16  # Standard key length

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not available")
    def test_dataframe_priority_config(self):
        """Test handler priority for DataFrame processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prefer pandas over polars
            config = CacheConfig(
                cache_dir=temp_dir,
                handler_priority=["pandas_dataframes", "polars_dataframes", "object_pickle"]
            )
            cache = UnifiedCache(config)
            
            # Create a pandas DataFrame
            df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            
            # Should use pandas handler (first in priority)
            handler = cache.handlers.get_handler(df)
            assert type(handler).__name__ == "PandasDataFrameHandler"
