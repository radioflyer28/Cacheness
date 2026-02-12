import tempfile
import numpy as np

from cacheness import cacheness, CacheConfig
from cacheness.config import SerializationConfig, HandlerConfig, CacheStorageConfig
from cacheness.serialization import serialize_for_cache_key
from cacheness.decorators import cached


class TestSerializationConfiguration:
    """Test various serialization configuration options."""

    def test_default_serialization_all_enabled(self):
        """Test that default configuration has all serialization methods enabled."""
        config = CacheConfig()

        # All methods should be enabled by default
        assert config.serialization.enable_basic_types is True
        assert config.serialization.enable_collections is True
        assert config.serialization.enable_special_cases is True
        assert config.serialization.enable_object_introspection is True
        assert config.serialization.enable_hashable_fallback is True
        assert config.serialization.enable_string_fallback is True

        # Test that all serialization types work
        assert serialize_for_cache_key("hello", config).startswith("str:")
        assert serialize_for_cache_key(42, config).startswith("int:")
        assert serialize_for_cache_key([1, 2, 3], config).startswith("list:")

        # NumPy array should use special handling
        arr = np.array([1, 2, 3])
        result = serialize_for_cache_key(arr, config)
        assert result.startswith("array:")

    def test_disable_basic_types(self):
        """Test disabling basic type serialization."""
        config = CacheConfig(
            serialization=SerializationConfig(enable_basic_types=False)
        )

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
        config = CacheConfig(
            serialization=SerializationConfig(enable_collections=False)
        )

        # Should fall back to string method for lists (not recursive introspection)
        result = serialize_for_cache_key([1, 2, 3], config)

        # Should start with "list:" but be the string representation, not recursive
        assert result.startswith("list:")
        # String fallback should be: "list:[1, 2, 3]" (literal string, not introspected)
        assert result == "list:[1, 2, 3]"

        # Compare with collections enabled to see the difference
        config_enabled = CacheConfig(
            serialization=SerializationConfig(enable_collections=True)
        )
        result_enabled = serialize_for_cache_key([1, 2, 3], config_enabled)

        # With collections enabled, should be introspected: "list:[int:1,int:2,int:3]"
        assert result_enabled.startswith("list:")
        assert "int:1" in result_enabled  # Should contain introspected elements

        # Results should be different
        assert result != result_enabled

    def test_disable_special_cases(self):
        """Test disabling special case handling."""
        config = CacheConfig(
            serialization=SerializationConfig(enable_special_cases=False)
        )

        # NumPy array should not use special handling
        arr = np.array([1, 2, 3])
        result = serialize_for_cache_key(arr, config)
        assert not result.startswith("array:")

    def test_tuple_recursive_length_threshold(self):
        """Test configurable tuple recursive serialization threshold."""
        # Small threshold - should use hash for small tuples
        config = CacheConfig(
            serialization=SerializationConfig(max_tuple_recursive_length=2)
        )

        small_tuple = (1, 2)
        large_tuple = (1, 2, 3, 4)

        result_small = serialize_for_cache_key(small_tuple, config)
        result_large = serialize_for_cache_key(large_tuple, config)

        # Small tuple should be recursively serialized
        assert result_small.startswith("tuple:")
        assert "int:1" in result_small

        # Large tuple should fall back to hash
        assert "hashed:" in result_large

    def test_collection_depth_limit(self):
        """Test configurable collection depth limit."""
        # Limit depth to 2
        config = CacheConfig(serialization=SerializationConfig(max_collection_depth=2))

        # Deep nested structure
        deep_list = [1, [2, [3, [4]]]]
        result = serialize_for_cache_key(deep_list, config)

        # Should have depth limitation applied
        assert "max_depth_exceeded" in result

    def test_unified_cache_key_with_config(self):
        """Test that cache keys respect configuration."""
        config = CacheConfig(
            serialization=SerializationConfig(enable_collections=False)
        )

        # Same data with different configs should produce different keys
        data = {"list": [1, 2, 3], "value": 42}

        # Direct key generation (not through cacheness instance)
        from cacheness.serialization import create_unified_cache_key

        key = create_unified_cache_key(data, config)

        # Should be a 16-character hex string
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)


class TestHandlerConfiguration:
    """Test handler configuration options."""

    def test_default_handler_registry(self):
        """Test that default configuration includes expected handlers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(storage=CacheStorageConfig(cache_dir=temp_dir))
            cache = cacheness(config)

            # Should have standard handlers available
            handler_types = [type(h).__name__ for h in cache.handlers.handlers]
            assert "ObjectHandler" in handler_types

            cache.close()

    def test_disable_specific_handlers(self):
        """Test disabling specific data type handlers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                storage=CacheStorageConfig(cache_dir=temp_dir),
                handlers=HandlerConfig(
                    enable_pandas_dataframes=False, enable_numpy_arrays=False
                ),
            )
            cache = cacheness(config)

            # Should not have disabled handlers
            handler_types = [type(h).__name__ for h in cache.handlers.handlers]
            # Note: Handler availability depends on what's actually imported
            # This test mainly ensures configuration doesn't break
            assert isinstance(handler_types, list)

            cache.close()

    def test_custom_handler_priority(self):
        """Test custom handler priority order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Specify custom priority - object handler first
            config = CacheConfig(
                storage=CacheStorageConfig(cache_dir=temp_dir),
                handlers=HandlerConfig(
                    handler_priority=[
                        "object_pickle",
                        "numpy_arrays",
                        "pandas_dataframes",
                    ]
                ),
            )
            cache = cacheness(config)

            handler_types = [type(h).__name__ for h in cache.handlers.handlers]

            # Object handler should be first (if present)
            if "ObjectHandler" in handler_types:
                assert handler_types[0] == "ObjectHandler"

            cache.close()


class TestDecoratorWithConfiguration:
    """Test decorators with custom configuration."""

    def test_decorator_with_custom_serialization(self):
        """Test decorator respects custom serialization configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                storage=CacheStorageConfig(cache_dir=temp_dir),
                serialization=SerializationConfig(enable_collections=False),
            )
            cache = cacheness(config)

            @cached(cache_instance=cache)
            def process_data(data):
                return f"Processed: {data}"

            # Test with list data
            result = process_data([1, 2, 3])
            assert result == "Processed: [1, 2, 3]"

            # Second call should hit cache
            result2 = process_data([1, 2, 3])
            assert result2 == result

            cache.close()

    def test_decorator_with_different_configs(self):
        """Test that different configurations produce different caching behavior."""
        with tempfile.TemporaryDirectory() as temp_dir1:
            with tempfile.TemporaryDirectory() as temp_dir2:
                config1 = CacheConfig(
                    storage=CacheStorageConfig(cache_dir=temp_dir1),
                    serialization=SerializationConfig(enable_collections=True),
                )
                config2 = CacheConfig(
                    storage=CacheStorageConfig(cache_dir=temp_dir2),
                    serialization=SerializationConfig(enable_collections=False),
                )

                cache1 = cacheness(config1)
                cache2 = cacheness(config2)

                @cached(cache_instance=cache1)
                def func1(data):
                    return f"Config1: {data}"

                @cached(cache_instance=cache2)
                def func2(data):
                    return f"Config2: {data}"

                # Same input should work with both caches
                data = [1, 2, 3]
                result1 = func1(data)
                result2 = func2(data)

                assert result1 == "Config1: [1, 2, 3]"
                assert result2 == "Config2: [1, 2, 3]"

                cache2.close()

            cache1.close()


class TestRealWorldUseCases:
    """Test realistic configuration scenarios."""

    def test_performance_optimized_config(self):
        """Test performance-optimized configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig.create_performance_optimized()
            config.storage.cache_dir = temp_dir
            cache = cacheness(config)

            # Should work with any data type
            @cached(cache_instance=cache)
            def compute(data):
                return sum(data) if isinstance(data, (list, tuple)) else data

            result = compute([1, 2, 3, 4, 5])
            assert result == 15

            cache.close()

    def test_precision_optimized_config(self):
        """Test precision-optimized configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig.create_size_optimized()
            config.storage.cache_dir = temp_dir
            cache = cacheness(config)

            # Should work with complex data
            @cached(cache_instance=cache)
            def process_complex_data(data):
                return {"processed": True, "count": len(data)}

            result = process_complex_data([1, 2, 3])
            assert result == {"processed": True, "count": 3}

            cache.close()

    def test_dataframe_priority_config(self):
        """Test configuration optimized for DataFrame processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                storage=CacheStorageConfig(cache_dir=temp_dir),
                serialization=SerializationConfig(
                    enable_collections=False,  # Skip expensive introspection
                    enable_object_introspection=False,  # Skip __dict__ inspection
                    max_collection_depth=3,
                ),
                handlers=HandlerConfig(
                    handler_priority=[
                        "pandas_dataframes",
                        "polars_dataframes",
                        "object_pickle",
                    ]
                ),
            )
            cache = cacheness(config)

            # Should work efficiently with DataFrames
            @cached(cache_instance=cache)
            def analyze_data(description):
                return f"Analysis: {description}"

            result = analyze_data("sample data")
            assert result == "Analysis: sample data"

            cache.close()
