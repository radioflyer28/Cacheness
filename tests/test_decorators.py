"""
Test Decorator Functionality
============================

Tests for the @cached decorator and related function-level caching utilities.
"""

import time
import tempfile
import numpy as np
import pytest
from pathlib import Path

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from cacheness import cached, cacheness, CacheConfig
from cacheness.decorators import cache_function, memoize, CacheContext


class TestCachedDecorator:
    """Test the @cached decorator functionality."""

    def test_basic_function_caching(self):
        """Test basic function caching with decorator."""
        # Use temporary directory to ensure clean cache state
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            cache_instance = cacheness(config)

            call_count = 0

            @cached(cache_instance=cache_instance)
            def expensive_function(x, y):
                nonlocal call_count
                call_count += 1
                time.sleep(0.01)  # Simulate some work
                return x * y

            # First call should execute function
            result1 = expensive_function(5, 10)
            assert result1 == 50
            assert call_count == 1

            # Second call should hit cache
            result2 = expensive_function(5, 10)
            assert result2 == 50
            assert call_count == 1  # No additional calls

            # Different arguments should execute function again
            result3 = expensive_function(3, 7)
            assert result3 == 21
            assert call_count == 2

    def test_multiple_return_values_tuple(self):
        """Test caching function that returns tuple of multiple values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            cache_instance = cacheness(config)

            call_count = 0

            @cached(cache_instance=cache_instance)
            def multi_return_function(x):
                nonlocal call_count
                call_count += 1
                return x, x**2, x**3, f"processed_{x}"

            # First call
            result1 = multi_return_function(5)
            assert result1 == (5, 25, 125, "processed_5")
            assert call_count == 1

            # Second call should hit cache
            result2 = multi_return_function(5)
            assert result2 == (5, 25, 125, "processed_5")
            assert call_count == 1  # No additional calls

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas not available")
    def test_multiple_dataframes_return(self):
        """Test caching functions that return multiple DataFrames."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            cache_instance = cacheness(config)

            call_count = 0

            @cached(cache_instance=cache_instance)
            def create_datasets():
                nonlocal call_count
                call_count += 1

                # Create test DataFrames
                df_train = pd.DataFrame(
                    {
                        "feature1": [1, 2, 3, 4],
                        "feature2": [5, 6, 7, 8],
                        "target": [0, 1, 0, 1],
                    }
                )

                df_test = pd.DataFrame(
                    {"feature1": [9, 10], "feature2": [11, 12], "target": [1, 0]}
                )

                metadata = {
                    "created_at": "2024-01-15",
                    "version": "1.0",
                    "samples": {"train": len(df_train), "test": len(df_test)},
                }

                return df_train, df_test, metadata

            # First call
            result1 = create_datasets()
            assert call_count == 1
            train1, test1, meta1 = result1

            # Verify types
            assert isinstance(train1, pd.DataFrame)
            assert isinstance(test1, pd.DataFrame)
            assert isinstance(meta1, dict)

            # Second call should hit cache
            result2 = create_datasets()
            assert call_count == 1  # No additional calls
            train2, test2, meta2 = result2

            # Verify data integrity
            assert train1.equals(train2)
            assert test1.equals(test2)
            assert meta1 == meta2

    def test_decorator_with_custom_ttl(self):
        """Test decorator with custom TTL."""

        @cached(ttl_hours=1)
        def time_sensitive_function(x):
            return f"result_{x}"

        result = time_sensitive_function(10)
        assert result == "result_10"

        # Should hit cache
        result2 = time_sensitive_function(10)
        assert result2 == "result_10"

    def test_decorator_with_key_prefix(self):
        """Test decorator with custom key prefix."""

        @cached(key_prefix="test_prefix")
        def prefixed_function(x):
            return x * 2

        result = prefixed_function(5)
        assert result == 10

        # Verify cache key generation includes prefix
        cache_key = prefixed_function.cache_key(5)  # type: ignore
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0

    def test_decorator_cache_management_methods(self):
        """Test cache management methods attached to decorated functions."""

        @cached()
        def managed_function(x):
            return x**2

        # Test cache_key method
        key1 = managed_function.cache_key(5)  # type: ignore
        key2 = managed_function.cache_key(5)  # type: ignore
        key3 = managed_function.cache_key(10)  # type: ignore

        assert key1 == key2  # Same args = same key
        assert key1 != key3  # Different args = different key

        # Test cache_info method
        info = managed_function.cache_info()  # type: ignore
        assert isinstance(info, dict)
        assert "function" in info
        assert "cache_dir" in info

        # Test cache_clear method (returns 0 for now as it's not fully implemented)
        cleared = managed_function.cache_clear()  # type: ignore
        assert isinstance(cleared, int)

    def test_custom_cache_instance(self):
        """Test decorator with custom cache instance."""

        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir, default_ttl_hours=48)
            custom_cache = cacheness(config)

            @cached(cache_instance=custom_cache)
            def custom_cached_function(x):
                return f"custom_{x}"

            result = custom_cached_function(100)
            assert result == "custom_100"

            # Verify it used the custom cache
            stats = custom_cache.get_stats()
            assert stats["total_entries"] >= 1

    def test_error_handling_ignore_errors_true(self):
        """Test decorator error handling with ignore_errors=True (default)."""

        @cached(ignore_errors=True)
        def function_with_bad_args(obj):
            return f"processed_{obj}"

        # Function should work even if caching fails
        class UnserializableObj:
            def __reduce__(self):
                raise TypeError("Cannot serialize this object")

        # This should not raise an error, just skip caching
        result = function_with_bad_args(UnserializableObj())
        assert "processed_" in result

    def test_error_handling_ignore_errors_false(self):
        """Test decorator error handling with ignore_errors=False."""

        # Note: This test is tricky because our serialization is very robust
        # We'll test with a custom key function that can fail

        def failing_key_func(func, args, kwargs):
            raise ValueError("Key generation failed!")

        @cached(ignore_errors=False, key_func=failing_key_func)
        def function_with_failing_key(x):
            return x * 2

        # Should raise RuntimeError wrapping the original error
        with pytest.raises(RuntimeError, match="Cache key generation failed"):
            function_with_failing_key(5)


class TestCacheFunctionInterface:
    """Test the cache_function alternative interface."""

    def test_cache_function_as_decorator(self):
        """Test cache_function used as a decorator."""

        @cache_function
        def simple_func(x):
            return x + 1

        result1 = simple_func(5)  # type: ignore
        result2 = simple_func(5)  # type: ignore
        assert result1 == result2 == 6

    def test_cache_function_with_args(self):
        """Test cache_function with arguments."""

        @cache_function(ttl_hours=2, key_prefix="func_test")
        def configured_func(x):
            return x * 3

        result = configured_func(4)
        assert result == 12

    def test_cache_function_wrapping(self):
        """Test cache_function for wrapping existing functions."""

        def existing_function(a, b):
            return a + b

        # Wrap the function with caching
        cached_func = cache_function(existing_function, ttl_hours=1)  # type: ignore

        result1 = cached_func(2, 3)  # type: ignore
        result2 = cached_func(2, 3)  # type: ignore
        assert result1 == result2 == 5


class TestMemoizeDecorator:
    """Test the memoize decorator."""

    def test_memoize_basic(self):
        """Test basic memoization."""

        call_count = 0

        @memoize
        def fibonacci(n):
            nonlocal call_count
            call_count += 1
            if n < 2:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        # Calculate fibonacci(5) - should cache intermediate results
        result = fibonacci(5)
        assert result == 5  # 0,1,1,2,3,5

        # Call count should be much less than 2^5 due to memoization
        assert call_count <= 6  # Should only calculate each unique n once

    def test_memoize_permanent_cache(self):
        """Test that memoize creates permanent cache (no TTL)."""

        @memoize
        def permanent_func(x):
            return f"permanent_{x}"

        result = permanent_func(10)
        assert result == "permanent_10"

        # The function should have ttl_hours=None (permanent)
        # This is implementation detail, but we can check via cache_info
        info = permanent_func.cache_info()  # type: ignore
        assert info["ttl_hours"] is None


class TestCacheContext:
    """Test the CacheContext context manager."""

    def test_cache_context_basic(self):
        """Test basic CacheContext usage."""

        with CacheContext(default_ttl_hours=1) as ctx:

            @ctx.cached()
            def context_function(x):
                return x * 10

            result1 = context_function(5)
            result2 = context_function(5)
            assert result1 == result2 == 50

    def test_cache_context_with_custom_dir(self):
        """Test CacheContext with custom cache directory."""

        with tempfile.TemporaryDirectory() as temp_dir:
            with CacheContext(cache_dir=temp_dir) as ctx:

                @ctx.cached(key_prefix="context_test")
                def context_dir_function(x):
                    return f"context_result_{x}"

                result = context_dir_function(100)
                assert result == "context_result_100"

                # Verify cache file was created in the temp directory
                cache_path = Path(temp_dir)
                assert cache_path.exists()
                # Should have at least metadata file
                cache_files = list(cache_path.iterdir())
                assert len(cache_files) > 0


class TestDecoratorEdgeCases:
    """Test edge cases and complex scenarios."""

    def test_nested_data_structures(self):
        """Test caching with complex nested data structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir)
            cache_instance = cacheness(config)

            call_count = 0

            @cached(cache_instance=cache_instance)
            def create_nested_structure():
                nonlocal call_count
                call_count += 1
                return {
                    "level1": {
                        "level2": {
                            "arrays": [np.array([1, 2, 3]), np.array([4, 5, 6])],
                            "metadata": {"created": True, "version": 1},
                        },
                        "simple_list": [1, 2, 3, 4, 5],
                    },
                    "top_level_array": np.array(
                        [0.1, 0.2, 0.3, 0.4, 0.5]
                    ),  # Fixed values for consistent comparison
                }

            result1 = create_nested_structure()
            assert call_count == 1

            result2 = create_nested_structure()
            assert call_count == 1  # Should hit cache

            # Compare individual components (can't directly compare dicts with numpy arrays)
            assert result1["level1"]["simple_list"] == result2["level1"]["simple_list"]
            assert (
                result1["level1"]["level2"]["metadata"]
                == result2["level1"]["level2"]["metadata"]
            )

            # Verify nested arrays are equal
            np.testing.assert_array_equal(
                result1["top_level_array"], result2["top_level_array"]
            )
            np.testing.assert_array_equal(
                result1["level1"]["level2"]["arrays"][0],
                result2["level1"]["level2"]["arrays"][0],
            )
            np.testing.assert_array_equal(
                result1["level1"]["level2"]["arrays"][1],
                result2["level1"]["level2"]["arrays"][1],
            )

    def test_function_with_defaults(self):
        """Test caching functions with default arguments."""

        @cached()
        def func_with_defaults(a, b=10, c=20):
            return a + b + c

        # Different ways of calling should generate different cache keys
        result1 = func_with_defaults(1)  # a=1, b=10, c=20
        result2 = func_with_defaults(1, 10)  # a=1, b=10, c=20 (same as above)
        result3 = func_with_defaults(1, 10, 20)  # a=1, b=10, c=20 (same as above)
        result4 = func_with_defaults(1, b=15)  # a=1, b=15, c=20 (different)

        assert result1 == result2 == result3 == 31
        assert result4 == 36

    def test_empty_return_value(self):
        """Test caching functions that return None or empty values."""

        @cached()
        def returns_none():
            return None

        @cached()
        def returns_empty_list():
            return []

        @cached()
        def returns_empty_dict():
            return {}

        # All should work and cache properly
        assert returns_none() is None
        assert returns_empty_list() == []
        assert returns_empty_dict() == {}

        # Second calls should hit cache
        assert returns_none() is None
        assert returns_empty_list() == []
        assert returns_empty_dict() == {}
