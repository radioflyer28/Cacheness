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

            cache_instance.close()

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

            cache_instance.close()

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

            cache_instance.close()

    def test_decorator_with_custom_ttl(self):
        """Test decorator with custom TTL."""

        @cached(ttl_seconds=3600)
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
            config = CacheConfig(cache_dir=temp_dir, default_ttl_seconds=172800)
            custom_cache = cacheness(config)

            @cached(cache_instance=custom_cache)
            def custom_cached_function(x):
                return f"custom_{x}"

            result = custom_cached_function(100)
            assert result == "custom_100"

            # Verify it used the custom cache
            stats = custom_cache.get_stats()
            assert stats["total_entries"] >= 1

            custom_cache.close()

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

        @cache_function(ttl_seconds=7200, key_prefix="func_test")
        def configured_func(x):
            return x * 3

        result = configured_func(4)
        assert result == 12

    def test_cache_function_wrapping(self):
        """Test cache_function for wrapping existing functions."""

        def existing_function(a, b):
            return a + b

        # Wrap the function with caching
        cached_func = cache_function(existing_function, ttl_seconds=3600)  # type: ignore

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

        # The function should have ttl_seconds=None (permanent)
        # This is implementation detail, but we can check via cache_info
        info = permanent_func.cache_info()  # type: ignore
        assert info["ttl_seconds"] is None


class TestCacheClearFunctionality:
    """Test cache_clear() method on decorated functions."""

    def test_cache_clear_removes_entries(self, tmp_path):
        """Test that cache_clear() actually deletes cached entries."""
        call_count = 0

        cache_config = CacheConfig(cache_dir=tmp_path)

        @cached(cache_instance=cacheness(cache_config))
        def func_to_clear(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # Create cache entries
        result1 = func_to_clear(1)
        result2 = func_to_clear(2)
        result3 = func_to_clear(3)
        assert result1 == 2
        assert result2 == 4
        assert result3 == 6
        assert call_count == 3

        # Verify entries are cached (calls don't increment count)
        func_to_clear(1)
        func_to_clear(2)
        func_to_clear(3)
        assert call_count == 3  # No new calls

        # Clear cache
        cleared = func_to_clear.cache_clear()
        assert cleared == 3  # Should have deleted 3 entries

        # Verify entries are gone (calls increment count)
        func_to_clear(1)
        func_to_clear(2)
        func_to_clear(3)
        assert call_count == 6  # 3 new calls

    def test_cache_clear_independent_functions(self, tmp_path):
        """Test that cache_clear() on one function doesn't affect another."""
        cache_config = CacheConfig(cache_dir=tmp_path)
        cache_instance = cacheness(cache_config)

        count_a = 0
        count_b = 0

        @cached(cache_instance=cache_instance, key_prefix="func_a")
        def func_a(x):
            nonlocal count_a
            count_a += 1
            return x * 2

        @cached(cache_instance=cache_instance, key_prefix="func_b")
        def func_b(x):
            nonlocal count_b
            count_b += 1
            return x * 3

        # Create entries for both functions
        func_a(1)
        func_a(2)
        func_b(1)
        func_b(2)
        assert count_a == 2
        assert count_b == 2

        # Clear only func_a
        cleared_a = func_a.cache_clear()
        assert cleared_a == 2

        # Verify func_a entries are gone but func_b entries remain
        func_a(1)
        func_a(2)
        func_b(1)
        func_b(2)
        assert count_a == 4  # func_a was called again
        assert count_b == 2  # func_b used cache

    def test_cache_clear_empty_cache(self, tmp_path):
        """Test that cache_clear() returns 0 when cache is empty."""
        cache_config = CacheConfig(cache_dir=tmp_path)

        @cached(cache_instance=cacheness(cache_config))
        def empty_func(x):
            return x + 1

        # Clear without creating any entries
        cleared = empty_func.cache_clear()
        assert cleared == 0

    def test_cache_clear_multiple_calls(self, tmp_path):
        """Test that calling cache_clear() multiple times is safe."""
        cache_config = CacheConfig(cache_dir=tmp_path)

        @cached(cache_instance=cacheness(cache_config))
        def multi_clear_func(x):
            return x * 2

        # Create entry
        multi_clear_func(5)

        # First clear
        cleared1 = multi_clear_func.cache_clear()
        assert cleared1 == 1

        # Second clear should return 0
        cleared2 = multi_clear_func.cache_clear()
        assert cleared2 == 0


class TestCacheInfoStatistics:
    """Test cache_info() hit/miss statistics."""

    def test_cache_info_tracks_hits_and_misses(self, tmp_path):
        """Test that cache_info() returns accurate hit/miss counts."""
        cache_config = CacheConfig(cache_dir=tmp_path)

        @cached(cache_instance=cacheness(cache_config))
        def stat_func(x):
            return x * 2

        # Get initial info (should be all zeros)
        info = stat_func.cache_info()
        assert info["hits"] == 0
        assert info["misses"] == 0
        assert info["size"] == 0

        # First call is a miss
        result1 = stat_func(5)
        assert result1 == 10

        info = stat_func.cache_info()
        assert info["hits"] == 0
        assert info["misses"] == 1
        assert info["size"] == 1

        # Second call with same arg is a hit
        result2 = stat_func(5)
        assert result2 == 10

        info = stat_func.cache_info()
        assert info["hits"] == 1
        assert info["misses"] == 1
        assert info["size"] == 1

        # Third call with different arg is a miss
        result3 = stat_func(10)
        assert result3 == 20

        info = stat_func.cache_info()
        assert info["hits"] == 1
        assert info["misses"] == 2
        assert info["size"] == 2

        # Fourth call with first arg is another hit
        result4 = stat_func(5)
        assert result4 == 10

        info = stat_func.cache_info()
        assert info["hits"] == 2
        assert info["misses"] == 2
        assert info["size"] == 2

    def test_cache_info_per_function_isolation(self, tmp_path):
        """Test that cache_info() stats are isolated per decorated function."""
        cache_config = CacheConfig(cache_dir=tmp_path)
        cache_instance = cacheness(cache_config)

        @cached(cache_instance=cache_instance, key_prefix="func_x")
        def func_x(n):
            return n + 1

        @cached(cache_instance=cache_instance, key_prefix="func_y")
        def func_y(n):
            return n + 2

        # Call func_x multiple times
        func_x(1)  # miss
        func_x(1)  # hit
        func_x(1)  # hit

        # Call func_y once
        func_y(5)  # miss

        # Check that stats are independent
        info_x = func_x.cache_info()
        assert info_x["hits"] == 2
        assert info_x["misses"] == 1
        assert info_x["size"] == 1

        info_y = func_y.cache_info()
        assert info_y["hits"] == 0
        assert info_y["misses"] == 1
        assert info_y["size"] == 1

    def test_cache_info_includes_config_fields(self, tmp_path):
        """Test that cache_info() still returns original config fields."""
        cache_config = CacheConfig(cache_dir=tmp_path)

        @cached(
            cache_instance=cacheness(cache_config),
            ttl_seconds=7200,
            key_prefix="test_prefix",
            ignore_errors=False,
        )
        def config_func(x):
            return x * 3

        # Call once to populate stats
        config_func(1)

        info = config_func.cache_info()

        # Check runtime stats
        assert "hits" in info
        assert "misses" in info
        assert "size" in info

        # Check config fields are still present
        assert info["ttl_seconds"] == 7200
        assert info["key_prefix"] == "test_prefix"
        assert info["ignore_errors"] is False
        assert "cache_dir" in info
        assert "function" in info

    def test_cache_info_after_cache_clear(self, tmp_path):
        """Test that size updates correctly after cache_clear()."""
        cache_config = CacheConfig(cache_dir=tmp_path)

        @cached(cache_instance=cacheness(cache_config))
        def clear_stat_func(x):
            return x * 4

        # Create entries
        clear_stat_func(1)
        clear_stat_func(2)
        clear_stat_func(3)

        info_before = clear_stat_func.cache_info()
        assert info_before["size"] == 3
        assert info_before["misses"] == 3

        # Clear cache
        cleared = clear_stat_func.cache_clear()
        assert cleared == 3

        info_after = clear_stat_func.cache_info()
        assert info_after["size"] == 0
        # Hits and misses should be unchanged (they persist)
        assert info_after["hits"] == 0
        assert info_after["misses"] == 3


class TestCacheContext:
    """Test the CacheContext context manager."""

    def test_cache_context_basic(self):
        """Test basic CacheContext usage."""

        with CacheContext(default_ttl_seconds=3600) as ctx:

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

            cache_instance.close()

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


class TestFactoryMethods:
    """Test the factory method decorators."""

    def test_cached_for_api(self):
        """Test @cached.for_api() decorator."""
        # Use temporary directory to ensure clean cache state
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            call_count = 0

            @cached.for_api(ttl_seconds=3600, cache_dir=temp_dir)
            def fetch_api_data(endpoint):
                nonlocal call_count
                call_count += 1
                return {"endpoint": endpoint, "data": "response"}

            # First call should execute function
            result1 = fetch_api_data("users")
            assert result1 == {"endpoint": "users", "data": "response"}
            assert call_count == 1

            # Second call should hit cache
            result2 = fetch_api_data("users")
            assert result2 == {"endpoint": "users", "data": "response"}
            assert call_count == 1  # No additional calls

            # Different endpoint should execute function
            result3 = fetch_api_data("posts")
            assert result3 == {"endpoint": "posts", "data": "response"}
            assert call_count == 2

            # Close the cache instance created by the decorator
            fetch_api_data._cache_instance.close()

    def test_cached_for_api_error_handling(self):
        """Test that @cached.for_api() has error handling enabled by default."""
        # Use temporary directory to ensure clean cache state
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:

            @cached.for_api(cache_dir=temp_dir)
            def might_fail():
                return "success"

            # This should work normally
            result = might_fail()
            assert result == "success"

            # The decorator should have ignore_errors=True by default
            # We can't easily test cache errors without mocking, but we can verify
            # the decorator was created with the right parameters
            assert hasattr(might_fail, "cache_clear")
            assert hasattr(might_fail, "cache_info")

            # Close the cache instance created by the decorator
            might_fail._cache_instance.close()


class TestCacheIfDecorator:
    """Test the @cache_if conditional caching decorator."""

    def test_cache_if_only_caches_when_condition_true(self, tmp_path):
        """Test that @cache_if only caches results when condition returns True."""
        config = CacheConfig(cache_dir=tmp_path)
        cache_instance = cacheness(config)

        call_count = 0

        from cacheness.decorators import cache_if

        @cache_if(
            condition=lambda result: result is not None,
            cache_instance=cache_instance,
        )
        def fetch_data(should_succeed):
            nonlocal call_count
            call_count += 1
            return "data" if should_succeed else None

        # First call with successful result (should cache)
        result1 = fetch_data(True)
        assert result1 == "data"
        assert call_count == 1

        # Second call with same args (should hit cache)
        result2 = fetch_data(True)
        assert result2 == "data"
        assert call_count == 1  # Still 1 - cache was hit

        # Call with unsuccessful result (should NOT cache)
        result3 = fetch_data(False)
        assert result3 is None
        assert call_count == 2

        # Calling again with unsuccessful result (should execute again, not cached)
        result4 = fetch_data(False)
        assert result4 is None
        assert call_count == 3  # Incremented - not cached

        cache_instance.close()

    def test_cache_if_predicate_receives_result(self, tmp_path):
        """Test that the condition predicate receives the actual return value."""
        config = CacheConfig(cache_dir=tmp_path)
        cache_instance = cacheness(config)

        received_results = []

        from cacheness.decorators import cache_if

        def capture_condition(result):
            received_results.append(result)
            return result.get("status") == "success" if isinstance(result, dict) else False

        @cache_if(condition=capture_condition, cache_instance=cache_instance)
        def api_call(status_code):
            return {"status": "success" if status_code == 200 else "error", "code": status_code}

        # Call with success
        result1 = api_call(200)
        assert result1 == {"status": "success", "code": 200}
        assert len(received_results) == 1
        assert received_results[0] == {"status": "success", "code": 200}

        # Call with error
        result2 = api_call(500)
        assert result2 == {"status": "error", "code": 500}
        assert len(received_results) == 2
        assert received_results[1] == {"status": "error", "code": 500}

        cache_instance.close()

    def test_cache_if_supports_ttl_parameter(self, tmp_path):
        """Test that @cache_if supports ttl_seconds parameter like @cached."""
        config = CacheConfig(cache_dir=tmp_path)
        cache_instance = cacheness(config)

        call_count = 0

        from cacheness.decorators import cache_if

        @cache_if(
            condition=lambda result: result is not None,
            ttl_seconds=1,  # 1 second TTL
            cache_instance=cache_instance,
        )
        def fetch_data():
            nonlocal call_count
            call_count += 1
            return "data"

        # First call
        result1 = fetch_data()
        assert result1 == "data"
        assert call_count == 1

        # Immediate second call (should hit cache)
        result2 = fetch_data()
        assert result2 == "data"
        assert call_count == 1

        # Wait for TTL to expire
        time.sleep(1.1)

        # Third call after TTL expires (should execute again)
        result3 = fetch_data()
        assert result3 == "data"
        assert call_count == 2

        cache_instance.close()

    def test_cache_if_supports_key_prefix(self, tmp_path):
        """Test that @cache_if supports key_prefix parameter."""
        config = CacheConfig(cache_dir=tmp_path)
        cache_instance = cacheness(config)

        from cacheness.decorators import cache_if

        @cache_if(
            condition=lambda result: True,
            key_prefix="v1",
            cache_instance=cache_instance,
        )
        def api_v1():
            return "v1_data"

        @cache_if(
            condition=lambda result: True,
            key_prefix="v2",
            cache_instance=cache_instance,
        )
        def api_v1():  # Same function name, different prefix
            return "v2_data"

        # Both should work independently
        result1 = api_v1()
        assert "data" in result1

        cache_instance.close()

    def test_cache_if_cache_clear(self, tmp_path):
        """Test that cache_clear() works on @cache_if decorated functions."""
        config = CacheConfig(cache_dir=tmp_path)
        cache_instance = cacheness(config)

        call_count = 0

        from cacheness.decorators import cache_if

        @cache_if(
            condition=lambda result: result is not None,
            cache_instance=cache_instance,
        )
        def fetch_data(value):
            nonlocal call_count
            call_count += 1
            return value

        # Cache some results
        fetch_data(1)
        fetch_data(2)
        fetch_data(3)
        assert call_count == 3

        # Hit cache
        fetch_data(1)
        assert call_count == 3

        # Clear cache
        cleared = fetch_data.cache_clear()
        assert cleared == 3  # Should have cleared 3 entries

        # Should execute again after clear
        fetch_data(1)
        assert call_count == 4

        cache_instance.close()

    def test_cache_if_cache_info(self, tmp_path):
        """Test that cache_info() works on @cache_if decorated functions."""
        config = CacheConfig(cache_dir=tmp_path)
        cache_instance = cacheness(config)

        from cacheness.decorators import cache_if

        @cache_if(
            condition=lambda result: result is not None,
            ttl_seconds=3600,
            cache_instance=cache_instance,
        )
        def fetch_data(value):
            return value

        # Cache some results
        fetch_data(1)  # miss
        fetch_data(1)  # hit
        fetch_data(2)  # miss

        info = fetch_data.cache_info()
        assert info["hits"] == 1
        assert info["misses"] == 2
        assert info["size"] == 2
        assert info["ttl_seconds"] == 3600
        assert "cache_dir" in info

        cache_instance.close()

    def test_cache_if_condition_failure_doesnt_cache(self, tmp_path):
        """Test that if condition raises exception, result is not cached."""
        config = CacheConfig(cache_dir=tmp_path)
        cache_instance = cacheness(config)

        call_count = 0

        from cacheness.decorators import cache_if

        def failing_condition(result):
            if result > 10:
                raise ValueError("Too large!")
            return True

        @cache_if(
            condition=failing_condition,
            cache_instance=cache_instance,
            ignore_errors=True,  # Don't propagate condition errors
        )
        def compute(value):
            nonlocal call_count
            call_count += 1
            return value

        # Call with small value (condition passes)
        result1 = compute(5)
        assert result1 == 5
        assert call_count == 1

        # Second call (should hit cache)
        result2 = compute(5)
        assert result2 == 5
        assert call_count == 1

        # Call with large value (condition fails with exception)
        result3 = compute(15)
        assert result3 == 15
        assert call_count == 2

        # Call again with large value (should not be cached due to condition failure)
        result4 = compute(15)
        assert result4 == 15
        assert call_count == 3  # Executed again

        cache_instance.close()

    def test_cache_if_with_different_arg_combinations(self, tmp_path):
        """Test that @cache_if generates different keys for different arguments."""
        config = CacheConfig(cache_dir=tmp_path)
        cache_instance = cacheness(config)

        call_count = 0

        from cacheness.decorators import cache_if

        @cache_if(
            condition=lambda result: result is not None,
            cache_instance=cache_instance,
        )
        def fetch_user(user_id, include_details=False):
            nonlocal call_count
            call_count += 1
            return {"id": user_id, "details": include_details}

        # Different positional args
        fetch_user(1)
        fetch_user(2)
        assert call_count == 2

        # Different keyword args
        fetch_user(1, include_details=True)
        assert call_count == 3  # New variation

        # Cache hit for existing combination
        fetch_user(1)
        assert call_count == 3  # Still 3

        cache_instance.close()

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not installed")
    def test_cache_if_with_dataframe_condition(self, tmp_path):
        """Test @cache_if with DataFrame-specific conditions."""
        config = CacheConfig(cache_dir=tmp_path)
        cache_instance = cacheness(config)

        call_count = 0

        from cacheness.decorators import cache_if

        @cache_if(
            condition=lambda df: not df.empty,  # Only cache non-empty DataFrames
            cache_instance=cache_instance,
        )
        def load_data(rows):
            nonlocal call_count
            call_count += 1
            return pd.DataFrame({"col": list(range(rows))})

        # Cache non-empty DataFrame
        df1 = load_data(5)
        assert len(df1) == 5
        assert call_count == 1

        # Hit cache
        df2 = load_data(5)
        assert len(df2) == 5
        assert call_count == 1

        # Don't cache empty DataFrame
        df3 = load_data(0)
        assert len(df3) == 0
        assert call_count == 2

        # Call again with empty (not cached, executes again)
        df4 = load_data(0)
        assert len(df4) == 0
        assert call_count == 3

        cache_instance.close()
