"""
Function Caching Decorators
===========================

This module provides decorators and utilities for function-level caching
using the UnifiedCache system with automatic key generation.
"""

import functools
import xxhash
from typing import Any, Callable, Optional, Union, Dict, Tuple, cast

from .core import UnifiedCache, CacheConfig
from .serialization import serialize_for_cache_key


def _generate_cache_key(
    func: Callable,
    args: Tuple,
    kwargs: Dict[str, Any],
    key_prefix: Optional[str] = None,
    config: Optional[CacheConfig] = None,
) -> str:
    """
    Generate a cache key for a function call using unified serialization.

    Args:
        func: The function being cached
        args: Positional arguments passed to the function
        kwargs: Keyword arguments passed to the function
        key_prefix: Optional prefix for the cache key
        config: Optional CacheConfig to control serialization behavior

    Returns:
        A deterministic hash string for the function call
    """
    # Start with function identification
    func_name = getattr(func, "__qualname__", getattr(func, "__name__", "unknown"))
    func_module = getattr(func, "__module__", "unknown")
    func_id = f"{func_module}.{func_name}"

    # Serialize arguments using unified approach with config
    args_str = serialize_for_cache_key(args, config)
    kwargs_str = serialize_for_cache_key(kwargs, config)

    # Combine all components
    if key_prefix:
        cache_key_base = f"{key_prefix}:{func_id}:args:{args_str}:kwargs:{kwargs_str}"
    else:
        cache_key_base = f"{func_id}:args:{args_str}:kwargs:{kwargs_str}"

    # Hash for consistent length and characters
    return xxhash.xxh3_64(cache_key_base.encode()).hexdigest()


class cached:
    """
    Decorator for function-level caching using UnifiedCache.

    Examples:
        # Basic usage
        @cached()
        def expensive_function(x, y):
            return x * y

        # With TTL
        @cached(ttl_hours=6)
        def fetch_data():
            return requests.get("https://api.example.com").json()

        # With custom key prefix
        @cached(key_prefix="ml_model")
        def train_model(data):
            return expensive_ml_training(data)

        # With custom cache instance
        custom_cache = UnifiedCache(CacheConfig(cache_dir="./custom_cache"))

        @cached(cache_instance=custom_cache)
        def specialized_function():
            return complex_computation()

        # Custom key generation
        @cached(key_func=lambda func, args, kwargs: f"custom_{args[0]}")
        def user_specific_function(user_id, data):
            return process_user_data(user_id, data)
    """

    def __init__(
        self,
        ttl_hours: Optional[int] = None,
        key_prefix: Optional[str] = None,
        cache_instance: Optional[UnifiedCache] = None,
        key_func: Optional[Callable[[Callable, Tuple, Dict], str]] = None,
        ignore_errors: bool = True,
    ):
        """
        Initialize the caching decorator.

        Args:
            ttl_hours: Time-to-live in hours (uses cache default if None)
            key_prefix: Prefix for cache keys (useful for versioning)
            cache_instance: Specific cache instance to use (creates default if None)
            key_func: Custom function for generating cache keys
            ignore_errors: If True, cache errors don't prevent function execution
        """
        self.ttl_hours = ttl_hours
        self.key_prefix = key_prefix
        self.cache_instance = cache_instance
        self.key_func = key_func
        self.ignore_errors = ignore_errors

        # Create default cache instance if none provided
        if self.cache_instance is None:
            self.cache_instance = UnifiedCache()

    def __call__(self, func: Callable) -> Callable:
        """Apply the caching decorator to a function."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            try:
                if self.key_func:
                    cache_key = self.key_func(func, args, kwargs)
                else:
                    cache_key = _generate_cache_key(
                        func, args, kwargs, self.key_prefix, self.cache_instance.config
                    )
            except Exception as e:
                if self.ignore_errors:
                    # If key generation fails, just call the function
                    return func(*args, **kwargs)
                else:
                    raise RuntimeError(f"Cache key generation failed: {e}") from e

            # Try to get from cache using a synthetic parameter containing the cache key
            try:
                cache_instance = cast(UnifiedCache, self.cache_instance)
                cached_result = cache_instance.get(
                    ttl_hours=self.ttl_hours,
                    __decorator_cache_key=cache_key,  # Use synthetic parameter with the cache key
                )
                if cached_result is not None:
                    return cached_result
            except Exception as e:
                if not self.ignore_errors:
                    raise RuntimeError(f"Cache retrieval failed: {e}") from e
                # If cache retrieval fails but we're ignoring errors, continue to function call

            # Call the original function
            result = func(*args, **kwargs)

            # Store result in cache using the same synthetic parameter
            try:
                cache_instance = cast(UnifiedCache, self.cache_instance)
                cache_instance.put(
                    result,
                    description=f"Cached result for {cache_key}",
                    __decorator_cache_key=cache_key,  # Use same synthetic parameter
                )
            except Exception as e:
                if not self.ignore_errors:
                    raise RuntimeError(f"Cache storage failed: {e}") from e
                # If cache storage fails but we're ignoring errors, still return the result

            return result

        # Add cache management methods to the wrapped function
        def cache_clear():
            return self._clear_cache(func)

        def cache_info():
            return self._cache_info(func)

        def cache_key(*args, **kwargs):
            if self.key_func:
                return self.key_func(func, args, kwargs)
            else:
                return _generate_cache_key(
                    func, args, kwargs, self.key_prefix, self.cache_instance.config
                )

        # Attach methods (these will be available as wrapper.cache_clear(), etc.)
        setattr(wrapper, "cache_clear", cache_clear)
        setattr(wrapper, "cache_info", cache_info)
        setattr(wrapper, "cache_key", cache_key)

        return wrapper

    def _clear_cache(self, func: Callable) -> int:
        """Clear all cache entries for this function."""
        # This is a simplified implementation - in practice, you might want
        # to track function-specific keys or use a pattern-based deletion
        # For now, we'll return 0 as we don't have pattern-based deletion
        # This could be enhanced by keeping track of keys per function
        return 0

    def _cache_info(self, func: Callable) -> Dict[str, Any]:
        """Get cache information for this function."""
        func_name = getattr(func, "__qualname__", getattr(func, "__name__", "unknown"))
        func_module = getattr(func, "__module__", "unknown")
        cache_instance = cast(UnifiedCache, self.cache_instance)
        return {
            "function": f"{func_module}.{func_name}",
            "ttl_hours": self.ttl_hours,
            "key_prefix": self.key_prefix,
            "cache_dir": str(cache_instance.config.cache_dir),
            "ignore_errors": self.ignore_errors,
        }

    @classmethod
    def for_api(cls, ttl_hours: int = 6, ignore_errors: bool = True, **kwargs):
        """
        Decorator optimized for API requests.
        
        Defaults:
        - TTL: 6 hours (good for most API data)  
        - ignore_errors: True (don't fail if cache has issues)
        - Fast compression for JSON/text data
        
        Example:
            @cached.for_api(ttl_hours=4)
            def fetch_weather(city):
                return requests.get(f"api.weather.com/{city}").json()
        """
        from .core import UnifiedCache
        cache_instance = UnifiedCache.for_api(ttl_hours=ttl_hours, **kwargs)
        return cls(cache_instance=cache_instance, ignore_errors=ignore_errors)


def cache_function(
    func: Optional[Callable] = None, **kwargs
) -> Union[Callable, cached]:
    """
    Alternative function-based interface for caching.

    Can be used as a decorator or to wrap function calls.

    Examples:
        # As decorator
        @cache_function
        def my_func():
            return expensive_computation()

        # As decorator with arguments
        @cache_function(ttl_hours=12)
        def my_func():
            return expensive_computation()

        # Wrapping function calls
        cached_func = cache_function(expensive_function, ttl_hours=6)
        result = cached_func(arg1, arg2)
    """
    if func is None:
        # Called with arguments: @cache_function(ttl_hours=6)
        return cached(**kwargs)
    else:
        # Called without arguments: @cache_function
        return cached(**kwargs)(func)


def memoize(func: Callable) -> Callable:
    """
    Simple memoization decorator using the cache system.

    This is a convenience wrapper around @cached() with no TTL (permanent caching).

    Example:
        @memoize
        def fibonacci(n):
            if n < 2:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
    """
    return cached(ttl_hours=None)(func)


class CacheContext:
    """
    Context manager for temporary cache configuration.

    Useful for testing or temporary cache behavior changes.

    Example:
        with CacheContext(ttl_hours=1, key_prefix="test") as cache:
            @cache.cached()
            def temp_function():
                return "temporary result"
    """

    def __init__(self, **cache_config_kwargs):
        """Initialize with cache configuration overrides."""
        self.cache_config_kwargs = cache_config_kwargs
        self.cache_instance = None

    def __enter__(self) -> "CacheContext":
        """Enter the context and create cache instance."""
        config = CacheConfig(**self.cache_config_kwargs)
        self.cache_instance = UnifiedCache(config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        self.cache_instance = None

    def cached(self, **kwargs) -> cached:
        """Get a cached decorator using this context's cache instance."""
        return cached(cache_instance=self.cache_instance, **kwargs)
