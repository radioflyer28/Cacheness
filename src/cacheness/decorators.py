"""
Function Caching Decorators
===========================

This module provides decorators and utilities for function-level caching
using the UnifiedCache system with automatic key generation.
"""

import atexit
import functools
import threading
import weakref
from typing import Any, Callable, Optional, Union, Dict, Tuple, Set, cast

from .core import UnifiedCache, CacheConfig, _normalize_function_args
from .serialization import create_unified_cache_key

# Track decorator-created cache instances for cleanup
_decorator_cache_instances: list[weakref.ref[UnifiedCache]] = []


def _cleanup_decorator_caches():
    """Clean up all decorator-created cache instances on exit."""
    global _decorator_cache_instances
    for ref in _decorator_cache_instances:
        instance = ref()
        if instance is not None:
            try:
                instance.close()
            except Exception:
                pass  # Ignore errors during cleanup
    _decorator_cache_instances.clear()


# Register cleanup on interpreter exit
atexit.register(_cleanup_decorator_caches)


def _generate_cache_key(
    func: Callable,
    args: Tuple,
    kwargs: Dict[str, Any],
    key_prefix: Optional[str] = None,
    config: Optional[CacheConfig] = None,
) -> str:
    """
    Generate a cache key for a function call using unified serialization.

    This uses the same key generation system as UnifiedCache to ensure consistency.

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

    # Normalize arguments to consistent parameter mapping
    normalized_params = _normalize_function_args(func, args, kwargs)

    # Add function identification to parameters for uniqueness
    enhanced_params = {**normalized_params, "__function__": func_id}

    # Add key prefix if provided
    if key_prefix:
        enhanced_params["__key_prefix__"] = key_prefix

    # Use the same unified cache key generation as UnifiedCache
    return create_unified_cache_key(enhanced_params, config)


class cached:
    """
    Decorator for function-level caching using UnifiedCache.

    Examples:
        # Basic usage
        @cached()
        def expensive_function(x, y):
            return x * y

        # With TTL
        @cached(ttl_seconds=21600)
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
        ttl_seconds: Optional[float] = None,
        key_prefix: Optional[str] = None,
        cache_instance: Optional[UnifiedCache] = None,
        key_func: Optional[Callable[[Callable, Tuple, Dict], str]] = None,
        ignore_errors: bool = True,
    ):
        """
        Initialize the caching decorator.

        Args:
            ttl_seconds: Time-to-live in seconds (uses cache default if None).
                Pass None explicitly to never expire.
            key_prefix: Prefix for cache keys (useful for versioning)
            cache_instance: Specific cache instance to use (creates default if None)
            key_func: Custom function for generating cache keys
            ignore_errors: If True, cache errors don't prevent function execution
        """
        self.ttl_seconds = ttl_seconds

        self.key_prefix = key_prefix
        self.cache_instance = cache_instance
        self.key_func = key_func
        self.ignore_errors = ignore_errors
        self._owns_cache = False  # Track if we created the cache instance

        # Track cache keys created by this decorator for cache_clear()
        self._cache_keys: Set[str] = set()
        self._lock = threading.Lock()

        # Create default cache instance if none provided
        if self.cache_instance is None:
            self.cache_instance = UnifiedCache()
            self._owns_cache = True
            # Track for cleanup using weak reference
            _decorator_cache_instances.append(weakref.ref(self.cache_instance))

    def __call__(self, func: Callable) -> Callable:
        """Apply the caching decorator to a function."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            try:
                if self.key_func:
                    cache_key = self.key_func(func, args, kwargs)
                else:
                    cache_instance = cast(UnifiedCache, self.cache_instance)
                    cache_key = _generate_cache_key(
                        func, args, kwargs, self.key_prefix, cache_instance.config
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
                    ttl_seconds=self.ttl_seconds,
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
                # Track the cache key for later cleanup
                with self._lock:
                    self._cache_keys.add(cache_key)
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
                cache_instance = cast(UnifiedCache, self.cache_instance)
                return _generate_cache_key(
                    func, args, kwargs, self.key_prefix, cache_instance.config
                )

        # Attach methods (these will be available as wrapper.cache_clear(), etc.)
        setattr(wrapper, "cache_clear", cache_clear)
        setattr(wrapper, "cache_info", cache_info)
        setattr(wrapper, "cache_key", cache_key)
        setattr(wrapper, "_cache_instance", self.cache_instance)  # For cleanup in tests

        return wrapper

    def _clear_cache(self, func: Callable) -> int:
        """Clear all cache entries for this function."""
        cache_instance = cast(UnifiedCache, self.cache_instance)
        deleted = 0

        with self._lock:
            # Make a copy of the keys to avoid modification during iteration
            keys_to_delete = self._cache_keys.copy()

        # Delete each tracked cache key
        for cache_key in keys_to_delete:
            try:
                # Invalidate using the cache key stored in __decorator_cache_key
                cache_instance.invalidate(__decorator_cache_key=cache_key)
                with self._lock:
                    self._cache_keys.discard(cache_key)
                deleted += 1
            except Exception:
                # Key might have been deleted externally or expired
                with self._lock:
                    self._cache_keys.discard(cache_key)

        return deleted

    def _cache_info(self, func: Callable) -> Dict[str, Any]:
        """Get cache information for this function."""
        func_name = getattr(func, "__qualname__", getattr(func, "__name__", "unknown"))
        func_module = getattr(func, "__module__", "unknown")
        cache_instance = cast(UnifiedCache, self.cache_instance)
        return {
            "function": f"{func_module}.{func_name}",
            "ttl_seconds": self.ttl_seconds,
            "key_prefix": self.key_prefix,
            "cache_dir": str(cache_instance.config.cache_dir),
            "ignore_errors": self.ignore_errors,
        }

    def close(self):
        """
        Close the cache instance if this decorator owns it.

        Call this method to explicitly release resources when the decorated
        function is no longer needed, especially in long-running applications.
        """
        if self._owns_cache and self.cache_instance is not None:
            try:
                self.cache_instance.close()
            except Exception:
                pass  # Ignore errors during cleanup

    @classmethod
    def for_api(cls, ttl_seconds: float = 21600, ignore_errors: bool = True, **kwargs):
        """
        Decorator optimized for API requests.

        Defaults:
        - TTL: 6 hours (21600 seconds)
        - ignore_errors: True (don't fail if cache has issues)
        - Fast compression for JSON/text data

        Example:
            @cached.for_api(ttl_seconds=14400)  # 4 hours
            def fetch_weather(city):
                return requests.get(f"api.weather.com/{city}").json()
        """
        from .core import UnifiedCache

        cache_instance = UnifiedCache.for_api(ttl_seconds=ttl_seconds, **kwargs)
        # Track for cleanup using weak reference
        _decorator_cache_instances.append(weakref.ref(cache_instance))
        decorator = cls(cache_instance=cache_instance, ignore_errors=ignore_errors)
        decorator._owns_cache = True  # Mark as owned for explicit close()
        return decorator


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
        @cache_function(ttl_seconds=43200)  # 12 hours
        def my_func():
            return expensive_computation()

        # Wrapping function calls
        cached_func = cache_function(expensive_function, ttl_seconds=21600)  # 6 hours
        result = cached_func(arg1, arg2)
    """
    if func is None:
        # Called with arguments: @cache_function(ttl_seconds=21600)
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
    return cached(ttl_seconds=None)(func)


class CacheContext:
    """
    Context manager for temporary cache configuration.

    Useful for testing or temporary cache behavior changes.

    Example:
        with CacheContext(ttl_seconds=3600, key_prefix="test") as cache:
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
        if self.cache_instance is not None:
            self.cache_instance.close()
        self.cache_instance = None

    def cached(self, **kwargs) -> cached:
        """Get a cached decorator using this context's cache instance."""
        return cached(cache_instance=self.cache_instance, **kwargs)
