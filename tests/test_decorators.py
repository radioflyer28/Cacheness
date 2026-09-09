"""Regression tests for the canonical explicit-cache decorator."""

from __future__ import annotations

from cacheness import CacheConfig, cached
from cacheness.core import UnifiedCache
from cacheness.storage.composition import BackendRef, StoreTopology


def _cache(tmp_path) -> UnifiedCache:
    """Create one initialized caller-owned memory cache for a test."""

    cache = UnifiedCache(
        CacheConfig(cache_dir=tmp_path),
        store=StoreTopology(
            payload=BackendRef(name="memory"),
            authority=BackendRef(name="memory"),
        ),
    )
    cache.initialize()
    return cache


def test_explicit_decorator_caches_basic_function_results(tmp_path) -> None:
    """Repeated calls with the same arguments execute the user function once."""

    cache = _cache(tmp_path)
    calls = 0
    try:
        @cached(cache=cache)
        def multiply(left: int, right: int) -> int:
            nonlocal calls
            calls += 1
            return left * right

        assert multiply(5, 10) == 50
        assert multiply(5, 10) == 50
        assert multiply(3, 7) == 21
        assert calls == 2
    finally:
        cache.close()


def test_explicit_decorator_preserves_tuple_results(tmp_path) -> None:
    """Arbitrary serializable values remain valid cache hits."""

    cache = _cache(tmp_path)
    calls = 0
    try:
        @cached(cache=cache)
        def powers(value: int) -> tuple[int, int, int, str]:
            nonlocal calls
            calls += 1
            return value, value**2, value**3, f"processed_{value}"

        assert powers(5) == (5, 25, 125, "processed_5")
        assert powers(5) == (5, 25, 125, "processed_5")
        assert calls == 1
    finally:
        cache.close()


def test_explicit_decorator_preserves_function_metadata(tmp_path) -> None:
    """The thin facade still provides normal ``functools.wraps`` behavior."""

    cache = _cache(tmp_path)
    try:
        @cached(cache=cache)
        def documented(value: int) -> int:
            """Return the provided value."""

            return value

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "Return the provided value."
        assert documented.__wrapped__(3) == 3
    finally:
        cache.close()


def test_explicit_decorator_clear_returns_the_cache_removal_report(tmp_path) -> None:
    """Decorator cleanup exposes canonical bounded removal accounting."""

    cache = _cache(tmp_path)
    try:
        @cached(cache=cache)
        def value(identifier: int) -> str:
            return f"value:{identifier}"

        assert value(1) == "value:1"
        report = value.cache_clear()

        assert report.attempted == 1
        assert report.removed == 1
        assert report.complete is True
    finally:
        cache.close()
