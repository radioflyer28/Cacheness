"""Key consistency across explicit cache and decorator policy entry points."""

from __future__ import annotations

from cacheness.config import CacheConfig, CacheStorageConfig
from cacheness.core import UnifiedCache
from cacheness.decorators import cached
from cacheness.serialization import create_unified_cache_key
from cacheness.storage.composition import BackendRef, StoreTopology


def _cache(tmp_path) -> UnifiedCache:
    """Build one current policy facade with its caller-selected topology."""

    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path)),
        store=StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        ),
    )
    cache.initialize()
    return cache


def test_public_cache_key_matches_the_shared_serializer(tmp_path) -> None:
    """Direct requests use the same deterministic serialization contract."""

    cache = _cache(tmp_path)
    try:
        values = {"nested": [1, 2], "label": "value"}
        assert cache._create_cache_key(values) == create_unified_cache_key(values, cache.config)
    finally:
        cache.close()


def test_decorator_reuses_one_explicit_cache_and_function_namespace(tmp_path) -> None:
    """Function values are retrieved through the same BlobStore-backed facade."""

    cache = _cache(tmp_path)
    calls = 0

    @cached(cache=cache)
    def render(value: int, *, scale: int = 1) -> int:
        nonlocal calls
        calls += 1
        return value * scale

    try:
        assert render(3, scale=2) == 6
        assert render(3, scale=2) == 6
        assert calls == 1
        assert render.cache_clear().removed == 1
    finally:
        cache.close()


def test_distinct_functions_do_not_share_function_call_keys(tmp_path) -> None:
    """Function namespace remains part of decorator policy identity."""

    cache = _cache(tmp_path)

    def first(value: int) -> int:
        return value

    def second(value: int) -> int:
        return value

    try:
        assert cache.function_namespace(first) != cache.function_namespace(second)
    finally:
        cache.close()
