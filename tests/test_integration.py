"""End-to-end integration through the explicit UnifiedCache/BlobStore seam."""

from __future__ import annotations

from cacheness.cache_policy import CacheOutcome
from cacheness.config import CacheConfig, CachePolicyConfig, CacheStorageConfig
from cacheness.core import UnifiedCache
from cacheness.decorators import cached
from cacheness.storage.composition import BackendRef, StoreTopology


def _cache(tmp_path, *, byte_limit: int = 2_000 * 1024 * 1024) -> UnifiedCache:
    """Compose an initialized one-process cache for end-to-end behavior."""

    cache = UnifiedCache(
        CacheConfig(
            storage=CacheStorageConfig(cache_dir=tmp_path),
            policy=CachePolicyConfig(max_authoritative_bytes=byte_limit),
        ),
        store=StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        ),
    )
    cache.initialize()
    return cache


def test_put_lookup_and_exact_invalidation_form_one_policy_workflow(tmp_path) -> None:
    """A cache entry moves only through the canonical BlobStore lifecycle."""

    cache = _cache(tmp_path)
    try:
        stored = cache.put({"payload": [1, 2]}, request_id="workflow")

        assert cache.lookup(cache_key=stored.receipt.key).value == {"payload": [1, 2]}
        assert cache.invalidate(cache_key=stored.receipt.key).removed == 1
        assert cache.lookup(cache_key=stored.receipt.key).outcome is CacheOutcome.ABSENT
    finally:
        cache.close()


def test_explicit_decorator_reuses_the_same_policy_store(tmp_path) -> None:
    """Function caching is opt-in and never discovers a global cache facade."""

    cache = _cache(tmp_path)
    calls = 0

    @cached(cache=cache)
    def request(value: int) -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"value": value}

    try:
        assert request(4) == {"value": 4}
        assert request(4) == {"value": 4}
        assert calls == 1
        assert request.cache_clear().removed == 1
    finally:
        cache.close()


def test_one_bounded_maintenance_result_never_relabels_a_committed_receipt(tmp_path) -> None:
    """Policy pressure produces a result object while the write receipt stays truthful."""

    cache = _cache(tmp_path, byte_limit=0)
    try:
        result = cache.put({"payload": "x" * 256}, request_id="maintenance")

        assert result.receipt.key
        assert result.maintenance.complete is False
        assert cache.lookup(cache_key=result.receipt.key, ttl_hours=None).outcome is CacheOutcome.HIT
    finally:
        cache.close()
