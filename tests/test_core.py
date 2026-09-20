"""Core UnifiedCache policy contracts after the BlobStore composition cutover."""

from __future__ import annotations

import numpy as np
import pytest

from cacheness.cache_policy import CacheOutcome
from cacheness.config import CacheConfig, CachePolicyConfig, CacheStorageConfig
from cacheness.core import UnifiedCache
from cacheness.storage.composition import BackendRef, StoreTopology


def _cache(tmp_path, *, ttl_hours: float | None = 24.0) -> UnifiedCache:
    """Create the explicit one-process topology for cache-policy behavior."""

    cache = UnifiedCache(
        CacheConfig(
            storage=CacheStorageConfig(cache_dir=tmp_path),
            policy=CachePolicyConfig(default_ttl_hours=ttl_hours),
        ),
        store=StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        ),
    )
    cache.initialize()
    return cache


def test_nested_config_is_retained_by_the_policy_facade(tmp_path) -> None:
    """UnifiedCache owns policy while its selected BlobStore owns lifecycle."""

    cache = _cache(tmp_path)
    try:
        assert cache.config.storage.cache_dir == tmp_path
        assert cache.store is cache._cache_blob_store
        assert cache.store.topology.qualified_profile.pair == ("memory", "memory")
    finally:
        cache.close()


def test_put_and_lookup_preserve_objects_and_cached_none(tmp_path) -> None:
    """Lookup exposes presence separately from a payload value of None."""

    cache = _cache(tmp_path)
    try:
        object_key = cache.put({"value": 1}, request_id="object").receipt.key
        none_key = cache.put(None, request_id="none").receipt.key

        assert cache.lookup(cache_key=object_key).value == {"value": 1}
        none_result = cache.lookup(cache_key=none_key)
        assert none_result.outcome is CacheOutcome.HIT
        assert none_result.value is None
    finally:
        cache.close()


def test_array_handler_round_trips_through_the_canonical_store(tmp_path) -> None:
    """Handler-selected payloads remain available through policy lookup results."""

    cache = _cache(tmp_path)
    try:
        array = np.array([1, 2, 3])
        key = cache.put(array, request_id="array").receipt.key

        assert np.array_equal(cache.lookup(cache_key=key).value, array)
    finally:
        cache.close()


def test_expiry_invalidation_and_statistics_use_typed_results(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Current policy reports outcomes and exact deletion without legacy metadata APIs."""

    cache = _cache(tmp_path)
    try:
        key = cache.put("value", request_id="expire").receipt.key
        deleted: list[str] = []
        original_delete = cache.store.delete

        def observed_delete(cache_key: str, *args, **kwargs):
            deleted.append(cache_key)
            return original_delete(cache_key, *args, **kwargs)

        monkeypatch.setattr(cache.store, "delete", observed_delete)

        expired = cache.lookup(cache_key=key, ttl_hours=-1)

        assert expired.outcome is CacheOutcome.EXPIRED
        assert expired.removal is not None
        assert expired.removal.removed == 1
        assert deleted == [key]
        assert cache.lookup(cache_key=key).outcome is CacheOutcome.ABSENT
        assert cache.statistics().expired == 1
        assert cache.statistics().absent == 1
    finally:
        cache.close()


def test_cache_key_derivation_is_deterministic_for_request_parameters(tmp_path) -> None:
    """The facade derives one stable identity before delegating to BlobStore."""

    cache = _cache(tmp_path)
    try:
        assert cache._create_cache_key({"left": 1, "right": 2}) == cache._create_cache_key(
            {"right": 2, "left": 1}
        )
    finally:
        cache.close()
