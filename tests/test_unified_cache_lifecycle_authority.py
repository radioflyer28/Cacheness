"""Canonical authority ownership contracts for UnifiedCache composition."""

from __future__ import annotations

import pytest

from cacheness.config import CacheConfig, CacheStorageConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import CacheBlobStoreClosedError
from cacheness.storage import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology


def _topology() -> StoreTopology:
    """Return the explicit one-process topology used by authority ownership tests."""

    return StoreTopology(
        payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
    )


def test_topology_form_creates_one_owned_blob_store(tmp_path) -> None:
    """A policy facade creates storage only from an explicit selected topology."""

    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path)), store=_topology()
    )
    try:
        cache.initialize()
        assert isinstance(cache.store, BlobStore)
        assert cache.store.lifecycle is cache.store._authority_lifecycle
    finally:
        cache.close()


def test_injected_store_remains_available_after_facade_close(tmp_path) -> None:
    """Caller-owned lifecycle resources are never closed by cache policy."""

    store = BlobStore(_topology(), cache_dir=tmp_path / "store")
    store.initialize()
    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path / "cache")),
        store=store,
    )
    try:
        cache.put({"value": "through-cache"}, request_id="cache")
        cache.close()

        with pytest.raises(CacheBlobStoreClosedError, match="Cache is closed"):
            cache.statistics()
        store.put({"value": "direct"}, key="direct")
        assert store.get("direct") == {"value": "direct"}
    finally:
        store.close()


def test_policy_replacement_keeps_the_latest_authoritative_generation(tmp_path) -> None:
    """Repeated policy writes share one key while BlobStore owns generation selection."""

    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path)), store=_topology()
    )
    try:
        cache.initialize()
        first = cache.put({"generation": 1}, request_id="replace")
        second = cache.put({"generation": 2}, request_id="replace")

        assert first.receipt.key == second.receipt.key
        assert cache.lookup(cache_key=second.receipt.key, ttl_hours=None).value == {
            "generation": 2
        }
    finally:
        cache.close()
