"""Canonical authority ownership contracts for UnifiedCache composition."""

from __future__ import annotations

from pathlib import Path

from obstore.store import MemoryStore
import pytest

from cacheness.config import CacheConfig, CacheStorageConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import CacheBlobStoreClosedError
from cacheness.storage import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.guarded_handler_io import GuardedHandlerIO
from cacheness.storage.obstore_generation_io import ObstoreGenerationIO


def _topology() -> StoreTopology:
    """Return the explicit one-process topology used by authority ownership tests."""

    return StoreTopology(
        payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
    )


class _ObservedPayloadProvider(ObstoreGenerationIO):
    """Count ownership-driven closes of one injected/shared participant."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True)
        super().__init__(
            MemoryStore(), GuardedHandlerIO(root), qualification_identity="memory"
        )
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


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


def test_payload_provider_ownership_closes_only_the_selected_owner(tmp_path) -> None:
    """The cache/BlobStore boundary never double-closes or adopts a provider."""

    topology_owned = _ObservedPayloadProvider(tmp_path / "topology-owned")
    registry = _topology().role_registry
    registry.register(
        "payload",
        "memory",
        lambda: topology_owned,
        capabilities=topology_owned.topology_capabilities,
        replace=True,
    )
    store_owned = BlobStore(
        StoreTopology(
            payload=BackendRef(name="memory"),
            authority=BackendRef(name="memory"),
            role_registry=registry,
        ),
        cache_dir=tmp_path / "owned-store",
    )
    store_owned.put({"value": "owned"}, key="owned")
    store_owned.close()
    assert topology_owned.close_calls == 1

    caller_owned = _ObservedPayloadProvider(tmp_path / "caller-owned")
    caller_store = BlobStore(
        StoreTopology(
            payload=BackendRef(instance=caller_owned), authority=BackendRef(name="memory")
        ),
        cache_dir=tmp_path / "caller-store",
    )
    caller_store.put({"value": "caller"}, key="caller")
    caller_store.close()
    assert caller_owned.close_calls == 0
    caller_owned.close()


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
