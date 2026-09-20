"""Integrity outcomes retained by the canonical BlobStore-backed cache API."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from cacheness.cache_policy import CacheOutcome
from cacheness.config import CacheConfig, CacheMetadataConfig, CacheStorageConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import CacheBlobPayloadTamperedError
from cacheness.storage.composition import BackendRef, StoreTopology


def _cache(tmp_path, *, verify: bool = True) -> UnifiedCache:
    """Create one explicit composition with the requested integrity observer policy."""

    cache = UnifiedCache(
        CacheConfig(
            storage=CacheStorageConfig(cache_dir=tmp_path),
            metadata=CacheMetadataConfig(verify_cache_integrity=verify),
        ),
        store=StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        ),
    )
    cache.initialize()
    return cache


def test_integrity_verification_is_enabled_by_default(tmp_path) -> None:
    """The nested metadata policy retains the secure default explicitly."""

    cache = _cache(tmp_path)
    try:
        assert cache.config.metadata.verify_cache_integrity is True
        key = cache.put({"trusted": "payload"}, request_id="integrity").receipt.key
        assert cache.lookup(cache_key=key).outcome is CacheOutcome.HIT
    finally:
        cache.close()


def test_tampered_payload_evidence_becomes_a_typed_corrupt_outcome(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Policy does not reinterpret an integrity failure as a cache miss."""

    cache = _cache(tmp_path)
    error = CacheBlobPayloadTamperedError("payload digest does not match manifest")

    @contextmanager
    def tampered_open_entry(_key):
        raise error
        yield None

    monkeypatch.setattr(cache.store, "open_entry", tampered_open_entry)
    try:
        result = cache.lookup(cache_key="tampered")

        assert result.outcome is CacheOutcome.CORRUPT
        assert result.cause is error
    finally:
        cache.close()


def test_integrity_observer_policy_stays_separate_from_payload_authority(tmp_path) -> None:
    """Disabling the observer setting does not create an alternate lifecycle store."""

    cache = _cache(tmp_path, verify=False)
    try:
        assert cache.config.metadata.verify_cache_integrity is False
        assert cache.store is cache._cache_blob_store
        assert cache.store.topology.qualified_profile.pair == ("memory", "memory")
    finally:
        cache.close()
