"""Phase 6 contracts for presence-bearing UnifiedCache lookups."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from cacheness.cache_policy import CacheOutcome
from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobPayloadTamperedError,
)
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology


def _memory_topology() -> StoreTopology:
    """Build the explicit same-process topology used by the tracer."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def test_explicit_memory_composition_returns_hit_for_cached_none(tmp_path):
    """A stored None remains a presence-bearing cache hit."""

    cache = UnifiedCache(CacheConfig(cache_dir=tmp_path), store=_memory_topology())
    cache.initialize()
    cache_key = cache.put(None, request_id="cached-none")

    result = cache.lookup(cache_key=cache_key)

    assert result.outcome is CacheOutcome.HIT
    assert result.value is None
    cache.close()


@pytest.mark.parametrize("stored", [False, True], ids=["absent", "present-none"])
def test_lookup_observes_blob_store_once_for_absent_and_present_none(
    tmp_path, monkeypatch, stored
):
    """Presence comes from exactly one BlobStore.open_entry observation."""

    cache = UnifiedCache(CacheConfig(cache_dir=tmp_path), store=_memory_topology())
    cache.initialize()
    cache_key = "missing-key"
    if stored:
        cache_key = cache.put(None, request_id="present-none")

    calls = 0
    original_open_entry = cache.store.open_entry

    @contextmanager
    def observed_open_entry(key):
        nonlocal calls
        calls += 1
        with original_open_entry(key) as entry:
            yield entry

    monkeypatch.setattr(cache.store, "open_entry", observed_open_entry)

    result = cache.lookup(cache_key=cache_key)

    assert calls == 1
    assert result.outcome is CacheOutcome.HIT if stored else CacheOutcome.ABSENT
    assert result.value is None
    cache.close()


def test_existing_store_is_retained_and_topology_creates_one_store(tmp_path):
    """Cache composition has one explicit, observable BlobStore owner."""

    topology = _memory_topology()
    caller_store = BlobStore(topology, cache_dir=tmp_path / "caller-store")
    cache = UnifiedCache(CacheConfig(cache_dir=tmp_path), store=caller_store)

    assert cache.store is caller_store
    cache.close()
    assert caller_store.lifecycle_authority is not None

    topology_cache = UnifiedCache(
        CacheConfig(cache_dir=tmp_path / "owned-cache"), store=_memory_topology()
    )
    assert isinstance(topology_cache.store, BlobStore)
    assert topology_cache.store.lifecycle is topology_cache.store._authority_lifecycle
    topology_cache.close()


@pytest.mark.parametrize(
    ("error", "expected_outcome"),
    [
        (
            CacheBlobPayloadTamperedError("payload evidence is invalid"),
            CacheOutcome.CORRUPT,
        ),
        (
            CacheBlobLifecycleConflictError("generation changed"),
            CacheOutcome.CONFLICT,
        ),
        (
            CacheBlobBackendError("authority is unavailable"),
            CacheOutcome.BACKEND_ERROR,
        ),
    ],
)
def test_lookup_classifies_declared_storage_failures_without_cleanup(
    tmp_path, monkeypatch, error, expected_outcome
):
    """Declared direct-read failures retain their category and exact cause."""

    cache = UnifiedCache(CacheConfig(cache_dir=tmp_path), store=_memory_topology())
    cache.initialize()
    delete_calls = 0

    @contextmanager
    def failing_open_entry(_cache_key):
        raise error
        yield None

    def unexpected_delete(*_args, **_kwargs):
        nonlocal delete_calls
        delete_calls += 1
        raise AssertionError("lookup failure classification must not clean up evidence")

    monkeypatch.setattr(cache.store, "open_entry", failing_open_entry)
    monkeypatch.setattr(cache.store, "delete", unexpected_delete)

    result = cache.lookup(cache_key="failure-key")

    assert result.outcome is expected_outcome
    assert result.cause is error
    assert delete_calls == 0
    cache.close()


def test_lookup_propagates_unclassified_programming_errors(tmp_path, monkeypatch):
    """Lookup never converts arbitrary control-flow errors into cache outcomes."""

    cache = UnifiedCache(CacheConfig(cache_dir=tmp_path), store=_memory_topology())
    cache.initialize()

    @contextmanager
    def failing_open_entry(_cache_key):
        raise ValueError("programming error")
        yield None

    monkeypatch.setattr(cache.store, "open_entry", failing_open_entry)

    with pytest.raises(ValueError, match="programming error"):
        cache.lookup(cache_key="failure-key")

    cache.close()
