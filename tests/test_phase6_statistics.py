"""Phase 6 contracts for immutable UnifiedCache outcome statistics."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError

import pytest

from cacheness.cache_policy import CacheOutcome, CacheStatistics
from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobPayloadTamperedError,
)
from cacheness.storage.composition import BackendRef, StoreTopology


def _memory_topology() -> StoreTopology:
    """Build the explicit same-process topology used by policy tests."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def _cache(tmp_path) -> UnifiedCache:
    """Create one initialized cache with no shared lifecycle state."""

    cache = UnifiedCache(CacheConfig(cache_dir=tmp_path), store=_memory_topology())
    cache.initialize()
    return cache


def _record_failure(cache, monkeypatch, error):
    """Perform one lookup whose sole storage observation raises ``error``."""

    @contextmanager
    def failing_open_entry(_cache_key):
        raise error
        yield None

    with monkeypatch.context() as patch:
        patch.setattr(cache.store, "open_entry", failing_open_entry)
        return cache.lookup(cache_key=f"failure-{type(error).__name__}")


def _record_outcome_schedule(cache, monkeypatch, outcomes):
    """Record each public lookup outcome through the canonical lookup boundary."""

    for outcome in outcomes:
        if outcome is CacheOutcome.ABSENT:
            assert cache.lookup(cache_key="absent-key").outcome is outcome
        elif outcome is CacheOutcome.HIT:
            key = cache.put("present", request_id="hit").receipt.key
            assert cache.lookup(cache_key=key).outcome is outcome
        elif outcome is CacheOutcome.EXPIRED:
            key = cache.put("expired", request_id="expired").receipt.key
            with monkeypatch.context() as patch:
                patch.setattr(cache, "_is_expired", lambda *_args, **_kwargs: True)
                assert cache.lookup(cache_key=key).outcome is outcome
        elif outcome is CacheOutcome.CORRUPT:
            assert _record_failure(
                cache,
                monkeypatch,
                CacheBlobPayloadTamperedError("payload evidence is invalid"),
            ).outcome is outcome
        elif outcome is CacheOutcome.CONFLICT:
            assert _record_failure(
                cache,
                monkeypatch,
                CacheBlobLifecycleConflictError("generation changed"),
            ).outcome is outcome
        elif outcome is CacheOutcome.BACKEND_ERROR:
            assert _record_failure(
                cache,
                monkeypatch,
                CacheBlobBackendError("authority is unavailable"),
            ).outcome is outcome
        else:  # pragma: no cover - the enum exhaustiveness contract is above.
            raise AssertionError(f"Unhandled outcome: {outcome}")


def test_statistics_snapshot_starts_at_zero_and_is_frozen(tmp_path):
    """A new observer reports all valid zero counts without storage I/O."""

    cache = _cache(tmp_path)

    statistics = cache.statistics()

    assert statistics == CacheStatistics()
    assert statistics.lookups == 0
    assert statistics.total_lookups == 0
    assert statistics.misses == 0
    assert statistics.hit_rate == 0.0
    with pytest.raises(FrozenInstanceError):
        statistics.hit = 1
    cache.close()


def test_statistics_count_every_outcome_once_and_are_order_independent(
    tmp_path, monkeypatch
):
    """The same lookup multiset yields one immutable aggregate in either order."""

    outcomes = tuple(CacheOutcome)
    first = _cache(tmp_path / "first")
    second = _cache(tmp_path / "second")
    try:
        _record_outcome_schedule(first, monkeypatch, outcomes)
        _record_outcome_schedule(second, monkeypatch, tuple(reversed(outcomes)))

        expected = CacheStatistics(
            hit=1,
            absent=1,
            expired=1,
            corrupt=1,
            conflict=1,
            backend_error=1,
        )
        assert first.statistics() == expected
        assert second.statistics() == expected
        assert expected.lookups == 6
        assert expected.total_lookups == 6
        assert expected.misses == 5
        assert expected.hit_rate == pytest.approx(1 / 6)
    finally:
        first.close()
        second.close()


def test_statistics_snapshot_never_observes_or_mutates_the_catalog(tmp_path, monkeypatch):
    """Statistics are a derived observer and cannot become lifecycle authority."""

    cache = _cache(tmp_path)

    def unexpected_catalog_work(*_args, **_kwargs):
        raise AssertionError("statistics must not inspect or mutate the catalog")

    monkeypatch.setattr(cache.store, "list", unexpected_catalog_work)
    monkeypatch.setattr(cache.store, "query_catalog", unexpected_catalog_work)
    monkeypatch.setattr(cache.store, "delete", unexpected_catalog_work)

    assert cache.statistics() == CacheStatistics()
    cache.close()
