"""Phase 8 contracts for public UnifiedCache policy boundaries."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace

import pytest

from cacheness.cache_policy import (
    CacheMaintenancePhase,
    CacheMaintenanceResult,
    CacheOutcome,
    CacheRemovalReport,
    CacheStatistics,
)
from cacheness.config import CacheConfig, CacheStorageConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobLifecycleTimeoutError,
    CacheBlobPayloadTamperedError,
)
from cacheness.storage.catalog import CatalogQuery, CatalogStaleCursorError
from cacheness.storage.composition import BackendRef, StoreTopology


def _phase8_cache(tmp_path) -> UnifiedCache:
    """Create an explicit same-process cache/store composition for policy tests."""

    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path)),
        store=StoreTopology(
            payload=BackendRef(name="memory"),
            authority=BackendRef(name="memory"),
        ),
    )
    cache.initialize()
    return cache


@contextmanager
def _open_entry_failure(cache: UnifiedCache, error: BaseException):
    """Make exactly one public storage observation raise a typed cause."""

    original_open_entry = cache.store.open_entry

    @contextmanager
    def failing_open_entry(_key: str):
        raise error
        yield None  # pragma: no cover - the preceding raise is the contract.

    cache.store.open_entry = failing_open_entry
    try:
        yield
    finally:
        cache.store.open_entry = original_open_entry


def test_cache_policy_preserves_declared_blobstore_lookup_cause(tmp_path) -> None:
    """A typed lifecycle conflict remains the public lookup cause and outcome."""

    cache = _phase8_cache(tmp_path)
    failure = CacheBlobLifecycleConflictError("exact generation changed")
    try:
        with _open_entry_failure(cache, failure):
            result = cache.lookup(cache_key="phase8-conflict")

        assert result.outcome is CacheOutcome.CONFLICT
        assert result.cause is failure
    finally:
        cache.close()


@pytest.mark.parametrize(
    ("failure", "outcome"),
    (
        (
            CacheBlobPayloadTamperedError("canonical bytes changed"),
            CacheOutcome.CORRUPT,
        ),
        (
            CacheBlobLifecycleConflictError("exact generation changed"),
            CacheOutcome.CONFLICT,
        ),
        (
            CacheBlobLifecycleTimeoutError("bounded progress exhausted"),
            CacheOutcome.BACKEND_ERROR,
        ),
        (CacheBlobBackendError("authority unavailable"), CacheOutcome.BACKEND_ERROR),
    ),
)
def test_cache_policy_maps_only_documented_lookup_failures(
    tmp_path, failure: BaseException, outcome: CacheOutcome
) -> None:
    """Each declared storage error maps once and keeps the original typed cause."""

    cache = _phase8_cache(tmp_path)
    try:
        with _open_entry_failure(cache, failure):
            result = cache.lookup(cache_key=f"phase8-{outcome.value}")

        assert result.outcome is outcome
        assert result.cause is failure
    finally:
        cache.close()


def test_cache_policy_rejects_undeclared_lookup_failures(tmp_path) -> None:
    """Policy does not relabel arbitrary implementation errors as cache misses."""

    cache = _phase8_cache(tmp_path)
    failure = RuntimeError("unclassified programming error")
    try:
        with _open_entry_failure(cache, failure):
            with pytest.raises(RuntimeError) as raised:
                cache.lookup(cache_key="phase8-undeclared")

        assert raised.value is failure
    finally:
        cache.close()


def test_cache_policy_invalidation_uses_exact_blobstore_result_accounting(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalidation forwards the observed generation expectation and reports its result."""

    cache = _phase8_cache(tmp_path)
    try:
        key = cache.put(
            {"generation": "current"}, request_id="phase8-exact"
        ).receipt.key
        snapshot = cache.store.get_entry_info(key)
        assert snapshot is not None
        observed_calls: list[tuple[str, object]] = []
        original_delete = cache.store.delete

        def observe_exact_delete(cache_key: str, *, expected: object) -> bool:
            observed_calls.append((cache_key, expected))
            return original_delete(cache_key, expected=expected)

        monkeypatch.setattr(cache.store, "delete", observe_exact_delete)
        report = cache.invalidate(cache_key=key)

        assert report == CacheRemovalReport(attempted=1, removed=1)
        assert observed_calls == [(key, snapshot.expectation)]
        assert cache.lookup(cache_key=key).outcome is CacheOutcome.ABSENT
    finally:
        cache.close()


def test_cache_policy_invalidation_preserves_conflict_and_backend_causes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conflict/retryability and backend failure remain distinct policy accounting."""

    cache = _phase8_cache(tmp_path)
    try:
        key = cache.put(
            {"generation": "current"}, request_id="phase8-outcomes"
        ).receipt.key
        monkeypatch.setattr(cache.store, "delete", lambda *_args, **_kwargs: False)

        conflict = cache.invalidate(cache_key=key)
        assert conflict == CacheRemovalReport(attempted=1, conflicted=1, retryable=1)

        failure = CacheBlobBackendError("authority unavailable")
        monkeypatch.setattr(
            cache.store,
            "delete",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )
        backend_failure = cache.invalidate(cache_key=key)
        assert backend_failure.attempted == 1
        assert backend_failure.removed == 0
        assert backend_failure.conflicted == 0
        assert backend_failure.failed == 1
        assert backend_failure.failures[0].key == key
        assert backend_failure.failures[0].cause is failure
    finally:
        cache.close()


def test_cache_policy_stale_continuation_restarts_through_the_bounded_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the documented restart token clears a stale revision cursor for a new page."""

    cache = _phase8_cache(tmp_path)
    try:
        key = cache.put(
            {"generation": "current"}, request_id="phase8-restart"
        ).receipt.key
        original_query = cache._query_cache_catalog
        cursors: list[str | None] = []

        def stale_then_fresh(query, *, cursor, page_size, work_cap):
            cursors.append(cursor)
            if cursor == "stale-phase8-cursor":
                raise CatalogStaleCursorError("cursor revision is stale")
            return original_query(
                query,
                cursor=cursor,
                page_size=page_size,
                work_cap=work_cap,
            )

        monkeypatch.setattr(cache, "_query_cache_catalog", stale_then_fresh)
        query = CatalogQuery(page_size=1)
        stale = cache.invalidate_where(
            query,
            cursor="stale-phase8-cursor",
            page_size=1,
            work_cap=1,
        )
        assert stale == CacheRemovalReport(
            retryable=1,
            complete=False,
            continuation="cache-policy:restart",
        )

        restarted = cache.invalidate_where(
            query,
            cursor=stale.continuation,
            page_size=1,
            work_cap=1,
        )
        assert restarted == CacheRemovalReport(attempted=1, removed=1)
        assert cursors == ["stale-phase8-cursor", None]
        assert cache.lookup(cache_key=key).outcome is CacheOutcome.ABSENT
    finally:
        cache.close()


def test_cache_policy_maintenance_results_are_immutable_and_validate_before_io(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed maintenance evidence is rejected before a policy resume inspects storage."""

    cache = _phase8_cache(tmp_path)
    try:
        state = cache._seal_maintenance_state(
            phase=CacheMaintenancePhase.INVENTORY,
            authority_revision=None,
            cursor=None,
        )
        malformed = replace(state, signature="0" * 64)

        def unexpected_storage(*_args, **_kwargs):
            raise AssertionError("invalid maintenance evidence reached storage")

        monkeypatch.setattr(cache, "_query_cache_catalog", unexpected_storage)
        with pytest.raises(ValueError, match="signature"):
            cache.resume_maintenance(malformed)

        with pytest.raises(ValueError, match="exactly one continuation or restart"):
            CacheMaintenanceResult(
                complete=False,
                retryable=True,
                removal=CacheRemovalReport(),
                state=state,
                cause=CacheBlobLifecycleConflictError("not a valid dual result"),
            )
    finally:
        cache.close()


def test_cache_policy_statistics_are_frozen_and_storage_free(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Statistics expose derived outcomes without catalog, payload, or delete calls."""

    cache = _phase8_cache(tmp_path)
    try:
        assert cache.lookup(cache_key="phase8-absent").outcome is CacheOutcome.ABSENT

        def unexpected_storage(*_args, **_kwargs):
            raise AssertionError("statistics must remain a storage-free observer")

        monkeypatch.setattr(cache.store, "get_entry_info", unexpected_storage)
        monkeypatch.setattr(cache.store, "query_catalog", unexpected_storage)
        monkeypatch.setattr(cache.store, "delete", unexpected_storage)

        statistics = cache.statistics()
        assert statistics == CacheStatistics(absent=1)
        with pytest.raises(FrozenInstanceError):
            statistics.absent = 2
    finally:
        cache.close()
