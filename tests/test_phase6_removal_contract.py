"""Phase 6 contracts for bounded, exact cache-policy removal."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from cacheness.cache_policy import CacheOutcome, CacheRemovalReport
from cacheness.config import CacheConfig, CacheStorageConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import CacheBlobBackendError
from cacheness.storage.catalog import (
    CatalogPredicate,
    CatalogQuery,
    CatalogQueryValidationError,
)
from cacheness.storage.composition import BackendRef, StoreTopology


def _memory_topology() -> StoreTopology:
    """Build the supported same-process topology used by removal contracts."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def _cache(tmp_path) -> UnifiedCache:
    """Create and initialize one explicit policy/store composition."""

    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path)),
        store=_memory_topology(),
    )
    cache.initialize()
    return cache


def test_expired_lookup_reports_the_exact_lifecycle_removal(tmp_path) -> None:
    """Expiry keeps its primary outcome while reporting a real exact delete."""

    cache = _cache(tmp_path)
    try:
        key = cache.put({"generation": "expired"}, request_id="expired").receipt.key

        result = cache.lookup(cache_key=key, ttl_hours=-1)

        assert result.outcome is CacheOutcome.EXPIRED
        assert isinstance(result.removal, CacheRemovalReport)
        assert result.removal.attempted == 1
        assert result.removal.removed == 1
        assert result.removal.conflicted == 0
        assert result.removal.failed == 0
        assert result.removal.complete is True
        assert cache.lookup(cache_key=key).outcome is CacheOutcome.ABSENT
    finally:
        cache.close()


def test_expired_lookup_preserves_a_replacement_and_reports_conflict(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selected generation cannot authorize deletion of its replacement."""

    cache = _cache(tmp_path)
    try:
        key = cache.put({"generation": "old"}, request_id="replacement").receipt.key
        replaced = False

        def replace_before_delete_promotion(boundary: str) -> None:
            nonlocal replaced
            if boundary == "delete.intent_prepared" and not replaced:
                replaced = True
                cache.put({"generation": "new"}, request_id="replacement")

        monkeypatch.setattr(
            cache.store.lifecycle,
            "test_hook",
            replace_before_delete_promotion,
        )

        result = cache.lookup(cache_key=key, ttl_hours=-1)

        assert result.outcome is CacheOutcome.EXPIRED
        assert result.removal is not None
        assert result.removal.attempted == 1
        assert result.removal.removed == 0
        assert result.removal.conflicted == 1
        assert result.removal.retryable == 1
        assert cache.lookup(cache_key=key, ttl_hours=None).value == {
            "generation": "new"
        }
    finally:
        cache.close()


def test_expired_cleanup_restarts_after_each_deleted_catalog_page(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TTL cleanup never returns a cursor invalidated by its own delete."""

    cache = _cache(tmp_path)
    try:
        keys = [
            cache.put({"generation": index}, request_id=f"expired-{index}").receipt.key
            for index in range(3)
        ]
        monkeypatch.setattr(cache, "_is_expired_at", lambda *_args: True)

        reports = []
        cursor = None
        for _ in range(3):
            report = cache._cleanup_expired(cursor=cursor, page_size=1, work_cap=1)
            reports.append(report)
            cursor = report.continuation

        assert [report.removed for report in reports] == [1, 1, 1]
        assert [report.complete for report in reports] == [False, False, True]
        assert all(
            report.continuation == "cache-policy:restart" for report in reports[:2]
        )
        assert reports[-1].continuation is None
        assert all(
            cache.lookup(cache_key=key).outcome is CacheOutcome.ABSENT for key in keys
        )
    finally:
        cache.close()


def test_malformed_expiry_facts_fail_closed_without_deletion(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unauthenticated or malformed policy facts never authorize cleanup."""

    cache = _cache(tmp_path)
    try:
        key = cache.put({"generation": "protected"}, request_id="malformed").receipt.key
        original_open = cache.store.open_entry
        delete_calls = 0

        @contextmanager
        def malformed_open_entry(cache_key: str):
            with original_open(cache_key) as entry:
                assert entry is not None
                metadata = dict(entry.metadata)
                metadata["file_size"] = "not-an-integer"
                entry.metadata = metadata
                yield entry

        def unexpected_delete(*_args, **_kwargs):
            nonlocal delete_calls
            delete_calls += 1
            raise AssertionError("malformed policy facts must not delete")

        monkeypatch.setattr(cache.store, "open_entry", malformed_open_entry)
        monkeypatch.setattr(cache.store, "delete", unexpected_delete)

        result = cache.lookup(cache_key=key, ttl_hours=-1)

        assert result.outcome is CacheOutcome.CORRUPT
        assert result.cause is not None
        assert delete_calls == 0
    finally:
        cache.close()


def test_single_key_invalidation_returns_a_truthful_report(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-key removal carries its observed expectation into BlobStore."""

    cache = _cache(tmp_path)
    try:
        key = cache.put({"generation": "one"}, request_id="one").receipt.key
        deleted: list[str] = []
        original_delete = cache.store.delete

        def observed_delete(cache_key: str, *args, **kwargs):
            deleted.append(cache_key)
            return original_delete(cache_key, *args, **kwargs)

        monkeypatch.setattr(cache.store, "delete", observed_delete)

        report = cache.invalidate(cache_key=key)

        assert report == CacheRemovalReport(attempted=1, removed=1)
        assert deleted == [key]
        assert cache.lookup(cache_key=key).outcome is CacheOutcome.ABSENT
    finally:
        cache.close()


def test_single_key_invalidation_preserves_backend_failure_details(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed lifecycle delete is not reported as a successful removal."""

    cache = _cache(tmp_path)
    try:
        key = cache.put(
            {"generation": "unavailable"}, request_id="unavailable"
        ).receipt.key
        failure = CacheBlobBackendError("authority unavailable")
        monkeypatch.setattr(
            cache.store,
            "delete",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        )

        report = cache.invalidate(cache_key=key)

        assert report.attempted == 1
        assert report.removed == 0
        assert report.conflicted == 0
        assert report.failed == 1
        assert report.failures[0].key == key
        assert report.failures[0].cause is failure
    finally:
        cache.close()


def test_empty_predicate_and_global_invalidation_report_complete_without_deletes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty selection is a successful zero-work policy result."""

    cache = _cache(tmp_path)
    try:
        delete_calls = 0
        original_delete = cache.store.delete

        def observed_delete(*args, **kwargs):
            nonlocal delete_calls
            delete_calls += 1
            return original_delete(*args, **kwargs)

        monkeypatch.setattr(cache.store, "delete", observed_delete)

        predicate_report = cache.invalidate_where(
            CatalogQuery(predicates=(CatalogPredicate("cache_prefix", "eq", "none"),))
        )
        global_report = cache.clear_all()

        assert predicate_report == CacheRemovalReport()
        assert global_report == CacheRemovalReport()
        assert delete_calls == 0
    finally:
        cache.close()


def test_predicate_removal_restarts_fresh_bounded_scans_until_complete(
    tmp_path,
) -> None:
    """Removal never reuses a cursor invalidated by its own exact deletion."""

    cache = _cache(tmp_path)
    try:
        for index in range(3):
            cache.put({"generation": index}, prefix="batch", request_id=index)

        query = CatalogQuery(
            predicates=(CatalogPredicate("cache_prefix", "eq", "batch"),),
            page_size=1,
        )
        first = cache.invalidate_where(query, page_size=1, work_cap=1)

        assert first.attempted == 1
        assert first.removed == 1
        assert first.complete is False
        assert isinstance(first.continuation, str)

        second = cache.invalidate_where(
            query, cursor=first.continuation, page_size=1, work_cap=1
        )
        third = cache.invalidate_where(
            query, cursor=second.continuation, page_size=1, work_cap=1
        )

        assert second == CacheRemovalReport(
            attempted=1,
            removed=1,
            complete=False,
            continuation=first.continuation,
        )
        assert third == CacheRemovalReport(attempted=1, removed=1)
        assert cache.invalidate_where(query, page_size=1, work_cap=1) == CacheRemovalReport()
    finally:
        cache.close()


def test_predicate_and_global_clear_delegate_exact_removal_to_blob_store(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Policy selection reaches payload removal only through ``BlobStore.delete``."""

    cache = _cache(tmp_path)
    try:
        predicate_key = cache.put(
            {"generation": "predicate"}, prefix="batch", request_id="predicate"
        ).receipt.key
        global_key = cache.put(
            {"generation": "global"}, prefix="other", request_id="global"
        ).receipt.key
        deleted: list[str] = []
        original_delete = cache.store.delete

        def observed_delete(cache_key: str, *args, **kwargs):
            deleted.append(cache_key)
            return original_delete(cache_key, *args, **kwargs)

        monkeypatch.setattr(cache.store, "delete", observed_delete)

        predicate = cache.invalidate_where(
            CatalogQuery(
                predicates=(CatalogPredicate("cache_prefix", "eq", "batch"),)
            )
        )
        global_clear = cache.clear_all()

        assert predicate == CacheRemovalReport(attempted=1, removed=1)
        assert global_clear == CacheRemovalReport(attempted=1, removed=1)
        assert deleted == [predicate_key, global_key]
    finally:
        cache.close()


def test_unsupported_predicate_fails_before_catalog_io(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public predicate boundary accepts only declared cache fields."""

    cache = _cache(tmp_path)
    try:
        def unexpected_query(*_args, **_kwargs):
            raise AssertionError("predicate validation must precede catalog I/O")

        monkeypatch.setattr(cache.store, "query_catalog", unexpected_query)

        with pytest.raises(CatalogQueryValidationError, match="queryable"):
            cache.invalidate_where(
                CatalogQuery(
                    predicates=(CatalogPredicate("untrusted_field", "eq", "value"),)
                )
            )
    finally:
        cache.close()
