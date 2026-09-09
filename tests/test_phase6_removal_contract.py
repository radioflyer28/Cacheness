"""Phase 6 contracts for bounded, exact cache-policy removal."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from cacheness.cache_policy import CacheOutcome, CacheRemovalReport
from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache
from cacheness.storage.catalog import CatalogPredicate, CatalogQuery
from cacheness.storage.composition import BackendRef, StoreTopology


def _memory_topology() -> StoreTopology:
    """Build the supported same-process topology used by removal contracts."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def _cache(tmp_path) -> UnifiedCache:
    """Create and initialize one explicit policy/store composition."""

    cache = UnifiedCache(CacheConfig(cache_dir=tmp_path), store=_memory_topology())
    cache.initialize()
    return cache


def test_expired_lookup_reports_the_exact_lifecycle_removal(tmp_path) -> None:
    """Expiry keeps its primary outcome while reporting a real exact delete."""

    cache = _cache(tmp_path)
    try:
        key = cache.put({"generation": "expired"}, request_id="expired")

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
        key = cache.put({"generation": "old"}, request_id="replacement")
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


def test_malformed_expiry_facts_fail_closed_without_deletion(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unauthenticated or malformed policy facts never authorize cleanup."""

    cache = _cache(tmp_path)
    try:
        key = cache.put({"generation": "protected"}, request_id="malformed")
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


def test_single_key_invalidation_returns_a_truthful_report(tmp_path) -> None:
    """Single-key removal carries its observed expectation into BlobStore."""

    cache = _cache(tmp_path)
    try:
        key = cache.put({"generation": "one"}, request_id="one")

        report = cache.invalidate(cache_key=key)

        assert report == CacheRemovalReport(attempted=1, removed=1)
        assert cache.lookup(cache_key=key).outcome is CacheOutcome.ABSENT
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


def test_predicate_removal_is_bounded_and_resumable_without_duplicate_success(
    tmp_path,
) -> None:
    """One invocation observes at most its cap and returns an opaque cursor."""

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

        resumed = cache.invalidate_where(
            query, cursor=first.continuation, page_size=1, work_cap=1
        )

        assert resumed.removed == 0
        assert resumed.complete is False
        assert resumed.retryable == 1
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

        with pytest.raises(ValueError, match="queryable"):
            cache.invalidate_where(
                CatalogQuery(
                    predicates=(CatalogPredicate("untrusted_field", "eq", "value"),)
                )
            )
    finally:
        cache.close()
