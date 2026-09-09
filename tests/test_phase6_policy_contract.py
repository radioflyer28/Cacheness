"""Phase 6 contracts for bounded, resumable cache size maintenance."""

from __future__ import annotations

import math

import pytest

from cacheness.cache_policy import (
    CacheMaintenancePhase,
    CacheMaintenanceResult,
    CachePutResult,
    CacheRemovalReport,
)
from cacheness.config import CacheConfig, CachePolicyConfig
from cacheness.core import _CACHE_NAMESPACE, _CACHE_POLICY_SCHEMA, UnifiedCache
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
)
from cacheness.storage import BlobReceipt
from cacheness.storage.catalog import CatalogQuery
from cacheness.storage.composition import BackendRef, StoreTopology


def _memory_topology() -> StoreTopology:
    """Build the supported same-process topology used by policy contracts."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def _cache(tmp_path, *, byte_limit: int = 0) -> UnifiedCache:
    """Create one initialized cache with deliberately tiny policy budgets."""

    config = CacheConfig(
        cache_dir=tmp_path,
        policy=CachePolicyConfig(
            max_authoritative_bytes=byte_limit,
            catalog_page_size=1,
            maintenance_work_cap=1,
        ),
    )
    cache = UnifiedCache(config, store=_memory_topology())
    cache.initialize()
    return cache


def _seed(cache: UnifiedCache, request_id: str, value: object) -> str:
    """Commit a cache entry without running the cache's put-policy seam."""

    key = cache._create_cache_key({"request_id": request_id})
    receipt = cache.store.put_entry(
        value,
        key=key,
        metadata={"prefix": "policy", "description": "size-contract"},
        catalog_schema=_CACHE_POLICY_SCHEMA,
        catalog_values={
            "cache_namespace": _CACHE_NAMESPACE,
            "cache_prefix": "policy",
        },
    )
    return receipt.key


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("catalog_page_size", 0),
        ("maintenance_work_cap", 0),
        ("maintenance_work_cap", math.inf),
        ("max_authoritative_bytes", -1),
    ),
)
def test_policy_configuration_rejects_non_finite_or_inconsistent_bounds(
    field: str, value: object
) -> None:
    """Policy limits reject invalid input before an authority can be queried."""

    with pytest.raises(ValueError, match=field):
        CachePolicyConfig(**{field: value})


def test_size_maintenance_is_page_bounded_and_requires_explicit_resumes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finite multi-page catalog converges only through caller-driven steps."""

    cache = _cache(tmp_path)
    try:
        for index in range(3):
            _seed(cache, str(index), {"payload": "x" * 128, "index": index})

        queries = deletes = 0
        original_query = cache.store.query_catalog
        original_delete = cache.store.delete

        def observed_query(*args, **kwargs):
            nonlocal queries
            queries += 1
            return original_query(*args, **kwargs)

        def observed_delete(*args, **kwargs):
            nonlocal deletes
            deletes += 1
            return original_delete(*args, **kwargs)

        monkeypatch.setattr(cache.store, "query_catalog", observed_query)
        monkeypatch.setattr(cache.store, "delete", observed_delete)

        result = cache.maintain_size()
        steps = 1
        assert isinstance(result, CacheMaintenanceResult)
        assert result.complete is False
        assert result.state is not None
        assert result.state.phase is CacheMaintenancePhase.INVENTORY

        while not result.complete:
            assert queries <= 1
            assert deletes <= 1
            assert result.state is not None
            queries = deletes = 0
            result = cache.resume_maintenance(result.state)
            steps += 1
            assert steps < 40

        assert steps > 3
        assert result.removal.complete is True
        assert cache.store.query_catalog(
            CatalogQuery(page_size=1),
            schema=_CACHE_POLICY_SCHEMA,
            limit=1,
            work_cap=1,
        ).entries == ()
    finally:
        cache.close()


def test_replacement_conflict_preserves_winner_and_returns_typed_retryable_result(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale exact candidate cannot delete a replacement generation."""

    cache = _cache(tmp_path)
    try:
        key = _seed(cache, "replacement", {"generation": "old", "x": "x" * 64})
        inventory = cache.maintain_size()
        assert inventory.state is not None
        candidate = cache.resume_maintenance(inventory.state)
        assert candidate.state is not None

        replaced = False

        def replace_before_delete(boundary: str) -> None:
            nonlocal replaced
            if boundary == "delete.intent_prepared" and not replaced:
                replaced = True
                _seed(cache, "replacement", {"generation": "new", "x": "x" * 64})

        monkeypatch.setattr(cache.store.lifecycle, "test_hook", replace_before_delete)
        result = cache.resume_maintenance(candidate.state)

        assert result.complete is False
        assert result.retryable is True
        assert isinstance(result.cause, CacheBlobLifecycleConflictError)
        assert cache.lookup(cache_key=key, ttl_hours=None).value["generation"] == "new"
    finally:
        cache.close()


def test_foreign_maintenance_state_is_rejected_before_authority_io(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A continuation capability cannot cross cache/store compositions."""

    first = _cache(tmp_path / "first")
    second = _cache(tmp_path / "second")
    try:
        _seed(first, "foreign", {"payload": "x" * 64})
        state = first.maintain_size().state
        assert state is not None

        def unexpected_io(*_args, **_kwargs):
            raise AssertionError("foreign state must fail before authority I/O")

        monkeypatch.setattr(second.store, "query_catalog", unexpected_io)
        monkeypatch.setattr(second.store, "delete", unexpected_io)

        with pytest.raises(ValueError, match="composition"):
            second.resume_maintenance(state)
    finally:
        first.close()
        second.close()


def test_put_retains_canonical_receipt_when_maintenance_is_incomplete(tmp_path) -> None:
    """A committed write remains inspectable while bounded work is pending."""

    cache = _cache(tmp_path, byte_limit=0)
    try:
        value = {"payload": "x" * 128}
        result = cache.put(value, request_id="post-commit-partial")

        assert isinstance(result, CachePutResult)
        assert isinstance(result.receipt, BlobReceipt)
        assert result.maintenance.complete is False
        assert result.maintenance.retryable is True
        assert result.maintenance.state is not None
        assert (
            cache.lookup(cache_key=result.receipt.key, ttl_hours=None).value == value
        )
    finally:
        cache.close()


def test_put_preserves_receipt_when_one_maintenance_step_reports_complete(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt and a complete policy report remain distinct immutable facts."""

    cache = _cache(tmp_path, byte_limit=1024 * 1024)
    complete = CacheMaintenanceResult(
        complete=True,
        retryable=False,
        removal=CacheRemovalReport(),
    )
    try:
        monkeypatch.setattr(cache, "maintain_size", lambda: complete)

        result = cache.put({"payload": "complete"}, request_id="post-commit-complete")

        assert isinstance(result, CachePutResult)
        assert isinstance(result.receipt, BlobReceipt)
        assert result.maintenance is complete
    finally:
        cache.close()


def test_put_preserves_receipt_when_maintenance_reports_backend_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-commit policy failure cannot falsify the BlobStore write result."""

    cache = _cache(tmp_path, byte_limit=0)
    try:
        def unavailable_catalog(*_args, **_kwargs):
            raise CacheBlobBackendError("catalog unavailable")

        monkeypatch.setattr(cache.store, "query_catalog", unavailable_catalog)

        result = cache.put(
            {"payload": "backend-failure"}, request_id="post-commit-failure"
        )

        assert isinstance(result, CachePutResult)
        assert isinstance(result.receipt, BlobReceipt)
        assert result.maintenance.complete is False
        assert result.maintenance.retryable is True
        assert isinstance(result.maintenance.cause, CacheBlobBackendError)
    finally:
        cache.close()


def test_failed_put_skips_size_maintenance(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed canonical write does not start policy work."""

    cache = _cache(tmp_path)
    maintenance_calls = 0
    try:
        def failed_put(*_args, **_kwargs):
            raise CacheBlobBackendError("canonical put failed")

        def unexpected_maintenance() -> CacheMaintenanceResult:
            nonlocal maintenance_calls
            maintenance_calls += 1
            raise AssertionError("maintenance must not run after a failed put")

        monkeypatch.setattr(cache.store, "put_entry", failed_put)
        monkeypatch.setattr(cache, "maintain_size", unexpected_maintenance)

        with pytest.raises(CacheBlobBackendError, match="canonical put failed"):
            cache.put({"payload": "never"}, request_id="pre-commit-failure")

        assert maintenance_calls == 0
    finally:
        cache.close()
