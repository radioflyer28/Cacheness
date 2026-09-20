"""Adversarial safety contracts for the cache-policy/BlobStore boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cacheness.config import CacheConfig, CacheStorageConfig
from cacheness.core import UnifiedCache
from cacheness.storage.composition import BackendRef, StoreTopology


def _cache(tmp_path) -> UnifiedCache:
    """Create one explicit in-process topology for deterministic safety checks."""

    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path)),
        store=StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        ),
    )
    cache.initialize()
    return cache


def test_policy_module_does_not_construct_or_import_the_payload_transport() -> None:
    """Only BlobStore is allowed to reach the selected payload participant."""

    module = ast.parse(Path("src/cacheness/core.py").read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "obstore" not in imported_modules


def test_replacement_after_an_invalidation_snapshot_is_not_deleted(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact expectations prevent a policy delete from retiring a newer generation."""

    cache = _cache(tmp_path)
    try:
        key = cache.put({"generation": "old"}, request_id="same").receipt.key
        replaced = False

        def replace_before_delete(boundary: str) -> None:
            nonlocal replaced
            if boundary == "delete.intent_prepared" and not replaced:
                replaced = True
                cache.put({"generation": "new"}, request_id="same")

        monkeypatch.setattr(cache.store.lifecycle, "test_hook", replace_before_delete)
        report = cache.invalidate(cache_key=key)

        assert report.removed == 0
        assert report.conflicted == 1
        assert cache.lookup(cache_key=key, ttl_hours=None).value == {"generation": "new"}
    finally:
        cache.close()


def test_clear_all_is_bounded_and_requires_explicit_continuation(tmp_path) -> None:
    """Clear policy exposes truthful bounded work instead of a hidden loop."""

    cache = _cache(tmp_path)
    try:
        for index in range(3):
            cache.put({"generation": index}, request_id=index)

        reports = []
        cursor = None
        while True:
            report = cache.clear_all(cursor=cursor, page_size=1, work_cap=1)
            reports.append(report)
            if report.complete:
                break
            cursor = report.continuation

        assert [report.removed for report in reports] == [1, 1, 1]
        assert reports[-1].complete is True
    finally:
        cache.close()
