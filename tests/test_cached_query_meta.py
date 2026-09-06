"""Public cached-metadata query capability regressions for Phase 3."""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap

import pytest

from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import CacheQueryValidationError
from cacheness.json_utils import dumps as json_dumps
from cacheness.metadata import CacheEntry, CachedMetadataBackend, SqliteBackend


@pytest.fixture
def cached_sqlite_cache(tmp_path):
    """Build the supported file-backed SQLite metadata wrapper."""
    cache = UnifiedCache(
        CacheConfig(
            cache_dir=str(tmp_path / "cached-sqlite"),
            metadata_backend="sqlite",
            enable_memory_cache=True,
            store_cache_key_params=True,
        )
    )
    assert isinstance(cache.metadata_backend, CachedMetadataBackend)
    try:
        yield cache
    finally:
        cache.close()


def test_cached_sqlite_query_meta_delegates_matching_and_live_keys(
    cached_sqlite_cache,
) -> None:
    """The wrapper preserves file-backed SQLite matching without stale rows."""
    cache = cached_sqlite_cache
    matching_key = cache.put(
        "matching",
        experiment="wrapped",
        score=1.5,
        active=True,
    )
    tombstoned_key = cache.put(
        "tombstoned",
        experiment="wrapped",
        score=3.0,
        active=False,
    )
    wrapped = cache.metadata_backend
    assert wrapped.supports_entry_metadata_query()
    assert not hasattr(wrapped, "SessionLocal")
    assert isinstance(wrapped.backend, SqliteBackend)

    with wrapped.backend.SessionLocal() as session:
        matching = session.get(CacheEntry, matching_key)
        assert matching is not None
        matching.cache_key_params = json_dumps(
            {
                "experiment": "str:wrapped",
                "score": "float:1.5",
                "active": "bool:True",
                "nested": {"region": "str:north"},
            }
        )
        session.commit()

    cache._cache_blob_store.delete(tombstoned_key)
    stale_entry = {
        "data_type": "object",
        "prefix": "",
        "description": "stale projection",
        "file_size": 1,
        "metadata": {
            "actual_path": str(cache.cache_dir / "stale.native"),
            "cache_key_params": {"experiment": "str:wrapped"},
        },
    }
    wrapped.backend.put_entry("stale-projection", stale_entry)

    direct = wrapped.backend.query_entries_by_key_params(
        {"experiment": "wrapped"}, live_keys={matching_key}
    )
    assert cache.query_meta(experiment="wrapped") == direct
    assert [entry["cache_key"] for entry in direct] == [matching_key]
    assert [entry["cache_key"] for entry in cache.query_meta(score=1.0)] == [
        matching_key
    ]
    assert [entry["cache_key"] for entry in cache.query_meta(active=True)] == [
        matching_key
    ]
    assert [
        entry["cache_key"] for entry in cache.query_meta(**{"nested.region": "north"})
    ] == [matching_key]


@pytest.mark.parametrize("backend", ("json", "memory", "sqlite_memory"))
def test_query_meta_keeps_unsupported_backend_warning_policy(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
    backend: str,
) -> None:
    """Unsupported metadata adapters retain the established public result."""
    cache = UnifiedCache(
        CacheConfig(
            cache_dir=str(tmp_path / backend),
            metadata_backend=backend,
            store_cache_key_params=True,
        )
    )
    try:
        with caplog.at_level(logging.WARNING):
            assert cache.query_meta(experiment="unsupported") is None
        assert "query_meta() requires SQLite backend" in caplog.text
    finally:
        cache.close()


def test_query_meta_disabled_key_parameter_storage_warns_after_capability(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Supported SQLite still keeps its established disabled-storage policy."""
    cache = UnifiedCache(
        CacheConfig(
            cache_dir=str(tmp_path / "disabled-params"),
            metadata_backend="sqlite",
            store_cache_key_params=False,
        )
    )
    try:
        with caplog.at_level(logging.WARNING):
            assert cache.query_meta(experiment="disabled") is None
        assert (
            "query_meta() requires store_cache_key_params=True in cache configuration"
            in caplog.text
        )
    finally:
        cache.close()


def test_sqlite_query_capability_revalidates_direct_filters(tmp_path) -> None:
    """Callers bypassing UnifiedCache still cannot derive unsafe JSON paths."""
    backend = SqliteBackend(str(tmp_path / "metadata.sqlite3"))
    try:
        with pytest.raises(CacheQueryValidationError):
            backend.query_entries_by_key_params(
                {"unsafe[0]": "blocked"}, live_keys=set()
            )
    finally:
        backend.close()


def test_query_meta_body_has_no_concrete_sql_session_dependency() -> None:
    """The facade asks a backend capability instead of owning SQL internals."""
    source = textwrap.dedent(inspect.getsource(UnifiedCache.query_meta))
    function = ast.parse(source).body[0]
    names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}

    assert {"SessionLocal", "engine", "CacheEntry", "SQLAlchemy"}.isdisjoint(names)

