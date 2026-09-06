"""Public cached-metadata query capability regressions for Phase 3."""

from __future__ import annotations

import ast
import inspect
import json
import logging
import sqlite3
import textwrap
from threading import Barrier, Thread

import pytest

from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import (
    CacheIntegrityError,
    CacheQueryValidationError,
    CacheReason,
)
from cacheness.json_utils import dumps as json_dumps
from cacheness.metadata import CacheEntry, CachedMetadataBackend, SqliteBackend
from cacheness.storage.manifest import (
    MAX_MANIFEST_BYTES,
    MAX_NESTING_DEPTH,
    MAX_STRING_UTF8_BYTES,
)


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
        matching_pair = (matching_key, matching.actual_path)
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
        {"experiment": "wrapped"}, live_pairs={matching_pair}
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


def test_public_query_meta_uses_fresh_read_only_pool_connections_under_writer(
    cached_sqlite_cache,
) -> None:
    """New pooled readers retain committed query visibility during a writer lock."""
    cache = cached_sqlite_cache
    cache.put("stable", cohort="read-wave", ordinal=1)
    expected = cache.query_meta(cohort="read-wave")
    assert isinstance(expected, list)
    assert expected

    backend = cache.metadata_backend.backend
    backend.engine.dispose()
    writer = sqlite3.connect(backend.db_file, isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")
    barrier = Barrier(7)
    results: list[list[dict]] = []
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(8):
                result = cache.query_meta(cohort="read-wave")
                assert result == expected
                results.append(result)
        except BaseException as error:  # pragma: no cover - asserted after joins.
            errors.append(error)

    threads = [Thread(target=reader) for _ in range(6)]
    try:
        for thread in threads:
            thread.start()
        barrier.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
    finally:
        writer.execute("ROLLBACK")
        writer.close()

    assert errors == []
    assert len(results) == 48
    assert backend.engine.pool.checkedout() == 0


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
                {"unsafe[0]": "blocked"}, live_pairs=set()
            )
    finally:
        backend.close()


def test_query_meta_body_has_no_concrete_sql_session_dependency() -> None:
    """The facade asks a backend capability instead of owning SQL internals."""
    source = textwrap.dedent(inspect.getsource(UnifiedCache.query_meta))
    function = ast.parse(source).body[0]
    names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}

    assert {"SessionLocal", "engine", "CacheEntry", "SQLAlchemy"}.isdisjoint(names)


@pytest.mark.parametrize("enable_memory_cache", (False, True))
@pytest.mark.parametrize(
    "filters",
    (
        {},
        {"experiment": "corrupt"},
        {"enabled": True},
        {"score": 1},
        {"nested.region": "north"},
    ),
)
def test_public_query_meta_rejects_malformed_live_parameter_evidence(
    tmp_path, enable_memory_cache: bool, filters: dict[str, object]
) -> None:
    """Exact live rows must fail closed before any empty or filtered query."""
    cache = UnifiedCache(
        CacheConfig(
            cache_dir=str(tmp_path / f"malformed-{enable_memory_cache}"),
            metadata_backend="sqlite",
            enable_memory_cache=enable_memory_cache,
            store_cache_key_params=True,
        )
    )
    try:
        cache_key = cache.put(
            "value", experiment="corrupt", enabled=True, score=1,
            nested={"region": "north"},
        )
        metadata_backend = cache.metadata_backend
        backend = (
            metadata_backend.backend
            if isinstance(metadata_backend, CachedMetadataBackend)
            else metadata_backend
        )
        assert isinstance(backend, SqliteBackend)
        with backend.SessionLocal() as session:
            entry = session.get(CacheEntry, cache_key)
            assert entry is not None
            entry.cache_key_params = "{"
            session.commit()

        with pytest.raises(CacheIntegrityError) as error:
            cache.query_meta(**filters)

        assert error.value.context == {
            "reason": CacheReason.METADATA_CORRUPT.value,
            "backend": "sqlite",
            "operation": "query_entries_by_key_params",
            "field": "cache_key_params",
            "cache_key": cache_key,
            "sqlite_type": "text",
            "byte_size": 1,
        }
        with sqlite3.connect(backend.db_file) as connection:
            assert connection.execute(
                "SELECT cache_key_params FROM cache_entries WHERE cache_key = ?",
                (cache_key,),
            ).fetchone() == ("{",)
        assert backend.engine.pool.checkedout() == 0
    finally:
        cache.close()


@pytest.mark.parametrize(
    ("expression", "parameters", "expected_type"),
    (
        ("NULL", (), "null"),
        # SQLite's declared TEXT affinity converts direct integer/real updates
        # to TEXT; their malformed textual representation is still rejected.
        ("1", (), "text"),
        ("1.5", (), "text"),
        ("CAST(? AS BLOB)", (b'{"experiment":"str:typed"}',), "blob"),
        ("CAST(X'FF' AS TEXT)", (), "text"),
        ("?", ("[",), "text"),
        ("?", ("[]",), "text"),
    ),
)
def test_public_query_meta_rejects_every_corrupt_live_storage_shape(
    tmp_path,
    expression: str,
    parameters: tuple[object, ...],
    expected_type: str,
) -> None:
    """Persisted parameter storage is exact SQLite TEXT containing one mapping."""
    cache = UnifiedCache(
        CacheConfig(
            cache_dir=str(tmp_path / "typed-corruption"),
            metadata_backend="sqlite",
            store_cache_key_params=True,
        )
    )
    try:
        cache_key = cache.put("value", experiment="typed")
        backend = cache.metadata_backend
        assert isinstance(backend, SqliteBackend)
        with sqlite3.connect(backend.db_file) as connection:
            connection.execute(
                f"UPDATE cache_entries SET cache_key_params = {expression} "
                "WHERE cache_key = ?",
                (*parameters, cache_key),
            )
            connection.commit()
            original = connection.execute(
                "SELECT typeof(cache_key_params), hex(cache_key_params) "
                "FROM cache_entries WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

        with pytest.raises(CacheIntegrityError) as error:
            cache.query_meta(experiment="typed")

        assert error.value.context["reason"] == CacheReason.METADATA_CORRUPT.value
        assert error.value.context["cache_key"] == cache_key
        assert error.value.context["sqlite_type"] == expected_type
        with sqlite3.connect(backend.db_file) as connection:
            assert connection.execute(
                "SELECT typeof(cache_key_params), hex(cache_key_params) "
                "FROM cache_entries WHERE cache_key = ?",
                (cache_key,),
            ).fetchone() == original
        assert backend.engine.pool.checkedout() == 0
    finally:
        cache.close()


@pytest.mark.parametrize(
    ("payload", "bound"),
    (
        (
            json.dumps(
                {"x": "a" * (MAX_STRING_UTF8_BYTES + 1)}, separators=(",", ":")
            ),
            "MAX_STRING_UTF8_BYTES",
        ),
        (
            '{"a":' * (MAX_NESTING_DEPTH + 1) + "0" + "}" * (MAX_NESTING_DEPTH + 1),
            "MAX_NESTING_DEPTH",
        ),
    ),
)
def test_public_query_meta_reports_canonical_parameter_bounds(
    tmp_path, payload: str, bound: str
) -> None:
    """Strict decoding reports the canonical bound without replacing evidence."""
    cache = UnifiedCache(
        CacheConfig(
            cache_dir=str(tmp_path / "bound-corruption"),
            metadata_backend="sqlite",
            store_cache_key_params=True,
        )
    )
    try:
        cache_key = cache.put("value", experiment="bound")
        backend = cache.metadata_backend
        assert isinstance(backend, SqliteBackend)
        with sqlite3.connect(backend.db_file) as connection:
            connection.execute(
                "UPDATE cache_entries SET cache_key_params = ? WHERE cache_key = ?",
                (payload, cache_key),
            )
            connection.commit()

        with pytest.raises(CacheIntegrityError) as error:
            cache.query_meta()

        assert error.value.context["bound"] == bound
        assert error.value.context["cache_key"] == cache_key
    finally:
        cache.close()


def test_public_query_meta_rejects_values_larger_than_manifest_byte_bound(tmp_path) -> None:
    """Byte size is checked before JSON parsing or an expensive structural walk."""
    cache = UnifiedCache(
        CacheConfig(
            cache_dir=str(tmp_path / "byte-bound"),
            metadata_backend="sqlite",
            store_cache_key_params=True,
        )
    )
    try:
        cache_key = cache.put("value", experiment="byte-bound")
        backend = cache.metadata_backend
        assert isinstance(backend, SqliteBackend)
        payload = "{" + '"x":"' + "a" * MAX_MANIFEST_BYTES + '"}'
        with sqlite3.connect(backend.db_file) as connection:
            connection.execute(
                "UPDATE cache_entries SET cache_key_params = ? WHERE cache_key = ?",
                (payload, cache_key),
            )
            connection.commit()

        with pytest.raises(CacheIntegrityError) as error:
            cache.query_meta()

        assert error.value.context["bound"] == "MAX_MANIFEST_BYTES"
        assert error.value.context["byte_size"] == len(payload.encode("utf-8"))
    finally:
        cache.close()


def test_stale_corrupt_projection_does_not_mask_an_exact_live_query(tmp_path) -> None:
    """Permissive observation hides stale corrupt JSON until authority selects rows."""
    cache = UnifiedCache(
        CacheConfig(
            cache_dir=str(tmp_path / "stale-corrupt"),
            metadata_backend="sqlite",
            store_cache_key_params=True,
        )
    )
    try:
        live_key = cache.put("live", experiment="live")
        backend = cache.metadata_backend
        assert isinstance(backend, SqliteBackend)
        backend.put_entry(
            "stale-key",
            {
                "data_type": "object",
                "metadata": {
                    "actual_path": str(tmp_path / "stale.native"),
                    "cache_key_params": {"experiment": "stale"},
                },
            },
        )
        with sqlite3.connect(backend.db_file) as connection:
            connection.execute(
                "UPDATE cache_entries SET cache_key_params = CAST(X'FF' AS TEXT) "
                "WHERE cache_key = 'stale-key'"
            )
            connection.commit()

        assert [entry["cache_key"] for entry in cache.query_meta(experiment="live")] == [
            live_key
        ]
        stale = next(
            entry for entry in backend.list_entries() if entry["cache_key"] == "stale-key"
        )
        assert "cache_key_params" not in stale["metadata"]
        assert backend.engine.pool.checkedout() == 0
    finally:
        cache.close()
