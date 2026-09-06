"""Public custom-metadata contracts through the memory-cached SQL wrapper."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest
from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import sessionmaker

from cacheness import CacheConfig, cacheness
from cacheness.config import SecurityConfig
from cacheness.custom_metadata import (
    CustomMetadataBase,
    _reset_registry,
    custom_metadata_model,
)
from cacheness.error_handling import CacheBlobLifecycleConflictError
from cacheness.error_handling import CacheStorageError
from cacheness.core import UnifiedCache
from cacheness.metadata import Base, CachedMetadataBackend
from cacheness.storage.backends.postgresql_backend import (
    PgCacheEntry,
    PgCacheStats,
    PostgresBackend,
    PostgresBase,
)


@pytest.fixture(autouse=True)
def reset_custom_metadata_registry():
    """Keep the globally registered SQLAlchemy test model isolated."""
    _reset_registry()
    yield
    _reset_registry()


def _cached_sqlite_cache(root: Path):
    """Construct the public facade with the metadata LRU layer enabled."""
    config = CacheConfig(
        cache_dir=str(root),
        metadata_backend="sqlite",
        enable_memory_cache=True,
        store_cache_key_params=True,
    )
    cache = cacheness(config)
    assert isinstance(cache.metadata_backend, CachedMetadataBackend)
    return cache


def _cached_postgresql_mapping() -> CachedMetadataBackend:
    """Build PostgreSQL behavior parity over a SQLite SQLAlchemy mapping.

    This is deliberately not a live PostgreSQL service qualification.  It
    proves the same facade-to-wrapper backend protocol without external setup.
    """
    backend = PostgresBackend.__new__(PostgresBackend)
    backend._lock = RLock()
    backend.engine = create_engine("sqlite://")
    backend.SessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=backend.engine
    )
    PostgresBase.metadata.create_all(backend.engine)
    Base.metadata.create_all(backend.engine)
    with backend.SessionLocal() as session:
        session.add(PgCacheStats(id=1))
        session.commit()
    return CachedMetadataBackend(
        backend,
        SimpleNamespace(
            enable_memory_cache=True,
            memory_cache_type="lru",
            memory_cache_maxsize=16,
            memory_cache_ttl_seconds=60,
            memory_cache_stats=False,
        ),
    )


def test_cached_sqlite_facade_preserves_live_custom_metadata_across_replacement(
    tmp_path: Path,
) -> None:
    """Public put/get/query/session paths retain only the M2-owned link."""

    @custom_metadata_model("cached_projection")
    class CachedProjectionMetadata(Base, CustomMetadataBase):
        __tablename__ = "custom_cached_projection_metadata"

        label = Column(String(100), nullable=False)

    cache = _cached_sqlite_cache(tmp_path / "cached-sqlite")
    try:
        key = cache.put(
            {"generation": "m1"},
            key="projection-key",
            custom_metadata=CachedProjectionMetadata(label="m1"),
        )
        assert cache.get_custom_metadata_for_entry(cache_key=key)["cached_projection"].label == "m1"
        assert [item.label for item in cache.query_custom("cached_projection")] == ["m1"]
        with cache.query_custom_session("cached_projection") as query:
            assert [item.label for item in query.all()] == ["m1"]

        assert cache.put(
            {"generation": "m2"},
            key="projection-key",
            custom_metadata=CachedProjectionMetadata(label="m2"),
        ) == key

        assert cache.get_custom_metadata_for_entry(cache_key=key)["cached_projection"].label == "m2"
        assert [item.label for item in cache.query_custom("cached_projection")] == ["m2"]
        with cache.query_custom_session("cached_projection") as query:
            assert [item.label for item in query.all()] == ["m2"]

        with pytest.raises(CacheBlobLifecycleConflictError):
            cache.metadata_backend.store_custom_metadata_if_current(
                key,
                "/stale/m1",
                [CachedProjectionMetadata(label="stale")],
            )
        assert [item.label for item in cache.query_custom("cached_projection")] == ["m2"]
    finally:
        cache.close()


def test_cached_postgresql_mapping_uses_the_same_public_custom_metadata_protocol() -> None:
    """Cached PostgreSQL parity retains exact-current M1/M2 custom links."""

    @custom_metadata_model("cached_postgresql_projection")
    class CachedPostgresqlProjectionMetadata(Base, CustomMetadataBase):
        __tablename__ = "custom_cached_postgresql_projection_metadata"

        label = Column(String(100), nullable=False)

    backend = _cached_postgresql_mapping()
    cache = object.__new__(UnifiedCache)
    cache.actual_backend = "postgresql"
    cache._custom_metadata_enabled = True
    cache.metadata_backend = backend
    cache._live_authority_projection_keys = lambda: {"0123456789abcdef"}
    cache._authority_snapshot_entry = lambda _key: (object(), object())
    key = "0123456789abcdef"
    try:
        assert backend.conditional_projection_mutation(
            key,
            expected_locator=None,
            replacement={
                "data_type": "object",
                "description": "m1",
                "file_size": 1,
                "metadata": {"actual_path": "/projection/m1"},
            },
        ).status == "applied"
        cache._store_custom_metadata(
            key,
            CachedPostgresqlProjectionMetadata(label="m1"),
            expected_locator="/projection/m1",
        )
        assert cache.get_custom_metadata_for_entry(cache_key=key)[
            "cached_postgresql_projection"
        ].label == "m1"

        assert backend.conditional_projection_mutation(
            key,
            expected_locator="/projection/m1",
            replacement={
                "data_type": "object",
                "description": "m2",
                "file_size": 1,
                "metadata": {"actual_path": "/projection/m2"},
            },
        ).status == "applied"
        cache._store_custom_metadata(
            key,
            CachedPostgresqlProjectionMetadata(label="m2"),
            expected_locator="/projection/m2",
        )
        assert [item.label for item in cache.query_custom("cached_postgresql_projection")] == [
            "m2"
        ]
        with pytest.raises(CacheBlobLifecycleConflictError):
            cache._store_custom_metadata(
                key,
                CachedPostgresqlProjectionMetadata(label="stale"),
                expected_locator="/projection/m1",
            )
    finally:
        backend.close()


def test_postgresql_cache_key_params_round_trip_at_the_signed_metadata_path() -> None:
    """PostgreSQL read-back preserves one decoded value at both API aliases."""
    cached = _cached_postgresql_mapping()
    backend = cached.backend
    key = "0123456789abcdef"
    try:
        backend.put_entry(
            key,
            {
                "data_type": "object",
                "description": "signed projection",
                "file_size": 1,
                "metadata": {
                    "actual_path": "/projection/signed",
                    "cache_key_params": {"run": "signed", "attempt": 2},
                },
            },
        )
        entry = backend.get_entry(key)
        assert entry is not None
        assert entry["metadata"]["cache_key_params"] == {
            "run": "signed",
            "attempt": 2,
        }
        assert entry["cache_key_params"] is entry["metadata"]["cache_key_params"]

        for malformed_value in ("{malformed", "[]", "null"):
            with backend.SessionLocal() as session:
                row = session.get(PgCacheEntry, key)
                assert row is not None
                row.cache_key_params = malformed_value
                session.commit()
            with pytest.raises(CacheStorageError):
                backend.get_entry(key)
            with backend.SessionLocal() as session:
                assert session.get(PgCacheEntry, key).cache_key_params == malformed_value
    finally:
        backend.close()


def test_signed_postgresql_cache_key_params_keep_a_valid_projection_live(
    tmp_path: Path,
) -> None:
    """Public get/list/stats keep a valid signed PostgreSQL projection committed."""
    signed_fields = [
        "cache_key",
        "data_type",
        "prefix",
        "file_size",
        "file_hash",
        "object_type",
        "storage_format",
        "serializer",
        "compression_codec",
        "actual_path",
        "created_at",
        "cache_key_params",
    ]
    cache = cacheness(
        CacheConfig(
            cache_dir=str(tmp_path / "signed-postgresql"),
            metadata_backend="sqlite",
            store_cache_key_params=True,
            security=SecurityConfig(
                enable_entry_signing=True,
                allow_unsigned_entries=False,
                delete_invalid_signatures=True,
                custom_signed_fields=signed_fields,
            ),
        )
    )
    original_backend = cache.metadata_backend
    cache.metadata_backend = _cached_postgresql_mapping()
    cache.actual_backend = "postgresql"
    try:
        key = cache.put({"value": "signed"}, run="signed", attempt=2)
        assert cache.get(cache_key=key) == {"value": "signed"}
        projection = cache.metadata_backend.get_entry(key)
        assert projection is not None
        assert projection["metadata"]["cache_key_params"] == {
            "run": "str:signed",
            "attempt": "int:2",
        }
        assert [entry["cache_key"] for entry in cache.list_entries()] == [key]
        assert cache.get_stats()["total_entries"] == 1
        assert cache._cache_blob_store.lifecycle_authority.read_entry(key) is not None
    finally:
        cache.close()
        original_backend.close()
