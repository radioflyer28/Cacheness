"""Public custom-metadata contracts through the memory-cached SQL wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Column, String

from cacheness import CacheConfig, cacheness
from cacheness.custom_metadata import (
    CustomMetadataBase,
    _reset_registry,
    custom_metadata_model,
)
from cacheness.error_handling import CacheBlobLifecycleConflictError
from cacheness.metadata import Base, CachedMetadataBackend


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


def test_cached_sqlite_facade_preserves_live_custom_metadata_across_replacement(
    tmp_path: Path,
) -> None:
    """Public put/get/query/session paths retain only the M2-owned link."""

    @custom_metadata_model("cached_projection")
    class CachedProjectionMetadata(Base, CustomMetadataBase):
        __tablename__ = "custom_cached_projection_metadata"

        label = Column(String(100), nullable=False)

    cache = _cached_sqlite_cache(tmp_path / "cached-sqlite")
    key = "0123456789abcdef"
    try:
        cache.put({"generation": "m1"}, key=key, custom_metadata=CachedProjectionMetadata(label="m1"))
        assert cache.get_custom_metadata_for_entry(cache_key=key)["cached_projection"].label == "m1"
        assert [item.label for item in cache.query_custom("cached_projection")] == ["m1"]
        with cache.query_custom_session("cached_projection") as query:
            assert [item.label for item in query.all()] == ["m1"]

        cache.put({"generation": "m2"}, key=key, custom_metadata=CachedProjectionMetadata(label="m2"))

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
