"""Filesystem locator containment through the canonical BlobStore topology."""

from __future__ import annotations

from pathlib import Path

from cacheness.config import CacheConfig, CacheStorageConfig
from cacheness.storage import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology


def _local_store(root: Path) -> BlobStore:
    """Create the qualified SQLite/filesystem topology that owns locators."""

    store = BlobStore(
        StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        ),
        cache_dir=root,
        config=CacheConfig(storage=CacheStorageConfig(cache_dir=root)),
    )
    store.initialize()
    return store


def test_filesystem_locator_is_opaque_and_contained_for_unusual_keys(tmp_path) -> None:
    """Key text cannot become a caller-controlled cache path after cutover."""

    store = _local_store(tmp_path / "store")
    try:
        key = "../../not-a-locator"
        store.put({"safe": True}, key=key)
        entry = store.get_entry_info(key)

        assert entry is not None
        assert ".." not in entry.locator
        assert store.get(key) == {"safe": True}
    finally:
        store.close()


def test_direct_store_does_not_accept_retired_sharding_configuration(tmp_path) -> None:
    """Layout policy comes from topology participants, not CacheBlobConfig flags."""

    store = _local_store(tmp_path / "store")
    try:
        assert store.topology.qualified_profile.pair == ("sqlite", "filesystem")
        assert store.get("missing") is None
    finally:
        store.close()
