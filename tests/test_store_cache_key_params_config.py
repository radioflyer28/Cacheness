"""Cache-key parameter metadata through the canonical receipt/store boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from cacheness.config import (
    CacheConfig,
    CacheMetadataConfig,
    CacheStorageConfig,
    load_config_from_json,
    load_config_from_yaml,
    save_config_to_json,
    save_config_to_yaml,
)
from cacheness.core import UnifiedCache
from cacheness.storage import BackendRef, BlobStore, StoreTopology


def _memory_topology() -> StoreTopology:
    """Return the same-process topology used by cache observer tests."""

    return StoreTopology(
        payload=BackendRef(name="memory"),
        authority=BackendRef(name="memory"),
    )


def _cache_with_store(
    tmp_path, *, store_cache_key_params: bool
) -> tuple[UnifiedCache, BlobStore]:
    """Create an initialized caller-owned store and inject it into policy."""

    store = BlobStore(_memory_topology(), cache_dir=tmp_path / "store")
    store.initialize()
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_path / "cache")),
        metadata=CacheMetadataConfig(store_cache_key_params=store_cache_key_params),
    )
    cache = UnifiedCache(config, store=store)
    cache.initialize()
    return cache, store


def _committed_metadata(cache: UnifiedCache, store: BlobStore, **params: object) -> dict:
    """Inspect the exact committed entry through its immutable put receipt."""

    result = cache.put("test data", description="Test", **params)
    metadata = store.get_metadata(result.receipt.key)
    assert metadata is not None
    return metadata


@pytest.mark.parametrize("enabled", (False, True))
def test_nested_option_controls_committed_key_parameter_metadata(
    tmp_path, enabled: bool
) -> None:
    """Default and enabled policy are visible only on the committed store entry."""

    cache, store = _cache_with_store(tmp_path, store_cache_key_params=enabled)
    try:
        metadata = _committed_metadata(cache, store, model="gpt-4", temperature=0.7)

        assert cache.config.metadata.store_cache_key_params is enabled
        user_metadata = metadata["metadata"]
        if enabled:
            assert user_metadata["cache_key_params"] == {
                "model": "'gpt-4'",
                "temperature": "0.7",
            }
        else:
            assert "cache_key_params" not in user_metadata
    finally:
        cache.close()
        store.close()


def test_explicit_false_preserves_cache_operation_without_parameter_observer(tmp_path) -> None:
    """Disabling observer metadata does not change the committed cache value."""

    cache, store = _cache_with_store(tmp_path, store_cache_key_params=False)
    try:
        result = cache.put({"key": "value", "number": 42}, experiment="test")

        assert cache.lookup(cache_key=result.receipt.key, ttl_hours=None).value == {
            "key": "value",
            "number": 42,
        }
        assert "cache_key_params" not in store.get_metadata(result.receipt.key)["metadata"]
    finally:
        cache.close()
        store.close()


def test_key_parameter_metadata_survives_cache_instance_replacement(tmp_path) -> None:
    """A caller-owned store remains the observer source across cache instances."""

    store = BlobStore(_memory_topology(), cache_dir=tmp_path / "shared-store")
    store.initialize()
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_path / "cache")),
        metadata=CacheMetadataConfig(store_cache_key_params=True),
    )
    first = UnifiedCache(config, store=store)
    second = None
    try:
        first_result = first.put("first", model="one")
        first.close()

        second = UnifiedCache(config, store=store)
        second_result = second.put("second", model="two")

        assert store.get_metadata(first_result.receipt.key)["metadata"]["cache_key_params"] == {
            "model": "'one'"
        }
        assert store.get_metadata(second_result.receipt.key)["metadata"]["cache_key_params"] == {
            "model": "'two'"
        }
    finally:
        first.close()
        if second is not None:
            second.close()
        store.close()


def test_complex_parameter_values_are_observer_metadata_on_committed_entry(tmp_path) -> None:
    """Complex key inputs are represented by the policy's stable repr boundary."""

    cache, store = _cache_with_store(tmp_path, store_cache_key_params=True)
    try:
        metadata = _committed_metadata(
            cache,
            store,
            model_path=Path("/tmp/model.pkl"),
            config={"lr": 0.001, "epochs": 100},
            tags=["ml", "experiment"],
        )

        stored = metadata["metadata"]["cache_key_params"]
        assert stored["model_path"] == "PosixPath('/tmp/model.pkl')"
        assert stored["config"] == "{'lr': 0.001, 'epochs': 100}"
        assert stored["tags"] == "['ml', 'experiment']"
    finally:
        cache.close()
        store.close()


@pytest.mark.parametrize(
    ("suffix", "save", "load"),
    (
        ("json", save_config_to_json, load_config_from_json),
        ("yaml", save_config_to_yaml, load_config_from_yaml),
    ),
)
def test_nested_parameter_option_round_trips_in_supported_config_formats(
    tmp_path,
    suffix: str,
    save,
    load,
) -> None:
    """JSON/YAML retain the nested observer setting without flat aliases."""

    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_path / "configured-cache")),
        metadata=CacheMetadataConfig(store_cache_key_params=True),
    )
    path = tmp_path / f"cache-config.{suffix}"

    save(config, path)

    restored = load(path)
    assert restored.metadata.store_cache_key_params is True
    assert restored.storage.cache_dir == str(tmp_path / "configured-cache")


def test_flat_parameter_option_remains_an_intentional_non_delegation_check() -> None:
    """The cutover does not revive a flat configuration compatibility shim."""

    with pytest.raises(TypeError, match="store_cache_key_params"):
        CacheConfig(store_cache_key_params=True)
