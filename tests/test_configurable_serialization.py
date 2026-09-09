"""Config-driven key and handler behavior through explicit cache composition."""

from __future__ import annotations

import numpy as np
import pytest

from cacheness.config import (
    CacheConfig,
    CacheStorageConfig,
    HandlerConfig,
    SerializationConfig,
)
from cacheness.core import UnifiedCache
from cacheness.decorators import cached
from cacheness.handlers import HandlerRegistry
from cacheness.serialization import create_unified_cache_key, serialize_for_cache_key
from cacheness.storage.composition import BackendRef, StoreTopology


def _cache(tmp_path, config: CacheConfig) -> UnifiedCache:
    """Create the explicit memory topology used for policy-only tests."""

    cache = UnifiedCache(
        config,
        store=StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        ),
    )
    cache.initialize()
    return cache


def test_serialization_configuration_changes_key_evidence() -> None:
    """The stable serializer consumes the nested SerializationConfig."""

    shallow = CacheConfig(serialization=SerializationConfig(max_collection_depth=1))
    deep = CacheConfig(serialization=SerializationConfig(max_collection_depth=8))
    value = {"outer": {"inner": [1, 2, 3]}}

    assert serialize_for_cache_key(value, shallow) != serialize_for_cache_key(value, deep)
    assert create_unified_cache_key({"value": value}, shallow) == create_unified_cache_key(
        {"value": value}, shallow
    )


def test_handler_configuration_disables_array_and_object_fallbacks() -> None:
    """Registry availability derives from the configured direct-store policy."""

    config = CacheConfig(
        handlers=HandlerConfig(
            enable_numpy_arrays=False,
            enable_object_pickle=False,
            enable_pandas_dataframes=False,
            enable_polars_dataframes=False,
            enable_pandas_series=False,
            enable_polars_series=False,
        )
    )
    registry = HandlerRegistry(config)

    assert registry.handlers == []
    with pytest.raises(ValueError, match="No handler available"):
        registry.get_handler(np.array([1, 2]))


def test_handler_priority_is_preserved_by_the_configured_registry() -> None:
    """Priority is selected at composition time, not retrofitted by a facade."""

    config = CacheConfig(
        handlers=HandlerConfig(
            handler_priority=["object_pickle", "numpy_arrays"],
            enable_pandas_dataframes=False,
            enable_polars_dataframes=False,
            enable_pandas_series=False,
            enable_polars_series=False,
        )
    )

    assert [handler.data_type for handler in HandlerRegistry(config).handlers] == [
        "object",
        "array",
    ]


def test_explicit_decorator_uses_the_same_configured_key_boundary(tmp_path) -> None:
    """Decorators require a caller-owned UnifiedCache and reuse its key policy."""

    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=tmp_path),
        serialization=SerializationConfig(max_collection_depth=4),
    )
    cache = _cache(tmp_path, config)
    calls = 0

    @cached(cache=cache)
    def render(value: dict[str, int]) -> int:
        nonlocal calls
        calls += 1
        return value["item"]

    try:
        assert render({"item": 3}) == 3
        assert render({"item": 3}) == 3
        assert calls == 1
    finally:
        cache.close()
