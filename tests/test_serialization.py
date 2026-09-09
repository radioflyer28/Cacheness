"""Deterministic serialization and explicit decorator integration contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cacheness.config import CacheConfig, CacheStorageConfig
from cacheness.core import UnifiedCache
from cacheness.decorators import cached
from cacheness.serialization import create_unified_cache_key, serialize_for_cache_key
from cacheness.storage.composition import BackendRef, StoreTopology


@dataclass(frozen=True)
class _Value:
    """A small custom value used to cover object serialization."""

    label: str
    count: int


def _cache(tmp_path) -> UnifiedCache:
    """Create one caller-selected memory composition."""

    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path)),
        store=StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        ),
    )
    cache.initialize()
    return cache


def test_basic_collections_and_mapping_order_have_stable_key_evidence() -> None:
    """Canonical keys are deterministic for equal nested request parameters."""

    first = {"letters": ["a", "b"], "number": 1}
    second = {"number": 1, "letters": ["a", "b"]}

    assert serialize_for_cache_key(first) == serialize_for_cache_key(second)
    assert create_unified_cache_key(first) == create_unified_cache_key(second)


def test_numpy_and_custom_objects_have_repeatable_serialization() -> None:
    """Array and object key inputs retain deterministic content identity."""

    assert serialize_for_cache_key(np.array([1, 2, 3])) == serialize_for_cache_key(
        np.array([1, 2, 3])
    )
    assert serialize_for_cache_key(_Value("item", 2)) == serialize_for_cache_key(
        _Value("item", 2)
    )


def test_path_key_evidence_is_supported_without_the_retired_facade(tmp_path) -> None:
    """Path arguments are serialized through the same current key boundary."""

    path = Path(tmp_path) / "payload.txt"
    path.write_text("value", encoding="utf-8")

    assert create_unified_cache_key({"path": path}) == create_unified_cache_key(
        {"path": path}
    )


def test_explicit_cached_function_reuses_the_canonical_function_key(tmp_path) -> None:
    """The only decorator API shares storage and key policy with UnifiedCache."""

    cache = _cache(tmp_path)
    calls = 0

    @cached(cache=cache)
    def combine(left: int, right: int = 0) -> int:
        nonlocal calls
        calls += 1
        return left + right

    try:
        assert combine(2, right=3) == 5
        assert combine(2, right=3) == 5
        assert calls == 1
    finally:
        cache.close()
