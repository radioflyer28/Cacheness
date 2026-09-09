"""Pandas handler coverage through the canonical explicit cache composition."""

from __future__ import annotations

import pytest

from cacheness.config import CacheConfig, CacheStorageConfig
from cacheness.core import UnifiedCache
from cacheness.storage.composition import BackendRef, StoreTopology


pd = pytest.importorskip("pandas")


def _cache(tmp_path) -> UnifiedCache:
    """Create a current cache facade with the optional DataFrame handler enabled."""

    cache = UnifiedCache(
        CacheConfig(storage=CacheStorageConfig(cache_dir=tmp_path)),
        store=StoreTopology(
            payload=BackendRef(name="memory"), authority=BackendRef(name="memory")
        ),
    )
    cache.initialize()
    return cache


def test_dataframe_round_trip_preserves_columns_index_and_dtypes(tmp_path) -> None:
    """A mixed DataFrame is selected and reconstructed by the pandas handler."""

    frame = pd.DataFrame(
        {
            "nullable": pd.array([1, None, 3], dtype="Int64"),
            "string": pd.array(["a", None, "c"], dtype="string"),
            "when": pd.date_range("2024-01-01", periods=3),
        },
        index=["first", "second", "third"],
    )
    cache = _cache(tmp_path)
    try:
        key = cache.put(frame, request_id="dataframe").receipt.key

        pd.testing.assert_frame_equal(cache.lookup(cache_key=key).value, frame)
    finally:
        cache.close()


def test_series_round_trip_preserves_name_and_values(tmp_path) -> None:
    """Pandas Series uses its dedicated canonical handler contract."""

    series = pd.Series([1, 2, 3], name="values", index=["a", "b", "c"])
    cache = _cache(tmp_path)
    try:
        key = cache.put(series, request_id="series").receipt.key

        pd.testing.assert_series_equal(cache.lookup(cache_key=key).value, series)
    finally:
        cache.close()
