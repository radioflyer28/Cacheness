"""
Validate that put() populates all 5 metadata columns correctly for every handler.

CACHE-198: Ensures storage_format, serializer, compression_codec, object_type,
and data_type are written to dedicated columns (not lost in nested metadata dicts).

Column semantics:
  data_type        -> which handler (e.g. "pandas_dataframe", "object")
  storage_format   -> file format (e.g. "parquet", "blosc2_array", "pickle")
  serializer       -> library within handler (e.g. "pickle", "dill"; NULL for parquet/blosc2)
  compression_codec -> compression algorithm (e.g. "lz4", "zstd", "snappy")
  object_type      -> Python type string (only for ObjectHandler)
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path

from cacheness.core import UnifiedCache, CacheConfig
from cacheness.config import SecurityConfig, CompressionConfig


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    d = tmp_path / "test_meta_columns"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_cache(cache_dir, **overrides):
    """Create a UnifiedCache with SQLite backend for column inspection.

    Accepts flat CacheConfig kwargs plus special keys:
      - enable_entry_signing -> SecurityConfig
      - use_blosc2_arrays / pickle_compression_codec / npz_compression /
        parquet_compression -> CompressionConfig
    """
    # Extract sub-config overrides
    signing = overrides.pop("enable_entry_signing", False)
    use_blosc2 = overrides.pop("use_blosc2_arrays", None)
    pickle_codec = overrides.pop("pickle_compression_codec", None)

    security = SecurityConfig(enable_entry_signing=signing)
    compression = CompressionConfig()
    if use_blosc2 is not None:
        compression.use_blosc2_arrays = use_blosc2
    if pickle_codec is not None:
        compression.pickle_compression_codec = pickle_codec

    defaults = dict(
        cache_dir=str(cache_dir),
        metadata_backend="sqlite",
        cleanup_on_init=False,
    )
    defaults.update(overrides)
    return UnifiedCache(
        CacheConfig(security=security, compression=compression, **defaults)
    )


def _get_entry_columns(cache, cache_key):
    """Extract the 5 metadata columns from a cache entry."""
    entries = cache.list_entries()
    for entry in entries:
        if entry.get("cache_key") == cache_key:
            meta = entry.get("metadata", {})
            return {
                "data_type": entry.get("data_type"),
                "storage_format": meta.get("storage_format"),
                "serializer": meta.get("serializer"),
                "compression_codec": meta.get("compression_codec"),
                "object_type": meta.get("object_type"),
            }
    pytest.fail(f"Entry {cache_key!r} not found in cache")


# ---------------------------------------------------------------------------
# Pandas DataFrame
# ---------------------------------------------------------------------------
class TestPandasDataFrameColumns:
    @pytest.fixture(autouse=True)
    def _skip_if_no_pandas(self):
        pytest.importorskip("pandas")

    def test_default_compression(self, cache_dir):
        import pandas as pd

        cache = _make_cache(cache_dir, parquet_compression="lz4")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        cache.put(df, hash_key="pd_df_lz4")
        cols = _get_entry_columns(
            cache, cache.get_metadata(hash_key="pd_df_lz4")["cache_key"]
        )

        assert cols["data_type"] == "pandas_dataframe"
        assert cols["storage_format"] == "parquet"
        assert cols["serializer"] is None
        assert cols["compression_codec"] == "lz4"
        assert cols["object_type"] is None
        cache.close()

    def test_snappy_compression(self, cache_dir):
        import pandas as pd

        cache = _make_cache(cache_dir, parquet_compression="snappy")
        df = pd.DataFrame({"x": [10]})
        cache.put(df, hash_key="pd_df_snappy")
        cols = _get_entry_columns(
            cache, cache.get_metadata(hash_key="pd_df_snappy")["cache_key"]
        )

        assert cols["data_type"] == "pandas_dataframe"
        assert cols["storage_format"] == "parquet"
        assert cols["compression_codec"] == "snappy"
        cache.close()


# ---------------------------------------------------------------------------
# Pandas Series
# ---------------------------------------------------------------------------
class TestPandasSeriesColumns:
    @pytest.fixture(autouse=True)
    def _skip_if_no_pandas(self):
        pytest.importorskip("pandas")

    def test_default(self, cache_dir):
        import pandas as pd

        cache = _make_cache(cache_dir, parquet_compression="lz4")
        s = pd.Series([1, 2, 3], name="vals")
        cache.put(s, hash_key="pd_series")
        cols = _get_entry_columns(
            cache, cache.get_metadata(hash_key="pd_series")["cache_key"]
        )

        assert cols["data_type"] == "pandas_series"
        assert cols["storage_format"] == "parquet"
        assert cols["serializer"] is None
        assert cols["compression_codec"] == "lz4"
        assert cols["object_type"] is None
        cache.close()


# ---------------------------------------------------------------------------
# Polars DataFrame
# ---------------------------------------------------------------------------
class TestPolarsDataFrameColumns:
    @pytest.fixture(autouse=True)
    def _skip_if_no_polars(self):
        pytest.importorskip("polars")

    def test_default_compression(self, cache_dir):
        import polars as pl

        cache = _make_cache(cache_dir, parquet_compression="lz4")
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        cache.put(df, hash_key="pl_df_lz4")
        cols = _get_entry_columns(
            cache, cache.get_metadata(hash_key="pl_df_lz4")["cache_key"]
        )

        assert cols["data_type"] == "polars_dataframe"
        assert cols["storage_format"] == "parquet"
        assert cols["serializer"] is None
        assert cols["compression_codec"] == "lz4"
        assert cols["object_type"] is None
        cache.close()


# ---------------------------------------------------------------------------
# Polars Series
# ---------------------------------------------------------------------------
class TestPolarsSeriesColumns:
    @pytest.fixture(autouse=True)
    def _skip_if_no_polars(self):
        pytest.importorskip("polars")

    def test_default(self, cache_dir):
        import polars as pl

        cache = _make_cache(cache_dir, parquet_compression="lz4")
        s = pl.Series("vals", [1, 2, 3])
        cache.put(s, hash_key="pl_series")
        cols = _get_entry_columns(
            cache, cache.get_metadata(hash_key="pl_series")["cache_key"]
        )

        assert cols["data_type"] == "polars_series"
        assert cols["storage_format"] == "parquet"
        assert cols["serializer"] is None
        assert cols["compression_codec"] == "lz4"
        assert cols["object_type"] is None
        cache.close()


# ---------------------------------------------------------------------------
# NumPy Array (blosc2)
# ---------------------------------------------------------------------------
class TestArrayBlosc2Columns:
    @pytest.fixture(autouse=True)
    def _skip_if_no_blosc2(self):
        pytest.importorskip("blosc2")

    def test_blosc2_array(self, cache_dir):
        cache = _make_cache(cache_dir, use_blosc2_arrays=True)
        arr = np.random.rand(100, 100)
        cache.put(arr, hash_key="np_blosc2")
        cols = _get_entry_columns(
            cache, cache.get_metadata(hash_key="np_blosc2")["cache_key"]
        )

        assert cols["data_type"] == "array"
        assert cols["storage_format"] == "blosc2_array"
        assert cols["serializer"] is None
        assert cols["compression_codec"] is not None  # default blosc2 codec
        assert cols["object_type"] is None
        cache.close()


# ---------------------------------------------------------------------------
# NumPy Array (npz fallback)
# ---------------------------------------------------------------------------
class TestArrayNpzColumns:
    def test_npz_compressed(self, cache_dir):
        cache = _make_cache(cache_dir, use_blosc2_arrays=False, npz_compression=True)
        arr = np.array([1, 2, 3])
        cache.put(arr, hash_key="np_npz")
        cols = _get_entry_columns(
            cache, cache.get_metadata(hash_key="np_npz")["cache_key"]
        )

        assert cols["data_type"] == "array"
        assert cols["storage_format"] == "npz"
        assert cols["serializer"] is None
        assert cols["compression_codec"] == "zlib"
        assert cols["object_type"] is None
        cache.close()

    def test_npz_uncompressed(self, cache_dir):
        cache = _make_cache(cache_dir, use_blosc2_arrays=False, npz_compression=False)
        arr = np.array([4, 5, 6])
        cache.put(arr, hash_key="np_npz_none")
        cols = _get_entry_columns(
            cache, cache.get_metadata(hash_key="np_npz_none")["cache_key"]
        )

        assert cols["data_type"] == "array"
        assert cols["storage_format"] == "npz"
        assert cols["compression_codec"] == "none"
        cache.close()


# ---------------------------------------------------------------------------
# Object (pickle)
# ---------------------------------------------------------------------------
class TestObjectPickleColumns:
    def test_pickle_uncompressed(self, cache_dir):
        # Small object below compression threshold -> uncompressed pickle
        cache = _make_cache(cache_dir, compression_threshold_bytes=999999)
        cache.put({"key": "val"}, hash_key="obj_pkl")
        cols = _get_entry_columns(
            cache, cache.get_metadata(hash_key="obj_pkl")["cache_key"]
        )

        assert cols["data_type"] == "object"
        assert cols["storage_format"] in ("pickle", "compressed_pickle")
        assert cols["serializer"] == "pickle"
        assert cols["object_type"] is not None
        assert "dict" in cols["object_type"]
        cache.close()

    def test_pickle_compressed(self, cache_dir):
        # Large object above compression threshold -> compressed pickle
        cache = _make_cache(
            cache_dir,
            compression_threshold_bytes=1,  # force compression
            pickle_compression_codec="zstd",
        )
        big_data = {"data": list(range(1000))}
        cache.put(big_data, hash_key="obj_pkl_zstd")
        cols = _get_entry_columns(
            cache, cache.get_metadata(hash_key="obj_pkl_zstd")["cache_key"]
        )

        assert cols["data_type"] == "object"
        assert cols["storage_format"] == "compressed_pickle"
        assert cols["serializer"] == "pickle"
        assert cols["compression_codec"] == "zstd"
        assert "dict" in cols["object_type"]
        cache.close()


# ---------------------------------------------------------------------------
# Object (dill)
# ---------------------------------------------------------------------------
class TestObjectDillColumns:
    @pytest.fixture(autouse=True)
    def _skip_if_no_dill(self):
        pytest.importorskip("dill")

    def test_dill_with_lambda(self, cache_dir):
        # Lambda forces dill serializer; use high threshold to skip blosc compression
        # (avoids blosc2 codec enum compatibility issues in test env)
        cache = _make_cache(
            cache_dir,
            compression_threshold_bytes=999_999_999,
        )
        fn = lambda x: x * 2  # noqa: E731
        cache.put(fn, hash_key="obj_dill")
        cols = _get_entry_columns(
            cache, cache.get_metadata(hash_key="obj_dill")["cache_key"]
        )

        assert cols["data_type"] == "object"
        assert cols["storage_format"] == "dill"
        assert cols["serializer"] == "dill"
        assert cols["object_type"] is not None
        assert "function" in cols["object_type"]
        cache.close()


# ---------------------------------------------------------------------------
# Backward compatibility: old "blosc2" entries still readable
# ---------------------------------------------------------------------------
class TestBackwardCompat:
    @pytest.fixture(autouse=True)
    def _skip_if_no_blosc2(self):
        pytest.importorskip("blosc2")

    def test_legacy_blosc2_storage_format_still_readable(self, cache_dir):
        """Entries written before the blosc2 -> blosc2_array rename should still load."""
        cache = _make_cache(cache_dir, use_blosc2_arrays=True)
        arr = np.random.rand(50, 50)
        cache.put(arr, hash_key="legacy_blosc2")

        # Tamper with metadata to simulate old "blosc2" value
        key = cache.get_metadata(hash_key="legacy_blosc2")["cache_key"]
        cache.metadata_backend.update_entry_metadata(
            cache_key=key,
            updates={"storage_format": "blosc2"},
        )

        # Should still be readable
        loaded = cache.get(hash_key="legacy_blosc2")
        assert loaded is not None
        np.testing.assert_array_equal(loaded, arr)
        cache.close()
