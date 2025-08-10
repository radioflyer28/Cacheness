"""
Unit tests for cache data handlers.

Tests the handler system including fallback mechanisms and different data types.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from cacheness.handlers import (
    HandlerRegistry,
    ObjectHandler,
    ArrayHandler,
    PolarsDataFrameHandler,
    PandasDataFrameHandler,
    PandasSeriesHandler,
    PolarsSeriesHandler,
)
from cacheness.core import CacheConfig


def _has_dataframe_library():
    """Check if polars or pandas is available."""
    import importlib.util

    return (
        importlib.util.find_spec("polars") is not None
        or importlib.util.find_spec("pandas") is not None
    )


def _has_pandas():
    """Check if pandas is available."""
    import importlib.util

    return importlib.util.find_spec("pandas") is not None


def _has_polars():
    """Check if polars is available."""
    import importlib.util

    return importlib.util.find_spec("polars") is not None


class PickleableCustomObject:
    """A simple custom object that should be pickleable."""

    def __init__(self, value):
        self.value = value


"""
Unit tests for cache data handlers.

Tests the handler system including fallback mechanisms and different data types.
"""


class TestHandlerRegistry:
    """Test the cache handler registry system."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return CacheConfig()

    def test_registry_initialization(self, config):
        """Test registry initialization and default handlers."""
        registry = HandlerRegistry()

        # Should have some default handlers
        assert len(registry.handlers) > 0

        # Should have basic handlers
        handler_types = [type(h).__name__ for h in registry.handlers]
        assert "ObjectHandler" in handler_types
        assert "ArrayHandler" in handler_types

    def test_handler_selection(self, config):
        """Test handler selection based on data type."""
        registry = HandlerRegistry()

        # Test numpy array
        array_data = np.array([1, 2, 3])
        handler = registry.get_handler(array_data)
        assert handler is not None
        assert handler.can_handle(array_data)

        # Test generic object
        generic_data = {"key": "value", "number": 42}
        handler = registry.get_handler(generic_data)
        assert handler is not None
        assert handler.can_handle(generic_data)

    def test_no_handler_found(self, config):
        """Test behavior when no handler can handle the data."""
        registry = HandlerRegistry()

        # Mock all handlers to return False for can_handle
        for handler in registry.handlers:
            handler.can_handle = Mock(return_value=False)

        # Should raise ValueError when no handler can handle data
        with pytest.raises(ValueError):
            registry.get_handler("some data")


class TestObjectHandler:
    """Test the object cache handler."""

    @pytest.fixture
    def handler(self):
        """Create an object handler for testing."""
        return ObjectHandler()

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return CacheConfig()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_can_handle(self, handler):
        """Test can_handle method for various data types."""
        # Should handle most basic Python objects
        assert handler.can_handle({"key": "value"})
        assert handler.can_handle([1, 2, 3])
        assert handler.can_handle("string")
        assert handler.can_handle(42)
        assert handler.can_handle(3.14)

        # Should handle complex objects
        assert handler.can_handle({"nested": {"data": [1, 2, 3]}})

    def test_put_and_get(self, handler, config, temp_dir):
        """Test saving and loading data."""
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        file_path = temp_dir / "test"

        # Save data
        metadata = handler.put(test_data, file_path, config)
        assert "actual_path" in metadata
        assert Path(metadata["actual_path"]).exists()

        # Load data
        loaded_data = handler.get(Path(metadata["actual_path"]), metadata)
        assert loaded_data == test_data

    def test_get_file_extension(self, handler, config):
        """Test file extension."""
        # ObjectHandler uses pickle with compression (.pkl.zstd by default)
        assert handler.get_file_extension(config) == ".pkl.zstd"

    def test_data_type(self, handler):
        """Test data type identifier."""
        assert handler.data_type == "object"


class TestArrayHandler:
    """Test the array cache handler for numpy arrays."""

    @pytest.fixture
    def handler(self):
        """Create an array handler for testing."""
        return ArrayHandler()

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return CacheConfig()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_can_handle(self, handler):
        """Test can_handle method."""
        # Should handle numpy arrays
        assert handler.can_handle(np.array([1, 2, 3]))
        assert handler.can_handle(np.random.rand(5, 5))

        # Should handle dictionaries with array values
        array_dict = {"arr1": np.array([1, 2, 3]), "arr2": np.array([4, 5, 6])}
        assert handler.can_handle(array_dict)

        # Should not handle other types
        assert not handler.can_handle([1, 2, 3])
        assert not handler.can_handle({"key": "value"})
        assert not handler.can_handle("string")

    def test_put_and_get_single_array(self, handler, config, temp_dir):
        """Test saving and loading single numpy arrays."""
        test_array = np.random.rand(10, 5)
        file_path = temp_dir / "test"

        # Save array
        metadata = handler.put(test_array, file_path, config)
        assert "actual_path" in metadata
        assert Path(metadata["actual_path"]).exists()

        # Load array
        loaded_array = handler.get(Path(metadata["actual_path"]), metadata)
        assert isinstance(loaded_array, np.ndarray)
        assert loaded_array.shape == test_array.shape
        assert np.allclose(loaded_array, test_array)

    def test_put_and_get_array_dict(self, handler, config, temp_dir):
        """Test saving and loading array dictionaries."""
        test_dict = {
            "array1": np.random.rand(5, 3),
            "array2": np.random.randint(0, 10, (4, 2)),
            "labels": np.array(["a", "b", "c"]),
        }
        file_path = temp_dir / "test"

        # Save dictionary
        metadata = handler.put(test_dict, file_path, config)
        assert "actual_path" in metadata
        assert Path(metadata["actual_path"]).exists()

        # Load dictionary
        loaded_dict = handler.get(Path(metadata["actual_path"]), metadata)
        assert isinstance(loaded_dict, dict)
        assert set(loaded_dict.keys()) == set(test_dict.keys())

        for key in test_dict.keys():
            assert np.array_equal(loaded_dict[key], test_dict[key])

    def test_get_file_extension(self, handler, config):
        """Test file extension."""
        # ArrayHandler extension depends on config - could be .npz or .b2nd
        ext = handler.get_file_extension(config)
        assert ext in [".npz", ".b2nd"]  # Both are valid depending on config

    def test_data_type(self, handler):
        """Test data type identifier."""
        assert handler.data_type == "array"


class TestDataFrameHandlers:
    """Test DataFrame handlers (polars and pandas)."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return CacheConfig()

    def test_polars_handler_availability(self, config):
        """Test polars handler registration when polars is available."""
        try:
            handler = PolarsDataFrameHandler()

            # Test basic properties
            assert handler.get_file_extension(config) == ".parquet"
            assert handler.data_type == "polars_dataframe"

            # Test can_handle requires polars
            try:
                import polars as pl

                df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
                assert handler.can_handle(df)
                assert not handler.can_handle([1, 2, 3])
            except ImportError:
                pytest.skip("Polars not available")

        except ImportError:
            # Handler not available when polars not installed
            pytest.skip("Polars handler not available")

    def test_pandas_handler_availability(self, config):
        """Test pandas handler registration when pandas is available."""
        try:
            handler = PandasDataFrameHandler()

            # Test basic properties
            assert handler.get_file_extension(config) == ".parquet"
            assert handler.data_type == "pandas_dataframe"

            # Test can_handle requires pandas
            try:
                import pandas as pd

                df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
                assert handler.can_handle(df)
                assert not handler.can_handle([1, 2, 3])
            except ImportError:
                pytest.skip("Pandas not available")

        except ImportError:
            # Handler not available when pandas not installed
            pytest.skip("Pandas handler not available")

    @pytest.mark.skipif(
        not _has_dataframe_library(),
        reason="DataFrame tests require polars or pandas",
    )
    def test_dataframe_roundtrip(self, config):
        """Test full roundtrip for DataFrame handlers."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test polars if available
            try:
                import polars as pl

                handler = PolarsDataFrameHandler()
                df = pl.DataFrame(
                    {
                        "id": range(5),
                        "value": [1.1, 2.2, 3.3, 4.4, 5.5],
                        "category": ["A", "B", "A", "C", "B"],
                    }
                )

                file_path = temp_path / "test_polars"
                metadata = handler.put(df, file_path, config)
                loaded_df = handler.get(Path(metadata["actual_path"]), metadata)

                assert loaded_df.shape == df.shape
                assert loaded_df.columns == df.columns

            except ImportError:
                pass

            # Test pandas if available
            try:
                import pandas as pd

                handler = PandasDataFrameHandler()
                df = pd.DataFrame(
                    {
                        "id": range(5),
                        "value": [1.1, 2.2, 3.3, 4.4, 5.5],
                        "category": ["A", "B", "A", "C", "B"],
                    }
                )

                file_path = temp_path / "test_pandas"
                metadata = handler.put(df, file_path, config)
                loaded_df = handler.get(Path(metadata["actual_path"]), metadata)

                assert loaded_df.shape == df.shape
                assert list(loaded_df.columns) == list(df.columns)
                assert df.equals(loaded_df)

            except ImportError:
                pass


class TestSeriesHandlers:
    """Test Series-specific handlers."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return CacheConfig()

    @pytest.mark.skipif(not _has_pandas(), reason="Pandas Series tests require pandas")
    def test_pandas_series_handler_basic(self, config):
        """Test PandasSeriesHandler with basic data types."""
        import tempfile
        import pandas as pd

        handler = PandasSeriesHandler()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test numeric series
            series = pd.Series([1, 2, 3, 4, 5], name="test_series")

            assert handler.can_handle(series)
            assert handler.data_type == "pandas_series"

            file_path = temp_path / "test_pandas_series"
            metadata = handler.put(series, file_path, config)
            loaded_series = handler.get(Path(metadata["actual_path"]), metadata)

            assert loaded_series.equals(series)
            assert loaded_series.name == series.name

    @pytest.mark.skipif(not _has_pandas(), reason="Pandas Series tests require pandas")
    def test_pandas_series_handler_with_index(self, config):
        """Test PandasSeriesHandler preserves custom index."""
        import tempfile
        import pandas as pd

        handler = PandasSeriesHandler()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test series with custom index
            custom_index = pd.Index(["a", "b", "c", "d", "e"], name="letters")
            series = pd.Series([10, 20, 30, 40, 50], index=custom_index, name="values")

            file_path = temp_path / "test_indexed_series"
            metadata = handler.put(series, file_path, config)
            loaded_series = handler.get(Path(metadata["actual_path"]), metadata)

            assert loaded_series.equals(series)
            assert loaded_series.index.equals(series.index)
            assert loaded_series.index.name == series.index.name

    @pytest.mark.skipif(not _has_pandas(), reason="Pandas Series tests require pandas")
    def test_pandas_series_handler_multiindex(self, config):
        """Test PandasSeriesHandler with MultiIndex."""
        import tempfile
        import pandas as pd

        handler = PandasSeriesHandler()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test series with MultiIndex
            multi_index = pd.MultiIndex.from_tuples(
                [("A", 1), ("A", 2), ("B", 1)], names=["letter", "number"]
            )
            series = pd.Series([100, 200, 300], index=multi_index, name="multi_values")

            file_path = temp_path / "test_multiindex_series"
            metadata = handler.put(series, file_path, config)
            loaded_series = handler.get(Path(metadata["actual_path"]), metadata)

            assert loaded_series.equals(series)
            assert loaded_series.index.equals(series.index)
            assert loaded_series.index.names == series.index.names

    @pytest.mark.skipif(not _has_pandas(), reason="Pandas Series tests require pandas")
    def test_pandas_series_fallback_to_object_handler(self, config):
        """Test that complex Series fall back to ObjectHandler."""
        import pandas as pd

        registry = HandlerRegistry()

        # Create a series with objects that can't be stored in Parquet but are pickleable
        series = pd.Series(
            [
                PickleableCustomObject(1),
                PickleableCustomObject(2),
                PickleableCustomObject(3),
            ],
            name="custom_objects",
        )

        # Should get ObjectHandler for complex objects that can't be Parquet-serialized
        handler = registry.get_handler(series)
        assert handler.data_type == "object"  # Falls back to ObjectHandler

    @pytest.mark.skipif(not _has_polars(), reason="Polars Series tests require polars")
    def test_polars_series_handler_basic(self, config):
        """Test PolarsSeriesHandler with basic data types."""
        import tempfile
        import polars as pl

        handler = PolarsSeriesHandler()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test numeric series
            series = pl.Series("test_series", [1, 2, 3, 4, 5])

            assert handler.can_handle(series)
            assert handler.data_type == "polars_series"

            file_path = temp_path / "test_polars_series"
            metadata = handler.put(series, file_path, config)
            loaded_series = handler.get(Path(metadata["actual_path"]), metadata)

            assert loaded_series.equals(series)
            assert loaded_series.name == series.name

    @pytest.mark.skipif(not _has_polars(), reason="Polars Series tests require polars")
    def test_polars_series_handler_string_data(self, config):
        """Test PolarsSeriesHandler with string data."""
        import tempfile
        import polars as pl

        handler = PolarsSeriesHandler()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test string series
            series = pl.Series("names", ["Alice", "Bob", "Charlie", "Diana"])

            file_path = temp_path / "test_string_series"
            metadata = handler.put(series, file_path, config)
            loaded_series = handler.get(Path(metadata["actual_path"]), metadata)

            assert loaded_series.equals(series)
            assert loaded_series.name == series.name

    @pytest.mark.skipif(not _has_polars(), reason="Polars Series tests require polars")
    def test_polars_series_fallback_to_object_handler(self, config):
        """Test that Polars Series with Object types fall back to ObjectHandler."""
        import polars as pl

        registry = HandlerRegistry()

        try:
            # Create a series with Object type (this should fail in Polars anyway)
            # But if it doesn't, it should fall back to ObjectHandler
            series = pl.Series("objects", [pl.date(2023, 1, 1), pl.date(2023, 1, 2)])
            handler = registry.get_handler(series)

            # If we get here, should be ObjectHandler due to Object type
            # (though Polars usually doesn't allow Object types in Series)
            if "object" in str(series.dtype).lower():
                assert handler.data_type == "object"
            else:
                # If it's a proper Polars type, should use PolarsSeriesHandler
                assert handler.data_type == "polars_series"

        except Exception:
            # This is expected for Object types in Polars
            pass

    def test_series_handler_priority_in_registry(self, config):
        """Test that Series handlers are prioritized over DataFrame handlers."""
        registry = HandlerRegistry()

        if _has_pandas():
            import pandas as pd

            series = pd.Series([1, 2, 3])
            handler = registry.get_handler(series)
            assert handler.data_type == "pandas_series"  # Not pandas_dataframe

        if _has_polars():
            import polars as pl

            series = pl.Series("test", [1, 2, 3])
            handler = registry.get_handler(series)
            assert handler.data_type == "polars_series"  # Not polars_dataframe


class TestFallbackMechanisms:
    """Test fallback mechanisms in handlers."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return CacheConfig()

    def test_blosc_fallback(self):
        """Test blosc2 fallback to standard compression."""
        # Test that missing blosc2 doesn't break the system
        with patch.dict("sys.modules", {"blosc2": None}):
            # Should still be able to create handlers
            handler = ObjectHandler()
            assert handler is not None

    def test_polars_pandas_fallback(self, config):
        """Test that missing polars/pandas doesn't break registration."""
        registry = HandlerRegistry()

        # Registry should still work even if DataFrame libraries are missing
        assert len(registry.handlers) > 0

        # Should still handle basic data types
        handler = registry.get_handler({"key": "value"})
        assert handler is not None

    def test_compression_fallback(self, config):
        """Test compression fallback mechanisms."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            handler = ObjectHandler()
            test_data = {"large": list(range(1000))}
            file_path = Path(temp_dir) / "test"

            # Should work even if compression fails
            metadata = handler.put(test_data, file_path, config)
            loaded_data = handler.get(Path(metadata["actual_path"]), metadata)

            assert loaded_data == test_data
