"""
Tests for the interfaces module.

This module tests all interface definitions, abstract method enforcement,
error handling, and contract compliance validation for cache handlers.
"""

import pytest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock

from cacheness.interfaces import (
    CacheabilityChecker,
    CacheWriter,
    CacheReader,
    FormatProvider,
    CacheHandler,
    DataFrameHandler,
    SeriesHandler,
    ArrayHandler,
    ObjectHandler,
    CacheHandlerError,
    CacheWriteError,
    CacheReadError,
    CacheFormatError,
    CacheValidationError,
    HandlerFactory,
    HandlerRegistry,
)


class TestCacheabilityCheckerInterface:
    """Test the CacheabilityChecker interface."""

    def test_concrete_implementation(self):
        """Test that concrete implementation works correctly."""

        class ConcreteChecker(CacheabilityChecker):
            def can_handle(self, data: Any) -> bool:
                return isinstance(data, str)

        checker = ConcreteChecker()
        assert checker.can_handle("test") is True
        assert checker.can_handle(123) is False

    def test_method_signature_requirements(self):
        """Test that can_handle method has correct signature."""

        class TestChecker(CacheabilityChecker):
            def can_handle(self, data: Any) -> bool:
                return True

        checker = TestChecker()
        # Test that method exists and can be called
        result = checker.can_handle("anything")
        assert isinstance(result, bool)


class TestCacheWriterInterface:
    """Test the CacheWriter interface."""

    def test_concrete_implementation(self):
        """Test that concrete implementation works correctly."""

        class ConcreteWriter(CacheWriter):
            def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
                return {
                    "storage_format": "test",
                    "file_size": 100,
                    "actual_path": str(file_path),
                    "metadata": {"type": "test"},
                }

        writer = ConcreteWriter()
        result = writer.put("data", Path("/test"), None)

        assert "storage_format" in result
        assert "file_size" in result
        assert "actual_path" in result
        assert "metadata" in result

    def test_return_value_structure(self):
        """Test that put method returns correctly structured data."""

        class TestWriter(CacheWriter):
            def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
                return {
                    "storage_format": "json",
                    "file_size": 256,
                    "actual_path": file_path.with_suffix(".json").as_posix(),
                    "metadata": {"timestamp": "2024-01-01"},
                }

        writer = TestWriter()
        result = writer.put({"key": "value"}, Path("/cache/file"), {"config": True})

        assert result["storage_format"] == "json"
        assert result["file_size"] == 256
        assert result["actual_path"] == "/cache/file.json"
        assert result["metadata"]["timestamp"] == "2024-01-01"


class TestCacheReaderInterface:
    """Test the CacheReader interface."""

    def test_concrete_implementation(self):
        """Test that concrete implementation works correctly."""

        class ConcreteReader(CacheReader):
            def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
                return f"data_from_{file_path.name}"

        reader = ConcreteReader()
        result = reader.get(Path("/cache/file.json"), {"format": "json"})
        assert result == "data_from_file.json"

    def test_metadata_usage(self):
        """Test that get method can use metadata parameter."""

        class MetadataReader(CacheReader):
            def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
                format_type = metadata.get("storage_format", "unknown")
                # Use as_posix() for cross-platform path representation
                return f"Reading {file_path.as_posix()} as {format_type}"

        reader = MetadataReader()
        result = reader.get(
            Path("/cache/data.parquet"),
            {"storage_format": "parquet", "compression": "snappy"},
        )
        assert result == "Reading /cache/data.parquet as parquet"


class TestFormatProviderInterface:
    """Test the FormatProvider interface."""

    def test_concrete_implementation(self):
        """Test that concrete implementation works correctly."""

        class ConcreteProvider(FormatProvider):
            def get_file_extension(self, config: Any) -> str:
                return ".json"

            @property
            def data_type(self) -> str:
                return "json_data"

        provider = ConcreteProvider()
        assert provider.get_file_extension(None) == ".json"
        assert provider.data_type == "json_data"

    def test_property_implementation(self):
        """Test that data_type property works correctly."""

        class TestProvider(FormatProvider):
            def get_file_extension(self, config: Any) -> str:
                return ".csv"

            @property
            def data_type(self) -> str:
                return "csv_data"

        provider = TestProvider()
        # Test property access
        assert provider.data_type == "csv_data"

        # Test that it's a property, not a method
        assert isinstance(type(provider).data_type, property)


class TestCacheHandlerInterface:
    """Test the combined CacheHandler interface."""

    def test_inherits_all_interfaces(self):
        """Test that CacheHandler inherits from all required interfaces."""
        assert issubclass(CacheHandler, CacheabilityChecker)
        assert issubclass(CacheHandler, CacheWriter)
        assert issubclass(CacheHandler, CacheReader)
        assert issubclass(CacheHandler, FormatProvider)

    def test_concrete_implementation(self):
        """Test that complete implementation works."""

        class CompleteCacheHandler(CacheHandler):
            def can_handle(self, data: Any) -> bool:
                return True

            def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
                return {
                    "storage_format": "test",
                    "file_size": 100,
                    "actual_path": str(file_path),
                    "metadata": {},
                }

            def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
                return "cached_data"

            def get_file_extension(self, config: Any) -> str:
                return ".cache"

            @property
            def data_type(self) -> str:
                return "test_data"

        handler = CompleteCacheHandler()

        # Test all inherited functionality
        assert handler.can_handle("test") is True

        result = handler.put("data", Path("/test"), None)
        assert result["storage_format"] == "test"

        data = handler.get(Path("/test.cache"), {})
        assert data == "cached_data"

        assert handler.get_file_extension(None) == ".cache"
        assert handler.data_type == "test_data"


class TestSpecializedHandlerInterfaces:
    """Test specialized handler interfaces."""

    def test_dataframe_handler(self):
        """Test DataFrameHandler specialized interface."""
        assert issubclass(DataFrameHandler, CacheHandler)

        class TestDataFrameHandler(DataFrameHandler):
            def can_handle(self, data: Any) -> bool:
                return hasattr(data, "columns")

            def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
                return {
                    "storage_format": "parquet",
                    "file_size": 1000,
                    "actual_path": str(file_path),
                    "metadata": {},
                }

            def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
                return "dataframe"

            def get_file_extension(self, config: Any) -> str:
                return ".parquet"

            @property
            def data_type(self) -> str:
                return "dataframe"

            def validate_dataframe(self, data: Any) -> bool:
                return hasattr(data, "dtypes")

        handler = TestDataFrameHandler()
        assert handler.validate_dataframe(Mock(dtypes={})) is True
        assert handler.validate_dataframe("not_dataframe") is False

    def test_series_handler(self):
        """Test SeriesHandler specialized interface."""
        assert issubclass(SeriesHandler, CacheHandler)

        class TestSeriesHandler(SeriesHandler):
            def can_handle(self, data: Any) -> bool:
                return hasattr(data, "name")

            def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
                return {
                    "storage_format": "pickle",
                    "file_size": 500,
                    "actual_path": str(file_path),
                    "metadata": {},
                }

            def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
                return "series"

            def get_file_extension(self, config: Any) -> str:
                return ".pkl"

            @property
            def data_type(self) -> str:
                return "series"

            def preserve_series_metadata(
                self, data: Any, metadata: Dict[str, Any]
            ) -> Dict[str, Any]:
                enhanced = metadata.copy()
                enhanced["series_name"] = getattr(data, "name", None)
                return enhanced

        handler = TestSeriesHandler()
        mock_series = Mock()
        mock_series.name = "test_series"  # Set name as attribute, not Mock name
        metadata = handler.preserve_series_metadata(mock_series, {"existing": "data"})
        assert metadata["series_name"] == "test_series"
        assert metadata["existing"] == "data"

    def test_array_handler(self):
        """Test ArrayHandler specialized interface."""
        assert issubclass(ArrayHandler, CacheHandler)

        class TestArrayHandler(ArrayHandler):
            def can_handle(self, data: Any) -> bool:
                return hasattr(data, "shape")

            def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
                return {
                    "storage_format": "npz",
                    "file_size": 2000,
                    "actual_path": str(file_path),
                    "metadata": {},
                }

            def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
                return "array"

            def get_file_extension(self, config: Any) -> str:
                return ".npz"

            @property
            def data_type(self) -> str:
                return "array"

            def optimize_array_storage(self, data: Any, config: Any) -> str:
                if hasattr(data, "nbytes") and data.nbytes > 1000000:
                    return "blosc2"
                return "npz"

        handler = TestArrayHandler()

        small_array = Mock(nbytes=1000)
        large_array = Mock(nbytes=2000000)

        assert handler.optimize_array_storage(small_array, None) == "npz"
        assert handler.optimize_array_storage(large_array, None) == "blosc2"

    def test_object_handler(self):
        """Test ObjectHandler specialized interface."""
        assert issubclass(ObjectHandler, CacheHandler)

        class TestObjectHandler(ObjectHandler):
            def can_handle(self, data: Any) -> bool:
                return True  # Handle any object

            def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
                return {
                    "storage_format": "pickle",
                    "file_size": 300,
                    "actual_path": str(file_path),
                    "metadata": {},
                }

            def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
                return "object"

            def get_file_extension(self, config: Any) -> str:
                return ".pkl"

            @property
            def data_type(self) -> str:
                return "object"

            def validate_pickleable(self, data: Any) -> bool:
                try:
                    import pickle

                    pickle.dumps(data)
                    return True
                except:
                    return False

        handler = TestObjectHandler()

        assert handler.validate_pickleable("string") is True
        assert handler.validate_pickleable([1, 2, 3]) is True

        # Test with unpickleable object
        unpickleable = lambda x: x  # lambdas can't be pickled
        assert handler.validate_pickleable(unpickleable) is False


class TestCacheHandlerErrors:
    """Test cache handler exception classes."""

    def test_cache_handler_error_base(self, caplog):
        """Test base CacheHandlerError functionality."""
        error = CacheHandlerError(
            "Test error", handler_type="TestHandler", data_type="test_data"
        )

        assert str(error) == "Test error"
        assert error.handler_type == "TestHandler"
        assert error.data_type == "test_data"

        # Check that error was logged
        assert "Cache handler error: Test error" in caplog.text
        assert "handler=TestHandler" in caplog.text
        assert "data_type=test_data" in caplog.text

    def test_cache_handler_error_without_optional_params(self, caplog):
        """Test CacheHandlerError with minimal parameters."""
        error = CacheHandlerError("Simple error")

        assert str(error) == "Simple error"
        assert error.handler_type is None
        assert error.data_type is None

        assert "Cache handler error: Simple error" in caplog.text
        assert "handler=None" in caplog.text
        assert "data_type=None" in caplog.text

    def test_specific_error_types(self):
        """Test that specific error types inherit correctly."""
        write_error = CacheWriteError("Write failed", "Writer", "data")
        read_error = CacheReadError("Read failed", "Reader", "data")
        format_error = CacheFormatError("Format invalid", "Handler", "data")
        validation_error = CacheValidationError(
            "Validation failed", "Validator", "data"
        )

        # Test inheritance
        assert isinstance(write_error, CacheHandlerError)
        assert isinstance(read_error, CacheHandlerError)
        assert isinstance(format_error, CacheHandlerError)
        assert isinstance(validation_error, CacheHandlerError)

        # Test error messages
        assert str(write_error) == "Write failed"
        assert str(read_error) == "Read failed"
        assert str(format_error) == "Format invalid"
        assert str(validation_error) == "Validation failed"

    def test_error_inheritance_chain(self):
        """Test that all errors inherit from Exception."""
        errors = [
            CacheHandlerError("test"),
            CacheWriteError("test"),
            CacheReadError("test"),
            CacheFormatError("test"),
            CacheValidationError("test"),
        ]

        for error in errors:
            assert isinstance(error, Exception)
            assert isinstance(error, CacheHandlerError)


class TestHandlerFactoryInterface:
    """Test the HandlerFactory interface."""

    def test_concrete_implementation(self):
        """Test that concrete implementation works correctly."""

        class MockHandler(CacheHandler):
            def can_handle(self, data: Any) -> bool:
                return True

            def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
                return {}

            def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
                return None

            def get_file_extension(self, config: Any) -> str:
                return ".test"

            @property
            def data_type(self) -> str:
                return "test"

        class ConcreteFactory(HandlerFactory):
            def create_handler(
                self, data_type: str, config: Any = None
            ) -> "CacheHandler":
                if data_type == "test":
                    return MockHandler()
                raise ValueError(f"No handler for {data_type}")

            def get_available_handlers(self) -> Dict[str, type]:
                return {"test": MockHandler}

        factory = ConcreteFactory()

        # Test create_handler
        handler = factory.create_handler("test")
        assert isinstance(handler, MockHandler)

        with pytest.raises(ValueError, match="No handler for unknown"):
            factory.create_handler("unknown")

        # Test get_available_handlers
        handlers = factory.get_available_handlers()
        assert "test" in handlers
        assert handlers["test"] == MockHandler


class TestHandlerRegistryInterface:
    """Test the HandlerRegistry interface."""

    def test_concrete_implementation(self):
        """Test that concrete implementation works correctly."""

        class MockHandler(CacheHandler):
            def __init__(self, data_type_name: str):
                self._data_type = data_type_name

            def can_handle(self, data: Any) -> bool:
                return isinstance(data, str)

            def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
                return {}

            def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
                return None

            def get_file_extension(self, config: Any) -> str:
                return ".test"

            @property
            def data_type(self) -> str:
                return self._data_type

        class ConcreteRegistry(HandlerRegistry):
            def __init__(self):
                self.handlers = {}
                self.priorities = {}

            def register_handler(
                self, handler: "CacheHandler", priority: int = 0
            ) -> None:
                self.handlers[handler.data_type] = handler
                self.priorities[handler.data_type] = priority

            def get_handler(self, data: Any) -> "CacheHandler":
                # Find handler that can handle the data
                for handler in self.handlers.values():
                    if handler.can_handle(data):
                        return handler
                raise ValueError("No suitable handler found")

            def get_handler_by_type(self, data_type: str) -> "CacheHandler":
                if data_type in self.handlers:
                    return self.handlers[data_type]
                raise ValueError(f"No handler found for {data_type}")

            def list_handlers(self) -> Dict[str, "CacheHandler"]:
                return self.handlers.copy()

        registry = ConcreteRegistry()
        handler = MockHandler("string")

        # Test registration
        registry.register_handler(handler, priority=1)

        # Test get_handler
        found_handler = registry.get_handler("test string")
        assert found_handler is handler

        with pytest.raises(ValueError, match="No suitable handler found"):
            registry.get_handler(123)  # No handler for int

        # Test get_handler_by_type
        type_handler = registry.get_handler_by_type("string")
        assert type_handler is handler

        with pytest.raises(ValueError, match="No handler found for unknown"):
            registry.get_handler_by_type("unknown")

        # Test list_handlers
        handlers = registry.list_handlers()
        assert "string" in handlers
        assert handlers["string"] is handler


class TestInterfaceContractCompliance:
    """Test that interfaces enforce proper contracts."""

    def test_return_type_expectations(self):
        """Test that implementations return expected types."""

        class TypeCheckingHandler(CacheHandler):
            def can_handle(self, data: Any) -> bool:
                return True

            def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
                # Must return a dictionary
                return {
                    "storage_format": "test",
                    "file_size": 100,
                    "actual_path": str(file_path),
                    "metadata": {},
                }

            def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
                return "any_type_allowed"

            def get_file_extension(self, config: Any) -> str:
                return ".ext"

            @property
            def data_type(self) -> str:
                return "test_type"

        handler = TypeCheckingHandler()

        # Test return types
        assert isinstance(handler.can_handle("data"), bool)

        put_result = handler.put("data", Path("/test"), None)
        assert isinstance(put_result, dict)
        assert "storage_format" in put_result
        assert "file_size" in put_result
        assert "actual_path" in put_result
        assert "metadata" in put_result

        get_result = handler.get(Path("/test"), {})
        # get can return Any, so no type checking needed

        extension = handler.get_file_extension(None)
        assert isinstance(extension, str)
        assert extension.startswith(".")

        data_type = handler.data_type
        assert isinstance(data_type, str)

    def test_error_propagation(self):
        """Test that interfaces allow proper error propagation."""

        class ErrorPropagatingHandler(CacheHandler):
            def can_handle(self, data: Any) -> bool:
                if data == "error":
                    raise CacheValidationError("Cannot handle error data")
                return True

            def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
                if data == "write_error":
                    raise CacheWriteError("Cannot write data")
                return {
                    "storage_format": "test",
                    "file_size": 100,
                    "actual_path": str(file_path),
                    "metadata": {},
                }

            def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
                if "error" in str(file_path):
                    raise CacheReadError("Cannot read file")
                return "data"

            def get_file_extension(self, config: Any) -> str:
                if config == "error":
                    raise CacheFormatError("Invalid configuration")
                return ".test"

            @property
            def data_type(self) -> str:
                return "test"

        handler = ErrorPropagatingHandler()

        # Test that appropriate errors are raised
        with pytest.raises(CacheValidationError):
            handler.can_handle("error")

        with pytest.raises(CacheWriteError):
            handler.put("write_error", Path("/test"), None)

        with pytest.raises(CacheReadError):
            handler.get(Path("/error"), {})

        with pytest.raises(CacheFormatError):
            handler.get_file_extension("error")

        # Test normal operation still works
        assert handler.can_handle("normal") is True
        assert handler.put("normal", Path("/test"), None)["storage_format"] == "test"
        assert handler.get(Path("/normal"), {}) == "data"
        assert handler.get_file_extension("normal") == ".test"
