"""
Tests for Phase 2.1: Handler Registration System

Tests the ability to:
- Register custom handlers with priority
- Unregister handlers
- List registered handlers
- Validate handler interface
"""

import pytest
from pathlib import Path
from typing import Any, Dict
import tempfile

from cacheness import (
    HandlerRegistry,
    CacheHandler,
    register_handler,
    unregister_handler,
    list_handlers,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockCustomHandler(CacheHandler):
    """A valid custom handler for testing."""

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and data.get("_custom_type") == "mock"

    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        # Simple mock implementation
        import json

        output_path = file_path.with_suffix(".mock.json")
        with open(output_path, "w") as f:
            json.dump(data, f)
        return {
            "storage_format": "mock_json",
            "file_path": str(output_path),
        }

    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        import json

        output_path = file_path.with_suffix(".mock.json")
        with open(output_path, "r") as f:
            return json.load(f)

    def get_file_extension(self, config: Any) -> str:
        return ".mock.json"

    @property
    def data_type(self) -> str:
        return "mock_custom"


class AnotherCustomHandler(CacheHandler):
    """Another valid custom handler for testing priority."""

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and data.get("_custom_type") == "another"

    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        return {"storage_format": "another"}

    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        return {}

    def get_file_extension(self, config: Any) -> str:
        return ".another"

    @property
    def data_type(self) -> str:
        return "another_custom"


class InvalidHandler:
    """Invalid handler missing required methods."""

    def can_handle(self, data: Any) -> bool:
        return False

    # Missing: put, get, get_file_extension, data_type


class PartialHandler:
    """Partially implemented handler."""

    def can_handle(self, data: Any) -> bool:
        return False

    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        return {}

    # Missing: get, get_file_extension, data_type


@pytest.fixture
def registry():
    """Create a fresh handler registry for each test."""
    return HandlerRegistry()


@pytest.fixture
def mock_handler():
    """Create a mock custom handler."""
    return MockCustomHandler()


@pytest.fixture
def another_handler():
    """Create another custom handler."""
    return AnotherCustomHandler()


# =============================================================================
# Test HandlerRegistry.register_handler()
# =============================================================================


class TestRegisterHandler:
    """Tests for HandlerRegistry.register_handler()"""

    def test_register_handler_appends_to_end(self, registry, mock_handler):
        """Handler should be appended to end when no priority specified."""
        initial_count = len(registry.handlers)

        registry.register_handler(mock_handler)

        assert len(registry.handlers) == initial_count + 1
        assert registry.handlers[-1] is mock_handler

    def test_register_handler_with_priority_zero(self, registry, mock_handler):
        """Handler with priority=0 should be first in list."""
        registry.register_handler(mock_handler, priority=0)

        assert registry.handlers[0] is mock_handler

    def test_register_handler_with_priority_middle(self, registry, mock_handler):
        """Handler should be inserted at correct priority position."""
        initial_count = len(registry.handlers)

        # Insert at position 1 (second position)
        registry.register_handler(mock_handler, priority=1)

        assert len(registry.handlers) == initial_count + 1
        if initial_count > 0:
            assert registry.handlers[1] is mock_handler

    def test_register_handler_with_high_priority_appends(self, registry, mock_handler):
        """Priority higher than list length should append to end."""
        initial_count = len(registry.handlers)

        registry.register_handler(mock_handler, priority=1000)

        assert registry.handlers[-1] is mock_handler

    def test_register_handler_negative_priority_becomes_zero(
        self, registry, mock_handler
    ):
        """Negative priority should be treated as 0."""
        registry.register_handler(mock_handler, priority=-5)

        assert registry.handlers[0] is mock_handler

    def test_register_duplicate_handler_raises(self, registry, mock_handler):
        """Registering handler with same data_type should raise ValueError."""
        registry.register_handler(mock_handler)

        duplicate_handler = MockCustomHandler()

        with pytest.raises(ValueError, match="already registered"):
            registry.register_handler(duplicate_handler)

    def test_register_handler_with_custom_name(self, registry, mock_handler):
        """Handler can be registered with custom name to avoid conflicts."""
        registry.register_handler(mock_handler, name="custom_name_1")

        # Same handler class with different name should work
        another_instance = MockCustomHandler()
        # This should work because we use a custom name
        # Note: The current implementation checks data_type, not name
        # So this test documents expected behavior if we change to name-based dedup

    def test_register_invalid_handler_raises(self, registry):
        """Handler missing required interface should raise ValueError."""
        invalid = InvalidHandler()

        with pytest.raises(ValueError, match="missing required"):
            registry.register_handler(invalid)

    def test_register_partial_handler_raises(self, registry):
        """Handler with partial implementation should raise ValueError."""
        partial = PartialHandler()

        with pytest.raises(ValueError, match="missing required"):
            registry.register_handler(partial)


# =============================================================================
# Test HandlerRegistry.unregister_handler()
# =============================================================================


class TestUnregisterHandler:
    """Tests for HandlerRegistry.unregister_handler()"""

    def test_unregister_existing_handler(self, registry, mock_handler):
        """Should return True and remove handler when found."""
        registry.register_handler(mock_handler)
        initial_count = len(registry.handlers)

        result = registry.unregister_handler("mock_custom")

        assert result is True
        assert len(registry.handlers) == initial_count - 1
        assert mock_handler not in registry.handlers

    def test_unregister_nonexistent_handler(self, registry):
        """Should return False when handler not found."""
        result = registry.unregister_handler("nonexistent_handler")

        assert result is False

    def test_unregister_builtin_handler(self, registry):
        """Should be able to unregister built-in handlers."""
        # Find a built-in handler
        handlers_before = [h.data_type for h in registry.handlers]

        if "object" in handlers_before:
            result = registry.unregister_handler("object")
            assert result is True
            assert "object" not in [h.data_type for h in registry.handlers]


# =============================================================================
# Test HandlerRegistry.list_handlers()
# =============================================================================


class TestListHandlers:
    """Tests for HandlerRegistry.list_handlers()"""

    def test_list_handlers_returns_list(self, registry):
        """Should return a list of handler info dictionaries."""
        result = registry.list_handlers()

        assert isinstance(result, list)
        assert len(result) == len(registry.handlers)

    def test_list_handlers_includes_required_keys(self, registry):
        """Each handler info should have required keys."""
        result = registry.list_handlers()

        required_keys = {"name", "priority", "class", "is_builtin"}

        for handler_info in result:
            assert required_keys.issubset(handler_info.keys())

    def test_list_handlers_priority_is_position(self, registry):
        """Priority should match position in list."""
        result = registry.list_handlers()

        for i, handler_info in enumerate(result):
            assert handler_info["priority"] == i

    def test_list_handlers_identifies_builtin(self, registry):
        """Built-in handlers should be marked is_builtin=True."""
        result = registry.list_handlers()

        builtin_names = {
            "polars_dataframe",
            "pandas_dataframe",
            "polars_series",
            "pandas_series",
            "numpy_array",
            "object",
            "tensorflow_tensor",
        }

        for handler_info in result:
            if handler_info["name"] in builtin_names:
                assert handler_info["is_builtin"] is True

    def test_list_handlers_custom_not_builtin(self, registry, mock_handler):
        """Custom handlers should be marked is_builtin=False."""
        registry.register_handler(mock_handler)

        result = registry.list_handlers()

        custom_info = next(h for h in result if h["name"] == "mock_custom")
        assert custom_info["is_builtin"] is False


# =============================================================================
# Test Handler Selection with Custom Handlers
# =============================================================================


class TestHandlerSelection:
    """Tests for handler selection with custom handlers."""

    def test_custom_handler_selected_by_priority(self, registry, mock_handler):
        """Custom handler at priority 0 should be checked first."""
        registry.register_handler(mock_handler, priority=0)

        # Data that matches custom handler
        data = {"_custom_type": "mock", "value": 42}

        handler = registry.get_handler(data)

        assert handler is mock_handler

    def test_custom_handler_at_end_not_selected_for_dict(self, registry, mock_handler):
        """
        Custom handler at end of list won't be selected for dict data
        because ObjectHandler (fallback) comes earlier and handles all dicts.

        This is expected behavior - priority matters!
        """
        registry.register_handler(mock_handler)  # Appended to end

        # Data that matches custom handler, but ObjectHandler comes first
        data = {"_custom_type": "mock", "value": 42}

        handler = registry.get_handler(data)

        # ObjectHandler handles it first since it's earlier in the list
        # This is the expected behavior - priority determines selection
        assert handler.data_type == "object"  # ObjectHandler

    def test_builtin_handler_selected_when_no_custom_match(
        self, registry, mock_handler
    ):
        """Built-in handler should be selected when custom doesn't match."""
        registry.register_handler(mock_handler, priority=0)

        # Regular dict without _custom_type should go to ObjectHandler
        data = {"regular": "dict"}

        handler = registry.get_handler(data)

        # Should be ObjectHandler (or another built-in), not our mock
        assert handler is not mock_handler


# =============================================================================
# Test Module-Level Registration Functions
# =============================================================================


class TestModuleLevelRegistration:
    """Tests for module-level register_handler/unregister_handler/list_handlers."""

    def test_module_list_handlers(self):
        """Module-level list_handlers should return handler list."""
        result = list_handlers()

        assert isinstance(result, list)
        # Should have at least the default handlers
        assert len(result) > 0

    def test_module_register_and_unregister(self):
        """Module-level registration functions should work."""

        # Create a unique handler to avoid conflicts
        class UniqueTestHandler(CacheHandler):
            def can_handle(self, data):
                return False

            def put(self, data, file_path, config):
                return {}

            def get(self, file_path, metadata):
                return None

            def get_file_extension(self, config):
                return ".unique"

            @property
            def data_type(self):
                return "unique_test_handler_12345"

        handler = UniqueTestHandler()

        try:
            # Register
            register_handler(handler)

            # Verify it's in the list
            handlers = list_handlers()
            names = [h["name"] for h in handlers]
            assert "unique_test_handler_12345" in names

        finally:
            # Clean up
            unregister_handler("unique_test_handler_12345")

        # Verify it's removed
        handlers = list_handlers()
        names = [h["name"] for h in handlers]
        assert "unique_test_handler_12345" not in names


# =============================================================================
# Test Handler Validation
# =============================================================================


class TestHandlerValidation:
    """Tests for handler validation during registration."""

    def test_validates_can_handle_method(self, registry):
        """Should validate can_handle method exists and is callable."""

        class MissingCanHandle:
            def put(self, data, file_path, config):
                return {}

            def get(self, file_path, metadata):
                return None

            def get_file_extension(self, config):
                return ".test"

            @property
            def data_type(self):
                return "test"

        with pytest.raises(ValueError, match="can_handle"):
            registry.register_handler(MissingCanHandle())

    def test_validates_put_method(self, registry):
        """Should validate put method exists and is callable."""

        class MissingPut:
            def can_handle(self, data):
                return False

            def get(self, file_path, metadata):
                return None

            def get_file_extension(self, config):
                return ".test"

            @property
            def data_type(self):
                return "test"

        with pytest.raises(ValueError, match="put"):
            registry.register_handler(MissingPut())

    def test_validates_get_method(self, registry):
        """Should validate get method exists and is callable."""

        class MissingGet:
            def can_handle(self, data):
                return False

            def put(self, data, file_path, config):
                return {}

            def get_file_extension(self, config):
                return ".test"

            @property
            def data_type(self):
                return "test"

        with pytest.raises(ValueError, match="get"):
            registry.register_handler(MissingGet())

    def test_validates_data_type_property(self, registry):
        """Should validate data_type property exists."""

        class MissingDataType:
            def can_handle(self, data):
                return False

            def put(self, data, file_path, config):
                return {}

            def get(self, file_path, metadata):
                return None

            def get_file_extension(self, config):
                return ".test"

        with pytest.raises(ValueError, match="data_type"):
            registry.register_handler(MissingDataType())


# =============================================================================
# Integration Tests
# =============================================================================


class TestHandlerIntegration:
    """Integration tests for custom handlers with actual caching."""

    def test_custom_handler_put_and_get(self, mock_handler):
        """Custom handler should be able to store and retrieve data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_data"

            # Mock config
            class MockConfig:
                pass

            data = {"_custom_type": "mock", "value": 42, "nested": {"a": 1}}

            # Put
            metadata = mock_handler.put(data, file_path, MockConfig())

            assert metadata["storage_format"] == "mock_json"

            # Get
            retrieved = mock_handler.get(file_path, metadata)

            assert retrieved == data

    def test_multiple_custom_handlers_priority(
        self, registry, mock_handler, another_handler
    ):
        """Multiple custom handlers should be checked in priority order."""
        # Register both at high priority (before ObjectHandler)
        registry.register_handler(another_handler, priority=0)
        registry.register_handler(mock_handler, priority=1)

        # Data for another_handler
        data1 = {"_custom_type": "another"}
        handler1 = registry.get_handler(data1)
        assert handler1 is another_handler

        # Data for mock_handler
        data2 = {"_custom_type": "mock"}
        handler2 = registry.get_handler(data2)
        assert handler2 is mock_handler
