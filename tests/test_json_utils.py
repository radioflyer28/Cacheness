"""
Tests for JSON utilities module.

Tests the cacheness-specific behavior of the JSON wrapper (dumps/loads),
not the underlying orjson/json library itself.
"""

import json
import pytest

from cacheness import json_utils


class TestJSONUtils:
    """Test cacheness JSON utility functions."""

    def test_dumps_and_loads_roundtrip(self):
        """Test that dumps -> loads returns the original data."""
        original_data = {
            "string": "test",
            "integer": 123,
            "float": 45.67,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "nested": {"inner": "value"},
        }

        json_str = json_utils.dumps(original_data)
        result = json_utils.loads(json_str)

        assert result == original_data

    def test_dumps_with_sort_keys(self):
        """Test serialization with key sorting for deterministic output."""
        data = {"z": 1, "a": 2, "m": 3}
        result = json_utils.dumps(data, sort_keys=True)

        assert result.index('"a"') < result.index('"m"')
        assert result.index('"m"') < result.index('"z"')

    def test_loads_with_bytes_input(self):
        """Test deserialization accepts bytes input."""
        json_bytes = b'{"key": "value", "number": 42}'
        result = json_utils.loads(json_bytes)

        assert result == {"key": "value", "number": 42}

    def test_dumps_with_custom_default(self):
        """Test serialization with custom default function."""

        class CustomObject:
            def __init__(self, value):
                self.value = value

        def custom_serializer(obj):
            if isinstance(obj, CustomObject):
                return {"custom_value": obj.value}
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        data = {"object": CustomObject("test")}
        result = json_utils.dumps(data, default=custom_serializer)

        parsed = json.loads(result)
        assert parsed == {"object": {"custom_value": "test"}}

    def test_dumps_non_serializable_raises(self):
        """Test that non-serializable objects raise without custom default."""

        class CustomObject:
            pass

        with pytest.raises((TypeError, ValueError)):
            json_utils.dumps({"object": CustomObject()})

    def test_loads_invalid_json_raises(self):
        """Test that invalid JSON raises an error."""
        with pytest.raises((json.JSONDecodeError, ValueError, SystemError)):
            json_utils.loads('{"incomplete": ')

    def test_unicode_roundtrip(self):
        """Test Unicode string handling through cacheness wrapper."""
        data = {
            "emoji": "\U0001f680",
            "chinese": "\u4f60\u597d",
            "special_chars": "caf\u00e9",
        }

        json_str = json_utils.dumps(data)
        result = json_utils.loads(json_str)
        assert result == data

    def test_get_json_backend(self):
        """Test backend detection returns expected value."""
        backend = json_utils.get_json_backend()
        assert backend in ["orjson", "builtin"]

    def test_repeated_serialization_consistency(self):
        """Test that repeated serialization is deterministic."""
        data = {"key": "value", "number": 42}

        results = [json_utils.dumps(data, sort_keys=True) for _ in range(10)]

        assert all(result == results[0] for result in results)

    def test_large_object_roundtrip(self):
        """Test serialization of large objects."""
        large_data = {
            f"key_{i}": {"value": f"value_{i}", "number": i} for i in range(1000)
        }

        json_str = json_utils.dumps(large_data)
        result = json_utils.loads(json_str)

        assert len(result) == 1000
        assert result["key_500"]["value"] == "value_500"
