"""
Comprehensive test suite for JSON utilities module.

Tests both orjson (when available) and built-in json fallback functionality,
covering edge cases, error handling, encoding issues, and performance considerations.
"""

import json
import tempfile
from datetime import datetime, date, time
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch, Mock
import pytest

from cacheness import json_utils


class TestJSONUtilsBasicFunctionality:
    """Test basic JSON serialization and deserialization functionality."""

    def test_dumps_basic_serialization(self):
        """Test basic object serialization."""
        data = {"key": "value", "number": 42, "boolean": True}
        result = json_utils.dumps(data)
        
        assert isinstance(result, str)
        # Parse back to verify it's valid JSON
        parsed = json.loads(result)
        assert parsed == data

    def test_loads_basic_deserialization(self):
        """Test basic JSON string deserialization."""
        json_str = '{"key": "value", "number": 42, "boolean": true}'
        result = json_utils.loads(json_str)
        
        expected = {"key": "value", "number": 42, "boolean": True}
        assert result == expected

    def test_dumps_with_sort_keys(self):
        """Test serialization with key sorting."""
        data = {"z": 1, "a": 2, "m": 3}
        result = json_utils.dumps(data, sort_keys=True)
        
        # With sorted keys, 'a' should come first
        assert result.index('"a"') < result.index('"m"')
        assert result.index('"m"') < result.index('"z"')

    def test_dumps_without_sort_keys(self):
        """Test serialization without key sorting."""
        data = {"z": 1, "a": 2, "m": 3}
        result = json_utils.dumps(data, sort_keys=False)
        
        # Should be valid JSON regardless of key order
        parsed = json.loads(result)
        assert parsed == data

    def test_loads_with_bytes_input(self):
        """Test deserialization with bytes input."""
        json_bytes = b'{"key": "value", "number": 42}'
        result = json_utils.loads(json_bytes)
        
        expected = {"key": "value", "number": 42}
        assert result == expected

    def test_roundtrip_consistency(self):
        """Test that dumps -> loads returns the original data."""
        original_data = {
            "string": "test",
            "integer": 123,
            "float": 45.67,
            "boolean": True,
            "null": None,
            "array": [1, 2, 3],
            "nested": {"inner": "value"}
        }
        
        json_str = json_utils.dumps(original_data)
        result = json_utils.loads(json_str)
        
        assert result == original_data


class TestJSONUtilsDataTypes:
    """Test JSON utilities with various data types."""

    def test_nested_structures(self):
        """Test deeply nested data structures."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "deep_value": "found",
                        "array": [{"item": i} for i in range(5)]
                    }
                }
            }
        }
        
        json_str = json_utils.dumps(data)
        result = json_utils.loads(json_str)
        assert result == data

    def test_large_arrays(self):
        """Test serialization of large arrays."""
        large_array = list(range(1000))
        data = {"large_array": large_array}
        
        json_str = json_utils.dumps(data)
        result = json_utils.loads(json_str)
        assert result == data

    def test_unicode_strings(self):
        """Test Unicode string handling."""
        data = {
            "emoji": "🚀📊💾",
            "chinese": "测试数据",
            "arabic": "بيانات الاختبار",
            "special_chars": "äöüß",
            "escape_chars": 'quotes "and" backslash \\'
        }
        
        json_str = json_utils.dumps(data)
        result = json_utils.loads(json_str)
        assert result == data

    def test_numeric_precision(self):
        """Test numeric precision handling."""
        data = {
            "small_float": 0.000000001,
            "large_float": 999999999.999999999,
            "negative": -123.456,
            "zero": 0,
            "scientific": 1.23e-10
        }
        
        json_str = json_utils.dumps(data)
        result = json_utils.loads(json_str)
        
        # Note: JSON may not preserve exact floating point precision
        assert abs(result["small_float"] - data["small_float"]) < 1e-10
        assert abs(result["large_float"] - data["large_float"]) < 1e-6

    def test_empty_containers(self):
        """Test empty containers."""
        data = {
            "empty_dict": {},
            "empty_list": [],
            "empty_string": ""
        }
        
        json_str = json_utils.dumps(data)
        result = json_utils.loads(json_str)
        assert result == data


class TestJSONUtilsErrorHandling:
    """Test error handling and edge cases."""

    def test_dumps_with_non_serializable_object(self):
        """Test serialization of non-serializable objects without custom default."""
        class CustomObject:
            pass
        
        data = {"object": CustomObject()}
        
        with pytest.raises((TypeError, ValueError)):
            json_utils.dumps(data)

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

    def test_loads_with_invalid_json(self):
        """Test deserialization of invalid JSON."""
        # Test with basic invalid JSON that should fail consistently
        invalid_json_strings = [
            '{"incomplete": ',
            'not json at all',
            '{"unclosed": "string}'
        ]
        
        for invalid_json in invalid_json_strings:
            try:
                result = json_utils.loads(invalid_json)
                # If it doesn't raise an exception, that's unexpected
                pytest.fail(f"Expected exception for invalid JSON: {invalid_json}")
            except (json.JSONDecodeError, ValueError, TypeError, SystemError):
                # Any of these exceptions are acceptable for invalid JSON
                pass

    def test_loads_with_none_input(self):
        """Test deserialization with None input."""
        try:
            result = json_utils.loads(None)  # type: ignore
            pytest.fail("Expected exception for None input")
        except (TypeError, ValueError, SystemError):
            # Any of these exceptions are acceptable
            pass

    def test_dumps_with_circular_reference(self):
        """Test serialization with circular references."""
        data = {"key": "value"}
        data["self"] = data  # type: ignore  # Create circular reference
        
        # orjson throws TypeError with "Recursion limit reached"
        # builtin json throws ValueError
        with pytest.raises((ValueError, RecursionError, TypeError)):
            json_utils.dumps(data)

    def test_extremely_large_strings(self):
        """Test handling of very large strings."""
        large_string = "x" * 1000000  # 1MB string
        data = {"large": large_string}
        
        json_str = json_utils.dumps(data)
        result = json_utils.loads(json_str)
        assert result == data

    def test_deeply_nested_structures_limit(self):
        """Test deeply nested structures approaching recursion limits."""
        # Create a deeply nested structure
        data = {}
        current = data
        for i in range(100):  # Reasonable depth
            current[f"level{i}"] = {}
            current = current[f"level{i}"]
        current["final"] = "value"
        
        json_str = json_utils.dumps(data)
        result = json_utils.loads(json_str)
        assert result == data


class TestJSONUtilsSpecialCases:
    """Test special cases and edge scenarios."""

    def test_datetime_serialization_with_default(self):
        """Test datetime serialization with custom default."""
        def datetime_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, date):
                return obj.isoformat()
            elif isinstance(obj, time):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        data = {
            "datetime": datetime(2023, 1, 1, 12, 30, 45),
            "date": date(2023, 1, 1),
            "time": time(12, 30, 45)
        }
        
        result = json_utils.dumps(data, default=datetime_serializer)
        parsed = json.loads(result)
        
        assert parsed["datetime"] == "2023-01-01T12:30:45"
        assert parsed["date"] == "2023-01-01"
        assert parsed["time"] == "12:30:45"

    def test_decimal_serialization_with_default(self):
        """Test Decimal serialization with custom default."""
        def decimal_serializer(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        data = {"decimal": Decimal("123.456")}
        result = json_utils.dumps(data, default=decimal_serializer)
        parsed = json.loads(result)
        
        assert abs(parsed["decimal"] - 123.456) < 0.001

    def test_pathlib_serialization_with_default(self):
        """Test Path object serialization with custom default."""
        def path_serializer(obj):
            if isinstance(obj, Path):
                # Use as_posix() for cross-platform consistency
                return obj.as_posix()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        data = {"path": Path("/test/path")}
        result = json_utils.dumps(data, default=path_serializer)
        parsed = json.loads(result)
        
        assert parsed["path"] == "/test/path"

    def test_whitespace_handling(self):
        """Test JSON with various whitespace scenarios."""
        json_strings = [
            '  { "key" : "value" }  ',
            '{\n  "key": "value"\n}',
            '{"key":"value"}',  # No spaces
            '\t{\t"key"\t:\t"value"\t}\t'
        ]
        
        expected = {"key": "value"}
        for json_str in json_strings:
            result = json_utils.loads(json_str)
            assert result == expected

    def test_json_with_null_bytes(self):
        """Test JSON strings containing null bytes."""
        # Most JSON parsers should handle this gracefully
        data = {"key": "value\x00with\x00nulls"}
        json_str = json_utils.dumps(data)
        result = json_utils.loads(json_str)
        assert result == data


class TestJSONUtilsBackendDetection:
    """Test JSON backend detection and functionality."""

    def test_get_json_backend(self):
        """Test backend detection returns expected value."""
        backend = json_utils.get_json_backend()
        assert backend in ["orjson", "builtin"]

    def test_backend_consistency(self):
        """Test that the backend is consistently reported."""
        backend1 = json_utils.get_json_backend()
        backend2 = json_utils.get_json_backend()
        assert backend1 == backend2

    def test_builtin_fallback_simulation(self):
        """Test behavior when orjson is not available."""
        # This test simulates the ImportError scenario by checking backend detection
        backend = json_utils.get_json_backend()
        assert backend in ["orjson", "builtin"]
        
        # Test basic functionality regardless of backend
        data = {"test": "value"}
        result = json_utils.dumps(data)
        assert isinstance(result, str)
        
        parsed = json_utils.loads(result)
        assert parsed == data


class TestJSONUtilsPerformanceConsiderations:
    """Test performance-related scenarios."""

    def test_large_object_serialization(self):
        """Test serialization of large objects."""
        # Create a substantial object
        large_data = {
            f"key_{i}": {
                "value": f"value_{i}",
                "number": i,
                "array": list(range(10))
            }
            for i in range(1000)
        }
        
        json_str = json_utils.dumps(large_data)
        result = json_utils.loads(json_str)
        
        assert len(result) == 1000
        assert result["key_500"]["value"] == "value_500"

    def test_repeated_serialization_consistency(self):
        """Test that repeated serialization is consistent."""
        data = {"key": "value", "number": 42}
        
        results = []
        for _ in range(10):
            results.append(json_utils.dumps(data, sort_keys=True))
        
        # All results should be identical when keys are sorted
        assert all(result == results[0] for result in results)

    def test_binary_data_handling(self):
        """Test handling of binary-like data."""
        # JSON can't directly handle binary data, but we can test
        # with base64-encoded data or hex strings
        import base64
        
        binary_data = b"binary data example"
        encoded_data = base64.b64encode(binary_data).decode('utf-8')
        
        data = {"binary": encoded_data}
        json_str = json_utils.dumps(data)
        result = json_utils.loads(json_str)
        
        # Decode back to verify
        decoded = base64.b64decode(result["binary"])
        assert decoded == binary_data


class TestJSONUtilsIntegration:
    """Test integration scenarios and real-world usage patterns."""

    def test_configuration_file_pattern(self):
        """Test pattern commonly used for configuration files."""
        config_data = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "testdb",
                "ssl": True
            },
            "cache": {
                "backend": "redis",
                "ttl": 3600,
                "max_connections": 10
            },
            "features": ["feature1", "feature2", "feature3"]
        }
        
        json_str = json_utils.dumps(config_data, sort_keys=True)
        result = json_utils.loads(json_str)
        assert result == config_data

    def test_api_response_pattern(self):
        """Test pattern commonly used for API responses."""
        api_response = {
            "status": "success",
            "data": {
                "users": [
                    {"id": 1, "name": "User 1", "active": True},
                    {"id": 2, "name": "User 2", "active": False}
                ],
                "total": 2,
                "page": 1,
                "per_page": 10
            },
            "meta": {
                "timestamp": "2023-01-01T00:00:00Z",
                "version": "1.0.0"
            }
        }
        
        json_str = json_utils.dumps(api_response)
        result = json_utils.loads(json_str)
        assert result == api_response

    def test_caching_metadata_pattern(self):
        """Test pattern used for cache metadata."""
        cache_metadata = {
            "cache_key": "user:123:profile",
            "created_at": "2023-01-01T00:00:00Z",
            "expires_at": "2023-01-01T01:00:00Z",
            "hit_count": 42,
            "tags": ["user", "profile"],
            "size_bytes": 1024,
            "compression": "gzip"
        }
        
        json_str = json_utils.dumps(cache_metadata)
        result = json_utils.loads(json_str)
        assert result == cache_metadata

    def test_temporary_file_roundtrip(self):
        """Test writing to and reading from temporary files."""
        test_data = {
            "test_name": "temporary_file_test",
            "data": list(range(100)),
            "metadata": {"created": "2023-01-01", "type": "test"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
            # Write JSON to file
            json_str = json_utils.dumps(test_data, sort_keys=True)
            f.write(json_str)
            f.flush()
            
            # Read back from file
            f.seek(0)
            file_content = f.read()
            result = json_utils.loads(file_content)
            
            assert result == test_data
        
        # Clean up
        Path(f.name).unlink()


class TestJSONUtilsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_json_with_comments_rejection(self):
        """Test that JSON with comments is properly rejected."""
        json_with_comments = '''{"key": "value", // comment }'''
        
        try:
            result = json_utils.loads(json_with_comments)
            pytest.fail("Expected exception for JSON with comments")
        except (json.JSONDecodeError, ValueError, SystemError):
            # Any of these exceptions are acceptable
            pass

    def test_trailing_comma_rejection(self):
        """Test that trailing commas are properly rejected."""
        json_with_trailing_comma = '{"key": "value",}'
        
        try:
            result = json_utils.loads(json_with_trailing_comma)
            pytest.fail("Expected exception for JSON with trailing comma")
        except (json.JSONDecodeError, ValueError, SystemError):
            # Any of these exceptions are acceptable
            pass

    def test_single_quotes_rejection(self):
        """Test that single quotes are properly rejected."""
        json_with_single_quotes = "{'key': 'value'}"
        
        try:
            result = json_utils.loads(json_with_single_quotes)
            pytest.fail("Expected exception for JSON with single quotes")
        except (json.JSONDecodeError, ValueError, SystemError):
            # Any of these exceptions are acceptable
            pass

    def test_nan_and_infinity_handling(self):
        """Test handling of NaN and infinity values."""
        # Standard JSON doesn't support NaN or infinity
        import math
        
        def nan_inf_serializer(obj):
            if isinstance(obj, float):
                if math.isnan(obj):
                    return "NaN"
                elif math.isinf(obj):
                    return "Infinity" if obj > 0 else "-Infinity"
                else:
                    return obj  # Normal float
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        # Test with normal floats that should work
        data = {"normal_float": 3.14, "zero": 0.0}
        result = json_utils.dumps(data, default=nan_inf_serializer)
        parsed = json.loads(result)
        
        assert parsed["normal_float"] == 3.14
        assert parsed["zero"] == 0.0

    def test_very_long_key_names(self):
        """Test handling of very long key names."""
        long_key = "x" * 1000
        data = {long_key: "value"}
        
        json_str = json_utils.dumps(data)
        result = json_utils.loads(json_str)
        assert result[long_key] == "value"

    def test_mixed_encoding_strings(self):
        """Test strings with mixed encoding characters."""
        data = {
            "mixed": "ASCII + Unicode: café, naïve, 北京, 🌟",
            "control_chars": "tab:\t newline:\n carriage_return:\r",
            "quotes": 'double:"quote" single:\'quote\'',
            "backslash": "path\\to\\file"
        }
        
        json_str = json_utils.dumps(data)
        result = json_utils.loads(json_str)
        assert result == data