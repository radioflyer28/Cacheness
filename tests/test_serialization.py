"""
Tests for unified serialization system.
"""

import numpy as np
from pathlib import Path
from cacheness.serialization import serialize_for_cache_key, create_unified_cache_key
from cacheness import cacheness, cached


class TestUnifiedSerialization:
    """Test the unified serialization system."""

    def test_basic_types(self):
        """Test serialization of basic Python types."""
        assert serialize_for_cache_key(None) == "None"
        assert serialize_for_cache_key(42) == "int:42"
        assert serialize_for_cache_key("hello") == "str:hello"
        assert serialize_for_cache_key(3.14) == "float:3.14"
        assert serialize_for_cache_key(True) == "bool:True"

    def test_collections(self):
        """Test serialization of collections."""
        # Lists (not hashable, will be serialized recursively)
        result = serialize_for_cache_key([1, 2, 3])
        assert "list:" in result and "int:1" in result

        # Tuples (small tuples get recursive treatment for better introspection)
        result = serialize_for_cache_key((1, "two", 3.0))
        assert (
            "tuple:[int:1,str:two,float:3.0]" in result
        )  # Small tuples use recursive serialization

        # Large tuples use hash for performance
        large_tuple = tuple(range(20))  # > 10 elements
        result = serialize_for_cache_key(large_tuple)
        assert "hashed:tuple:" in result  # Large tuples use hash

        # Dictionaries (not hashable, serialized recursively and sorted by key)
        result = serialize_for_cache_key({"b": 2, "a": 1})
        assert "dict:" in result

        # Sets (not hashable, should be sorted)
        result = serialize_for_cache_key({3, 1, 2})
        assert "set:" in result

    def test_hashable_objects(self):
        """Test serialization of objects with __hash__ method."""

        class HashableObj:
            def __init__(self, value):
                self.value = value

            def __hash__(self):
                return hash(self.value)

            def __eq__(self, other):
                return isinstance(other, HashableObj) and self.value == other.value

        obj = HashableObj(42)
        result = serialize_for_cache_key(obj)
        # With improved ordering, objects with __dict__ get introspection treatment first
        assert result.startswith("HashableObj:dict:")
        assert "int:42" in result

        # Test hashable object without useful __dict__ (e.g., built-in types with custom __hash__)
        class HashableWithoutDict:
            __slots__ = ["value"]  # No __dict__

            def __init__(self, value):
                self.value = value

            def __hash__(self):
                return hash(self.value)

        obj_no_dict = HashableWithoutDict(99)
        result = serialize_for_cache_key(obj_no_dict)
        # Should fall back to hash since no __dict__
        assert "hashed:HashableWithoutDict:" in result

    def test_non_hashable_objects(self):
        """Test serialization of objects without __hash__ method."""

        class NonHashableObj:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"NonHashable({self.value})"

        obj = NonHashableObj("test")
        result = serialize_for_cache_key(obj)
        # Should fall back to string representation since it has __hash__ but it's unreliable
        assert "NonHashableObj" in result

    def test_custom_objects_with_dict(self):
        """Test serialization of custom objects with __dict__."""

        class CustomObj:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        obj = CustomObj(1, 2)
        result = serialize_for_cache_key(obj)
        assert "CustomObj:" in result

    def test_numpy_arrays(self):
        """Test serialization of numpy arrays."""
        arr = np.array([1, 2, 3])
        result = serialize_for_cache_key(arr)
        assert "array:" in result
        assert "(3,)" in result  # Shape
        assert "int64" in result or "int32" in result  # Dtype

    def test_cache_key_generation(self):
        """Test unified cache key generation."""
        params = {
            "string_param": "hello",
            "int_param": 42,
            "array_param": np.array([1, 2, 3]),
            "dict_param": {"nested": True},
        }

        key1 = create_unified_cache_key(params)
        key2 = create_unified_cache_key(params)

        # Should be consistent
        assert key1 == key2
        assert len(key1) == 16  # 16-character hex string

    def test_deterministic_ordering(self):
        """Test that parameter order doesn't affect cache key."""
        params1 = {"a": 1, "b": 2, "c": 3}
        params2 = {"c": 3, "a": 1, "b": 2}

        key1 = create_unified_cache_key(params1)
        key2 = create_unified_cache_key(params2)

        assert key1 == key2


class TestcachenessIntegration:
    """Test integration of unified serialization with cache system."""

    def test_cache_consistency(self):
        """Test that cacheness and decorators use consistent serialization."""

        class TestObj:
            def __init__(self, value):
                self.value = value

            def __hash__(self):
                return hash(self.value)

            def __eq__(self, other):
                return isinstance(other, TestObj) and self.value == other.value

        test_obj = TestObj(42)

        # Test with cacheness
        cache = cacheness()
        cache.put("result1", test_param=test_obj, other_param="test")
        result1 = cache.get(test_param=test_obj, other_param="test")
        assert result1 == "result1"

        # Test with decorator
        @cached()
        def test_function(test_param, other_param):
            return f"result2_{test_param.value}_{other_param}"

        result2 = test_function(test_obj, "test")
        result3 = test_function(test_obj, "test")  # Should hit cache

        assert result2 == result3
        assert "result2_42_test" == result2

    def test_complex_object_caching(self):
        """Test caching with complex objects."""

        complex_data = {
            "numpy_array": np.array([[1, 2], [3, 4]]),
            "nested_dict": {"level1": {"level2": [1, 2, 3]}},
            "tuple_data": (1, "two", 3.0, np.array([4, 5])),
        }

        @cached()
        def process_complex_data(**kwargs):
            return f"processed_{len(kwargs)}_items"

        result1 = process_complex_data(**complex_data)
        result2 = process_complex_data(**complex_data)  # Should hit cache

        assert result1 == result2
        assert "processed_3_items" == result1

    def test_path_object_handling(self):
        """Test that Path objects work with unified serialization."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            test_path = Path(f.name)

        try:
            cache = cacheness()
            cache.put("path_result", file_path=test_path)
            result = cache.get(file_path=test_path)
            assert result == "path_result"
        finally:
            test_path.unlink()  # Clean up
