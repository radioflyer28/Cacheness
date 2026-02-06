#!/usr/bin/env python3
"""
Performance comparison of the improved serialization ordering.
"""

import time
import numpy as np
from cacheness.serialization import serialize_for_cache_key

# Test objects
basic_int = 42
basic_str = "hello"
small_list = [1, 2, 3]
small_dict = {"a": 1, "b": 2}
small_tuple = (1, 2, 3)
large_tuple = tuple(range(20))  # Should use hash
numpy_array = np.array([1, 2, 3, 4, 5])


# Custom objects
class HashableObj:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)


class HashableObjNoDict:
    __slots__ = ["value"]

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)


hashable_with_dict = HashableObj(42)
hashable_no_dict = HashableObjNoDict(99)

# Test cases
test_cases = [
    ("Basic int", basic_int),
    ("Basic string", basic_str),
    ("Small list", small_list),
    ("Small dict", small_dict),
    ("Small tuple", small_tuple),
    ("Large tuple", large_tuple),
    ("NumPy array", numpy_array),
    ("Hashable with __dict__", hashable_with_dict),
    ("Hashable without __dict__", hashable_no_dict),
]


def benchmark_serialization():
    """Benchmark our improved serialization ordering."""
    print("🚀 Performance Benchmark - Improved Serialization Ordering")
    print("=" * 60)

    for name, obj in test_cases:
        # Warm up
        for _ in range(100):
            serialize_for_cache_key(obj)

        # Benchmark
        start_time = time.perf_counter()
        for _ in range(10000):
            result = serialize_for_cache_key(obj)
        end_time = time.perf_counter()

        avg_time_us = ((end_time - start_time) / 10000) * 1_000_000

        print(f"{name:25} | {avg_time_us:8.2f} μs | {result[:50]}")

    print("=" * 60)
    print("✅ Benchmark complete! Shows ordering strategy performance.")


if __name__ == "__main__":
    benchmark_serialization()
