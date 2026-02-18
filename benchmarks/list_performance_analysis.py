#!/usr/bin/env python3
"""
Detailed analysis of list_entries() performance across backends.

Measures raw backend performance (memory cache layer disabled) at increasing
cache sizes to compare scaling behaviour between JSON, SQLite, and
SQLite in-memory backends.

CALIBRATION NOTE (CACHE-nck)
-----------------------------
list_entries() and get_entry() were switched from full ORM hydration
(select(CacheEntry)) to SQLAlchemy Core column selects in commit 114db6e.
The SQLAlchemy session identity-map previously tracked every row for the
lifetime of the session, adding per-row overhead that compounded
super-linearly.  Core column selects return lightweight Row tuples and
bypass the identity-map entirely, making SQLite scale closer to linear.
"""

import time
import tempfile
import os
from cacheness import CacheConfig, cacheness


def analyze_list_performance():
    """Detailed analysis of list_entries() performance patterns."""
    print("List Performance Deep Dive")
    print("=" * 60)
    print(
        "NOTE: Memory cache layer DISABLED for JSON/SQLite to test raw backend performance"
    )

    backends = ["json", "sqlite", "sqlite_memory"]
    cache_sizes = [50, 200, 500, 1000]

    for size in cache_sizes:
        print(f"\nTesting with {size} entries:")

        for backend in backends:
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = os.path.join(temp_dir, f"{backend}_test")

                config = CacheConfig(
                    cache_dir=cache_dir,
                    metadata_backend=backend,
                    max_cache_size="5GB",  # Large enough for test
                    # Explicitly disable memory cache layer for JSON/SQLite
                    enable_memory_cache=False,
                    memory_cache_stats=False,
                )
                cache = cacheness(config)

                # Populate cache
                test_data = {"value": f"test_data_{size}"}
                for i in range(size):
                    cache.put(test_data, test_id=i, description=f"Entry {i}")

                # Test first call (should be pure backend performance)
                start = time.time()
                entries1 = cache.list_entries()
                first_call_time = (time.time() - start) * 1000

                # Test second call (should be same for JSON/SQLite, may be faster for memory if it has internal caching)
                start = time.time()
                entries2 = cache.list_entries()
                second_call_time = (time.time() - start) * 1000

                # Test third call
                start = time.time()
                entries3 = cache.list_entries()
                third_call_time = (time.time() - start) * 1000

                # Calculate any caching benefit
                speedup = (
                    first_call_time / second_call_time if second_call_time > 0 else 0
                )

                print(
                    f"  {backend:10} | 1st: {first_call_time:6.1f}ms | 2nd: {second_call_time:6.1f}ms | 3rd: {third_call_time:6.1f}ms | Speedup: {speedup:4.1f}x"
                )

                # Verify all calls return same number of entries
                assert len(entries1) == len(entries2) == len(entries3) == size
                cache.close()


def analyze_list_operations_detail():
    """Analyze what operations are actually happening in list_entries()."""
    print("\n\nList Operations Analysis")
    print("=" * 60)
    print("Testing with memory cache layer DISABLED for all backends")

    # Test with moderate size to see patterns clearly
    size = 500

    for backend in ["json", "sqlite", "sqlite_memory"]:
        print(f"\n{backend.upper()} Backend Analysis:")

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, f"{backend}_analysis")

            config = CacheConfig(
                cache_dir=cache_dir,
                metadata_backend=backend,
                max_cache_size="5GB",
                # Explicitly disable memory cache layer
                enable_memory_cache=False,
                memory_cache_stats=False,
            )
            cache = cacheness(config)

            # Populate with varied data
            for i in range(size):
                test_data = {"index": i, "data": f"entry_{i}"}
                cache.put(test_data, test_id=i, description=f"Test entry {i}")

            # Time multiple consecutive calls
            times = []
            for call_num in range(5):
                start = time.time()
                entries = cache.list_entries()
                call_time = (time.time() - start) * 1000
                times.append(call_time)
                print(
                    f"   Call {call_num + 1}: {call_time:6.1f}ms ({len(entries)} entries)"
                )

            # Analysis
            avg_time = sum(times) / len(times)
            first_vs_rest = (
                times[0] / (sum(times[1:]) / len(times[1:])) if len(times) > 1 else 1
            )

            print(f"   Average: {avg_time:6.1f}ms")
            print(f"   First vs Rest ratio: {first_vs_rest:4.1f}x")

            # Check if results are cached (same object reference) - should be False for JSON/SQLite now
            entries_a = cache.list_entries()
            entries_b = cache.list_entries()
            is_cached = entries_a is entries_b
            print(f"   Result caching: {'Yes' if is_cached else 'No'}")
            cache.close()


def compare_scaling_patterns():
    """Compare how each backend scales with cache size."""
    print("\n\nScaling Pattern Comparison")
    print("=" * 60)
    print("Testing RAW backend performance (no memory cache layer)")
    print("SQLite uses Core column selects (CACHE-nck) -- expect near-linear scaling.")

    sizes = [10, 50, 100, 200, 500, 1000, 2000]

    print(
        "Size     | SQLite-Mem (ms) | JSON (ms)   | SQLite (ms) | SQLiteMem/JSON | SQLite/JSON"
    )
    print("-" * 80)

    results = {}

    for size in sizes:
        size_results = {}

        for backend in ["json", "sqlite", "sqlite_memory"]:
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = os.path.join(temp_dir, f"scale_test_{backend}")

                config = CacheConfig(
                    cache_dir=cache_dir,
                    metadata_backend=backend,
                    max_cache_size="10GB",
                    # Disable memory cache layer to test raw performance
                    enable_memory_cache=False,
                    memory_cache_stats=False,
                )
                cache = cacheness(config)

                # Populate cache
                test_data = {"data": "x" * 100}  # Small consistent data
                for i in range(size):
                    cache.put(test_data, test_id=i)

                # Time list operation (first call, pure backend performance)
                start = time.time()
                entries = cache.list_entries()
                list_time = (time.time() - start) * 1000

                size_results[backend] = list_time
                cache.close()

        results[size] = size_results

        # Calculate ratios
        sqlite_mem_time = size_results["sqlite_memory"]
        json_time = size_results["json"]
        sqlite_time = size_results["sqlite"]

        sqlite_mem_json_ratio = sqlite_mem_time / json_time if json_time > 0 else 0
        sqlite_json_ratio = sqlite_time / json_time if json_time > 0 else 0

        print(
            f"{size:4d}     | {sqlite_mem_time:8.1f}       | {json_time:8.1f}    | {sqlite_time:8.1f}     | {sqlite_mem_json_ratio:8.2f}       | {sqlite_json_ratio:8.2f}"
        )

    return results


if __name__ == "__main__":
    analyze_list_performance()
    analyze_list_operations_detail()
    results = compare_scaling_patterns()

    print("\n\nKey Insights:")
    print("=" * 60)
    print("  SQLite In-Memory : Pure :memory: DB -- no file I/O, fastest raw reads")
    print("  JSON backend     : Entire file loaded to RAM on startup; list_entries()")
    print("                     iterates an in-memory dict -- O(n).")
    print("  SQLite (file)    : Core column selects (CACHE-nck) bypass the ORM")
    print("                     identity-map, eliminating per-row session overhead.")
    print("                     Scaling is now near-linear like JSON.")
    print("  Memory cache layer: wraps any backend; list_entries() returns a cached")
    print("                     snapshot -- effectively O(1) after first call.")
    print("")
    print("Recommendations:")
    print("  - Use SQLite for production caches (concurrent-safe, near-linear scaling)")
    print("  - Use JSON only for dev / single-process caches < ~200 entries")
    print("  - Enable memory cache layer when list_entries() is called in hot loops")
    print("  - SQLite in-memory is useful for isolated tests requiring top speed")
