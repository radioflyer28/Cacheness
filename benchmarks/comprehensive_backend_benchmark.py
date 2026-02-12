#!/usr/bin/env python3
"""
Comprehensive Backend Performance Benchmark
===========================================

This consolidates multiple backend benchmarks into a single comprehensive test:
1. Raw backend performance comparison (JSON, SQLite, SQLite In-Memory)
2. Memory cache layer impact (enabled vs disabled for JSON/SQLite)
3. Realistic workload patterns with different access patterns
4. Scaling characteristics across different cache sizes
5. Concurrent access patterns

Replaces: backend_comparison_benchmark.py, test_entry_caching.py,
         test_realistic_caching.py, quick_backend_demo.py, list_performance_analysis.py
"""

import time
import tempfile
import os
import random
import numpy as np
from cacheness import CacheConfig, cacheness


def create_test_data(size_category="small"):
    """Create test data of different sizes for realistic testing."""
    if size_category == "small":
        return {
            "array": np.random.random(50).tolist(),
            "metadata": {"category": size_category, "created": time.time()},
        }
    elif size_category == "medium":
        return {
            "array": np.random.random(500).tolist(),
            "dataframe_data": np.random.random((20, 5)).tolist(),
            "metadata": {"category": size_category, "created": time.time()},
        }
    else:  # large
        return {
            "array": np.random.random(2000).tolist(),
            "dataframe_data": np.random.random((100, 10)).tolist(),
            "metadata": {"category": size_category, "created": time.time()},
        }


def benchmark_raw_backend_performance():
    """Test 1: Raw backend performance without memory cache layer."""
    print("🏁 Test 1: Raw Backend Performance Comparison")
    print("=" * 80)
    print("Testing pure backend performance with memory cache layer DISABLED")
    print()

    backends = ["json", "sqlite", "sqlite_memory"]
    cache_sizes = [50, 100, 200, 500]

    print("Backend   | Cache Size | PUT ops/sec | GET ops/sec | LIST time (ms)")
    print("-" * 70)

    for backend in backends:
        for size in cache_sizes:
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = os.path.join(temp_dir, f"{backend}_test")

                config = CacheConfig(
                    cache_dir=cache_dir,
                    metadata_backend=backend,
                    max_cache_size_mb=5000,
                    # Disable memory cache layer for raw performance
                    enable_memory_cache=False,
                    memory_cache_stats=False,
                )
                cache = cacheness(config)

                # Generate test data
                test_data = create_test_data("small")

                # Benchmark PUT operations
                start = time.time()
                cache_keys = []
                for i in range(size):
                    key = cache.put(test_data, test_id=i)
                    cache_keys.append(key)
                put_time = time.time() - start
                put_ops_per_sec = size / put_time

                # Benchmark GET operations
                start = time.time()
                for key in cache_keys:
                    result = cache.get(cache_key=key)
                    assert result is not None
                get_time = time.time() - start
                get_ops_per_sec = size / get_time

                # Benchmark LIST operations
                start = time.time()
                cache.list_entries()
                list_time = (time.time() - start) * 1000

                print(
                    f"{backend:9} | {size:10d} | {put_ops_per_sec:8.0f}    | {get_ops_per_sec:8.0f}    | {list_time:8.1f}"
                )
                cache.close()


def benchmark_memory_cache_layer_impact():
    """Test 2: Memory cache layer performance impact for JSON and SQLite."""
    print("\n\n🚀 Test 2: Memory Cache Layer Impact")
    print("=" * 80)
    print("Comparing JSON and SQLite backends with memory cache layer ON vs OFF")
    print()

    backends = ["json", "sqlite"]
    cache_size = 100
    num_operations = 200

    print("Backend | Cache Layer | PUT ops/sec | GET ops/sec | Hit Rate | Cache Stats")
    print("-" * 85)

    for backend in backends:
        for cache_enabled in [False, True]:
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = os.path.join(temp_dir, f"{backend}_cache_test")

                config = CacheConfig(
                    cache_dir=cache_dir,
                    metadata_backend=backend,
                    max_cache_size_mb=5000,
                    enable_memory_cache=cache_enabled,
                    memory_cache_type="lru",
                    memory_cache_maxsize=100,
                    memory_cache_ttl_seconds=300,
                    memory_cache_stats=True,
                )
                cache = cacheness(config)

                # Populate cache first
                test_data = create_test_data("small")
                cache_keys = []
                for i in range(cache_size):
                    key = cache.put(test_data, test_id=i)
                    cache_keys.append(key)

                # Benchmark repeated GET operations (80% hot data pattern)
                hot_keys = cache_keys[: int(cache_size * 0.2)]  # 20% of keys are "hot"

                start = time.time()
                for _ in range(num_operations):
                    # 80% chance to access hot data, 20% chance to access cold data
                    if random.random() < 0.8:
                        key = random.choice(hot_keys)
                    else:
                        key = random.choice(cache_keys)
                    result = cache.get(cache_key=key)
                    assert result is not None

                get_time = time.time() - start
                get_ops_per_sec = num_operations / get_time

                # Get cache statistics
                stats = cache.get_stats()
                cache_hit_rate = stats.get("memory_cache_hit_rate", "N/A")
                cache_size_info = f"{stats.get('memory_cache_size', 'N/A')}/{stats.get('memory_cache_maxsize', 'N/A')}"

                cache_status = "ENABLED " if cache_enabled else "DISABLED"
                print(
                    f"{backend:7} | {cache_status:11} | {'N/A':8}     | {get_ops_per_sec:8.0f}    | {cache_hit_rate:8}  | {cache_size_info}"
                )
                cache.close()


def benchmark_realistic_workload_patterns():
    """Test 3: Realistic workload patterns with different access distributions."""
    print("\n\n📊 Test 3: Realistic Workload Patterns")
    print("=" * 80)
    print(
        "Testing different access patterns: Sequential, Random, Hot-Spot (80/20 rule)"
    )
    print()

    patterns = [
        ("Sequential", lambda keys, n: [keys[i % len(keys)] for i in range(n)]),
        ("Random", lambda keys, n: [random.choice(keys) for _ in range(n)]),
        (
            "Hot-Spot",
            lambda keys, n: [
                random.choice(keys[: len(keys) // 5])
                if random.random() < 0.8
                else random.choice(keys)
                for _ in range(n)
            ],
        ),
    ]

    cache_size = 200
    num_operations = 500

    print("Pattern     | Backend | Cache Layer | ops/sec | Cache Hit Rate")
    print("-" * 60)

    for pattern_name, pattern_func in patterns:
        for backend in ["json", "sqlite"]:
            for cache_enabled in [False, True]:
                with tempfile.TemporaryDirectory() as temp_dir:
                    cache_dir = os.path.join(temp_dir, f"pattern_test_{backend}")

                    config = CacheConfig(
                        cache_dir=cache_dir,
                        metadata_backend=backend,
                        max_cache_size_mb=5000,
                        enable_memory_cache=cache_enabled,
                        memory_cache_type="lru",
                        memory_cache_maxsize=200,  # 20% of cache size
                        memory_cache_ttl_seconds=300,
                        memory_cache_stats=True,
                    )
                    cache = cacheness(config)

                    # Populate cache
                    test_data = create_test_data("small")
                    cache_keys = []
                    for i in range(cache_size):
                        key = cache.put(test_data, test_id=i)
                        cache_keys.append(key)

                    # Generate access pattern
                    access_keys = pattern_func(cache_keys, num_operations)

                    # Benchmark the pattern
                    start = time.time()
                    for key in access_keys:
                        result = cache.get(cache_key=key)
                        assert result is not None

                    pattern_time = time.time() - start
                    ops_per_sec = num_operations / pattern_time

                    # Get cache statistics
                    stats = cache.get_stats()
                    hit_rate = stats.get("memory_cache_hit_rate", "N/A")

                    cache_status = "ON " if cache_enabled else "OFF"
                    print(
                        f"{pattern_name:11} | {backend:7} | {cache_status:11} | {ops_per_sec:7.0f} | {hit_rate}"
                    )
                    cache.close()


def benchmark_scaling_characteristics():
    """Test 4: How backends scale with increasing cache sizes."""
    print("\n\n📈 Test 4: Backend Scaling Characteristics")
    print("=" * 80)
    print("Testing how list_entries() performance scales with cache size")
    print()

    sizes = [10, 50, 100, 200, 500]
    backends = ["json", "sqlite", "sqlite_memory"]

    print(
        "Cache Size | SQLite-Mem (ms) | JSON (ms)   | SQLite (ms) | SQLite/JSON Ratio"
    )
    print("-" * 75)

    for size in sizes:
        results = {}

        for backend in backends:
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = os.path.join(temp_dir, f"scale_test_{backend}")

                config = CacheConfig(
                    cache_dir=cache_dir,
                    metadata_backend=backend,
                    max_cache_size_mb=10000,
                    enable_memory_cache=False,  # Test raw performance
                )
                cache = cacheness(config)

                # Populate cache
                test_data = create_test_data("small")
                for i in range(size):
                    cache.put(test_data, test_id=i)

                # Time list operation
                start = time.time()
                cache.list_entries()
                list_time = (time.time() - start) * 1000

                results[backend] = list_time
                cache.close()

        # Calculate ratio
        ratio = results["sqlite"] / results["json"] if results["json"] > 0 else 0

        print(
            f"{size:10d} | {results['sqlite_memory']:8.1f}       | {results['json']:8.1f}    | {results['sqlite']:8.1f}     | {ratio:8.1f}x"
        )


def benchmark_memory_cache_effectiveness():
    """Test 5: Detailed memory cache effectiveness analysis."""
    print("\n\n🎯 Test 5: Memory Cache Effectiveness Analysis")
    print("=" * 80)
    print("Detailed analysis of memory cache layer benefits for repeated operations")
    print()

    backend = "sqlite"  # Use SQLite as it benefits most from caching
    cache_size = 100

    for cache_enabled in [False, True]:
        cache_status = "ENABLED" if cache_enabled else "DISABLED"
        print(f"\n🔍 Memory Cache {cache_status}:")

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "effectiveness_test")

            config = CacheConfig(
                cache_dir=cache_dir,
                metadata_backend=backend,
                max_cache_size_mb=5000,
                enable_memory_cache=cache_enabled,
                memory_cache_type="lru",
                memory_cache_maxsize=100,
                memory_cache_ttl_seconds=300,
                memory_cache_stats=True,
            )
            cache = cacheness(config)

            # Populate cache
            test_data = create_test_data("small")
            cache_keys = []
            for i in range(cache_size):
                key = cache.put(test_data, test_id=i)
                cache_keys.append(key)

            # Test repeated get_entry calls on same keys
            test_keys = cache_keys[:20]  # Test with subset of keys

            times = []
            for round_num in range(5):
                start = time.time()
                for key in test_keys:
                    result = cache.get(cache_key=key)
                    assert result is not None
                round_time = (time.time() - start) * 1000
                times.append(round_time)
                print(
                    f"   Round {round_num + 1}: {round_time:6.1f}ms ({len(test_keys)} gets)"
                )

            # Show statistics
            avg_time = sum(times) / len(times)
            first_vs_avg = times[0] / avg_time if avg_time > 0 else 1

            print(f"   Average: {avg_time:6.1f}ms")
            print(f"   First vs Avg ratio: {first_vs_avg:4.1f}x")

            if cache_enabled:
                stats = cache.get_stats()
                hit_rate = stats.get("memory_cache_hit_rate", 0)
                cache_size_used = stats.get("memory_cache_size", 0)
                print(f"   Cache hit rate: {hit_rate:.3f}")
                print(f"   Cache utilization: {cache_size_used}/100")
            cache.close()


def main():
    """Run comprehensive backend benchmark suite."""
    print("🏆 Cacheness Comprehensive Backend Benchmark")
    print("=" * 80)
    print("Testing all backend types and memory cache layer configurations")
    print("This may take several minutes to complete...")
    print()

    try:
        benchmark_raw_backend_performance()
        benchmark_memory_cache_layer_impact()
        benchmark_realistic_workload_patterns()
        benchmark_scaling_characteristics()
        benchmark_memory_cache_effectiveness()

        print("\n\n🎯 Summary and Recommendations")
        print("=" * 80)
        print("📋 Backend Selection Guide:")
        print("• SQLite In-Memory: Fast ephemeral caching, no persistence")
        print(
            "• JSON Backend: Good for small caches (<500 entries), single-process apps"
        )
        print("• SQLite Backend: Best for large caches (>500 entries), production apps")
        print()
        print("🚀 Memory Cache Layer:")
        print("• Significant benefit for SQLite backend (database overhead reduction)")
        print("• Moderate benefit for JSON backend (file I/O reduction)")
        print("• Not applicable to SQLite In-Memory (already in-memory)")
        print()
        print("⚡ Performance Patterns:")
        print("• Hot-spot access patterns benefit most from memory cache layer")
        print("• Sequential access has minimal caching benefit")
        print("• Random access shows moderate caching benefit")
        print()
        print("✅ Benchmark complete!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")


if __name__ == "__main__":
    main()
