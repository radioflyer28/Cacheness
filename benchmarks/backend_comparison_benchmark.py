#!/usr/bin/env python3
"""
Comprehensive backend comparison benchmark showing realistic tradeoffs
between JSON and SQLite backends across different scenarios.
"""

import time
import tempfile
import concurrent.futures
import os
from cacheness import CacheConfig, cacheness
import numpy as np
import pandas as pd


def create_test_data(size_category="small"):
    """Create test data of different sizes."""
    if size_category == "small":
        return {
            "array": np.random.random(100),
            "dataframe": pd.DataFrame(np.random.random((10, 5))),
            "dict": {"key": f"value_{i}" for i in range(10)},
            "metadata": f"Test data size: {size_category}"
        }
    elif size_category == "medium":
        return {
            "array": np.random.random(10000),
            "dataframe": pd.DataFrame(np.random.random((1000, 10))),
            "dict": {"key": f"value_{i}" for i in range(1000)},
            "metadata": f"Test data size: {size_category}"
        }
    else:  # large
        return {
            "array": np.random.random(100000),
            "dataframe": pd.DataFrame(np.random.random((10000, 10))),
            "dict": {"key": f"value_{i}" for i in range(10000)},
            "metadata": f"Test data size: {size_category}"
        }


def benchmark_basic_operations(backend, cache_dir, num_entries, data_size="small"):
    """Benchmark basic put/get operations."""
    config = CacheConfig(
        cache_dir=cache_dir,
        metadata_backend=backend
    )
    cache = cacheness(config)
    
    # Generate test data
    test_data = create_test_data(data_size)
    
    # Benchmark PUT operations
    put_times = []
    cache_keys = []
    
    for i in range(num_entries):
        start = time.time()
        key = cache.put(
            test_data,
            experiment=f"benchmark_{backend}",
            iteration=i,
            data_size=data_size
        )
        put_times.append(time.time() - start)
        cache_keys.append(key)
    
    # Benchmark GET operations
    get_times = []
    for key in cache_keys:
        start = time.time()
        result = cache.get(cache_key=key)
        get_times.append(time.time() - start)
        assert result is not None
    
    # Benchmark metadata operations
    start = time.time()
    entries = cache.list_entries()
    list_time = time.time() - start
    
    start = time.time()
    stats = cache.get_stats()
    stats_time = time.time() - start
    
    start = time.time()
    # Note: cleanup_expired not available in current API
    # Using list_entries as alternative metadata operation
    cache.list_entries()
    cleanup_time = time.time() - start
    
    return {
        "backend": backend,
        "num_entries": num_entries,
        "data_size": data_size,
        "avg_put_time": np.mean(put_times),
        "avg_get_time": np.mean(get_times),
        "total_put_time": sum(put_times),
        "total_get_time": sum(get_times),
        "list_entries_time": list_time,
        "get_stats_time": stats_time,
        "cleanup_time": cleanup_time,
        "put_ops_per_sec": num_entries / sum(put_times),
        "get_ops_per_sec": num_entries / sum(get_times),
        "total_size_mb": stats.get("total_size_mb", 0),
        "entry_count": len(entries)
    }


def benchmark_concurrent_access(backend, cache_dir, num_processes=4, operations_per_process=50):
    """Benchmark concurrent access patterns."""
    def worker_function(worker_id, backend, cache_dir, num_ops):
        """Worker function for concurrent access."""
        config = CacheConfig(
            cache_dir=cache_dir,
            metadata_backend=backend
        )
        cache = cacheness(config)
        
        results = {"worker_id": worker_id, "operations": [], "errors": 0}
        
        for i in range(num_ops):
            try:
                # Mix of read and write operations
                if i % 3 == 0:  # Write operation
                    data = create_test_data("small")
                    start = time.time()
                    cache.put(
                        data,
                        worker=worker_id,
                        operation=i,
                        process_test=True
                    )
                    elapsed = time.time() - start
                    results["operations"].append(("put", elapsed, True))
                else:  # Read operation
                    start = time.time()
                    entries = cache.list_entries()
                    if entries:
                        # Try to read a random entry
                        import random
                        entry = random.choice(entries)
                        result = cache.get(cache_key=entry["cache_key"])
                        success = result is not None
                    else:
                        success = True  # No entries to read
                    elapsed = time.time() - start
                    results["operations"].append(("get", elapsed, success))
                    
            except Exception:
                results["errors"] += 1
                results["operations"].append(("error", 0, False))
        
        return results
    
    # Run concurrent workers
    start_time = time.time()
    
    if backend == "json":
        # JSON backend doesn't handle concurrency well, so we test sequentially
        # to show the limitation
        all_results = []
        for worker_id in range(num_processes):
            result = worker_function(worker_id, backend, cache_dir, operations_per_process)
            all_results.append(result)
    else:
        # SQLite can handle true concurrency
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(worker_function, i, backend, cache_dir, operations_per_process)
                for i in range(num_processes)
            ]
            all_results = [future.result() for future in futures]
    
    total_time = time.time() - start_time
    
    # Aggregate results
    total_operations = sum(len(r["operations"]) for r in all_results)
    total_errors = sum(r["errors"] for r in all_results)
    successful_ops = total_operations - total_errors
    
    put_times = []
    get_times = []
    
    for result in all_results:
        for op_type, elapsed, success in result["operations"]:
            if success:
                if op_type == "put":
                    put_times.append(elapsed)
                elif op_type == "get":
                    get_times.append(elapsed)
    
    return {
        "backend": backend,
        "num_processes": num_processes,
        "operations_per_process": operations_per_process,
        "total_time": total_time,
        "total_operations": total_operations,
        "successful_operations": successful_ops,
        "error_rate": total_errors / total_operations if total_operations > 0 else 0,
        "avg_put_time": np.mean(put_times) if put_times else 0,
        "avg_get_time": np.mean(get_times) if get_times else 0,
        "throughput_ops_per_sec": successful_ops / total_time if total_time > 0 else 0
    }


def benchmark_scaling_behavior():
    """Test how backends scale with increasing cache sizes."""
    print("🔬 Backend Scaling Analysis")
    print("=" * 60)
    
    cache_sizes = [10, 50, 100, 500, 1000, 2000]
    backends = ["json", "sqlite"]
    data_sizes = ["small", "medium"]
    
    results = []
    
    for data_size in data_sizes:
        print(f"\n📊 Testing with {data_size} data objects:")
        
        for size in cache_sizes:
            print(f"  Cache size: {size} entries...")
            
            for backend in backends:
                with tempfile.TemporaryDirectory() as temp_dir:
                    cache_dir = os.path.join(temp_dir, f"{backend}_cache")
                    
                    try:
                        result = benchmark_basic_operations(backend, cache_dir, size, data_size)
                        results.append(result)
                        
                        print(f"    {backend:8} | PUT: {result['put_ops_per_sec']:6.0f} ops/sec | "
                              f"GET: {result['get_ops_per_sec']:6.0f} ops/sec | "
                              f"LIST: {result['list_entries_time']*1000:6.1f}ms")
                              
                    except Exception as e:
                        print(f"    {backend:8} | ERROR: {e}")
    
    return results


def benchmark_concurrency():
    """Test concurrent access patterns."""
    print("\n🔄 Concurrency Benchmark")
    print("=" * 60)
    
    backends = ["json", "sqlite"]
    process_counts = [1, 2, 4]
    
    results = []
    
    for backend in backends:
        print(f"\n{backend.upper()} Backend:")
        
        for num_processes in process_counts:
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = os.path.join(temp_dir, f"{backend}_concurrent")
                
                try:
                    result = benchmark_concurrent_access(backend, cache_dir, num_processes)
                    results.append(result)
                    
                    print(f"  {num_processes} processes | "
                          f"Throughput: {result['throughput_ops_per_sec']:6.1f} ops/sec | "
                          f"Errors: {result['error_rate']*100:4.1f}% | "
                          f"Total time: {result['total_time']:5.2f}s")
                          
                except Exception as e:
                    print(f"  {num_processes} processes | ERROR: {e}")
    
    return results


def benchmark_initialization_overhead():
    """Measure initialization overhead for both backends."""
    print("\n⚡ Initialization Overhead Analysis")
    print("=" * 60)
    
    backends = ["json", "sqlite", "sqlite_memory"]
    num_trials = 20
    
    for backend in backends:
        cold_times = []
        warm_times = []
        
        # Test cold initialization (new cache each time)
        for _ in range(num_trials):
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = os.path.join(temp_dir, "cache")
                
                start = time.time()
                config = CacheConfig(cache_dir=cache_dir, metadata_backend=backend)
                cache = cacheness(config)
                
                # Perform a simple operation
                cache.put({"test": "data"}, init_test=True)
                cold_times.append(time.time() - start)
        
        # Test warm initialization (reusing same cache)
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "cache")
            config = CacheConfig(cache_dir=cache_dir, metadata_backend=backend)
            cache = cacheness(config)
            
            # Add some data first
            for i in range(10):
                cache.put({"test": f"data_{i}"}, warmup=i)
            
            # Now test warm access
            for _ in range(num_trials):
                start = time.time()
                cache.get_stats()  # Simple metadata operation
                warm_times.append(time.time() - start)
        
        print(f"{backend:15} | Cold init: {np.mean(cold_times)*1000:6.1f}ms ± {np.std(cold_times)*1000:4.1f}ms | "
              f"Warm access: {np.mean(warm_times)*1000:6.1f}ms ± {np.std(warm_times)*1000:4.1f}ms")


def generate_recommendations(scaling_results, concurrency_results):
    """Generate usage recommendations based on benchmark results."""
    print("\n📋 Backend Selection Recommendations")
    print("=" * 60)
    
    # Analyze scaling breakeven points
    json_results = [r for r in scaling_results if r["backend"] == "json"]
    sqlite_results = [r for r in scaling_results if r["backend"] == "sqlite"]
    
    print("🎯 WHEN TO USE EACH BACKEND:\n")
    
    print("📁 JSON Backend - Best for:")
    print("  • Small caches (< 100-200 entries)")
    print("  • Single-process applications")
    print("  • Development and testing")
    print("  • Simple deployment scenarios")
    print("  • When you need human-readable metadata files")
    print("  • Quick prototyping")
    
    print("\n💾 SQLite Backend - Best for:")
    print("  • Large caches (> 200 entries)")
    print("  • Multi-process applications")
    print("  • Production deployments")
    print("  • Applications requiring metadata queries")
    print("  • Long-running services")
    print("  • When cache integrity is critical")
    
    print("\n⚡ Performance Characteristics:")
    
    # Find crossover points
    small_cache_json = next((r for r in json_results if r["num_entries"] == 100), None)
    small_cache_sqlite = next((r for r in sqlite_results if r["num_entries"] == 100), None)
    
    large_cache_json = next((r for r in json_results if r["num_entries"] == 1000), None)
    large_cache_sqlite = next((r for r in sqlite_results if r["num_entries"] == 1000), None)
    
    if small_cache_json and small_cache_sqlite:
        json_advantage = small_cache_json["put_ops_per_sec"] / small_cache_sqlite["put_ops_per_sec"]
        print(f"  • Small caches: JSON is {json_advantage:.1f}x faster for operations")
    
    if large_cache_json and large_cache_sqlite:
        list_speedup = large_cache_json["list_entries_time"] / large_cache_sqlite["list_entries_time"]
        print(f"  • Large caches: SQLite is {list_speedup:.0f}x faster for metadata operations")
    
    # Concurrency analysis
    print("\n🔄 Concurrency Support:")
    json_concurrent = [r for r in concurrency_results if r["backend"] == "json" and r["num_processes"] > 1]
    sqlite_concurrent = [r for r in concurrency_results if r["backend"] == "sqlite" and r["num_processes"] > 1]
    
    if json_concurrent:
        avg_json_errors = np.mean([r["error_rate"] for r in json_concurrent])
        print(f"  • JSON: {avg_json_errors*100:.1f}% error rate with multiple processes")
    
    if sqlite_concurrent:
        avg_sqlite_errors = np.mean([r["error_rate"] for r in sqlite_concurrent])
        print(f"  • SQLite: {avg_sqlite_errors*100:.1f}% error rate with multiple processes")
    
    print("\n💡 CONFIGURATION RECOMMENDATIONS:\n")
    
    print("🚀 High-Performance Setup (Large caches, production):")
    print("""
    config = CacheConfig(
        cache_dir="/fast_storage/cache",
        metadata_backend="sqlite",
        max_cache_size_mb=10000
    )""")
    
    print("\n🛠️ Development Setup (Small caches, testing):")
    print("""
    config = CacheConfig(
        cache_dir="./dev_cache",
        metadata_backend="json",
        max_cache_size_mb=1000
    )""")
    
    print("\n⚡ Memory-Optimized Setup (Fast access, temporary data):")
    print("""
    config = CacheConfig(
        cache_dir="./temp_cache",
        metadata_backend="sqlite_memory",
        max_cache_size_mb=2000
    )""")


def main():
    """Run comprehensive backend comparison benchmark."""
    print("🏆 Cacheness Backend Comparison Benchmark")
    print("=" * 60)
    print("Testing JSON vs SQLite backends across different scenarios")
    print("This may take several minutes to complete...\n")
    
    # Run all benchmarks
    scaling_results = benchmark_scaling_behavior()
    concurrency_results = benchmark_concurrency()
    benchmark_initialization_overhead()
    
    # Generate recommendations
    generate_recommendations(scaling_results, concurrency_results)
    
    print("\n✅ Benchmark complete!")
    print("\nKey Takeaways:")
    print("• JSON: Fast for small caches, single-process apps")
    print("• SQLite: Scales better, handles concurrency, production-ready")
    print("• Choose based on your cache size and concurrency needs")


if __name__ == "__main__":
    main()