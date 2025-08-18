#!/usr/bin/env python3
"""
Detailed analysis of list_entries() performance across backends.
This benchmark reveals why SQLite appears slow and Memory/JSON have similar times.
"""

import time
import tempfile
import os
from cacheness import CacheConfig, cacheness


def analyze_list_performance():
    """Detailed analysis of list_entries() performance patterns."""
    print("🔍 List Performance Deep Dive")
    print("=" * 60)
    print("NOTE: Memory cache layer DISABLED for JSON/SQLite to test raw backend performance")
    
    backends = ["memory", "json", "sqlite"]
    cache_sizes = [50, 200, 500, 1000]
    
    for size in cache_sizes:
        print(f"\n📊 Testing with {size} entries:")
        
        for backend in backends:
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = os.path.join(temp_dir, f"{backend}_test")
                
                config = CacheConfig(
                    cache_dir=cache_dir,
                    metadata_backend=backend,
                    max_cache_size_mb=5000,  # Large enough for test
                    # Explicitly disable memory cache layer for JSON/SQLite
                    enable_memory_cache=False,
                    memory_cache_stats=False
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
                speedup = first_call_time / second_call_time if second_call_time > 0 else 0
                
                print(f"  {backend:10} | 1st: {first_call_time:6.1f}ms | 2nd: {second_call_time:6.1f}ms | 3rd: {third_call_time:6.1f}ms | Speedup: {speedup:4.1f}x")
                
                # Verify all calls return same number of entries
                assert len(entries1) == len(entries2) == len(entries3) == size


def analyze_list_operations_detail():
    """Analyze what operations are actually happening in list_entries()."""
    print("\n\n🔬 List Operations Analysis")
    print("=" * 60)
    print("Testing with memory cache layer DISABLED for all backends")
    
    # Test with moderate size to see patterns clearly
    size = 500
    
    for backend in ["memory", "json", "sqlite"]:
        print(f"\n🔍 {backend.upper()} Backend Analysis:")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, f"{backend}_analysis")
            
            config = CacheConfig(
                cache_dir=cache_dir,
                metadata_backend=backend,
                max_cache_size_mb=5000,
                # Explicitly disable memory cache layer
                enable_memory_cache=False,
                memory_cache_stats=False
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
                print(f"   Call {call_num + 1}: {call_time:6.1f}ms ({len(entries)} entries)")
            
            # Analysis
            avg_time = sum(times) / len(times)
            first_vs_rest = times[0] / (sum(times[1:]) / len(times[1:])) if len(times) > 1 else 1
            
            print(f"   Average: {avg_time:6.1f}ms")
            print(f"   First vs Rest ratio: {first_vs_rest:4.1f}x")
            
            # Check if results are cached (same object reference) - should be False for JSON/SQLite now
            entries_a = cache.list_entries()
            entries_b = cache.list_entries()
            is_cached = entries_a is entries_b
            print(f"   Result caching: {'Yes' if is_cached else 'No'}")


def compare_scaling_patterns():
    """Compare how each backend scales with cache size."""
    print("\n\n📈 Scaling Pattern Comparison")
    print("=" * 60)
    print("Testing RAW backend performance (no memory cache layer)")
    
    sizes = [10, 50, 100, 200, 500, 1000, 2000]
    
    print("Size     | Memory (ms) | JSON (ms)   | SQLite (ms) | Memory/JSON | SQLite/JSON")
    print("-" * 80)
    
    results = {}
    
    for size in sizes:
        size_results = {}
        
        for backend in ["memory", "json", "sqlite"]:
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_dir = os.path.join(temp_dir, f"scale_test_{backend}")
                
                config = CacheConfig(
                    cache_dir=cache_dir,
                    metadata_backend=backend,
                    max_cache_size_mb=10000,
                    # Disable memory cache layer to test raw performance
                    enable_memory_cache=False,
                    memory_cache_stats=False
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
        
        results[size] = size_results
        
        # Calculate ratios
        memory_time = size_results["memory"]
        json_time = size_results["json"] 
        sqlite_time = size_results["sqlite"]
        
        memory_json_ratio = memory_time / json_time if json_time > 0 else 0
        sqlite_json_ratio = sqlite_time / json_time if json_time > 0 else 0
        
        print(f"{size:4d}     | {memory_time:8.1f}    | {json_time:8.1f}    | {sqlite_time:8.1f}     | {memory_json_ratio:8.2f}    | {sqlite_json_ratio:8.2f}")
    
    return results


if __name__ == "__main__":
    analyze_list_performance()
    analyze_list_operations_detail()  
    results = compare_scaling_patterns()
    
    print("\n\n🎯 Key Insights:")
    print("=" * 60)
    print("• InMemoryBackend: Pure in-memory, no internal result caching")
    print("• JSONBackend: File-based storage, no internal caching")  
    print("• SQLiteBackend: Database storage + JSON parsing overhead per entry")
    print("• Memory cache layer: Can be enabled/disabled for JSON/SQLite backends")
    print("• InMemoryBackend shouldn't have internal caching (fixed in this version)")
    print("\n💡 Recommendations:")
    print("• SQLite: Significant performance overhead due to DB + JSON parsing")
    print("• JSON vs Memory: Should have similar performance for list operations") 
    print("• Enable memory cache layer for JSON/SQLite if doing frequent list_entries() calls")
    print("• InMemoryBackend: Best raw performance, no persistence")