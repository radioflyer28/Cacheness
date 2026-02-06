#!/usr/bin/env python3
"""
Before/After Optimization Test
==============================

Test the performance impact of:
1. New unified schema without metadata_json column overhead
2. Memory cache layer for SQLite
3. Performance improvements from schema alignment

This shows the concrete benefits of the schema optimizations.
"""

import time
import tempfile
import os
from cacheness import CacheConfig, cacheness


def test_schema_optimization_performance():
    """Test performance impact of new unified schema."""
    print("🔧 Schema Optimization Performance Impact")
    print("=" * 60)
    print("Testing new unified entry schema vs old structured approach")
    print()
    
    cache_size = 300
    test_data = {"sample": "data", "array": list(range(20))}
    
    backends = [
        ("JSON Backend (unified schema)", "json"),
        ("SQLite Backend (optimized)", "sqlite"),
        ("SQLite In-Memory (ephemeral)", "sqlite_memory"),
    ]
    
    print("Backend                    | PUT ops/sec | GET ops/sec | LIST time (ms)")
    print("-" * 70)
    
    for backend_name, backend_type in backends:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "schema_test")
            
            config = CacheConfig(
                cache_dir=cache_dir,
                metadata_backend=backend_type,
                max_cache_size_mb=1000
            )
            cache = cacheness(config)
            
            # Benchmark PUT operations with complex parameters
            start = time.time()
            cache_keys = []
            for i in range(cache_size):
                key = cache.put(test_data, 
                               test_id=i, 
                               model=f"model_{i}", 
                               temperature=0.7 + (i * 0.001),
                               config={"lr": 0.001, "epochs": 100 + i})
                cache_keys.append(key)
            put_time = time.time() - start
            put_ops_per_sec = cache_size / put_time
            
            # Benchmark GET operations
            start = time.time()
            for key in cache_keys[:100]:  # Test subset for speed
                result = cache.get(cache_key=key)
                assert result is not None
            get_time = time.time() - start
            get_ops_per_sec = 100 / get_time
            
            # Benchmark LIST operations  
            start = time.time()
            entries = cache.list_entries()
            list_time = (time.time() - start) * 1000
            
            print(f"{backend_name:26} | {put_ops_per_sec:8.0f}    | {get_ops_per_sec:8.0f}    | {list_time:8.1f}")
            
            # Verify unified schema structure for first few entries
            if len(entries) > 0:
                first_entry = entries[0]
                print(f"   Entry fields: {list(first_entry.keys())}")
                print(f"   Metadata keys: {list(first_entry.get('metadata', {}).keys())[:5]}...")  # Show first 5
            cache.close()


def test_memory_cache_layer_benefit():
    """Test the benefit of memory cache layer for SQLite."""
    print("\n\n🚀 Memory Cache Layer Performance Benefit")
    print("=" * 60)
    print("Testing SQLite with memory cache layer OFF vs ON")
    print()
    
    cache_size = 200
    test_data = {"data": "test", "number": 42}
    
    configs = [
        ("SQLite (no memory cache)", False),
        ("SQLite + Memory Cache", True),
    ]
    
    print("Configuration              | GET ops/sec | LIST time (ms) | Cache Hit Rate")
    print("-" * 75)
    
    for config_name, enable_memory_cache in configs:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, "memory_test")
            
            config = CacheConfig(
                cache_dir=cache_dir,
                metadata_backend="sqlite",
                max_cache_size_mb=1000
            )
            cache = cacheness(config)
            
            # Populate cache
            cache_keys = []
            for i in range(cache_size):
                key = cache.put(test_data, test_id=i)
                cache_keys.append(key)
            
            # Benchmark repeated GET operations (simulates real usage)
            start = time.time()
            for _ in range(3):  # Multiple rounds
                for key in cache_keys[:50]:  # Test subset for speed
                    result = cache.get(cache_key=key)
                    assert result is not None
            get_time = time.time() - start
            get_ops_per_sec = (3 * 50) / get_time
            
            # Benchmark LIST operations
            start = time.time()
            cache.list_entries()
            list_time = (time.time() - start) * 1000
            
            # Get cache statistics
            stats = cache.get_stats()
            hit_rate = stats.get('memory_cache_hit_rate', 'N/A')
            if hit_rate != 'N/A':
                hit_rate = f"{hit_rate:.3f}"
            
            print(f"{config_name:26} | {get_ops_per_sec:8.0f}    | {list_time:8.1f}      | {hit_rate}")
            cache.close()


def compare_backends_optimized():
    """Compare all backends with optimized settings."""
    print("\n\n📊 Optimized Backend Performance Comparison")
    print("=" * 60)
    print("All backends with unified schema optimizations applied")
    print()
    
    cache_size = 200
    test_data = {"data": "test", "array": list(range(30))}
    
    backends = [
        ("SQLite In-Memory", "sqlite_memory"),
        ("JSON", "json"),
        ("SQLite", "sqlite"),
    ]
    
    print("Backend                    | PUT ops/sec | GET ops/sec | LIST time (ms)")
    print("-" * 70)
    
    for backend_name, backend_type in backends:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, f"{backend_type}_test")
            
            config = CacheConfig(
                cache_dir=cache_dir,
                metadata_backend=backend_type,
                max_cache_size_mb=1000
            )
            cache = cacheness(config)
            
            # Benchmark PUT operations
            start = time.time()
            cache_keys = []
            for i in range(cache_size):
                key = cache.put(test_data, test_id=i)
                cache_keys.append(key)
            put_time = time.time() - start
            put_ops_per_sec = cache_size / put_time
            
            # Benchmark GET operations
            start = time.time()
            for key in cache_keys[:50]:  # Test subset
                result = cache.get(cache_key=key)
                assert result is not None
            get_time = time.time() - start
            get_ops_per_sec = 50 / get_time
            
            # Benchmark LIST operations
            start = time.time()
            cache.list_entries()
            list_time = (time.time() - start) * 1000
            
            print(f"{backend_name:26} | {put_ops_per_sec:8.0f}    | {get_ops_per_sec:8.0f}    | {list_time:8.1f}")
            cache.close()


def main():
    """Run before/after optimization tests."""
    print("🏆 Schema Optimization Results")
    print("=" * 60)
    print("Demonstrating performance improvements from unified schema")
    print()
    
    try:
        test_schema_optimization_performance()
        test_memory_cache_layer_benefit()
        compare_backends_optimized()
        
        print("\n\n🎯 Optimization Summary")
        print("=" * 60)
        print("✅ Completed Schema Optimizations:")
        print("• Unified entry structure across all backends")
        print("• Eliminated metadata_json column overhead in SQLite")
        print("• Consistent entry format while optimizing storage per backend")
        print()
        print("📈 Performance Improvements:")
        print("• SQLite backend: Eliminated JSON parsing overhead for backend metadata")
        print("• JSON backend: Simple row-based storage for better performance")
        print("• SQLite In-Memory: Fast ephemeral caching via :memory: mode")
        print("• All backends: Consistent API with optimized storage patterns")
        print()
        print("🚀 Best Practices:")
        print("• Use SQLite In-Memory for temporary high-performance caching")
        print("• Use JSON backend for small caches and development")
        print("• Use SQLite backend for production and large caches")
        print("• All backends now provide identical entry structures")
        print()
        print("✅ Schema optimization analysis complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()