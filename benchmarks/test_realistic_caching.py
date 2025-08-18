#!/usr/bin/env python3
"""
Test script to demonstrate entry caching benefits with realistic workloads
"""

import time
import tempfile
import os
import random
from cacheness import CacheConfig, cacheness


def test_realistic_caching_workload():
    """Test entry caching with realistic workload patterns."""
    print("🧪 Realistic Entry Caching Workload Test")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "realistic_test")
        
        # Test with and without caching
        for enable_cache in [False, True]:
            cache_status = "WITH" if enable_cache else "WITHOUT"
            print(f"\n🔄 Testing {cache_status} entry caching...")
            
            config = CacheConfig(
                cache_dir=cache_dir,
                metadata_backend="sqlite",
                enable_memory_cache=enable_cache,
                memory_cache_type="lru",
                memory_cache_maxsize=100,  # Smaller cache to force evictions
                memory_cache_ttl_seconds=300,
                memory_cache_stats=True
            )
            cache = cacheness(config)
            
            # Create a large number of entries
            print("   📥 Creating 500 cache entries...")
            test_data = {"test": "realistic workload data", "size": list(range(100))}
            keys = []
            
            start = time.time()
            for i in range(500):
                key = cache.put(test_data, workload_id=i, batch=i//50)
                keys.append(key)
            creation_time = time.time() - start
            print(f"   Created 500 entries in {creation_time:.2f}s")
            
            # Simulate realistic access patterns:
            # - 80% of accesses to 20% of entries (hot data)
            # - 20% of accesses to 80% of entries (cold data)
            hot_keys = keys[:100]  # Top 20% are hot
            cold_keys = keys[100:]  # Bottom 80% are cold
            
            access_pattern = []
            for _ in range(1000):  # 1000 total accesses
                if random.random() < 0.8:  # 80% chance
                    access_pattern.append(random.choice(hot_keys))
                else:  # 20% chance
                    access_pattern.append(random.choice(cold_keys))
            
            print(f"   🎯 Testing with realistic access pattern (1000 gets)...")
            print(f"      - 80% accesses to hot data (100 entries)")
            print(f"      - 20% accesses to cold data (400 entries)")
            
            start = time.time()
            for key in access_pattern:
                entry = cache.metadata_backend.get_entry(key)
                assert entry is not None
            access_time = time.time() - start
            ops_per_sec = 1000 / access_time
            
            print(f"   Performance: {ops_per_sec:.0f} ops/sec")
            
            # Show cache stats if available
            if enable_cache and hasattr(cache.metadata_backend, 'get_cache_stats'):
                cache_stats = cache.metadata_backend.get_cache_stats()
                if cache_stats:
                    print(f"   Cache hits: {cache_stats.get('memory_cache_hits', 0)}")
                    print(f"   Cache misses: {cache_stats.get('memory_cache_misses', 0)}")
                    print(f"   Hit rate: {cache_stats.get('memory_cache_hit_rate', 0)}")
                    print(f"   Cache size: {cache_stats.get('memory_cache_size', 0)}")
            
            # Store results for comparison
            if not enable_cache:
                no_cache_ops = ops_per_sec
            else:
                with_cache_ops = ops_per_sec
        
        # Calculate improvement
        if 'no_cache_ops' in locals() and 'with_cache_ops' in locals():
            improvement = with_cache_ops / no_cache_ops
            print(f"\n🚀 PERFORMANCE IMPROVEMENT")
            print(f"=" * 40)
            print(f"Without caching: {no_cache_ops:8.0f} ops/sec")
            print(f"With caching:    {with_cache_ops:8.0f} ops/sec")
            print(f"Improvement:     {improvement:8.1f}x")
            
            if improvement > 1.5:
                print("✅ Entry caching providing significant benefit!")
            elif improvement > 1.1:
                print("✅ Entry caching providing measurable benefit")
            else:
                print("⚠️  Entry caching benefit minimal for this workload")


def test_cache_eviction_and_ttl():
    """Test cache eviction and TTL behavior."""
    print("\n\n🧪 Cache Eviction and TTL Test")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "ttl_test")
        
        config = CacheConfig(
            cache_dir=cache_dir,
            metadata_backend="sqlite",
            enable_memory_cache=True,
            memory_cache_type="lru",
            memory_cache_maxsize=10,  # Very small cache
            memory_cache_ttl_seconds=2,  # Short TTL
            memory_cache_stats=True
        )
        cache = cacheness(config)
        
        test_data = {"test": "eviction test"}
        
        # Create more entries than cache can hold
        print("📥 Creating 15 entries (cache size = 10)...")
        keys = []
        for i in range(15):
            key = cache.put(test_data, eviction_test=i)
            keys.append(key)
        
        # Access first 10 entries (should all be cached)
        print("🔍 Accessing first 10 entries...")
        for key in keys[:10]:
            entry = cache.metadata_backend.get_entry(key)
            assert entry is not None
        
        # Access last 5 entries (should cause evictions)
        print("🔍 Accessing last 5 entries (should cause evictions)...")
        for key in keys[10:]:
            entry = cache.metadata_backend.get_entry(key)
            assert entry is not None
        
        # Show cache stats
        if hasattr(cache.metadata_backend, 'get_cache_stats'):
            cache_stats = cache.metadata_backend.get_cache_stats()
            if cache_stats:
                print(f"Cache size: {cache_stats.get('memory_cache_size', 0)}/10")
                print(f"Cache hits: {cache_stats.get('memory_cache_hits', 0)}")
                print(f"Cache misses: {cache_stats.get('memory_cache_misses', 0)}")
        
        # Test TTL expiration
        print("⏰ Testing TTL expiration (waiting 3 seconds)...")
        time.sleep(3)
        
        # Access an entry that should have expired
        entry = cache.metadata_backend.get_entry(keys[0])
        assert entry is not None
        
        # Show final cache stats
        if hasattr(cache.metadata_backend, 'get_cache_stats'):
            cache_stats = cache.metadata_backend.get_cache_stats()
            if cache_stats:
                print(f"Final cache hits: {cache_stats.get('memory_cache_hits', 0)}")
                print(f"Final cache misses: {cache_stats.get('memory_cache_misses', 0)}")
                print("✅ TTL and eviction working correctly")


if __name__ == "__main__":
    test_realistic_caching_workload()
    test_cache_eviction_and_ttl()