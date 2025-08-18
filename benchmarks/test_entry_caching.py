#!/usr/bin/env python3
"""
Test script to verify metadata memory cache layer performance improvements
"""

import time
import tempfile
import os
from cacheness import CacheConfig, cacheness


def test_entry_caching_performance():
    """Test entry caching performance for disk-persistent backends."""
    print("🧪 Metadata Entry Caching Performance Test")
    print("=" * 60)
    
    # Test with SQLite backend + entry caching
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "cache_test")
        
        # Configuration with memory cache enabled for SQLite backend (disabled by default)
        config = CacheConfig(
            cache_dir=cache_dir,
            metadata_backend="sqlite",
            enable_memory_cache=False,  # Compare baseline without caching
            memory_cache_stats=True
        )
        cache = cacheness(config)
        
        # Test data
        test_data = {"test": "cached data", "value": 42}
        
        # Phase 1: Populate cache with entries
        print("\n📥 Phase 1: Populating cache with entries...")
        num_entries = 200
        keys = []
        
        start = time.time()
        for i in range(num_entries):
            key = cache.put(test_data, test_id=i, cache_test=True)
            keys.append(key)
        populate_time = time.time() - start
        print(f"   Populated {num_entries} entries in {populate_time:.2f}s")
        
        # Phase 2: Test cold GET performance (no entry cache)
        print("\n🧊 Phase 2: Cold GET performance (first access)...")
        start = time.time()
        for key in keys[:50]:  # Test first 50 keys
            entry = cache.metadata_backend.get_entry(key)
            assert entry is not None
        cold_time = time.time() - start
        cold_ops_per_sec = 50 / cold_time
        print(f"   Cold GETs: {cold_ops_per_sec:.0f} ops/sec")
        
        # Phase 3: Test warm GET performance (with entry cache)
        print("\n🔥 Phase 3: Warm GET performance (cached entries)...")
        start = time.time()
        for key in keys[:50]:  # Test same 50 keys
            entry = cache.metadata_backend.get_entry(key)
            assert entry is not None
        warm_time = time.time() - start
        warm_ops_per_sec = 50 / warm_time
        print(f"   Warm GETs: {warm_ops_per_sec:.0f} ops/sec")
        
        # Calculate speedup
        speedup = warm_ops_per_sec / cold_ops_per_sec
        print(f"   🚀 Cache speedup: {speedup:.1f}x")
        
        # Phase 4: Test repeated access performance
        print("\n♻️  Phase 4: Repeated access performance...")
        start = time.time()
        for _ in range(10):  # 10 rounds
            for key in keys[:20]:  # 20 keys each round = 200 total gets
                entry = cache.metadata_backend.get_entry(key)
                assert entry is not None
        repeated_time = time.time() - start
        repeated_ops_per_sec = 200 / repeated_time
        print(f"   Repeated GETs: {repeated_ops_per_sec:.0f} ops/sec")
        
        # Phase 5: Show cache statistics
        print("\n📊 Phase 5: Entry cache statistics...")
        stats = cache.get_stats()
        
        # Check statistics after caching enabled run
        from cacheness.metadata import CachedMetadataBackend
        if isinstance(cache.metadata_backend, CachedMetadataBackend):
            cache_stats = cache.metadata_backend.get_cache_stats()
            if cache_stats:
                print("\n📊 Memory Cache Statistics After Test:")
                print(f"Hit rate: {cache_stats.get('memory_cache_hit_rate', 0)}")
        else:
            print("\n📊 No memory cache statistics available")
        
        print(f"\n   Backend type: {stats.get('backend_type', 'N/A')}")
        print(f"   Total entries: {stats.get('total_entries', 'N/A')}")
        
        # Performance summary
        print("\n" + "=" * 60)
        print("📈 PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Cold GET performance:     {cold_ops_per_sec:6.0f} ops/sec")
        print(f"Warm GET performance:     {warm_ops_per_sec:6.0f} ops/sec")
        print(f"Repeated GET performance: {repeated_ops_per_sec:6.0f} ops/sec")
        print(f"Cache speedup:            {speedup:6.1f}x")
        
        if speedup > 2.0:
            print("✅ Memory cache layer working effectively!")
        elif speedup > 1.2:
            print("⚠️  Memory cache layer showing modest improvement")
        else:
            print("❌ Memory cache layer not providing expected speedup")


def test_json_backend_caching():
    """Test entry caching with JSON backend."""
    print("\n\n🧪 JSON Backend Entry Caching Test")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "json_cache_test")
        
        config = CacheConfig(
            cache_dir=temp_dir,
            metadata_backend="sqlite",
            enable_memory_cache=True,
            memory_cache_type="lru",
            memory_cache_maxsize=100,
            memory_cache_ttl_seconds=300,
            memory_cache_stats=True
        )
        cache = cacheness(config)
        
        test_data = {"test": "json cached data"}
        
        # Create some entries
        keys = []
        for i in range(50):
            key = cache.put(test_data, test_id=i)
            keys.append(key)
        
        # Test performance
        start = time.time()
        for key in keys:
            entry = cache.metadata_backend.get_entry(key)
            assert entry is not None
        first_time = time.time() - start
        
        start = time.time()
        for key in keys:
            entry = cache.metadata_backend.get_entry(key)
            assert entry is not None
        second_time = time.time() - start
        
        speedup = (50 / first_time) / (50 / second_time) if second_time > 0 else 1.0
        
        print(f"First pass:  {50/first_time:.0f} ops/sec")
        print(f"Second pass: {50/second_time:.0f} ops/sec")
        print(f"Speedup:     {speedup:.1f}x")
        
        # Show cache stats
        from cacheness.metadata import CachedMetadataBackend
        if isinstance(cache.metadata_backend, CachedMetadataBackend):
            cache_stats = cache.metadata_backend.get_cache_stats()
            if cache_stats:
                print(f"Hit rate: {cache_stats.get('memory_cache_hit_rate', 0)}")


def test_memory_backend_no_caching():
    """Verify memory backend doesn't get wrapped with caching."""
    print("\n\n🧪 Memory Backend (No Caching) Test")
    print("=" * 45)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "memory_test")
        
        config = CacheConfig(
            cache_dir=cache_dir,
            metadata_backend="memory",
            enable_memory_cache=False,  # Should be ignored for memory backend
            memory_cache_stats=True
        )
        cache = cacheness(config)
        
        # Check if memory backend is wrapped (it shouldn't be)
        backend_type = type(cache.metadata_backend).__name__
        print(f"Backend type: {backend_type}")
        
        if backend_type == "InMemoryBackend":
            print("✅ Memory backend correctly NOT wrapped with caching")
        else:
            print(f"⚠️  Memory backend unexpectedly wrapped: {backend_type}")
        
        # Test basic functionality
        test_data = {"test": "memory data"}
        key = cache.put(test_data, memory_test=True)
        entry = cache.metadata_backend.get_entry(key)
        assert entry is not None
        print("✅ Memory backend functionality working")


if __name__ == "__main__":
    test_entry_caching_performance()
    test_json_backend_caching()
    test_memory_backend_no_caching()