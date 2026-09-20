#!/usr/bin/env python3
"""
Test script to verify memory backend performance
"""

import time
import tempfile
import os
from cacheness import CacheConfig, cacheness


def test_memory_backend():
    """Test the memory backend performance in isolation."""
    print("🧪 Memory Backend Performance Test")
    print("=" * 50)
    
    # Test pure memory backend without any file I/O
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "memory_test")
        
        config = CacheConfig(
            cache_dir=cache_dir,
            metadata_backend="memory"
        )
        cache = cacheness(config)
        
        # Test data
        test_data = {"test": "simple data", "value": 42}
        
        # Test PUT performance  
        num_ops = 1000
        keys = []
        
        start = time.time()
        for i in range(num_ops):
            key = cache.put(test_data, test_id=i, memory_test=True)
            keys.append(key)
        put_time = time.time() - start
        put_ops_per_sec = num_ops / put_time
        
        print(f"PUT operations: {put_ops_per_sec:.0f} ops/sec")
        
        # Test GET performance
        start = time.time()
        for key in keys:
            result = cache.get(cache_key=key)
            assert result is not None
        get_time = time.time() - start
        get_ops_per_sec = num_ops / get_time
        
        print(f"GET operations: {get_ops_per_sec:.0f} ops/sec")
        
        # Test list_entries performance (first call - should be slow)
        start = time.time()
        entries = cache.list_entries()
        first_list_time = (time.time() - start) * 1000
        
        print(f"First list_entries(): {first_list_time:.1f}ms ({len(entries)} entries)")
        
        # Test list_entries performance (cached call - should be fast)
        start = time.time()
        entries = cache.list_entries()
        cached_list_time = (time.time() - start) * 1000
        
        print(f"Cached list_entries(): {cached_list_time:.1f}ms ({len(entries)} entries)")
        print(f"Cache speedup: {first_list_time / cached_list_time:.1f}x")
        
        # Test stats
        start = time.time()
        stats = cache.get_stats()
        stats_time = (time.time() - start) * 1000
        
        print(f"get_stats(): {stats_time:.1f}ms")
        print(f"Stats: {stats}")


if __name__ == "__main__":
    test_memory_backend()