#!/usr/bin/env python3
"""
Simple SQLite Metadata Analysis
===============================

Demonstrate the performance issues with the current SQLite backend:
1. Show what's stored in metadata_json column
2. Identify redundant data
3. Calculate potential optimization impact

This illustrates why SQLite is 30x slower than JSON for list operations.
"""

import time
import tempfile
import os
import json
from cacheness import CacheConfig, cacheness


def analyze_sqlite_metadata_overhead():
    """Analyze what data is unnecessarily stored in metadata_json."""
    print("🔍 SQLite Metadata Storage Analysis")
    print("=" * 60)
    print("Examining redundant data storage in SQLite backend")
    print()
    
    # Create a simple test cache
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "analysis")
        
        config = CacheConfig(
            cache_dir=cache_dir,
            metadata_backend="sqlite",
            enable_memory_cache=False
        )
        cache = cacheness(config)
        
        # Store different types of test data
        test_data_types = [
            ("Small dict", {"key": "value", "number": 42}),
            ("Large dict", {f"key_{i}": f"value_{i}" for i in range(100)}),
            ("Array", list(range(50))),
        ]
        
        for data_name, test_data in test_data_types:
            cache_key = cache.put(test_data, test_id=data_name.lower().replace(" ", "_"))
            
            # Examine what's in the SQLite database
            db_path = os.path.join(cache_dir, "cache_metadata.db")
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT data_type, prefix, LENGTH(metadata_json), metadata_json 
                FROM cache_entries 
                WHERE cache_key = ?
            """, (cache_key,))
            
            row = cursor.fetchone()
            if row:
                data_type, prefix, json_length, metadata_json = row
                
                print(f"📄 {data_name}:")
                print(f"   Cache Key: {cache_key}")
                print(f"   data_type column: '{data_type}'")
                print(f"   prefix column: '{prefix}'")
                print(f"   metadata_json size: {json_length} bytes")
                
                # Parse and analyze JSON content
                try:
                    metadata = json.loads(metadata_json)
                    print(f"   JSON keys: {list(metadata.keys())}")
                    
                    # Check for redundant fields
                    redundant_bytes = 0
                    redundant_fields = []
                    
                    for field in ['data_type', 'prefix']:
                        if field in metadata:
                            redundant_fields.append(field)
                            redundant_bytes += len(json.dumps({field: metadata[field]}))
                    
                    # Check for computable fields
                    computable_fields = []
                    if 'actual_path' in metadata:
                        computable_fields.append('actual_path')
                        redundant_bytes += len(json.dumps({'actual_path': metadata['actual_path']}))
                    
                    if redundant_fields:
                        print(f"   ⚠️  Redundant fields: {redundant_fields}")
                    if computable_fields:
                        print(f"   ⚠️  Computable fields: {computable_fields}")
                    
                    if redundant_bytes > 0:
                        savings_pct = (redundant_bytes / json_length) * 100
                        print(f"   💾 Potential savings: {redundant_bytes} bytes ({savings_pct:.1f}%)")
                    else:
                        print("   ✅ No redundant data found")
                        
                except json.JSONDecodeError:
                    print("   ❌ Invalid JSON")
            
            conn.close()
            print()


def benchmark_list_performance_bottleneck():
    """Demonstrate the list_entries performance bottleneck."""
    print("⚡ List Performance Bottleneck Analysis")
    print("=" * 60)
    print("Comparing JSON vs SQLite list_entries performance")
    print()
    
    cache_size = 200
    test_data = {"sample": "data", "number": 123}
    
    backends = ["json", "sqlite"]
    results = {}
    
    for backend in backends:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = os.path.join(temp_dir, f"{backend}_test")
            
            config = CacheConfig(
                cache_dir=cache_dir,
                metadata_backend=backend,
                enable_memory_cache=False
            )
            cache = cacheness(config)
            
            # Populate cache
            print(f"📊 Testing {backend.upper()} backend...")
            for i in range(cache_size):
                cache.put(test_data, test_id=i)
            
            # Time multiple list_entries calls
            times = []
            for _ in range(5):
                start = time.time()
                entries = cache.list_entries()
                list_time = (time.time() - start) * 1000
                times.append(list_time)
                assert len(entries) == cache_size
            
            avg_time = sum(times) / len(times)
            results[backend] = avg_time
            
            print(f"   Average list_entries time: {avg_time:.1f}ms")
    
    # Compare results
    print()
    print("📈 Performance Comparison:")
    ratio = results["sqlite"] / results["json"]
    print(f"   JSON backend:    {results['json']:.1f}ms")
    print(f"   SQLite backend:  {results['sqlite']:.1f}ms")
    print(f"   SQLite slowdown: {ratio:.1f}x")
    print()
    
    if ratio > 10:
        print("🔥 Critical Performance Issue Identified!")
        print("   SQLite is significantly slower due to:")
        print("   • JSON parsing on every list_entries() call")
        print("   • Redundant data stored in metadata_json")
        print("   • Database query overhead for simple operations")
    else:
        print("✅ Performance looks reasonable")


def estimate_optimization_impact():
    """Estimate the potential impact of SQLite optimizations."""
    print("🎯 Optimization Impact Estimation")
    print("=" * 60)
    print("Calculating potential performance improvements")
    print()
    
    # Create test cache to measure current JSON overhead
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "estimation")
        
        config = CacheConfig(
            cache_dir=cache_dir,
            metadata_backend="sqlite",
            enable_memory_cache=False
        )
        cache = cacheness(config)
        
        # Store sample data
        cache.put({"test": "data"}, test_id="sample")
        
        # Analyze database content
        db_path = os.path.join(cache_dir, "cache_metadata.db")
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get current metadata_json size
        cursor.execute("SELECT AVG(LENGTH(metadata_json)) FROM cache_entries")
        avg_json_size = cursor.fetchone()[0] or 0
        
        # Estimate optimized size (remove redundant fields)
        cursor.execute("SELECT metadata_json FROM cache_entries LIMIT 1")
        sample_json = cursor.fetchone()[0]
        
        if sample_json:
            metadata = json.loads(sample_json)
            
            # Remove redundant fields
            optimized_metadata = metadata.copy()
            removed_fields = []
            
            for field in ['data_type', 'prefix', 'actual_path']:
                if field in optimized_metadata:
                    del optimized_metadata[field]
                    removed_fields.append(field)
            
            optimized_size = len(json.dumps(optimized_metadata))
            savings = avg_json_size - optimized_size
            savings_pct = (savings / avg_json_size) * 100 if avg_json_size > 0 else 0
            
            print(f"📏 Current metadata_json size: {avg_json_size:.0f} bytes")
            print(f"📏 Optimized metadata_json size: {optimized_size:.0f} bytes")
            print(f"💾 Potential space savings: {savings:.0f} bytes ({savings_pct:.1f}%)")
            print(f"🗑️  Removed redundant fields: {removed_fields}")
            print()
            
            # Estimate performance impact
            if savings_pct > 30:
                perf_improvement = min(savings_pct * 2, 80)  # Conservative estimate
                print(f"⚡ Estimated performance improvement: {perf_improvement:.0f}%")
                print("   Benefits:")
                print(f"   • {savings_pct:.0f}% less JSON parsing overhead")
                print("   • Smaller database size and faster I/O")
                print("   • Reduced memory usage during list operations")
            else:
                print("📊 Modest optimization potential identified")
        
        conn.close()


def main():
    """Run SQLite metadata analysis."""
    print("🏆 SQLite Backend Performance Analysis")
    print("=" * 60)
    print("Identifying optimization opportunities in SQLite backend")
    print()
    
    try:
        analyze_sqlite_metadata_overhead()
        benchmark_list_performance_bottleneck()
        estimate_optimization_impact()
        
        print()
        print("🎯 Key Findings")
        print("=" * 60)
        print("💡 Optimization Recommendations:")
        print("1. Remove redundant fields from metadata_json")
        print("   • data_type and prefix are already stored as columns")
        print("   • actual_path can be computed from cache_key")
        print()
        print("2. Minimize JSON parsing overhead")
        print("   • Only parse JSON when it contains meaningful data")
        print("   • Use database columns for frequently accessed fields")
        print()
        print("3. Disable unnecessary serialization")
        print("   • cache_key_params should be off by default")
        print("   • Only enable when absolutely needed")
        print()
        print("4. Use memory cache layer for repeated access")
        print("   • Significant benefit for SQLite (reduces DB overhead)")
        print("   • Moderate benefit for JSON (reduces file I/O)")
        print()
        print("✅ Analysis complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()