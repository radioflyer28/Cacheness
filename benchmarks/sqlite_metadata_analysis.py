#!/usr/bin/env python3
"""
SQLite Backend Schema Analysis
==============================

Analyze the current SQLite backend schema efficiency:
1. Show what's stored in each dedicated column
2. Measure column storage overhead
3. Benchmark query performance on dedicated columns
4. Compare list_entries performance across backends

This validates that the schema design (dedicated columns instead of a
monolithic metadata_json blob) is efficient and identifies any remaining
optimization opportunities.
"""

import time
import tempfile
import os
import json
from cacheness import CacheConfig, cacheness


def analyze_sqlite_schema():
    """Analyze current SQLite schema and column utilization."""
    print("🔍 SQLite Schema Analysis")
    print("=" * 60)
    print("Examining dedicated column storage in SQLite backend")
    print()

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "analysis")

        config = CacheConfig(
            cache_dir=cache_dir,
            metadata_backend="sqlite",
            enable_memory_cache=False,
        )
        cache = cacheness(config)

        # Store different types of test data
        import numpy as np

        test_data_types = [
            ("Small dict", {"key": "value", "number": 42}),
            ("Large dict", {f"key_{i}": f"value_{i}" for i in range(100)}),
            ("Numpy array", np.random.random((10, 5))),
        ]

        for data_name, test_data in test_data_types:
            cache.put(test_data, test_id=data_name.lower().replace(" ", "_"))

        # Examine the SQLite database directly
        db_path = os.path.join(cache_dir, "cache_metadata.db")
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get column info
        cursor.execute("PRAGMA table_info(cache_entries)")
        columns = cursor.fetchall()
        print("📋 Schema Columns:")
        for col in columns:
            cid, name, col_type, not_null, default, pk = col
            flags = []
            if pk:
                flags.append("PK")
            if not_null:
                flags.append("NOT NULL")
            if default is not None:
                flags.append(f"DEFAULT={default}")
            flag_str = f" ({', '.join(flags)})" if flags else ""
            print(f"   {name:25} {col_type:10}{flag_str}")

        print()

        # Analyze each stored entry
        cursor.execute("""
            SELECT cache_key, data_type, prefix, file_size,
                   object_type, storage_format, serializer, compression_codec,
                   actual_path, cache_key_params, metadata_dict
            FROM cache_entries
        """)

        rows = cursor.fetchall()
        print(f"📊 Stored Entries ({len(rows)}):")
        for row in rows:
            (
                cache_key, data_type, prefix, file_size,
                object_type, storage_format, serializer, compression_codec,
                actual_path, cache_key_params, metadata_dict,
            ) = row

            print(f"\n   Cache Key: {cache_key}")
            print(f"   data_type: '{data_type}'")
            print(f"   storage_format: '{storage_format}'")
            print(f"   serializer: '{serializer}'")
            print(f"   compression_codec: '{compression_codec}'")
            print(f"   file_size: {file_size} bytes")
            print(f"   object_type: '{object_type}'")

            if cache_key_params:
                params_len = len(cache_key_params)
                print(f"   cache_key_params: {params_len} bytes")
            else:
                print(f"   cache_key_params: NULL (not stored)")

            if metadata_dict:
                meta_len = len(metadata_dict)
                print(f"   metadata_dict: {meta_len} bytes")
            else:
                print(f"   metadata_dict: NULL")

        # Calculate total row sizes
        cursor.execute("""
            SELECT
                COUNT(*) as total_entries,
                AVG(LENGTH(cache_key)) as avg_key_len,
                AVG(file_size) as avg_file_size,
                AVG(COALESCE(LENGTH(cache_key_params), 0)) as avg_params_len,
                AVG(COALESCE(LENGTH(metadata_dict), 0)) as avg_metadata_len,
                AVG(COALESCE(LENGTH(actual_path), 0)) as avg_path_len
            FROM cache_entries
        """)
        stats = cursor.fetchone()
        total, avg_key, avg_size, avg_params, avg_meta, avg_path = stats

        print(f"\n\n📏 Storage Statistics:")
        print(f"   Total entries: {total}")
        print(f"   Avg cache_key length: {avg_key:.0f} chars")
        print(f"   Avg file_size: {avg_size:.0f} bytes")
        print(f"   Avg cache_key_params length: {avg_params:.0f} bytes")
        print(f"   Avg metadata_dict length: {avg_meta:.0f} bytes")
        print(f"   Avg actual_path length: {avg_path:.0f} chars")

        # Check database file size
        db_size = os.path.getsize(db_path)
        print(f"\n   Database file size: {db_size:,} bytes ({db_size / 1024:.1f} KB)")
        print(f"   Bytes per entry: {db_size / total:.0f}")

        conn.close()
        cache.close()


def benchmark_column_query_performance():
    """Benchmark query performance using dedicated columns."""
    print("\n\n⚡ Column Query Performance Benchmark")
    print("=" * 60)
    print("Testing query speed on dedicated columns vs full table scan")
    print()

    import numpy as np

    cache_size = 500

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, "query_bench")

        config = CacheConfig(
            cache_dir=cache_dir,
            metadata_backend="sqlite",
            enable_memory_cache=False,
        )
        cache = cacheness(config)

        # Populate with mixed data types
        for i in range(cache_size):
            if i % 3 == 0:
                data = np.random.random((5, 5))
            elif i % 3 == 1:
                data = {"key": f"value_{i}", "index": i}
            else:
                data = [1, 2, 3, i]
            cache.put(data, test_id=i, batch=f"batch_{i // 50}")

        # Direct SQLite queries on dedicated columns
        db_path = os.path.join(cache_dir, "cache_metadata.db")
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Test 1: Query by data_type column
        start = time.time()
        for _ in range(100):
            cursor.execute(
                "SELECT cache_key, file_size FROM cache_entries WHERE data_type = ?",
                ("array",),
            )
            cursor.fetchall()
        column_query_time = (time.time() - start) * 1000 / 100

        # Test 2: Full table scan
        start = time.time()
        for _ in range(100):
            cursor.execute("SELECT * FROM cache_entries")
            cursor.fetchall()
        full_scan_time = (time.time() - start) * 1000 / 100

        # Test 3: Aggregate queries (used by get_stats)
        start = time.time()
        for _ in range(100):
            cursor.execute("""
                SELECT
                    COUNT(*),
                    SUM(file_size),
                    COUNT(CASE WHEN data_type = 'pandas_dataframe' THEN 1 END),
                    COUNT(CASE WHEN data_type = 'array' THEN 1 END)
                FROM cache_entries
            """)
            cursor.fetchone()
        aggregate_time = (time.time() - start) * 1000 / 100

        print(f"   Column filter query (data_type):  {column_query_time:.2f}ms avg")
        print(f"   Full table scan:                  {full_scan_time:.2f}ms avg")
        print(f"   Aggregate query (get_stats):      {aggregate_time:.2f}ms avg")

        ratio = full_scan_time / column_query_time if column_query_time > 0 else 0
        print(f"\n   Column query speedup vs scan: {ratio:.1f}x")

        conn.close()

        # Test 4: list_entries via cache API
        start = time.time()
        for _ in range(10):
            cache.list_entries()
        list_time = (time.time() - start) * 1000 / 10
        print(f"   cache.list_entries() ({cache_size} entries): {list_time:.1f}ms avg")

        # Test 5: get_stats via cache API
        start = time.time()
        for _ in range(100):
            cache.get_stats()
        stats_time = (time.time() - start) * 1000 / 100
        print(f"   cache.get_stats():                {stats_time:.2f}ms avg")

        cache.close()


def benchmark_list_performance_comparison():
    """Compare list_entries performance across backends."""
    print("\n\n📊 List Performance: JSON vs SQLite")
    print("=" * 60)

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
                enable_memory_cache=False,
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
            cache.close()

    # Compare results
    print()
    print("📈 Performance Comparison:")
    ratio = results["sqlite"] / results["json"] if results["json"] > 0 else 0
    print(f"   JSON backend:    {results['json']:.1f}ms")
    print(f"   SQLite backend:  {results['sqlite']:.1f}ms")
    print(f"   SQLite/JSON ratio: {ratio:.1f}x")
    print()

    if ratio > 5:
        print("⚠️  SQLite is significantly slower for list_entries.")
        print("   Consider enabling memory cache layer for repeated access.")
    elif ratio > 2:
        print("📊 SQLite has moderate overhead compared to JSON.")
        print("   Expected for database-backed storage. Memory cache helps.")
    else:
        print("✅ Performance gap is acceptable.")


def main():
    """Run SQLite schema analysis."""
    print("🏆 SQLite Backend Schema Analysis")
    print("=" * 60)
    print("Analyzing current dedicated-column schema efficiency")
    print()

    try:
        analyze_sqlite_schema()
        benchmark_column_query_performance()
        benchmark_list_performance_comparison()

        print()
        print("🎯 Key Findings")
        print("=" * 60)
        print("Current Schema Benefits:")
        print("• Dedicated columns eliminate JSON parsing overhead")
        print("• Column-based queries are faster than full table scans")
        print("• get_stats() uses SQL aggregates instead of loading all rows")
        print("• cache_key_params stored only when configured (NULL by default)")
        print()
        print("💡 Optimization Notes:")
        print("• metadata_dict column stores query_meta() params as JSON")
        print("• cache_key_params is optional - only stored when enabled")
        print("• actual_path stored for direct file access without recomputation")
        print("• Enable memory cache layer for frequent repeated access patterns")
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