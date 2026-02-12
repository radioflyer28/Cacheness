#!/usr/bin/env python3
"""
Management Operations Benchmark
================================

Benchmark the Phase 3 management operations that were recently added:
  - update_data()      : Replace blob data at an existing cache key
  - touch()            : Reset entry timestamp to extend TTL
  - touch_batch()      : Touch matching entries in bulk
  - delete_where()     : Delete entries matching a filter function
  - delete_matching()  : Delete entries matching key/value metadata
  - get_batch()        : Retrieve multiple entries at once
  - delete_batch()     : Delete multiple entries at once

These operations involve both metadata backend I/O and blob I/O.
This benchmark measures ops/sec for each at various cache sizes
across JSON, SQLite, and SQLite in-memory backends.
"""

import time
import tempfile
import os
import statistics
from typing import List
from cacheness import CacheConfig, cacheness


# ── Helpers ──────────────────────────────────────────────────────────────────


def populate_cache(cache, count: int, prefix_pattern: str = "item") -> List[str]:
    """Populate a cache with test data and return the cache keys."""
    keys = []
    for i in range(count):
        data = {"index": i, "payload": f"data_{i}", "group": f"group_{i % 5}"}
        key = cache.put(data, test_id=i, prefix=prefix_pattern)
        keys.append(key)
    return keys


def time_operation(func, iterations: int = 1) -> float:
    """Time an operation and return ms per iteration."""
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed / iterations


# ── Individual Benchmarks ────────────────────────────────────────────────────


def benchmark_update_data(backends: List[str], cache_sizes: List[int]):
    """Benchmark update_data() — replace blob data at an existing key."""
    print("\n📝 update_data() Benchmark")
    print("-" * 60)

    for size in cache_sizes:
        print(f"\n  Cache size: {size} entries")
        for backend in backends:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend=backend,
                    enable_memory_cache=False,
                )
                cache = cacheness(config)
                keys = populate_cache(cache, size)

                # Update 10 random entries
                update_count = min(10, size)
                new_data = {"updated": True, "payload": "new_value"}

                times = []
                for i in range(update_count):
                    t = time_operation(
                        lambda k=keys[i]: cache.update_data(new_data, cache_key=k)
                    )
                    times.append(t)

                avg = statistics.mean(times)
                ops_sec = 1000.0 / avg if avg > 0 else 0
                print(f"    {backend:15} {avg:8.2f}ms avg  ({ops_sec:.0f} ops/sec)")
                cache.close()


def benchmark_touch(backends: List[str], cache_sizes: List[int]):
    """Benchmark touch() — reset entry timestamp."""
    print("\n👆 touch() Benchmark")
    print("-" * 60)

    for size in cache_sizes:
        print(f"\n  Cache size: {size} entries")
        for backend in backends:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend=backend,
                    enable_memory_cache=False,
                )
                cache = cacheness(config)
                keys = populate_cache(cache, size)

                touch_count = min(20, size)
                times = []
                for i in range(touch_count):
                    t = time_operation(lambda k=keys[i]: cache.touch(cache_key=k))
                    times.append(t)

                avg = statistics.mean(times)
                ops_sec = 1000.0 / avg if avg > 0 else 0
                print(f"    {backend:15} {avg:8.2f}ms avg  ({ops_sec:.0f} ops/sec)")
                cache.close()


def benchmark_touch_batch(backends: List[str], cache_sizes: List[int]):
    """Benchmark touch_batch() — touch matching entries."""
    print("\n👆 touch_batch() Benchmark")
    print("-" * 60)

    for size in cache_sizes:
        print(f"\n  Cache size: {size} entries")
        for backend in backends:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend=backend,
                    enable_memory_cache=False,
                    store_full_metadata=True,
                )
                cache = cacheness(config)

                # Store entries with metadata we can filter on
                for i in range(size):
                    data = {"index": i}
                    cache.put(data, test_id=i, project="bench", group=f"g{i % 5}")

                t = time_operation(lambda: cache.touch_batch(project="bench"))
                print(f"    {backend:15} {t:8.2f}ms  (touch all {size} entries)")
                cache.close()


def benchmark_delete_where(backends: List[str], cache_sizes: List[int]):
    """Benchmark delete_where() — filter-based bulk delete."""
    print("\n🗑️  delete_where() Benchmark")
    print("-" * 60)

    for size in cache_sizes:
        print(f"\n  Cache size: {size} entries")
        for backend in backends:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend=backend,
                    enable_memory_cache=False,
                )
                cache = cacheness(config)
                populate_cache(cache, size)

                # Delete entries with even-ish index (filter on description/data)
                start = time.perf_counter()
                deleted = cache.delete_where(lambda e: "0" in e.get("cache_key", ""))
                elapsed = (time.perf_counter() - start) * 1000

                print(f"    {backend:15} {elapsed:8.2f}ms  (deleted {deleted}/{size})")
                cache.close()


def benchmark_delete_matching(backends: List[str], cache_sizes: List[int]):
    """Benchmark delete_matching() — keyword-based bulk delete."""
    print("\n🗑️  delete_matching() Benchmark")
    print("-" * 60)

    for size in cache_sizes:
        print(f"\n  Cache size: {size} entries")
        for backend in backends:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend=backend,
                    enable_memory_cache=False,
                    store_full_metadata=True,
                )
                cache = cacheness(config)

                for i in range(size):
                    data = {"index": i}
                    cache.put(data, test_id=i, category="benchmark")

                start = time.perf_counter()
                deleted = cache.delete_matching(category="benchmark")
                elapsed = (time.perf_counter() - start) * 1000

                print(f"    {backend:15} {elapsed:8.2f}ms  (deleted {deleted}/{size})")
                cache.close()


def benchmark_invalidate(backends: List[str], cache_sizes: List[int]):
    """Benchmark invalidate() — remove a single cache entry."""
    print("\n🗑️  invalidate() Benchmark")
    print("-" * 60)

    for size in cache_sizes:
        print(f"\n  Cache size: {size} entries")
        invalidate_count = min(10, size)

        for backend in backends:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend=backend,
                    enable_memory_cache=False,
                )
                cache = cacheness(config)
                keys = populate_cache(cache, size)

                times = []
                for i in range(invalidate_count):
                    t = time_operation(lambda k=keys[i]: cache.invalidate(cache_key=k))
                    times.append(t)

                avg = statistics.mean(times)
                ops_sec = 1000.0 / avg if avg > 0 else 0
                print(f"    {backend:15} {avg:8.2f}ms avg  ({ops_sec:.0f} ops/sec)")
                cache.close()


def benchmark_clear_all(backends: List[str], cache_sizes: List[int]):
    """Benchmark clear_all() — wipe entire cache."""
    print("\n💥 clear_all() Benchmark")
    print("-" * 60)

    for size in cache_sizes:
        print(f"\n  Cache size: {size} entries")
        for backend in backends:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend=backend,
                    enable_memory_cache=False,
                )
                cache = cacheness(config)
                populate_cache(cache, size)

                start = time.perf_counter()
                cache.clear_all()
                elapsed = (time.perf_counter() - start) * 1000

                remaining = len(cache.list_entries())
                print(f"    {backend:15} {elapsed:8.2f}ms  (remaining: {remaining})")
                cache.close()


def benchmark_get_batch(backends: List[str], cache_sizes: List[int]):
    """Benchmark get_batch() — retrieve multiple entries at once."""
    print("\n📦 get_batch() Benchmark")
    print("-" * 60)

    for size in cache_sizes:
        print(f"\n  Cache size: {size} entries")
        batch_size = min(20, size)

        for backend in backends:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend=backend,
                    enable_memory_cache=False,
                )
                cache = cacheness(config)
                populate_cache(cache, size)

                # Build batch of kwargs to retrieve
                kwargs_list = [
                    {"test_id": i, "prefix": "item"} for i in range(batch_size)
                ]

                times = []
                for _ in range(3):
                    t = time_operation(lambda: cache.get_batch(kwargs_list))
                    times.append(t)

                avg = statistics.mean(times)
                per_item = avg / batch_size
                print(
                    f"    {backend:15} {avg:8.2f}ms total  ({per_item:.2f}ms/item, batch={batch_size})"
                )
                cache.close()


def benchmark_delete_batch(backends: List[str], cache_sizes: List[int]):
    """Benchmark delete_batch() — delete multiple entries at once."""
    print("\n🗑️  delete_batch() Benchmark")
    print("-" * 60)

    for size in cache_sizes:
        print(f"\n  Cache size: {size} entries")
        batch_size = min(20, size)

        for backend in backends:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend=backend,
                    enable_memory_cache=False,
                )
                cache = cacheness(config)
                populate_cache(cache, size)

                kwargs_list = [
                    {"test_id": i, "prefix": "item"} for i in range(batch_size)
                ]

                start = time.perf_counter()
                deleted = cache.delete_batch(kwargs_list)
                elapsed = (time.perf_counter() - start) * 1000

                per_item = elapsed / batch_size if batch_size > 0 else 0
                print(
                    f"    {backend:15} {elapsed:8.2f}ms total  ({per_item:.2f}ms/item, deleted {deleted}/{batch_size})"
                )
                cache.close()


# ── Summary ──────────────────────────────────────────────────────────────────


def benchmark_operation_summary(backends: List[str]):
    """Quick summary: time all ops once on a medium cache."""
    print("\n\n🏆 Operation Summary (100 entries)")
    print("=" * 60)
    print(f"{'Operation':25} ", end="")
    for b in backends:
        print(f"{b:>15}", end="")
    print()
    print("-" * (25 + 15 * len(backends)))

    size = 100
    operations = [
        ("put (create)", "put"),
        ("get (hit)", "get"),
        ("update_data", "update_data"),
        ("touch", "touch"),
        ("invalidate", "invalidate"),
        ("list_entries", "list_entries"),
        ("get_stats", "get_stats"),
        ("get_batch (10)", "get_batch"),
        ("delete_batch (10)", "delete_batch"),
        ("clear_all", "clear_all"),
    ]

    for op_name, op_key in operations:
        print(f"  {op_name:23} ", end="")
        for backend in backends:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend=backend,
                    enable_memory_cache=False,
                )
                cache = cacheness(config)
                keys = populate_cache(cache, size)

                if op_key == "put":
                    t = time_operation(
                        lambda: cache.put({"new": True}, test_id=9999),
                        iterations=5,
                    )
                elif op_key == "get":
                    t = time_operation(
                        lambda: cache.get(test_id=0, prefix="item"),
                        iterations=5,
                    )
                elif op_key == "update_data":
                    t = time_operation(
                        lambda: cache.update_data({"updated": True}, cache_key=keys[0]),
                        iterations=5,
                    )
                elif op_key == "touch":
                    t = time_operation(
                        lambda: cache.touch(cache_key=keys[0]),
                        iterations=5,
                    )
                elif op_key == "list_entries":
                    t = time_operation(lambda: cache.list_entries(), iterations=5)
                elif op_key == "get_stats":
                    t = time_operation(lambda: cache.get_stats(), iterations=10)
                elif op_key == "get_batch":
                    kw_list = [{"test_id": i, "prefix": "item"} for i in range(10)]
                    t = time_operation(lambda: cache.get_batch(kw_list), iterations=3)
                elif op_key == "delete_batch":
                    kw_list = [{"test_id": i, "prefix": "item"} for i in range(10)]
                    t = time_operation(lambda: cache.delete_batch(kw_list))
                elif op_key == "invalidate":
                    t = time_operation(
                        lambda: cache.invalidate(cache_key=keys[0]),
                        iterations=5,
                    )
                elif op_key == "clear_all":
                    # Re-populate so clear_all has entries to clear
                    populate_cache(cache, 20)
                    t = time_operation(lambda: cache.clear_all())
                else:
                    t = 0

                print(f"{t:13.2f}ms", end="")
                cache.close()
        print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("🏆 Management Operations Benchmark")
    print("=" * 60)
    print("Benchmarking Phase 3 management operations across backends")
    print()

    backends = ["json", "sqlite", "sqlite_memory"]
    cache_sizes = [50, 200]

    try:
        benchmark_update_data(backends, cache_sizes)
        benchmark_touch(backends, cache_sizes)
        benchmark_invalidate(backends, cache_sizes)
        benchmark_clear_all(backends, cache_sizes)
        benchmark_get_batch(backends, cache_sizes)
        benchmark_delete_batch(backends, cache_sizes)
        benchmark_delete_where(backends, cache_sizes)
        benchmark_delete_matching(backends, cache_sizes)
        benchmark_touch_batch(backends, cache_sizes)
        benchmark_operation_summary(backends)

        print()
        print()
        print("🎯 Interpretation Guide")
        print("=" * 60)
        print("• update_data: blob I/O + metadata update (expect ~5-20ms)")
        print("• touch: metadata-only update, no blob I/O (expect <5ms)")
        print("• invalidate: remove one entry — metadata + file delete (expect <10ms)")
        print("• clear_all: wipe all entries — scales with cache size")
        print("• get_batch/delete_batch: N sequential ops (linear scaling)")
        print("• delete_where: full scan + filter (O(n) in cache size)")
        print(
            "• delete_matching: query_meta fast path on SQLite with store_full_metadata"
        )
        print("• touch_batch: scan + filter + N touches (most expensive)")
        print()
        print("⚠️  Regressions to watch for:")
        print("• update_data >50ms suggests handler or blob write bottleneck")
        print("• touch >10ms suggests metadata backend overhead")
        print("• delete_where on SQLite >>3x JSON indicates query overhead")
        print()
        print("✅ Benchmark complete!")

    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
