#!/usr/bin/env python3
"""
Compression Benchmark
======================

Benchmark compression codec impact on put/get performance and file size.

Measures:
  - Per-codec compress/decompress speed via compress_pickle.benchmark_codecs()
  - End-to-end put/get with each pickle_compression_codec config
  - Compression ratio vs speed trade-off across data types
  - Codec impact on different data shapes (dict, numpy array, DataFrame)

Available codecs (blosc2-compatible): lz4, lz4hc, zstd, zlib, blosclz
"""

import time
import os
import tempfile
import statistics
import numpy as np
from typing import List, Tuple, Any

from cacheness import CacheConfig, cacheness
from cacheness.config import CompressionConfig
from cacheness.compress_pickle import benchmark_codecs, list_available_codecs

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False


# ── Data Generators ──────────────────────────────────────────────────────────

def make_test_data() -> List[Tuple[str, Any]]:
    """Generate labeled test datasets of varying compressibility."""
    cases = [
        ("small dict", {"a": 1, "b": "hello", "c": [1, 2, 3]}),
        ("large dict", {f"key_{i}": list(range(100)) for i in range(100)}),
        ("numpy random (low compress)", np.random.rand(10_000)),
        ("numpy zeros (high compress)", np.zeros(10_000)),
        ("numpy ints (medium compress)", np.arange(10_000)),
    ]

    if PANDAS_AVAILABLE:
        cases.append((
            "pandas DataFrame",
            pd.DataFrame({f"c{i}": np.random.rand(2000) for i in range(10)})
        ))

    return cases


# ── Benchmarks ───────────────────────────────────────────────────────────────

def benchmark_codec_microbench():
    """Use compress_pickle.benchmark_codecs() for raw codec comparison."""
    print("\n🗜️  Raw Codec Micro-Benchmark (via benchmark_codecs)")
    print("-" * 70)

    codecs = list_available_codecs()[:5]  # Main codecs only
    print(f"  Testing codecs: {', '.join(codecs)}")
    print()

    # Test with a moderately sized dict
    data = {f"key_{i}": list(range(100)) for i in range(50)}

    with tempfile.TemporaryDirectory() as tmp:
        results = benchmark_codecs(data, codecs=codecs, temp_dir=tmp)

    print(f"  {'Codec':12} {'Write ms':>10} {'Read ms':>10} {'Ratio':>8} {'Match':>6}")
    print("  " + "-" * 48)
    for codec, r in results.items():
        if "error" in r:
            print(f"  {codec:12} ERROR: {r['error']}")
        else:
            print(
                f"  {codec:12} "
                f"{r['write_time'] * 1000:10.2f} "
                f"{r['read_time'] * 1000:10.2f} "
                f"{r['compression_ratio']:8.2f}x "
                f"{'✓' if r['data_match'] else '✗':>6}"
            )

    # Test with numpy array
    print()
    print("  (numpy 10k random float64):")
    np_data = np.random.rand(10_000)
    with tempfile.TemporaryDirectory() as tmp:
        results = benchmark_codecs(np_data, codecs=codecs, temp_dir=tmp)

    print(f"  {'Codec':12} {'Write ms':>10} {'Read ms':>10} {'Ratio':>8} {'Match':>6}")
    print("  " + "-" * 48)
    for codec, r in results.items():
        if "error" in r:
            print(f"  {codec:12} ERROR: {r['error']}")
        else:
            print(
                f"  {codec:12} "
                f"{r['write_time'] * 1000:10.2f} "
                f"{r['read_time'] * 1000:10.2f} "
                f"{r['compression_ratio']:8.2f}x "
                f"{'✓' if r['data_match'] else '✗':>6}"
            )


def benchmark_end_to_end_codecs():
    """Benchmark full put/get cycle with different pickle_compression_codec settings."""
    print("\n📦 End-to-End Put/Get by Codec")
    print("-" * 70)

    # lz4 and zstd are valid for both pickle and blosc2 paths
    codecs = ["lz4", "zstd"]
    data_cases = make_test_data()

    for label, data in data_cases:
        print(f"\n  {label}:")
        print(f"    {'Codec':8} {'Put ms':>8} {'Get ms':>8} {'File KB':>8} {'Ratio':>8}")
        print("    " + "-" * 44)

        for codec in codecs:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend="sqlite_memory",
                    enable_memory_cache=False,
                    compression=CompressionConfig(pickle_compression_codec=codec),
                )
                cache = cacheness(config)

                # Put 3 times, take median
                put_times = []
                keys = []
                for i in range(3):
                    start = time.perf_counter()
                    key = cache.put(data, codec=codec, run=i)
                    put_times.append((time.perf_counter() - start) * 1000)
                    keys.append(key)

                # Get 3 times
                get_times = []
                for key in keys:
                    start = time.perf_counter()
                    cache.get(cache_key=key)
                    get_times.append((time.perf_counter() - start) * 1000)

                # File size
                entries = cache.list_entries()
                file_kb = 0
                if entries:
                    fs = entries[0].get("file_size", 0)
                    file_kb = fs / 1024 if fs else 0

                p = statistics.median(put_times)
                g = statistics.median(get_times)

                # Crude compression ratio estimate
                import pickle
                try:
                    raw_size = len(pickle.dumps(data, -1))
                    ratio = raw_size / (file_kb * 1024) if file_kb > 0 else 0
                except Exception:
                    ratio = 0

                print(f"    {codec:8} {p:8.2f} {g:8.2f} {file_kb:8.1f} {ratio:7.1f}x")

                cache.close()


def benchmark_codec_comparison_table():
    """Summary table: best codec per metric for medium data."""
    print("\n\n🏆 Codec Comparison Summary (large dict)")
    print("=" * 60)

    codecs = ["lz4", "zstd"]
    data = {f"key_{i}": list(range(100)) for i in range(100)}

    results = {}
    for codec in codecs:
        with tempfile.TemporaryDirectory() as tmp:
            config = CacheConfig(
                cache_dir=os.path.join(tmp, "cache"),
                metadata_backend="sqlite_memory",
                enable_memory_cache=False,
                compression=CompressionConfig(pickle_compression_codec=codec),
            )
            cache = cacheness(config)

            # 5 iterations
            put_times = []
            get_times = []
            for i in range(5):
                start = time.perf_counter()
                key = cache.put(data, codec=codec, run=i)
                put_times.append((time.perf_counter() - start) * 1000)

                start = time.perf_counter()
                cache.get(cache_key=key)
                get_times.append((time.perf_counter() - start) * 1000)

            entries = cache.list_entries()
            file_kb = 0
            if entries:
                fs = entries[0].get("file_size", 0)
                file_kb = fs / 1024 if fs else 0

            results[codec] = {
                "put": statistics.median(put_times),
                "get": statistics.median(get_times),
                "kb": file_kb,
            }
            cache.close()

    print(f"  {'Codec':8} {'Put ms':>8} {'Get ms':>8} {'File KB':>8} {'Total ms':>8}")
    print("  " + "-" * 40)
    for codec, r in results.items():
        total = r["put"] + r["get"]
        print(f"  {codec:8} {r['put']:8.2f} {r['get']:8.2f} {r['kb']:8.1f} {total:8.2f}")

    # Identify winners
    fastest_put = min(results, key=lambda c: results[c]["put"])
    fastest_get = min(results, key=lambda c: results[c]["get"])
    smallest = min(results, key=lambda c: results[c]["kb"])
    print()
    print(f"  🏅 Fastest put:   {fastest_put}")
    print(f"  🏅 Fastest get:   {fastest_get}")
    print(f"  🏅 Smallest file: {smallest}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("🗜️  Compression Benchmark")
    print("=" * 60)
    print("Benchmarking compression codec impact on cache performance")
    print()

    try:
        benchmark_codec_microbench()
        benchmark_end_to_end_codecs()
        benchmark_codec_comparison_table()

        print()
        print()
        print("🎯 Interpretation Guide")
        print("=" * 60)
        print("• lz4: fastest speed, lowest ratio — best for latency-sensitive use")
        print("• zstd: good balance of speed and ratio — default codec")
        print("• zlib: highest ratio, slowest — best for storage-constrained use")
        print("• Random floats compress poorly; structured/zero data compresses well")
        print()
        print("⚠️  Regressions to watch for:")
        print("• zstd >2x slower than lz4 suggests library issue")
        print("• Any codec with ratio <1.0x is expanding data (should not happen)")
        print("• Put time >100ms for small data suggests compression overhead")
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
