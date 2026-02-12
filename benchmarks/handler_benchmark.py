#!/usr/bin/env python3
"""
Handler Benchmark
==================

Benchmark put/get performance per data-type handler.

All existing benchmarks only store plain dicts (ObjectHandler).
This benchmark exercises EVERY handler path:
  - ObjectHandler:             dicts, custom objects (pickle)
  - ArrayHandler:              NumPy arrays (NPZ / blosc2)
  - PandasDataFrameHandler:    pandas DataFrames (parquet)
  - PandasSeriesHandler:       pandas Series (parquet)
  - PolarsDataFrameHandler:    polars DataFrames (parquet)
  - PolarsSeriesHandler:       polars Series (parquet)

Measures round-trip (put + get) time and file size for small/medium data.
TensorFlowTensorHandler is skipped unless TF is installed.
"""

import time
import os
import tempfile
import statistics
from typing import List, Tuple, Any

import numpy as np

from cacheness import CacheConfig, cacheness

# Optional imports — graceful skip
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False


# ── Data Generators ──────────────────────────────────────────────────────────


def make_test_data() -> List[Tuple[str, str, Any, Any]]:
    """
    Generate (label, handler_name, small_data, medium_data) tuples.

    Returns a list of test cases. Each case exercises a different handler.
    """
    cases = []

    # ObjectHandler — plain dict
    cases.append(
        (
            "dict (ObjectHandler)",
            "ObjectHandler",
            {"a": 1, "b": "hello", "c": [1, 2, 3]},
            {f"key_{i}": list(range(50)) for i in range(50)},
        )
    )

    # ArrayHandler — NumPy array
    cases.append(
        (
            "numpy array (ArrayHandler)",
            "ArrayHandler",
            np.random.rand(100),
            np.random.rand(1000, 50),
        )
    )

    # PandasDataFrameHandler
    if PANDAS_AVAILABLE:
        small_pdf = pd.DataFrame({"x": range(100), "y": np.random.rand(100)})
        medium_pdf = pd.DataFrame({f"col_{i}": np.random.rand(5000) for i in range(20)})
        cases.append(
            ("pandas DataFrame", "PandasDataFrameHandler", small_pdf, medium_pdf)
        )

        # PandasSeriesHandler
        cases.append(
            (
                "pandas Series",
                "PandasSeriesHandler",
                pd.Series(np.random.rand(100), name="values"),
                pd.Series(np.random.rand(50_000), name="big_values"),
            )
        )

    # PolarsDataFrameHandler
    if POLARS_AVAILABLE:
        small_pldf = pl.DataFrame({"x": range(100), "y": np.random.rand(100).tolist()})
        medium_pldf = pl.DataFrame(
            {f"col_{i}": np.random.rand(5000).tolist() for i in range(20)}
        )
        cases.append(
            ("polars DataFrame", "PolarsDataFrameHandler", small_pldf, medium_pldf)
        )

        # PolarsSeriesHandler
        cases.append(
            (
                "polars Series",
                "PolarSeriesHandler",
                pl.Series("values", np.random.rand(100).tolist()),
                pl.Series("big_values", np.random.rand(50_000).tolist()),
            )
        )

    # TensorFlowTensorHandler
    if TF_AVAILABLE:
        cases.append(
            (
                "TF tensor",
                "TensorFlowTensorHandler",
                tf.constant(np.random.rand(100).astype(np.float32)),
                tf.constant(np.random.rand(1000, 50).astype(np.float32)),
            )
        )

    return cases


# ── Benchmarks ───────────────────────────────────────────────────────────────


def benchmark_put_get_roundtrip():
    """Benchmark put + get round-trip per handler for small & medium data."""
    print("\n🔄 Put/Get Round-Trip by Handler")
    print("-" * 70)
    print(f"  {'Data Type':30} {'Size':8} {'Put ms':>8} {'Get ms':>8} {'File KB':>8}")
    print("  " + "-" * 66)

    cases = make_test_data()

    for label, handler_name, small, medium in cases:
        for size_label, data in [("small", small), ("medium", medium)]:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend="sqlite_memory",
                    enable_memory_cache=False,
                )
                cache = cacheness(config)

                # Measure put
                put_times = []
                keys = []
                for i in range(5):
                    start = time.perf_counter()
                    key = cache.put(data, bench_id=i, handler=handler_name)
                    put_times.append((time.perf_counter() - start) * 1000)
                    keys.append(key)

                # Measure get
                get_times = []
                for key in keys:
                    start = time.perf_counter()
                    result = cache.get(cache_key=key)
                    get_times.append((time.perf_counter() - start) * 1000)

                # File size of one entry (read from disk for accuracy)
                entries = cache.list_entries()
                file_size_kb = 0
                if entries:
                    actual_path = entries[0].get("metadata", {}).get("actual_path", "")
                    if actual_path and os.path.exists(actual_path):
                        file_size_kb = os.path.getsize(actual_path) / 1024

                avg_put = statistics.mean(put_times)
                avg_get = statistics.mean(get_times)

                print(
                    f"  {label:30} {size_label:8} {avg_put:8.2f} {avg_get:8.2f} {file_size_kb:8.1f}"
                )

                cache.close()


def benchmark_handler_scaling():
    """Test how each handler scales with data size (numpy & pandas only)."""
    print("\n📈 Handler Scaling (rows/elements)")
    print("-" * 70)

    sizes = [100, 1_000, 10_000]

    # NumPy scaling
    print("\n  NumPy array (float64):")
    print(f"    {'Elements':>10} {'Put ms':>8} {'Get ms':>8} {'KB':>8}")
    for n in sizes:
        data = np.random.rand(n)
        with tempfile.TemporaryDirectory() as tmp:
            config = CacheConfig(
                cache_dir=os.path.join(tmp, "cache"),
                metadata_backend="sqlite_memory",
                enable_memory_cache=False,
            )
            cache = cacheness(config)

            start = time.perf_counter()
            key = cache.put(data, size=n)
            put_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            cache.get(cache_key=key)
            get_ms = (time.perf_counter() - start) * 1000

            entries = cache.list_entries()
            kb = 0
            if entries:
                ap = entries[0].get("metadata", {}).get("actual_path", "")
                if ap and os.path.exists(ap):
                    kb = os.path.getsize(ap) / 1024

            print(f"    {n:>10,} {put_ms:8.2f} {get_ms:8.2f} {kb:8.1f}")
            cache.close()

    # Pandas scaling
    if PANDAS_AVAILABLE:
        print("\n  Pandas DataFrame (10 cols, float64):")
        print(f"    {'Rows':>10} {'Put ms':>8} {'Get ms':>8} {'KB':>8}")
        for n in sizes:
            data = pd.DataFrame({f"c{i}": np.random.rand(n) for i in range(10)})
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend="sqlite_memory",
                    enable_memory_cache=False,
                )
                cache = cacheness(config)

                start = time.perf_counter()
                key = cache.put(data, rows=n)
                put_ms = (time.perf_counter() - start) * 1000

                start = time.perf_counter()
                cache.get(cache_key=key)
                get_ms = (time.perf_counter() - start) * 1000

                entries = cache.list_entries()
                kb = 0
                if entries:
                    ap = entries[0].get("metadata", {}).get("actual_path", "")
                    if ap and os.path.exists(ap):
                        kb = os.path.getsize(ap) / 1024

                print(f"    {n:>10,} {put_ms:8.2f} {get_ms:8.2f} {kb:8.1f}")
                cache.close()

    # Polars scaling
    if POLARS_AVAILABLE:
        print("\n  Polars DataFrame (10 cols, float64):")
        print(f"    {'Rows':>10} {'Put ms':>8} {'Get ms':>8} {'KB':>8}")
        for n in sizes:
            data = pl.DataFrame(
                {f"c{i}": np.random.rand(n).tolist() for i in range(10)}
            )
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend="sqlite_memory",
                    enable_memory_cache=False,
                )
                cache = cacheness(config)

                start = time.perf_counter()
                key = cache.put(data, rows=n)
                put_ms = (time.perf_counter() - start) * 1000

                start = time.perf_counter()
                cache.get(cache_key=key)
                get_ms = (time.perf_counter() - start) * 1000

                entries = cache.list_entries()
                kb = 0
                if entries:
                    ap = entries[0].get("metadata", {}).get("actual_path", "")
                    if ap and os.path.exists(ap):
                        kb = os.path.getsize(ap) / 1024

                print(f"    {n:>10,} {put_ms:8.2f} {get_ms:8.2f} {kb:8.1f}")
                cache.close()


def benchmark_handler_summary():
    """Quick comparison table: all handlers at one size."""
    print("\n\n🏆 Handler Summary (medium data)")
    print("=" * 60)
    print(f"  {'Handler':30} {'Put ms':>8} {'Get ms':>8} {'Total ms':>8}")
    print("  " + "-" * 56)

    cases = make_test_data()

    for label, handler_name, _small, medium in cases:
        with tempfile.TemporaryDirectory() as tmp:
            config = CacheConfig(
                cache_dir=os.path.join(tmp, "cache"),
                metadata_backend="sqlite_memory",
                enable_memory_cache=False,
            )
            cache = cacheness(config)

            # 3 iterations, take median
            put_times = []
            get_times = []
            for i in range(3):
                start = time.perf_counter()
                key = cache.put(medium, run=i, handler=handler_name)
                put_times.append((time.perf_counter() - start) * 1000)

                start = time.perf_counter()
                cache.get(cache_key=key)
                get_times.append((time.perf_counter() - start) * 1000)

            p = statistics.median(put_times)
            g = statistics.median(get_times)
            print(f"  {label:30} {p:8.2f} {g:8.2f} {p + g:8.2f}")

            cache.close()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("🔧 Handler Benchmark")
    print("=" * 60)
    print("Benchmarking put/get performance per data-type handler")
    print()

    avail = ["numpy"]
    if PANDAS_AVAILABLE:
        avail.append("pandas")
    if POLARS_AVAILABLE:
        avail.append("polars")
    if TF_AVAILABLE:
        avail.append("tensorflow")
    print(f"Available libraries: {', '.join(avail)}")

    try:
        benchmark_put_get_roundtrip()
        benchmark_handler_scaling()
        benchmark_handler_summary()

        print()
        print()
        print("🎯 Interpretation Guide")
        print("=" * 60)
        print("• ObjectHandler (pickle): baseline, good for small/medium dicts")
        print("• ArrayHandler (NPZ/blosc2): fast for numeric arrays, good compression")
        print("• PandasDataFrameHandler (parquet): efficient columnar storage")
        print("• PolarsDataFrameHandler (parquet): similar to pandas, often faster")
        print("• Series handlers: wraps as 1-col DataFrame → parquet")
        print()
        print("⚠️  Regressions to watch for:")
        print("• Parquet put >100ms for 5k rows suggests I/O bottleneck")
        print("• Array get >50ms for 50k elements suggests decompression issue")
        print("• Any handler >10x slower than ObjectHandler on same data volume")
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
