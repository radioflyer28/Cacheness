#!/usr/bin/env python3
"""
Security Benchmark
===================

Benchmark the overhead of cache entry signing and verification.

CacheEntrySigner uses HMAC-SHA256 and signs 11 fields by default.
Signing happens on every put(), verification on every get().

Measures:
  - Raw sign_entry / verify_entry micro-cost
  - End-to-end put/get with signing enabled vs disabled
  - Throughput impact at scale (many entries)
  - Impact of custom_fields list length on signing cost
"""

import time
import os
import tempfile
import statistics
import logging
from pathlib import Path

from cacheness import CacheConfig, cacheness
from cacheness.config import SecurityConfig
from cacheness.security import CacheEntrySigner

# Suppress noisy signature verification warnings during benchmarks
logging.getLogger("cacheness.security").setLevel(logging.ERROR)


# ── Helpers ──────────────────────────────────────────────────────────────────


def time_op(func, iterations: int = 100) -> float:
    """Return average ms per call."""
    for _ in range(min(5, iterations)):
        func()
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    return ((time.perf_counter() - start) / iterations) * 1000


# ── Benchmarks ───────────────────────────────────────────────────────────────


def benchmark_raw_sign_verify():
    """Micro-benchmark of sign_entry and verify_entry alone."""
    print("\n🔐 Raw sign_entry / verify_entry Micro-Benchmark")
    print("-" * 60)

    signer = CacheEntrySigner(
        key_file_path=Path("/dev/null"),  # unused
        use_in_memory_key=True,
    )

    # Realistic entry_data dict matching what core.py passes
    entry_data = {
        "cache_key": "abc123def456",
        "data_type": "dict",
        "prefix": "benchmark",
        "file_size": 4096,
        "file_hash": "sha256_" + "a" * 60,
        "object_type": "dict",
        "storage_format": "pickle",
        "serializer": "pickle",
        "compression_codec": "zstd",
        "actual_path": "/tmp/cache/abc123def456.pkl.zstd",
        "created_at": "2025-01-15T10:30:00",
    }

    # Sign
    sign_t = time_op(lambda: signer.sign_entry(entry_data), iterations=5000)
    sig = signer.sign_entry(entry_data)

    # Verify (valid)
    verify_t = time_op(lambda: signer.verify_entry(entry_data, sig), iterations=5000)

    # Verify (invalid — should fail)
    bad_sig = "0" * 64
    verify_bad_t = time_op(
        lambda: signer.verify_entry(entry_data, bad_sig), iterations=5000
    )

    print(f"  sign_entry:          {sign_t * 1000:8.2f} μs")
    print(f"  verify_entry (valid):{verify_t * 1000:8.2f} μs")
    print(f"  verify_entry (bad):  {verify_bad_t * 1000:8.2f} μs")
    print(f"  Signature length:    {len(sig)} chars (hex SHA-256)")


def benchmark_field_count_impact():
    """Measure how the number of signed fields affects signing time."""
    print("\n📊 Signed Fields Count Impact")
    print("-" * 60)

    all_fields = [
        "cache_key",
        "data_type",
        "prefix",
        "file_size",
        "file_hash",
        "object_type",
        "storage_format",
        "serializer",
        "compression_codec",
        "actual_path",
        "created_at",
    ]

    entry_data = {f: f"value_{i}" for i, f in enumerate(all_fields)}

    field_counts = [3, 5, 8, 11]  # Subsets of increasing size

    print(f"  {'Fields':>8} {'Sign μs':>10} {'Verify μs':>10}")
    print("  " + "-" * 30)

    for n in field_counts:
        subset = all_fields[:n]
        signer = CacheEntrySigner(
            key_file_path=Path("/dev/null"),
            custom_fields=subset,
            use_in_memory_key=True,
        )

        sign_t = time_op(lambda: signer.sign_entry(entry_data), iterations=3000)
        sig = signer.sign_entry(entry_data)
        verify_t = time_op(
            lambda: signer.verify_entry(entry_data, sig), iterations=3000
        )

        print(f"  {n:>8} {sign_t * 1000:10.2f} {verify_t * 1000:10.2f}")


def benchmark_signing_enabled_vs_disabled():
    """End-to-end put/get with entry signing ON vs OFF."""
    print("\n⚡ Put/Get: Signing Enabled vs Disabled")
    print("-" * 60)

    backends = ["sqlite", "sqlite_memory"]
    count = 50

    for backend in backends:
        print(f"\n  Backend: {backend}")
        print(f"    {'Mode':20} {'Put ms':>8} {'Get ms':>8} {'Total ms':>8}")
        print("    " + "-" * 48)

        for signing in [False, True]:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend=backend,
                    enable_memory_cache=False,
                    security=SecurityConfig(enable_entry_signing=signing),
                )
                cache = cacheness(config)

                # Put
                put_times = []
                keys = []
                for i in range(count):
                    data = {"index": i, "payload": f"data_{i}"}
                    start = time.perf_counter()
                    key = cache.put(data, bench_id=i, prefix="sec")
                    put_times.append((time.perf_counter() - start) * 1000)
                    keys.append(key)

                # Get
                get_times = []
                for key in keys:
                    start = time.perf_counter()
                    cache.get(cache_key=key)
                    get_times.append((time.perf_counter() - start) * 1000)

                avg_put = statistics.mean(put_times)
                avg_get = statistics.mean(get_times)
                label = f"signing={'ON' if signing else 'OFF'}"
                print(
                    f"    {label:20} {avg_put:8.2f} {avg_get:8.2f} {avg_put + avg_get:8.2f}"
                )

                cache.close()


def benchmark_signing_throughput():
    """Throughput: entries/sec with signing at increasing cache sizes."""
    print("\n📈 Signing Throughput at Scale")
    print("-" * 60)

    sizes = [20, 50, 100]

    print(f"  {'Entries':>8} {'Put/s OFF':>10} {'Put/s ON':>10} {'Overhead':>10}")
    print("  " + "-" * 40)

    for size in sizes:
        results = {}
        for signing in [False, True]:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend="sqlite_memory",
                    enable_memory_cache=False,
                    security=SecurityConfig(enable_entry_signing=signing),
                )
                cache = cacheness(config)

                start = time.perf_counter()
                for i in range(size):
                    cache.put({"i": i}, idx=i)
                elapsed = time.perf_counter() - start

                ops_sec = size / elapsed if elapsed > 0 else 0
                results["ON" if signing else "OFF"] = ops_sec

                cache.close()

        overhead_pct = 0
        if results.get("OFF", 0) > 0:
            overhead_pct = (1 - results.get("ON", 0) / results["OFF"]) * 100

        print(
            f"  {size:>8} "
            f"{results.get('OFF', 0):10.0f} "
            f"{results.get('ON', 0):10.0f} "
            f"{overhead_pct:9.1f}%"
        )


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("🔐 Security Benchmark")
    print("=" * 60)
    print("Benchmarking cache entry signing/verification overhead")
    print()

    try:
        benchmark_raw_sign_verify()
        benchmark_field_count_impact()
        benchmark_signing_enabled_vs_disabled()
        benchmark_signing_throughput()

        print()
        print()
        print("🎯 Interpretation Guide")
        print("=" * 60)
        print("• HMAC-SHA256 is fast (~1-5 μs per sign) — crypto overhead is minimal")
        print("• Signing overhead should be <5% of total put/get time")
        print("• Field count has linear impact but even 11 fields is cheap")
        print("• If signing overhead >20%, investigate entry_data construction cost")
        print()
        print("⚠️  Regressions to watch for:")
        print("• sign_entry >50μs suggests payload construction regression")
        print("• Signing-ON put >2x signing-OFF put indicates a problem")
        print("• verify_entry significantly slower than sign_entry (they do same work)")
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
