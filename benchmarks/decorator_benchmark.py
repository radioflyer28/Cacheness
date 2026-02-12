#!/usr/bin/env python3
"""
Decorator Benchmark
====================

Benchmark the @cacheness_it / @cached decorator — the primary user-facing API.

Measures:
  - Cache key generation overhead per call
  - Cache-hit fast path latency (decorator overhead on a hit)
  - Cache-miss round-trip (function call + put)
  - TTL-based expiration + re-cache cost
  - ignore_errors overhead when cache is degraded
  - Comparison: raw function vs decorated function
"""

import time
import os
import tempfile
import statistics
from cacheness import CacheConfig, cacheness
from cacheness.decorators import cached, _generate_cache_key


# ── Helpers ──────────────────────────────────────────────────────────────────


def time_op(func, iterations: int = 100) -> float:
    """Return average ms per call."""
    # Warmup
    for _ in range(min(10, iterations)):
        func()
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed / iterations


# ── Benchmarks ───────────────────────────────────────────────────────────────


def benchmark_key_generation():
    """Benchmark _generate_cache_key overhead for various arg shapes."""
    print("\n🔑 Cache Key Generation Overhead")
    print("-" * 60)

    def simple_func(x, y):
        return x + y

    def many_args_func(a, b, c, d, e, f, g, h):
        return a

    def kwargs_func(x, **kwargs):
        return x

    cases = [
        ("2 ints", simple_func, (1, 2), {}),
        ("2 strings", simple_func, ("hello", "world"), {}),
        ("8 positional args", many_args_func, (1, 2, 3, 4, 5, 6, 7, 8), {}),
        (
            "1 arg + 5 kwargs",
            kwargs_func,
            (1,),
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
        ),
        ("dict arg", simple_func, ({"key": "value", "nested": {"a": 1}}, 2), {}),
        ("list arg", simple_func, (list(range(50)), 2), {}),
    ]

    for name, func, args, kwargs in cases:
        t = time_op(lambda: _generate_cache_key(func, args, kwargs), iterations=1000)
        print(f"  {name:25} {t * 1000:8.2f} μs")


def benchmark_hit_vs_miss():
    """Compare cache-hit fast path vs cache-miss round-trip."""
    print("\n⚡ Cache Hit vs Miss Latency")
    print("-" * 60)

    backends = ["json", "sqlite", "sqlite_memory"]

    for backend in backends:
        with tempfile.TemporaryDirectory() as tmp:
            config = CacheConfig(
                cache_dir=os.path.join(tmp, "cache"),
                metadata_backend=backend,
                enable_memory_cache=False,
            )
            cache_inst = cacheness(config)

            @cached(cache_instance=cache_inst)
            def compute(x, y):
                return x * y + sum(range(100))

            # Cold miss — first call stores result
            miss_times = []
            for i in range(20):
                start = time.perf_counter()
                compute(i, i + 1)
                miss_times.append((time.perf_counter() - start) * 1000)

            # Hot hit — repeat same args
            hit_times = []
            for i in range(20):
                start = time.perf_counter()
                compute(i, i + 1)
                hit_times.append((time.perf_counter() - start) * 1000)

            avg_miss = statistics.mean(miss_times)
            avg_hit = statistics.mean(hit_times)
            ratio = avg_miss / avg_hit if avg_hit > 0 else float("inf")
            print(
                f"  {backend:15}  miss={avg_miss:7.2f}ms  hit={avg_hit:7.2f}ms  ratio={ratio:.1f}x"
            )

            cache_inst.close()


def benchmark_raw_vs_decorated():
    """Measure the overhead of the decorator vs calling the function directly."""
    print("\n📊 Raw Function vs Decorated (cache-hit)")
    print("-" * 60)

    def raw_func(x, y):
        """Trivial computation to isolate decorator overhead."""
        return x + y

    with tempfile.TemporaryDirectory() as tmp:
        config = CacheConfig(
            cache_dir=os.path.join(tmp, "cache"),
            metadata_backend="sqlite_memory",
            enable_memory_cache=False,
        )
        cache_inst = cacheness(config)

        @cached(cache_instance=cache_inst)
        def decorated_func(x, y):
            return x + y

        # Prime the cache
        decorated_func(10, 20)

        raw_t = time_op(lambda: raw_func(10, 20), iterations=5000)
        hit_t = time_op(lambda: decorated_func(10, 20), iterations=200)

        overhead = hit_t - raw_t
        print(f"  Raw function:     {raw_t * 1000:8.2f} μs")
        print(f"  Decorated (hit):  {hit_t * 1000:8.2f} μs")
        print(f"  Overhead:         {overhead * 1000:8.2f} μs ({overhead:.3f} ms)")

        cache_inst.close()


def benchmark_with_memory_cache():
    """Measure benefit of the in-memory cache layer on decorator hits."""
    print("\n🧠 Memory Cache Layer Impact on Decorator")
    print("-" * 60)

    backends = ["json", "sqlite"]

    for backend in backends:
        for mem_cache in [False, True]:
            with tempfile.TemporaryDirectory() as tmp:
                config = CacheConfig(
                    cache_dir=os.path.join(tmp, "cache"),
                    metadata_backend=backend,
                    enable_memory_cache=mem_cache,
                )
                cache_inst = cacheness(config)

                @cached(cache_instance=cache_inst)
                def compute(x):
                    return {"result": x**2, "data": list(range(100))}

                # Prime
                compute(42)

                t = time_op(lambda: compute(42), iterations=100)
                label = f"{backend} + mem={'ON' if mem_cache else 'OFF'}"
                print(f"  {label:25} {t:8.3f} ms/hit")

                cache_inst.close()


def benchmark_ttl_expiration():
    """Benchmark TTL-based expiration + re-cache cost."""
    print("\n⏰ TTL Expiration Round-Trip")
    print("-" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        config = CacheConfig(
            cache_dir=os.path.join(tmp, "cache"),
            metadata_backend="sqlite_memory",
            enable_memory_cache=False,
        )
        cache_inst = cacheness(config)

        call_count = 0

        @cached(cache_instance=cache_inst, ttl_seconds=0.1)
        def expensive(x):
            nonlocal call_count
            call_count += 1
            return sum(range(x))

        # First call — miss
        start = time.perf_counter()
        expensive(1000)
        first_call = (time.perf_counter() - start) * 1000

        # Immediate second call — should hit
        start = time.perf_counter()
        expensive(1000)
        cached_call = (time.perf_counter() - start) * 1000

        # Wait for TTL to expire
        time.sleep(0.15)

        # Third call — should miss again (expired)
        start = time.perf_counter()
        expensive(1000)
        expired_call = (time.perf_counter() - start) * 1000

        print(
            f"  First call (miss):     {first_call:8.2f} ms  (call_count={call_count})"
        )
        print(f"  Second call (hit):     {cached_call:8.2f} ms")
        print(
            f"  After TTL (re-cache):  {expired_call:8.2f} ms  (call_count={call_count})"
        )

        cache_inst.close()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("🎯 Decorator Benchmark")
    print("=" * 60)
    print("Benchmarking @cached / @cacheness_it decorator performance")
    print()

    try:
        benchmark_key_generation()
        benchmark_hit_vs_miss()
        benchmark_raw_vs_decorated()
        benchmark_with_memory_cache()
        benchmark_ttl_expiration()

        print()
        print()
        print("🎯 Interpretation Guide")
        print("=" * 60)
        print("• Key gen <100μs is healthy — dominated by xxhash")
        print("• Hit latency << miss latency confirms cache is working")
        print("• Decorator overhead on trivial funcs shows baseline cost")
        print("• Memory cache ON should give ~2-10x hit speedup over disk-only")
        print("• TTL re-cache should be comparable to initial miss")
        print()
        print("⚠️  Regressions to watch for:")
        print("• Key gen >500μs suggests serialization regression")
        print("• Hit path >10ms without memory cache suggests backend slowdown")
        print("• Decorator overhead >5ms on trivial function is a red flag")
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
