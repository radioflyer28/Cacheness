#!/usr/bin/env python3
"""
Configurable Serialization
===========================

Advanced: fine-tune how cacheness serialises parameters and chooses
handlers. Most users never need this — the defaults are good.

Usage:
    uv run python examples/configurable_serialization_demo.py
"""

import time
import numpy as np
from cacheness import cacheness, CacheConfig, cached


def demo_serialization_configs():
    """Different configs produce different cache keys for the same data."""
    print("1. Serialisation modes")

    configs = {
        "default": CacheConfig(),
        "fast": CacheConfig(
            enable_collections=False,
            enable_object_introspection=False,
            max_tuple_recursive_length=2,
        ),
        "precise": CacheConfig(
            enable_collections=True,
            enable_object_introspection=True,
            max_tuple_recursive_length=50,
        ),
    }

    data = {"arrays": [np.array([1, 2, 3])], "nested": {"a": {"b": "c"}}}

    for name, cfg in configs.items():
        cache = cacheness(cfg)
        key = cache._create_cache_key({"data": data})
        print(f"   {name:>8}: {key}")
    print()


def demo_custom_cache_instances():
    """Two cache instances with different configs, same decorator API."""
    print("2. Decorator with custom cache instances")

    fast_cache = cacheness(
        CacheConfig(cache_dir="./cache_fast", enable_collections=False)
    )
    precise_cache = cacheness(
        CacheConfig(cache_dir="./cache_precise", enable_collections=True)
    )

    @cached(cache_instance=fast_cache)
    def fast_fn(data, params):
        print("     fast_fn executed")
        return sum(data) * len(params)

    @cached(cache_instance=precise_cache)
    def precise_fn(data, params):
        print("     precise_fn executed")
        return sum(data) * len(params)

    args = ([1, 2, 3], {"a": 1, "b": 2})

    t0 = time.time()
    r1 = fast_fn(*args)
    first = time.time() - t0

    t0 = time.time()
    r2 = fast_fn(*args)
    hit = time.time() - t0

    print(f"   fast  — first: {first:.4f}s  cached: {hit:.4f}s  match: {r1 == r2}")

    t0 = time.time()
    r3 = precise_fn(*args)
    first = time.time() - t0

    t0 = time.time()
    r4 = precise_fn(*args)
    hit = time.time() - t0

    print(f"   precise — first: {first:.4f}s  cached: {hit:.4f}s  match: {r3 == r4}")
    print()


if __name__ == "__main__":
    demo_serialization_configs()
    demo_custom_cache_instances()
    print("Done.")
