#!/usr/bin/env python3
"""
Dill Serialisation (Advanced)
=============================

Cache objects that contain closures, lambdas, or dynamic methods.
Standard pickle can't handle these — ``enable_dill_fallback=True``
transparently falls back to dill when needed.

**Security note:** dill can execute arbitrary code on deserialisation.
Only enable it when you trust every cache file on disk.

Usage:
    uv run python examples/dill_class_caching_demo.py

Requires:
    uv add dill
"""

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from cacheness import cacheness, CacheConfig


@dataclass
class Experiment:
    """ML experiment whose data_processor is a closure (needs dill)."""

    name: str
    weights: np.ndarray
    history: dict
    hyperparams: dict
    processor: Callable = field(init=False)

    def __post_init__(self):
        lr = self.hyperparams.get("learning_rate", 0.001)
        dr = self.hyperparams.get("dropout_rate", 0.2)

        def process(data):
            mask = np.random.random(data.shape) > dr
            return data * mask * lr

        self.processor = process


def build_experiment() -> Experiment:
    """Create a moderately large experiment."""
    print("  Building experiment (takes time)...")
    weights = np.random.randn(2000, 500).astype(np.float32)
    return Experiment(
        name="transformer_v3",
        weights=weights,
        history={"loss": [2.5, 1.2, 0.6, 0.3], "acc": [0.1, 0.5, 0.8, 0.94]},
        hyperparams={"learning_rate": 0.001, "dropout_rate": 0.3},
    )


if __name__ == "__main__":
    config = CacheConfig(
        cache_dir="./cache_dill_demo",
        enable_dill_fallback=True,
        default_ttl="2d",
    )
    cache = cacheness(config)

    # Build & store
    t0 = time.time()
    exp = build_experiment()
    print(f"  Created in {time.time() - t0:.2f}s  ({exp.weights.nbytes / 1e6:.1f} MB)")

    cache.put(exp, project="demo", model=exp.name)

    # Retrieve & verify
    loaded = cache.get(project="demo", model=exp.name)
    assert loaded is not None

    sample = np.random.randn(10, 5)
    orig = exp.processor(sample)
    cached_out = loaded.processor(sample)

    print(f"  Weights match:    {np.array_equal(exp.weights, loaded.weights)}")
    print(f"  History match:    {exp.history == loaded.history}")
    print(f"  Processor works:  {cached_out.shape == orig.shape}")

    stats = cache.get_stats()
    print(
        f"\n  Cache: {stats['total_entries']} entries, "
        f"{stats['total_size_bytes']:,} bytes"
    )
