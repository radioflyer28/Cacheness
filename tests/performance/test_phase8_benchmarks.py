"""Contract tests for the Phase 8 controlled-performance harness."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

_BENCHMARKS_DIR = Path(__file__).parents[2] / "benchmarks"
sys.path.insert(0, str(_BENCHMARKS_DIR))

from phase8_workloads import (
    LAYERS,
    access_labels,
    build_layer_workload,
    reviewed_workloads,
)


def test_workloads_inventory_is_representative_without_topology_cross_product() -> None:
    """Each reviewed tier occurs once and names its intended topology."""
    workloads = reviewed_workloads()

    assert {workload.name for workload in workloads} == {
        "small-generic-object",
        "numpy-npz-medium",
        "numpy-npz-large",
        "pandas-parquet",
        "polars-parquet",
    }
    assert {workload.layer for workload in workloads} == set(LAYERS)
    assert {workload.access for workload in workloads} == set(access_labels())
    assert {workload.topology for workload in workloads} == {
        "memory-memory",
        "sqlite-filesystem",
    }

    canonical_sizes = {workload.name: workload.canonical_size for workload in workloads}
    canonical_units = {workload.name: workload.size_unit for workload in workloads}
    assert canonical_sizes["small-generic-object"] == 4 * 1024
    assert canonical_sizes["numpy-npz-medium"] == 16 * 1024 * 1024
    assert canonical_sizes["numpy-npz-large"] == 128 * 1024 * 1024
    assert canonical_sizes["pandas-parquet"] == 100_000
    assert canonical_sizes["polars-parquet"] == 100_000
    assert canonical_units["numpy-npz-large"] == "bytes"
    assert canonical_units["pandas-parquet"] == "rows"
    assert all("blosc2" not in workload.name for workload in workloads)


def test_reduced_workload_callbacks_use_current_public_composition() -> None:
    """Reduced fixtures remain deterministic and exercise public layers only."""
    descriptor = next(
        workload
        for workload in reviewed_workloads()
        if workload.name == "small-generic-object"
        and workload.layer == "blobstore"
        and workload.access == "cold-put"
    )

    first = build_layer_workload(descriptor, reduced=True)
    second = build_layer_workload(descriptor, reduced=True)
    try:
        assert first.fixture_digest == second.fixture_digest
        assert first.fixture_digest == hashlib.sha256(first.fixture_bytes).hexdigest()
        assert first.layer == "blobstore"
        assert first.uses_public_composition is True
        assert first.timed_callback() is not None
    finally:
        first.close()
        second.close()


def test_workload_layers_and_access_labels_remain_explicit() -> None:
    """Cold and warm operations are not conflated in one timing sample."""
    assert LAYERS == ("handler", "blobstore", "unified-cache")
    assert access_labels() == ("cold-put", "warm-get")
