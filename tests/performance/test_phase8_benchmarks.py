"""Contract tests for the Phase 8 controlled-performance harness."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

import pytest

_BENCHMARKS_DIR = Path(__file__).parents[2] / "benchmarks"
sys.path.insert(0, str(_BENCHMARKS_DIR))

_WORKLOADS = importlib.import_module("phase8_workloads")
LAYERS = _WORKLOADS.LAYERS
access_labels = _WORKLOADS.access_labels
build_layer_workload = _WORKLOADS.build_layer_workload
reviewed_workloads = _WORKLOADS.reviewed_workloads

_BENCHMARKS = importlib.import_module("phase8_benchmarks")
BaselineVerificationError = _BENCHMARKS.BaselineVerificationError
HASH_SIZES = _BENCHMARKS.HASH_SIZES
atomic_replace_baseline = _BENCHMARKS.atomic_replace_baseline
build_baseline_document = _BENCHMARKS.build_baseline_document
distribution = _BENCHMARKS.distribution
measure_hashes = _BENCHMARKS.measure_hashes
verify_baseline = _BENCHMARKS.verify_baseline


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


def test_reduced_workloads_use_current_public_composition() -> None:
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


def test_workloads_layers_and_access_labels_remain_explicit() -> None:
    """Cold and warm operations are not conflated in one timing sample."""
    assert LAYERS == ("handler", "blobstore", "unified-cache")
    assert access_labels() == ("cold-put", "warm-get")


def _baseline_document() -> dict[str, object]:
    return build_baseline_document(
        revision="a" * 40,
        source_digest="b" * 64,
        runner_identity="cacheness-perf-linux-x64",
        environment={"platform": "Linux", "machine": "x86_64"},
        distributions={
            "small-generic-object__blobstore__cold-put": distribution(
                [100, 110, 120, 130, 140]
            )
        },
        envelopes={
            "small-generic-object__blobstore__cold-put": {
                "median_relative_limit": 1.2,
                "tail_relative_limit": 1.3,
            }
        },
    )


def test_distribution_preserves_raw_samples_and_deterministic_tails() -> None:
    """The reviewed record keeps derivations traceable to raw nanoseconds."""
    result = distribution([1, 2, 3, 4, 5])

    assert result["samples_ns"] == [1, 2, 3, 4, 5]
    assert result["p50_ns"] == 3.0
    assert result["p95_ns"] == 4.8
    assert result["p99_ns"] == 4.96


def test_hash_measurements_cover_canonical_sizes_without_changing_sha256() -> None:
    """XXH3 is comparative evidence while SHA-256 remains the stored digest."""
    assert HASH_SIZES == (4 * 1024, 1 * 1024 * 1024, 16 * 1024 * 1024, 128 * 1024 * 1024)

    observations = measure_hashes(
        size_bytes=4 * 1024,
        lifecycle_distribution=distribution([1_000, 1_200, 1_400]),
        loops=2,
    )

    assert {observation["algorithm"] for observation in observations} == {"sha256", "xxh3_64"}
    assert all(observation["throughput_bytes_per_second"] > 0 for observation in observations)
    assert all(0 < observation["lifecycle_share_p50"] for observation in observations)
    assert _BENCHMARKS.PAYLOAD_DIGEST_ALGORITHM == "sha256"


def test_verify_baseline_rejects_mismatched_identity_revision_environment_and_regression() -> None:
    """Only one exact controlled environment can qualify its reviewed envelope."""
    baseline = _baseline_document()
    current = _baseline_document()
    assert verify_baseline(baseline, current) == []

    mismatched_runner = _baseline_document()
    mismatched_runner["runner_identity"] = "ordinary-linux"
    with pytest.raises(BaselineVerificationError, match="runner identity"):
        verify_baseline(baseline, mismatched_runner)

    stale_revision = _baseline_document()
    stale_revision["revision"] = "c" * 40
    with pytest.raises(BaselineVerificationError, match="revision"):
        verify_baseline(baseline, stale_revision)

    environment_drift = _baseline_document()
    environment_drift["environment"] = {"platform": "Linux", "machine": "arm64"}
    with pytest.raises(BaselineVerificationError, match="environment"):
        verify_baseline(baseline, environment_drift)

    regressed = _baseline_document()
    record = regressed["distributions"]["small-generic-object__blobstore__cold-put"]
    record["samples_ns"] = [200, 220, 240, 260, 280]
    record.update(distribution(record["samples_ns"]))
    with pytest.raises(BaselineVerificationError, match="median"):
        verify_baseline(baseline, regressed)


def test_baseline_capture_and_recalibration_are_explicit_and_atomic(tmp_path: Path) -> None:
    """Verify reads evidence; reviewed capture/recalibration own all mutations."""
    destination = tmp_path / "phase8_baseline.json"
    baseline = _baseline_document()

    atomic_replace_baseline(destination, baseline, mode="capture")
    original = destination.read_bytes()
    assert verify_baseline(destination, baseline) == []
    assert destination.read_bytes() == original

    with pytest.raises(BaselineVerificationError, match="already exists"):
        atomic_replace_baseline(destination, baseline, mode="capture")
    with pytest.raises(BaselineVerificationError, match="justification"):
        atomic_replace_baseline(destination, baseline, mode="recalibrate")

    atomic_replace_baseline(
        destination,
        baseline,
        mode="recalibrate",
        justification="controlled runner kernel update",
    )
    recalibrated = json.loads(destination.read_text(encoding="utf-8"))
    assert recalibrated["baseline_change"] == {
        "mode": "recalibrate",
        "justification": "controlled runner kernel update",
    }


def test_controlled_workflow_requires_exact_sha_and_named_linux_runner() -> None:
    """Only a detached exact SHA on the reviewed runner can block performance."""
    workflow = (
        Path(__file__).parents[2] / ".github" / "workflows" / "performance.yml"
    ).read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert re.search(r"candidate_sha:\s*\n\s*description:.*\n\s*required: true", workflow)
    assert "cacheness-perf-linux-x64" in workflow
    assert "environment: controlled-performance" in workflow
    assert "ref: ${{ inputs.candidate_sha }}" in workflow
    assert 'ACTUAL_SHA="$(git rev-parse HEAD)"' in workflow
    assert '"$ACTUAL_SHA" = "$CANDIDATE_SHA"' in workflow
    assert "--verify-baseline benchmarks/phase8_baseline.json" in workflow
    assert "--all-workloads" in workflow
    assert "--runner-identity cacheness-perf-linux-x64" in workflow
    assert "remote PostgreSQL/S3 timing" in workflow
    assert "ordinary macOS timing" in workflow


def test_controlled_workflow_pins_actions_and_uploads_one_sanitized_envelope() -> None:
    """The qualifying artifact has a fixed name and bounded diagnostic retention."""
    workflow = (
        Path(__file__).parents[2] / ".github" / "workflows" / "performance.yml"
    ).read_text(encoding="utf-8")

    action_references = re.findall(r"uses:\s+[^@\s]+@([^\s]+)", workflow)
    assert action_references
    assert all(re.fullmatch(r"[0-9a-f]{40}", reference) for reference in action_references)
    assert "name: controlled-performance-envelope" in workflow
    assert "retention-days: 30" in workflow
    assert "build/phase8/controlled-performance.json" in workflow
    assert "build/phase8/raw-performance.json" in workflow
