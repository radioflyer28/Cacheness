"""Isolated peak-memory contracts for Phase 8 structural evidence.

Peak RSS is collected in fresh child processes.  These checks intentionally do
not measure elapsed time or promise bounded-memory S3 streaming.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys

import pytest

from cacheness.storage.obstore_generation_io import (
    DEFAULT_MAX_TRANSFER_BYTES,
    ObstoreGenerationIO,
)


_RUNNER_PATH = Path(__file__).parents[2] / "tools" / "run_phase8_scale_gates.py"
sys.path.insert(0, str(_RUNNER_PATH.parent))
_RUNNER = importlib.import_module("run_phase8_scale_gates")

MemoryObservation = _RUNNER.MemoryObservation
MemoryProbeError = _RUNNER.MemoryProbeError
assert_peak_memory_formula = _RUNNER.assert_peak_memory_formula
run_isolated_peak_probe = _RUNNER.run_isolated_peak_probe
synthetic_bounded_probe = _RUNNER.synthetic_bounded_probe
synthetic_failing_probe = _RUNNER.synthetic_failing_probe
write_structural_evidence = _RUNNER.write_structural_evidence


def _observation(cardinality: int, peak_rss_bytes: int) -> MemoryObservation:
    return MemoryObservation(
        operation="inventory",
        cardinality=cardinality,
        workload_bytes=cardinality * 64,
        peak_rss_bytes=peak_rss_bytes,
        resource_peak_rss_bytes=peak_rss_bytes,
        proc_peak_rss_bytes=peak_rss_bytes,
        unit="bytes",
        platform="linux",
        child_exit_code=0,
    )


def test_peak_memory_formula_accepts_page_bounded_growth() -> None:
    """A larger store cannot drive memory linearly with total cardinality."""

    assert_peak_memory_formula(
        _observation(100, 10_000),
        _observation(10_000, 18_000),
        max_growth_ratio=2.0,
    )


def test_peak_memory_formula_rejects_cardinality_proportional_growth() -> None:
    """Structural comparison catches retained whole-store inventory state."""

    with pytest.raises(MemoryProbeError, match="peak RSS"):
        assert_peak_memory_formula(
            _observation(100, 10_000),
            _observation(10_000, 1_000_000),
            max_growth_ratio=2.0,
        )


@pytest.mark.parametrize("operation", ["inventory", "reconciliation", "clear", "maintenance"])
@pytest.mark.parametrize("cardinality", [10, 100, 1_000, 10_000])
def test_isolated_probe_records_complete_child_measurement(
    operation: str, cardinality: int
) -> None:
    """Every structural operation receives a fresh child result with byte units."""

    observation = run_isolated_peak_probe(
        operation=operation,
        cardinality=cardinality,
        workload_bytes=cardinality * 64,
        probe=synthetic_bounded_probe,
    )

    assert observation.operation == operation
    assert observation.cardinality == cardinality
    assert observation.workload_bytes == cardinality * 64
    assert observation.unit == "bytes"
    assert observation.child_exit_code == 0
    assert observation.peak_rss_bytes > 0


def test_missing_or_unit_mismatched_child_evidence_fails_closed() -> None:
    """A parent cannot turn missing or ambiguous RSS units into a passing result."""

    with pytest.raises(MemoryProbeError, match="child result"):
        MemoryObservation.from_child_message(None, expected_operation="inventory")

    with pytest.raises(MemoryProbeError, match="unit"):
        MemoryObservation.from_child_message(
            {
                "operation": "inventory",
                "cardinality": 10,
                "workload_bytes": 640,
                "peak_rss_bytes": 10_000,
                "resource_peak_rss_bytes": 10_000,
                "proc_peak_rss_bytes": 10_000,
                "unit": "kilobytes",
                "platform": "linux",
                "child_exit_code": 0,
            },
            expected_operation="inventory",
        )


def test_crashed_child_probe_cannot_be_recorded_as_peak_memory_evidence() -> None:
    """Child exceptions have no usable measurement envelope."""

    with pytest.raises(MemoryProbeError, match="child probe failed"):
        run_isolated_peak_probe(
            operation="inventory",
            cardinality=10,
            workload_bytes=640,
            probe=synthetic_failing_probe,
        )


def test_structural_evidence_keeps_counts_and_peak_rss_without_a_timing_claim(
    tmp_path: Path,
) -> None:
    """The shared envelope preserves bounded raw facts under the structural class."""

    output = tmp_path / "structural.json"
    write_structural_evidence(
        output,
        call_observations=[
            _RUNNER.ScaleObservation(
                operation="catalog",
                seeded_entries=10_000,
                page_size=16,
                work_cap=16,
                selected_entries=16,
                counters=_RUNNER.CallCounters(authority_pages=1),
            )
        ],
        memory_observations=[_observation(10_000, 18_000)],
        revision="a" * 40,
        source_digest="b" * 64,
    )

    envelope = _RUNNER.phase8_evidence.load_envelope(output)

    assert envelope.evidence_class == "structural"
    assert envelope.payload["claim_categories"]["performance"] == "NOT_QUALIFIED"
    assert envelope.payload["observations"][0]["counters"]["authority_pages"] == 1
    assert envelope.payload["observations"][1]["peak_rss_bytes"] == 18_000


def test_direct_conditional_upload_keeps_its_explicit_resource_cap() -> None:
    """Structural RSS evidence must not accidentally imply streaming S3 uploads."""

    assert DEFAULT_MAX_TRANSFER_BYTES == 128 * 1024 * 1024
    assert "use_multipart=False" in inspect.getsource(
        ObstoreGenerationIO.publish_generation
    )
