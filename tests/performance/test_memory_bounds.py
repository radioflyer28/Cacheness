"""Isolated peak-memory contracts for Phase 8 structural evidence.

Peak RSS is collected in fresh child processes.  These checks intentionally do
not measure elapsed time or promise bounded-memory S3 streaming.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_RUNNER_PATH = Path(__file__).parents[2] / "tools" / "run_phase8_scale_gates.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location("phase8_scale_memory", _RUNNER_PATH)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = _RUNNER
_RUNNER_SPEC.loader.exec_module(_RUNNER)

MemoryObservation = _RUNNER.MemoryObservation
MemoryProbeError = _RUNNER.MemoryProbeError
assert_peak_memory_formula = _RUNNER.assert_peak_memory_formula
run_isolated_peak_probe = _RUNNER.run_isolated_peak_probe
synthetic_bounded_probe = _RUNNER.synthetic_bounded_probe


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
def test_isolated_probe_records_complete_child_measurement(operation: str) -> None:
    """Every structural operation receives a fresh child result with byte units."""

    observation = run_isolated_peak_probe(
        operation=operation,
        cardinality=10,
        workload_bytes=640,
        probe=synthetic_bounded_probe,
    )

    assert observation.operation == operation
    assert observation.cardinality == 10
    assert observation.workload_bytes == 640
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
