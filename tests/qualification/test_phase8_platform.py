"""Contracts for Phase 8 Python and operating-system qualification evidence."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path

import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "run_phase8_platform_gates.py"
)


@pytest.fixture()
def runner():
    """Load the standalone platform evidence runner."""
    spec = importlib.util.spec_from_file_location("phase8_platform_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stable_rows(runner, *, profile: str = "core") -> list[dict[str, object]]:
    return [
        runner.build_row(
            expected_os="Linux",
            python_minor=minor,
            feature_profile=profile,
            command_status="PASS",
        )
        for minor in runner.STABLE_PYTHON_MINORS
    ]


def test_python_stable_matrix_is_exact_and_complete(runner) -> None:
    """Every supported stable Python minor has exactly one Linux core row."""
    assert runner.STABLE_PYTHON_MINORS == ("3.11", "3.12", "3.13", "3.14")
    evidence = runner.aggregate_rows(_stable_rows(runner))
    assert evidence["status"] == "QUALIFIED"
    assert evidence["qualified_slots"] == [
        f"Linux/{minor}/core" for minor in runner.STABLE_PYTHON_MINORS
    ]


def test_python_missing_or_duplicate_stable_rows_fail_closed(runner) -> None:
    """Omissions and duplicate rows cannot be hidden by an aggregate pass."""
    rows = _stable_rows(runner)
    with pytest.raises(ValueError, match="missing required"):
        runner.aggregate_rows(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        runner.aggregate_rows([*rows, rows[0]])


def test_python_advisory_result_cannot_satisfy_or_invalidate_stable_slot(runner) -> None:
    """Prerelease evidence is retained as advisory without changing stable outcome."""
    rows = _stable_rows(runner)
    rows.append(
        runner.build_row(
            expected_os="Linux",
            python_minor="3.15",
            feature_profile="core",
            command_status="FAIL",
            advisory=True,
        )
    )
    assert runner.aggregate_rows(rows)["status"] == "QUALIFIED"

    contradictory = _stable_rows(runner)
    contradictory[0]["advisory"] = True
    with pytest.raises(ValueError, match="stable row cannot be advisory"):
        runner.aggregate_rows(contradictory)


def test_python_tensorflow_profile_has_explicit_stable_compatibility(runner) -> None:
    """TensorFlow gaps are nonqualifying results, never skip-based qualification."""
    assert runner.TENSORFLOW_COMPATIBLE_MINORS == ("3.11", "3.12", "3.13")
    rows = _stable_rows(runner, profile="tensorflow")
    assert runner.aggregate_feature_rows(rows)["status"] == "QUALIFIED"

    rows[-1]["command_status"] = "SKIPPED"
    with pytest.raises(ValueError, match="not compatible"):
        runner.aggregate_feature_rows(rows)


def test_python_runtime_identity_mismatch_is_rejected(runner, tmp_path: Path) -> None:
    """A caller cannot label one executing interpreter as another row."""
    current_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    wrong_minor = "3.11" if current_minor != "3.11" else "3.12"
    output = tmp_path / "evidence.json"

    exit_code = runner.run_platform_gate(
        expected_os=platform.system(),
        python_minor=wrong_minor,
        feature_profile="core",
        output=output,
    )

    assert exit_code == 2
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["status"] == "UNAVAILABLE"
    assert evidence["reason"] == "runtime_identity_mismatch"

