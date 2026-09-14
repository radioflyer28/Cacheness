"""Focused failure-mode contracts for the Phase 8 coverage ratchet."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "verify_phase8_coverage.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("phase8_coverage", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Phase 8 coverage verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _coverage_report(*, missing_branch: bool = False) -> dict[str, object]:
    summary: dict[str, object] = {
        "covered_lines": 90,
        "num_statements": 100,
        "covered_branches": 36,
        "num_branches": 40,
    }
    if missing_branch:
        summary.pop("covered_branches")

    return {
        "meta": {"version": "7.0", "branch_coverage": True},
        "files": {
            "src/cacheness/core.py": {
                "summary": {
                    "covered_lines": 9,
                    "num_statements": 10,
                    "covered_branches": 3,
                    "num_branches": 4,
                }
            }
        },
        "totals": summary,
    }


def _baseline() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_revision": "a" * 40,
        "environment": {"python": "3.13", "platform": "test"},
        "command": "uv run pytest --cov-branch",
        "named_selectors": [
            "tests/test_phase8_lifecycle_coverage.py::test_postgresql_error_classification",
            "tests/test_phase8_lifecycle_coverage.py::test_postgresql_replay",
            "tests/test_phase8_lifecycle_coverage.py::test_postgresql_pagination",
            "tests/test_phase8_lifecycle_coverage.py::test_postgresql_transaction_rollback",
        ],
        "repository": {
            "covered_statements": 90,
            "total_statements": 100,
            "covered_branches": 36,
            "total_branches": 40,
            "statement_rate": 0.9,
            "branch_rate": 0.9,
        },
        "critical": {
            "covered_statements": 9,
            "total_statements": 10,
            "covered_branches": 3,
            "total_branches": 4,
            "statement_rate": 0.9,
            "branch_rate": 0.75,
        },
    }


def test_rejects_report_without_branch_counts(tmp_path: Path) -> None:
    verifier = _load_verifier()
    report_path = tmp_path / "coverage.json"
    _write_json(report_path, _coverage_report(missing_branch=True))

    with pytest.raises(verifier.CoverageGateError, match="covered_branches"):
        verifier.parse_coverage_report(report_path)


def test_rejects_missing_critical_source_file(tmp_path: Path) -> None:
    verifier = _load_verifier()
    report_path = tmp_path / "coverage.json"
    _write_json(report_path, _coverage_report())

    with pytest.raises(verifier.CoverageGateError, match="critical source"):
        verifier.parse_coverage_report(report_path)


def test_rejects_any_regressed_total_or_critical_dimension() -> None:
    verifier = _load_verifier()
    baseline = _baseline()
    current = _baseline()
    current["critical"] = {
        **current["critical"],
        "covered_branches": 2,
        "branch_rate": 0.5,
    }

    with pytest.raises(verifier.CoverageGateError, match="critical branch"):
        verifier.compare_to_baseline(current, baseline)


def test_requires_literal_postgresql_selector_families() -> None:
    verifier = _load_verifier()

    with pytest.raises(verifier.CoverageGateError, match="postgresql_replay"):
        verifier.validate_named_selectors(
            [
                selector
                for selector in _baseline()["named_selectors"]
                if "postgresql_replay" not in selector
            ]
        )


def test_verify_mode_does_not_rewrite_baseline(tmp_path: Path) -> None:
    verifier = _load_verifier()
    report_path = tmp_path / "coverage.json"
    baseline_path = tmp_path / "baseline.json"
    _write_json(report_path, _coverage_report())
    _write_json(baseline_path, _baseline())
    original = baseline_path.read_bytes()

    with pytest.raises(verifier.CoverageGateError):
        verifier.verify(report_path, baseline_path)

    assert baseline_path.read_bytes() == original


def test_ruff_scope_is_bounded_and_contains_fixed_critical_inventory(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    changed = tmp_path / "changed.txt"
    changed.write_text(
        "src/cacheness/core.py\n"
        "../escape.py\n"
        "non-python.txt\n",
        encoding="utf-8",
    )

    with pytest.raises(verifier.CoverageGateError, match="unsafe"):
        verifier.build_ruff_scope(changed.read_text(encoding="utf-8").splitlines())

    scope = verifier.build_ruff_scope(["src/cacheness/core.py"])
    assert "src/cacheness/core.py" in scope
    assert set(verifier.RUFF_CRITICAL_PATHS).issubset(scope)
