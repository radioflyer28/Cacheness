"""Focused failure-mode contracts for the Phase 8 coverage ratchet."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "verify_phase8_coverage.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("phase8_coverage", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Phase 8 coverage verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _coverage_report(
    *,
    missing_branch: bool = False,
    critical_paths: tuple[str, ...] = ("src/cacheness/core.py",),
) -> dict[str, object]:
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
            path: {
                "summary": {
                    "covered_lines": 9,
                    "num_statements": 10,
                    "covered_branches": 3,
                    "num_branches": 4,
                }
            }
            for path in critical_paths
        },
        "totals": summary,
    }


def _baseline() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_revision": "a" * 40,
        "environment": {
            "implementation": "CPython",
            "python": "3.13",
            "platform": "test",
        },
        "command": "uv run pytest --cov-branch",
        "justification": "Phase 8 initial ratchet establishment",
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


def test_rejects_critical_scope_shrink_even_if_coverage_rate_rises() -> None:
    verifier = _load_verifier()
    baseline = _baseline()
    current = _baseline()
    current["critical"] = {
        **current["critical"],
        "total_branches": 3,
        "branch_rate": 1.0,
    }

    with pytest.raises(verifier.CoverageGateError, match="critical branch total"):
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


def test_named_selector_ast_validation_accepts_only_module_level_tests(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    source_path = tmp_path / "selector_source.py"
    source_path.write_text(
        "class TestNested:\n"
        "    def test_not_module_level(self):\n"
        "        pass\n\n"
        "def test_module_level():\n"
        "    pass\n",
        encoding="utf-8",
    )

    assert verifier._top_level_test_names(source_path) == {"test_module_level"}


def test_rejects_baseline_with_an_omitted_named_contract(tmp_path: Path) -> None:
    verifier = _load_verifier()
    baseline_path = tmp_path / "baseline.json"
    baseline = _baseline()
    baseline["named_selectors"] = list(verifier.NAMED_SELECTORS[:-1])
    baseline_path.write_bytes(verifier._canonical_json(baseline))

    with pytest.raises(verifier.CoverageGateError, match="named selector inventory"):
        verifier.load_baseline(baseline_path)


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


def test_capture_does_not_write_when_named_selector_preflight_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _load_verifier()
    report_path = tmp_path / "coverage.json"
    baseline_path = tmp_path / "baseline.json"
    _write_json(
        report_path,
        _coverage_report(critical_paths=verifier.CRITICAL_SOURCE_FILES),
    )
    baseline_path.write_text("unchanged baseline", encoding="utf-8")

    def fail_preflight() -> None:
        raise verifier.CoverageGateError("postgresql_replay did not pass")

    monkeypatch.setattr(verifier, "_preflight_named_selectors", fail_preflight)
    with pytest.raises(verifier.CoverageGateError, match="postgresql_replay"):
        verifier.capture(
            report_path,
            baseline_path,
            justification="Phase 8 initial ratchet establishment",
            command="pytest --cov-branch",
        )

    assert baseline_path.read_text(encoding="utf-8") == "unchanged baseline"


def test_ruff_scope_is_bounded_and_contains_fixed_critical_inventory(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    changed = tmp_path / "changed.txt"
    changed.write_text(
        "src/cacheness/core.py\n../escape.py\nnon-python.txt\n",
        encoding="utf-8",
    )

    with pytest.raises(verifier.CoverageGateError, match="unsafe"):
        verifier.build_ruff_scope(changed.read_text(encoding="utf-8").splitlines())

    scope = verifier.build_ruff_scope(["src/cacheness/core.py"])
    assert "src/cacheness/core.py" in scope
    assert set(verifier.RUFF_CRITICAL_PATHS).issubset(scope)


def test_default_measurement_command_excludes_every_live_service_marker() -> None:
    verifier = _load_verifier()

    assert verifier.NON_LIVE_MARKER_EXPRESSION == (
        "not (live_postgresql or live_aws_s3 or live_remote)"
    )
    assert (
        f"-m '{verifier.NON_LIVE_MARKER_EXPRESSION}'"
        in verifier.DEFAULT_MEASUREMENT_COMMAND
    )


def test_changed_python_paths_accepts_the_documented_parent_ref() -> None:
    verifier = _load_verifier()

    assert isinstance(verifier.changed_python_paths("HEAD^"), tuple)
