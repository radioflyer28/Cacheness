"""Regression contracts for the complete repository test environment."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).parent.parent
QUALITY_GATE_PATH = PROJECT_ROOT / "tests" / "test_phase1_quality_gates.py"
DOCUMENTED_FULL_SUITE_COMMAND = (
    "uv run --isolated --all-extras --group dev --frozen pytest -q -o "
    "log_cli=false"
)

TEST_ISOLATION_CLASSIFICATION = {
    "bare_collection": "uv run pytest --collect-only",
    "bare_collection_missing": (
        "SQLAlchemy for test_custom_metadata.py and pandas for the three SQL-cache "
        "modules"
    ),
    "mutating_cascade": (
        "three public-import failures, 23 S3/botocore setup errors, and 13 "
        "SQL/pandas failures after nested Ruff dependency resolution"
    ),
}


def _load_quality_gate_module():
    spec = importlib.util.spec_from_file_location(
        "phase1_quality_gates_under_test", QUALITY_GATE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ruff_gate_executes_the_running_environment_binary_without_uv(monkeypatch):
    """The in-suite quality gate cannot re-resolve packages beneath pytest."""
    quality_gates = _load_quality_gate_module()
    invocations: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        invocations.append(list(argv))
        return SimpleNamespace(returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(quality_gates.subprocess, "run", fake_run)

    assert quality_gates._ruff_findings() == []

    expected_executable = Path(sys.executable).parent / (
        "ruff.exe" if os.name == "nt" else "ruff"
    )
    assert invocations == [
        [
            str(expected_executable),
            "check",
            "src",
            "tests",
            "--output-format",
            "json",
        ]
    ]
    assert "uv" not in invocations[0]


def test_documented_full_suite_command_uses_locked_extras_and_dev_group():
    """The supported command collects optional-feature tests without base expansion."""
    guide = (PROJECT_ROOT / "docs" / "CROSS_PLATFORM_GUIDE.md").read_text(
        encoding="utf-8"
    )
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert DOCUMENTED_FULL_SUITE_COMMAND in guide
    assert '"sqlalchemy>=2.0.0"' in pyproject
    assert '"pandas>=2.0.0,<4.0.0"' in pyproject
    assert '"boto3>=1.26.0"' in pyproject
    assert '"pytest>=8.4.1"' in pyproject
    assert '"ruff>=0.12.8"' in pyproject
    assert TEST_ISOLATION_CLASSIFICATION["bare_collection"]
    assert "test_custom_metadata.py" in TEST_ISOLATION_CLASSIFICATION[
        "bare_collection_missing"
    ]
    assert "23 S3/botocore" in TEST_ISOLATION_CLASSIFICATION["mutating_cascade"]


def test_isolation_classification_is_structured_as_data_not_a_product_failure():
    """The historical cascade is test-invocation evidence, not a defect waiver."""
    encoded = json.dumps(TEST_ISOLATION_CLASSIFICATION, sort_keys=True)
    assert "bare_collection" in encoded
    assert "mutating_cascade" in encoded
    assert "public-import" in encoded
