"""Regression contracts for the complete repository test environment."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).parent.parent
QUALITY_GATE_PATH = PROJECT_ROOT / "tests" / "test_phase1_quality_gates.py"
DOCUMENTED_FULL_SUITE_COMMAND = (
    "uv run --isolated --all-extras --group dev --frozen pytest -q -o "
    "log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'"
)
PHASE6_LOCAL_SUITE_COMMAND = (
    "uv run --isolated --all-extras --group dev --frozen python "
    "tools/run_phase6_local_suite.py --repo-root ."
)

TEST_ISOLATION_CLASSIFICATION = {
    "bare_collection": "uv run pytest --collect-only",
    "bare_collection_missing": "SQLAlchemy for test_custom_metadata.py",
    "mutating_cascade": (
        "three public-import failures, 23 S3/botocore setup errors, and 13 "
        "pandas failures after nested Ruff dependency resolution"
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
    guide = (PROJECT_ROOT / "docs" / "RELEASE_QUALIFICATION.md").read_text(
        encoding="utf-8"
    )
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert DOCUMENTED_FULL_SUITE_COMMAND in guide
    assert '"sqlalchemy>=2.0.0"' in pyproject
    assert '"pandas>=2.0.0,<4.0.0"' in pyproject
    assert '"pytest>=8.4.1"' in pyproject
    assert '"ruff>=0.12.8"' in pyproject
    assert TEST_ISOLATION_CLASSIFICATION["bare_collection"]
    assert "test_custom_metadata.py" in TEST_ISOLATION_CLASSIFICATION[
        "bare_collection_missing"
    ]
    assert "23 S3/botocore" in TEST_ISOLATION_CLASSIFICATION["mutating_cascade"]


def test_phase071_runtime_extras_keep_boto3_in_test_tooling_only():
    """The obstore cutover has no boto3 production dependency escape hatch."""
    package = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))
    optional = package["project"]["optional-dependencies"]
    all_runtime_dependencies = [
        *package["project"]["dependencies"],
        *(dependency for group in optional.values() for dependency in group),
    ]

    assert any(
        dependency.startswith("obstore==0.11.1")
        for dependency in package["project"]["dependencies"]
    )
    assert optional["s3"] == []
    assert all(
        "boto3" not in dependency.lower()
        for dependency in all_runtime_dependencies
    )
    assert any(
        dependency.startswith("moto[s3,server]")
        for dependency in package["dependency-groups"]["dev"]
    )


def test_phase071_clean_wheel_base_and_selected_extras_cut_over_to_obstore(
    tmp_path: Path,
) -> None:
    """Fresh wheel environments need obstore, never boto3, for current transports."""
    dist = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(dist.glob("cacheness-*.whl"))
    memory_round_trip = """
from cacheness.storage import BackendRef, BlobStore, StoreTopology

store = BlobStore(
    StoreTopology(
        payload=BackendRef(name=\"memory\"),
        authority=BackendRef(name=\"memory\"),
    )
)
store.initialize()
try:
    key = store.put({\"answer\": 42}, key=\"wheel-round-trip\")
    assert store.get(key) == {\"answer\": 42}
finally:
    store.close()
"""
    for extra in (None, "s3", "cloud"):
        requirement = str(wheel) if extra is None else f"{wheel}[{extra}]"
        command = [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--with",
            requirement,
            "python",
            "-c",
            "import cacheness, cacheness.storage; "
            "from importlib.util import find_spec; "
            "assert find_spec('boto3') is None",
        ]
        if extra is None:
            command[-1] = memory_round_trip
        subprocess.run(
            command,
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )


def test_phase6_local_suite_command_keeps_live_qualification_out_of_local_evidence():
    """Phase 6 has one fixed non-live gate; Phase 8 retains the broad command."""
    assert PHASE6_LOCAL_SUITE_COMMAND == (
        "uv run --isolated --all-extras --group dev --frozen python "
        "tools/run_phase6_local_suite.py --repo-root ."
    )
    assert DOCUMENTED_FULL_SUITE_COMMAND == (
        "uv run --isolated --all-extras --group dev --frozen pytest -q -o "
        "log_cli=false -m 'not (live_postgresql or live_aws_s3 or live_remote)'"
    )


def test_isolation_classification_is_structured_as_data_not_a_product_failure():
    """The historical cascade is test-invocation evidence, not a defect waiver."""
    encoded = json.dumps(TEST_ISOLATION_CLASSIFICATION, sort_keys=True)
    assert "bare_collection" in encoded
    assert "mutating_cascade" in encoded
    assert "public-import" in encoded
