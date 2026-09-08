"""Contracts for the fail-closed Phase 5 live-service qualification gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPOSITORY_ROOT / "tools" / "run_phase5_qualification.py"


def _load_runner():
    """Load the standalone runner without requiring ``tools`` to be a package."""
    spec = importlib.util.spec_from_file_location("phase5_qualification_runner", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_missing_configuration_writes_sanitized_unavailable_evidence(
    tmp_path: Path,
) -> None:
    """Absent services remain a reproducible, non-passing qualification state."""
    runner = _load_runner()
    output = tmp_path / "qualification.json"

    exit_code = runner.run_qualification(output=output, environment={})

    assert exit_code == 2
    evidence = runner.load_evidence(output)
    assert evidence["schema"] == "phase5-live-qualification-v1"
    assert evidence["status"] == "UNAVAILABLE"
    assert evidence["missing_configuration"] == sorted(
        {
            "CACHENESS_TEST_MANIFEST_KEY_B64",
            "CACHENESS_TEST_POSTGRES_DSN",
            "CACHENESS_TEST_S3_BUCKET",
        }
    )
    runner.validate_evidence(evidence)


@pytest.mark.parametrize(
    ("completed", "cleanup_status", "expected_exit_code"),
    [
        (subprocess.CompletedProcess([], 1, "1 failed", ""), "CLEAN", 1),
        (subprocess.CompletedProcess([], 0, "2 passed, 1 skipped", ""), "CLEAN", 1),
        (subprocess.CompletedProcess([], 0, "0 items collected", ""), "CLEAN", 1),
        (subprocess.CompletedProcess([], 0, "3 passed", ""), "RESIDUE", 1),
        (subprocess.CompletedProcess([], 0, "3 passed", ""), "CLEAN", 0),
    ],
)
def test_only_a_complete_live_run_and_clean_cleanup_can_qualify(
    tmp_path: Path,
    completed: subprocess.CompletedProcess[str],
    cleanup_status: str,
    expected_exit_code: int,
) -> None:
    """Skipped, empty, failed, or uncleared runs cannot create a claim."""
    runner = _load_runner()
    output = tmp_path / "qualification.json"
    environment = {
        "CACHENESS_TEST_POSTGRES_DSN": "postgresql://test:password@db/qualification",
        "CACHENESS_TEST_S3_BUCKET": "qualification-bucket",
        "CACHENESS_TEST_MANIFEST_KEY_B64": "dGVzdC1tYW5pZmVzdC1rZXktZm9yLXJlZGFjdGlvbi1vbmx5",
    }

    exit_code = runner.run_qualification(
        output=output,
        environment=environment,
        run_tests=lambda _arguments, _timeout: completed,
        resolve_aws=lambda _environment: runner.AwsServiceIdentity(
            region="us-east-1", provider="standard"
        ),
        cleanup=lambda _environment, _run_namespace: cleanup_status,
    )

    assert exit_code == expected_exit_code
    evidence = runner.load_evidence(output)
    assert evidence["status"] == (
        "QUALIFIED" if expected_exit_code == 0 else "NOT_QUALIFIED"
    )
    runner.validate_evidence(evidence)


def test_evidence_serializer_rejects_secret_fragments_and_non_allowlisted_fields(
    tmp_path: Path,
) -> None:
    """Evidence may never serialize connection, signing, or payload material."""
    runner = _load_runner()
    output = tmp_path / "qualification.json"
    evidence = runner.make_evidence(
        status="NOT_QUALIFIED",
        missing_configuration=[],
        run_namespace="phase5-sentinel-run",
        aws_identity=runner.AwsServiceIdentity(region="us-east-1", provider="standard"),
        result="failed",
        cleanup_status="CLEAN",
    )

    runner.write_evidence(
        output,
        evidence,
        forbidden_fragments=("postgresql://test:password@db", "sentinel-manifest-key"),
    )
    serialized = output.read_text(encoding="utf-8")
    assert "postgresql://" not in serialized
    assert "sentinel-manifest-key" not in serialized

    with pytest.raises(ValueError, match="allow-list|secret|credential|URL"):
        runner.write_evidence(
            output,
            {**evidence, "dsn": "postgresql://test:password@db"},
            forbidden_fragments=(),
        )
