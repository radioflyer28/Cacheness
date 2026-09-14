"""Fail-closed contracts for the Phase 8 real-service qualification runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPOSITORY_ROOT / "tools" / "run_phase8_qualification.py"


def _load_runner():
    """Load the standalone runner without importing ``tools`` as a package."""
    spec = importlib.util.spec_from_file_location("phase8_qualification_runner", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _configured_environment() -> dict[str, str]:
    """Return representative configuration values that must never be serialized."""
    return {
        "CACHENESS_TEST_POSTGRES_DSN": "postgresql://test:password@db/qualification",
        "CACHENESS_TEST_S3_BUCKET": "qualification-bucket",
        "CACHENESS_TEST_MANIFEST_KEY_B64": "bW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW0=",
        "CACHENESS_TEST_AWS_REGION": "us-east-1",
    }


def test_runner_uses_only_the_frozen_live_obstore_suite() -> None:
    """The real-service gate names the three exact modules and current participant."""
    runner = _load_runner()

    assert runner.LIVE_TEST_MODULES == (
        "tests/integration/test_postgresql_authority.py",
        "tests/integration/test_s3_generation.py",
        "tests/integration/test_remote_topology.py",
    )
    assert runner.LIVE_MARKER_EXPRESSION == "live_postgresql or live_aws_s3 or live_remote"
    assert "src/cacheness/storage/obstore_generation_io.py" in runner.QUALIFICATION_SOURCE_PATHS
    assert "tools/run_phase8_qualification.py" in runner.QUALIFICATION_SOURCE_PATHS
    assert "tools/run_phase5_qualification.py" not in runner.QUALIFICATION_SOURCE_PATHS


def test_missing_configuration_writes_sanitized_unavailable_phase8_evidence(
    tmp_path: Path,
) -> None:
    """Absent service configuration is an explicit, non-qualifying terminal state."""
    runner = _load_runner()
    output = tmp_path / "live.json"

    assert runner.run_qualification(output=output, environment={}) == 2

    evidence = runner.load_evidence(output)
    assert evidence["schema"] == "phase8-live-qualification-v1"
    assert evidence["status"] == "UNAVAILABLE"
    assert evidence["result"] == "not_run"
    assert evidence["cleanup_status"] == "NOT_ATTEMPTED"
    assert evidence["missing_configuration"] == sorted(runner.REQUIRED_CONFIGURATION)
    runner.validate_evidence(evidence)


@pytest.mark.parametrize(
    "completed",
    (
        subprocess.CompletedProcess([], 0, "4 passed", ""),
        subprocess.CompletedProcess([], 0, "4 passed, 1 skipped", ""),
        subprocess.CompletedProcess([], 0, "4 passed, 1 deselected", ""),
        subprocess.CompletedProcess([], 0, "4 passed, 1 xfailed", ""),
        subprocess.CompletedProcess([], 1, "1 failed", ""),
    ),
)
def test_runner_requires_complete_clean_current_source_proof(
    tmp_path: Path,
    completed: subprocess.CompletedProcess[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an unskipped clean run at one stable source revision can qualify."""
    runner = _load_runner()
    monkeypatch.setattr(
        runner,
        "_qualification_source_identity",
        lambda: runner.SourceIdentity(revision="a" * 40, digest="b" * 64),
    )
    output = tmp_path / "live.json"

    exit_code = runner.run_qualification(
        output=output,
        environment=_configured_environment(),
        run_tests=lambda _arguments, _timeout: completed,
        resolve_aws=lambda _environment: runner.AwsServiceIdentity(
            region="us-east-1", provider="standard"
        ),
        cleanup=lambda _environment, _run_id: "CLEAN",
        run_namespace="phase8-" + "a" * 32,
    )

    evidence = runner.load_evidence(output)
    assert evidence["status"] == ("QUALIFIED" if completed.returncode == 0 and "passed, " not in completed.stdout else "NOT_QUALIFIED")
    assert evidence["revision"] == "a" * 40
    assert isinstance(evidence["source_digest"], str)
    assert evidence["services"]["aws"] == {
        "provider": "standard",
        "region": "us-east-1",
        "service": "amazon-s3",
    }
    assert evidence["services"]["payload_participant"] == "ObstoreGenerationIO"
    runner.validate_evidence(evidence)
    assert exit_code == (0 if evidence["status"] == "QUALIFIED" else 1)


def test_runner_rejects_endpoint_override_and_owner_pinning_without_running(
    tmp_path: Path,
) -> None:
    """Amazon S3 qualification uses standard identity with no endpoint/owner claim."""
    runner = _load_runner()
    called = False

    def should_not_run(
        _arguments: list[str], _timeout: int
    ) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        return subprocess.CompletedProcess([], 0, "4 passed", "")

    for disallowed_name, disallowed_value in (
        ("AWS_ENDPOINT_URL_S3", "http://localhost:4566"),
        ("CACHENESS_TEST_EXPECTED_BUCKET_OWNER", "123456789012"),
    ):
        output = tmp_path / f"{disallowed_name}.json"
        assert runner.run_qualification(
            output=output,
            environment={**_configured_environment(), disallowed_name: disallowed_value},
            run_tests=should_not_run,
        ) == 1
        assert runner.load_evidence(output)["status"] == "NOT_QUALIFIED"

    assert called is False


def test_runner_refuses_dirty_relevant_sources_before_service_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale runner, fixture, workflow, test, or participant cannot qualify HEAD."""
    runner = _load_runner()
    monkeypatch.setattr(runner, "_qualification_source_identity", lambda: None)

    assert runner.run_qualification(
        output=tmp_path / "live.json",
        environment=_configured_environment(),
    ) == 1

    evidence = runner.load_evidence(tmp_path / "live.json")
    assert evidence["status"] == "NOT_QUALIFIED"
    assert evidence["result"] == "not_run"
    assert evidence["cleanup_status"] == "NOT_ATTEMPTED"
