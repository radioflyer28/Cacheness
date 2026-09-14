"""Fail-closed contracts for the Phase 8 real-service qualification runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPOSITORY_ROOT / "tools" / "run_phase8_qualification.py"
FIXTURES_PATH = REPOSITORY_ROOT / "tests" / "qualification" / "conftest.py"


def _load_runner():
    """Load the standalone runner without importing ``tools`` as a package."""
    spec = importlib.util.spec_from_file_location(
        "phase8_qualification_runner", RUNNER_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_fixtures():
    """Load the real cleanup helpers without making ``tests`` a package."""
    spec = importlib.util.spec_from_file_location(
        "phase8_qualification_fixtures", FIXTURES_PATH
    )
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
    assert (
        runner.LIVE_MARKER_EXPRESSION == "live_postgresql or live_aws_s3 or live_remote"
    )
    assert (
        "src/cacheness/storage/obstore_generation_io.py"
        in runner.QUALIFICATION_SOURCE_PATHS
    )
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
    assert evidence["status"] == (
        "QUALIFIED"
        if completed.returncode == 0 and "passed, " not in completed.stdout
        else "NOT_QUALIFIED"
    )
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
        assert (
            runner.run_qualification(
                output=output,
                environment={
                    **_configured_environment(),
                    disallowed_name: disallowed_value,
                },
                run_tests=should_not_run,
            )
            == 1
        )
        assert runner.load_evidence(output)["status"] == "NOT_QUALIFIED"

    assert called is False


def test_runner_refuses_dirty_relevant_sources_before_service_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale runner, fixture, workflow, test, or participant cannot qualify HEAD."""
    runner = _load_runner()
    monkeypatch.setattr(runner, "_qualification_source_identity", lambda: None)

    assert (
        runner.run_qualification(
            output=tmp_path / "live.json",
            environment=_configured_environment(),
        )
        == 1
    )

    evidence = runner.load_evidence(tmp_path / "live.json")
    assert evidence["status"] == "NOT_QUALIFIED"
    assert evidence["result"] == "not_run"
    assert evidence["cleanup_status"] == "NOT_ATTEMPTED"


def test_phase8_cleanup_namespace_is_exact_and_cannot_collide_with_phase5() -> None:
    """A Phase 8 runner has one run-owned schema/prefix without changing Phase 5."""
    fixtures = _load_fixtures()

    phase5 = fixtures.qualification_namespace("phase5-" + "5" * 32)
    phase8 = fixtures.qualification_namespace("phase8-" + "8" * 32)

    assert phase5.schema == "cacheness_q5_" + "5" * 32
    assert phase8.schema == "cacheness_q8_" + "8" * 32
    assert phase8.prefix == "cacheness-qualification/phase8-" + "8" * 32 + "/"
    assert phase5.prefix != phase8.prefix


class _Body:
    """Small S3 body stand-in with the bounded reader surface used by cleanup."""

    def __init__(self, value: bytes) -> None:
        self._value = value

    def read(self, _size: int) -> bytes:
        return self._value

    def close(self) -> None:
        return None


def test_cleanup_rejects_malformed_or_escaping_prefixes_without_delete() -> None:
    """Marker ambiguity and out-of-prefix listings remain residue, never cleanup."""
    fixtures = _load_fixtures()
    namespace = fixtures.qualification_namespace("phase8-" + "a" * 32)

    class FakeS3:
        deleted = False

        def get_object(self, **_kwargs: object) -> dict[str, object]:
            return {"Body": _Body(b'{"run_id":"forged"}')}

        def list_objects_v2(self, **_kwargs: object) -> dict[str, object]:
            return {"Contents": [{"Key": "outside/never-delete", "Size": 1}]}

        def list_multipart_uploads(self, **_kwargs: object) -> dict[str, object]:
            return {"Uploads": []}

        def delete_objects(self, **_kwargs: object) -> dict[str, object]:
            self.deleted = True
            return {}

    client = FakeS3()
    assert fixtures.cleanup_s3_run(client, "bucket", namespace) == "RESIDUE"
    assert client.deleted is False


def test_cleanup_cap_or_partial_delete_keeps_phase8_owner_marker() -> None:
    """Bound exhaustion and service partial success do not authorize marker deletion."""
    fixtures = _load_fixtures()
    namespace = fixtures.qualification_namespace("phase8-" + "b" * 32)

    class FakeS3:
        marker_deleted = False
        delete_attempted = False

        def get_object(self, **_kwargs: object) -> dict[str, object]:
            return {"Body": _Body(fixtures._owner_marker_payload(namespace))}

        def list_multipart_uploads(self, **_kwargs: object) -> dict[str, object]:
            return {"Uploads": [], "IsTruncated": False}

        def list_objects_v2(self, **_kwargs: object) -> dict[str, object]:
            return {
                "Contents": [
                    {"Key": f"{namespace.prefix}payload/{index}", "Size": 1}
                    for index in range(fixtures._MAX_CLEANUP_OBJECTS + 1)
                ],
                "IsTruncated": False,
            }

        def delete_objects(self, **_kwargs: object) -> dict[str, object]:
            self.delete_attempted = True
            return {"Errors": [{"Code": "AccessDenied"}]}

        def delete_object(self, **_kwargs: object) -> None:
            self.marker_deleted = True

    client = FakeS3()
    assert fixtures.cleanup_s3_run(client, "bucket", namespace) == "RESIDUE"
    assert client.delete_attempted is False
    assert client.marker_deleted is False


def test_cleanup_rejects_postgresql_owner_mismatch_and_rolls_back_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema is dropped only after its one exact owner marker is confirmed."""
    fixtures = _load_fixtures()
    namespace = fixtures.qualification_namespace("phase8-" + "c" * 32)

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class Connection:
        committed = False
        rolled_back = False
        closed = False

        def cursor(self) -> Cursor:
            return Cursor()

        def commit(self) -> None:
            self.committed = True

        def rollback(self) -> None:
            self.rolled_back = True

        def close(self) -> None:
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(fixtures, "_postgresql_connection", lambda _config: connection)
    monkeypatch.setattr(fixtures, "_postgresql_schema_exists", lambda *_args: True)
    monkeypatch.setattr(fixtures, "_postgresql_owner_matches", lambda *_args: False)

    assert fixtures.cleanup_postgresql_run(object(), namespace) == "RESIDUE"
    assert connection.committed is False
    assert connection.closed is True


def test_secret_safe_evidence_and_cleanup_exception_remain_nonqualifying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Secret values and cleanup exceptions cannot become clean service proof."""
    runner = _load_runner()
    identity = runner.SourceIdentity(revision="a" * 40, digest="b" * 64)
    evidence = runner.make_evidence(
        status="NOT_QUALIFIED",
        missing_configuration=[],
        run_namespace="phase8-sentinel",
        aws_identity=runner.AwsServiceIdentity(region="us-east-1", provider="standard"),
        result="failed",
        cleanup_status="ERROR",
        source_identity=identity,
    )
    output = tmp_path / "live.json"
    runner.write_evidence(
        output,
        evidence,
        forbidden_fragments=("postgresql://test:password@db/qualification", "m" * 32),
    )
    serialized = output.read_text(encoding="utf-8")
    assert "postgresql://" not in serialized
    assert "m" * 32 not in serialized

    monkeypatch.setattr(runner, "_qualification_source_identity", lambda: identity)
    assert (
        runner.run_qualification(
            output=output,
            environment=_configured_environment(),
            run_tests=lambda _arguments, _timeout: subprocess.CompletedProcess(
                [], 0, "4 passed", ""
            ),
            resolve_aws=lambda _environment: runner.AwsServiceIdentity(
                "us-east-1", "standard"
            ),
            cleanup=lambda _environment, _run_id: (_ for _ in ()).throw(
                RuntimeError("cleanup")
            ),
            run_namespace="phase8-" + "d" * 32,
        )
        == 1
    )
    assert runner.load_evidence(output)["cleanup_status"] == "ERROR"


def test_scheduled_diagnostic_role_cannot_emit_release_candidate_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A healthy scheduled service probe remains diagnostic rather than QUALIFIED."""
    runner = _load_runner()
    identity = runner.SourceIdentity(revision="a" * 40, digest="b" * 64)
    monkeypatch.setattr(runner, "_qualification_source_identity", lambda: identity)

    assert (
        runner.run_qualification(
            output=tmp_path / "scheduled.json",
            environment=_configured_environment(),
            run_tests=lambda _arguments, _timeout: subprocess.CompletedProcess(
                [], 0, "4 passed", ""
            ),
            resolve_aws=lambda _environment: runner.AwsServiceIdentity(
                "us-east-1", "standard"
            ),
            cleanup=lambda _environment, _run_id: "CLEAN",
            run_namespace="phase8-" + "e" * 32,
            run_role="scheduled_diagnostic",
        )
        == 1
    )

    evidence = runner.load_evidence(tmp_path / "scheduled.json")
    assert evidence["run_role"] == "scheduled_diagnostic"
    assert evidence["result"] == "passed"
    assert evidence["cleanup_status"] == "CLEAN"
    assert evidence["status"] == "NOT_QUALIFIED"
    runner.validate_evidence(evidence)
