"""Contracts for the fail-closed Phase 5 live-service qualification gate."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPOSITORY_ROOT / "tools" / "run_phase5_qualification.py"
FIXTURES_PATH = REPOSITORY_ROOT / "tests" / "qualification" / "conftest.py"
VERIFIER_PATH = REPOSITORY_ROOT / "tools" / "verify_phase5_contracts.py"


def _load_runner():
    """Load the standalone runner without requiring ``tools`` to be a package."""
    spec = importlib.util.spec_from_file_location("phase5_qualification_runner", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_fixtures():
    """Load qualification fixtures directly for non-live ownership contracts."""
    spec = importlib.util.spec_from_file_location("phase5_qualification_fixtures", FIXTURES_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_verifier():
    """Load the read-only evidence verifier without importing ``tools`` as a package."""
    spec = importlib.util.spec_from_file_location("phase5_contract_verifier", VERIFIER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_live_qualification_requires_an_explicit_region_without_owner_pinning() -> None:
    """Phase 8's live gate keeps D-16's explicit AWS configuration boundary."""
    fixtures = _load_fixtures()
    environment = {
        "CACHENESS_TEST_POSTGRES_DSN": "postgresql://test:password@db/qualification",
        "CACHENESS_TEST_S3_BUCKET": "qualification-bucket",
        "CACHENESS_TEST_MANIFEST_KEY_B64": "bW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW0=",
    }

    with pytest.raises(fixtures.QualificationConfigurationError, match="AWS region"):
        fixtures.qualification_config_from_environment(environment)

    config = fixtures.qualification_config_from_environment(
        {**environment, "CACHENESS_TEST_AWS_REGION": "us-east-1"}
    )
    assert config.region == "us-east-1"
    assert not hasattr(config, "expected_bucket_owner")
    assert "ExpectedBucketOwner" not in FIXTURES_PATH.read_text(encoding="utf-8")


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
            "CACHENESS_TEST_AWS_REGION",
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipped, empty, failed, or uncleared runs cannot create a claim."""
    runner = _load_runner()
    monkeypatch.setattr(runner, "_qualification_source_revision", lambda: "a" * 40)
    output = tmp_path / "qualification.json"
    environment = {
        "CACHENESS_TEST_POSTGRES_DSN": "postgresql://test:password@db/qualification",
        "CACHENESS_TEST_S3_BUCKET": "qualification-bucket",
        "CACHENESS_TEST_MANIFEST_KEY_B64": "bW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW0=",
        "CACHENESS_TEST_AWS_REGION": "us-east-1",
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


def test_runner_passes_only_the_exact_run_identifier_to_its_live_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture cleanup backstop and child process address the same namespace."""
    runner = _load_runner()
    observed_environment: dict[str, str] = {}

    def fake_suite(
        _arguments: list[str], _timeout: int, environment: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        observed_environment.update(environment)
        return subprocess.CompletedProcess([], 0, "3 passed", "")

    monkeypatch.setattr(runner, "_run_fixed_suite", fake_suite)
    monkeypatch.setattr(runner, "_qualification_source_revision", lambda: "a" * 40)
    exit_code = runner.run_qualification(
        output=tmp_path / "qualification.json",
        environment={
            "CACHENESS_TEST_POSTGRES_DSN": "postgresql://test:password@db/qualification",
            "CACHENESS_TEST_S3_BUCKET": "qualification-bucket",
            "CACHENESS_TEST_MANIFEST_KEY_B64": "bW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW0=",
            "CACHENESS_TEST_AWS_REGION": "us-east-1",
        },
        resolve_aws=lambda _environment: runner.AwsServiceIdentity(
            region="us-east-1", provider="standard"
        ),
        cleanup=lambda _environment, _run_namespace: "CLEAN",
        run_namespace="phase5-" + "c" * 32,
    )

    assert exit_code == 0
    assert observed_environment["CACHENESS_PHASE5_QUALIFICATION_RUN_ID"] == (
        "phase5-" + "c" * 32
    )


def test_frozen_live_suite_loads_the_qualification_fixture_plugin() -> None:
    """The fixed subprocess resolves fixtures before missing services stop it."""
    runner = _load_runner()
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("AWS_", "CACHENESS_TEST_", "CACHENESS_PHASE5_"))
    }

    completed = subprocess.run(
        runner._qualification_arguments(),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    output = f"{completed.stdout}\n{completed.stderr}"

    assert completed.returncode != 0
    assert "explicit AWS region is required" in output
    assert "fixture 'live_qualification_resources' not found" not in output


def test_dirty_qualification_source_prevents_live_subprocess_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dirty implementation/test/tool/doc path is never eligible for a claim."""
    runner = _load_runner()
    called = False

    def should_not_run(
        _arguments: list[str], _timeout: int
    ) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        return subprocess.CompletedProcess([], 0, "3 passed", "")

    monkeypatch.setattr(runner, "_qualification_source_revision", lambda: None)
    output = tmp_path / "qualification.json"
    exit_code = runner.run_qualification(
        output=output,
        environment={
            "CACHENESS_TEST_POSTGRES_DSN": "postgresql://test:password@db/qualification",
            "CACHENESS_TEST_S3_BUCKET": "qualification-bucket",
            "CACHENESS_TEST_MANIFEST_KEY_B64": "bW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW0=",
            "CACHENESS_TEST_AWS_REGION": "us-east-1",
        },
        run_tests=should_not_run,
    )

    evidence = runner.load_evidence(output)
    assert exit_code == 1
    assert called is False
    assert evidence["status"] == "NOT_QUALIFIED"
    assert evidence["result"] == "not_run"
    assert evidence["cleanup_status"] == "NOT_ATTEMPTED"


def test_source_revision_rejects_a_dirty_imported_production_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every imported production module is part of the qualified revision."""
    runner = _load_runner()
    revision = "a" * 40

    def git_with_dirty_runner_path(*arguments, **_kwargs):
        command = arguments[0]
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, f"{revision}\n", "")
        assert command[:3] == ["git", "status", "--porcelain"]
        assert "src/cacheness" in command
        assert "tests" in command
        assert "src/cacheness/storage" not in command
        assert "tests/integration" not in command
        return subprocess.CompletedProcess(
            command, 0, " M src/cacheness/config.py\n", ""
        )

    monkeypatch.setattr(runner.subprocess, "run", git_with_dirty_runner_path)

    assert runner._qualification_source_revision() is None


def test_endpoint_override_is_not_a_live_amazon_s3_configuration(tmp_path: Path) -> None:
    """A compatible endpoint cannot be substituted for the real AWS gate."""
    runner = _load_runner()
    called = False

    def should_not_run(
        _arguments: list[str], _timeout: int
    ) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        return subprocess.CompletedProcess([], 0, "3 passed", "")

    exit_code = runner.run_qualification(
        output=tmp_path / "qualification.json",
        environment={
            "CACHENESS_TEST_POSTGRES_DSN": "postgresql://test:password@db/qualification",
            "CACHENESS_TEST_S3_BUCKET": "qualification-bucket",
            "CACHENESS_TEST_MANIFEST_KEY_B64": "bW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW0=",
            "CACHENESS_TEST_AWS_REGION": "us-east-1",
            "AWS_ENDPOINT_URL_S3": "http://localhost:4566",
        },
        run_tests=should_not_run,
    )

    assert exit_code == 1
    assert called is False
    assert runner.load_evidence(tmp_path / "qualification.json")["status"] == "NOT_QUALIFIED"


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


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        (
            "qualified result failed",
            lambda evidence: evidence.update(result="failed"),
        ),
        (
            "qualified cleanup has residue",
            lambda evidence: evidence.update(cleanup_status="RESIDUE"),
        ),
        (
            "qualified has missing configuration",
            lambda evidence: evidence.update(
                missing_configuration=["CACHENESS_TEST_S3_BUCKET"]
            ),
        ),
        (
            "qualified lacks AWS identity",
            lambda evidence: evidence["services"].update(aws=None),
        ),
        (
            "qualified has nonstandard AWS identity",
            lambda evidence: evidence["services"].update(
                aws={"provider": "emulated", "region": "us-east-1", "service": "amazon-s3"}
            ),
        ),
        (
            "qualified has unavailable revision",
            lambda evidence: evidence.update(revision="unavailable"),
        ),
        (
            "qualified has malformed revision",
            lambda evidence: evidence.update(revision="A" * 40),
        ),
        (
            "unavailable claims a completed run",
            lambda evidence: evidence.update(
                status="UNAVAILABLE", result="passed", cleanup_status="NOT_ATTEMPTED"
            ),
        ),
        (
            "unavailable claims cleanup completed",
            lambda evidence: evidence.update(
                status="UNAVAILABLE", result="not_run", cleanup_status="CLEAN"
            ),
        ),
    ],
)
def test_forged_contradictory_evidence_is_invalid_to_runner_and_verifier(
    tmp_path: Path, case: str, mutate
) -> None:
    """Shape-valid fields cannot be combined into an unsupported release claim."""
    runner = _load_runner()
    verifier = _load_verifier()
    evidence = runner.make_evidence(
        status="QUALIFIED",
        missing_configuration=[],
        run_namespace="phase5-forged-evidence",
        aws_identity=runner.AwsServiceIdentity(region="us-east-1", provider="standard"),
        result="passed",
        cleanup_status="CLEAN",
    )
    mutate(evidence)
    artifact = tmp_path / f"{case.replace(' ', '-')}.json"
    artifact.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ValueError):
        runner.validate_evidence(evidence)
    assert verifier.read_live_evidence_status(artifact) == "invalid"


def test_fixture_configuration_uses_one_bounded_owned_namespace_and_two_signers() -> None:
    """Fixtures derive service resource names from one exact random run identifier."""
    fixtures = _load_fixtures()
    environment = {
        "CACHENESS_TEST_POSTGRES_DSN": "postgresql://test:password@db/qualification",
        "CACHENESS_TEST_S3_BUCKET": "qualification-bucket",
        "CACHENESS_TEST_MANIFEST_KEY_B64": "bW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW1tbW0=",
        "CACHENESS_TEST_AWS_REGION": "us-east-1",
    }

    config = fixtures.qualification_config_from_environment(environment)
    namespace = fixtures.qualification_namespace("phase5-" + "a" * 32)
    first, second = fixtures.independent_manifest_signers(config)

    assert namespace.schema == "cacheness_q5_" + "a" * 32
    assert namespace.prefix == "cacheness-qualification/phase5-" + "a" * 32 + "/"
    assert first is not second
    assert first.get_key() == second.get_key() == b"m" * 32
    assert "m" * 32 not in repr(first)


def test_bounded_cleanup_refuses_unowned_or_prefix_escaping_s3_objects() -> None:
    """Cleanup reports residue rather than deleting outside its owned run namespace."""
    fixtures = _load_fixtures()
    namespace = fixtures.qualification_namespace("phase5-" + "b" * 32)

    class FakeS3:
        def __init__(self) -> None:
            self.deleted = False

        def get_object(self, **_kwargs: object) -> dict[str, object]:
            return {"Body": _Body(b'{"run_id":"wrong-run"}')}

        def list_objects_v2(self, **_kwargs: object) -> dict[str, object]:
            return {"Contents": [{"Key": "outside/never-delete", "Size": 1}]}

        def delete_objects(self, **_kwargs: object) -> dict[str, object]:
            self.deleted = True
            return {}

    class _Body:
        def __init__(self, value: bytes) -> None:
            self._value = value

        def read(self, _size: int) -> bytes:
            return self._value

        def close(self) -> None:
            return None

    client = FakeS3()
    assert fixtures.cleanup_s3_run(client, "bucket", namespace) == "RESIDUE"
    assert client.deleted is False


def test_bounded_cleanup_keeps_its_marker_until_all_later_pages_finish() -> None:
    """A failed later delete batch leaves the owned prefix safely retryable."""
    fixtures = _load_fixtures()
    namespace = fixtures.qualification_namespace("phase5-" + "d" * 32)

    class _Body:
        def __init__(self, value: bytes) -> None:
            self._value = value

        def read(self, _size: int) -> bytes:
            return self._value

        def close(self) -> None:
            return None

    class FakeS3:
        def __init__(self) -> None:
            self.objects = {
                namespace.s3_owner_marker_key: fixtures._owner_marker_payload(namespace),
                f"{namespace.prefix}payload/first": b"one",
                f"{namespace.prefix}payload/second": b"two",
            }
            self.fail_later_batch = True
            self.delete_batches = 0
            self.marker_deletions: list[str] = []

        def get_object(self, *, Key: str, **_kwargs: object) -> dict[str, object]:
            return {"Body": _Body(self.objects[Key])}

        def list_objects_v2(self, **kwargs: object) -> dict[str, object]:
            keys = sorted(self.objects)
            continuation = kwargs.get("ContinuationToken")
            if continuation is None and len(keys) > 2:
                page_keys = keys[:2]
                return {
                    "Contents": [
                        {"Key": key, "Size": len(self.objects[key])}
                        for key in page_keys
                    ],
                    "IsTruncated": True,
                    "NextContinuationToken": "later-page",
                }
            if continuation == "later-page":
                page_keys = [
                    key for key in keys if key == f"{namespace.prefix}payload/second"
                ]
            else:
                page_keys = keys
            return {
                "Contents": [
                    {"Key": key, "Size": len(self.objects[key])}
                    for key in page_keys
                ],
                "IsTruncated": False,
            }

        def delete_objects(self, *, Delete: dict[str, object], **_kwargs: object) -> dict[str, object]:
            self.delete_batches += 1
            if self.fail_later_batch and self.delete_batches == 2:
                raise RuntimeError("simulated later page failure")
            for item in Delete["Objects"]:
                self.objects.pop(item["Key"])
            return {}

        def list_multipart_uploads(self, **_kwargs: object) -> dict[str, object]:
            return {"Uploads": [], "IsTruncated": False}

        def abort_multipart_upload(self, **_kwargs: object) -> None:
            raise AssertionError("no uploads should be aborted")

        def delete_object(self, *, Key: str, **_kwargs: object) -> None:
            self.marker_deletions.append(Key)
            self.objects.pop(Key)

    client = FakeS3()

    assert fixtures.cleanup_s3_run(client, "bucket", namespace) == "RESIDUE"
    assert namespace.s3_owner_marker_key in client.objects
    assert client.marker_deletions == []

    client.fail_later_batch = False
    assert fixtures.cleanup_s3_run(client, "bucket", namespace) == "CLEAN"
    assert client.objects == {}
    assert client.marker_deletions == [namespace.s3_owner_marker_key]
