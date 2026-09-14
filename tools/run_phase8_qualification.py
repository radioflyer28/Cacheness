#!/usr/bin/env python3
"""Write fail-closed real PostgreSQL/Amazon-S3 qualification evidence for Phase 8.

This runner is intentionally an evidence producer, not a readiness probe.  Its
only successful state is a complete frozen suite on a clean exact revision,
using standard AWS identity and leaving no qualification resources behind.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "build" / "phase8" / "live_qualification.json"
EVIDENCE_SCHEMA = "phase8-live-qualification-v1"
REQUIRED_CONFIGURATION = (
    "CACHENESS_TEST_POSTGRES_DSN",
    "CACHENESS_TEST_S3_BUCKET",
    "CACHENESS_TEST_MANIFEST_KEY_B64",
    "CACHENESS_TEST_AWS_REGION",
)
LIVE_TEST_MODULES = (
    "tests/integration/test_postgresql_authority.py",
    "tests/integration/test_s3_generation.py",
    "tests/integration/test_remote_topology.py",
)
LIVE_MARKER_EXPRESSION = "live_postgresql or live_aws_s3 or live_remote"
QUALIFICATION_FIXTURE_PLUGIN = "tests.qualification.conftest"
QUALIFICATION_SOURCE_PATHS = (
    "pyproject.toml",
    "docs/CATALOG_AND_TOPOLOGY.md",
    "docs/STORAGE_INITIALIZATION.md",
    "src/cacheness/cache_policy.py",
    "src/cacheness/core.py",
    "src/cacheness/storage/blob_store.py",
    "src/cacheness/storage/composition.py",
    "src/cacheness/storage/lifecycle.py",
    "src/cacheness/storage/obstore_generation_io.py",
    "src/cacheness/storage/backends/postgresql_lifecycle_authority.py",
    "tests/qualification/conftest.py",
    "tests/qualification/test_phase8_evidence.py",
    *LIVE_TEST_MODULES,
    "tools/phase8_evidence.py",
    "tools/run_phase8_qualification.py",
    ".github/workflows/live_qualification.yml",
)
DEFAULT_TIMEOUT_SECONDS = 900
MIN_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 3600

_ALLOWED_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "status",
        "revision",
        "source_digest",
        "generated_at_utc",
        "run_namespace",
        "missing_configuration",
        "services",
        "fixed_suite",
        "result",
        "cleanup_status",
    }
)
_ALLOWED_SERVICE_KEYS = frozenset(
    {
        "python",
        "pytest",
        "obstore",
        "psycopg",
        "postgresql",
        "payload_participant",
        "aws",
    }
)
_ALLOWED_AWS_KEYS = frozenset({"provider", "region", "service"})
_ALLOWED_SUITE_KEYS = frozenset({"modules", "marker_expression", "complete"})
_STATUS_VALUES = frozenset({"QUALIFIED", "UNAVAILABLE", "NOT_QUALIFIED"})
_RESULT_VALUES = frozenset(
    {"not_run", "passed", "failed", "incomplete", "configuration_error"}
)
_CLEANUP_VALUES = frozenset({"NOT_ATTEMPTED", "CLEAN", "RESIDUE", "ERROR"})
_FORBIDDEN_VALUE_PATTERN = re.compile(
    r"://|access[_-]?key|credential|password|secret|token|private[_-]?key|payload",
    re.IGNORECASE,
)
_SECRET_ENVIRONMENT_NAME = re.compile(
    r"(?:^AWS_(?:ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN)$|"
    r"(?:DSN|KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL))",
    re.IGNORECASE,
)
_DISALLOWED_S3_CONFIGURATION = (
    "CACHENESS_TEST_S3_ENDPOINT",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_S3",
    "CACHENESS_TEST_EXPECTED_BUCKET_OWNER",
    "EXPECTED_BUCKET_OWNER",
)
_NON_AWS_LIVE_SUITE_SOURCE = re.compile(
    r"\b(?:moto|mock_aws|localstack|minio)\b|endpoint_url|127\.0\.0\.1|localhost",
    re.IGNORECASE,
)
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class AwsServiceIdentity:
    """The non-secret AWS identity eligible for Amazon S3 qualification."""

    region: str
    provider: str
    service: str = "amazon-s3"


@dataclass(frozen=True)
class SourceIdentity:
    """One reviewed immutable source identity for a live candidate."""

    revision: str
    digest: str


class QualificationUnavailableError(RuntimeError):
    """An external prerequisite is unavailable without exposing its details."""


def _git_revision() -> str:
    """Return the current revision without propagating command diagnostics."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    revision = completed.stdout.strip()
    return revision if _REVISION_PATTERN.fullmatch(revision) else "unavailable"


def _source_files() -> tuple[Path, ...] | None:
    """Resolve the fixed reviewed source inventory without adaptive discovery."""
    files: list[Path] = []
    for relative_path in QUALIFICATION_SOURCE_PATHS:
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            return None
        files.append(path)
    return tuple(files)


def _relevant_source_digest() -> str | None:
    """Hash the fixed source inventory by repository-relative path and bytes."""
    files = _source_files()
    if files is None:
        return None
    digest = hashlib.sha256()
    try:
        for path in files:
            relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    except OSError:
        return None
    return digest.hexdigest()


def _qualification_source_identity() -> SourceIdentity | None:
    """Return a clean revision plus fixed-inventory digest eligible for live proof."""
    revision = _git_revision()
    digest = _relevant_source_digest()
    if not _REVISION_PATTERN.fullmatch(revision) or digest is None:
        return None
    try:
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *QUALIFICATION_SOURCE_PATHS,
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if status.stdout.strip():
        return None
    return SourceIdentity(revision=revision, digest=digest)


def _package_version(package: str) -> str:
    """Return an installed package version without importing its runtime API."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _new_run_namespace() -> str:
    """Generate one random exact namespace for the one qualification invocation."""
    return f"phase8-{secrets.token_hex(16)}"


def _redacted_namespace(run_namespace: str) -> str:
    """Write a stable non-resource reference instead of a schema or S3 prefix."""
    return f"sha256:{hashlib.sha256(run_namespace.encode('utf-8')).hexdigest()[:16]}"


def _missing_configuration(environment: Mapping[str, str]) -> list[str]:
    """Return configuration names only, never their values."""
    return sorted(name for name in REQUIRED_CONFIGURATION if not environment.get(name))


def _supplied_secret_fragments(environment: Mapping[str, str]) -> tuple[str, ...]:
    """Capture secret-shaped supplied values for the final evidence scan only."""
    return tuple(
        value
        for name, value in environment.items()
        if value and len(value) >= 8 and _SECRET_ENVIRONMENT_NAME.search(name)
    )


def _validate_external_configuration(environment: Mapping[str, str]) -> None:
    """Validate required configuration in memory before any service interaction."""
    bucket = environment["CACHENESS_TEST_S3_BUCKET"]
    region = environment["CACHENESS_TEST_AWS_REGION"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
        raise ValueError("invalid external configuration")
    if not re.fullmatch(r"[a-z]{2}-[a-z]+-\d", region):
        raise ValueError("invalid external configuration")
    try:
        manifest_key = base64.b64decode(
            environment["CACHENESS_TEST_MANIFEST_KEY_B64"], validate=True
        )
    except (ValueError, TypeError) as error:
        raise ValueError("invalid external configuration") from error
    if len(manifest_key) != 32 or any(
        environment.get(name) for name in _DISALLOWED_S3_CONFIGURATION
    ):
        raise ValueError("invalid external configuration")


def resolve_aws_service_identity(environment: Mapping[str, str]) -> AwsServiceIdentity:
    """Confirm the standard AWS provider chain without serializing credentials."""
    try:
        import boto3
    except ImportError as error:
        raise QualificationUnavailableError("AWS SDK unavailable") from error
    try:
        session = boto3.Session(region_name=environment["CACHENESS_TEST_AWS_REGION"])
        credentials = session.get_credentials()
    except Exception as error:
        raise QualificationUnavailableError("AWS credentials unavailable") from error
    if credentials is None:
        raise QualificationUnavailableError("AWS credentials unavailable")
    return AwsServiceIdentity(
        region=environment["CACHENESS_TEST_AWS_REGION"], provider="standard"
    )


def make_evidence(
    *,
    status: str,
    missing_configuration: Sequence[str],
    run_namespace: str,
    aws_identity: AwsServiceIdentity | None,
    result: str,
    cleanup_status: str,
    source_identity: SourceIdentity | None = None,
) -> dict[str, object]:
    """Build the exact sanitized evidence record before it reaches disk."""
    services: dict[str, object] = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "pytest": _package_version("pytest"),
        "obstore": _package_version("obstore"),
        "psycopg": _package_version("psycopg"),
        "postgresql": "real-postgresql",
        "payload_participant": "ObstoreGenerationIO",
        "aws": None,
    }
    if aws_identity is not None:
        services["aws"] = {
            "provider": aws_identity.provider,
            "region": aws_identity.region,
            "service": aws_identity.service,
        }
    return {
        "schema": EVIDENCE_SCHEMA,
        "status": status,
        "revision": source_identity.revision if source_identity else "unavailable",
        "source_digest": source_identity.digest if source_identity else "unavailable",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_namespace": _redacted_namespace(run_namespace),
        "missing_configuration": list(missing_configuration),
        "services": services,
        "fixed_suite": {
            "modules": list(LIVE_TEST_MODULES),
            "marker_expression": LIVE_MARKER_EXPRESSION,
            "complete": result == "passed",
        },
        "result": result,
        "cleanup_status": cleanup_status,
    }


def _validate_safe_text(value: object) -> None:
    if not isinstance(value, str) or not value or _FORBIDDEN_VALUE_PATTERN.search(value):
        raise ValueError("evidence contains unsafe text")


def _is_standard_amazon_s3_identity(aws: object) -> bool:
    return (
        isinstance(aws, Mapping)
        and set(aws) == _ALLOWED_AWS_KEYS
        and aws.get("provider") == "standard"
        and aws.get("service") == "amazon-s3"
        and isinstance(aws.get("region"), str)
        and bool(aws["region"])
    )


def validate_evidence(evidence: Mapping[str, object]) -> None:
    """Reject any contradictory, incomplete, secret-bearing evidence document."""
    if set(evidence) != _ALLOWED_EVIDENCE_KEYS:
        raise ValueError("evidence violates the exact allow-list")
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("evidence schema is invalid")
    if evidence.get("status") not in _STATUS_VALUES:
        raise ValueError("evidence status is invalid")
    if evidence.get("result") not in _RESULT_VALUES:
        raise ValueError("evidence result is invalid")
    if evidence.get("cleanup_status") not in _CLEANUP_VALUES:
        raise ValueError("evidence cleanup status is invalid")
    revision = evidence.get("revision")
    source_digest = evidence.get("source_digest")
    if revision != "unavailable" and not (
        isinstance(revision, str) and _REVISION_PATTERN.fullmatch(revision)
    ):
        raise ValueError("evidence revision is invalid")
    if source_digest != "unavailable" and not (
        isinstance(source_digest, str) and _DIGEST_PATTERN.fullmatch(source_digest)
    ):
        raise ValueError("evidence source digest is invalid")
    timestamp = evidence.get("generated_at_utc")
    if not isinstance(timestamp, str):
        raise ValueError("evidence timestamp is invalid")
    try:
        if datetime.fromisoformat(timestamp).tzinfo is None:
            raise ValueError("evidence timestamp is invalid")
    except ValueError as error:
        raise ValueError("evidence timestamp is invalid") from error
    namespace = evidence.get("run_namespace")
    if not isinstance(namespace, str) or not re.fullmatch(r"sha256:[0-9a-f]{16}", namespace):
        raise ValueError("evidence namespace is invalid")
    missing = evidence.get("missing_configuration")
    if not isinstance(missing, list) or any(
        not isinstance(name, str) or name not in REQUIRED_CONFIGURATION for name in missing
    ):
        raise ValueError("evidence missing configuration is invalid")
    services = evidence.get("services")
    if not isinstance(services, Mapping) or set(services) != _ALLOWED_SERVICE_KEYS:
        raise ValueError("evidence services violate the exact allow-list")
    for name in ("python", "pytest", "obstore", "psycopg", "postgresql", "payload_participant"):
        _validate_safe_text(services.get(name))
    if services.get("postgresql") != "real-postgresql" or services.get("payload_participant") != "ObstoreGenerationIO":
        raise ValueError("evidence service topology is invalid")
    aws = services.get("aws")
    if aws is not None and not _is_standard_amazon_s3_identity(aws):
        raise ValueError("evidence AWS identity is invalid")
    if isinstance(aws, Mapping):
        for value in aws.values():
            _validate_safe_text(value)
    fixed_suite = evidence.get("fixed_suite")
    if not isinstance(fixed_suite, Mapping) or set(fixed_suite) != _ALLOWED_SUITE_KEYS:
        raise ValueError("evidence fixed suite is invalid")
    if fixed_suite.get("modules") != list(LIVE_TEST_MODULES) or fixed_suite.get("marker_expression") != LIVE_MARKER_EXPRESSION or not isinstance(fixed_suite.get("complete"), bool):
        raise ValueError("evidence fixed suite is invalid")
    for value in (evidence["schema"], evidence["status"], timestamp, namespace, evidence["result"], evidence["cleanup_status"]):
        _validate_safe_text(value)

    status = evidence["status"]
    complete_proof = (
        evidence["result"] == "passed"
        and evidence["cleanup_status"] == "CLEAN"
        and not missing
        and fixed_suite["complete"] is True
        and _is_standard_amazon_s3_identity(aws)
        and isinstance(revision, str)
        and _REVISION_PATTERN.fullmatch(revision)
        and isinstance(source_digest, str)
        and _DIGEST_PATTERN.fullmatch(source_digest)
    )
    if status == "QUALIFIED" and not complete_proof:
        raise ValueError("qualified evidence contradicts its required service proof")
    if status == "UNAVAILABLE" and (
        evidence["result"] != "not_run" or evidence["cleanup_status"] != "NOT_ATTEMPTED"
    ):
        raise ValueError("unavailable evidence contradicts its terminal state")
    if status != "QUALIFIED" and complete_proof:
        raise ValueError("complete qualification evidence must be marked qualified")


def write_evidence(output: Path, evidence: Mapping[str, object], *, forbidden_fragments: Sequence[str]) -> None:
    """Validate and atomically write one bounded evidence document."""
    validate_evidence(evidence)
    serialized = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    for fragment in forbidden_fragments:
        if fragment and fragment in serialized:
            raise ValueError("evidence contains a supplied secret fragment")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as temporary:
        temporary.write(serialized)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_evidence(path: Path) -> dict[str, object]:
    """Read one evidence document for deterministic contract tests."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("evidence root must be an object")
    return loaded


def _qualification_arguments() -> list[str]:
    """Return the one frozen module set; callers cannot supply a subset."""
    return [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        QUALIFICATION_FIXTURE_PLUGIN,
        "-q",
        "-ra",
        "-o",
        "log_cli=false",
        *LIVE_TEST_MODULES,
        "-m",
        LIVE_MARKER_EXPRESSION,
    ]


def _fixed_live_suite_is_real() -> bool:
    """Reject an absent, mocked, or endpoint-overridden frozen suite."""
    for relative_path in LIVE_TEST_MODULES:
        try:
            source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        except OSError:
            return False
        if _NON_AWS_LIVE_SUITE_SOURCE.search(source):
            return False
    return True


def _timeout_seconds(environment: Mapping[str, str]) -> int:
    raw_timeout = environment.get("CACHENESS_TEST_QUALIFICATION_TIMEOUT_SECONDS")
    if raw_timeout is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = int(raw_timeout)
    except ValueError as error:
        raise ValueError("invalid qualification timeout") from error
    if not MIN_TIMEOUT_SECONDS <= timeout <= MAX_TIMEOUT_SECONDS:
        raise ValueError("invalid qualification timeout")
    return timeout


def _run_fixed_suite(arguments: Sequence[str], timeout: int, environment: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    """Run the sealed suite without forwarding possibly secret-bearing output."""
    if not _fixed_live_suite_is_real():
        return subprocess.CompletedProcess(list(arguments), 1, "", "")
    return subprocess.run(
        list(arguments), cwd=REPOSITORY_ROOT, capture_output=True, text=True,
        timeout=timeout, check=False, env=dict(environment)
    )


def _cleanup_from_fixtures(environment: Mapping[str, str], run_namespace: str) -> str:
    """Use fixture-owned exact cleanup without importing ``tests`` as a package."""
    path = REPOSITORY_ROOT / "tests" / "qualification" / "conftest.py"
    spec = importlib.util.spec_from_file_location("phase8_qualification_fixtures", path)
    if spec is None or spec.loader is None:
        return "ERROR"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cleanup = getattr(module, "cleanup_qualification_resources", None)
    if not callable(cleanup):
        return "ERROR"
    result = cleanup(environment, run_namespace)
    return result if result in _CLEANUP_VALUES else "ERROR"


def _is_complete_pass(completed: subprocess.CompletedProcess[str]) -> bool:
    """Reject all non-pass, skip, deselection, timeout, and xfail outcomes."""
    output = f"{completed.stdout or ''}\n{completed.stderr or ''}".lower()
    incomplete_markers = (
        " skipped", " deselected", "no tests ran", "0 items collected", "collected 0 items",
        " xfailed", " xpassed", " interrupted", " timeout", " timed out", " error",
    )
    return completed.returncode == 0 and not any(marker in output for marker in incomplete_markers) and re.search(r"\b[1-9][0-9]* passed\b", output) is not None


def _write_terminal_evidence(*, output: Path, status: str, missing_configuration: Sequence[str], run_namespace: str, aws_identity: AwsServiceIdentity | None, result: str, cleanup_status: str, forbidden_fragments: Sequence[str], source_identity: SourceIdentity | None = None) -> None:
    write_evidence(
        output,
        make_evidence(
            status=status, missing_configuration=missing_configuration,
            run_namespace=run_namespace, aws_identity=aws_identity, result=result,
            cleanup_status=cleanup_status, source_identity=source_identity,
        ),
        forbidden_fragments=forbidden_fragments,
    )


def run_qualification(*, output: Path, environment: Mapping[str, str] | None = None, run_tests: Callable[[Sequence[str], int], subprocess.CompletedProcess[str]] | None = None, resolve_aws: Callable[[Mapping[str, str]], AwsServiceIdentity] = resolve_aws_service_identity, cleanup: Callable[[Mapping[str, str], str], str] = _cleanup_from_fixtures, run_namespace: str | None = None) -> int:
    """Write one terminal record and return 0, 1, or 2 for its fixed status."""
    supplied_environment = dict(os.environ if environment is None else environment)
    namespace = run_namespace or _new_run_namespace()
    forbidden_fragments = _supplied_secret_fragments(supplied_environment)
    missing = _missing_configuration(supplied_environment)
    if missing:
        _write_terminal_evidence(output=output, status="UNAVAILABLE", missing_configuration=missing, run_namespace=namespace, aws_identity=None, result="not_run", cleanup_status="NOT_ATTEMPTED", forbidden_fragments=forbidden_fragments)
        return 2
    try:
        _validate_external_configuration(supplied_environment)
        timeout = _timeout_seconds(supplied_environment)
    except ValueError:
        _write_terminal_evidence(output=output, status="NOT_QUALIFIED", missing_configuration=(), run_namespace=namespace, aws_identity=None, result="configuration_error", cleanup_status="NOT_ATTEMPTED", forbidden_fragments=forbidden_fragments)
        return 1
    source_identity = _qualification_source_identity()
    if source_identity is None:
        _write_terminal_evidence(output=output, status="NOT_QUALIFIED", missing_configuration=(), run_namespace=namespace, aws_identity=None, result="not_run", cleanup_status="NOT_ATTEMPTED", forbidden_fragments=forbidden_fragments)
        return 1
    try:
        aws_identity = resolve_aws(supplied_environment)
    except QualificationUnavailableError:
        _write_terminal_evidence(output=output, status="UNAVAILABLE", missing_configuration=(), run_namespace=namespace, aws_identity=None, result="not_run", cleanup_status="NOT_ATTEMPTED", forbidden_fragments=forbidden_fragments, source_identity=source_identity)
        return 2

    completed: subprocess.CompletedProcess[str] | None = None
    cleanup_status = "ERROR"
    child_environment = dict(supplied_environment)
    child_environment["CACHENESS_PHASE8_QUALIFICATION_RUN_ID"] = namespace
    try:
        completed = run_tests(_qualification_arguments(), timeout) if run_tests else _run_fixed_suite(_qualification_arguments(), timeout, child_environment)
    except (OSError, subprocess.SubprocessError):
        completed = None
    finally:
        try:
            cleanup_status = cleanup(supplied_environment, namespace)
        except Exception:
            cleanup_status = "ERROR"
    complete = completed is not None and _is_complete_pass(completed)
    source_stable = _qualification_source_identity() == source_identity
    qualified = complete and cleanup_status == "CLEAN" and source_stable
    _write_terminal_evidence(
        output=output, status="QUALIFIED" if qualified else "NOT_QUALIFIED",
        missing_configuration=(), run_namespace=namespace, aws_identity=aws_identity,
        result="passed" if complete and source_stable else "incomplete" if completed is None or not source_stable else "failed",
        cleanup_status=cleanup_status, forbidden_fragments=forbidden_fragments,
        source_identity=source_identity,
    )
    return 0 if qualified else 1


def main(arguments: Sequence[str] | None = None) -> int:
    """Parse one output location without permitting selector substitution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return run_qualification(output=parser.parse_args(arguments).output)


if __name__ == "__main__":
    raise SystemExit(main())
