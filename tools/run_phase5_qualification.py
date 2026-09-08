#!/usr/bin/env python3
"""Run the fail-closed real-service qualification gate for Phase 5.

The command deliberately has only three terminal states.  It is an evidence
writer, not a readiness probe: unavailable configuration and incomplete test
execution leave the PostgreSQL/Amazon-S3 topology unqualified.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
from typing import Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / ".planning/phases/05-payload-backends-and-supported-topology-qualification"
    / "05-LIVE-QUALIFICATION.json"
)
EVIDENCE_SCHEMA = "phase5-live-qualification-v1"
REQUIRED_CONFIGURATION = (
    "CACHENESS_TEST_POSTGRES_DSN",
    "CACHENESS_TEST_S3_BUCKET",
    "CACHENESS_TEST_MANIFEST_KEY_B64",
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
    "src/cacheness",
    "tests",
    "tools/run_phase5_qualification.py",
    "tools/verify_phase5_contracts.py",
)
DEFAULT_TIMEOUT_SECONDS = 900
MIN_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 3600

_ALLOWED_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "status",
        "revision",
        "generated_at_utc",
        "run_namespace",
        "missing_configuration",
        "services",
        "result",
        "cleanup_status",
    }
)
_ALLOWED_SERVICE_KEYS = frozenset({"python", "pytest", "boto3", "psycopg", "aws"})
_ALLOWED_AWS_KEYS = frozenset({"provider", "region", "service"})
_STATUS_VALUES = frozenset({"QUALIFIED", "UNAVAILABLE", "NOT_QUALIFIED"})
_RESULT_VALUES = frozenset(
    {
        "not_run",
        "passed",
        "failed",
        "incomplete",
        "configuration_error",
    }
)
_CLEANUP_VALUES = frozenset({"NOT_ATTEMPTED", "CLEAN", "RESIDUE", "ERROR"})
_FORBIDDEN_VALUE_PATTERN = re.compile(
    r"://|access[_-]?key|credential|manifest|password|secret|token|payload",
    re.IGNORECASE,
)
_SECRET_ENVIRONMENT_NAME = re.compile(
    r"(?:^AWS_(?:ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN)$|"
    r"(?:DSN|KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL))",
    re.IGNORECASE,
)
_DISALLOWED_S3_ENDPOINT_ENVIRONMENT = (
    "CACHENESS_TEST_S3_ENDPOINT",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_S3",
)
_NON_AWS_LIVE_SUITE_SOURCE = re.compile(
    r"\b(?:moto|mock_aws|localstack|minio)\b|endpoint_url|127\.0\.0\.1|localhost",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AwsServiceIdentity:
    """The non-secret AWS metadata permitted in qualification evidence."""

    region: str
    provider: str
    service: str = "amazon-s3"


class QualificationUnavailableError(RuntimeError):
    """A standard external dependency was unavailable without exposing detail."""


def _git_revision() -> str:
    """Return the tested source revision without propagating command diagnostics."""
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
    return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else "unavailable"


def _qualification_source_revision() -> str | None:
    """Return a clean, immutable source revision eligible for live evidence."""
    revision = _git_revision()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
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
    return revision if not status.stdout.strip() else None


def _package_version(package: str) -> str:
    """Return an installed version without importing optional service clients."""
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:  # Package discovery is report-only and intentionally narrow.
        return "unavailable"


def _redacted_namespace(run_namespace: str) -> str:
    """Produce a stable evidence reference without naming cloud/database resources."""
    digest = hashlib.sha256(run_namespace.encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest}"


def _new_run_namespace() -> str:
    """Generate a bounded cryptographically random namespace for one test run."""
    return f"phase5-{secrets.token_hex(16)}"


def _missing_configuration(environment: Mapping[str, str]) -> list[str]:
    """Return names only; callers must never include the associated values."""
    return sorted(name for name in REQUIRED_CONFIGURATION if not environment.get(name))


def _supplied_secret_fragments(environment: Mapping[str, str]) -> tuple[str, ...]:
    """Return only secret-shaped values for a final serialized-evidence scan."""
    return tuple(
        value
        for name, value in environment.items()
        if value and len(value) >= 8 and _SECRET_ENVIRONMENT_NAME.search(name)
    )


def _validate_external_configuration(environment: Mapping[str, str]) -> None:
    """Validate only shape and key material in memory before a live run."""
    bucket = environment["CACHENESS_TEST_S3_BUCKET"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
        raise ValueError("invalid external configuration")

    try:
        manifest_key = base64.b64decode(
            environment["CACHENESS_TEST_MANIFEST_KEY_B64"], validate=True
        )
    except Exception as error:
        raise ValueError("invalid external configuration") from error
    if len(manifest_key) != 32:
        raise ValueError("invalid external configuration")
    if any(environment.get(name) for name in _DISALLOWED_S3_ENDPOINT_ENVIRONMENT):
        raise ValueError("invalid external configuration")


def resolve_aws_service_identity(environment: Mapping[str, str]) -> AwsServiceIdentity:
    """Resolve standard AWS credentials without serializing their values.

    A live S3 operation in the fixed suite remains the service-access proof.  This
    preflight only distinguishes absent standard-provider credentials from a
    runnable externally configured test environment.
    """
    try:
        import boto3
    except ImportError as error:
        raise QualificationUnavailableError("AWS SDK unavailable") from error

    region = (
        environment.get("CACHENESS_TEST_AWS_REGION")
        or environment.get("AWS_REGION")
        or environment.get("AWS_DEFAULT_REGION")
        or "unspecified"
    )
    try:
        session = boto3.Session(region_name=None if region == "unspecified" else region)
        credentials = session.get_credentials()
    except Exception as error:
        raise QualificationUnavailableError("AWS credentials unavailable") from error
    if credentials is None:
        raise QualificationUnavailableError("AWS credentials unavailable")
    return AwsServiceIdentity(region=region, provider="standard")


def make_evidence(
    *,
    status: str,
    missing_configuration: Sequence[str],
    run_namespace: str,
    aws_identity: AwsServiceIdentity | None,
    result: str,
    cleanup_status: str,
    revision: str | None = None,
) -> dict[str, object]:
    """Build the exact allow-listed evidence record before serializing it."""
    services: dict[str, object] = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "pytest": _package_version("pytest"),
        "boto3": _package_version("boto3"),
        "psycopg": _package_version("psycopg"),
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
        "revision": _git_revision() if revision is None else revision,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_namespace": _redacted_namespace(run_namespace),
        "missing_configuration": list(missing_configuration),
        "services": services,
        "result": result,
        "cleanup_status": cleanup_status,
    }


def _validate_safe_text(value: str) -> None:
    if _FORBIDDEN_VALUE_PATTERN.search(value):
        raise ValueError("evidence contains a secret, credential, or URL-like value")


def _is_standard_amazon_s3_identity(aws: object) -> bool:
    """Recognize the one non-secret service identity eligible for qualification."""
    return (
        isinstance(aws, Mapping)
        and aws.get("provider") == "standard"
        and aws.get("service") == "amazon-s3"
        and isinstance(aws.get("region"), str)
        and bool(aws["region"])
    )


def validate_evidence(evidence: Mapping[str, object]) -> None:
    """Fail closed unless evidence has the exact public release-evidence shape."""
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
    if not isinstance(evidence.get("revision"), str):
        raise ValueError("evidence revision is invalid")
    if not isinstance(evidence.get("generated_at_utc"), str):
        raise ValueError("evidence timestamp is invalid")
    run_namespace = evidence.get("run_namespace")
    if not isinstance(run_namespace, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{16}", run_namespace
    ):
        raise ValueError("evidence namespace is invalid")

    missing = evidence.get("missing_configuration")
    if not isinstance(missing, list) or any(
        not isinstance(name, str) or name not in REQUIRED_CONFIGURATION for name in missing
    ):
        raise ValueError("evidence missing configuration is invalid")

    services = evidence.get("services")
    if not isinstance(services, Mapping) or set(services) != _ALLOWED_SERVICE_KEYS:
        raise ValueError("evidence services violate the exact allow-list")
    for name in ("python", "pytest", "boto3", "psycopg"):
        if not isinstance(services.get(name), str):
            raise ValueError("evidence service version is invalid")
    aws = services.get("aws")
    if aws is not None:
        if not isinstance(aws, Mapping) or set(aws) != _ALLOWED_AWS_KEYS:
            raise ValueError("evidence AWS identity violates the exact allow-list")
        if not all(isinstance(value, str) for value in aws.values()):
            raise ValueError("evidence AWS identity is invalid")

    # Missing environment *names* are release-relevant and deliberately allowed;
    # all emitted values other than those fixed names must be safe text.
    for name in (
        "schema",
        "status",
        "revision",
        "generated_at_utc",
        "run_namespace",
        "result",
        "cleanup_status",
    ):
        _validate_safe_text(str(evidence[name]))
    for name in ("python", "pytest", "boto3", "psycopg"):
        _validate_safe_text(str(services[name]))
    if isinstance(aws, Mapping):
        for value in aws.values():
            _validate_safe_text(str(value))

    status = evidence["status"]
    result = evidence["result"]
    cleanup_status = evidence["cleanup_status"]
    complete_qualified_evidence = (
        result == "passed"
        and cleanup_status == "CLEAN"
        and not missing
        and _is_standard_amazon_s3_identity(aws)
    )
    if status == "QUALIFIED":
        if not complete_qualified_evidence or not re.fullmatch(
            r"[0-9a-f]{40}", evidence["revision"]
        ):
            raise ValueError("qualified evidence contradicts its required service proof")
    elif status == "UNAVAILABLE":
        if result != "not_run" or cleanup_status != "NOT_ATTEMPTED":
            raise ValueError("unavailable evidence contradicts its terminal state")
    elif complete_qualified_evidence:
        raise ValueError("complete qualification evidence must be marked qualified")


def write_evidence(
    output: Path,
    evidence: Mapping[str, object],
    *,
    forbidden_fragments: Sequence[str],
) -> None:
    """Serialize exact evidence only after shape and supplied-secret scans pass."""
    validate_evidence(evidence)
    serialized = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    for fragment in forbidden_fragments:
        if fragment and fragment in serialized:
            raise ValueError("evidence contains a supplied secret fragment")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")


def load_evidence(path: Path) -> dict[str, object]:
    """Load one evidence document for the runner contract tests."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("evidence root must be an object")
    return loaded


def _qualification_arguments() -> list[str]:
    """Return the fixed suite; callers cannot substitute mocks or a subset."""
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
    """Reject absent or emulator-oriented fixed modules before they can qualify."""
    for relative_path in LIVE_TEST_MODULES:
        path = REPOSITORY_ROOT / relative_path
        try:
            source = path.read_text(encoding="utf-8")
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


def _run_fixed_suite(
    arguments: Sequence[str], timeout: int, environment: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run the live suite without forwarding its potentially sensitive output."""
    if not _fixed_live_suite_is_real():
        return subprocess.CompletedProcess(list(arguments), 1, "", "")
    return subprocess.run(
        list(arguments),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=dict(environment),
    )


def _cleanup_from_fixtures(
    environment: Mapping[str, str], run_namespace: str
) -> str:
    """Load the exact-run cleanup implementation without making ``tests`` a package."""
    conftest_path = REPOSITORY_ROOT / "tests/qualification/conftest.py"
    if not conftest_path.is_file():
        return "ERROR"
    spec = importlib.util.spec_from_file_location("phase5_qualification_fixtures", conftest_path)
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
    """Reject skips, deselection, empty collection, and all non-passing outcomes."""
    output = f"{completed.stdout or ''}\n{completed.stderr or ''}".lower()
    incomplete_markers = (
        " skipped",
        " deselected",
        "no tests ran",
        "0 items collected",
        "collected 0 items",
        " xfailed",
        " xpassed",
        " interrupted",
        " error",
    )
    if completed.returncode != 0 or any(marker in output for marker in incomplete_markers):
        return False
    return re.search(r"\b[1-9][0-9]* passed\b", output) is not None


def _write_terminal_evidence(
    *,
    output: Path,
    status: str,
    missing_configuration: Sequence[str],
    run_namespace: str,
    aws_identity: AwsServiceIdentity | None,
    result: str,
    cleanup_status: str,
    forbidden_fragments: Sequence[str],
    revision: str | None = None,
) -> None:
    evidence = make_evidence(
        status=status,
        missing_configuration=missing_configuration,
        run_namespace=run_namespace,
        aws_identity=aws_identity,
        result=result,
        cleanup_status=cleanup_status,
        revision=revision,
    )
    write_evidence(output, evidence, forbidden_fragments=forbidden_fragments)


def run_qualification(
    *,
    output: Path,
    environment: Mapping[str, str] | None = None,
    run_tests: Callable[[Sequence[str], int], subprocess.CompletedProcess[str]] | None = None,
    resolve_aws: Callable[[Mapping[str, str]], AwsServiceIdentity] = resolve_aws_service_identity,
    cleanup: Callable[[Mapping[str, str], str], str] = _cleanup_from_fixtures,
    run_namespace: str | None = None,
) -> int:
    """Write one terminal evidence record and return its fixed process status."""
    supplied_environment = dict(os.environ if environment is None else environment)
    namespace = run_namespace or _new_run_namespace()
    forbidden_fragments = _supplied_secret_fragments(supplied_environment)
    missing = _missing_configuration(supplied_environment)
    if missing:
        _write_terminal_evidence(
            output=output,
            status="UNAVAILABLE",
            missing_configuration=missing,
            run_namespace=namespace,
            aws_identity=None,
            result="not_run",
            cleanup_status="NOT_ATTEMPTED",
            forbidden_fragments=forbidden_fragments,
        )
        return 2

    try:
        _validate_external_configuration(supplied_environment)
        timeout = _timeout_seconds(supplied_environment)
    except ValueError:
        _write_terminal_evidence(
            output=output,
            status="NOT_QUALIFIED",
            missing_configuration=[],
            run_namespace=namespace,
            aws_identity=None,
            result="configuration_error",
            cleanup_status="NOT_ATTEMPTED",
            forbidden_fragments=forbidden_fragments,
        )
        return 1

    source_revision = _qualification_source_revision()
    if source_revision is None:
        _write_terminal_evidence(
            output=output,
            status="NOT_QUALIFIED",
            missing_configuration=[],
            run_namespace=namespace,
            aws_identity=None,
            result="not_run",
            cleanup_status="NOT_ATTEMPTED",
            forbidden_fragments=forbidden_fragments,
        )
        return 1

    try:
        aws_identity = resolve_aws(supplied_environment)
    except QualificationUnavailableError:
        _write_terminal_evidence(
            output=output,
            status="UNAVAILABLE",
            missing_configuration=[],
            run_namespace=namespace,
            aws_identity=None,
            result="not_run",
            cleanup_status="NOT_ATTEMPTED",
            forbidden_fragments=forbidden_fragments,
        )
        return 2

    completed: subprocess.CompletedProcess[str] | None = None
    cleanup_status = "ERROR"
    child_environment = dict(supplied_environment)
    child_environment["CACHENESS_PHASE5_QUALIFICATION_RUN_ID"] = namespace
    try:
        if run_tests is None:
            completed = _run_fixed_suite(
                _qualification_arguments(), timeout, child_environment
            )
        else:
            completed = run_tests(_qualification_arguments(), timeout)
    except (OSError, subprocess.SubprocessError):
        completed = None
    finally:
        try:
            cleanup_status = cleanup(supplied_environment, namespace)
        except Exception:
            cleanup_status = "ERROR"

    is_complete_pass = completed is not None and _is_complete_pass(completed)
    source_is_stable = _qualification_source_revision() == source_revision
    qualified = is_complete_pass and cleanup_status == "CLEAN" and source_is_stable
    _write_terminal_evidence(
        output=output,
        status="QUALIFIED" if qualified else "NOT_QUALIFIED",
        missing_configuration=[],
        run_namespace=namespace,
        aws_identity=aws_identity,
        result=(
            "passed"
            if is_complete_pass and source_is_stable
            else "incomplete"
            if completed is None or not source_is_stable
            else "failed"
        ),
        cleanup_status=cleanup_status,
        forbidden_fragments=forbidden_fragments,
        revision=source_revision,
    )
    return 0 if qualified else 1


def main(arguments: Sequence[str] | None = None) -> int:
    """Parse the one optional output location without permitting test selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parsed = parser.parse_args(arguments)
    return run_qualification(output=parsed.output)


if __name__ == "__main__":
    raise SystemExit(main())
