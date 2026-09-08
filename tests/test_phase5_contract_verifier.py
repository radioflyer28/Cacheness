"""Contract tests for the immutable Phase 5 topology publication.

The support matrix is deliberately compared to the runtime catalog rather than
being manually copied into a release checklist.  It describes qualification
requirements only: the release-evidence artifact is the sole place an observed
live status may appear.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import subprocess

import pytest

from cacheness.storage.composition import qualified_topology_profiles


REPOSITORY_ROOT = Path(__file__).parents[1]
CATALOG_PATH = REPOSITORY_ROOT / "docs" / "CATALOG_AND_TOPOLOGY.md"
INITIALIZATION_PATH = REPOSITORY_ROOT / "docs" / "STORAGE_INITIALIZATION.md"
COVERAGE_PATH = REPOSITORY_ROOT / (
    ".planning/phases/05-payload-backends-and-supported-topology-qualification/"
    "05-COVERAGE.md"
)

TOPOLOGY_START = "<!-- phase5-topology-matrix:start -->"
TOPOLOGY_END = "<!-- phase5-topology-matrix:end -->"
COVERAGE_START = "<!-- phase5-api-coverage:start -->"
COVERAGE_END = "<!-- phase5-api-coverage:end -->"
INITIALIZATION_START = "<!-- phase5-initialization-contract:start -->"
INITIALIZATION_END = "<!-- phase5-initialization-contract:end -->"

TOPOLOGY_HEADERS = (
    "profile",
    "authority",
    "payload",
    "coordination",
    "durability / atomicity boundary",
    "progress outcomes",
    "required service configuration",
    "derived JSON projection",
    "evidence requirement ID",
    "evidence schema ID",
)
COVERAGE_HEADERS = ("role", "capability", "decision", "reason")

EXPECTED_COVERAGE_CAPABILITIES = frozenset(
    {
        "Standard boto3 credential-provider chain",
        "Injected boto3 S3 client",
        "Region plus ExpectedBucketOwner",
        'PutObject + IfNoneMatch="*"',
        "CreateMultipartUpload",
        "UploadPart",
        'CompleteMultipartUpload + IfNoneMatch="*"',
        "AbortMultipartUpload",
        "ListMultipartUploads",
        "HeadObject",
        "GetObject / StreamingBody",
        "DeleteObject",
        "ListObjectsV2 / continuation token",
        "SDK transport checksums",
        "ETag response metadata",
        "Botocore modeled service/credential/timeout errors",
        "Bucket versioning",
        "S3 Object Lock/WORM",
        "Cross-region replication",
        "Bucket creation/deletion or policy management",
        "Bucket lifecycle-rule management",
        "Event notifications/SQS/SNS/Lambda",
        "Presigned URLs",
        "CopyObject / mutable stable-key replacement",
        "S3 Select, inventory reports, batch operations",
        "Custom S3-compatible endpoint support",
        "External DSN / injected connection factory or pool",
        "Connection.transaction()",
        "Bound SQL values",
        "psycopg.sql.Identifier",
        "INSERT ... ON CONFLICT",
        "Conditional UPDATE ... RETURNING",
        "Keyset queries with LIMIT",
        "Transaction-local lock_timeout",
        "Transaction-local statement_timeout",
        "SQLSTATE exception classes (40001, 40P01, 55P03, 57014)",
        "Connection operational/timeout errors",
        "Server/version identity query",
        "Explicit DDL in initialize()",
        "Advisory locks",
        "Two-phase commit/prepared transactions",
        "LISTEN/NOTIFY",
        "Async psycopg API",
        "Logical/physical replication APIs",
        "SQLAlchemy ORM lifecycle layer",
        "Implicit schema migration on connect/open",
        "Server-side retry-until-success procedure",
        "Cross-resource foreign-data wrapper or filesystem authority",
    }
)


def _load_verifier():
    """Load the standalone verifier without making ``tools`` a package."""
    verifier_path = REPOSITORY_ROOT / "tools" / "verify_phase5_contracts.py"
    spec = spec_from_file_location("phase5_contract_verifier", verifier_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _marker_text(path: Path, start: str, end: str) -> str:
    """Return one exact marker-bounded documentation region."""
    text = path.read_text(encoding="utf-8")
    assert text.count(start) == 1
    assert text.count(end) == 1
    start_index = text.index(start) + len(start)
    end_index = text.index(end)
    assert start_index < end_index
    return text[start_index:end_index]


def _pipe_rows(text: str, headers: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    """Parse one simple exact Markdown table without accepting prose rows."""
    rows = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    assert len(rows) >= 3
    parsed = [tuple(part.strip() for part in line.strip("|").split("|")) for line in rows]
    assert parsed[0] == headers
    assert all(set(cell) <= {"-", ":", " "} for cell in parsed[1])
    assert all(len(row) == len(headers) for row in parsed[2:])
    return tuple(dict(zip(headers, row, strict=True)) for row in parsed[2:])


def _runtime_profile_rows() -> dict[tuple[str, str], dict[str, str]]:
    """Normalize immutable runtime records to their documented contract fields."""
    return {
        pair: {
            "authority": profile.authority_identity,
            "payload": profile.payload_identity,
            "coordination": profile.requirements.coordination_scope,
            "durability / atomicity boundary": (
                profile.requirements.durability_atomicity_boundary
            ),
            "progress outcomes": ", ".join(
                sorted(profile.requirements.progress_outcomes)
            ),
            "required service configuration": "; ".join(
                profile.requirements.service_prerequisites
            ),
            "derived JSON projection": str(profile.requirements.projection_available).lower(),
            "evidence requirement ID": profile.requirements.evidence_requirement_id,
            "evidence schema ID": profile.requirements.evidence_schema_id,
        }
        for pair, profile in qualified_topology_profiles().items()
    }


def test_published_topology_matrix_matches_immutable_runtime_profiles() -> None:
    """The marker-bounded table documents exactly the three runtime pairings."""
    rows = _pipe_rows(
        _marker_text(CATALOG_PATH, TOPOLOGY_START, TOPOLOGY_END), TOPOLOGY_HEADERS
    )
    runtime_rows = _runtime_profile_rows()

    assert len(rows) == 3
    documented_pairs = {(row["authority"], row["payload"]) for row in rows}
    assert documented_pairs == set(runtime_rows) == {
        ("memory", "memory"),
        ("sqlite", "filesystem"),
        ("postgresql", "s3"),
    }
    for row in rows:
        expected = runtime_rows[(row["authority"], row["payload"])]
        assert {field: row[field] for field in expected} == expected


def test_documentation_declares_requirements_not_an_observed_remote_status() -> None:
    """Docs cannot mirror mutable live evidence or turn local green into BACK-05."""
    for path in (CATALOG_PATH, INITIALIZATION_PATH, COVERAGE_PATH):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\b(?:QUALIFIED|UNAVAILABLE|NOT_QUALIFIED)\b", text)
        assert "latest observed status" not in text.lower()

    initialization = _marker_text(
        INITIALIZATION_PATH, INITIALIZATION_START, INITIALIZATION_END
    )
    assert "explicit PostgreSQL initialization before shared workers" in initialization
    assert "read-only version validation" in initialization
    assert re.search(r"stopped-worker\s+Phase 7 migration", initialization)
    assert "Amazon S3" in initialization
    assert "shared external manifest signing key" in initialization


def test_api_coverage_is_complete_and_every_opt_out_is_reasoned() -> None:
    """Relevant S3 and psycopg surfaces are either integrated or explicit opt-outs."""
    rows = _pipe_rows(
        _marker_text(COVERAGE_PATH, COVERAGE_START, COVERAGE_END), COVERAGE_HEADERS
    )
    assert {row["capability"] for row in rows} == EXPECTED_COVERAGE_CAPABILITIES
    assert {row["role"] for row in rows} == {"Amazon S3 / boto3", "PostgreSQL / psycopg"}
    assert {row["decision"] for row in rows} <= {"INTEGRATE", "OPT-OUT"}
    assert all(row["reason"] for row in rows)
    assert all(
        row["reason"]
        for row in rows
        if row["decision"] == "OPT-OUT"
    )


@pytest.mark.parametrize(
    ("path", "start", "end"),
    [
        (CATALOG_PATH, TOPOLOGY_START, TOPOLOGY_END),
        (COVERAGE_PATH, COVERAGE_START, COVERAGE_END),
        (INITIALIZATION_PATH, INITIALIZATION_START, INITIALIZATION_END),
    ],
)
def test_contract_markers_are_unique(path: Path, start: str, end: str) -> None:
    """One fixed source region makes documentation drift deterministic."""
    _marker_text(path, start, end)


@pytest.mark.parametrize(
    ("source", "filename", "expected"),
    [
        (
            "class AnotherLifecycleEngine:\n    pass\n",
            "src/cacheness/storage/parallel.py",
            ("second lifecycle engine: AnotherLifecycleEngine",),
        ),
        (
            "class MutationCoordinator:\n    pass\n",
            "src/cacheness/storage/coordinator.py",
            ("second lifecycle coordinator: MutationCoordinator",),
        ),
        (
            "def publish(authority):\n    authority.promote_mutation(None)\n",
            "src/cacheness/storage/backends/s3_backend.py",
            ("payload adapter invokes authority transition: promote_mutation",),
        ),
        (
            "def visible(client):\n    if client.head_object(Bucket='x', Key='y'):\n        return True\n",
            "src/cacheness/storage/backends/s3_backend.py",
            ("S3 observation used as visibility authority",),
        ),
        (
            "cursor.execute('SELECT pg_advisory_lock(1)')\n",
            "src/cacheness/storage/backends/postgresql_lifecycle_authority.py",
            ("PostgreSQL advisory lock call",),
        ),
        (
            "client.list_objects_v2(Bucket='bucket', Prefix='prefix')\n",
            "src/cacheness/storage/backends/s3_backend.py",
            ("unbounded S3 listing: list_objects_v2",),
        ),
        (
            "if response['ETag'] == manifest.digest:\n    return True\n",
            "src/cacheness/storage/backends/s3_backend.py",
            ("ETag used as integrity authority",),
        ),
        (
            "MANIFEST_SECRET = 'do-not-commit'\n",
            "src/cacheness/storage/backends/s3_backend.py",
            ("inline secret assignment: MANIFEST_SECRET",),
        ),
    ],
)
def test_architecture_audit_rejects_prohibited_executable_forms(
    source: str, filename: str, expected: tuple[str, ...]
) -> None:
    """AST rules catch executable architecture drift, not just documentation text."""
    verifier = _load_verifier()
    assert verifier.audit_source(source, filename) == expected


@pytest.mark.parametrize(
    "source",
    [
        "# class AnotherLifecycleEngine: pass\n",
        "text = 'pg_advisory_lock and list_objects_v2 without MaxKeys'\n",
        "doc = 'ETag used as integrity authority'\n",
        "client.list_objects_v2(Bucket='bucket', Prefix='prefix', MaxKeys=1)\n",
        "def publish():\n    return client.head_object(Bucket='x', Key='y')\n",
    ],
)
def test_architecture_audit_ignores_comments_strings_and_bounded_nonvisibility_code(
    source: str,
) -> None:
    """Source rules must not replace semantic checks with broad text greps."""
    verifier = _load_verifier()
    assert verifier.audit_source(source, "fixture.py") == ()


def test_local_verifier_inventory_and_evidence_boundary_are_fixed(tmp_path: Path) -> None:
    """Local verification reports evidence but cannot write or upgrade it."""
    verifier = _load_verifier()

    assert set(verifier.CONTRACT_TEST_MODULES) == {
        "tests/test_supported_topologies.py",
        "tests/test_topology_capabilities.py",
        "tests/contracts/test_payload_generation_io.py",
        "tests/test_s3_blob_backend.py",
        "tests/contracts/test_s3_generation_io.py",
        "tests/contracts/test_postgresql_lifecycle_authority.py",
        "tests/contracts/test_lifecycle_authority.py",
        "tests/test_lifecycle_authority_contract.py",
        "tests/test_payload_faults.py",
        "tests/contracts/test_topology_lifecycle.py",
        "tests/test_blob_store_composition.py",
        "tests/qualification/test_live_evidence.py",
        "tests/test_phase5_contract_verifier.py",
    }
    assert set(verifier.LIVE_COLLECTION_MODULES) == {
        "tests/integration/test_postgresql_authority.py",
        "tests/integration/test_s3_generation.py",
        "tests/integration/test_remote_topology.py",
    }
    assert verifier.read_live_evidence_status(tmp_path / "missing.json") == "absent"

    unavailable = tmp_path / "evidence.json"
    unavailable.write_text('{"status": "UNAVAILABLE"}', encoding="utf-8")
    assert verifier.read_live_evidence_status(unavailable) == "invalid"
    assert unavailable.read_text(encoding="utf-8") == '{"status": "UNAVAILABLE"}'


def test_local_pytest_subprocess_has_a_fixed_timeout(monkeypatch) -> None:
    """The verification gate cannot wait forever on a stuck child process."""
    verifier = _load_verifier()
    observed: dict[str, object] = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="1 passed")

    monkeypatch.setattr(verifier.subprocess, "run", run)

    ok, diagnostic = verifier._run_pytest(("tests/example.py",), label="fixed suite")

    assert ok is True
    assert diagnostic == "fixed suite:\n1 passed"
    assert observed["timeout"] == verifier.PYTEST_TIMEOUT_SECONDS == 300


def test_local_pytest_timeout_fails_closed_with_diagnostic(monkeypatch) -> None:
    """A stuck child becomes a bounded failed result, not an uncaught exception."""
    verifier = _load_verifier()

    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(
            command,
            kwargs["timeout"],
            output="partial output",
        )

    monkeypatch.setattr(verifier.subprocess, "run", run)

    ok, diagnostic = verifier._run_pytest(("tests/example.py",), label="fixed suite")

    assert ok is False
    assert diagnostic == "fixed suite: timed out after 300 seconds\npartial output"
