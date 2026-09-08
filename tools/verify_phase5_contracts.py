#!/usr/bin/env python3
"""Verify the immutable Phase 5 topology contract without claiming live evidence.

This is a deliberately fixed, local architecture gate.  It checks only the
Phase 5 sources and test inventory named below; it neither scans historical
race-fix artifacts nor writes release evidence.  A green local run proves the
contract, integrity/recovery boundaries, and progress declarations.  Real
PostgreSQL/Amazon S3 qualification remains an observation in the separately
sanitized evidence artifact.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Sequence
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PHASE_DIRECTORY = REPOSITORY_ROOT / (
    ".planning/phases/05-payload-backends-and-supported-topology-qualification"
)
CATALOG_PATH = REPOSITORY_ROOT / "docs/CATALOG_AND_TOPOLOGY.md"
INITIALIZATION_PATH = REPOSITORY_ROOT / "docs/STORAGE_INITIALIZATION.md"
COVERAGE_PATH = PHASE_DIRECTORY / "05-COVERAGE.md"
DEFAULT_EVIDENCE_PATH = PHASE_DIRECTORY / "05-LIVE-QUALIFICATION.json"

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
EXPECTED_PAIRS = frozenset(
    {("memory", "memory"), ("sqlite", "filesystem"), ("postgresql", "s3")}
)

# These are fixed Phase 5 contract suites from Plans 01-08.  The real-service
# modules are collected separately below, because this local command must not
# turn a skipped or absent external service into an executable substitute.
CONTRACT_TEST_MODULES = (
    "tests/test_supported_topologies.py",
    "tests/contracts/test_payload_generation_io.py",
    "tests/contracts/test_s3_generation_io.py",
    "tests/contracts/test_postgresql_lifecycle_authority.py",
    "tests/contracts/test_lifecycle_authority.py",
    "tests/test_lifecycle_authority_contract.py",
    "tests/test_payload_faults.py",
    "tests/contracts/test_topology_lifecycle.py",
    "tests/test_blob_store_composition.py",
    "tests/qualification/test_live_evidence.py",
    "tests/test_phase5_contract_verifier.py",
)
LIVE_COLLECTION_MODULES = (
    "tests/integration/test_postgresql_authority.py",
    "tests/integration/test_s3_generation.py",
    "tests/integration/test_remote_topology.py",
)
ARCHITECTURE_SOURCE_MODULES = (
    "src/cacheness/storage/lifecycle.py",
    "src/cacheness/storage/blob_store.py",
    "src/cacheness/storage/composition.py",
    "src/cacheness/storage/backends/s3_backend.py",
    "src/cacheness/storage/backends/postgresql_lifecycle_authority.py",
)

_AUTHORITY_TRANSITIONS = frozenset(
    {
        "prepare_mutation",
        "record_verification",
        "promote_mutation",
        "abort_mutation",
        "retire_cleanup_debt",
        "begin_clear",
        "complete_clear",
    }
)
_S3_OBSERVATION_CALLS = frozenset(
    {"head_object", "get_object", "list_objects_v2", "list_multipart_uploads"}
)
_S3_LISTING_BOUNDS = {
    "list_objects_v2": "MaxKeys",
    "list_multipart_uploads": "MaxUploads",
}
_INLINE_SECRET_NAME = re.compile(
    r"(?:secret|password|token|access[_-]?key|private[_-]?key)", re.IGNORECASE
)


def _read_marker_text(path: Path, start: str, end: str) -> str:
    """Read exactly one bounded contract region from an owned Markdown file."""
    text = path.read_text(encoding="utf-8")
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"{path.relative_to(REPOSITORY_ROOT)} has invalid contract markers")
    beginning = text.index(start) + len(start)
    finish = text.index(end)
    if beginning >= finish:
        raise ValueError(f"{path.relative_to(REPOSITORY_ROOT)} has inverted contract markers")
    return text[beginning:finish]


def _parse_table(text: str, headers: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    """Parse one marker-bounded simple Markdown table with exact headers."""
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        raise ValueError("contract region must contain a header, delimiter, and one row")
    rows = [tuple(part.strip() for part in line.strip("|").split("|")) for line in lines]
    if rows[0] != headers:
        raise ValueError(f"unexpected table headers: {rows[0]!r}")
    if any(set(cell) - {"-", ":", " "} for cell in rows[1]):
        raise ValueError("contract table has an invalid delimiter row")
    if any(len(row) != len(headers) for row in rows[2:]):
        raise ValueError("contract table has an uneven row")
    return tuple(dict(zip(headers, row, strict=True)) for row in rows[2:])


def _runtime_topology_rows() -> dict[tuple[str, str], dict[str, str]]:
    """Normalize runtime declarations to exactly the published table columns."""
    from cacheness.storage.composition import qualified_topology_profiles

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


def verify_document_contract() -> tuple[str, ...]:
    """Return contract-document errors without reading any mutable service result."""
    errors: list[str] = []
    try:
        topology_rows = _parse_table(
            _read_marker_text(CATALOG_PATH, TOPOLOGY_START, TOPOLOGY_END),
            TOPOLOGY_HEADERS,
        )
        runtime_rows = _runtime_topology_rows()
        pairs = {(row["authority"], row["payload"]) for row in topology_rows}
        if len(topology_rows) != 3 or pairs != EXPECTED_PAIRS or pairs != set(runtime_rows):
            errors.append("topology table is not the exact three-profile runtime catalog")
        for row in topology_rows:
            pair = (row["authority"], row["payload"])
            expected = runtime_rows.get(pair)
            if expected is None or {field: row[field] for field in expected} != expected:
                errors.append(f"topology row differs from runtime profile: {pair!r}")

        coverage_rows = _parse_table(
            _read_marker_text(COVERAGE_PATH, COVERAGE_START, COVERAGE_END),
            COVERAGE_HEADERS,
        )
        if not coverage_rows:
            errors.append("API coverage table is empty")
        for row in coverage_rows:
            if row["decision"] not in {"INTEGRATE", "OPT-OUT"} or not row["reason"]:
                errors.append(f"unreasoned API coverage row: {row['capability']}")

        initialization = _read_marker_text(
            INITIALIZATION_PATH, INITIALIZATION_START, INITIALIZATION_END
        )
        for required in (
            "explicit PostgreSQL initialization before shared workers",
            "read-only version validation",
            "Phase 7 migration",
            "Amazon S3",
            "shared external manifest signing key",
        ):
            if required not in initialization:
                errors.append(f"initialization contract omits: {required}")

        for path in (CATALOG_PATH, INITIALIZATION_PATH, COVERAGE_PATH):
            text = path.read_text(encoding="utf-8")
            if re.search(r"\b(?:QUALIFIED|UNAVAILABLE|NOT_QUALIFIED)\b", text):
                errors.append(
                    f"mutable service status appears outside release evidence: "
                    f"{path.relative_to(REPOSITORY_ROOT)}"
                )
    except (OSError, ValueError, KeyError) as error:
        errors.append(f"document contract unreadable: {error}")
    return tuple(errors)


def _call_name(node: ast.Call) -> str | None:
    """Return an attribute call's final method name without resolving bindings."""
    return node.func.attr if isinstance(node.func, ast.Attribute) else None


def _contains_s3_observation(node: ast.AST) -> bool:
    """Check an executable condition for a direct S3 observation call."""
    return any(
        isinstance(candidate, ast.Call) and _call_name(candidate) in _S3_OBSERVATION_CALLS
        for candidate in ast.walk(node)
    )


def _contains_etag_subscript(node: ast.AST) -> bool:
    """Recognize a real dictionary/key expression, not comment or string text."""
    return any(
        isinstance(candidate, ast.Subscript)
        and isinstance(candidate.slice, ast.Constant)
        and candidate.slice.value == "ETag"
        for candidate in ast.walk(node)
    )


class _ArchitectureVisitor(ast.NodeVisitor):
    """AST-only guardrails for the Phase 5 one-engine architecture."""

    def __init__(self, filename: str) -> None:
        self.filename = PurePosixPath(filename).as_posix()
        self.findings: list[str] = []
        self._function_names: list[str] = []
        self._bounded_request_names: list[set[str]] = [set()]

    @property
    def _is_payload_adapter(self) -> bool:
        return self.filename.endswith("/backends/s3_backend.py")

    def _add(self, finding: str) -> None:
        if finding not in self.findings:
            self.findings.append(finding)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name.endswith("LifecycleEngine") and node.name != "AuthorityLifecycleEngine":
            self._add(f"second lifecycle engine: {node.name}")
        if node.name.endswith("Coordinator"):
            self._add(f"second lifecycle coordinator: {node.name}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_names.append(node.name.lower())
        self._bounded_request_names.append(set())
        self.generic_visit(node)
        self._bounded_request_names.pop()
        self._function_names.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if isinstance(node.value, ast.Call) and _call_name(node.value) == "_request_kwargs":
            declared = {keyword.arg for keyword in node.value.keywords if keyword.arg}
            if declared & {"MaxKeys", "MaxUploads"}:
                self._bounded_request_names[-1].update(names)
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (str, bytes)):
            for name in names:
                if _INLINE_SECRET_NAME.search(name):
                    self._add(f"inline secret assignment: {name}")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, (str, bytes))
            and _INLINE_SECRET_NAME.search(node.target.id)
        ):
            self._add(f"inline secret assignment: {node.target.id}")
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if self._is_payload_adapter and _contains_s3_observation(node.test):
            if any(
                token in function_name
                for function_name in self._function_names
                for token in ("visible", "visibility", "promote", "authorize", "catalog")
            ):
                self._add("S3 observation used as visibility authority")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if self._is_payload_adapter and _contains_etag_subscript(node):
            self._add("ETag used as integrity authority")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        if self._is_payload_adapter and name in _AUTHORITY_TRANSITIONS:
            self._add(f"payload adapter invokes authority transition: {name}")
        if self._is_payload_adapter and name in _S3_LISTING_BOUNDS:
            required_keyword = _S3_LISTING_BOUNDS[name]
            has_direct_bound = any(
                keyword.arg == required_keyword for keyword in node.keywords
            )
            has_bounded_request = any(
                keyword.arg is None
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id in self._bounded_request_names[-1]
                for keyword in node.keywords
            )
            if not has_direct_bound and not has_bounded_request:
                self._add(f"unbounded S3 listing: {name}")
        if name == "execute":
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if "pg_advisory_lock" in argument.value.lower():
                        self._add("PostgreSQL advisory lock call")
        self.generic_visit(node)


def audit_source(source: str, filename: str) -> tuple[str, ...]:
    """Return deterministic Phase 5 architecture findings for executable AST nodes."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        return (f"invalid Python source: {error.msg}",)
    visitor = _ArchitectureVisitor(filename)
    visitor.visit(tree)
    return tuple(visitor.findings)


def run_architecture_audit() -> tuple[str, ...]:
    """Audit the fixed production inventory, never historical or untracked files."""
    findings: list[str] = []
    for relative_path in ARCHITECTURE_SOURCE_MODULES:
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            findings.append(f"missing architecture source: {relative_path}")
            continue
        for finding in audit_source(path.read_text(encoding="utf-8"), relative_path):
            findings.append(f"{relative_path}: {finding}")

    blob_store_path = REPOSITORY_ROOT / "src/cacheness/storage/blob_store.py"
    blob_store_source = blob_store_path.read_text(encoding="utf-8")
    if "self.lifecycle = AuthorityLifecycleEngine(self, self.lifecycle_authority)" not in blob_store_source:
        findings.append("BlobStore no longer materializes AuthorityLifecycleEngine")
    return tuple(findings)


def _load_qualification_runner() -> Any:
    """Load the evidence schema validator without importing ``tools`` as a package."""
    runner_path = REPOSITORY_ROOT / "tools/run_phase5_qualification.py"
    spec = spec_from_file_location("phase5_qualification_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Phase 5 qualification runner")
    module = module_from_spec(spec)
    # Dataclass annotations resolve through ``sys.modules`` while the runner
    # is executing, so mirror normal import machinery before loading it.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def read_live_evidence_status(path: Path) -> str:
    """Read a sanitized observation without writing, repairing, or upgrading it."""
    if not path.exists():
        return "absent"
    try:
        runner = _load_qualification_runner()
        evidence = runner.load_evidence(path)
        runner.validate_evidence(evidence)
        status = evidence.get("status")
        return status if isinstance(status, str) else "invalid"
    except (OSError, RuntimeError, ValueError, TypeError):
        return "invalid"


def _run_pytest(arguments: Sequence[str], *, label: str) -> tuple[bool, str]:
    """Run one fixed local test command and return a compact diagnostic."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    summary = "\n".join(completed.stdout.splitlines()[-12:])
    return completed.returncode == 0, f"{label}:\n{summary}"


def run_local_contract_suite() -> tuple[bool, str]:
    """Run the precise local suite and only collect externally configured tests."""
    missing = [
        path for path in (*CONTRACT_TEST_MODULES, *LIVE_COLLECTION_MODULES)
        if not (REPOSITORY_ROOT / path).is_file()
    ]
    if missing:
        return False, f"missing fixed Phase 5 test modules: {', '.join(missing)}"
    contracts_ok, contracts_output = _run_pytest(
        ("-q", *CONTRACT_TEST_MODULES, "-x", "-o", "log_cli=false"),
        label="integrity/recovery/progress local contracts",
    )
    collection_ok, collection_output = _run_pytest(
        (
            "--collect-only",
            "-q",
            *LIVE_COLLECTION_MODULES,
            "-m",
            "live_postgresql or live_aws_s3 or live_remote",
            "-o",
            "log_cli=false",
        ),
        label="remote qualification suite collection",
    )
    return contracts_ok and collection_ok, f"{contracts_output}\n{collection_output}"


def _print_findings(title: str, findings: Iterable[str]) -> bool:
    """Render one guarantee-class result and return whether it is clean."""
    findings = tuple(findings)
    if not findings:
        print(f"{title}: PASS")
        return True
    print(f"{title}: FAIL")
    for finding in findings:
        print(f"  - {finding}")
    return False


def main(arguments: Sequence[str] | None = None) -> int:
    """Run fixed local checks and report, but never change, service evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE_PATH,
        help="Sanitized live evidence to report read-only (default: Phase 5 artifact)",
    )
    parsed = parser.parse_args(arguments)

    documents_ok = _print_findings("Topology declaration contract", verify_document_contract())
    architecture_ok = _print_findings("One-engine architecture contract", run_architecture_audit())
    tests_ok, test_output = run_local_contract_suite()
    print("Local integrity/recovery/progress contracts: " + ("PASS" if tests_ok else "FAIL"))
    if not tests_ok:
        print(test_output)
    print("Performance boundary: PASS (no performance threshold is evaluated here; Phase 8 owns budgets)")
    print(f"Live qualification evidence (read-only): {read_live_evidence_status(parsed.evidence)}")
    return 0 if documents_ok and architecture_ok and tests_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
