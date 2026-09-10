#!/usr/bin/env python3
"""Verify Phase 7's fixed offline migration and rebuild contract.

This verifier intentionally uses a literal reviewed inventory.  It neither
consults a Git diff nor discovers tests, services, candidates, or maintenance
runs.  A green result is deterministic local evidence only: live PostgreSQL,
AWS S3, Windows, Python-version qualification, and performance distributions
remain explicit Phase 8 work.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Mapping, Sequence
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTEST_TIMEOUT_SECONDS = 300

# Keep reviewed and executed inventories distinct.  The comparison below makes
# an accidental edit to the execution tuple observable rather than letting the
# mutable tuple become its own oracle.
_PHASE7_REVIEWED_PRODUCTION_PATHS = (
    "src/cacheness/core.py",
    "src/cacheness/error_handling.py",
    "src/cacheness/handlers.py",
    "src/cacheness/interfaces.py",
    "src/cacheness/storage/__init__.py",
    "src/cacheness/storage/blob_store.py",
    "src/cacheness/storage/memory_lifecycle_authority.py",
    "src/cacheness/storage/migration.py",
    "src/cacheness/storage/migration_authority.py",
    "src/cacheness/storage/migration_evidence.py",
    "src/cacheness/storage/projections.py",
    "src/cacheness/storage/sqlite_lifecycle_authority.py",
    "src/cacheness/storage/backends/postgresql_lifecycle_authority.py",
    "src/cacheness/storage/backends/s3_backend.py",
    "docs/STORAGE_MIGRATION.md",
    "docs/STORAGE_INITIALIZATION.md",
    "docs/BACKEND_SELECTION.md",
    ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-COVERAGE.md",
)
PHASE7_PRODUCTION_PATHS = tuple(_PHASE7_REVIEWED_PRODUCTION_PATHS)

_PHASE7_REVIEWED_TEST_NODES = (
    "tests/test_migration_cutover.py",
    "tests/test_stored_compatibility.py",
    "tests/test_migration_plan_contract.py",
    "tests/test_migration_inspection.py",
    "tests/test_lifecycle_authority_contract.py",
    "tests/contracts/test_postgresql_lifecycle_authority.py",
    "tests/test_migration_run_evidence.py",
    "tests/test_projection_sql_atomicity.py",
    "tests/test_migration_remote_contract.py",
    "tests/test_s3_blob_backend.py",
    "tests/test_rebuild_workflow.py",
    "tests/test_handler_registration.py",
    "tests/test_blob_store_read_contract.py",
    "tests/test_migration_public_contract.py",
    "tests/test_phase7_contract_verifier.py",
)
PHASE7_TEST_NODES = tuple(_PHASE7_REVIEWED_TEST_NODES)
# The base runtime intentionally does not install optional PostgreSQL drivers.
# Quick feedback therefore exercises every deterministic local module it can
# import and reports this one fixed module as requiring ``--all`` in the locked
# all-extras environment.  It is neither skipped as a pass nor qualification.
PHASE7_QUICK_TEST_NODES = (
    "tests/test_migration_cutover.py",
    "tests/test_stored_compatibility.py",
    "tests/test_migration_plan_contract.py",
    "tests/test_migration_inspection.py",
    "tests/test_lifecycle_authority_contract.py",
    "tests/test_migration_run_evidence.py",
    "tests/test_projection_sql_atomicity.py",
    "tests/test_migration_remote_contract.py",
    "tests/test_s3_blob_backend.py",
    "tests/test_rebuild_workflow.py",
    "tests/test_handler_registration.py",
    "tests/test_blob_store_read_contract.py",
    "tests/test_migration_public_contract.py",
    "tests/test_phase7_contract_verifier.py",
)
PHASE8_LIVE_UNQUALIFIED_NODES = (
    "tests/integration/test_postgresql_authority.py",
    "tests/integration/test_remote_topology.py",
    "tests/integration/test_s3_generation.py",
)

PHASE7_PLAN_PATHS = (
    ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-01-PLAN.md",
    ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-02-PLAN.md",
    ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-03-PLAN.md",
    ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-04-PLAN.md",
    ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-05-PLAN.md",
    ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-06-PLAN.md",
    ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-07-PLAN.md",
    ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-08-PLAN.md",
    ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-09-PLAN.md",
    ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-10-PLAN.md",
    ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-11-PLAN.md",
)
PHASE7_CONTEXT_PATH = (
    ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-CONTEXT.md"
)

MIGRATION_REQUIREMENT_NODES = {
    "MIGR-03": (
        "tests/test_migration_inspection.py",
        "tests/test_migration_plan_contract.py",
        "tests/test_migration_public_contract.py",
    ),
    "MIGR-04": (
        "tests/test_migration_cutover.py",
        "tests/test_stored_compatibility.py",
        "tests/test_lifecycle_authority_contract.py",
    ),
    "MIGR-05": (
        "tests/test_migration_run_evidence.py",
        "tests/test_migration_cutover.py",
        "tests/test_projection_sql_atomicity.py",
        "tests/test_migration_remote_contract.py",
    ),
    "MIGR-06": (
        "tests/test_rebuild_workflow.py",
        "tests/test_handler_registration.py",
        "tests/test_blob_store_read_contract.py",
    ),
}

DECISION_NODES = {
    "D-01": ("tests/test_stored_compatibility.py",),
    "D-02": ("tests/test_migration_plan_contract.py",),
    "D-03": ("tests/test_migration_plan_contract.py",),
    "D-04": ("tests/test_migration_inspection.py",),
    "D-05": ("tests/test_stored_compatibility.py",),
    "D-06": ("tests/test_migration_inspection.py",),
    "D-07": ("tests/test_migration_cutover.py",),
    "D-08": ("tests/test_rebuild_workflow.py",),
    "D-09": ("tests/test_migration_run_evidence.py",),
    "D-10": ("tests/test_handler_registration.py",),
    "D-11": ("tests/test_projection_sql_atomicity.py",),
    "D-12": ("tests/test_migration_cutover.py",),
    "D-13": ("tests/test_migration_cutover.py",),
    "D-14": ("tests/test_migration_cutover.py",),
    "D-15": ("tests/test_migration_cutover.py",),
    "D-16": ("tests/test_migration_cutover.py",),
    "D-17": ("tests/test_migration_run_evidence.py",),
    "D-18": ("tests/test_migration_plan_contract.py",),
    "D-19": ("tests/test_migration_run_evidence.py",),
    "D-20": ("tests/test_migration_run_evidence.py",),
    "D-21": ("tests/test_migration_cutover.py",),
    "D-22": ("tests/test_migration_run_evidence.py",),
}

_DECLARED_PHASE7_THREAT_IDS = (
    *(f"T-07-{number:02d}" for number in range(1, 25)),
    *(f"T-07-{number:02d}" for number in range(26, 30)),
    *(f"T-07-{number:02d}" for number in range(31, 35)),
    *(f"T-07-{number:02d}" for number in range(36, 41)),
    *(f"T-07-{number:02d}" for number in range(42, 51)),
)
SECURITY_THREAT_NODES = {
    "T-07-01": ("tests/test_migration_cutover.py",),
    "T-07-02": ("tests/test_stored_compatibility.py",),
    "T-07-03": ("tests/test_blob_store_read_contract.py",),
    "T-07-04": ("tests/test_migration_inspection.py",),
    "T-07-05": ("tests/test_migration_run_evidence.py",),
    "T-07-06": ("tests/test_migration_plan_contract.py",),
    "T-07-07": ("tests/test_migration_plan_contract.py",),
    "T-07-08": ("tests/test_migration_plan_contract.py",),
    "T-07-09": ("tests/test_migration_plan_contract.py",),
    "T-07-10": ("tests/test_migration_plan_contract.py",),
    "T-07-11": ("tests/test_migration_run_evidence.py",),
    "T-07-12": ("tests/test_migration_inspection.py",),
    "T-07-13": ("tests/test_migration_inspection.py",),
    "T-07-14": ("tests/test_migration_inspection.py",),
    "T-07-15": ("tests/test_migration_inspection.py",),
    "T-07-16": ("tests/test_migration_run_evidence.py",),
    "T-07-17": ("tests/test_migration_run_evidence.py",),
    "T-07-18": ("tests/test_migration_run_evidence.py",),
    "T-07-19": ("tests/test_migration_run_evidence.py",),
    "T-07-20": ("tests/test_migration_run_evidence.py",),
    "T-07-21": ("tests/test_migration_cutover.py",),
    "T-07-22": ("tests/test_migration_cutover.py",),
    "T-07-23": ("tests/test_migration_cutover.py",),
    "T-07-24": ("tests/test_projection_sql_atomicity.py",),
    "T-07-26": ("tests/contracts/test_postgresql_lifecycle_authority.py",),
    "T-07-27": ("tests/test_migration_remote_contract.py",),
    "T-07-28": ("tests/test_migration_remote_contract.py",),
    "T-07-29": ("tests/test_migration_remote_contract.py",),
    "T-07-31": ("tests/test_migration_cutover.py",),
    "T-07-32": ("tests/test_migration_cutover.py",),
    "T-07-33": ("tests/test_migration_cutover.py",),
    "T-07-34": ("tests/test_migration_cutover.py",),
    "T-07-36": ("tests/test_rebuild_workflow.py",),
    "T-07-37": ("tests/test_rebuild_workflow.py",),
    "T-07-38": ("tests/test_handler_registration.py",),
    "T-07-39": ("tests/test_rebuild_workflow.py",),
    "T-07-40": ("tests/test_rebuild_workflow.py",),
    "T-07-42": ("tests/test_migration_public_contract.py",),
    "T-07-43": ("tests/test_migration_public_contract.py",),
    "T-07-44": ("tests/test_migration_public_contract.py",),
    "T-07-45": ("tests/test_migration_public_contract.py",),
    "T-07-46": ("tests/test_phase7_contract_verifier.py",),
    "T-07-47": ("tests/test_phase7_contract_verifier.py",),
    "T-07-48": ("tests/test_phase7_contract_verifier.py",),
    "T-07-49": ("tests/test_phase7_contract_verifier.py",),
    "T-07-50": ("tests/test_phase7_contract_verifier.py",),
}

FLAGGED_ASSUMPTION_NODES = {
    "MIGR-03": ("tests/test_migration_inspection.py",),
    "MIGR-04": ("tests/test_migration_cutover.py",),
    "MIGR-05": ("tests/test_migration_run_evidence.py",),
    "MIGR-06": ("tests/test_rebuild_workflow.py",),
}

PLAN01_PROHIBITIONS = (
    "No ordinary constructor, open, initialize, read, cache policy, reconciliation, or cleanup path may migrate, rebuild, adopt, activate, or purge a store.",
    "No work-directory file, filesystem path, symlink, object listing, candidate presence, or derived projection may become lifecycle or cutover authority.",
    "No new lock, queue, lease, sidecar, daemon, scheduler, online writer protocol, or second lifecycle state machine may be added.",
    "No historical compatibility reader, manufactured persisted version, universal native payload converter, or force override may be introduced.",
    "No partial candidate may activate, and activation/finalize may not physically delete the prior valid store.",
    "No unauthenticated evidence, stale source fingerprint, unexplained candidate, or incomplete catalog may be adopted.",
    "No signing key bytes, raw credentials, or secret provider paths may appear in plans, reports, evidence, or logs.",
    "No unbounded inventory/evidence or Phase 8 live PostgreSQL/AWS S3/performance qualification claim may enter Phase 7.",
)

PHASE7_RUFF_PATHS = (
    "src/cacheness/error_handling.py",
    "src/cacheness/handlers.py",
    "src/cacheness/interfaces.py",
    "src/cacheness/storage/__init__.py",
    "src/cacheness/storage/blob_store.py",
    "src/cacheness/storage/memory_lifecycle_authority.py",
    "src/cacheness/storage/migration.py",
    "src/cacheness/storage/migration_authority.py",
    "src/cacheness/storage/migration_evidence.py",
    "src/cacheness/storage/projections.py",
    "src/cacheness/storage/sqlite_lifecycle_authority.py",
    "src/cacheness/storage/backends/postgresql_lifecycle_authority.py",
    "src/cacheness/storage/backends/s3_backend.py",
    *PHASE7_TEST_NODES,
    "tests/test_phase7_contract_verifier.py",
)

_ORDINARY_MODULES = frozenset(
    {
        "src/cacheness/core.py",
        "src/cacheness/storage/blob_store.py",
        "src/cacheness/storage/projections.py",
    }
)
_MAINTENANCE_NAMES = frozenset(
    {
        "OfflineMigrationService",
        "inspect_migration_store",
        "create_rebuild_plan",
        "activate_verified_candidate",
        "finalize_verified_candidate",
        "purge",
    }
)


def _dotted_name(node: ast.AST) -> str | None:
    """Return a static dotted name while rejecting dynamic expressions."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return None if prefix is None else f"{prefix}.{node.attr}"
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    return None


def _call_name(node: ast.Call) -> str | None:
    """Return the terminal component of a statically known call."""
    dotted = _dotted_name(node.func)
    return None if dotted is None else dotted.rsplit(".", maxsplit=1)[-1]


def _mapping_difference(
    actual: Iterable[str], expected: Iterable[str], label: str
) -> tuple[str, ...]:
    """Render a deterministic fixed-inventory mismatch."""
    actual_set = frozenset(actual)
    expected_set = frozenset(expected)
    if actual_set == expected_set:
        return ()
    return (
        f"{label} differs from the fixed Phase 7 set: "
        f"missing={sorted(expected_set - actual_set)!r}, "
        f"unexpected={sorted(actual_set - expected_set)!r}",
    )


def validate_paths(root: Path, paths: Iterable[str]) -> tuple[str, ...]:
    """Reject root escape and missing fixed artifacts before any execution."""
    errors: list[str] = []
    for path in paths:
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            errors.append(f"manifest path is not repository-relative: {path}")
        elif not (root / candidate).is_file():
            errors.append(f"manifest artifact is missing: {path}")
    return tuple(errors)


def validate_mapping_inventory(
    *,
    requirements: Mapping[str, Sequence[str]],
    decisions: Mapping[str, Sequence[str]],
    threats: Mapping[str, Sequence[str]],
    assumptions: Mapping[str, Sequence[str]],
    prohibitions: Sequence[str],
) -> tuple[str, ...]:
    """Reject a silent coverage-map deletion without inferring replacements."""
    errors: list[str] = []
    errors.extend(
        _mapping_difference(requirements, MIGRATION_REQUIREMENT_NODES, "MIGR mapping")
    )
    errors.extend(_mapping_difference(decisions, DECISION_NODES, "D mapping"))
    errors.extend(
        _mapping_difference(threats, _DECLARED_PHASE7_THREAT_IDS, "Threat mapping")
    )
    errors.extend(
        _mapping_difference(
            assumptions, MIGRATION_REQUIREMENT_NODES, "Flagged-assumption mapping"
        )
    )
    if tuple(prohibitions) != PLAN01_PROHIBITIONS:
        errors.append("Plan 01 prohibition inventory differs from the fixed eight-row set")
    return tuple(errors)


def validate_fixed_manifest(root: Path) -> tuple[str, ...]:
    """Validate the literal path and cross-reference inventories."""
    errors: list[str] = []
    errors.extend(
        _mapping_difference(
            PHASE7_PRODUCTION_PATHS,
            _PHASE7_REVIEWED_PRODUCTION_PATHS,
            "Phase 7 production inventory",
        )
    )
    actual_tests = frozenset(PHASE7_TEST_NODES)
    reviewed_tests = frozenset(_PHASE7_REVIEWED_TEST_NODES)
    if actual_tests != reviewed_tests:
        errors.append(
            "Phase 7 test inventory differs from the fixed reviewed set: "
            f"missing={sorted(reviewed_tests - actual_tests)!r}, "
            f"unexpected={sorted(actual_tests - reviewed_tests)!r}"
        )
    if errors:
        return tuple(errors)
    errors.extend(validate_paths(root, (*PHASE7_PRODUCTION_PATHS, *PHASE7_TEST_NODES)))
    errors.extend(validate_paths(root, (*PHASE7_PLAN_PATHS, PHASE7_CONTEXT_PATH)))
    errors.extend(
        validate_mapping_inventory(
            requirements=MIGRATION_REQUIREMENT_NODES,
            decisions=DECISION_NODES,
            threats=SECURITY_THREAT_NODES,
            assumptions=FLAGGED_ASSUMPTION_NODES,
            prohibitions=PLAN01_PROHIBITIONS,
        )
    )
    known_nodes = frozenset(PHASE7_TEST_NODES)
    for label, mapping in (
        ("MIGR", MIGRATION_REQUIREMENT_NODES),
        ("D", DECISION_NODES),
        ("threat", SECURITY_THREAT_NODES),
        ("flagged assumption", FLAGGED_ASSUMPTION_NODES),
    ):
        for item, nodes in mapping.items():
            if not nodes:
                errors.append(f"{label} mapping has no executable evidence: {item}")
            for node in nodes:
                if node not in known_nodes:
                    errors.append(f"{label} mapping references non-manifest node: {item}: {node}")
    return tuple(errors)


class _ArchitectureVisitor(ast.NodeVisitor):
    """Audit executable migration shapes while ignoring comments and prose."""

    def __init__(self, filename: str) -> None:
        self.filename = PurePosixPath(filename).as_posix()
        self.findings: list[str] = []
        self._function_context: list[dict[str, bool]] = []

    def _add(self, finding: str) -> None:
        if finding not in self.findings:
            self.findings.append(finding)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if (
            self.filename == "src/cacheness/storage/migration.py"
            and node.name.endswith("MigrationAuthority")
            and node.name != "MigrationAuthority"
        ):
            self._add(f"second migration authority: {node.name}")
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._function_context.append({"listing": False, "integrity": False})
        self.generic_visit(node)
        self._function_context.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted_name(node.func) or ""
        name = _call_name(node)
        context = self._function_context[-1] if self._function_context else None
        if self.filename in _ORDINARY_MODULES and name in _MAINTENANCE_NAMES:
            self._add(f"ordinary lifecycle path invokes offline maintenance: {name}")
        if self.filename == "src/cacheness/storage/migration.py":
            if dotted in {"threading.Lock", "threading.RLock", "queue.Queue", "asyncio.Lock", "asyncio.Queue"}:
                self._add(f"migration coordination primitive: {dotted}")
            if name in {"listdir", "list_objects_v2", "list_objects"} and context is not None:
                context["listing"] = True
            if name == "activate_verified_candidate" and context and context["listing"]:
                self._add("candidate/listing adoption reaches activation")
            if name and name.startswith("verify_") and context is not None:
                context["integrity"] = True
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "handler"
                and node.func.attr in {"read", "load", "deserialize"}
                and context is not None
                and not context["integrity"]
            ):
                self._add("handler read precedes integrity verification")
            if name == "inventory_page":
                keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
                if not ({"limit", "page_size"} & keywords) or "work_cap" not in keywords:
                    self._add("unbounded inventory/evidence operation: inventory_page")
        if name in {"print", "warning", "error", "info", "debug"}:
            for argument in node.args:
                if isinstance(argument, ast.Name) and re.search(
                    r"(?:key|secret|credential|password|token)", argument.id, re.I
                ):
                    self._add(f"secret value flows to output: {argument.id}")
        self.generic_visit(node)


def audit_source(source: str, filename: str) -> tuple[str, ...]:
    """Return deterministic executable-only architecture findings."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        return (f"unreadable source: {error.msg}",)
    visitor = _ArchitectureVisitor(filename)
    visitor.visit(tree)
    return tuple(visitor.findings)


def audit_phase7_text(text: str) -> tuple[str, ...]:
    """Reject only positive support, performance, and secret-rendering claims."""
    normalized = " ".join(text.lower().split())
    findings: list[str] = []
    if (
        "phase 7" in normalized
        and "does not qualify" not in normalized
        and "do not qualify" not in normalized
        and "phase 8" not in normalized
        and "live postgresql" in normalized
        and "aws s3" in normalized
        and re.search(r"qualif|support", normalized)
    ):
        findings.append("false Phase 7 live-service qualification claim")
    if "performance distribution" in normalized and re.search(
        r"guarantee|qualif|pass", normalized
    ):
        findings.append("Phase 7 performance qualification claim")
    if re.search(r"(?:credential|secret|signing)[ _-]?path\s*=", normalized):
        findings.append("secret provider path appears in rendered text")
    return tuple(findings)


def audit_coverage_document(coverage: str) -> tuple[str, ...]:
    """Validate the stored detector record and explicit Phase 8 boundary."""
    start_marker = "<!-- phase7-api-coverage:detector-result:start -->\n```json\n"
    end_marker = "\n```\n<!-- phase7-api-coverage:detector-result:end -->"
    try:
        start = coverage.index(start_marker) + len(start_marker)
        end = coverage.index(end_marker, start)
        result = json.loads(coverage[start:end])
    except (ValueError, json.JSONDecodeError) as error:
        return (f"coverage detector record is unreadable: {error}",)
    if not isinstance(result, dict) or result.get("detected") is not True:
        return ("coverage detector result no longer preserves the recorded public-API signal",)
    required = (
        "No external API integration:",
        "PostgresqlLifecycleAuthority",
        "S3BlobBackend",
        "tests/contracts/test_postgresql_lifecycle_authority.py",
        "tests/test_migration_remote_contract.py",
        "tests/test_s3_blob_backend.py",
        "do not qualify live PostgreSQL/AWS S3",
        "Phase 8 alone",
    )
    if any(item not in coverage for item in required):
        return ("coverage declaration omits a required deterministic-adapter or Phase 8 boundary",)
    if "| capability | decision | reason |" in coverage.lower():
        return ("coverage declaration fabricates an external capability matrix",)
    if "qualifies live PostgreSQL/AWS S3" in coverage:
        return ("coverage declaration makes a false live-service qualification claim",)
    return ()


def _assignment_value(source: str, variable: str) -> object | None:
    """Read a simple assigned literal without treating comments as evidence."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == variable for target in node.targets):
                if isinstance(node.value, ast.Constant):
                    return node.value.value
    return None


def _audit_persisted_baselines(root: Path) -> tuple[str, ...]:
    """Pin the released SQLite/PostgreSQL publication baselines to 8 and 4."""
    sqlite = (root / "src/cacheness/storage/sqlite_lifecycle_authority.py").read_text(
        encoding="utf-8"
    )
    postgresql = (
        root / "src/cacheness/storage/backends/postgresql_lifecycle_authority.py"
    ).read_text(encoding="utf-8")
    errors: list[str] = []
    if _assignment_value(sqlite, "SQLITE_USER_VERSION") != 8:
        errors.append("SQLite authority publication schema is not pinned to 8")
    if _assignment_value(postgresql, "POSTGRESQL_AUTHORITY_SCHEMA_VERSION") != 4:
        errors.append("PostgreSQL authority publication schema is not pinned to 4")
    if _assignment_value(postgresql, "POSTGRESQL_AUTHORITY_CAPABILITY") != "postgresql-lifecycle-authority-v4":
        errors.append("PostgreSQL authority capability is not pinned to schema 4")
    return tuple(errors)


def _audit_worker_fence(root: Path) -> tuple[str, ...]:
    """Check that ordinary BlobStore entry reuses the authority offline fence."""
    source = (root / "src/cacheness/storage/blob_store.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="src/cacheness/storage/blob_store.py")
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    canonical = methods.get("_require_canonical_store")
    if canonical is None:
        return ("BlobStore lacks the ordinary-worker canonical-store fence",)
    has_fence_lookup = any(
        isinstance(node, ast.Call)
        and _call_name(node) == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "require_ordinary_worker_access"
        for node in ast.walk(canonical)
    )
    has_fence_call = any(
        isinstance(node, ast.Call) and _call_name(node) == "require_worker_access"
        for node in ast.walk(canonical)
    )
    if not (has_fence_lookup and has_fence_call):
        return ("BlobStore canonical-store fence does not reject activated_offline workers",)
    return ()


def _phase7_detector_scope(root: Path) -> str:
    """Assemble the exact reviewed scope passed to the active detector."""
    roadmap = (root / ".planning/ROADMAP.md").read_text(encoding="utf-8")
    start = roadmap.index("### Phase 7: Explicit Migration and Rebuild Cutover")
    end = roadmap.index("### Phase 8: Production Gates and Performance Stabilization", start)
    bodies: list[str] = []
    for relative_path in PHASE7_PLAN_PATHS:
        plan = (root / relative_path).read_text(encoding="utf-8")
        parts = plan.split("\n---\n", 1)
        if len(parts) != 2:
            raise ValueError(f"{relative_path} lacks a frontmatter/body boundary")
        bodies.append(parts[1])
    return roadmap[start:end] + "".join(bodies)


def _detector_path() -> Path | None:
    """Locate only the installed GSD detector; never substitute a local clone."""
    roots: list[Path] = []
    configured_root = os.environ.get("CODEX_HOME")
    if configured_root:
        roots.append(Path(configured_root))
    roots.append(Path.home() / ".codex")
    for root in roots:
        candidate = root / "gsd-core/bin/lib/api-coverage.cjs"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _validate_live_detector(root: Path, coverage: str) -> tuple[str, ...]:
    """Compare stored detector JSON against the active runtime's actual output."""
    detector = _detector_path()
    if detector is None:
        return ("active GSD API-coverage detector is unavailable",)
    start_marker = "<!-- phase7-api-coverage:detector-result:start -->\n```json\n"
    end_marker = "\n```\n<!-- phase7-api-coverage:detector-result:end -->"
    start = coverage.index(start_marker) + len(start_marker)
    end = coverage.index(end_marker, start)
    stored = json.loads(coverage[start:end])
    try:
        completed = subprocess.run(
            ["node", str(detector), "--json"],
            input=_phase7_detector_scope(root),
            text=True,
            capture_output=True,
            check=False,
            timeout=PYTEST_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return (f"active GSD API-coverage detector could not run: {error}",)
    if completed.returncode not in {0, 1}:
        return (f"active GSD API-coverage detector exited {completed.returncode}: {completed.stderr.strip()}",)
    try:
        observed = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        return (f"active GSD API-coverage detector emitted invalid JSON: {error}",)
    if observed != stored:
        return ("stored API-coverage detector result differs from the active detector",)
    return ()


def _run_pytest(
    root: Path,
    nodes: Sequence[str],
    label: str,
    *,
    options: Sequence[str] = (),
) -> tuple[bool, str]:
    """Run one finite pytest invocation and preserve non-green output."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *nodes,
        *options,
        "-o",
        "log_cli=false",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=PYTEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        return False, f"{label}: timed out after {PYTEST_TIMEOUT_SECONDS} seconds\n{error.output or ''}"
    except OSError as error:
        return False, f"{label}: could not start pytest: {error}"
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        return False, f"{label}: pytest exited {completed.returncode}\n{output}"
    return True, f"{label}: {output}"


def _run_ruff(root: Path) -> tuple[bool, str]:
    """Lint only the reviewed Phase 7 Python surface; no baseline is hidden."""
    command = [sys.executable, "-m", "ruff", "check", *PHASE7_RUFF_PATHS]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=PYTEST_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"Phase 7 scoped Ruff could not run: {error}"
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        return False, f"Phase 7 scoped Ruff exited {completed.returncode}\n{output}"
    return True, f"Phase 7 scoped Ruff: {output or 'PASS'}"


def verify_repository(root: Path, quick: bool) -> tuple[bool, tuple[str, ...]]:
    """Run fixed static checks and the reviewed behavioral evidence.

    ``quick`` omits only the full repository suite and scoped Ruff.  It never
    turns an unavailable external service into a skip/pass and never makes a
    performance deadline part of storage correctness.
    """
    root = root.resolve()
    errors: list[str] = []
    errors.extend(validate_fixed_manifest(root))
    for relative_path in (
        "src/cacheness/core.py",
        "src/cacheness/storage/blob_store.py",
        "src/cacheness/storage/migration.py",
        "src/cacheness/storage/migration_evidence.py",
        "src/cacheness/storage/projections.py",
    ):
        try:
            errors.extend(audit_source((root / relative_path).read_text(encoding="utf-8"), relative_path))
        except OSError as error:
            errors.append(f"architecture source unreadable: {relative_path}: {error}")
    try:
        errors.extend(_audit_persisted_baselines(root))
        errors.extend(_audit_worker_fence(root))
    except (OSError, SyntaxError) as error:
        errors.append(f"persisted publication audit unreadable: {error}")
    coverage_path = root / ".planning/phases/07-explicit-migration-and-rebuild-cutover/07-COVERAGE.md"
    try:
        coverage = coverage_path.read_text(encoding="utf-8")
        errors.extend(audit_coverage_document(coverage))
        if not quick:
            errors.extend(_validate_live_detector(root, coverage))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"coverage audit unreadable: {error}")
    for relative_path in (
        "docs/STORAGE_MIGRATION.md",
        "docs/STORAGE_INITIALIZATION.md",
        "docs/BACKEND_SELECTION.md",
    ):
        try:
            errors.extend(audit_phase7_text((root / relative_path).read_text(encoding="utf-8")))
        except OSError as error:
            errors.append(f"documentation audit unreadable: {relative_path}: {error}")
    context_path = root / PHASE7_CONTEXT_PATH
    try:
        context = context_path.read_text(encoding="utf-8")
        for decision in DECISION_NODES:
            if decision not in context:
                errors.append(f"Phase 7 context omits declared decision: {decision}")
    except OSError as error:
        errors.append(f"Phase 7 context audit unreadable: {error}")
    if not errors:
        nodes = PHASE7_QUICK_TEST_NODES if quick else PHASE7_TEST_NODES
        passed, evidence = _run_pytest(root, nodes, "fixed Phase 7 behavioral inventory")
        if not passed:
            errors.append(evidence)
    if not quick and not errors:
        live_options = tuple(f"--ignore={node}" for node in PHASE8_LIVE_UNQUALIFIED_NODES)
        passed, evidence = _run_pytest(
            root,
            (),
            "full deterministic non-live suite",
            options=live_options,
        )
        if not passed:
            errors.append(evidence)
        else:
            passed, evidence = _run_ruff(root)
            if not passed:
                errors.append(evidence)
    return not errors, tuple(errors)


def _diagnostic_requirements(errors: Iterable[str]) -> frozenset[str]:
    """Map failed behavior or static evidence to the public MIGR labels."""
    labels: set[str] = set()
    for error in errors:
        for requirement in MIGRATION_REQUIREMENT_NODES:
            if error.startswith(f"{requirement}:"):
                labels.add(requirement)
        if error.startswith(
            (
                "MIGR mapping",
                "D mapping",
                "Threat mapping",
                "Flagged-assumption mapping",
                "Plan 01 prohibition",
                "Phase 7 test inventory",
                "Phase 7 production inventory",
            )
        ):
            labels.update(MIGRATION_REQUIREMENT_NODES)
        if error.startswith(("ordinary lifecycle", "SQLite authority", "PostgreSQL authority", "BlobStore canonical")):
            labels.update({"MIGR-04", "MIGR-05"})
        if error.startswith(("candidate/listing", "handler read", "unbounded", "secret value")):
            labels.update({"MIGR-03", "MIGR-05", "MIGR-06"})
    return frozenset(labels)


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 7 verifier from any current working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="run fixed phase evidence")
    mode.add_argument("--all", action="store_true", help="also run full suite and scoped Ruff")
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args(argv)

    passed, errors = verify_repository(args.repo_root, args.quick)
    failed_requirements = _diagnostic_requirements(errors)
    print("Phase 7 fixed migration/rebuild contract verifier")
    for requirement in MIGRATION_REQUIREMENT_NODES:
        status = "see diagnostics" if requirement in failed_requirements else "PASS"
        print(f"{requirement}: {status}")
    print("SQLite/PostgreSQL publication baseline: 8/4 deterministic contract")
    if args.quick:
        print(
            "Deterministic PostgreSQL adapter contract: NOT RUN in --quick; "
            "--all requires the locked all-extras environment"
        )
    else:
        print(
            "Phase 8 live PostgreSQL/AWS S3 modules: NOT RUN and NOT QUALIFIED; "
            "they are excluded by the fixed non-live inventory"
        )
    print("Remote/platform/performance: Phase 8 only; deterministic adapters are not qualification")
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("Fixed Phase 7 inventory: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
