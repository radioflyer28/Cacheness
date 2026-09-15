#!/usr/bin/env python3
"""Verify the closed Phase 8 qualification contract without test discovery.

This tool is deliberately an inventory rather than a convenience test runner.
It checks the reviewed plans, requirements, decisions, threats, source scopes,
workflows, and exact test nodes before it delegates to a bounded local gate.
External evidence is reported as a prerequisite and never inferred from local
or mocked execution.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
import json
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PHASE_DIRECTORY = ".planning/phases/08-production-gates-and-performance-stabilization"
PHASE8_CONTEXT_PATH = f"{PHASE_DIRECTORY}/08-CONTEXT.md"
PHASE8_VALIDATION_PATH = f"{PHASE_DIRECTORY}/08-VALIDATION.md"
PYTEST_TIMEOUT_SECONDS = 900
MAX_LOCAL_EVIDENCE_BYTES = 64 * 1024

# The reviewed contract is intentionally literal.  In particular, it must not
# discover planning files, requirements, threats, or tests at runtime: deleting
# an item from a mutable input must fail the verifier rather than shrink it.
_REVIEWED_PLAN_PATHS = (
    f"{PHASE_DIRECTORY}/08-01-PLAN.md",
    f"{PHASE_DIRECTORY}/08-02-PLAN.md",
    f"{PHASE_DIRECTORY}/08-03-PLAN.md",
    f"{PHASE_DIRECTORY}/08-04-PLAN.md",
    f"{PHASE_DIRECTORY}/08-05-PLAN.md",
    f"{PHASE_DIRECTORY}/08-06-PLAN.md",
    f"{PHASE_DIRECTORY}/08-07-PLAN.md",
    f"{PHASE_DIRECTORY}/08-08-PLAN.md",
    f"{PHASE_DIRECTORY}/08-09-PLAN.md",
    f"{PHASE_DIRECTORY}/08-10-PLAN.md",
    f"{PHASE_DIRECTORY}/08-11-PLAN.md",
    f"{PHASE_DIRECTORY}/08-12-PLAN.md",
    f"{PHASE_DIRECTORY}/08-13-PLAN.md",
    f"{PHASE_DIRECTORY}/08-14-PLAN.md",
    f"{PHASE_DIRECTORY}/08-15-PLAN.md",
)
PHASE8_PLAN_PATHS = tuple(_REVIEWED_PLAN_PATHS)

_REVIEWED_REQUIREMENTS = (
    "BACK-05",
    "QUAL-01",
    "QUAL-02",
    "QUAL-03",
    "QUAL-04",
    "QUAL-05",
    "QUAL-06",
    "QUAL-07",
)
DEFERRED_REQUIREMENTS = ("QUAL-06",)
PHASE8_REQUIREMENTS = tuple(
    requirement
    for requirement in _REVIEWED_REQUIREMENTS
    if requirement not in DEFERRED_REQUIREMENTS
)

_REVIEWED_THREATS = (
    "T-08-01-01",
    "T-08-01-02",
    "T-08-01-03",
    "T-08-01-04",
    "T-08-01-05",
    "T-08-02-01",
    "T-08-02-02",
    "T-08-02-03",
    "T-08-02-04",
    "T-08-02-SC",
    "T-08-03-01",
    "T-08-03-02",
    "T-08-03-03",
    "T-08-03-04",
    "T-08-04-01",
    "T-08-04-02",
    "T-08-04-03",
    "T-08-04-04",
    "T-08-05-01",
    "T-08-05-02",
    "T-08-05-03",
    "T-08-05-04",
    "T-08-06-01",
    "T-08-06-02",
    "T-08-06-03",
    "T-08-06-04",
    "T-08-07-01",
    "T-08-07-02",
    "T-08-07-03",
    "T-08-07-04",
    "T-08-07-05",
    "T-08-08-01",
    "T-08-08-02",
    "T-08-08-03",
    "T-08-08-04",
    "T-08-08-05",
    "T-08-08-06",
    "T-08-09-01",
    "T-08-09-02",
    "T-08-09-03",
    "T-08-09-04",
    "T-08-09-SC",
    "T-08-10-01",
    "T-08-10-02",
    "T-08-10-03",
    "T-08-10-04",
    "T-08-10-05",
    "T-08-10-06",
    "T-08-11-01",
    "T-08-11-02",
    "T-08-11-03",
    "T-08-11-04",
    "T-08-11-05",
    "T-08-11-06",
    "T-08-12-01",
    "T-08-12-02",
    "T-08-12-03",
    "T-08-12-04",
    "T-08-12-05",
    "T-08-12-06",
    "T-08-13-01",
    "T-08-13-02",
    "T-08-13-03",
)
PHASE8_THREATS = tuple(_REVIEWED_THREATS)

_DECISION_TESTS = {
    "D-01": (
        "tests/test_phase8_release_tracer.py::test_tracer_writes_one_validated_exact_commit_deterministic_pass",
    ),
    "D-02": (
        "tests/test_phase8_lifecycle_coverage.py::test_lifecycle_integrity_before_handler_read",
    ),
    "D-03": (
        "tests/qualification/test_phase8_platform.py::test_platform_adr_progress_outcomes_remain_valid",
    ),
    "D-04": (
        "tests/test_phase8_lifecycle_coverage.py::test_lifecycle_cleanup_debt_reconciliation_retires_only_exact_work",
    ),
    "D-05": (
        "tests/qualification/test_phase8_platform.py::test_python_stable_matrix_is_exact_and_complete",
    ),
    "D-06": (
        "tests/qualification/test_phase8_platform.py::test_platform_windows_is_non_native_unavailable_without_substitute_execution",
    ),
    "D-07": (
        "tests/packaging/test_wheel_matrix.py::test_base_wheel_qualification_uses_one_artifact_and_public_round_trips",
    ),
    "D-08": (
        "tests/packaging/test_wheel_matrix.py::test_optional_group_wheel_qualification_runs_each_compatible_extra",
    ),
    "D-09": (
        "tests/qualification/test_phase8_live_workflow.py::test_live_workflow_has_only_protected_rc_and_scheduled_triggers",
    ),
    "D-10": (
        "tests/qualification/test_phase8_quality_workflow.py::test_release_candidate_binds_one_detached_sha_and_uploads_fixed_class_artifacts",
    ),
    "D-11": (
        "tests/qualification/test_phase8_evidence.py::test_scheduled_diagnostic_role_cannot_emit_release_candidate_evidence",
    ),
    "D-12": (
        "tests/qualification/test_phase8_evidence.py::test_runner_rejects_endpoint_override_and_owner_pinning_without_running",
    ),
    "D-13": (
        "tests/qualification/test_phase8_live_workflow.py::test_live_artifacts_are_fixed_retained_and_run_id_addressable",
    ),
    "D-14": (
        "tests/test_phase8_coverage_gate.py::test_rejects_any_regressed_total_or_critical_dimension",
    ),
    "D-15": (
        "tests/test_phase8_coverage_gate.py::test_requires_literal_postgresql_selector_families",
    ),
    "D-16": (
        "tests/test_phase8_coverage_gate.py::test_ruff_scope_is_bounded_and_contains_fixed_critical_inventory",
    ),
    "D-17": (
        "tests/test_phase8_coverage_gate.py::test_changed_python_paths_accepts_the_documented_parent_ref",
    ),
    "D-18": (
        "tests/performance/test_phase8_benchmarks.py::test_workloads_inventory_is_representative_without_topology_cross_product",
    ),
    "D-19": (
        "tests/performance/test_phase8_benchmarks.py::test_hash_measurements_cover_canonical_sizes_without_changing_sha256",
    ),
    "D-20": (
        "tests/performance/test_phase8_benchmarks.py::test_verify_baseline_rejects_mismatched_identity_revision_environment_and_regression",
    ),
    "D-21": (
        "tests/performance/test_complexity_contracts.py::test_reconciliation_formula_has_independent_authority_and_inventory_bounds",
    ),
    "D-22": (
        "tests/performance/test_memory_bounds.py::test_structural_evidence_keeps_counts_and_peak_rss_without_a_timing_claim",
    ),
    "D-23": (
        "tests/test_phase8_contract_verifier.py::test_fixed_manifest_covers_deferred_performance_decision",
    ),
}
DECISION_NODES = dict(_DECISION_TESTS)

# Each threat has one concrete fixed test family.  A test can cover multiple
# threat IDs, but every threat still remains independently named in the
# reviewed map so a threat cannot vanish with a future test refactor.
_THREAT_TEST_MODULES = {
    "01": "tests/test_phase8_release_tracer.py::test_evidence_rejects_unknown_keys_and_contradictory_canonical_json",
    "02": "tests/packaging/test_wheel_matrix.py::test_optional_group_inventory_is_exact_and_rejects_metadata_drift",
    "03": "tests/qualification/test_phase8_platform.py::test_platform_adr_progress_outcomes_remain_valid",
    "04": "tests/test_phase8_lifecycle_coverage.py::test_lifecycle_ambiguous_publication_settles_only_exact_locator_identity",
    "05": "tests/test_phase8_coverage_gate.py::test_rejects_any_regressed_total_or_critical_dimension",
    "06": "tests/performance/test_complexity_contracts.py::test_reconciliation_formula_has_independent_authority_and_inventory_bounds",
    "07": "tests/performance/test_phase8_benchmarks.py::test_hash_measurements_cover_canonical_sizes_without_changing_sha256",
    "08": "tests/qualification/test_phase8_evidence.py::test_runner_requires_complete_clean_current_source_proof",
    "09": "tests/qualification/test_phase8_quality_workflow.py::test_quality_workflow_keeps_live_and_controlled_performance_out_of_pr_jobs",
    "10": "tests/test_phase8_contract_verifier.py::test_ast_source_audit_rejects_prohibited_architecture_regressions",
    "11": "tests/test_phase8_contract_verifier.py::test_external_statuses_are_explicit_nonclaims",
    "12": "tests/test_phase8_contract_verifier.py::test_external_statuses_are_explicit_nonclaims",
    "13": "tests/performance/test_phase8_benchmarks.py::test_preflight_runner_emits_only_the_bounded_eligibility_record",
    "14": "tests/test_phase8_contract_verifier.py::test_fixed_manifest_covers_deferred_performance_decision",
}
_REVIEWED_THREAT_NODES = {
    threat: (_THREAT_TEST_MODULES[threat.split("-")[2]],)
    for threat in _REVIEWED_THREATS
}
_REVIEWED_THREAT_NODES.update(
    {
        "T-08-14-01": (
            "tests/qualification/test_phase8_release.py::test_required_release_collection_excludes_deferred_performance",
        ),
        "T-08-14-02": (
            "tests/qualification/test_phase8_release.py::test_deferred_performance_artifacts_and_macos_diagnostics_are_rejected",
        ),
        "T-08-14-03": (
            "tests/qualification/test_phase8_release.py::test_aggregate_records_exact_deferred_performance_nonclaim",
        ),
        "T-08-14-04": (
            "tests/test_phase8_contract_verifier.py::test_fixed_manifest_covers_deferred_performance_decision",
        ),
        "T-08-15-01": (
            "tests/qualification/test_phase8_evidence.py::test_preflight_accepts_sanitized_configuration_without_external_effects",
        ),
        "T-08-15-02": (
            "tests/qualification/test_phase8_evidence.py::test_preflight_rejects_missing_invalid_and_disallowed_configuration_without_external_effects",
        ),
        "T-08-15-03": (
            "tests/qualification/test_phase8_evidence.py::test_preflight_requires_clean_source_and_reviewed_cleanup_contract",
        ),
        "T-08-15-04": (
            "tests/qualification/test_phase8_evidence.py::test_preflight_requires_clean_source_and_reviewed_cleanup_contract",
        ),
        "T-08-15-05": (
            "tests/qualification/test_phase8_evidence.py::test_preflight_cli_never_runs_or_writes_qualification_evidence",
        ),
    }
)
_REVIEWED_THREATS = (
    *_REVIEWED_THREATS,
    "T-08-14-01",
    "T-08-14-02",
    "T-08-14-03",
    "T-08-14-04",
    "T-08-15-01",
    "T-08-15-02",
    "T-08-15-03",
    "T-08-15-04",
    "T-08-15-05",
)
PHASE8_THREATS = tuple(_REVIEWED_THREATS)
THREAT_NODES = dict(_REVIEWED_THREAT_NODES)

_REVIEWED_SOURCE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "docs/adr/0001-topology-specific-storage-guarantees.md",
    "docs/RELEASE_QUALIFICATION.md",
    "src/cacheness/storage/blob_store.py",
    "src/cacheness/storage/composition.py",
    "src/cacheness/storage/lifecycle.py",
    "src/cacheness/storage/obstore_generation_io.py",
    "src/cacheness/cache_policy.py",
    "tools/phase8_evidence.py",
    "tools/run_phase8_local_gates.py",
    "tools/run_phase8_packaging.py",
    "tools/run_phase8_platform_gates.py",
    "tools/run_phase8_qualification.py",
    "tools/run_phase8_scale_gates.py",
    "tools/verify_phase8_coverage.py",
    "tools/verify_phase8_contracts.py",
    "tools/verify_phase8_release.py",
    "benchmarks/phase8_benchmarks.py",
    "tests/packaging/test_wheel_matrix.py",
    "tests/qualification/test_phase8_evidence.py",
    "tests/qualification/test_phase8_live_workflow.py",
    "tests/qualification/test_phase8_platform.py",
    "tests/qualification/test_phase8_quality_workflow.py",
    "tests/qualification/test_phase8_release.py",
    "tests/performance/test_complexity_contracts.py",
    "tests/performance/test_memory_bounds.py",
    "tests/performance/test_phase8_benchmarks.py",
    "tests/test_phase8_cache_policy_coverage.py",
    "tests/test_phase8_contract_verifier.py",
    "tests/test_phase8_coverage_gate.py",
    "tests/test_phase8_lifecycle_coverage.py",
    "tests/test_phase8_release_tracer.py",
)
SOURCE_PATHS = tuple(_REVIEWED_SOURCE_PATHS)
WORKFLOW_PATHS = (
    ".github/workflows/quality.yml",
    ".github/workflows/performance.yml",
    ".github/workflows/live_qualification.yml",
)
EXTERNAL_EVIDENCE_CLASSES = ("controlled_performance", "live_services")
CONTROLLED_PERFORMANCE_SEED = (
    ".planning/seeds/SEED-006-qualify-controlled-linux-performance.md"
)


def _module_level_test_names(path: Path) -> set[str]:
    """Return module-level pytest test functions without importing untrusted code."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def validate_selector(selector: str, root: Path = REPOSITORY_ROOT) -> tuple[str, ...]:
    """Return exact-node errors before pytest can treat deselection as success."""
    if selector.count("::") != 1:
        return (f"malformed selector: {selector}",)
    raw_path, name = selector.split("::", 1)
    path = PurePosixPath(raw_path)
    if not name.startswith("test_") or path.is_absolute() or ".." in path.parts:
        return (f"unsafe selector: {selector}",)
    candidate = root / path
    if not candidate.is_file() or candidate.is_symlink():
        return (f"missing selector module: {selector}",)
    if name not in _module_level_test_names(candidate):
        return (f"missing selector node: {selector}",)
    return ()


def validate_mapping_inventory(
    *,
    decisions: Mapping[str, Sequence[str]],
    threats: Mapping[str, Sequence[str]],
    plans: Sequence[str],
) -> tuple[str, ...]:
    """Validate maps as closed inventories, not caller-selected partial input."""
    errors: list[str] = []
    if tuple(plans) != PHASE8_PLAN_PATHS:
        errors.append("plan inventory differs from reviewed Phase 8 plan paths")
    if set(decisions) != set(DECISION_NODES):
        errors.append("decision inventory differs from reviewed decisions")
    if set(threats) != set(THREAT_NODES):
        errors.append("threat inventory differs from reviewed threats")
    for label, mapping in (("decision", decisions), ("threat", threats)):
        for identifier, selectors in mapping.items():
            if not selectors or len(set(selectors)) != len(selectors):
                errors.append(f"{label} {identifier} has duplicate or empty selectors")
    return tuple(errors)


class _ArchitectureVisitor(ast.NodeVisitor):
    """Reject executable topology regressions without trusting comments or text."""

    def __init__(self, filename: str) -> None:
        self.filename = PurePosixPath(filename).as_posix()
        self.findings: list[str] = []

    def _add(self, value: str) -> None:
        if value not in self.findings:
            self.findings.append(value)

    def visit_Import(self, node: ast.Import) -> None:
        if self.filename.startswith("src/") and any(
            alias.name == "boto3" for alias in node.names
        ):
            self._add("production boto3 import")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if (
            self.filename.startswith("src/")
            and node.module
            and "legacy_payload" in node.module
        ):
            self._add("legacy payload mechanic")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if (
            node.name.endswith("LifecycleAuthority")
            and node.name != "LifecycleAuthority"
        ):
            self._add("parallel lifecycle authority")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        name = (
            function.id
            if isinstance(function, ast.Name)
            else getattr(function, "attr", "")
        )
        if self.filename.startswith("src/cacheness/storage/") and name in {
            "Lock",
            "RLock",
            "Queue",
            "Semaphore",
            "Event",
        }:
            self._add("lifecycle coordination primitive")
        if (
            self.filename.endswith(
                ("blob_store.py", "lifecycle.py", "obstore_generation_io.py")
            )
            and "xxh" in name.casefold()
        ):
            self._add("canonical digest change")
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        if (
            isinstance(node.body, ast.Constant)
            and node.body.value == "PASS"
            and "unavailable" in ast.unparse(node.test).casefold()
        ):
            self._add("unavailable-to-pass conversion")
        self.generic_visit(node)


def audit_source(source: str, filename: str) -> tuple[str, ...]:
    """Return architecture findings from executable syntax only."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return ("source syntax is invalid",)
    visitor = _ArchitectureVisitor(filename)
    visitor.visit(tree)
    return tuple(visitor.findings)


def _validate_static_exports() -> list[str]:
    errors: list[str] = []
    if PHASE8_PLAN_PATHS != _REVIEWED_PLAN_PATHS:
        errors.append("plan inventory was mutated")
    if PHASE8_REQUIREMENTS != tuple(
        requirement
        for requirement in _REVIEWED_REQUIREMENTS
        if requirement not in DEFERRED_REQUIREMENTS
    ):
        errors.append("requirement inventory was mutated")
    if PHASE8_THREATS != _REVIEWED_THREATS:
        errors.append("threat inventory was mutated")
    if DECISION_NODES != _DECISION_TESTS:
        errors.append("decision map was mutated")
    if THREAT_NODES != _REVIEWED_THREAT_NODES:
        errors.append("threat map was mutated")
    return errors


def validate_fixed_manifest(root: Path = REPOSITORY_ROOT) -> tuple[str, ...]:
    """Verify every reviewed plan, source, workflow, threat, and selector."""
    errors = _validate_static_exports()
    errors.extend(
        validate_mapping_inventory(
            decisions=DECISION_NODES, threats=THREAT_NODES, plans=PHASE8_PLAN_PATHS
        )
    )
    for path in (*PHASE8_PLAN_PATHS, PHASE8_CONTEXT_PATH, PHASE8_VALIDATION_PATH):
        candidate = root / path
        if not candidate.is_file() or candidate.is_symlink():
            errors.append(f"missing reviewed planning source: {path}")
            continue
        text = candidate.read_text(encoding="utf-8")
        if path == PHASE8_CONTEXT_PATH:
            for decision in DECISION_NODES:
                if decision not in text:
                    errors.append(f"context omits {decision}")
        if path in PHASE8_PLAN_PATHS:
            plan_number = path.rsplit("-", 2)[-2]
            for threat in (
                item
                for item in PHASE8_THREATS
                if item.startswith(f"T-08-{plan_number}-")
            ):
                if threat not in text:
                    errors.append(f"plan omits threat {threat}")
    for requirement in (*PHASE8_REQUIREMENTS, *DEFERRED_REQUIREMENTS):
        if requirement not in (root / ".planning/REQUIREMENTS.md").read_text(
            encoding="utf-8"
        ):
            errors.append(f"requirements source omits {requirement}")
    for path in (*SOURCE_PATHS, *WORKFLOW_PATHS):
        candidate = root / path
        if not candidate.is_file() or candidate.is_symlink():
            errors.append(f"missing reviewed source: {path}")
            continue
        if path.endswith(".py"):
            errors.extend(
                f"{path}: {item}"
                for item in audit_source(candidate.read_text(encoding="utf-8"), path)
            )
    for selectors in (*DECISION_NODES.values(), *THREAT_NODES.values()):
        for selector in selectors:
            errors.extend(validate_selector(selector, root))
    return tuple(errors)


def render_external_statuses(statuses: Mapping[str, str]) -> str:
    """Render explicit nonclaims without choosing a substitute evidence class."""
    packaging_status = statuses.get("packaging", "NOT_RUN")
    platform_status = statuses.get("platform", "NOT_RUN")
    controlled = statuses.get("controlled_performance", "DEFERRED")
    live = statuses.get("live_services", "UNAVAILABLE")
    windows = statuses.get("windows", "NOT_QUALIFIED")
    return "\n".join(
        (
            f"packaging: {packaging_status}",
            f"platform: {platform_status}",
            f"controlled_performance: {controlled}",
            "controlled_performance qualification: NOT_QUALIFIED",
            "controlled_performance requirement: QUAL-06",
            f"controlled_performance deferred_to: {CONTROLLED_PERFORMANCE_SEED}",
            f"live_services: {live}",
            f"windows: {windows}",
        )
    )


def external_status_is_blocking(evidence_class: str, status: str) -> bool:
    """Return whether one external status blocks the current release boundary."""
    if evidence_class == "controlled_performance":
        return False
    if evidence_class == "live_services":
        return status != "QUALIFIED"
    return status != "PASS"


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(command),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=PYTEST_TIMEOUT_SECONDS,
    )


def local_gate_commands(output_directory: Path) -> tuple[tuple[str, ...], ...]:
    """Return every fixed local evidence command for the current host identity."""
    python_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    prefix = (
        "uv",
        "run",
        "--isolated",
        "--all-extras",
        "--group",
        "dev",
        "--frozen",
        "python",
        "tools/run_phase8_local_gates.py",
    )
    return (
        (
            *prefix,
            "all",
            "--output-dir",
            str(output_directory / "all"),
        ),
        (
            *prefix,
            "platform",
            "--expected-os",
            platform.system(),
            "--python-minor",
            python_minor,
            "--feature-profile",
            "core",
            "--output",
            str(output_directory / "platform.json"),
        ),
    )


def _unavailable_gate_status(command: Sequence[str]) -> tuple[str, str] | None:
    """Read one bounded local nonclaim envelope without upgrading it to a pass."""
    gate = command[command.index("tools/run_phase8_local_gates.py") + 1]
    if gate == "all":
        output = Path(command[command.index("--output-dir") + 1]) / "packaging.json"
        evidence_class = "packaging"
    elif gate == "platform":
        output = Path(command[command.index("--output") + 1])
        evidence_class = "platform"
    else:
        return None
    try:
        if output.is_symlink() or not output.is_file():
            return None
        if output.stat().st_size > MAX_LOCAL_EVIDENCE_BYTES:
            return None
        envelope = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(envelope, dict):
        return None
    if (
        envelope.get("evidence_class") != evidence_class
        or envelope.get("status") != "UNAVAILABLE"
    ):
        return None
    return gate if gate != "all" else evidence_class, "UNAVAILABLE"


def _run_local(mode: str) -> tuple[int, dict[str, str]]:
    command = (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "log_cli=false",
        "tests/test_phase8_contract_verifier.py",
        "tests/qualification/test_phase8_release.py",
        "-x",
    )
    if mode == "quick":
        command = command[:-3] + ("tests/test_phase8_contract_verifier.py", "-x")
    try:
        result = _run(command)
    except (OSError, subprocess.TimeoutExpired):
        return 1, {}
    if result.returncode != 0:
        return 1, {}
    if mode == "quick":
        return 0, {}
    statuses: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="phase8-contract-") as temporary:
        for local_command in local_gate_commands(Path(temporary)):
            try:
                local_result = _run(local_command)
            except (OSError, subprocess.TimeoutExpired):
                return 1, statuses
            gate = local_command[
                local_command.index("tools/run_phase8_local_gates.py") + 1
            ]
            unavailable = _unavailable_gate_status(local_command)
            if local_result.returncode != 0 and unavailable is not None:
                evidence_class, status = unavailable
                statuses[evidence_class] = status
                continue
            if local_result.returncode != 0:
                return 1, statuses
            if gate == "platform":
                statuses["platform"] = "PASS"
    return 0, statuses


def main(arguments: Sequence[str] | None = None) -> int:
    """Run fixed local verifier tests only after the complete static audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--all", action="store_true")
    parsed = parser.parse_args(arguments)
    errors = validate_fixed_manifest()
    if errors:
        print("Phase 8 fixed contract failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    mode = "quick" if parsed.quick else "all"
    local_exit_code, statuses = _run_local(mode)
    if local_exit_code:
        return 1
    print("phase 8 fixed contract passed")
    print(render_external_statuses(statuses))
    return 0 if mode == "quick" else 2


if __name__ == "__main__":
    raise SystemExit(main())
