#!/usr/bin/env python3
"""Verify the fixed Phase 6 cache-policy contract without remote claims.

The verifier intentionally has a finite inventory.  It never derives scope
from a branch diff, recursively collects the repository, installs packages, or
contacts a service.  A passing run proves only deterministic local behavior
and source structure; PostgreSQL/Amazon S3 release qualification remains a
Phase 8 responsibility.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTEST_TIMEOUT_SECONDS = 300
REMOTE_EVIDENCE_LABEL = "mocked-candidate; BACK-05 remains Phase 8"
STRICT_PROJECTION_LABEL = "SC-06 strict projection rejection"

# This is the complete Phase 6 behavioral manifest.  Keep the incompatible
# projection node exact: a file-level test invocation is not a substitute for
# proving the hostile format_version=999 observation fails closed.
PHASE6_CONTRACT_NODES = (
    "tests/test_phase6_lookup_contract.py",
    "tests/test_phase6_removal_contract.py",
    "tests/test_phase6_statistics.py",
    "tests/test_phase6_decorator_contract.py",
    "tests/test_phase6_public_api_contract.py",
    "tests/test_phase6_policy_contract.py",
    "tests/contracts/test_phase6_topology_policy.py",
    "tests/test_catalog_projection.py::"
    "test_json_projection_rejects_incompatible_derived_documents",
    "tests/test_sql_cache.py",
)
RETAINED_LIFECYCLE_NODES = (
    "tests/test_phase3_local_workflows.py",
    "tests/test_blob_store_atomic_lifecycle.py",
    "tests/test_blob_store_integrity.py",
    "tests/test_blob_store_reconciliation.py",
    "tests/contracts/test_topology_lifecycle.py",
    "tests/test_lifecycle_authority_contract.py",
    "tests/test_supported_topologies.py",
)
CACH_REQUIREMENT_NODES = {
    "CACH-01": ("tests/contracts/test_phase6_topology_policy.py",),
    "CACH-02": ("tests/test_phase6_policy_contract.py",),
    "CACH-03": ("tests/test_phase6_removal_contract.py",),
    "CACH-04": (
        "tests/test_phase6_lookup_contract.py",
        "tests/test_phase6_decorator_contract.py",
    ),
    "CACH-05": ("tests/test_phase6_statistics.py",),
    "CACH-06": ("tests/test_phase6_public_api_contract.py",),
}
FIXED_REGRESSION_NODES = {
    "CACH-07 SqlCache regression": ("tests/test_sql_cache.py",),
}
ARCHITECTURE_MODULES = (
    "src/cacheness/core.py",
    "src/cacheness/cache_policy.py",
    "src/cacheness/decorators.py",
    "src/cacheness/storage/blob_store.py",
    "src/cacheness/storage/lifecycle.py",
)
DOCUMENTATION_MODULES = ("docs/CACHE_POLICY.md",)
_POLICY_MODULES = frozenset(
    {
        "src/cacheness/core.py",
        "src/cacheness/cache_policy.py",
        "src/cacheness/decorators.py",
    }
)
_RETIRED_ROUTES = frozenset(
    {"cache_function", "memoize", "CacheContext", "get_cache"}
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


def _dotted_name(node: ast.AST) -> str | None:
    """Return a static dotted name while refusing dynamic expressions."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return None if prefix is None else f"{prefix}.{node.attr}"
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    return None


def _call_name(node: ast.Call) -> str | None:
    """Return the final static call component when one is available."""
    dotted = _dotted_name(node.func)
    return None if dotted is None else dotted.rsplit(".", maxsplit=1)[-1]


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    """Return straightforward assignment targets for structural diagnostics."""
    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
    return tuple(target.id for target in targets if isinstance(target, ast.Name))


class _ArchitectureVisitor(ast.NodeVisitor):
    """Audit executable cache-policy shapes without interpreting comments."""

    def __init__(self, filename: str) -> None:
        self.filename = PurePosixPath(filename).as_posix()
        self.findings: list[str] = []
        self._scope_depth = 0

    @property
    def _is_policy_module(self) -> bool:
        return self.filename in _POLICY_MODULES

    def _add(self, finding: str) -> None:
        if finding not in self.findings:
            self.findings.append(finding)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name.endswith("LifecycleEngine") and node.name != "AuthorityLifecycleEngine":
            self._add(f"second lifecycle engine: {node.name}")
        if node.name.endswith("LifecycleCoordinator") or node.name.endswith("Coordinator"):
            self._add(f"second lifecycle coordinator: {node.name}")
        if "Readiness" in node.name and node.name.endswith("Registry"):
            self._add(f"readiness registry: {node.name}")
        if self._is_policy_module and node.name in _RETIRED_ROUTES:
            self._add(f"retired compatibility route: {node.name}")
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._is_policy_module and node.name in _RETIRED_ROUTES:
            self._add(f"retired compatibility route: {node.name}")
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        if self._is_policy_module:
            for alias in node.names:
                if alias.name in {"atexit", "weakref"}:
                    self._add(f"hidden cache owner import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._is_policy_module and node.module in {"atexit", "weakref"}:
            self._add(f"hidden cache owner import: {node.module}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._is_policy_module and self._scope_depth == 0:
            for name in _assignment_names(node):
                if name in {"_global_cache", "global_cache"}:
                    self._add(f"hidden global cache: {name}")
        self._visit_admission_assignment(node, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._visit_admission_assignment(node, node.value)
        self.generic_visit(node)

    def _visit_admission_assignment(
        self, node: ast.Assign | ast.AnnAssign, value: ast.AST | None
    ) -> None:
        if not self._is_policy_module or not isinstance(value, ast.Call):
            return
        dotted = _dotted_name(value.func)
        names = _assignment_names(node)
        if dotted not in {"threading.Lock", "asyncio.Lock", "threading.RLock"}:
            if dotted not in {"queue.Queue", "asyncio.Queue"}:
                return
            if any("queue" in name or "admission" in name for name in names):
                self._add(f"lifecycle admission queue: {dotted}")
            return
        if any("lock" in name or "lifecycle" in name for name in names):
            self._add(f"lifecycle lock: {dotted}")

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_policy_module:
            dotted = _dotted_name(node.func) or ""
            name = _call_name(node)
            if name == "query_catalog":
                keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
                if (
                    not ({"page_size", "limit"} & keywords)
                    or "work_cap" not in keywords
                ):
                    self._add("unbounded cache catalog query")
            if name in {"unlink", "rmdir", "remove"}:
                self._add("direct resource deletion from cache policy")
            if name == "delete_or_prove_absent" and "_materialize_authority_store" in dotted:
                self._add("direct resource deletion from cache policy")
            if name in _AUTHORITY_TRANSITIONS:
                self._add(f"cache policy invokes authority transition: {name}")
            if name == "ProjectionController":
                self._add("compatibility projection authority in cache policy")
        self.generic_visit(node)


def audit_source(source: str, filename: str) -> tuple[str, ...]:
    """Return deterministic AST-only errors for one source module."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        return (f"unreadable source: {error.msg}",)
    visitor = _ArchitectureVisitor(filename)
    visitor.visit(tree)
    return tuple(visitor.findings)


def audit_contract_text(text: str) -> tuple[str, ...]:
    """Reject positive cache-contract promises prohibited by ADR 0001.

    This small semantic text check is reserved for user-facing cache contract
    documentation.  It is not used to count comment tokens in Python sources,
    so code comments/docstrings cannot create architecture false positives.
    """
    findings: list[str] = []
    lowered = " ".join(text.lower().split())
    checks = (
        (
            r"(?:every|all) (?:concurrent )?contender(?:s)? succeed",
            "cache contract promises universal contender success",
        ),
        (
            r"(?:is|are) one (?:cross-resource )?acid transaction",
            "cache contract promises cross-resource ACID",
        ),
        (
            r"benchmark.*\bis a runtime (?:correctness )?deadline",
            "cache contract turns benchmark into runtime correctness deadline",
        ),
        (
            r"global-oldest (?:eviction|ordering) is (?:exact|guaranteed)",
            "cache contract promises exact global-oldest ordering",
        ),
    )
    for pattern, finding in checks:
        if re.search(pattern, lowered):
            findings.append(finding)
    return tuple(findings)


def _verify_static_composition(root: Path) -> tuple[str, ...]:
    """Verify the one-store/one-engine relationship with narrow AST evidence."""
    errors: list[str] = []
    core_source = (root / "src/cacheness/core.py").read_text(encoding="utf-8")
    store_source = (root / "src/cacheness/storage/blob_store.py").read_text(
        encoding="utf-8"
    )
    lifecycle_source = (root / "src/cacheness/storage/lifecycle.py").read_text(
        encoding="utf-8"
    )
    if "from .storage.blob_store import BlobStore" not in core_source:
        errors.append("UnifiedCache does not import the canonical BlobStore")
    if "self.store = BlobStore(" not in core_source:
        errors.append("UnifiedCache does not construct one selected BlobStore")
    if "self._cache_blob_store = self.store" not in core_source:
        errors.append("UnifiedCache lacks its single BlobStore alias")
    if "self.lifecycle = AuthorityLifecycleEngine(" not in store_source:
        errors.append("BlobStore does not construct AuthorityLifecycleEngine")
    lifecycle_tree = ast.parse(lifecycle_source, filename="src/cacheness/storage/lifecycle.py")
    engines = [
        node.name
        for node in lifecycle_tree.body
        if isinstance(node, ast.ClassDef) and node.name.endswith("LifecycleEngine")
    ]
    if engines != ["AuthorityLifecycleEngine"]:
        errors.append(f"lifecycle module has unexpected engine classes: {engines!r}")
    return tuple(errors)


def _validate_manifest(root: Path) -> tuple[str, ...]:
    """Ensure every fixed path/node is valid before pytest receives it."""
    errors: list[str] = []
    for node in (*PHASE6_CONTRACT_NODES, *RETAINED_LIFECYCLE_NODES):
        path = node.split("::", maxsplit=1)[0]
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            errors.append(f"manifest node is not repository-relative: {node}")
        elif not (root / candidate).is_file():
            errors.append(f"manifest test is missing: {node}")
    return tuple(errors)


def _run_pytest(root: Path, nodes: Iterable[str], *, label: str) -> tuple[bool, str]:
    """Run one known finite test group and convert faults into evidence."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *nodes,
        "-o",
        "log_cli=false",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=PYTEST_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        output = error.output or ""
        return False, f"{label}: timed out after {PYTEST_TIMEOUT_SECONDS} seconds\n{output}"
    except OSError as error:
        return False, f"{label}: could not start pytest: {error}"

    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        return False, f"{label}: pytest exited {completed.returncode}\n{output}"
    return True, f"{label}: {output}"


def verify_repository(root: Path) -> tuple[bool, tuple[str, ...]]:
    """Run the finite source and behavior checks for an explicit repository root."""
    errors: list[str] = []
    root = root.resolve()
    errors.extend(_validate_manifest(root))
    for relative_path in ARCHITECTURE_MODULES:
        path = root / relative_path
        try:
            errors.extend(audit_source(path.read_text(encoding="utf-8"), relative_path))
        except OSError as error:
            errors.append(f"architecture source unreadable: {relative_path}: {error}")
    try:
        errors.extend(_verify_static_composition(root))
    except (OSError, SyntaxError) as error:
        errors.append(f"one-engine static contract unreadable: {error}")
    for relative_path in DOCUMENTATION_MODULES:
        path = root / relative_path
        try:
            errors.extend(audit_contract_text(path.read_text(encoding="utf-8")))
        except OSError as error:
            errors.append(f"cache contract unreadable: {relative_path}: {error}")

    for requirement, nodes in CACH_REQUIREMENT_NODES.items():
        passed, evidence = _run_pytest(root, nodes, label=requirement)
        if not passed:
            errors.append(evidence)
    for regression, nodes in FIXED_REGRESSION_NODES.items():
        passed, evidence = _run_pytest(root, nodes, label=regression)
        if not passed:
            errors.append(evidence)
    passed, evidence = _run_pytest(
        root,
        (
            "tests/test_catalog_projection.py::"
            "test_json_projection_rejects_incompatible_derived_documents",
        ),
        label=STRICT_PROJECTION_LABEL,
    )
    if not passed:
        errors.append(evidence)
    passed, evidence = _run_pytest(
        root, RETAINED_LIFECYCLE_NODES, label="retained Phase 3-5 contracts"
    )
    if not passed:
        errors.append(evidence)
    return not errors, tuple(errors)


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 6 verifier from any current working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args(argv)

    passed, errors = verify_repository(args.repo_root)
    print("Phase 6 fixed contract verifier")
    for requirement in CACH_REQUIREMENT_NODES:
        requirement_passed = not any(error.startswith(f"{requirement}:") for error in errors)
        print(f"{requirement}: {'PASS' if requirement_passed else 'see diagnostics'}")
    for regression in FIXED_REGRESSION_NODES:
        regression_passed = not any(error.startswith(f"{regression}:") for error in errors)
        print(f"{regression}: {'PASS' if regression_passed else 'see diagnostics'}")
    projection_passed = not any(
        error.startswith(f"{STRICT_PROJECTION_LABEL}:") for error in errors
    )
    print(
        f"{STRICT_PROJECTION_LABEL}: "
        + ("PASS" if projection_passed else "see diagnostics")
    )
    print(f"Remote evidence: {REMOTE_EVIDENCE_LABEL}")
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("Retained Phase 3-5 lifecycle contracts: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
