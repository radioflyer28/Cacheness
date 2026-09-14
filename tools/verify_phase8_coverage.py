#!/usr/bin/env python3
"""Validate and capture the explicit, branch-aware Phase 8 coverage ratchet.

Normal verification only reads Coverage.py evidence and the checked baseline.
Baseline capture is a separately named, review-justified operation that first
proves every named Phase 8 lifecycle and cache-policy selector collects and
passes.  This tool deliberately compares raw counts and their derived rates;
Coverage.py's combined display is not a release contract.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCHEMA_VERSION = 1
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_SAFE_GIT_REF_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
DEFAULT_BASE_REF = "HEAD^"

# These are the released architecture's lifecycle authority, payload, and
# cache-policy sources.  The set is literal so a deleted critical path cannot
# quietly shrink the aggregate measured by the ratchet.
CRITICAL_SOURCE_FILES = (
    "src/cacheness/cache_policy.py",
    "src/cacheness/core.py",
    "src/cacheness/storage/blob_store.py",
    "src/cacheness/storage/lifecycle.py",
    "src/cacheness/storage/lifecycle_authority.py",
    "src/cacheness/storage/memory_lifecycle_authority.py",
    "src/cacheness/storage/sqlite_lifecycle_authority.py",
    "src/cacheness/storage/backends/postgresql_lifecycle_authority.py",
    "src/cacheness/storage/reconciliation.py",
    "src/cacheness/storage/composition.py",
    "src/cacheness/storage/guarded_handler_io.py",
    "src/cacheness/storage/obstore_generation_io.py",
    "src/cacheness/storage/transport_evidence.py",
)

# Keep every Plan 04 proof node literal and AST validated.  The four named
# PostgreSQL families are intentionally represented by their concrete tests,
# not an easy-to-bypass `-k` expression or runtime discovery.
NAMED_SELECTORS = (
    "tests/test_phase8_lifecycle_coverage.py::test_postgresql_error_classification_preserves_typed_progress_outcomes",
    "tests/test_phase8_lifecycle_coverage.py::test_postgresql_error_classification_handles_driver_classes_and_redacts_unknowns",
    "tests/test_phase8_lifecycle_coverage.py::test_postgresql_replay_accepts_only_identical_operation_and_proof",
    "tests/test_phase8_lifecycle_coverage.py::test_postgresql_pagination_bounds_work_and_keeps_unemitted_rows_reachable",
    "tests/test_phase8_lifecycle_coverage.py::test_postgresql_pagination_reconciliation_cursor_reaches_each_emitted_row",
    "tests/test_phase8_lifecycle_coverage.py::test_postgresql_transaction_rollback_preserves_no_partial_commit",
    "tests/test_phase8_lifecycle_coverage.py::test_lifecycle_integrity_before_handler_read",
    "tests/test_phase8_lifecycle_coverage.py::test_lifecycle_ambiguous_publication_settles_only_exact_locator_identity",
    "tests/test_phase8_lifecycle_coverage.py::test_lifecycle_cleanup_debt_reconciliation_retires_only_exact_work",
    "tests/test_phase8_lifecycle_coverage.py::test_lifecycle_bounded_clear_preserves_changed_generations",
    "tests/test_phase8_cache_policy_coverage.py::test_cache_policy_preserves_declared_blobstore_lookup_cause",
    "tests/test_phase8_cache_policy_coverage.py::test_cache_policy_maps_only_documented_lookup_failures",
    "tests/test_phase8_cache_policy_coverage.py::test_cache_policy_rejects_undeclared_lookup_failures",
    "tests/test_phase8_cache_policy_coverage.py::test_cache_policy_invalidation_uses_exact_blobstore_result_accounting",
    "tests/test_phase8_cache_policy_coverage.py::test_cache_policy_invalidation_preserves_conflict_and_backend_causes",
    "tests/test_phase8_cache_policy_coverage.py::test_cache_policy_stale_continuation_restarts_through_the_bounded_path",
    "tests/test_phase8_cache_policy_coverage.py::test_cache_policy_maintenance_results_are_immutable_and_validate_before_io",
    "tests/test_phase8_cache_policy_coverage.py::test_cache_policy_statistics_are_frozen_and_storage_free",
)

POSTGRESQL_SELECTOR_FAMILIES = (
    "postgresql_error_classification",
    "postgresql_replay",
    "postgresql_pagination",
    "postgresql_transaction_rollback",
)

NON_LIVE_MARKER_EXPRESSION = "not (live_postgresql or live_aws_s3 or live_remote)"
DEFAULT_MEASUREMENT_COMMAND = (
    "uv run --isolated --all-extras --group dev --frozen pytest "
    "-q -o log_cli=false "
    f"-m '{NON_LIVE_MARKER_EXPRESSION}' "
    "--cov=cacheness --cov-branch "
    "--cov-report=json:build/phase8/coverage.json "
    "--cov-report=xml:build/phase8/coverage.xml"
)

# The fixed scope represents the complete Phase 8 lifecycle, cache-policy,
# qualification, and packaging surface already delivered by prior plans.  New
# changed Python files are added separately from a safely parsed Git diff.
RUFF_CRITICAL_PATHS = (
    *CRITICAL_SOURCE_FILES,
    "tests/test_phase8_lifecycle_coverage.py",
    "tests/test_phase8_cache_policy_coverage.py",
    "tests/test_phase8_coverage_gate.py",
    "tests/test_phase8_release_tracer.py",
    "tests/packaging/test_wheel_matrix.py",
    "tests/qualification/test_phase8_platform.py",
    "tests/qualification/test_phase8_evidence.py",
    "tests/qualification/test_phase8_live_workflow.py",
    "tools/verify_phase8_coverage.py",
)


class CoverageGateError(ValueError):
    """Raised when coverage evidence, baseline data, or scope is untrustworthy."""


@dataclass(frozen=True)
class CoverageCounts:
    """Raw Coverage.py counts and their independently derived rates."""

    covered_statements: int
    total_statements: int
    covered_branches: int
    total_branches: int

    def __post_init__(self) -> None:
        for name, value in (
            ("covered_statements", self.covered_statements),
            ("total_statements", self.total_statements),
            ("covered_branches", self.covered_branches),
            ("total_branches", self.total_branches),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CoverageGateError(f"{name} must be a non-negative integer")
        if self.total_statements <= 0:
            raise CoverageGateError("total_statements must be greater than zero")
        if self.total_branches <= 0:
            raise CoverageGateError("total_branches must be greater than zero")
        if self.covered_statements > self.total_statements:
            raise CoverageGateError("covered_statements exceeds total_statements")
        if self.covered_branches > self.total_branches:
            raise CoverageGateError("covered_branches exceeds total_branches")

    @property
    def statement_rate(self) -> float:
        """Return the raw statement fraction without Coverage.py display rounding."""

        return self.covered_statements / self.total_statements

    @property
    def branch_rate(self) -> float:
        """Return the raw branch fraction without Coverage.py display rounding."""

        return self.covered_branches / self.total_branches

    def to_dict(self) -> dict[str, int | float]:
        """Serialize raw counts plus reproducible derived rates."""

        return {
            "covered_statements": self.covered_statements,
            "total_statements": self.total_statements,
            "covered_branches": self.covered_branches,
            "total_branches": self.total_branches,
            "statement_rate": self.statement_rate,
            "branch_rate": self.branch_rate,
        }


@dataclass(frozen=True)
class CoverageEvidence:
    """The repository-total and literal critical-scope coverage aggregates."""

    repository: CoverageCounts
    critical: CoverageCounts

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        """Render the two independently measured scopes for a baseline."""

        return {
            "repository": self.repository.to_dict(),
            "critical": self.critical.to_dict(),
        }


def _read_json(path: Path, *, canonical: bool = False) -> Mapping[str, Any]:
    """Read a JSON object, rejecting duplicate keys and non-canonical baselines."""

    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CoverageGateError(f"cannot read JSON evidence {path}: {error}") from error

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CoverageGateError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CoverageGateError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CoverageGateError(f"JSON evidence must be an object: {path}")
    if canonical and raw != _canonical_json(payload):
        raise CoverageGateError(f"baseline JSON is not canonical: {path}")
    return payload


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    """Return the one reviewed on-disk encoding for checked baseline evidence."""

    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CoverageGateError(f"{name} must be an object")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CoverageGateError(f"{name} must be an integer")
    return value


def _counts_from_coverage_summary(summary: object, *, name: str) -> CoverageCounts:
    data = _mapping(summary, name=name)
    return CoverageCounts(
        covered_statements=_integer(
            data.get("covered_lines"), name=f"{name}.covered_lines"
        ),
        total_statements=_integer(
            data.get("num_statements"), name=f"{name}.num_statements"
        ),
        covered_branches=_integer(
            data.get("covered_branches"), name=f"{name}.covered_branches"
        ),
        total_branches=_integer(data.get("num_branches"), name=f"{name}.num_branches"),
    )


def parse_coverage_report(path: Path) -> CoverageEvidence:
    """Validate branch-aware Coverage.py JSON and aggregate its fixed modules."""

    report = _read_json(path)
    meta = _mapping(report.get("meta"), name="coverage meta")
    if meta.get("branch_coverage") is not True:
        raise CoverageGateError("coverage meta.branch_coverage must be true")
    repository = _counts_from_coverage_summary(
        report.get("totals"), name="coverage totals"
    )
    files = _mapping(report.get("files"), name="coverage files")

    critical_counts: list[CoverageCounts] = []
    for relative_path in CRITICAL_SOURCE_FILES:
        file_entry = files.get(relative_path)
        if file_entry is None:
            raise CoverageGateError(
                f"coverage report is missing critical source file: {relative_path}"
            )
        entry = _mapping(file_entry, name=f"coverage files.{relative_path}")
        critical_counts.append(
            _counts_from_coverage_summary(
                entry.get("summary"), name=f"coverage summary for {relative_path}"
            )
        )

    critical = CoverageCounts(
        covered_statements=sum(item.covered_statements for item in critical_counts),
        total_statements=sum(item.total_statements for item in critical_counts),
        covered_branches=sum(item.covered_branches for item in critical_counts),
        total_branches=sum(item.total_branches for item in critical_counts),
    )
    return CoverageEvidence(repository=repository, critical=critical)


def _counts_from_baseline(value: object, *, name: str) -> CoverageCounts:
    data = _mapping(value, name=name)
    expected_keys = {
        "covered_statements",
        "total_statements",
        "covered_branches",
        "total_branches",
        "statement_rate",
        "branch_rate",
    }
    if set(data) != expected_keys:
        raise CoverageGateError(
            f"{name} must contain exactly {sorted(expected_keys)}, got {sorted(data)}"
        )
    counts = CoverageCounts(
        covered_statements=_integer(
            data["covered_statements"], name=f"{name}.covered_statements"
        ),
        total_statements=_integer(
            data["total_statements"], name=f"{name}.total_statements"
        ),
        covered_branches=_integer(
            data["covered_branches"], name=f"{name}.covered_branches"
        ),
        total_branches=_integer(data["total_branches"], name=f"{name}.total_branches"),
    )
    for field, derived in (
        ("statement_rate", counts.statement_rate),
        ("branch_rate", counts.branch_rate),
    ):
        observed = data[field]
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise CoverageGateError(f"{name}.{field} must be a number")
        if not math.isfinite(observed) or not math.isclose(
            observed, derived, abs_tol=1e-12, rel_tol=0.0
        ):
            raise CoverageGateError(f"{name}.{field} is not derived from raw counts")
    return counts


def _top_level_test_names(path: Path) -> set[str]:
    """Return defined test functions without importing an untrusted test module."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise CoverageGateError(
            f"cannot parse selector source {path}: {error}"
        ) from error
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def validate_named_selectors(
    selectors: Sequence[str], root: Path = REPOSITORY_ROOT
) -> None:
    """Reject selector drift, duplicates, unsafe paths, and absent test functions."""

    if tuple(selectors) != NAMED_SELECTORS:
        missing = sorted(set(NAMED_SELECTORS).difference(selectors))
        unexpected = sorted(set(selectors).difference(NAMED_SELECTORS))
        details = []
        if missing:
            details.append(f"missing selectors: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected selectors: {', '.join(unexpected)}")
        if len(set(selectors)) != len(selectors):
            details.append("duplicate selectors")
        raise CoverageGateError(
            "named selector inventory changed: " + "; ".join(details)
        )

    for family in POSTGRESQL_SELECTOR_FAMILIES:
        if not any(family in selector for selector in selectors):
            raise CoverageGateError(f"missing PostgreSQL selector family: {family}")
    for selector in selectors:
        if selector.count("::") != 1:
            raise CoverageGateError(
                f"selector must have one path::test split: {selector}"
            )
        relative_path, test_name = selector.split("::", 1)
        path = PurePosixPath(relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not relative_path.startswith("tests/")
            or not test_name.isidentifier()
            or not test_name.startswith("test_")
        ):
            raise CoverageGateError(f"unsafe named selector: {selector}")
        source_path = root / Path(path)
        if not source_path.is_file():
            raise CoverageGateError(f"named selector source is absent: {selector}")
        if test_name not in _top_level_test_names(source_path):
            raise CoverageGateError(f"named selector test is absent: {selector}")


def _environment_identity() -> dict[str, str]:
    """Capture enough context to explain a measured ratchet without a claim of parity."""

    return {
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _source_revision(root: Path = REPOSITORY_ROOT) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip()
    if result.returncode or not _REVISION_PATTERN.fullmatch(revision):
        raise CoverageGateError(
            "could not determine exact 40-character source revision"
        )
    return revision


def load_baseline(path: Path) -> tuple[CoverageEvidence, Mapping[str, Any]]:
    """Load a canonical baseline and validate all evidence-bearing fields."""

    baseline = _read_json(path, canonical=True)
    expected_keys = {
        "command",
        "critical",
        "environment",
        "justification",
        "named_selectors",
        "repository",
        "schema_version",
        "source_revision",
    }
    if set(baseline) != expected_keys:
        raise CoverageGateError(
            "baseline schema keys differ from the reviewed schema: "
            f"expected {sorted(expected_keys)}, got {sorted(baseline)}"
        )
    if baseline["schema_version"] != BASELINE_SCHEMA_VERSION:
        raise CoverageGateError("baseline schema_version is unsupported")
    revision = baseline["source_revision"]
    if not isinstance(revision, str) or not _REVISION_PATTERN.fullmatch(revision):
        raise CoverageGateError("baseline source_revision must be a 40-character hash")
    command = baseline["command"]
    justification = baseline["justification"]
    if not isinstance(command, str) or not command.strip():
        raise CoverageGateError("baseline command must be a non-empty string")
    if not isinstance(justification, str) or not justification.strip():
        raise CoverageGateError("baseline justification must be a non-empty string")
    environment = _mapping(baseline["environment"], name="baseline environment")
    if set(environment) != {"implementation", "platform", "python"} or any(
        not isinstance(value, str) or not value.strip()
        for value in environment.values()
    ):
        raise CoverageGateError("baseline environment is incomplete")
    selectors = baseline["named_selectors"]
    if not isinstance(selectors, list) or not all(
        isinstance(selector, str) for selector in selectors
    ):
        raise CoverageGateError("baseline named_selectors must be a string list")
    validate_named_selectors(selectors)
    return (
        CoverageEvidence(
            repository=_counts_from_baseline(baseline["repository"], name="repository"),
            critical=_counts_from_baseline(baseline["critical"], name="critical"),
        ),
        baseline,
    )


def _compare_scope(
    current: CoverageCounts, baseline: CoverageCounts, *, scope: str
) -> list[str]:
    errors: list[str] = []
    for dimension, current_value, baseline_value in (
        (
            "statement covered count",
            current.covered_statements,
            baseline.covered_statements,
        ),
        (
            "statement total",
            current.total_statements,
            baseline.total_statements,
        ),
        ("branch covered count", current.covered_branches, baseline.covered_branches),
        ("branch total", current.total_branches, baseline.total_branches),
        ("statement rate", current.statement_rate, baseline.statement_rate),
        ("branch rate", current.branch_rate, baseline.branch_rate),
    ):
        if current_value < baseline_value:
            errors.append(
                f"{scope} {dimension} regressed: {current_value} < {baseline_value}"
            )
    return errors


def compare_to_baseline(
    current: CoverageEvidence | Mapping[str, object],
    baseline: CoverageEvidence | Mapping[str, object],
) -> None:
    """Fail any total/critical statement or branch raw-count/rate regression."""

    def evidence(value: CoverageEvidence | Mapping[str, object]) -> CoverageEvidence:
        if isinstance(value, CoverageEvidence):
            return value
        return CoverageEvidence(
            repository=_counts_from_baseline(
                value.get("repository"), name="repository"
            ),
            critical=_counts_from_baseline(value.get("critical"), name="critical"),
        )

    current_evidence = evidence(current)
    baseline_evidence = evidence(baseline)
    errors = [
        *_compare_scope(
            current_evidence.repository,
            baseline_evidence.repository,
            scope="repository",
        ),
        *_compare_scope(
            current_evidence.critical, baseline_evidence.critical, scope="critical"
        ),
    ]
    if errors:
        raise CoverageGateError("; ".join(errors))


def verify(report_path: Path, baseline_path: Path) -> CoverageEvidence:
    """Read and compare evidence without modifying its checked baseline."""

    current = parse_coverage_report(report_path)
    baseline, _ = load_baseline(baseline_path)
    compare_to_baseline(current, baseline)
    return current


def _safe_relative_python_path(relative_path: str) -> str:
    if "\x00" in relative_path:
        raise CoverageGateError("unsafe changed path contains a NUL byte")
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not relative_path.endswith(".py")
        or not relative_path.startswith(("src/", "tests/", "tools/"))
    ):
        raise CoverageGateError(f"unsafe changed Python path: {relative_path}")
    candidate = (REPOSITORY_ROOT / Path(path)).resolve()
    root = REPOSITORY_ROOT.resolve()
    if root not in (candidate, *candidate.parents) or not candidate.is_file():
        raise CoverageGateError(
            f"changed Python path is not an owned file: {relative_path}"
        )
    return path.as_posix()


def build_ruff_scope(changed_paths: Iterable[str]) -> tuple[str, ...]:
    """Build direct Ruff argv paths from safe changed files plus literal scope."""

    scope = set(RUFF_CRITICAL_PATHS)
    for critical_path in RUFF_CRITICAL_PATHS:
        _safe_relative_python_path(critical_path)
    for changed_path in changed_paths:
        if not isinstance(changed_path, str):
            raise CoverageGateError("changed path must be text")
        if changed_path.endswith(".py"):
            scope.add(_safe_relative_python_path(changed_path))
        elif ".." in PurePosixPath(changed_path).parts or changed_path.startswith("/"):
            raise CoverageGateError(f"unsafe changed path: {changed_path}")
    return tuple(sorted(scope))


def changed_python_paths(
    base_ref: str, root: Path = REPOSITORY_ROOT
) -> tuple[str, ...]:
    """Obtain a NUL-delimited merge-base diff without shell interpolation."""

    if base_ref != DEFAULT_BASE_REF and (
        not _SAFE_GIT_REF_PATTERN.fullmatch(base_ref) or base_ref.startswith("-")
    ):
        raise CoverageGateError(f"unsafe base revision: {base_ref}")
    merge_base = subprocess.run(
        ["git", "merge-base", base_ref, "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if merge_base.returncode:
        raise CoverageGateError(
            f"could not determine merge base for {base_ref}: {merge_base.stderr.strip()}"
        )
    revision = merge_base.stdout.strip()
    if not _REVISION_PATTERN.fullmatch(revision):
        raise CoverageGateError("git merge-base did not return an exact commit")
    result = subprocess.run(
        ["git", "diff", "--name-only", "-z", revision, "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise CoverageGateError(
            f"git diff failed: {result.stderr.decode(errors='replace')}"
        )
    try:
        paths = tuple(
            item.decode("utf-8") for item in result.stdout.split(b"\0") if item
        )
    except UnicodeDecodeError as error:
        raise CoverageGateError("git diff returned a non-UTF-8 path") from error
    return tuple(path for path in paths if path.endswith(".py"))


def run_ruff(scope: Sequence[str]) -> None:
    """Run direct lint and format checks using a list argv, never a shell string."""

    for command in (
        [sys.executable, "-m", "ruff", "check", *scope],
        [sys.executable, "-m", "ruff", "format", "--check", *scope],
    ):
        result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
        if result.returncode:
            raise CoverageGateError(f"Ruff command failed: {' '.join(command[:5])} ...")


def _preflight_named_selectors() -> None:
    """Collect and run every reviewed selector before baseline mutation."""

    validate_named_selectors(NAMED_SELECTORS)
    common = [sys.executable, "-m", "pytest", "-q", "-o", "log_cli=false", "-x"]
    for label, command in (
        ("collection", [*common, "--collect-only", *NAMED_SELECTORS]),
        ("execution", [*common, *NAMED_SELECTORS]),
    ):
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        output = f"{result.stdout}\n{result.stderr}".lower()
        if result.returncode or "deselected" in output or "skipped" in output:
            raise CoverageGateError(
                f"named selector {label} did not cleanly pass: {result.stdout}{result.stderr}"
            )


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish canonical baseline bytes only after all capture checks pass."""

    parent = path.parent
    if not parent.is_dir():
        raise CoverageGateError(f"baseline parent is absent: {parent}")
    encoded = _canonical_json(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise CoverageGateError(
            f"could not atomically replace baseline {path}: {error}"
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def capture(
    report_path: Path,
    baseline_path: Path,
    *,
    justification: str,
    command: str,
) -> CoverageEvidence:
    """Preflight named evidence and atomically establish a reviewed first floor."""

    if not justification.strip():
        raise CoverageGateError("baseline capture requires an explicit justification")
    if not command.strip():
        raise CoverageGateError("baseline capture requires the measurement command")
    evidence = parse_coverage_report(report_path)
    _preflight_named_selectors()
    payload: dict[str, Any] = {
        "command": command,
        "critical": evidence.critical.to_dict(),
        "environment": _environment_identity(),
        "justification": justification,
        "named_selectors": list(NAMED_SELECTORS),
        "repository": evidence.repository.to_dict(),
        "schema_version": BASELINE_SCHEMA_VERSION,
        "source_revision": _source_revision(),
    }
    _atomic_write(baseline_path, payload)
    return evidence


def _path_argument(value: str) -> Path:
    candidate = (REPOSITORY_ROOT / value).resolve()
    root = REPOSITORY_ROOT.resolve()
    if root not in (candidate, *candidate.parents):
        raise argparse.ArgumentTypeError("path must stay inside the repository")
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    """Verify a report or explicitly capture a new canonical baseline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=_path_argument)
    parser.add_argument("--baseline", required=True, type=_path_argument)
    parser.add_argument(
        "--capture",
        action="store_true",
        help="preflight selectors and replace baseline",
    )
    parser.add_argument(
        "--justification", help="review reason required only for explicit capture"
    )
    parser.add_argument(
        "--command",
        default=DEFAULT_MEASUREMENT_COMMAND,
        help="measurement command recorded in a baseline capture",
    )
    parser.add_argument(
        "--ruff", action="store_true", help="run direct lint and format checks"
    )
    parser.add_argument(
        "--base-ref",
        default=DEFAULT_BASE_REF,
        help="Git ref used to construct changed scope",
    )
    arguments = parser.parse_args(argv)

    try:
        if arguments.capture:
            if arguments.justification is None:
                raise CoverageGateError(
                    "--capture requires --justification; ordinary verify never writes"
                )
            evidence = capture(
                arguments.report,
                arguments.baseline,
                justification=arguments.justification,
                command=arguments.command,
            )
            mode = "captured"
        else:
            if arguments.justification is not None:
                raise CoverageGateError("--justification is valid only with --capture")
            evidence = verify(arguments.report, arguments.baseline)
            mode = "verified"
        if arguments.ruff:
            run_ruff(build_ruff_scope(changed_python_paths(arguments.base_ref)))
    except CoverageGateError as error:
        print(f"Phase 8 coverage gate failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Phase 8 coverage {mode}: "
        f"repository={evidence.repository.statement_rate:.4%}/{evidence.repository.branch_rate:.4%}; "
        f"critical={evidence.critical.statement_rate:.4%}/{evidence.critical.branch_rate:.4%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
