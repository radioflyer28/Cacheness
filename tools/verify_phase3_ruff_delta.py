"""Enforce the Phase 3 Ruff baseline without grandfathering new debt.

The baseline intentionally fingerprints diagnostics without their positions so a
harmless line movement does not create a false failure.  It does include the
path, rule, message, and source line, so a changed or newly introduced finding
is still rejected deterministically.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "phase3_ruff_baseline.json"
SCHEMA_VERSION = 1

EXISTING_PATHS = [
    "src/cacheness/__init__.py",
    "src/cacheness/config.py",
    "src/cacheness/core.py",
    "src/cacheness/error_handling.py",
    "src/cacheness/metadata.py",
    "src/cacheness/storage/__init__.py",
    "src/cacheness/storage/blob_store.py",
    "src/cacheness/storage/coordination.py",
    "src/cacheness/storage/guarded_handler_io.py",
    "src/cacheness/storage/lifecycle.py",
    "src/cacheness/storage/manifest_repository.py",
    "src/cacheness/storage/path_security.py",
    "src/cacheness/storage/backends/postgresql_backend.py",
    "src/cacheness/storage/reconciliation.py",
    "tests/test_blob_store_atomic_lifecycle.py",
    "tests/test_blob_store_close_contract.py",
    "tests/test_blob_store_concurrency.py",
    "tests/test_blob_store_read_contract.py",
    "tests/test_blob_store_reconciliation.py",
    "tests/test_clear_recovery.py",
    "tests/test_filesystem_containment.py",
    "tests/test_manifest_repository_cas.py",
    "tests/test_public_api_contract.py",
    "verify_platform.py",
]

NEW_PATHS = [
    "benchmarks/lifecycle_authority_benchmark.py",
    "src/cacheness/storage/lifecycle_authority.py",
    "src/cacheness/storage/sqlite_lifecycle_authority.py",
    "src/cacheness/storage/memory_lifecycle_authority.py",
    "tests/_lifecycle_test_support.py",
    "tests/test_lifecycle_authority_contract.py",
    "tests/test_sqlite_lifecycle_authority.py",
    "tests/test_phase3_release_evidence.py",
    "tests/test_phase3_scheduler_retirement.py",
    "tests/test_phase3_windows_contract.py",
    "tests/test_phase3_ruff_delta.py",
    "tests/test_projection_mutation_contract.py",
    "tests/test_unified_cache_lifecycle_authority.py",
    "tools/verify_phase3_ruff_delta.py",
]

EXPECTED_RETIRED_PATHS = [
    "src/cacheness/storage/operation_repository.py",
    "src/cacheness/storage/operation_record.py",
    "src/cacheness/storage/clear_recovery.py",
]


class BaselineError(ValueError):
    """Raised when the checked-in baseline is malformed or has scope drift."""


def normalize_path(path: str) -> str:
    """Return one normalized repository-relative path or reject invalid input."""
    if not isinstance(path, str) or not path.strip():
        raise BaselineError("path must be a non-empty string")
    normalized = path.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise BaselineError(f"path must be repository-relative: {path!r}")
    normalized = candidate.as_posix()
    if normalized in {"", "."}:
        raise BaselineError(f"path must name a file: {path!r}")
    return normalized


def normalize_whitespace(value: object) -> str:
    """Normalize display-only whitespace for stable diagnostic fingerprints."""
    return " ".join(str(value).split())


def _diagnostic_path(diagnostic: dict[str, Any], root: Path | None = None) -> str:
    raw_path = diagnostic.get("path", diagnostic.get("filename"))
    if not isinstance(raw_path, str):
        raise BaselineError("diagnostic path must be a string")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        if root is None:
            raise BaselineError("absolute diagnostic path requires a repository root")
        try:
            candidate = candidate.relative_to(root)
        except ValueError as error:
            raise BaselineError(
                f"diagnostic path is outside repository root: {raw_path!r}"
            ) from error
    return normalize_path(candidate.as_posix())


def diagnostic_fingerprint(diagnostic: dict[str, Any], root: Path | None = None) -> str:
    """Hash the stable diagnostic identity, deliberately excluding line/column."""
    path = _diagnostic_path(diagnostic, root)
    code = diagnostic.get("code")
    message = diagnostic.get("message")
    source = diagnostic.get("source")
    if not all(isinstance(value, str) for value in (code, message, source)):
        raise BaselineError("diagnostic requires string code, message, and source")
    material = "\x1f".join(
        (path, code, normalize_whitespace(message), normalize_whitespace(source))
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _normalized_path_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise BaselineError(f"{field} must be a list of paths")
    normalized = [normalize_path(item) for item in value]
    duplicates = sorted(path for path, count in Counter(normalized).items() if count > 1)
    if duplicates:
        raise BaselineError(f"duplicate path in {field}: {', '.join(duplicates)}")
    return normalized


def validate_baseline(
    baseline: dict[str, Any], *, enforce_phase_scope: bool = False
) -> None:
    """Reject malformed baselines and all declared-scope drift."""
    if not isinstance(baseline, dict):
        raise BaselineError("baseline must be a JSON object")
    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise BaselineError(f"schema_version must be {SCHEMA_VERSION}")

    declared = {
        "existing_paths": _normalized_path_list(
            baseline.get("existing_paths"), "existing_paths"
        ),
        "new_paths": _normalized_path_list(baseline.get("new_paths"), "new_paths"),
        "expected_retired_paths": _normalized_path_list(
            baseline.get("expected_retired_paths"), "expected_retired_paths"
        ),
    }
    all_paths = [path for paths in declared.values() for path in paths]
    duplicates = sorted(path for path, count in Counter(all_paths).items() if count > 1)
    if duplicates:
        raise BaselineError(f"duplicate path across scope lists: {', '.join(duplicates)}")

    if enforce_phase_scope:
        expected = {
            "existing_paths": EXISTING_PATHS,
            "new_paths": NEW_PATHS,
            "expected_retired_paths": EXPECTED_RETIRED_PATHS,
        }
        for field, required_paths in expected.items():
            if declared[field] != required_paths:
                raise BaselineError(f"scope drift in {field}")

    diagnostics = baseline.get("diagnostics")
    if not isinstance(diagnostics, list):
        raise BaselineError("diagnostics must be a list")
    fingerprints: list[str] = []
    allowed_diagnostic_paths = set(
        declared["existing_paths"] + declared["expected_retired_paths"]
    )
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            raise BaselineError("diagnostic entry must be an object")
        fingerprint = diagnostic.get("fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise BaselineError("diagnostic fingerprint must be a SHA-256 hex value")
        if fingerprint != diagnostic_fingerprint(diagnostic):
            raise BaselineError("diagnostic fingerprint does not match its content")
        path = _diagnostic_path(diagnostic)
        if path not in allowed_diagnostic_paths:
            raise BaselineError(f"diagnostic path is outside baseline scope: {path}")
        fingerprints.append(fingerprint)
    duplicates = sorted(
        fingerprint for fingerprint, count in Counter(fingerprints).items() if count > 1
    )
    if duplicates:
        raise BaselineError("duplicate diagnostic fingerprint in baseline")


def resolve_declared_paths(root: Path, baseline: dict[str, Any]) -> list[str]:
    """Return declared files that currently exist, enforcing mandatory scope files."""
    validate_baseline(baseline)
    present: list[str] = []
    for path in baseline["existing_paths"]:
        if not (root / path).is_file():
            raise BaselineError(f"required existing path is missing: {path}")
        present.append(path)
    for field in ("new_paths", "expected_retired_paths"):
        for path in baseline[field]:
            candidate = root / path
            if candidate.exists():
                if not candidate.is_file():
                    raise BaselineError(f"declared path is not a regular file: {path}")
                present.append(path)
    return sorted(present)


def _source_line(root: Path, diagnostic: dict[str, Any]) -> str:
    path = root / _diagnostic_path(diagnostic, root)
    location = diagnostic.get("location")
    if not isinstance(location, dict) or not isinstance(location.get("row"), int):
        raise BaselineError("Ruff diagnostic has no source row")
    try:
        return path.read_text(encoding="utf-8").splitlines()[location["row"] - 1]
    except (IndexError, OSError) as error:
        raise BaselineError(
            f"could not read diagnostic source line for {_diagnostic_path(diagnostic, root)}"
        ) from error


def run_ruff(root: Path, paths: list[str]) -> list[dict[str, Any]]:
    """Run the repository's Ruff executable and attach stable source snippets."""
    ruff = Path(sys.executable).with_name("ruff")
    executable = str(ruff) if ruff.is_file() else shutil.which("ruff")
    if executable is None:
        raise BaselineError("Ruff executable is unavailable")
    completed = subprocess.run(
        [executable, "check", "--output-format", "json", *paths],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise BaselineError(f"Ruff failed: {completed.stderr.strip()}")
    try:
        raw_diagnostics = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise BaselineError("Ruff did not produce JSON diagnostics") from error
    if not isinstance(raw_diagnostics, list):
        raise BaselineError("Ruff JSON output must be a list")

    diagnostics: list[dict[str, Any]] = []
    for diagnostic in raw_diagnostics:
        if not isinstance(diagnostic, dict):
            raise BaselineError("Ruff emitted a malformed diagnostic")
        normalized = dict(diagnostic)
        normalized["path"] = _diagnostic_path(diagnostic, root)
        normalized["source"] = _source_line(root, normalized)
        diagnostics.append(normalized)
    return diagnostics


def unmatched_diagnostics(
    baseline: dict[str, Any], current: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return current diagnostics not covered by the baseline multiset."""
    validate_baseline(baseline)
    available = Counter(item["fingerprint"] for item in baseline["diagnostics"])
    unmatched: list[dict[str, Any]] = []
    for diagnostic in current:
        fingerprint = diagnostic_fingerprint(diagnostic)
        if available[fingerprint]:
            available[fingerprint] -= 1
        else:
            unmatched.append(diagnostic)
    return sorted(
        unmatched,
        key=lambda item: (
            _diagnostic_path(item),
            str(item["code"]),
            normalize_whitespace(item["message"]),
            normalize_whitespace(item["source"]),
        ),
    )


def scope_errors(
    root: Path, baseline: dict[str, Any], current: list[dict[str, Any]]
) -> list[str]:
    """Report required-path and clean-new-file failures deterministically."""
    errors: list[str] = []
    try:
        resolve_declared_paths(root, baseline)
    except BaselineError as error:
        return [str(error)]
    new_paths = set(baseline["new_paths"])
    findings_by_path = Counter(_diagnostic_path(diagnostic, root) for diagnostic in current)
    for path in sorted(new_paths.intersection(findings_by_path)):
        errors.append(f"new path has Ruff findings: {path}")
    return errors


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    """Load and validate the checked-in baseline fixture."""
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineError(f"could not load baseline: {path}") from error
    validate_baseline(baseline, enforce_phase_scope=True)
    return baseline


def write_baseline(path: Path = BASELINE_PATH, root: Path = REPOSITORY_ROOT) -> None:
    """Generate the pre-phase baseline from current existing and retired paths."""
    declared_paths = [
        path
        for path in EXISTING_PATHS + EXPECTED_RETIRED_PATHS
        if (root / path).is_file()
    ]
    diagnostics = run_ruff(root, declared_paths)
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "existing_paths": EXISTING_PATHS,
        "new_paths": NEW_PATHS,
        "expected_retired_paths": EXPECTED_RETIRED_PATHS,
        "diagnostics": [
            {
                "fingerprint": diagnostic_fingerprint(diagnostic),
                "path": _diagnostic_path(diagnostic, root),
                "code": diagnostic["code"],
                "message": diagnostic["message"],
                "source": diagnostic["source"],
            }
            for diagnostic in sorted(
                diagnostics,
                key=lambda item: (
                    _diagnostic_path(item, root),
                    str(item["code"]),
                    normalize_whitespace(item["message"]),
                    normalize_whitespace(item["source"]),
                ),
            )
        ],
    }
    validate_baseline(fixture, enforce_phase_scope=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify(root: Path = REPOSITORY_ROOT, baseline_path: Path = BASELINE_PATH) -> int:
    """Print every unmatched diagnostic and return a shell-friendly status code."""
    baseline = load_baseline(baseline_path)
    paths = resolve_declared_paths(root, baseline)
    current = run_ruff(root, paths)
    errors = scope_errors(root, baseline, current)
    unmatched = unmatched_diagnostics(baseline, current)
    if errors or unmatched:
        for error in errors:
            print(f"ERROR: {error}")
        for diagnostic in unmatched:
            print(
                "UNMATCHED: "
                f"{_diagnostic_path(diagnostic, root)} "
                f"{diagnostic['code']} {normalize_whitespace(diagnostic['message'])} "
                f"[{normalize_whitespace(diagnostic['source'])}]"
            )
        return 1
    print("Phase 3 Ruff delta: no unmatched findings")
    return 0


def main() -> int:
    """Run verification or intentionally regenerate the frozen baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="regenerate the baseline from the declared existing/retired paths",
    )
    arguments = parser.parse_args()
    if arguments.write_baseline:
        write_baseline()
        print(f"Wrote Phase 3 Ruff baseline: {BASELINE_PATH.relative_to(REPOSITORY_ROOT)}")
        return 0
    return verify()


if __name__ == "__main__":
    raise SystemExit(main())
