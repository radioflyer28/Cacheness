"""Enforce the Phase 4 Ruff baseline without grandfathering new debt.

The checked-in baseline captures a stable fingerprint for every diagnostic in
the Python files Phase 4 owns before it edits production code.  Position data
is deliberately excluded: moving a known line is harmless, whereas changing
the path, rule, message, or source line is a regression.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PHASE_DIRECTORY = (
    REPOSITORY_ROOT / ".planning" / "phases" / "04-metadata-composition-and-topology-contracts"
)
BASELINE_PATH = REPOSITORY_ROOT / "tests" / "fixtures" / "phase4_ruff_baseline.json"
SCHEMA_VERSION = 1

# These paths are deliberately retired by the compatibility reset.  Their
# disappearance is allowed, but a surprise deletion anywhere else is scope
# drift that needs an explicit plan amendment.
EXPECTED_RETIRED_PATHS: tuple[str, ...] = (
    "src/cacheness/custom_metadata.py",
    "src/cacheness/storage/backends/base.py",
    "src/cacheness/storage/manifest_repository.py",
)


def _normalise_path(path: Path) -> str:
    """Return a POSIX repository-relative path or reject an escaped path."""
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path escapes repository root: {path}") from error


def _phase_python_paths() -> list[str]:
    """Read the committed Phase 4 plan inventories without a YAML dependency."""
    paths: set[str] = set()
    for plan_path in sorted(PHASE_DIRECTORY.glob("04-??-PLAN.md")):
        text = plan_path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise ValueError(f"plan has no frontmatter: {plan_path.name}")
        frontmatter = text.split("---\n", 2)[1]
        in_files = False
        for line in frontmatter.splitlines():
            if line == "files_modified:":
                in_files = True
                continue
            if in_files and line.startswith("  - "):
                candidate = line.removeprefix("  - ").strip()
                if candidate.endswith(".py"):
                    paths.add(candidate)
                continue
            if in_files and line and not line.startswith(" "):
                in_files = False
    if not paths:
        raise ValueError("Phase 4 plan inventory contains no Python paths")
    return sorted(paths)


def _run_ruff(paths: list[str]) -> list[dict[str, Any]]:
    """Return Ruff's JSON diagnostics for existing paths in the declared scope."""
    existing_paths = [path for path in paths if (REPOSITORY_ROOT / path).is_file()]
    if not existing_paths:
        return []
    result = subprocess.run(
        ["ruff", "check", "--output-format", "json", *existing_paths],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Ruff failed ({result.returncode}): {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Ruff did not emit JSON diagnostics") from error
    if not isinstance(payload, list):
        raise RuntimeError("Ruff JSON output was not a diagnostic list")
    return payload


def _fingerprint(diagnostic: dict[str, Any]) -> dict[str, str]:
    """Create a line-independent, stable representation of one diagnostic."""
    filename = diagnostic.get("filename")
    code = diagnostic.get("code")
    message = diagnostic.get("message")
    if not all(isinstance(value, str) for value in (filename, code, message)):
        raise ValueError("Ruff diagnostic is missing filename, code, or message")
    path = _normalise_path(Path(filename))
    source = diagnostic.get("source")
    if not isinstance(source, str):
        location = diagnostic.get("location")
        row = location.get("row") if isinstance(location, dict) else None
        if type(row) is not int or row < 1:
            raise ValueError("Ruff diagnostic is missing source and a valid location")
        try:
            source = (REPOSITORY_ROOT / path).read_text(encoding="utf-8").splitlines()[
                row - 1
            ]
        except (IndexError, OSError) as error:
            raise ValueError("cannot derive source line for Ruff diagnostic") from error
    payload = {"path": path, "code": code, "message": message, "source": source}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**payload, "fingerprint": fingerprint}


def _validate_paths(name: str, value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"baseline {name} must be a list of paths")
    if len(value) != len(set(value)):
        raise ValueError(f"baseline {name} contains duplicate paths")
    if value != sorted(value):
        raise ValueError(f"baseline {name} must be sorted")
    for path in value:
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"baseline {name} contains unsafe path: {path}")
    return value


def _validate_fingerprint(value: object) -> dict[str, str]:
    """Validate one canonical baseline item and recompute its digest."""
    if not isinstance(value, dict):
        raise ValueError("baseline contains malformed diagnostic")
    path = value.get("path")
    code = value.get("code")
    message = value.get("message")
    source = value.get("source")
    fingerprint = value.get("fingerprint")
    if not all(isinstance(item, str) for item in (path, code, message, source, fingerprint)):
        raise ValueError("baseline diagnostic has invalid fields")
    expected = hashlib.sha256(
        json.dumps(
            {"path": path, "code": code, "message": message, "source": source},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if fingerprint != expected:
        raise ValueError("baseline diagnostic fingerprint does not match its content")
    return {"path": path, "code": code, "message": message, "source": source, "fingerprint": fingerprint}


def _load_baseline() -> dict[str, Any]:
    try:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Phase 4 Ruff baseline: {error}") from error
    if not isinstance(baseline, dict) or baseline.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("malformed or unsupported Phase 4 Ruff baseline")
    for name in ("existing_paths", "new_paths", "expected_retired_paths"):
        baseline[name] = _validate_paths(name, baseline.get(name))
    if set(baseline["existing_paths"]) & set(baseline["new_paths"]):
        raise ValueError("baseline path may not be both existing and new")
    diagnostics = baseline.get("diagnostics")
    if not isinstance(diagnostics, list):
        raise ValueError("baseline diagnostics must be a list")
    validated = [_validate_fingerprint(item) for item in diagnostics]
    if len({json.dumps(item, sort_keys=True) for item in validated}) != len(validated):
        raise ValueError("baseline contains duplicate diagnostics")
    baseline["diagnostics"] = validated
    return baseline


def _capture_baseline() -> dict[str, Any]:
    declared_paths = _phase_python_paths()
    existing_paths = sorted(path for path in declared_paths if (REPOSITORY_ROOT / path).is_file())
    new_paths = sorted(path for path in declared_paths if path not in existing_paths)
    diagnostics = sorted(
        (_fingerprint(item) for item in _run_ruff(existing_paths)),
        key=lambda item: (item["path"], item["fingerprint"]),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostics": diagnostics,
        "existing_paths": existing_paths,
        "expected_retired_paths": sorted(EXPECTED_RETIRED_PATHS),
        "new_paths": new_paths,
    }


def _verify() -> list[str]:
    baseline = _load_baseline()
    errors: list[str] = []
    declared_paths = set(_phase_python_paths())
    baseline_paths = set(baseline["existing_paths"]) | set(baseline["new_paths"])
    retired_paths = set(baseline["expected_retired_paths"])
    if declared_paths != baseline_paths:
        errors.append(
            "declared Phase 4 Python scope drifted from the frozen baseline: "
            f"expected {sorted(baseline_paths)}, found {sorted(declared_paths)}"
        )
    if retired_paths - baseline_paths:
        errors.append("expected retired paths must be declared Phase 4 paths")

    for path in baseline["existing_paths"]:
        if not (REPOSITORY_ROOT / path).is_file() and path not in retired_paths:
            errors.append(f"undeclared deletion of existing Phase 4 path: {path}")
    current = [_fingerprint(item) for item in _run_ruff(sorted(declared_paths))]
    current_by_path: dict[str, Counter[str]] = {}
    for item in current:
        current_by_path.setdefault(item["path"], Counter())[item["fingerprint"]] += 1
    baseline_by_path: dict[str, Counter[str]] = {}
    for item in baseline["diagnostics"]:
        baseline_by_path.setdefault(item["path"], Counter())[item["fingerprint"]] += 1

    for path in baseline["existing_paths"]:
        if not (REPOSITORY_ROOT / path).is_file():
            continue
        unexpected = current_by_path.get(path, Counter()) - baseline_by_path.get(path, Counter())
        if unexpected:
            errors.append(f"new or changed Ruff finding in existing path {path}: {sorted(unexpected)}")
    for path in baseline["new_paths"]:
        findings = current_by_path.get(path, Counter())
        if findings:
            errors.append(f"new Phase 4 path must be Ruff-clean ({path}): {sorted(findings)}")
    undeclared_diagnostics = set(current_by_path) - baseline_paths
    if undeclared_diagnostics:
        errors.append(f"Ruff reported paths outside declared Phase 4 scope: {sorted(undeclared_diagnostics)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        action="store_true",
        help="write the initial Phase 4 baseline from the unedited declared scope",
    )
    arguments = parser.parse_args()
    try:
        if arguments.capture:
            if BASELINE_PATH.exists():
                raise ValueError("refusing to overwrite an existing Phase 4 Ruff baseline")
            baseline = _capture_baseline()
            BASELINE_PATH.write_text(
                json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"captured {len(baseline['diagnostics'])} diagnostics in {BASELINE_PATH}")
            return 0
        errors = _verify()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Phase 4 Ruff delta gate failed: {error}", file=sys.stderr)
        return 1
    if errors:
        print("Phase 4 Ruff delta gate failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("Phase 4 Ruff delta gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
