"""Deterministic quality gates for the Phase 1 compatibility baseline."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
VALIDATION_FILE = (
    PROJECT_ROOT
    / ".planning"
    / "phases"
    / "01-compatibility-and-security-baseline"
    / "01-VALIDATION.md"
)
HANDLERS_FILE = PROJECT_ROOT / "src" / "cacheness" / "handlers.py"
CATALOG_FILE = PROJECT_ROOT / "src" / "cacheness" / "storage" / "catalog.py"
BLOB_STORE_FILE = PROJECT_ROOT / "src" / "cacheness" / "storage" / "blob_store.py"

WAVE_ZERO_FILES = (
    "tests/test_public_api_contract.py",
    "tests/test_stored_compatibility.py",
    "tests/test_filesystem_containment.py",
    "tests/test_legacy_array_security.py",
    "tests/test_query_meta_security.py",
    "tests/test_security_documentation.py",
)

PHASE_CREATED_PYTHON_FILES = (
    "src/cacheness/storage/path_security.py",
    "src/cacheness/storage/guarded_handler_io.py",
    "src/cacheness/storage/clear_recovery.py",
    "src/cacheness/query_validation.py",
    "tests/test_public_api_contract.py",
    "tests/test_stored_compatibility.py",
    "tests/test_filesystem_containment.py",
    "tests/test_clear_recovery.py",
    "tests/test_legacy_array_security.py",
    "tests/test_query_meta_security.py",
    "tests/test_security_documentation.py",
    "tests/test_phase1_quality_gates.py",
)


def _module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _method_node(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(_module_tree(path)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == method_name:
                    return method
    raise AssertionError(f"Missing {class_name}.{method_name} in {path}")


def _function_node(path: Path, function_name: str) -> ast.FunctionDef:
    for node in ast.walk(_module_tree(path)):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"Missing {function_name} in {path}")


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _contains_executable_metadata_evaluator(node: ast.AST) -> bool:
    return any(
        isinstance(candidate, ast.Call) and _call_name(candidate) in {"eval", "exec"}
        for candidate in ast.walk(node)
    )


def _contains_permissive_numpy_load(node: ast.AST) -> bool:
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Call) or _call_name(candidate) != "load":
            continue
        if not isinstance(candidate.func, ast.Attribute):
            continue
        if not isinstance(candidate.func.value, ast.Name) or candidate.func.value.id != "np":
            continue
        allow_pickle = next(
            (
                keyword.value
                for keyword in candidate.keywords
                if keyword.arg == "allow_pickle"
            ),
            None,
        )
        if not isinstance(allow_pickle, ast.Constant) or allow_pickle.value is not False:
            return True
    return False


def _contains_query_field_interpolation(node: ast.AST) -> bool:
    field_names = {"field", "filter_key", "key"}

    def is_caller_field_value(value: ast.AST) -> bool:
        if isinstance(value, ast.Name):
            return value.id in field_names
        if isinstance(value, ast.Attribute):
            return value.attr in field_names or is_caller_field_value(value.value)
        if isinstance(value, ast.Subscript):
            return is_caller_field_value(value.value) or is_caller_field_value(
                value.slice
            )
        if isinstance(value, ast.Slice):
            return any(
                part is not None and is_caller_field_value(part)
                for part in (value.lower, value.upper, value.step)
            )
        return False

    for candidate in ast.walk(node):
        if isinstance(candidate, ast.JoinedStr) and any(
            isinstance(value, ast.FormattedValue)
            and is_caller_field_value(value.value)
            for value in candidate.values
        ):
            return True
        if isinstance(candidate, ast.BinOp) and isinstance(candidate.op, (ast.Add, ast.Mod)):
            if is_caller_field_value(candidate.left) or is_caller_field_value(
                candidate.right
            ):
                return True
    return False


def _contains_direct_print(node: ast.AST) -> bool:
    return any(
        isinstance(candidate, ast.Call) and _call_name(candidate) == "print"
        for candidate in ast.walk(node)
    )


def _ruff_findings() -> list[dict[str, object]]:
    ruff_executable = Path(sys.executable).parent / (
        "ruff.exe" if os.name == "nt" else "ruff"
    )
    if not ruff_executable.is_file():
        pytest.fail(
            "Ruff is unavailable from the running test environment: "
            f"{ruff_executable}"
        )
    result = subprocess.run(
        [
            str(ruff_executable),
            "check",
            "src",
            "tests",
            "--output-format",
            "json",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        pytest.fail(f"Ruff did not return a valid result: {result.stderr}")
    try:
        findings = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(f"Ruff did not return JSON: {exc}")
    if not isinstance(findings, list):
        pytest.fail("Ruff JSON result must be a list of findings")
    return findings


def test_every_wave_zero_file_exists() -> None:
    """The phase cannot close with an uncreated Wave 0 test dependency."""
    missing = [path for path in WAVE_ZERO_FILES if not (PROJECT_ROOT / path).is_file()]
    assert not missing


def test_ruff_baseline_is_bounded_and_phase_created_python_is_clean() -> None:
    """Ruff must run non-vacuously without permitting new phase lint debt."""
    findings = _ruff_findings()
    assert len(findings) <= 123
    phase_files = {(PROJECT_ROOT / path).resolve() for path in PHASE_CREATED_PYTHON_FILES}
    phase_findings = [
        finding
        for finding in findings
        if Path(str(finding["filename"])).resolve() in phase_files
    ]
    assert not phase_findings


def test_legacy_reader_has_no_executable_metadata_evaluator() -> None:
    """The sentinel is non-vacuous and the bounded reader remains non-executing."""
    assert _contains_executable_metadata_evaluator(ast.parse("eval(shape_text)"))
    legacy_reader = _method_node(HANDLERS_FILE, "ArrayHandler", "_read_blosc2_array")
    assert not _contains_executable_metadata_evaluator(legacy_reader)


def test_ordinary_array_loading_cannot_enable_pickle() -> None:
    """The sentinel detects permissive loads and ArrayHandler keeps NPZ strict."""
    assert _contains_permissive_numpy_load(ast.parse("np.load(path, allow_pickle=True)"))
    array_get = _method_node(HANDLERS_FILE, "ArrayHandler", "get")
    assert not _contains_permissive_numpy_load(array_get)


def test_catalog_query_boundary_has_no_caller_field_interpolation() -> None:
    """The sentinel rejects caller-field interpolation at the live query boundary."""
    assert _contains_query_field_interpolation(ast.parse('query = f"$.{field}"'))
    assert _contains_query_field_interpolation(
        ast.parse('query = f"$.{request.field}"')
    )
    assert _contains_query_field_interpolation(
        ast.parse('query = f"$.{filters[field]}"')
    )
    catalog_validation = _function_node(CATALOG_FILE, "validate_catalog_query")
    catalog_query = _method_node(BLOB_STORE_FILE, "BlobStore", "query_catalog")
    assert not _contains_query_field_interpolation(catalog_validation)
    assert not _contains_query_field_interpolation(catalog_query)


def test_blob_store_has_no_direct_print_failure_path() -> None:
    """The sentinel catches direct printing and the BlobStore remains logged."""
    assert _contains_direct_print(ast.parse('def fetch():\n    print("failure")'))
    blob_store = next(
        node
        for node in ast.walk(_module_tree(BLOB_STORE_FILE))
        if isinstance(node, ast.ClassDef) and node.name == "BlobStore"
    )
    assert not _contains_direct_print(blob_store)


def test_validation_artifact_records_terminal_approval_and_gap_wave_history() -> None:
    """Terminal approval is executable while the temporary gap state stays documented."""
    validation = VALIDATION_FILE.read_text(encoding="utf-8")
    frontmatter = validation.split("---", 2)[1]
    assert "status: validated" in frontmatter
    assert "nyquist_compliant: true" in frontmatter
    assert "wave_0_complete: true" in frontmatter
    assert "**Approval:** approved 2026-08-30" in validation
    assert "During Waves 8-9, the only accepted validation state" in validation
    assert "`status: draft`, `nyquist_compliant: false`, `wave_0_complete: false`" in validation
    assert "`Approval: pending`" in validation
    assert "01-13" in validation
    assert "01-14" in validation
    assert "01-15" in validation
    for finding in ("CR-01", "CR-02", "CR-03", "CR-04", "CR-05", "CR-06"):
        assert finding in validation
    for threat in ("T-01-38..T-01-40", "T-01-41..T-01-45", "T-01-46..T-01-48"):
        assert threat in validation
    assert "STOR-03, STOR-04, STOR-05, STOR-06, and CACH-03 remain **INCOMPLETE**" in validation
    assert "Executors do not modify `01-REVIEW.md` or `01-REVIEW-FIX.md`" in validation
    assert "T-01-09..T-01-11, T-01-28..T-01-36" in validation
    assert "T-01-20..T-01-23" in validation
    assert "T-01-01..T-01-07, T-01-12, T-01-37" in validation
    assert "T-01-13..T-01-16" in validation
    assert "T-01-17..T-01-19" in validation
    assert "T-01-24" in validation
    assert "Phase-gate threats T-01-25..T-01-27" in validation
    for path in WAVE_ZERO_FILES:
        assert f"- [x] `{path}`" in validation
    assert validation.count("✅ green") >= 6
    assert re.search(r"Full pytest: \d+ passed, \d+ skipped", validation)
    assert re.search(r"Ruff: \d+ findings", validation)
