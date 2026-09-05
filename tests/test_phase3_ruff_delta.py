"""Tests for the exact-scope Phase 3 Ruff delta verifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = REPOSITORY_ROOT / "tools" / "verify_phase3_ruff_delta.py"


def _load_verifier():
    """Load the verifier without requiring ``tools`` to be a package."""
    specification = importlib.util.spec_from_file_location(
        "phase3_ruff_delta_verifier", VERIFIER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load the Phase 3 Ruff delta verifier")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _diagnostic(
    path: str,
    code: str = "F401",
    message: str = "unused import",
    source: str = "import unused",
    row: int = 1,
) -> dict[str, object]:
    """Build a minimal Ruff JSON diagnostic for deterministic unit tests."""
    return {
        "filename": path,
        "code": code,
        "message": message,
        "location": {"row": row, "column": 1},
        "end_location": {"row": row, "column": len(source) + 1},
        "fix": None,
        "source": source,
    }


def _baseline(
    verifier,
    *,
    existing: list[str] | None = None,
    new: list[str] | None = None,
    retired: list[str] | None = None,
    diagnostics: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Create a valid baseline fixture with supplied Phase 3 scope."""
    existing = existing or ["src/existing.py"]
    new = new or ["src/new.py"]
    retired = retired or ["src/retired.py"]
    diagnostics = diagnostics or [_diagnostic("src/existing.py")]
    return {
        "schema_version": 1,
        "existing_paths": existing,
        "new_paths": new,
        "expected_retired_paths": retired,
        "diagnostics": [
            {
                "fingerprint": verifier.diagnostic_fingerprint(diagnostic),
                "path": diagnostic["filename"],
                "code": diagnostic["code"],
                "message": diagnostic["message"],
                "source": diagnostic["source"],
            }
            for diagnostic in diagnostics
        ],
    }


def test_fingerprint_ignores_line_movement_and_normalizes_paths():
    """Line movement and Windows separators do not produce a new finding."""
    verifier = _load_verifier()

    original = _diagnostic("src/module.py", source="import unused", row=1)
    moved = _diagnostic("src\\module.py", source=" import   unused ", row=99)

    assert verifier.diagnostic_fingerprint(original) == verifier.diagnostic_fingerprint(
        moved
    )


def test_baseline_rejects_duplicate_or_malformed_scope_declarations():
    """A baseline must declare one normalized path in exactly one scope list."""
    verifier = _load_verifier()
    baseline = _baseline(verifier, existing=["src/existing.py", "src\\existing.py"])

    with pytest.raises(verifier.BaselineError, match="duplicate"):
        verifier.validate_baseline(baseline)

    malformed = _baseline(verifier)
    malformed["diagnostics"] = [{"fingerprint": "not-a-sha256"}]
    with pytest.raises(verifier.BaselineError, match="diagnostic"):
        verifier.validate_baseline(malformed)


def test_comparison_allows_removed_findings_but_rejects_changed_or_added_ones():
    """Existing findings are a multiset upper bound, never a blanket allowlist."""
    verifier = _load_verifier()
    baseline = _baseline(verifier)
    verifier.validate_baseline(baseline)

    assert verifier.unmatched_diagnostics(baseline, []) == []

    changed = _diagnostic("src/existing.py", message="redefined unused import")
    unmatched = verifier.unmatched_diagnostics(baseline, [changed])
    assert len(unmatched) == 1
    assert unmatched[0]["message"] == "redefined unused import"

    duplicate_current = [_diagnostic("src/existing.py"), _diagnostic("src/existing.py")]
    assert len(verifier.unmatched_diagnostics(baseline, duplicate_current)) == 1


def test_scope_rules_allow_delayed_new_and_missing_retired_paths(tmp_path: Path):
    """Only existing paths are mandatory; retired and delayed new files may be absent."""
    verifier = _load_verifier()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "existing.py").write_text("value = 1\n", encoding="utf-8")
    baseline = _baseline(verifier)

    present = verifier.resolve_declared_paths(tmp_path, baseline)

    assert present == ["src/existing.py"]


def test_new_paths_must_be_clean_when_they_exist(tmp_path: Path):
    """New Phase 3 files can never inherit baseline debt."""
    verifier = _load_verifier()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "existing.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "src" / "new.py").write_text("value = 2\n", encoding="utf-8")
    baseline = _baseline(verifier)

    findings = [_diagnostic("src/new.py")]
    errors = verifier.scope_errors(tmp_path, baseline, findings)

    assert errors == ["new path has Ruff findings: src/new.py"]


def test_retired_diagnostics_need_not_remain_after_the_file_is_removed(tmp_path: Path):
    """Scheduler retirement removes both an allowed path and its old diagnostics."""
    verifier = _load_verifier()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "existing.py").write_text("value = 1\n", encoding="utf-8")
    retired_diagnostic = _diagnostic("src/retired.py")
    baseline = _baseline(verifier, diagnostics=[retired_diagnostic])

    assert verifier.scope_errors(tmp_path, baseline, []) == []
    assert verifier.unmatched_diagnostics(baseline, []) == []
