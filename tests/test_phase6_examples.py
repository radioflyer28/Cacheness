"""Executable public examples for the Phase 6 canonical cache surface."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DOC = ROOT / "docs" / "CACHE_POLICY.md"


def _run_example(name: str, temporary_directory: Path) -> subprocess.CompletedProcess[str]:
    """Run one example from a fresh working directory without project globals."""

    environment = os.environ.copy()
    environment.pop("CACHENESS_CACHE_DIR", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        cwd=temporary_directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_simple_object_example_executes_in_an_isolated_subprocess(tmp_path: Path) -> None:
    """The canonical object lifecycle has no implicit cache or service dependency."""

    completed = _run_example("simple_object_caching.py", tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "PUT_COMMITTED=" in completed.stdout
    assert "LOOKUP_OUTCOME=hit" in completed.stdout
    assert "INVALIDATION=attempted:1,removed:1,complete:True" in completed.stdout
    assert "CANONICAL_OBJECT_CACHE_EXAMPLE_OK" in completed.stdout


def test_cache_policy_docs_cover_the_canonical_outcome_contract() -> None:
    """The guide names canonical APIs, limits, ownership, and all outcomes."""

    guide = DOC.read_text(encoding="utf-8")

    for outcome in (
        "`hit`",
        "`absent`",
        "`expired`",
        "`corrupt`",
        "`conflict`",
        "`backend_error`",
    ):
        assert outcome in guide
    for required_claim in (
        "BlobStore",
        "CacheMaintenanceState",
        "CacheRemovalReport",
        "cache_last_lookup",
        "Phase 7",
        "SqlCache",
    ):
        assert required_claim in guide
    assert "never upgrades it" in guide
