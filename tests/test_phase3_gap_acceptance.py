"""Phase 3 gap-closure acceptance inventory and SQLite teardown contracts."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

from cacheness.metadata import SqliteBackend


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_GAP_NODES = {
    "CR-01": (
        "tests/test_unified_cache_adversarial_lifecycle.py::"
        "test_facade_put_admission_blocks_close_until_authority_promotion_exits",
    ),
    "CR-02": (
        "tests/test_unified_cache_adversarial_lifecycle.py::"
        "test_linked_m1_survives_a_pre_promotion_verification_failure",
    ),
    "CR-03": (
        "tests/test_unified_cache_adversarial_lifecycle.py::"
        "test_two_pending_candidates_preserve_m1_links_until_one_promotes",
    ),
    "CR-04": (
        "tests/test_projection_sql_atomicity.py::"
        "test_sqlite_spawned_processes_have_one_projection_winner",
    ),
    "CR-05": (
        "tests/test_sqlite_bootstrap_concurrency.py::"
        "test_spawned_fresh_authorities_join_one_root_and_commit_distinct_keys",
    ),
    "CR-06": (
        "tests/test_cached_custom_metadata.py::"
        "test_cached_sqlite_facade_preserves_live_custom_metadata_across_replacement",
    ),
    "CR-07": (
        "tests/test_unified_cache_adversarial_lifecycle.py::"
        "test_empty_authority_clear_preserves_a_peer_first_put_after_durable_intent",
    ),
    "CR-08": (
        "tests/test_unified_cache_adversarial_lifecycle.py::"
        "test_hostile_locator_rejects_same_key_put_without_mutating_m1",
    ),
    "CR-09": (
        "tests/test_cached_custom_metadata.py::"
        "test_signed_postgresql_cache_key_params_keep_a_valid_projection_live",
    ),
    "WR-01": "tests/test_phase3_gap_acceptance.py::test_sqlite_backend_close_is_explicit_idempotent_and_silent",
    "WR-02": "tests/test_phase3_gap_acceptance.py::test_phase3_gap_acceptance_inventory",
}


def _node_function_source(node: str) -> tuple[ast.FunctionDef, str]:
    """Load one declared real-path test function without executing it."""
    raw_path, function_name = node.split("::", maxsplit=1)
    source = (REPOSITORY_ROOT / raw_path).read_text(encoding="utf-8")
    module = ast.parse(source, filename=raw_path)
    for candidate in module.body:
        if isinstance(candidate, ast.FunctionDef) and candidate.name == function_name:
            return candidate, ast.get_source_segment(source, candidate) or ""
    raise AssertionError(f"required Phase 3 test node is missing: {node}")


def test_sqlite_backend_close_is_explicit_idempotent_and_silent(tmp_path: Path) -> None:
    """Explicit close detaches SQLAlchemy resources; teardown does no finalizer work."""
    database = tmp_path / "metadata.sqlite3"
    backend = SqliteBackend(str(database))
    with backend:
        backend.put_entry(
            "entry",
            {
                "description": "teardown contract",
                "data_type": "object",
                "prefix": "",
                "file_size": 1,
                "metadata": {"actual_path": "/entry"},
            },
        )
        assert backend.get_entry("entry") is not None
    assert backend.engine is None
    assert backend.SessionLocal is None
    assert "__del__" not in SqliteBackend.__dict__
    backend.close()

    script = """
import sys
from cacheness.metadata import SqliteBackend

backend = SqliteBackend(sys.argv[1])
backend.put_entry(
    "entry",
    {
        "description": "subprocess teardown",
        "data_type": "object",
        "prefix": "",
        "file_size": 1,
        "metadata": {"actual_path": "/entry"},
    },
)
if sys.argv[2] == "explicit":
    backend.close()
"""
    for mode in ("explicit", "teardown"):
        completed = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / f"{mode}.sqlite3"), mode],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "Exception ignored" not in completed.stderr
        assert "ImportError" not in completed.stderr
        assert "logging" not in completed.stderr.lower()


def test_phase3_gap_acceptance_inventory() -> None:
    """Every review finding retains an active deterministic real-path schedule."""
    assert set(REQUIRED_GAP_NODES) == {
        *(f"CR-{number:02d}" for number in range(1, 10)),
        "WR-01",
        "WR-02",
    }
    nodes = tuple(REQUIRED_GAP_NODES.values())
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *nodes],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert collected.returncode == 0, collected.stderr
    for node in nodes:
        assert node in collected.stdout

    for finding, node in REQUIRED_GAP_NODES.items():
        function, function_source = _node_function_source(node)
        marker_names = {
            decorator.func.attr
            for decorator in function.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Attribute)
            and decorator.func.value.attr == "mark"
        }
        assert marker_names.isdisjoint({"skip", "skipif", "xfail"}), (
            f"{finding} must remain active: {node}"
        )
        assert "pytest.skip" not in function_source, f"{finding} skips its schedule: {node}"
        assert "pytest.xfail" not in function_source, f"{finding} xfails its schedule: {node}"
        assert "tests/test_" in node, f"{finding} is not a real test node: {node}"

