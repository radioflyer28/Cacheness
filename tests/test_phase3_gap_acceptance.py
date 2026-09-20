"""Historical Phase 3 inventory on the current BlobStore authority seams."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

from cacheness.storage.blob_store import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.sqlite_lifecycle_authority import SqliteLifecycleAuthority


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_GAP_NODES = {
    "CR-01": "tests/test_blob_store_concurrency.py::"
    "test_clear_and_delete_converge_after_an_exact_snapshot",
    "CR-02": "tests/test_lifecycle_authority_contract.py::"
    "test_sqlite_promotion_rolls_back_every_participating_authority_row",
    "CR-03": "tests/test_blob_store_concurrency.py::"
    "test_independent_write_write_race_has_one_cas_winner",
    "CR-04": "tests/test_projection_sql_atomicity.py::"
    "test_sql_projection_rebuild_publishes_only_after_an_isolated_candidate_completes",
    "CR-05": "tests/test_sqlite_bootstrap_concurrency.py::"
    "test_spawned_fresh_authorities_join_one_root_and_commit_distinct_keys",
    "CR-06": "tests/test_cached_custom_metadata.py::"
    "test_cached_read_model_refresh_reports_committed_partial_with_receipt",
    "CR-07": "tests/test_blob_store_atomic_lifecycle.py::"
    "test_clear_preserves_post_snapshot_writes",
    "CR-08": "tests/test_filesystem_containment.py::"
    "test_blob_store_encodes_hostile_key_without_mutating_outside_target",
    "CR-09": "tests/test_cached_custom_metadata.py::"
    "test_cached_read_model_best_effort_failure_preserves_the_committed_receipt",
    "WR-01": "tests/test_phase3_gap_acceptance.py::"
    "test_sqlite_authority_close_is_explicit_idempotent_and_silent",
    "WR-02": "tests/test_phase3_gap_acceptance.py::test_phase3_gap_acceptance_inventory",
}


def _local_topology(root: Path) -> StoreTopology:
    """Build the supported filesystem-payload/SQLite-authority topology."""
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


def _node_function_source(node: str) -> tuple[ast.FunctionDef, str]:
    """Load one declared real-path test function without executing it."""
    raw_path, function_name = node.split("::", maxsplit=1)
    source = (REPOSITORY_ROOT / raw_path).read_text(encoding="utf-8")
    module = ast.parse(source, filename=raw_path)
    for candidate in module.body:
        if isinstance(candidate, ast.FunctionDef) and candidate.name == function_name:
            return candidate, ast.get_source_segment(source, candidate) or ""
    raise AssertionError(f"required Phase 3 test node is missing: {node}")


def test_sqlite_authority_close_is_explicit_idempotent_and_silent(
    tmp_path: Path,
) -> None:
    """Explicit authority/store close owns teardown; no metadata adapter remains."""
    root = tmp_path / "authority"
    store = BlobStore(_local_topology(root), cache_dir=root)
    try:
        store.initialize()
        assert store.put({"description": "teardown contract"}, key="entry") == "entry"
        assert store.get("entry") == {"description": "teardown contract"}
        assert isinstance(store.lifecycle_authority, SqliteLifecycleAuthority)
        assert "__del__" not in SqliteLifecycleAuthority.__dict__
    finally:
        store.close()
    store.close()

    script = """
import sys
from pathlib import Path

from cacheness.storage.blob_store import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology

root = Path(sys.argv[1])
store = BlobStore(
    StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    ),
    cache_dir=root,
)
store.initialize()
store.put({"description": "subprocess teardown"}, key="entry")
if sys.argv[2] == "explicit":
    store.close()
"""
    for mode in ("explicit", "teardown"):
        completed = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / mode), mode],
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
        [sys.executable, "-m", "pytest", "--collect-only", *nodes],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert collected.returncode == 0, collected.stderr
    for node in nodes:
        assert node in collected.stdout

    for finding, node in REQUIRED_GAP_NODES.items():
        function, _function_source = _node_function_source(node)
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
        terminal_marks = {
            call.func.attr
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "pytest"
        }
        assert terminal_marks.isdisjoint({"skip", "xfail"}), (
            f"{finding} skips or xfails its schedule: {node}"
        )
        assert "tests/test_" in node, f"{finding} is not a real test node: {node}"
