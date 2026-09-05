"""Negative-reachability gates for the retired file-native scheduler."""

import ast
import importlib
from importlib.util import find_spec
from pathlib import Path

import pytest


RETIRED_SCHEDULER_MODULES = (
    "cacheness.storage.operation_repository",
    "cacheness.storage.operation_record",
    "cacheness.storage.clear_recovery",
)


@pytest.mark.parametrize("module_name", RETIRED_SCHEDULER_MODULES)
def test_retired_scheduler_modules_are_not_importable(module_name: str) -> None:
    """The public package cannot resolve a second lifecycle authority."""
    assert find_spec(module_name) is None


def test_retired_scheduler_has_no_package_or_runtime_reachability() -> None:
    """No barrel, import, or fallback factory can revive the retired engine."""
    source_root = Path(__file__).parents[1] / "src" / "cacheness"
    retired_modules = {module_name.rsplit(".", 1)[-1] for module_name in RETIRED_SCHEDULER_MODULES}

    for source_path in source_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_modules = {
            alias.name.rsplit(".", 1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert retired_modules.isdisjoint(imported_modules), source_path

    import cacheness
    import cacheness.storage as storage

    for package in (cacheness, storage):
        assert retired_modules.isdisjoint(vars(package))
        assert retired_modules.isdisjoint(getattr(package, "__all__", ()))

    reconciliation = importlib.import_module("cacheness.storage.reconciliation")
    assert "_Reconciler" not in vars(reconciliation)
    blob_store_source = (source_root / "storage" / "blob_store.py").read_text(
        encoding="utf-8"
    )
    assert "operation_repository" not in blob_store_source
