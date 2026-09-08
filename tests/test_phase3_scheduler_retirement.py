"""Negative-reachability gates for the retired file-native scheduler."""

import ast
import importlib
from importlib.util import find_spec
from pathlib import Path

import pytest

from cacheness.storage import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology


RETIRED_SCHEDULER_MODULES = (
    "cacheness.storage.operation_repository",
    "cacheness.storage.operation_record",
    "cacheness.storage.clear_recovery",
)


def _topology(root: Path) -> StoreTopology:
    """Create the supported local direct-store composition."""
    return StoreTopology(
        payload=BackendRef(name="filesystem", options={"base_dir": root}),
        authority=BackendRef(name="sqlite", options={"root": root}),
    )


def _defined_names(source_path: Path) -> set[str]:
    """Return the top-level implementation names owned by one module."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


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
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    imported_modules.add(node.module)
                imported_modules.update(alias.name for alias in node.names)
        assert retired_modules.isdisjoint(imported_modules), source_path
        assert set(RETIRED_SCHEDULER_MODULES).isdisjoint(imported_modules), source_path

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


def test_fresh_and_reopened_stores_never_create_retired_control_artifacts(
    tmp_path: Path,
) -> None:
    """Normal authority-backed use never creates a retired control path."""
    root = tmp_path / "runtime-tree"
    store = BlobStore(_topology(root), cache_dir=root)
    try:
        assert store.get("absent") is None
        assert store.put({"value": "present"}, key="present") == "present"
        assert store.lifecycle_authority.read_entry("present") is not None
    finally:
        store.close()

    assert not (root / "operations").exists()
    reopened = BlobStore(_topology(root), cache_dir=root)
    try:
        assert reopened.get("present") == {"value": "present"}
        assert reopened.lifecycle_authority.read_entry("present") is not None
    finally:
        reopened.close()
    assert not (root / "operations").exists()


def test_retained_helpers_have_no_file_native_lock_or_control_authority() -> None:
    """Retained helpers only order local work and protect payload I/O."""
    storage_root = Path(__file__).parents[1] / "src" / "cacheness" / "storage"
    coordination_source = storage_root / "coordination.py"
    path_security_source = storage_root / "path_security.py"

    assert _defined_names(coordination_source).isdisjoint(
        {
            "InterprocessLockUnavailable",
            "_WindowsLockApi",
            "_NativeWindowsLockApi",
            "_windows_lock_api",
            "interprocess_open_file_lock",
            "interprocess_file_lock",
            "lock_stripe_index",
            "StoreAdmissionBarrier",
        }
    )
    assert _defined_names(path_security_source).isdisjoint(
        {
            "_WindowsRegistryAuthorityApi",
            "_windows_registry_authority_api",
        }
    )

    coordination_text = coordination_source.read_text(encoding="utf-8")
    path_security_text = path_security_source.read_text(encoding="utf-8")
    assert all(
        forbidden not in coordination_text
        for forbidden in ("LockFileEx", "UnlockFileEx", "fcntl.flock")
    )
    assert all(
        forbidden not in path_security_text
        for forbidden in (
            "CreateMutexW",
            "RegCreateKeyExW",
            "user.cacheness.lifecycle-lock.",
            "ensure_lifecycle_lock",
            "assert_retained_lock_identity",
            "promote_durable_pending_control",
        )
    )


def test_blob_store_has_one_authority_and_no_legacy_selector_or_repository_path() -> None:
    """The direct facade retains only StoreTopology-based lifecycle composition."""
    storage_root = Path(__file__).parents[1] / "src" / "cacheness" / "storage"
    blob_store_source = storage_root / "blob_store.py"

    blob_store_text = blob_store_source.read_text(encoding="utf-8")
    assert "StoreTopology" in blob_store_text
    assert all(
        forbidden not in blob_store_text
        for forbidden in (
            "_authority_mode",
            "self.manifest_repository",
            "manifest_repository",
            "ManifestCursor",
            "ManifestExpectation",
            "_authority_lifecycle is not None",
            "_reconciler",
            "backend=",
            "_select_projection_backend",
            "_create_lifecycle_authority",
        )
    )
