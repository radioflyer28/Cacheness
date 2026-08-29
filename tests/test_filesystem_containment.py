"""Security contract tests for filesystem-backed blob containment."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cacheness.error_handling import CacheReason, CacheUnsafePathError
from cacheness.storage.path_security import (
    ManagedFileOps,
    resolve_managed_locator,
    resolve_storage_root,
    validate_blob_id,
)


@pytest.mark.parametrize(
    ("blob_id", "reason"),
    [
        ("../escape", CacheReason.PATH_TRAVERSAL),
        ("..\\escape", CacheReason.PATH_TRAVERSAL),
        ("/tmp/escape", CacheReason.PATH_ABSOLUTE),
        ("C:\\escape", CacheReason.PATH_DRIVE),
        ("C:/escape", CacheReason.PATH_DRIVE),
        ("\\\\server\\share\\escape", CacheReason.PATH_UNC),
        ("\\rooted", CacheReason.PATH_ROOTED),
        ("safe/mixed\\escape", CacheReason.INVALID_IDENTIFIER),
        ("contains\x00nul", CacheReason.INVALID_IDENTIFIER),
        ("a" * 257, CacheReason.INVALID_IDENTIFIER),
        (".hidden", CacheReason.INVALID_IDENTIFIER),
    ],
)
def test_validate_blob_id_rejects_hostile_cross_platform_identifiers(blob_id, reason):
    """Opaque backend IDs reject unsafe forms before any path allocation."""
    with pytest.raises(CacheUnsafePathError) as exc_info:
        validate_blob_id(blob_id)

    assert exc_info.value.context["reason"] == reason.value


@pytest.mark.parametrize("blob_id", ["a", "safe_id-1.2", "A" * 256])
def test_validate_blob_id_accepts_only_opaque_backend_identifiers(blob_id):
    """Valid IDs preserve the restrictive D-12 grammar exactly."""
    assert validate_blob_id(blob_id) == blob_id


def test_resolved_root_allows_a_configured_root_symlink(tmp_path):
    """A configured root alias anchors once to its resolved target."""
    target = tmp_path / "target"
    target.mkdir()
    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(target, target_is_directory=True)

    root = resolve_storage_root(root_alias)

    assert root == target.resolve()
    assert resolve_managed_locator(root, "entry", operation="read") == root / "entry"


@pytest.mark.parametrize("link_name", ["ancestor", "leaf", "broken"])
def test_resolve_managed_locator_rejects_managed_symlink_components(tmp_path, link_name):
    """Existing, including broken, managed links fail closed before access."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_bytes(b"outside")

    if link_name == "ancestor":
        (root / "ancestor").symlink_to(outside, target_is_directory=True)
        locator = root / "ancestor" / "secret"
    elif link_name == "leaf":
        (root / "leaf").symlink_to(outside / "secret")
        locator = root / "leaf"
    else:
        (root / "broken").symlink_to(outside / "missing")
        locator = root / "broken"

    with pytest.raises(CacheUnsafePathError) as exc_info:
        resolve_managed_locator(root, locator, operation="read")

    assert exc_info.value.context["reason"] == CacheReason.PATH_RACE.value
    assert (outside / "secret").read_bytes() == b"outside"


def test_resolve_managed_locator_rejects_outside_locator_without_mutation(tmp_path):
    """Persisted locators outside the anchored root are never normalized or touched."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"unchanged")

    with pytest.raises(CacheUnsafePathError) as exc_info:
        resolve_managed_locator(root, sentinel, operation="delete")

    assert exc_info.value.context["reason"] == CacheReason.PATH_OUTSIDE_ROOT.value
    assert sentinel.read_bytes() == b"unchanged"


def test_resolve_managed_locator_does_not_create_paths_during_validation(tmp_path):
    """Validation may permit a missing leaf but never creates it or its parent."""
    root = tmp_path / "root"
    root.mkdir()

    locator = resolve_managed_locator(
        root,
        "new-parent/new-leaf",
        operation="write",
        allow_missing_leaf=True,
    )

    assert locator == root / "new-parent" / "new-leaf"
    assert not (root / "new-parent").exists()


def test_managed_operations_reject_a_deterministic_between_check_retarget(tmp_path):
    """A same-process ancestor swap at the operation seam cannot read outside data."""
    root = tmp_path / "root"
    managed = root / "managed"
    managed.mkdir(parents=True)
    (managed / "entry").write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "entry").write_bytes(b"outside")

    operations = ManagedFileOps(resolve_storage_root(root))

    def retarget(_operation: str, _locator: Path) -> None:
        if managed.exists() and not managed.is_symlink():
            managed.rename(root / "managed-before-swap")
            managed.symlink_to(outside, target_is_directory=True)

    operations.before_operation = retarget

    try:
        with pytest.raises(CacheUnsafePathError) as exc_info:
            operations.read_bytes(root / "managed" / "entry")
    finally:
        operations.close()

    assert exc_info.value.context["reason"] == CacheReason.PATH_RACE.value
    assert (outside / "entry").read_bytes() == b"outside"
    if os.name != "nt":
        assert (root / "managed-before-swap" / "entry").read_bytes() == b"inside"
