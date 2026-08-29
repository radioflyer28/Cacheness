"""Security contract tests for filesystem-backed blob containment."""

from __future__ import annotations

import os
import subprocess
import threading
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


def test_managed_operations_revalidate_a_stale_locator_before_access(tmp_path):
    """Validation from a prior operation is never reused after an ancestor swap."""
    root = tmp_path / "root"
    managed = root / "managed"
    managed.mkdir(parents=True)
    locator = managed / "entry"
    locator.write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "entry").write_bytes(b"outside")

    anchored_root = resolve_storage_root(root)
    assert resolve_managed_locator(anchored_root, locator, operation="read") == locator
    managed.rename(root / "managed-before-swap")
    managed.symlink_to(outside, target_is_directory=True)

    operations = ManagedFileOps(anchored_root)
    try:
        with pytest.raises(CacheUnsafePathError) as exc_info:
            operations.read_bytes(locator)
    finally:
        operations.close()

    assert exc_info.value.context["reason"] == CacheReason.PATH_RACE.value
    assert (outside / "entry").read_bytes() == b"outside"


def test_anchored_operations_ignore_later_root_alias_retargeting(tmp_path):
    """An instance retains its initialization target while a new root resolves anew."""
    first_target = tmp_path / "first"
    second_target = tmp_path / "second"
    first_target.mkdir()
    second_target.mkdir()
    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(first_target, target_is_directory=True)

    operations = ManagedFileOps(resolve_storage_root(root_alias))
    try:
        root_alias.unlink()
        root_alias.symlink_to(second_target, target_is_directory=True)
        locator = operations.write_bytes("anchored", b"first", shard_chars=0)
    finally:
        operations.close()

    assert locator == first_target / "anchored"
    assert (first_target / "anchored").read_bytes() == b"first"
    assert not (second_target / "anchored").exists()
    assert resolve_storage_root(root_alias) == second_target.resolve()


def test_descriptor_mode_never_reads_outside_during_pathname_swap_stress(tmp_path):
    """Descriptor-capable Unix reads stay inside during concurrent name swapping."""
    root = tmp_path / "root"
    managed = root / "managed"
    managed.mkdir(parents=True)
    (managed / "entry").write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_entry = outside / "entry"
    outside_entry.write_bytes(b"outside")

    operations = ManagedFileOps(resolve_storage_root(root))
    if not operations.descriptor_mode:
        operations.close()
        pytest.skip("descriptor-relative no-follow operations unavailable")

    failures: list[BaseException] = []

    def swap_managed_path() -> None:
        parked = root / "parked"
        try:
            for _ in range(20):
                if managed.is_symlink():
                    managed.unlink()
                    os.rename(parked, managed)
                else:
                    os.rename(managed, parked)
                    candidate_link = root / "candidate-link"
                    candidate_link.symlink_to(outside, target_is_directory=True)
                    os.rename(candidate_link, managed)
        except BaseException as exc:  # pragma: no cover - test thread reporting
            failures.append(exc)

    thread = threading.Thread(target=swap_managed_path)
    thread.start()
    results: list[bytes] = []
    try:
        for _ in range(40):
            try:
                results.append(operations.read_bytes(root / "managed" / "entry"))
            except (CacheUnsafePathError, FileNotFoundError):
                pass
    finally:
        thread.join()
        operations.close()

    assert not failures
    assert all(result == b"inside" for result in results)
    assert outside_entry.read_bytes() == b"outside"


@pytest.mark.skipif(os.name != "nt", reason="Windows junction fixture")
def test_windows_junction_is_rejected_as_a_managed_reparse_component(tmp_path):
    """Windows exercises a junction even when privileged symlinks are unavailable."""
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "entry").write_bytes(b"outside")
    junction = root / "junction"

    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    with pytest.raises(CacheUnsafePathError) as exc_info:
        resolve_managed_locator(root, junction / "entry", operation="read")

    assert exc_info.value.context["reason"] == CacheReason.PATH_RACE.value
    assert (outside / "entry").read_bytes() == b"outside"
