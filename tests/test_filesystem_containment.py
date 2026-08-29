"""Security contract tests for filesystem-backed blob containment."""

from __future__ import annotations

import os
import subprocess
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import CacheReason, CacheUnsafePathError
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.backends.blob_backends import FilesystemBlobBackend
from cacheness.storage import path_security
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


@pytest.mark.parametrize(
    "blob_id",
    [
        "../escape",
        "/tmp/escape",
        "C:\\escape",
        "\\\\server\\share\\escape",
        "\\rooted",
        "safe/mixed\\escape",
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_backend_rejects_hostile_ids_before_creating_shards(tmp_path, blob_id, streaming):
    """Both write entry points reject direct unsafe IDs without creating residue."""
    backend = FilesystemBlobBackend(tmp_path / "root")

    with pytest.raises(CacheUnsafePathError):
        if streaming:
            backend.write_blob_stream(blob_id, BytesIO(b"payload"))
        else:
            backend.write_blob(blob_id, b"payload")

    assert list(backend.base_dir.iterdir()) == []


@pytest.mark.parametrize(
    "operation", ["read", "delete", "exists", "read_stream", "size"]
)
@pytest.mark.parametrize("locator_kind", ["outside", "ancestor_link", "leaf_link"])
def test_backend_rejects_unsafe_locators_for_every_direct_operation(
    tmp_path, operation, locator_kind
):
    """Unsafe locators never become a miss-like result or touch outside bytes."""
    root = tmp_path / "root"
    backend = FilesystemBlobBackend(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"outside")

    if locator_kind == "outside":
        locator = sentinel
    elif locator_kind == "ancestor_link":
        (root / "managed").symlink_to(outside, target_is_directory=True)
        locator = root / "managed" / "sentinel"
    else:
        (root / "leaf").symlink_to(sentinel)
        locator = root / "leaf"

    with pytest.raises(CacheUnsafePathError):
        if operation == "read":
            backend.read_blob(str(locator))
        elif operation == "delete":
            backend.delete_blob(str(locator))
        elif operation == "exists":
            backend.exists(str(locator))
        elif operation == "read_stream":
            stream = backend.read_blob_stream(str(locator))
            stream.close()
        else:
            backend.get_size(str(locator))

    assert sentinel.read_bytes() == b"outside"


@pytest.mark.parametrize(
    "locator",
    ["../escape", "/tmp/escape", "C:\\escape", "\\\\server\\share\\escape", "\\rooted"],
)
def test_backend_rejects_cross_platform_direct_locator_shapes(tmp_path, locator):
    """Direct locators apply the same host-independent path-shape policy."""
    backend = FilesystemBlobBackend(tmp_path / "root")

    with pytest.raises(CacheUnsafePathError):
        backend.exists(locator)


def test_backend_preserves_valid_sharded_atomic_stream_lifecycle(tmp_path):
    """Guarded writes retain compatible paths, streams, size, and deletion behavior."""
    backend = FilesystemBlobBackend(tmp_path / "root", shard_chars=2)
    first_locator = backend.write_blob("ab-entry", b"first")
    second_locator = backend.write_blob_stream("ab-entry", BytesIO(b"second"))

    assert first_locator == second_locator
    assert Path(second_locator).parent == backend.base_dir / "ab"
    assert backend.read_blob(second_locator) == b"second"
    with backend.read_blob_stream(second_locator) as stream:
        assert stream.read() == b"second"
    assert backend.exists(second_locator)
    assert backend.get_size(second_locator) == len(b"second")
    assert backend.delete_blob(second_locator)
    assert not backend.exists(second_locator)
    assert not list(backend.base_dir.rglob("*.tmp"))


# =============================================================================
# High-level guarded handler I/O (Plan 01-03)
# =============================================================================


class _InstrumentedHandler:
    """Small handler that records every path the high-level APIs expose."""

    data_type = "instrumented"

    def __init__(self):
        self.put_paths: list[Path] = []
        self.get_paths: list[Path] = []

    def put(self, data: Any, file_path: Path, _config: Any) -> dict[str, Any]:
        self.put_paths.append(file_path)
        artifact = file_path.with_suffix(".guarded")
        artifact.write_text(str(data), encoding="utf-8")
        return {
            "storage_format": "guarded",
            "file_size": artifact.stat().st_size,
            "actual_path": str(artifact),
            "metadata": {"instrumented": True},
        }

    def get(self, file_path: Path, _metadata: dict[str, Any]) -> str:
        self.get_paths.append(file_path)
        return file_path.read_text(encoding="utf-8")


class _SingleHandlerRegistry:
    """Registry double selecting one instrumented handler for all test data."""

    def __init__(self, handler: _InstrumentedHandler):
        self.handler = handler

    def get_handler(self, _data: Any) -> _InstrumentedHandler:
        return self.handler

    def get_handler_by_type(self, data_type: str) -> _InstrumentedHandler:
        assert data_type == self.handler.data_type
        return self.handler


def _is_descendant(path: Path, root: Path) -> bool:
    """Return whether a path is contained by root without trusting its spelling."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def test_high_level_physical_name_encoder_is_stable_domain_separated_and_backend_safe():
    """Logical values never become managed path components at the backend boundary."""
    encoder = path_security.encode_physical_name
    ordinary = encoder("key", "prefix", namespace="blob-store")

    assert ordinary == encoder("key", "prefix", namespace="blob-store")
    assert ordinary != encoder("keyprefix", "", namespace="blob-store")
    assert ordinary != encoder("key", "prefix", namespace="unified-cache")
    assert ordinary != encoder("key", "prefix ", namespace="blob-store")
    assert len(ordinary) == 64
    assert ordinary == ordinary.lower()
    assert all(character in "0123456789abcdef" for character in ordinary)
    assert validate_blob_id(ordinary) == ordinary


def test_high_level_blob_store_keeps_logical_key_while_handlers_only_see_private_paths(tmp_path):
    """BlobStore keeps the public key exact and never passes cache-root paths to handlers."""
    root = tmp_path / "blob-root"
    handler = _InstrumentedHandler()
    store = BlobStore(root)
    store.handlers = _SingleHandlerRegistry(handler)
    logical_key = "../tenant/key with spaces"

    try:
        stored_key = store.put("payload", key=logical_key)
        entry = store.get_metadata(logical_key)

        assert stored_key == logical_key
        assert entry is not None
        assert entry["cache_key"] == logical_key
        assert store.list(prefix="../tenant") == [logical_key]
        assert store.get(logical_key) == "payload"
        assert all(not _is_descendant(path, root) for path in handler.put_paths)
        assert all(not _is_descendant(path, root) for path in handler.get_paths)
        assert all(path.exists() for path in handler.get_paths)
        actual_path = Path(entry["metadata"]["actual_path"])
        assert _is_descendant(actual_path, root)
        assert logical_key not in str(actual_path)
    finally:
        store.close()


def test_high_level_persisted_locator_raises_before_deserialization(tmp_path):
    """Unsafe persisted locators remain typed errors, not reads or cache misses."""
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")

    blob_handler = _InstrumentedHandler()
    store = BlobStore(tmp_path / "blobs")
    store.handlers = _SingleHandlerRegistry(blob_handler)
    try:
        key = store.put("inside", key="safe")
        entry = store.backend.get_entry(key)
        assert entry is not None
        entry["metadata"]["actual_path"] = str(outside)
        store.backend.put_entry(key, entry)

        with pytest.raises(CacheUnsafePathError):
            store.get(key)
        assert blob_handler.get_paths == []
        assert outside.read_text(encoding="utf-8") == "outside"
    finally:
        store.close()

    cache_handler = _InstrumentedHandler()
    config = CacheConfig(
        cache_dir=str(tmp_path / "cache"),
        metadata_backend="memory",
        cleanup_on_init=False,
        verify_cache_integrity=False,
    )
    cache = UnifiedCache(config)
    cache.handlers = _SingleHandlerRegistry(cache_handler)
    key = cache.put("inside", identity="safe")
    entry = cache.metadata_backend.get_entry(key)
    assert entry is not None
    entry["metadata"]["actual_path"] = str(outside)

    with pytest.raises(CacheUnsafePathError):
        cache.get(cache_key=key)
    assert cache_handler.get_paths == []
    assert outside.read_text(encoding="utf-8") == "outside"


def test_unified_cache_encodes_hostile_prefix_without_mutating_outside_target(tmp_path):
    """An authored prefix remains metadata while only its encoded physical name is used."""
    root = tmp_path / "cache"
    handler = _InstrumentedHandler()
    cache = UnifiedCache(
        CacheConfig(
            cache_dir=str(root),
            metadata_backend="memory",
            cleanup_on_init=False,
        )
    )
    cache.handlers = _SingleHandlerRegistry(handler)
    prefix = "../outside prefix"

    key = cache.put("payload", prefix=prefix, identity="prefix")
    entry = cache.metadata_backend.get_entry(key)

    assert entry is not None
    assert entry["prefix"] == prefix
    actual_path = Path(entry["metadata"]["actual_path"])
    assert _is_descendant(actual_path, root)
    assert prefix not in str(actual_path)
    assert cache.get(cache_key=key) == "payload"
    assert all(not _is_descendant(path, root) for path in handler.put_paths)
    assert all(not _is_descendant(path, root) for path in handler.get_paths)
