"""Security contract tests for filesystem-backed blob containment."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheReason,
    CacheStorageError,
    CacheUnsafePathError,
)
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.backends.blob_backends import FilesystemBlobBackend
from cacheness.storage.integrity import sign_hmac_sha256
from cacheness.storage.manifest import BlobManifestV1
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
        self.get_paths_alive: list[bool] = []
        self.events: list[str] = []

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
        self.get_paths_alive.append(file_path.exists())
        self.events.append("handler")
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


class _FormatHandler:
    """Handler double whose suffix and serialized format identify one payload."""

    def __init__(
        self,
        data_type: str,
        suffix: str,
        storage_format: str,
        serializer: str,
        compression: str,
    ) -> None:
        self.data_type = data_type
        self.suffix = suffix
        self.storage_format = storage_format
        self.serializer = serializer
        self.compression = compression

    def put(self, data: Any, file_path: Path, _config: Any) -> dict[str, Any]:
        """Write an exact, format-marked private staging artifact."""
        artifact = file_path.with_suffix(self.suffix)
        artifact.write_text(str(data), encoding="utf-8")
        return {
            "storage_format": self.storage_format,
            "file_size": artifact.stat().st_size,
            "actual_path": str(artifact),
            "metadata": {
                "serializer": self.serializer,
                "compression": self.compression,
            },
        }

    def get(self, file_path: Path, metadata: dict[str, Any]) -> str:
        """Read only a matching format, making stale metadata observable."""
        assert metadata["storage_format"] == self.storage_format
        return file_path.read_text(encoding="utf-8")


class _SwitchingHandlerRegistry:
    """Registry double that writes through a selected handler and reads by type."""

    def __init__(self, *handlers: _FormatHandler) -> None:
        self.current = handlers[0]
        self._handlers = {handler.data_type: handler for handler in handlers}

    def get_handler(self, _data: Any) -> _FormatHandler:
        """Return the handler selected for the next write."""
        return self.current

    def get_handler_by_type(self, data_type: str) -> _FormatHandler:
        """Resolve persisted metadata through its recorded handler type."""
        return self._handlers[data_type]


def _format_handlers() -> tuple[_FormatHandler, _FormatHandler]:
    """Return two incompatible handler formats for overwrite preservation tests."""
    return (
        _FormatHandler("text_v1", ".v1", "plain-v1", "json", "none"),
        _FormatHandler("text_v2", ".v2", "plain-v2", "msgpack", "zstd"),
    )


def _payloads_for_key(store: BlobStore, key: str) -> list[Path]:
    """Return all managed payloads sharing a logical key's physical base."""
    storage_id = store._storage_id_for_key(key)
    return sorted(store.cache_dir.glob(f"{storage_id}*"))


def _is_descendant(path: Path, root: Path) -> bool:
    """Return whether a path is contained by root without trusting its spelling."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def test_physical_name_encoder_is_stable_domain_separated_and_backend_safe():
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


def test_blob_store_keeps_logical_key_while_handlers_only_see_private_paths(tmp_path):
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
        assert all(handler.get_paths_alive)
        actual_path = Path(entry["metadata"]["actual_path"])
        assert _is_descendant(actual_path, root)
        assert logical_key not in str(actual_path)
    finally:
        store.close()


@pytest.mark.parametrize("handler_index", [0, 1])
def test_blob_store_first_write_metadata_failure_removes_candidate(
    tmp_path, monkeypatch, handler_index
):
    """A first-write metadata failure leaves no entry or unowned payload bytes."""
    root = tmp_path / "blob-root"
    handlers = _format_handlers()
    registry = _SwitchingHandlerRegistry(*handlers)
    registry.current = handlers[handler_index]
    store = BlobStore(root)
    store.handlers = registry

    def fail_metadata_write(_key: str, _entry: dict[str, Any]) -> None:
        raise RuntimeError("metadata unavailable")

    try:
        monkeypatch.setattr(store.backend, "put_entry", fail_metadata_write)

        with pytest.raises(RuntimeError, match="metadata unavailable"):
            store.put("replacement", key="first-write")

        assert store.get_metadata("first-write") is None
        assert _payloads_for_key(store, "first-write") == []
    finally:
        store.close()


def test_blob_store_cross_format_overwrite_metadata_failure_preserves_prior_evidence(
    tmp_path, monkeypatch
):
    """A failed replacement preserves exact old metadata, bytes, and readability."""
    root = tmp_path / "blob-root"
    old_handler, replacement_handler = _format_handlers()
    registry = _SwitchingHandlerRegistry(old_handler, replacement_handler)
    store = BlobStore(root)
    store.handlers = registry
    key = "cross-format"
    initial_metadata = {
        "serializer": old_handler.serializer,
        "handler_compression": old_handler.compression,
    }

    def fail_metadata_write(_key: str, _entry: dict[str, Any]) -> None:
        raise RuntimeError("metadata unavailable")

    try:
        store.put("old value", key=key, metadata=initial_metadata)
        entry_before = deepcopy(store.get_metadata(key))
        assert entry_before is not None
        old_path = Path(entry_before["metadata"]["actual_path"])
        old_bytes = old_path.read_bytes()
        assert old_path.suffix == old_handler.suffix
        assert entry_before["data_type"] == old_handler.data_type
        assert entry_before["metadata"]["storage_format"] == old_handler.storage_format
        assert entry_before["metadata"]["serializer"] == old_handler.serializer
        assert (
            entry_before["metadata"]["handler_compression"]
            == old_handler.compression
        )

        registry.current = replacement_handler
        monkeypatch.setattr(store.backend, "put_entry", fail_metadata_write)

        with pytest.raises(RuntimeError, match="metadata unavailable"):
            store.put(
                "replacement value",
                key=key,
                metadata={
                    "serializer": replacement_handler.serializer,
                    "handler_compression": replacement_handler.compression,
                },
            )

        assert store.get_metadata(key) == entry_before
        assert old_path.read_bytes() == old_bytes
        assert _payloads_for_key(store, key) == [old_path]
        assert store.get(key) == "old value"
    finally:
        store.close()


@pytest.mark.parametrize("cleanup_outcome", ["false", "raise"])
def test_blob_store_candidate_cleanup_failure_is_explicit_and_chained(
    tmp_path, monkeypatch, cleanup_outcome
):
    """An unprovable candidate deletion is never silently converted to a miss."""
    root = tmp_path / "blob-root"
    handler, _ = _format_handlers()
    store = BlobStore(root)
    store.handlers = _SwitchingHandlerRegistry(handler)
    cleanup_attempts: list[Path] = []

    def fail_metadata_write(_key: str, _entry: dict[str, Any]) -> None:
        raise RuntimeError("metadata unavailable")

    def cannot_prove_cleanup(locator: Path | str) -> bool:
        cleanup_attempts.append(Path(locator))
        if cleanup_outcome == "raise":
            raise OSError("candidate cleanup unavailable")
        return False

    try:
        monkeypatch.setattr(store.backend, "put_entry", fail_metadata_write)
        monkeypatch.setattr(
            store.guarded_handler_io.file_ops,
            "delete",
            cannot_prove_cleanup,
        )

        with pytest.raises(CacheStorageError) as exc_info:
            store.put("replacement", key="cleanup-proof")

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "metadata unavailable" in str(exc_info.value.__cause__)
        assert len(cleanup_attempts) == 1
        assert store.get_metadata("cleanup-proof") is None
    finally:
        store.close()


def test_blob_store_post_commit_prior_cleanup_keeps_new_metadata_authoritative(
    tmp_path, monkeypatch
):
    """A failed old-payload cleanup cannot restore stale replacement metadata."""
    root = tmp_path / "blob-root"
    old_handler, replacement_handler = _format_handlers()
    registry = _SwitchingHandlerRegistry(old_handler, replacement_handler)
    store = BlobStore(root)
    store.handlers = registry
    key = "post-commit"

    try:
        store.put("old value", key=key)
        entry_before = store.get_metadata(key)
        assert entry_before is not None
        old_path = Path(entry_before["metadata"]["actual_path"])
        delete = store.guarded_handler_io.file_ops.delete

        def fail_old_payload_cleanup(locator: Path | str) -> bool:
            if Path(locator) == old_path:
                return False
            return delete(locator)

        registry.current = replacement_handler
        monkeypatch.setattr(
            store.guarded_handler_io.file_ops,
            "delete",
            fail_old_payload_cleanup,
        )

        with pytest.raises(CacheStorageError, match="cleanup"):
            store.put("replacement value", key=key)

        entry_after = store.get_metadata(key)
        assert entry_after is not None
        new_path = Path(entry_after["metadata"]["actual_path"])
        assert entry_after["data_type"] == replacement_handler.data_type
        assert entry_after["metadata"]["storage_format"] == replacement_handler.storage_format
        assert new_path != old_path
        assert new_path.exists()
        assert old_path.exists()
        assert store.get(key) == "replacement value"
    finally:
        store.close()


def test_blob_store_clear_removes_guarded_payloads_before_metadata(tmp_path):
    """A successful clear leaves neither reachable metadata nor orphan payload bytes."""
    root = tmp_path / "blob-root"
    handler = _InstrumentedHandler()
    store = BlobStore(root)
    store.handlers = _SingleHandlerRegistry(handler)

    try:
        first_key = store.put("first", key="first")
        second_key = store.put("second", key="second")
        payload_paths = [
            Path(store.get_metadata(key)["metadata"]["actual_path"])
            for key in (first_key, second_key)
        ]

        assert all(path.exists() for path in payload_paths)
        assert store.clear() == 2

        assert all(not path.exists() for path in payload_paths)
        assert store.backend.list_entries() == []
        assert store.get(first_key) is None
        assert store.get(second_key) is None
    finally:
        store.close()


@pytest.mark.parametrize("failing_payload_delete", [1, 2])
def test_blob_store_clear_rolls_back_every_payload_when_staging_delete_fails(
    tmp_path, monkeypatch, failing_payload_delete
):
    """A staged clear restores every still-committed payload after delete failure."""
    root = tmp_path / "blob-root"
    handler = _InstrumentedHandler()
    store = BlobStore(root)
    store.handlers = _SingleHandlerRegistry(handler)

    try:
        keys = [store.put("first", key="first"), store.put("second", key="second")]
        entries_before = [store.get_metadata(key) for key in keys]
        assert all(entry is not None for entry in entries_before)
        payload_paths = [
            Path(entry["metadata"]["actual_path"])
            for entry in entries_before
            if entry is not None
        ]
        payload_bytes = {path: path.read_bytes() for path in payload_paths}
        delete = store.guarded_handler_io.file_ops.delete
        payload_delete_count = 0

        def fail_one_payload_delete(locator):
            nonlocal payload_delete_count
            if Path(locator) in payload_paths:
                payload_delete_count += 1
                if payload_delete_count == failing_payload_delete:
                    raise RuntimeError("payload delete unavailable")
            return delete(locator)

        monkeypatch.setattr(store.guarded_handler_io.file_ops, "delete", fail_one_payload_delete)

        with pytest.raises(CacheBlobBackendError) as error:
            store.clear()

        assert isinstance(error.value.__cause__, RuntimeError)
        assert str(error.value.__cause__) == "payload delete unavailable"
        assert [store.get_metadata(key) for key in keys] == entries_before
        assert {path: path.read_bytes() for path in payload_paths} == payload_bytes
        assert [store.get(key) for key in keys] == ["first", "second"]
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        store.close()


def test_blob_store_clear_rolls_back_payloads_when_metadata_clear_fails(
    tmp_path, monkeypatch
):
    """A metadata failure restores every staged payload to its original locator."""
    root = tmp_path / "blob-root"
    handler = _InstrumentedHandler()
    store = BlobStore(root)
    store.handlers = _SingleHandlerRegistry(handler)

    try:
        keys = [store.put("first", key="first"), store.put("second", key="second")]
        entries_before = [store.get_metadata(key) for key in keys]
        assert all(entry is not None for entry in entries_before)
        payload_paths = [
            Path(entry["metadata"]["actual_path"])
            for entry in entries_before
            if entry is not None
        ]
        payload_bytes = {path: path.read_bytes() for path in payload_paths}

        remove_entry = store.backend.remove_entry

        def fail_metadata_clear() -> int:
            remove_entry(keys[0])
            raise RuntimeError("metadata unavailable after partial clear")

        monkeypatch.setattr(store.backend, "clear_all", fail_metadata_clear)

        with pytest.raises(CacheBlobBackendError) as error:
            store.clear()

        assert isinstance(error.value.__cause__, RuntimeError)
        assert str(error.value.__cause__) == "metadata unavailable after partial clear"
        assert [store.get_metadata(key) for key in keys] == entries_before
        assert {path: path.read_bytes() for path in payload_paths} == payload_bytes
        assert [store.get(key) for key in keys] == ["first", "second"]
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        store.close()


def test_blob_store_clear_discards_tombstone_if_staging_copy_raises(
    tmp_path, monkeypatch
):
    """A stage-write failure cannot leak an unregistered tombstone payload."""
    root = tmp_path / "blob-root"
    handler = _InstrumentedHandler()
    store = BlobStore(root)
    store.handlers = _SingleHandlerRegistry(handler)

    try:
        key = store.put("payload", key="entry")
        entry_before = store.get_metadata(key)
        assert entry_before is not None
        payload_path = Path(entry_before["metadata"]["actual_path"])
        payload_bytes = payload_path.read_bytes()
        write_stream_to_locator = store.guarded_handler_io.file_ops.write_stream_to_locator

        def write_tombstone_then_raise(locator, source):
            write_stream_to_locator(locator, source)
            raise RuntimeError("tombstone staging unavailable")

        monkeypatch.setattr(
            store.guarded_handler_io.file_ops,
            "write_stream_to_locator",
            write_tombstone_then_raise,
        )

        with pytest.raises(CacheBlobBackendError) as error:
            store.clear()

        assert isinstance(error.value.__cause__, RuntimeError)
        assert str(error.value.__cause__) == "tombstone staging unavailable"
        assert store.get_metadata(key) == entry_before
        assert payload_path.read_bytes() == payload_bytes
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        store.close()


def test_blob_store_clear_leaves_only_recoverable_tombstones_when_final_delete_fails(
    tmp_path, monkeypatch
):
    """Post-commit payload cleanup cannot leave live metadata pointing at a miss."""
    root = tmp_path / "blob-root"
    handler = _InstrumentedHandler()
    store = BlobStore(root)
    store.handlers = _SingleHandlerRegistry(handler)

    try:
        keys = [store.put("first", key="first"), store.put("second", key="second")]
        payload_paths = [
            Path(store.get_metadata(key)["metadata"]["actual_path"])
            for key in keys
        ]
        delete = store.guarded_handler_io.file_ops.delete

        def fail_tombstone_delete(locator):
            if Path(locator).name.startswith("clear-tombstone-"):
                raise RuntimeError("tombstone delete unavailable")
            return delete(locator)

        monkeypatch.setattr(store.guarded_handler_io.file_ops, "delete", fail_tombstone_delete)

        with pytest.raises(CacheBlobBackendError) as error:
            store.clear()

        assert isinstance(error.value.__cause__, RuntimeError)
        assert str(error.value.__cause__) == "tombstone delete unavailable"
        assert store.backend.list_entries() == []
        assert [store.get(key) for key in keys] == [None, None]
        assert all(not path.exists() for path in payload_paths)
        assert len(list(root.glob("clear-tombstone-*"))) == len(keys)
    finally:
        store.close()


def test_persisted_locator_raises_before_deserialization(tmp_path):
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

        # BlobStore reads the authenticated canonical manifest, not the legacy
        # backend projection. Re-sign this trusted fixture so locator containment
        # is the first rejected boundary rather than manifest integrity.
        raw_manifest = store.manifest_repository.get_raw(key)
        assert raw_manifest is not None
        manifest_data = BlobManifestV1.from_canonical_bytes(raw_manifest).to_mapping()
        manifest_data["locator"] = str(outside)
        manifest_data.pop("signature")
        unsigned_manifest = BlobManifestV1(**manifest_data)
        tampered_manifest = unsigned_manifest.with_signature(
            sign_hmac_sha256(unsigned_manifest.signing_bytes(), store._manifest_key())
        )
        store.manifest_repository.put_raw(
            key,
            tampered_manifest.canonical_bytes(),
            entry_data=entry,
        )

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
    assert all(handler.get_paths_alive)


def test_guarded_handler_io_copies_one_private_snapshot_without_deserializing(tmp_path):
    """The adapter owns the managed open/copy and yields a live private artifact."""
    from cacheness.storage.guarded_handler_io import GuardedHandlerIO

    root = tmp_path / "root"
    root.mkdir()
    handler = _InstrumentedHandler()
    io = GuardedHandlerIO(root)
    try:
        result = io.put(handler, "payload", "a" * 64, CacheConfig(cache_dir=str(root)))
        final_path = Path(result["actual_path"])

        assert _is_descendant(final_path, root)
        assert not _is_descendant(handler.put_paths[0], root)
        with io.open_snapshot(final_path, result["metadata"]) as snapshot:
            assert snapshot.path.exists()
            assert not _is_descendant(snapshot.path, root)
            assert snapshot.path.read_text(encoding="utf-8") == "payload"
            assert handler.get_paths == []
    finally:
        io.close()


def test_guarded_handler_io_rejects_handler_created_symlink_ancestor(tmp_path):
    """A returned stage artifact cannot tunnel through a handler-created link."""
    from cacheness.storage.guarded_handler_io import GuardedHandlerIO

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_artifact = outside / "payload.guarded"
    outside_artifact.write_text("outside", encoding="utf-8")

    class SymlinkStageHandler:
        data_type = "symlink-stage"

        def put(self, _data: Any, file_path: Path, _config: Any) -> dict[str, Any]:
            alias = file_path.parent / "alias"
            alias.symlink_to(outside, target_is_directory=True)
            return {
                "storage_format": "guarded",
                "file_size": outside_artifact.stat().st_size,
                "actual_path": str(alias / outside_artifact.name),
                "metadata": {},
            }

    io = GuardedHandlerIO(root)
    try:
        with pytest.raises(CacheUnsafePathError):
            io.put(
                SymlinkStageHandler(),
                "payload",
                "b" * 64,
                CacheConfig(cache_dir=str(root)),
            )
        assert list(root.iterdir()) == []
        assert outside_artifact.read_text(encoding="utf-8") == "outside"
    finally:
        io.close()


@pytest.mark.parametrize("path_kind", ["relative", "absolute"])
def test_guarded_handler_io_rejects_stage_parent_traversal(tmp_path, path_kind):
    """Handler artifacts cannot escape the private stage through ``..`` spelling."""
    from cacheness.storage.guarded_handler_io import GuardedHandlerIO

    root = tmp_path / "root"
    root.mkdir()

    class ParentTraversalHandler:
        data_type = "parent-traversal"

        def __init__(self):
            self.outside_artifact: Path | None = None

        def put(self, _data: Any, file_path: Path, _config: Any) -> dict[str, Any]:
            outside_artifact = file_path.parent.parent / "payload.guarded"
            outside_artifact.write_text("outside", encoding="utf-8")
            self.outside_artifact = outside_artifact
            actual_path: Path | str
            if path_kind == "relative":
                actual_path = Path("..") / outside_artifact.name
            else:
                actual_path = file_path.parent / ".." / outside_artifact.name
            return {
                "storage_format": "guarded",
                "file_size": outside_artifact.stat().st_size,
                "actual_path": actual_path,
                "metadata": {},
            }

    handler = ParentTraversalHandler()
    io = GuardedHandlerIO(root)
    try:
        with pytest.raises(CacheUnsafePathError):
            io.put(handler, "payload", "c" * 64, CacheConfig(cache_dir=str(root)))
        assert handler.outside_artifact is not None
        assert handler.outside_artifact.read_text(encoding="utf-8") == "outside"
        assert list(root.iterdir()) == []
    finally:
        io.close()


def test_guarded_handler_io_rejects_stage_artifact_swapped_after_validation(
    tmp_path, monkeypatch
):
    """Publication reads the verified descriptor, not a handler-swapped pathname."""
    from cacheness.storage.guarded_handler_io import GuardedHandlerIO

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.guarded"
    outside.write_text("outside", encoding="utf-8")
    handler = _InstrumentedHandler()
    io = GuardedHandlerIO(root)
    original_staged_artifact = GuardedHandlerIO._staged_artifact

    def swap_after_validation(
        stage_root: Path, stage_base: Path, result: dict[str, Any]
    ) -> Path:
        artifact = original_staged_artifact(stage_root, stage_base, result)
        artifact.unlink()
        artifact.symlink_to(outside)
        return artifact

    monkeypatch.setattr(
        GuardedHandlerIO,
        "_staged_artifact",
        staticmethod(swap_after_validation),
    )
    try:
        with pytest.raises(CacheUnsafePathError):
            io.put(handler, "payload", "d" * 64, CacheConfig(cache_dir=str(root)))
        assert outside.read_text(encoding="utf-8") == "outside"
        assert list(root.iterdir()) == []
    finally:
        io.close()


@pytest.mark.parametrize("force_fallback", [False, True])
def test_guarded_handler_io_rejects_ordinary_leaf_replacement_after_validation(
    tmp_path, monkeypatch, force_fallback: bool
):
    """A validated leaf inode cannot be replaced before managed publication."""
    from cacheness.storage import guarded_handler_io
    from cacheness.storage.guarded_handler_io import GuardedHandlerIO

    if not force_fallback and (
        os.open not in os.supports_dir_fd
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
    ):
        pytest.skip("descriptor-relative staged-artifact open is unsupported")
    if force_fallback:
        monkeypatch.setattr(guarded_handler_io.os, "supports_dir_fd", set())

    root = tmp_path / "root"
    root.mkdir()
    handler = _InstrumentedHandler()
    io = GuardedHandlerIO(root)
    original_staged_artifact = GuardedHandlerIO._staged_artifact

    def replace_leaf_after_validation(
        stage_root: Path, stage_base: Path, result: dict[str, Any]
    ) -> Path:
        artifact = original_staged_artifact(stage_root, stage_base, result)
        replacement = artifact.with_name("replacement.guarded")
        replacement.write_text("swapped", encoding="utf-8")
        replacement.replace(artifact)
        return artifact

    monkeypatch.setattr(
        GuardedHandlerIO,
        "_staged_artifact",
        staticmethod(replace_leaf_after_validation),
    )
    try:
        with pytest.raises(CacheUnsafePathError) as exc_info:
            io.put(handler, "payload", "g" * 64, CacheConfig(cache_dir=str(root)))
        assert exc_info.value.context["reason"] == CacheReason.PATH_RACE.value
        assert list(root.iterdir()) == []
    finally:
        io.close()


@pytest.mark.parametrize("force_fallback", [False, True])
def test_guarded_handler_io_rejects_ordinary_ancestor_replacement_after_validation(
    tmp_path, monkeypatch, force_fallback: bool
):
    """A validated ancestor directory cannot be replaced before publication."""
    from cacheness.storage import guarded_handler_io
    from cacheness.storage.guarded_handler_io import GuardedHandlerIO

    if not force_fallback and (
        os.open not in os.supports_dir_fd
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
    ):
        pytest.skip("descriptor-relative staged-artifact open is unsupported")
    if force_fallback:
        monkeypatch.setattr(guarded_handler_io.os, "supports_dir_fd", set())

    root = tmp_path / "root"
    root.mkdir()

    class NestedStageHandler:
        data_type = "nested-stage"

        def put(self, _data: Any, file_path: Path, _config: Any) -> dict[str, Any]:
            artifact = file_path.parent / "nested" / "payload.guarded"
            artifact.parent.mkdir()
            artifact.write_text("inside", encoding="utf-8")
            return {
                "storage_format": "guarded",
                "file_size": artifact.stat().st_size,
                "actual_path": artifact,
                "metadata": {},
            }

    io = GuardedHandlerIO(root)
    original_staged_artifact = GuardedHandlerIO._staged_artifact

    def replace_ancestor_after_validation(
        stage_root: Path, stage_base: Path, result: dict[str, Any]
    ) -> Path:
        artifact = original_staged_artifact(stage_root, stage_base, result)
        original_parent = artifact.parent
        original_parent.rename(original_parent.with_name("original-nested"))
        artifact.parent.mkdir()
        artifact.write_text("swapped", encoding="utf-8")
        return artifact

    monkeypatch.setattr(
        GuardedHandlerIO,
        "_staged_artifact",
        staticmethod(replace_ancestor_after_validation),
    )
    try:
        with pytest.raises(CacheUnsafePathError) as exc_info:
            io.put(NestedStageHandler(), "payload", "h" * 64, CacheConfig(cache_dir=str(root)))
        assert exc_info.value.context["reason"] == CacheReason.PATH_RACE.value
        assert list(root.iterdir()) == []
    finally:
        io.close()


def test_guarded_handler_io_rejects_hard_linked_external_stage_artifact(tmp_path):
    """A stage path may not alias external bytes through a hard link."""
    from cacheness.storage.guarded_handler_io import GuardedHandlerIO

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.guarded"
    outside.write_text("outside", encoding="utf-8")

    class HardLinkStageHandler:
        data_type = "hard-link-stage"

        def put(self, _data: Any, file_path: Path, _config: Any) -> dict[str, Any]:
            artifact = file_path.with_suffix(".guarded")
            os.link(outside, artifact)
            return {
                "storage_format": "guarded",
                "file_size": artifact.stat().st_size,
                "actual_path": artifact,
                "metadata": {},
            }

    io = GuardedHandlerIO(root)
    try:
        with pytest.raises(CacheUnsafePathError):
            io.put(
                HardLinkStageHandler(),
                "payload",
                "e" * 64,
                CacheConfig(cache_dir=str(root)),
            )
        assert outside.read_text(encoding="utf-8") == "outside"
        assert list(root.iterdir()) == []
    finally:
        io.close()


def test_guarded_handler_io_rejects_stage_ancestor_swapped_after_validation(
    tmp_path, monkeypatch
):
    """A swapped stage directory cannot redirect an absolute artifact path."""
    from cacheness.storage.guarded_handler_io import GuardedHandlerIO

    root = tmp_path / "root"
    root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_payload = outside_dir / "payload.guarded"
    outside_payload.write_text("outside", encoding="utf-8")

    class NestedStageHandler:
        data_type = "nested-stage"

        def put(self, _data: Any, file_path: Path, _config: Any) -> dict[str, Any]:
            artifact = file_path.parent / "nested" / "payload.guarded"
            artifact.parent.mkdir()
            artifact.write_text("inside", encoding="utf-8")
            return {
                "storage_format": "guarded",
                "file_size": artifact.stat().st_size,
                "actual_path": artifact,
                "metadata": {},
            }

    io = GuardedHandlerIO(root)
    original_staged_artifact = GuardedHandlerIO._staged_artifact

    def swap_ancestor_after_validation(
        stage_root: Path, stage_base: Path, result: dict[str, Any]
    ) -> Path:
        artifact = original_staged_artifact(stage_root, stage_base, result)
        shutil.rmtree(artifact.parent)
        artifact.parent.symlink_to(outside_dir, target_is_directory=True)
        return artifact

    monkeypatch.setattr(
        GuardedHandlerIO,
        "_staged_artifact",
        staticmethod(swap_ancestor_after_validation),
    )
    try:
        with pytest.raises(CacheUnsafePathError):
            io.put(
                NestedStageHandler(),
                "payload",
                "f" * 64,
                CacheConfig(cache_dir=str(root)),
            )
        assert outside_payload.read_text(encoding="utf-8") == "outside"
        assert list(root.iterdir()) == []
    finally:
        io.close()


def _instrumented_cache(tmp_path, *, delete_invalid_signatures: bool = True):
    """Build a cache whose handler makes guarded ordering observable."""
    root = tmp_path / "cache"
    handler = _InstrumentedHandler()
    cache = UnifiedCache(
        CacheConfig(
            cache_dir=str(root),
            metadata_backend="memory",
            cleanup_on_init=False,
            delete_invalid_signatures=delete_invalid_signatures,
        )
    )
    cache.handlers = _SingleHandlerRegistry(handler)
    return cache, handler


def test_high_level_handler_io_verifies_private_snapshot_before_handler(tmp_path):
    """Digest and signature verification precede deserialization on one snapshot."""
    cache, handler = _instrumented_cache(tmp_path)
    key = cache.put("payload", identity="ordered")
    events: list[str] = []
    original_hash = cache._calculate_file_hash
    original_verify = cache.signer.verify_entry

    def record_hash(path: Path):
        events.append("hash")
        assert not _is_descendant(path, cache.cache_dir)
        return original_hash(path)

    def record_verify(entry_data: dict[str, Any], signature: str):
        events.append("signature")
        return original_verify(entry_data, signature)

    cache._calculate_file_hash = record_hash
    cache.signer.verify_entry = record_verify

    assert cache.get(cache_key=key) == "payload"
    assert events == ["hash", "signature"]
    assert handler.events == ["handler"]
    assert handler.get_paths_alive == [True]


@pytest.mark.parametrize("rejection", ["hash", "signature", "legacy", "unsigned"])
def test_high_level_handler_io_rejects_untrusted_entries_before_deserialization(
    tmp_path, rejection
):
    """Bad integrity/current-or-legacy signatures never reach a handler."""
    cache, handler = _instrumented_cache(tmp_path, delete_invalid_signatures=False)
    key = cache.put("payload", identity=rejection)
    entry = cache.metadata_backend.get_entry(key)
    assert entry is not None
    metadata = entry["metadata"]

    if rejection == "hash":
        metadata["file_hash"] = "not-the-payload-hash"
    elif rejection == "signature":
        metadata["entry_signature"] = "not-a-valid-signature"
    elif rejection == "legacy":
        metadata.pop("entry_signature")
        metadata["legacy_entry_signature"] = "not-a-valid-legacy-signature"
    else:
        metadata.pop("entry_signature")
        cache.config.security.allow_unsigned_entries = False

    assert cache.get(cache_key=key) is None
    assert handler.events == []
    assert cache.metadata_backend.get_entry(key) is entry


def test_high_level_locator_preflight_blocks_multi_entry_mutation(tmp_path):
    """One hostile locator prevents list/clear/cleanup from touching safe siblings."""
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")

    blob_handler = _InstrumentedHandler()
    store = BlobStore(tmp_path / "blobs")
    store.handlers = _SingleHandlerRegistry(blob_handler)
    try:
        safe_key = store.put("safe", key="safe")
        unsafe_key = store.put("unsafe", key="unsafe")
        unsafe_entry = store.backend.get_entry(unsafe_key)
        assert unsafe_entry is not None
        unsafe_entry["metadata"]["actual_path"] = str(outside)
        store.backend.put_entry(unsafe_key, unsafe_entry)

        with pytest.raises(CacheUnsafePathError):
            store.list()
        with pytest.raises(CacheUnsafePathError):
            store.clear()
        assert store.get(safe_key) == "safe"
        assert store.get_metadata(safe_key) is not None
    finally:
        store.close()

    cache, cache_handler = _instrumented_cache(tmp_path / "unified")
    safe_key = cache.put("safe", identity="safe")
    unsafe_key = cache.put("unsafe", identity="unsafe")
    unsafe_entry = cache.metadata_backend.get_entry(unsafe_key)
    assert unsafe_entry is not None
    unsafe_entry["metadata"]["actual_path"] = str(outside)

    stats_before = cache.metadata_backend.get_stats()
    with pytest.raises(CacheUnsafePathError):
        cache.list_entries()
    with pytest.raises(CacheUnsafePathError):
        cache.clear_all()
    with pytest.raises(CacheUnsafePathError):
        cache._cleanup_expired()
    with pytest.raises(CacheUnsafePathError):
        cache._enforce_size_limit()
    with pytest.raises(CacheUnsafePathError):
        cache.get(cache_key=unsafe_key)
    assert cache.metadata_backend.get_stats() == stats_before
    assert cache.get(cache_key=safe_key) == "safe"
    assert cache_handler.events == ["handler"]
    assert outside.read_text(encoding="utf-8") == "outside"
