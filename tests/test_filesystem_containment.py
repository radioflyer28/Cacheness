"""Security contract tests for filesystem-backed blob containment."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
import shutil
import socket
import stat
import subprocess
import threading
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from cacheness.config import CacheConfig
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobRecoverableCleanupError,
    CacheReason,
    CacheStorageError,
    CacheUnsafePathError,
)
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology
from cacheness.storage.backends.blob_backends import FilesystemBlobBackend
from cacheness.storage.integrity import sign_hmac_sha256
from cacheness.storage.manifest import BlobManifest
from cacheness.storage import path_security
from cacheness.storage.path_security import (
    ManagedFileOps,
    _WindowsFileApi,
    resolve_managed_locator,
    resolve_storage_root,
    validate_blob_id,
)


def _store(root: Path) -> BlobStore:
    """Create the qualified topology for direct containment regressions."""
    return BlobStore(
        StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        ),
        cache_dir=root,
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


@pytest.mark.parametrize(
    ("error_number", "error_type"),
    [
        (2, FileNotFoundError),
        (3, FileNotFoundError),
        (80, FileExistsError),
        (183, FileExistsError),
        (5, OSError),
    ],
)
def test_windows_file_api_maps_native_error_codes_without_constructing_win32(
    monkeypatch: pytest.MonkeyPatch,
    error_number: int,
    error_type: type[OSError],
) -> None:
    """The injectable native-error seam never masks an OS failure with AttributeError."""
    monkeypatch.setattr(_WindowsFileApi, "_last_error", staticmethod(lambda: error_number))

    with pytest.raises(error_type) as captured:
        _WindowsFileApi._raise_last_error("test-native-operation")

    assert captured.value.errno == error_number


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


@pytest.mark.skipif(os.name != "posix", reason="POSIX special-node fixtures")
@pytest.mark.parametrize("node_kind", ("fifo", "socket", "directory"))
def test_managed_reads_reject_special_nodes_before_they_can_block(
    tmp_path: Path, node_kind: str
) -> None:
    """No managed read, size check, or existence check accepts a special node."""
    root = tmp_path / f"special-{node_kind}"
    root.mkdir()
    locator = root / "control"
    socket_handle: socket.socket | None = None
    if node_kind == "fifo":
        os.mkfifo(locator)
    elif node_kind == "socket":
        socket_handle = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        original_directory = Path.cwd()
        try:
            os.chdir(root)
            try:
                socket_handle.bind(locator.name)
            except PermissionError:
                socket_handle.close()
                socket_handle = None
                pytest.skip("AF_UNIX socket creation is unavailable in this sandbox")
        finally:
            os.chdir(original_directory)
    else:
        locator.mkdir()

    operations = ManagedFileOps(root)
    try:
        for action in (
            lambda: operations.read_bytes(locator),
            lambda: operations.read_bytes_bounded(locator, max_bytes=128),
            lambda: operations.open_read(locator),
            lambda: operations.exists(locator),
            lambda: operations.get_size(locator),
        ):
            with pytest.raises(CacheUnsafePathError):
                action()
    finally:
        operations.close()
        if socket_handle is not None:
            socket_handle.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX device-node fixture")
def test_managed_reads_reject_a_device_before_evidence_parsing(tmp_path: Path) -> None:
    """A device authority node cannot reach a parser or block initialization."""
    root = tmp_path / "special-device"
    root.mkdir()
    locator = root / "device"
    try:
        os.mknod(locator, stat.S_IFCHR | 0o600, os.makedev(1, 3))
    except (AttributeError, OSError, PermissionError):
        pytest.skip("device-node creation is unavailable to this test user")

    operations = ManagedFileOps(root)
    try:
        with pytest.raises(CacheUnsafePathError):
            operations.read_bytes_bounded(locator, max_bytes=128)
    finally:
        operations.close()


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
    """Return immutable authority-generation payloads for one logical key."""
    storage_id = store._storage_id_for_key(key)
    generation_dir = store.cache_dir / "generations" / storage_id
    return sorted(generation_dir.glob("*")) if generation_dir.exists() else []


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
    store = _store(root)
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
        actual_path = store.cache_dir / entry["metadata"]["actual_path"]
        assert _is_descendant(actual_path, root)
        assert logical_key not in str(actual_path)
    finally:
        store.close()


@pytest.mark.parametrize("handler_index", [0, 1])
def test_blob_store_first_write_authority_conflict_removes_candidate(
    tmp_path, monkeypatch, handler_index
):
    """A first-write CAS loss leaves no entry, candidate, or operation residue."""
    root = tmp_path / "blob-root"
    handlers = _format_handlers()
    registry = _SwitchingHandlerRegistry(*handlers)
    registry.current = handlers[handler_index]
    store = _store(root)
    store.handlers = registry

    def reject_authority_promotion(*_args: Any, **_kwargs: Any) -> None:
        raise CacheBlobLifecycleConflictError("authority promotion lost")

    try:
        monkeypatch.setattr(
            store.lifecycle_authority,
            "promote_mutation",
            reject_authority_promotion,
        )

        with pytest.raises(CacheBlobLifecycleConflictError, match="authority promotion lost"):
            store.put("replacement", key="first-write")

        assert store.get_metadata("first-write") is None
        assert _payloads_for_key(store, "first-write") == []
        assert store.lifecycle_authority.pending_mutations() == ()
        assert store.lifecycle_authority.pending_cleanup_debts() == ()
    finally:
        store.close()


def test_blob_store_cross_format_authority_conflict_preserves_prior_evidence(
    tmp_path, monkeypatch
):
    """A failed replacement CAS preserves exact old bytes and authority."""
    root = tmp_path / "blob-root"
    old_handler, replacement_handler = _format_handlers()
    registry = _SwitchingHandlerRegistry(old_handler, replacement_handler)
    store = _store(root)
    store.handlers = registry
    key = "cross-format"
    initial_metadata = {
        "serializer": old_handler.serializer,
        "handler_compression": old_handler.compression,
    }

    def reject_authority_promotion(*_args: Any, **_kwargs: Any) -> None:
        raise CacheBlobLifecycleConflictError("authority promotion lost")

    try:
        store.put("old value", key=key, metadata=initial_metadata)
        entry_before = deepcopy(store.get_metadata(key))
        assert entry_before is not None
        old_path = root / entry_before["metadata"]["actual_path"]
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
        monkeypatch.setattr(
            store.lifecycle_authority,
            "promote_mutation",
            reject_authority_promotion,
        )

        with pytest.raises(CacheBlobLifecycleConflictError, match="authority promotion lost"):
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
        assert store.lifecycle_authority.pending_mutations() == ()
    finally:
        store.close()


@pytest.mark.parametrize("cleanup_outcome", ["false", "raise"])
def test_blob_store_candidate_cleanup_failure_is_explicit_and_chained(
    tmp_path, monkeypatch, cleanup_outcome
):
    """A CAS-loser candidate remains recoverable when cleanup cannot be proved."""
    root = tmp_path / "blob-root"
    handler, _ = _format_handlers()
    store = _store(root)
    store.handlers = _SwitchingHandlerRegistry(handler)
    cleanup_attempts: list[Path] = []

    def reject_authority_promotion(*_args: Any, **_kwargs: Any) -> None:
        raise CacheBlobLifecycleConflictError("authority promotion lost")

    def cannot_prove_cleanup(locator: Path) -> None:
        cleanup_attempts.append(Path(locator))
        if cleanup_outcome == "raise":
            raise OSError("candidate cleanup unavailable")
        raise CacheStorageError("candidate cleanup could not be proved")

    try:
        monkeypatch.setattr(
            store.lifecycle_authority,
            "promote_mutation",
            reject_authority_promotion,
        )
        monkeypatch.setattr(
            store,
            "_delete_or_prove_absent",
            cannot_prove_cleanup,
        )

        with pytest.raises(CacheBlobRecoverableCleanupError) as exc_info:
            store.put("replacement", key="cleanup-proof")

        assert exc_info.value.context["operation_id"]
        assert len(cleanup_attempts) == 1
        assert store.get_metadata("cleanup-proof") is None
        candidates = _payloads_for_key(store, "cleanup-proof")
        assert len(candidates) == 1
        debt = store.lifecycle_authority.pending_cleanup_debts()
        assert len(debt) == 1
        assert debt[0].locator == candidates[0].relative_to(root).as_posix()
    finally:
        store.close()


def test_blob_store_post_commit_prior_cleanup_keeps_new_metadata_authoritative(
    tmp_path, monkeypatch
):
    """A failed old-payload cleanup cannot restore stale replacement metadata."""
    root = tmp_path / "blob-root"
    old_handler, replacement_handler = _format_handlers()
    registry = _SwitchingHandlerRegistry(old_handler, replacement_handler)
    store = _store(root)
    store.handlers = registry
    key = "post-commit"

    try:
        store.put("old value", key=key)
        entry_before = store.get_metadata(key)
        assert entry_before is not None
        old_locator = Path(entry_before["metadata"]["actual_path"])
        old_path = root / old_locator
        delete_or_prove_absent = store._delete_or_prove_absent

        def fail_old_payload_cleanup(locator: Path) -> None:
            if locator == old_locator:
                raise CacheStorageError("old payload cleanup unavailable")
            delete_or_prove_absent(locator)

        registry.current = replacement_handler
        monkeypatch.setattr(
            store,
            "_delete_or_prove_absent",
            fail_old_payload_cleanup,
        )

        with pytest.raises(CacheStorageError, match="cleanup"):
            store.put("replacement value", key=key)

        entry_after = store.get_metadata(key)
        assert entry_after is not None
        new_path = root / entry_after["metadata"]["actual_path"]
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
    store = _store(root)
    store.handlers = _SingleHandlerRegistry(handler)

    try:
        first_key = store.put("first", key="first")
        second_key = store.put("second", key="second")
        payload_paths = [
            root / store.get_metadata(key)["metadata"]["actual_path"]
            for key in (first_key, second_key)
        ]

        assert all(path.exists() for path in payload_paths)
        assert store.clear() == 2

        assert all(not path.exists() for path in payload_paths)
        assert store.lifecycle_authority.list_entries() == ()
        assert store.get(first_key) is None
        assert store.get(second_key) is None
    finally:
        store.close()


@pytest.mark.parametrize("failing_payload_delete", [1, 2])
def test_blob_store_clear_keeps_tombstone_authority_when_reclamation_fails(
    tmp_path, monkeypatch, failing_payload_delete
):
    """A post-authority clear failure retains signed recovery evidence to converge."""
    root = tmp_path / "blob-root"
    handler = _InstrumentedHandler()
    store = _store(root)
    store.handlers = _SingleHandlerRegistry(handler)

    failed_key: str | None = None
    try:
        keys = [store.put("first", key="first"), store.put("second", key="second")]
        entries_before = [store.get_metadata(key) for key in keys]
        assert all(entry is not None for entry in entries_before)
        payload_paths = [
            root / entry["metadata"]["actual_path"]
            for entry in entries_before
            if entry is not None
        ]
        delete = store.guarded_handler_io.file_ops.delete
        payload_delete_count = 0

        def fail_one_payload_delete(locator):
            nonlocal payload_delete_count
            if root / Path(locator) in payload_paths:
                payload_delete_count += 1
                if payload_delete_count == failing_payload_delete:
                    raise RuntimeError("payload delete unavailable")
            return delete(locator)

        monkeypatch.setattr(store.guarded_handler_io.file_ops, "delete", fail_one_payload_delete)

        with pytest.raises(CacheBlobBackendError) as error:
            store.clear()

        assert error.value.context["operation"] == "clear"
        assert isinstance(error.value.__cause__, CacheBlobRecoverableCleanupError)
        assert isinstance(error.value.__cause__.__cause__, CacheStorageError)
        failed_key = keys[failing_payload_delete - 1]
        entry = store.lifecycle_authority.read_entry(failed_key)
        assert entry is not None
        tombstone = BlobManifest.from_canonical_bytes(entry.manifest)
        assert tombstone.state == "tombstoned"
        assert tombstone.signature
        assert store.get(failed_key) is None
        assert payload_paths[failing_payload_delete - 1].exists()
        assert store.lifecycle_authority.pending_cleanup_debts()
    finally:
        store.close()

    assert failed_key is not None
    reopened = _store(root)
    reopened.handlers = _SingleHandlerRegistry(_InstrumentedHandler())
    try:
        assert reopened.reconcile(apply=True).applied is True
        assert all(reopened.get(key) is None for key in keys[:failing_payload_delete])
        assert [reopened.get(key) for key in keys[failing_payload_delete:]] == [
            "first",
            "second",
        ][failing_payload_delete:]
        assert all(not path.exists() for path in payload_paths[:failing_payload_delete])
        assert all(path.exists() for path in payload_paths[failing_payload_delete:])
        assert reopened.lifecycle_authority.pending_cleanup_debts() == ()
    finally:
        reopened.close()


def test_blob_store_clear_commits_authority_tombstones(tmp_path):
    """Current clear removes each entry through the selected authority lifecycle."""
    root = tmp_path / "blob-root"
    handler = _InstrumentedHandler()
    store = _store(root)
    store.handlers = _SingleHandlerRegistry(handler)

    try:
        keys = [store.put("first", key="first"), store.put("second", key="second")]
        entries_before = [store.get_metadata(key) for key in keys]
        assert all(entry is not None for entry in entries_before)
        payload_paths = [
            root / entry["metadata"]["actual_path"]
            for entry in entries_before
            if entry is not None
        ]
        assert store.clear() == len(keys)
        assert all(store.get(key) is None for key in keys)
        assert all(not path.exists() for path in payload_paths)
        assert store.lifecycle_authority.list_entries() == ()
    finally:
        store.close()


def test_blob_store_clear_pre_authority_fault_preserves_committed_payload(
    tmp_path,
):
    """A pre-promotion clear fault aborts without revoking the committed entry."""
    root = tmp_path / "blob-root"
    handler = _InstrumentedHandler()
    store = _store(root)
    store.handlers = _SingleHandlerRegistry(handler)

    try:
        key = store.put("payload", key="entry")
        entry_before = store.get_metadata(key)
        assert entry_before is not None
        payload_path = root / entry_before["metadata"]["actual_path"]
        payload_bytes = payload_path.read_bytes()
        authority_before = store.lifecycle_authority.read_entry(key)
        assert authority_before is not None

        def interrupt_before_tombstone_authority(seam):
            if seam == "delete.before_tombstone_promotion":
                raise RuntimeError("tombstone publication unavailable")

        store.lifecycle.fault_hook = interrupt_before_tombstone_authority

        with pytest.raises(RuntimeError, match="tombstone publication unavailable"):
            store.clear()

        assert store.get_metadata(key) == entry_before
        assert payload_path.read_bytes() == payload_bytes
        assert store.lifecycle_authority.read_entry(key) == authority_before
        assert store.get(key) == "payload"
        assert store.lifecycle_authority.pending_mutations() == ()
        assert store.lifecycle_authority.pending_cleanup_debts() == ()
    finally:
        store.close()

    reopened = _store(root)
    reopened.handlers = _SingleHandlerRegistry(_InstrumentedHandler())
    try:
        assert reopened.get(key) == "payload"
        assert payload_path.exists()
    finally:
        reopened.close()


def test_blob_store_clear_records_tombstone_before_current_payload_cleanup_fails(
    tmp_path,
):
    """A clear records signed absence before a post-authority cleanup failure."""
    root = tmp_path / "blob-root"
    handler = _InstrumentedHandler()
    store = _store(root)
    store.handlers = _SingleHandlerRegistry(handler)

    interrupted_key: str | None = None
    try:
        keys = [store.put("first", key="first"), store.put("second", key="second")]
        payload_paths = [
            root / store.get_metadata(key)["metadata"]["actual_path"]
            for key in keys
        ]

        def interrupt_post_authority_cleanup(seam):
            nonlocal interrupted_key
            if seam != "cleanup.before_payload_delete" or interrupted_key is not None:
                return
            entries = store.lifecycle_authority.list_entries()
            tombstone_entry = next(
                entry
                for entry in entries
                if entry.key in keys
                and BlobManifest.from_canonical_bytes(entry.manifest).state
                == "tombstoned"
            )
            tombstone = BlobManifest.from_canonical_bytes(tombstone_entry.manifest)
            assert tombstone.state == "tombstoned"
            assert tombstone.signature
            interrupted_key = tombstone_entry.key
            raise RuntimeError("payload cleanup unavailable")

        store.lifecycle.fault_hook = interrupt_post_authority_cleanup

        with pytest.raises(CacheBlobBackendError) as error:
            store.clear()

        assert error.value.context["operation"] == "clear"
        assert isinstance(error.value.__cause__, CacheBlobRecoverableCleanupError)
        assert interrupted_key is not None
        assert store.get(interrupted_key) is None
        assert payload_paths[keys.index(interrupted_key)].exists()
        assert store.lifecycle_authority.pending_cleanup_debts()
    finally:
        store.close()

    reopened = _store(root)
    reopened.handlers = _SingleHandlerRegistry(_InstrumentedHandler())
    try:
        assert reopened.reconcile(apply=True).applied is True
        assert reopened.get(interrupted_key) is None
        assert not payload_paths[keys.index(interrupted_key)].exists()
        pending_key = next(key for key in keys if key != interrupted_key)
        assert reopened.get(pending_key) == "second"
        assert payload_paths[keys.index(pending_key)].exists()
    finally:
        reopened.close()


def test_persisted_locator_raises_before_deserialization(tmp_path, monkeypatch):
    """Unsafe persisted locators remain typed errors, not reads or cache misses."""
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")

    blob_handler = _InstrumentedHandler()
    store = _store(tmp_path / "blobs")
    store.handlers = _SingleHandlerRegistry(blob_handler)
    try:
        key = store.put("inside", key="safe")
        entry = store.lifecycle_authority.read_entry(key)
        assert entry is not None

        # Inject a re-signed authority entry rather than a projection.  Locator
        # containment must reject it before a handler sees any payload bytes.
        manifest_data = BlobManifest.from_canonical_bytes(entry.manifest).to_mapping()
        manifest_data["locator"] = str(outside)
        manifest_data.pop("signature")
        unsigned_manifest = BlobManifest.from_mapping({**manifest_data, "signature": ""})
        tampered_manifest = unsigned_manifest.with_signature(
            sign_hmac_sha256(
                unsigned_manifest.signing_bytes(), store._authority_manifest_key()
            )
        )
        tampered_raw = tampered_manifest.canonical_bytes()
        tampered_entry = replace(
            entry,
            locator=str(outside),
            manifest=tampered_raw,
            expectation=replace(
                entry.expectation,
                manifest_digest=hashlib.sha256(tampered_raw).hexdigest(),
            ),
        )
        monkeypatch.setattr(
            store.lifecycle_authority,
            "read_entry",
            lambda requested_key: tampered_entry if requested_key == key else None,
        )

        with pytest.raises(CacheUnsafePathError):
            store.get(key)
        assert blob_handler.get_paths == []
        assert outside.read_text(encoding="utf-8") == "outside"
    finally:
        store.close()

def test_blob_store_encodes_hostile_key_without_mutating_outside_target(tmp_path):
    """An authored key remains logical while only an encoded name reaches storage."""
    root = tmp_path / "cache"
    handler = _InstrumentedHandler()
    store = _store(root)
    store.handlers = _SingleHandlerRegistry(handler)
    key = "../outside prefix"

    try:
        assert store.put("payload", key=key) == key
        entry = store.get_metadata(key)

        assert entry is not None
        actual_path = root / entry["metadata"]["actual_path"]
        assert _is_descendant(actual_path, root)
        assert key not in str(actual_path)
        assert store.get(key) == "payload"
        assert all(not _is_descendant(path, root) for path in handler.put_paths)
        assert all(not _is_descendant(path, root) for path in handler.get_paths)
        assert all(handler.get_paths_alive)
    finally:
        store.close()


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


def test_locator_preflight_blocks_multi_entry_mutation(tmp_path, monkeypatch):
    """Unsafe signed authority locators fail closed before multi-entry mutation."""
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")

    blob_handler = _InstrumentedHandler()
    store = _store(tmp_path / "blobs")
    store.handlers = _SingleHandlerRegistry(blob_handler)
    try:
        safe_key = store.put("safe", key="safe")
        unsafe_key = store.put("unsafe", key="unsafe")
        unsafe_entry = store.lifecycle_authority.read_entry(unsafe_key)
        assert unsafe_entry is not None
        manifest_data = BlobManifest.from_canonical_bytes(
            unsafe_entry.manifest
        ).to_mapping()
        manifest_data["locator"] = str(outside)
        manifest_data.pop("signature")
        unsigned_manifest = BlobManifest.from_mapping({**manifest_data, "signature": ""})
        tampered_raw = unsigned_manifest.with_signature(
            sign_hmac_sha256(
                unsigned_manifest.signing_bytes(), store._authority_manifest_key()
            )
        ).canonical_bytes()
        tampered_entry = replace(
            unsafe_entry,
            locator=str(outside),
            manifest=tampered_raw,
            expectation=replace(
                unsafe_entry.expectation,
                manifest_digest=hashlib.sha256(tampered_raw).hexdigest(),
            ),
        )
        original_read_entry = store.lifecycle_authority.read_entry
        original_list_entries = store.lifecycle_authority.list_entries
        monkeypatch.setattr(
            store.lifecycle_authority,
            "read_entry",
            lambda key: tampered_entry if key == unsafe_key else original_read_entry(key),
        )
        monkeypatch.setattr(
            store.lifecycle_authority,
            "list_entries",
            lambda: tuple(
                tampered_entry if entry.key == unsafe_key else entry
                for entry in original_list_entries()
            ),
        )

        with pytest.raises(CacheUnsafePathError):
            store.list()
        with pytest.raises(CacheBlobBackendError) as clear_error:
            store.clear()
        assert isinstance(clear_error.value.__cause__, CacheUnsafePathError)
        assert store.get(safe_key) == "safe"
        with pytest.raises(CacheUnsafePathError):
            store.get(unsafe_key)
        assert outside.read_text(encoding="utf-8") == "outside"
    finally:
        store.close()
