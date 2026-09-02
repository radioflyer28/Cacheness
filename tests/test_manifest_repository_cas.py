"""Exact-record conditional publication contracts for local manifests."""

from __future__ import annotations

import builtins
import ctypes
import errno
import hashlib
import multiprocessing
import os
from pathlib import Path
from threading import Barrier, Lock, Thread

import pytest

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobMigrationRequiredError,
    CacheReason,
    CacheUnsafePathError,
)
from cacheness.metadata import InMemoryBackend, JsonBackend, SqliteBackend
from cacheness.storage.manifest_repository import (
    InMemoryManifestRepository,
    JsonManifestRepository,
    ManifestExpectation,
    SqliteManifestRepository,
)
from cacheness.storage import coordination
from cacheness.storage import path_security
from cacheness.storage import manifest_repository as manifest_repository_module
from cacheness.storage.path_security import ManagedFileOps, resolve_managed_locator
from cacheness.storage.operation_repository import FileOperationRecordRepository


def _record(label: str) -> bytes:
    """Return deliberately opaque canonical-record stand-ins for repository tests."""
    return f"canonical-manifest-record::{label}".encode("utf-8")


def test_pending_recovery_filters_unrelated_names_before_its_action_bound(
    tmp_path: Path,
) -> None:
    """Malformed and ordinary siblings cannot starve one valid pending record."""
    root = tmp_path / "pending-recovery-filter"
    root.mkdir()
    limits = LifecycleLimits(max_reconcile_actions=1)
    file_ops = ManagedFileOps(root)
    repository = FileOperationRecordRepository(file_ops, lifecycle_limits=limits)
    operation_id = "f" * 32
    raw = b'{"pending":"exact"}'
    digest = hashlib.sha256(raw).hexdigest()
    operations = root / "operations"
    operations.mkdir()
    # These sort before the valid pending name but are not eligible controls.
    (operations / ".000-malformed.tmp").write_bytes(b"noise")
    (operations / ".111.json.pending.not-a-digest.tmp").write_bytes(b"noise")
    (operations / "clear-target-page-not-an-operation.json").write_bytes(b"noise")
    pending = operations / f".{operation_id}.json.pending.{digest}.{'a' * 32}.tmp"
    pending.write_bytes(raw)
    try:
        assert repository.recover_pending_operation_records() == (operation_id,)
        assert repository.get_raw(operation_id) == raw
        # Repeated bounded reopen/recovery calls do not get stuck on siblings.
        assert repository.recover_pending_operation_records() == ()
    finally:
        repository.close()
        file_ops.close()


def test_directory_inventory_is_lexical_after_reverse_creation_and_has_a_distinct_bound(
    tmp_path: Path,
) -> None:
    """A lexical cursor never combines a partial scandir order with its position."""
    root = tmp_path / "stable-directory-inventory"
    root.mkdir()
    file_ops = ManagedFileOps(root)
    directory = root / "operations"
    directory.mkdir()
    for name in ("z.json", "a.json", "b.json"):
        (directory / name).write_bytes(name.encode("ascii"))
    try:
        first, cursor = file_ops.list_directory_names_bounded(
            directory,
            cursor=None,
            max_names=2,
            max_inventory_names=3,
            operation="test_inventory",
        )
        second, terminal = file_ops.list_directory_names_bounded(
            directory,
            cursor=cursor,
            max_names=2,
            max_inventory_names=3,
            operation="test_inventory",
        )
        assert first == ("a.json", "b.json")
        assert cursor == "b.json"
        assert second == ("z.json",)
        assert terminal is None
    finally:
        file_ops.close()


def test_preindex_inventory_refuses_unbounded_unrelated_legacy_namespace(
    tmp_path: Path,
) -> None:
    """An upgrade never scans an unbounded directory merely to prove it empty."""
    root = tmp_path / "legacy-unrelated-namespace"
    root.mkdir()
    operations = root / "operations"
    operations.mkdir()
    for index in range(3):
        (operations / f"unrelated-{index}").write_bytes(b"noise")
    file_ops = ManagedFileOps(root)
    repository = FileOperationRecordRepository(
        file_ops, lifecycle_limits=LifecycleLimits(max_inventory_items=2)
    )
    try:
        with pytest.raises(CacheBlobMigrationRequiredError):
            repository.list_page()
    finally:
        repository.close()
        file_ops.close()


def _race_json_manifest_cas_process(
    metadata_path: str,
    record: bytes,
    ready: multiprocessing.queues.Queue,
    release: multiprocessing.synchronize.Event,
    outcomes: multiprocessing.queues.Queue,
) -> None:
    """Race a fresh JSON authority repository without shared Python locks."""
    backend = JsonBackend(metadata_path)
    repository = JsonManifestRepository(backend)
    try:
        ready.put("ready")
        if not release.wait(timeout=10):
            outcomes.put("timeout")
            return
        repository.publish_if_expected("key", ManifestExpectation.absent(), record)
        outcomes.put("won")
    except CacheBlobLifecycleConflictError:
        outcomes.put("conflict")
    except BaseException as exc:  # pragma: no cover - surfaced by parent assertion.
        outcomes.put(f"error:{exc!r}")
    finally:
        repository.close()
        backend.close()


def _publish_json_while_parent_swaps_lock_name(
    metadata_path: str,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
    outcomes: multiprocessing.queues.Queue,
) -> None:
    """Hold the original descriptor while a parent swaps only its lock pathname."""
    backend = JsonBackend(metadata_path)
    repository = JsonManifestRepository(backend)
    repository.after_json_lock_acquisition = lambda: (ready.set(), release.wait(timeout=10))
    try:
        repository.publish_if_expected(
            "key", ManifestExpectation.absent(), _record("original-authority")
        )
        outcomes.put("original-won")
    except BaseException as exc:  # pragma: no cover - surfaced by parent assertion.
        outcomes.put(f"original-error:{exc!r}")
    finally:
        repository.close()
        backend.close()


def _attempt_json_publish_after_lock_swap(
    metadata_path: str, outcomes: multiprocessing.queues.Queue
) -> None:
    """A later opener must reject the replacement lock before it can enter CAS."""
    backend = JsonBackend(metadata_path)
    try:
        repository = JsonManifestRepository(backend)
    except CacheUnsafePathError:
        outcomes.put("replacement-blocked")
    except BaseException as exc:  # pragma: no cover - surfaced by parent assertion.
        outcomes.put(f"replacement-error:{exc!r}")
    else:
        try:
            repository.publish_if_expected(
                "key", ManifestExpectation.absent(), _record("replacement-authority")
            )
            outcomes.put("replacement-won")
        except CacheBlobLifecycleConflictError:
            outcomes.put("replacement-conflict")
        finally:
            repository.close()
    finally:
        backend.close()


def _repository_pair(tmp_path: Path, backend_name: str):
    """Build independently constructed repositories over one local topology."""
    if backend_name == "memory":
        backend = InMemoryBackend()
        return (
            InMemoryManifestRepository(backend),
            InMemoryManifestRepository(backend),
            (backend,),
        )
    if backend_name == "json":
        metadata_path = tmp_path / "metadata.json"
        first_backend = JsonBackend(metadata_path)
        second_backend = JsonBackend(metadata_path)
        return (
            JsonManifestRepository(first_backend),
            JsonManifestRepository(second_backend),
            (first_backend, second_backend),
        )
    metadata_path = tmp_path / "metadata.db"
    first_backend = SqliteBackend(metadata_path)
    second_backend = SqliteBackend(metadata_path)
    return (
        SqliteManifestRepository(first_backend),
        SqliteManifestRepository(second_backend),
        (first_backend, second_backend),
    )


@pytest.mark.parametrize("backend_name", ("memory", "json", "sqlite"))
def test_create_if_absent_has_one_independent_repository_winner(
    tmp_path: Path, backend_name: str
) -> None:
    """A second absence contender loses without altering the exact winner bytes."""
    first_repo, second_repo, backends = _repository_pair(tmp_path, backend_name)
    winner = _record("winner")
    loser = _record("loser")
    try:
        first_repo.publish_if_expected(
            "key", ManifestExpectation.absent(), winner
        )

        with pytest.raises(CacheBlobLifecycleConflictError):
            second_repo.publish_if_expected(
                "key", ManifestExpectation.absent(), loser
            )

        assert first_repo.get_raw("key") == winner
        assert second_repo.get_raw("key") == winner
    finally:
        for backend in backends:
            backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json", "sqlite"))
def test_exact_record_cas_rejects_same_generation_stale_patch(
    tmp_path: Path, backend_name: str
) -> None:
    """A record digest prevents last-write-wins updates within one generation."""
    first_repo, second_repo, backends = _repository_pair(tmp_path, backend_name)
    original = _record("generation-one-metadata-one")
    winner = _record("generation-one-metadata-two")
    stale_loser = _record("generation-one-metadata-three")
    expectation = ManifestExpectation.from_authenticated_record("generation-one", original)
    try:
        first_repo.put_raw("key", original)
        first_repo.publish_if_expected("key", expectation, winner)

        with pytest.raises(CacheBlobLifecycleConflictError):
            second_repo.publish_if_expected("key", expectation, stale_loser)

        assert first_repo.get_raw("key") == winner
        assert second_repo.get_raw("key") == winner
    finally:
        for backend in backends:
            backend.close()


def test_json_cas_has_one_exact_cross_process_winner(tmp_path: Path) -> None:
    """The JSON authority boundary is a real interprocess winner selection."""
    metadata_path = tmp_path / "concurrent.json"
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    release = context.Event()
    outcomes = context.Queue()
    records = (_record("process-a"), _record("process-b"))
    workers = [
        context.Process(
            target=_race_json_manifest_cas_process,
            args=(str(metadata_path), record, ready, release, outcomes),
        )
        for record in records
    ]
    for worker in workers:
        worker.start()
    try:
        assert ready.get(timeout=10) == "ready"
        assert ready.get(timeout=10) == "ready"
        release.set()
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
        assert sorted(outcomes.get(timeout=10) for _ in workers) == ["conflict", "won"]

        backend = JsonBackend(metadata_path)
        repository = JsonManifestRepository(backend)
        try:
            assert repository.get_raw("key") in records
        finally:
            repository.close()
            backend.close()
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


@pytest.mark.parametrize(
    "error_number",
    (
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
        errno.ENOSYS,
        errno.ENOTTY,
        errno.EINVAL,
        errno.EACCES,
        errno.EPERM,
        errno.EROFS,
    ),
)
def test_root_xattr_capability_and_policy_failures_are_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_number: int
) -> None:
    """Required root-xattr capability failures never escape as raw OS errors."""
    root = tmp_path / "xattr-capability"
    root.mkdir()
    file_ops = ManagedFileOps(root)
    locator = root / ".authority.lock"
    injected = OSError(error_number, "xattr unavailable")

    def fail_set(_name: str, _authority: bytes) -> None:
        raise injected

    monkeypatch.setattr(file_ops, "_set_root_authority_xattr", fail_set)
    try:
        with pytest.raises(CacheBlobBackendError) as error:
            file_ops._ensure_root_authority_binding(locator, b"authority")
        assert error.value.context["reason"] == (
            CacheReason.BLOB_BACKEND_CAPABILITY_UNSUPPORTED.value
        )
        assert error.value.context["capability"] == "root_xattr"
        assert error.value.context["errno"] == error_number
        assert error.value.__cause__ is injected
    finally:
        file_ops.close()


def test_root_xattr_api_unavailability_is_a_typed_capability_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Python runtime without xattr APIs has the same stable public outcome."""
    root = tmp_path / "xattr-api-unavailable"
    root.mkdir()
    file_ops = ManagedFileOps(root)
    locator = root / ".authority.lock"
    injected = AttributeError("setxattr unavailable")

    def fail_set(_name: str, _authority: bytes) -> None:
        raise injected

    monkeypatch.setattr(file_ops, "_set_root_authority_xattr", fail_set)
    try:
        with pytest.raises(CacheBlobBackendError) as error:
            file_ops._ensure_root_authority_binding(locator, b"authority")
        assert error.value.context == {
            "operation": "lifecycle_lock_authority",
            "capability": "root_xattr",
            "reason": CacheReason.BLOB_BACKEND_CAPABILITY_UNSUPPORTED.value,
        }
        assert error.value.__cause__ is injected
    finally:
        file_ops.close()


def test_root_xattr_operational_failure_is_a_typed_backend_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unexpected xattr I/O is still stable at the BlobStore boundary."""
    root = tmp_path / "xattr-operational-failure"
    root.mkdir()
    file_ops = ManagedFileOps(root)
    locator = root / ".authority.lock"
    injected = OSError(errno.EIO, "xattr I/O failed")

    def fail_set(_name: str, _authority: bytes) -> None:
        raise injected

    monkeypatch.setattr(file_ops, "_set_root_authority_xattr", fail_set)
    try:
        with pytest.raises(CacheBlobBackendError) as error:
            file_ops._ensure_root_authority_binding(locator, b"authority")
        assert error.value.context["reason"] == CacheReason.BLOB_BACKEND_FAILURE.value
        assert error.value.context["capability"] == "root_xattr"
        assert error.value.__cause__ is injected
    finally:
        file_ops.close()


def test_existing_root_xattr_failure_is_typed_without_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Policy denial while verifying a prior binding cannot adopt a new lock."""
    root = tmp_path / "xattr-verify-policy"
    root.mkdir()
    file_ops = ManagedFileOps(root)
    locator = root / ".authority.lock"
    injected = OSError(errno.EACCES, "xattr read denied")

    def existing_set(_name: str, _authority: bytes) -> None:
        raise FileExistsError(errno.EEXIST, "already bound")

    def denied_read(_name: str) -> bytes:
        raise injected

    monkeypatch.setattr(file_ops, "_set_root_authority_xattr", existing_set)
    monkeypatch.setattr(file_ops, "_get_root_authority_xattr", denied_read)
    try:
        with pytest.raises(CacheBlobBackendError) as error:
            file_ops._ensure_root_authority_binding(locator, b"authority")
        assert error.value.context["reason"] == (
            CacheReason.BLOB_BACKEND_CAPABILITY_UNSUPPORTED.value
        )
        assert error.value.__cause__ is injected
    finally:
        file_ops.close()


def test_existing_root_xattr_disappearance_fails_closed_without_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only initial create may initialize a missing D-21 root binding."""
    root = tmp_path / "xattr-missing"
    root.mkdir()
    file_ops = ManagedFileOps(root)
    locator = root / ".authority.lock"
    missing_errno = getattr(errno, "ENODATA", errno.ENOENT)

    def existing_set(_name: str, _authority: bytes) -> None:
        raise FileExistsError(errno.EEXIST, "already bound")

    def missing_read(_name: str) -> bytes:
        raise OSError(missing_errno, "binding disappeared")

    monkeypatch.setattr(file_ops, "_set_root_authority_xattr", existing_set)
    monkeypatch.setattr(file_ops, "_get_root_authority_xattr", missing_read)
    try:
        with pytest.raises(CacheUnsafePathError) as error:
            file_ops._ensure_root_authority_binding(locator, b"authority")
        assert error.value.context["reason"] == CacheReason.PATH_RACE.value
    finally:
        file_ops.close()


def test_existing_root_xattr_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An extant authority value must match exactly rather than be replaced."""
    root = tmp_path / "xattr-mismatch"
    root.mkdir()
    file_ops = ManagedFileOps(root)
    locator = root / ".authority.lock"

    def existing_set(_name: str, _authority: bytes) -> None:
        raise FileExistsError(errno.EEXIST, "already bound")

    monkeypatch.setattr(file_ops, "_set_root_authority_xattr", existing_set)
    monkeypatch.setattr(
        file_ops, "_get_root_authority_xattr", lambda _name: b"other-authority"
    )
    try:
        with pytest.raises(CacheUnsafePathError) as error:
            file_ops._ensure_root_authority_binding(locator, b"authority")
        assert error.value.context["reason"] == CacheReason.PATH_RACE.value
    finally:
        file_ops.close()


def test_root_xattr_initialization_is_only_the_create_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh trusted-store root binds exactly once without a verification read."""
    root = tmp_path / "xattr-initialization"
    root.mkdir()
    file_ops = ManagedFileOps(root)
    locator = root / ".authority.lock"
    calls: list[str] = []

    monkeypatch.setattr(
        file_ops,
        "_set_root_authority_xattr",
        lambda _name, _authority: calls.append("create"),
    )
    monkeypatch.setattr(
        file_ops,
        "_get_root_authority_xattr",
        lambda _name: calls.append("read") or b"authority",
    )
    try:
        file_ops._ensure_root_authority_binding(locator, b"authority")
        assert calls == ["create"]
    finally:
        file_ops.close()


def test_json_lock_name_swap_cannot_create_a_second_cross_process_authority(
    tmp_path: Path,
) -> None:
    """Replacing every JSON lock control name cannot form a new authority.

    The actual lock pathname is mutable by design.  The authoritative lock
    identity is instead attached to the root object, so a contender that sees
    a replacement file is rejected before it can enter JSON CAS even while the
    original retained descriptor remains in its critical section.
    """
    root = tmp_path / "json-lock-swap-process"
    root.mkdir()
    metadata_path = root / "metadata.json"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    outcomes = context.Queue()
    original = context.Process(
        target=_publish_json_while_parent_swaps_lock_name,
        args=(str(metadata_path), ready, release, outcomes),
    )
    original.start()
    assert ready.wait(timeout=10)

    lock_path = root / ".metadata.json.manifest-cas.lock"
    replacement = root / ".replacement-lock"
    replacement.write_bytes(b"lock\n")
    os.replace(replacement, lock_path)
    contender = context.Process(
        target=_attempt_json_publish_after_lock_swap,
        args=(str(metadata_path), outcomes),
    )
    contender.start()
    contender.join(timeout=10)
    assert contender.exitcode == 0

    release.set()
    original.join(timeout=10)
    assert original.exitcode == 0
    assert sorted(outcomes.get(timeout=5) for _ in range(2)) == [
        "original-won",
        "replacement-blocked",
    ]

    # The old descriptor was permitted to finish; the intentionally replaced
    # name is then a fail-closed topology, never a new authority to adopt.
    backend = JsonBackend(metadata_path)
    try:
        with pytest.raises(CacheUnsafePathError):
            JsonManifestRepository(backend)
    finally:
        backend.close()


def test_json_cas_uses_the_win32_adapter_when_fcntl_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The simulated one-user/session Windows path never imports POSIX locking."""
    shared_lock = Lock()
    calls: list[str] = []

    class FakeWindowsLockApi:
        def lock(
            self, _descriptor: int, *, exclusive: bool, nonblocking: bool = False
        ) -> object:
            assert exclusive
            shared_lock.acquire()
            calls.append("lock")
            return object()

        def unlock(self, _descriptor: int, _token: object) -> object:
            calls.append("unlock")
            shared_lock.release()
            return None

    original_import = builtins.__import__

    def reject_fcntl(name: str, *args: object, **kwargs: object):
        if name == "fcntl":
            raise ImportError("fcntl is unavailable on Windows")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(coordination, "_platform_name", lambda: "nt")
    monkeypatch.setattr(coordination, "_windows_lock_api", FakeWindowsLockApi)
    monkeypatch.setattr(builtins, "__import__", reject_fcntl)

    metadata_path = tmp_path / "windows.json"
    first_backend = JsonBackend(metadata_path)
    second_backend = JsonBackend(metadata_path)
    first = JsonManifestRepository(first_backend)
    second = JsonManifestRepository(second_backend)
    gate = Barrier(2)
    outcomes: list[str] = []

    def contender(repository: JsonManifestRepository, record: bytes) -> None:
        gate.wait(timeout=5)
        try:
            repository.publish_if_expected("key", ManifestExpectation.absent(), record)
            outcomes.append("won")
        except CacheBlobLifecycleConflictError:
            outcomes.append("conflict")

    threads = [
        Thread(target=contender, args=(first, _record("windows-a"))),
        Thread(target=contender, args=(second, _record("windows-b"))),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
        assert sorted(outcomes) == ["conflict", "won"]
        assert calls.count("lock") >= 2
        assert calls.count("unlock") == calls.count("lock")
    finally:
        first.close()
        second.close()
        first_backend.close()
        second_backend.close()


@pytest.mark.parametrize("substitute_during_open", (False, True))
def test_json_authority_lock_rejects_symlink_substitution_without_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitute_during_open: bool,
) -> None:
    """A lock symlink cannot win before or between managed create/open checks."""
    outside = tmp_path / "outside-lock"
    root = tmp_path / "inside"
    root.mkdir()
    backend = JsonBackend(root / "metadata.json")
    file_ops = ManagedFileOps(root)
    lock_locator = resolve_managed_locator(
        root,
        ".metadata.json.manifest-cas.lock",
        operation="manifest_repository_authority_lock",
        allow_missing_leaf=True,
    )
    try:
        if substitute_during_open:
            def swap_lock(operation: str, locator: Path) -> None:
                if operation == "read_stream" and locator == lock_locator:
                    lock_locator.unlink()
                    lock_locator.symlink_to(outside)

            monkeypatch.setattr(file_ops, "before_operation", swap_lock)
        else:
            lock_locator.symlink_to(outside)

        with pytest.raises(CacheUnsafePathError):
            JsonManifestRepository(backend, file_ops=file_ops)
        assert not outside.exists()
        assert backend._metadata.get("entries", {}).get("key") is None
    finally:
        file_ops.close()
        backend.close()


def test_json_authority_lock_rejects_a_live_lock_name_replacement(
    tmp_path: Path,
) -> None:
    """A retained lock descriptor rejects later name/inode substitution."""
    root = tmp_path / "live-lock-substitution"
    root.mkdir()
    outside = tmp_path / "outside-lock"
    backend = JsonBackend(root / "metadata.json")
    repository = JsonManifestRepository(backend)
    assert repository._json_lock_locator is not None
    lock_locator = repository._json_lock_locator
    try:
        lock_locator.unlink()
        lock_locator.symlink_to(outside)
        with pytest.raises(CacheBlobBackendError):
            repository.publish_if_expected(
                "key", ManifestExpectation.absent(), _record("blocked")
            )
        assert not outside.exists()
        assert backend._metadata.get("entries", {}).get("key") is None
    finally:
        repository.close()
        backend.close()


@pytest.mark.parametrize("seam", ("validation", "acquisition"))
def test_json_authority_never_publishes_into_a_replaced_root(
    tmp_path: Path, seam: str
) -> None:
    """Descriptor-anchored JSON CAS rejects either root-replacement boundary."""
    root = tmp_path / f"replaced-json-root-{seam}"
    root.mkdir()
    retired = tmp_path / f"retired-json-root-{seam}"
    backend = JsonBackend(root / "metadata.json")
    repository = JsonManifestRepository(backend)

    def replace_root() -> None:
        root.rename(retired)
        root.mkdir()

    if seam == "validation":
        repository.after_json_lock_validation = replace_root
    else:
        repository.after_json_lock_acquisition = replace_root
    try:
        with pytest.raises(CacheBlobBackendError):
            repository.publish_if_expected(
                "key", ManifestExpectation.absent(), _record(f"blocked-{seam}")
            )
        assert not (root / "metadata.json").exists()
        assert not (retired / "metadata.json").exists()
    finally:
        repository.close()
        backend.close()


def test_json_authority_lock_rejects_a_hard_link_from_outside_the_store(
    tmp_path: Path,
) -> None:
    """An outside hard link cannot become the canonical JSON authority inode."""
    root = tmp_path / "hard-linked-json-lock"
    root.mkdir()
    outside = tmp_path / "outside-lock"
    outside.write_bytes(b"lock\n")
    backend = JsonBackend(root / "metadata.json")
    lock_locator = root / ".metadata.json.manifest-cas.lock"
    os.link(outside, lock_locator)
    try:
        with pytest.raises(CacheUnsafePathError):
            JsonManifestRepository(backend)
        assert outside.stat().st_nlink == 2
        assert not (root / "metadata.json").exists()
    finally:
        backend.close()


def test_json_authority_lock_rejects_regular_inode_replacement(
    tmp_path: Path,
) -> None:
    """A regular-file swap cannot partition a retained authority descriptor."""
    root = tmp_path / "regular-json-lock-swap"
    root.mkdir()
    backend = JsonBackend(root / "metadata.json")
    repository = JsonManifestRepository(backend)
    assert repository._json_lock_locator is not None
    replacement = tmp_path / "replacement-lock"
    replacement.write_bytes(b"replacement\n")
    try:
        os.replace(replacement, repository._json_lock_locator)
        with pytest.raises(CacheBlobBackendError):
            repository.publish_if_expected(
                "key", ManifestExpectation.absent(), _record("regular-swap")
            )
        assert not (root / "metadata.json").exists()
    finally:
        repository.close()
        backend.close()


def test_json_repository_uses_native_windows_control_durability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The simulated one-user/session path uses documented file/move primitives."""
    root = tmp_path / "windows-json"
    root.mkdir()
    backend = JsonBackend(root / "metadata.json")
    calls: list[str] = []
    authorities: dict[str, bytes] = {}

    class FakeWindowsFileApi:
        def rename_no_replace(self, temporary: Path, destination: Path) -> None:
            calls.append("rename")
            if destination.exists():
                raise FileExistsError(destination)
            os.rename(temporary, destination)

        def replace_write_through(self, temporary: Path, destination: Path) -> None:
            calls.append("replace_write_through")
            os.replace(temporary, destination)

        def flush_regular_file(self, locator: Path) -> None:
            assert locator.parent == root
            calls.append("flush_regular_file")

    class FakeWindowsRegistryAuthorityApi:
        def ensure(self, name: str, value: bytes) -> None:
            assert authorities.setdefault(name, value) == value

    monkeypatch.setattr(path_security, "_platform_name", lambda: "nt")
    monkeypatch.setattr(path_security, "_windows_file_api", FakeWindowsFileApi)
    monkeypatch.setattr(
        path_security,
        "_windows_registry_authority_api",
        FakeWindowsRegistryAuthorityApi,
    )
    try:
        repository = JsonManifestRepository(backend)
        try:
            repository.publish_if_expected(
                "key", ManifestExpectation.absent(), _record("windows-control")
            )
            assert repository.get_raw("key") == _record("windows-control")
            assert "flush_regular_file" in calls
            assert authorities
        finally:
            repository.close()
    finally:
        backend.close()


def test_windows_local_authority_scope_is_not_cross_principal() -> None:
    """The implementation documents its deliberately local Windows namespace."""
    assert path_security._WINDOWS_LOCAL_STORE_COORDINATION_SCOPE == (
        "one_os_user_one_session"
    )


@pytest.mark.parametrize(
    ("status", "reason"),
    (
        (5, CacheReason.BLOB_BACKEND_CAPABILITY_UNSUPPORTED),
        (258, CacheReason.BLOB_BACKEND_FAILURE),
    ),
)
def test_windows_registry_authority_statuses_use_the_typed_backend_taxonomy(
    status: int, reason: CacheReason
) -> None:
    """Registry/mutex policy and operational failures never escape as OSError."""
    with pytest.raises(CacheBlobBackendError) as error:
        path_security._WindowsRegistryAuthorityApi._raise_status(
            status, "ReleaseMutex"
        )

    assert error.value.context["reason"] == reason.value
    assert error.value.context["operation"] == "ReleaseMutex"
    assert isinstance(error.value.__cause__, OSError)
    assert (
        path_security._WindowsRegistryAuthorityApi._HKEY_CURRENT_USER
        == 0x80000001
    )


@pytest.mark.parametrize("error_number", (80, 183))
def test_native_windows_already_exists_errors_preserve_fileexists_contract(
    monkeypatch: pytest.MonkeyPatch, error_number: int
) -> None:
    """Cross-platform exclusive-create callers receive ``FileExistsError``."""
    monkeypatch.setattr(
        path_security._WindowsFileApi,
        "_last_error",
        staticmethod(lambda: error_number),
    )
    with pytest.raises(FileExistsError) as error:
        path_security._WindowsFileApi._raise_last_error("MoveFileExW")
    assert error.value.errno == error_number


def test_native_windows_regular_file_flush_uses_documented_handle_and_closes(
    tmp_path: Path,
) -> None:
    """The native adapter flushes an ordinary file, never a directory handle."""
    api = object.__new__(path_security._WindowsFileApi)
    calls: list[object] = []
    api._create_file = lambda *args: calls.append(args) or ctypes.c_void_p(101).value
    api._flush_file_buffers = lambda handle: calls.append(("flush", handle)) or 1
    api._close_handle = lambda handle: calls.append(("close", handle)) or 1
    api._last_error = lambda: 5

    def get_information(_handle, information) -> int:
        contents = ctypes.cast(
            information, ctypes.POINTER(api._ByHandleFileInformation)
        ).contents
        contents.FileAttributes = 0
        contents.NumberOfLinks = 1
        return 1

    api._get_file_information = get_information

    api.flush_regular_file(tmp_path / "control.json")

    create_arguments = calls[0]
    assert create_arguments[1] == api._GENERIC_WRITE
    assert create_arguments[5] == api._FILE_FLAG_OPEN_REPARSE_POINT
    assert ("flush", ctypes.c_void_p(101).value) in calls
    assert ("close", ctypes.c_void_p(101).value) in calls


def test_native_windows_delete_uses_a_reparse_safe_disposition_handle(
    tmp_path: Path,
) -> None:
    """Immediate deletion does not use MoveFileExW with a NULL destination."""
    api = object.__new__(path_security._WindowsFileApi)
    calls: list[object] = []
    api._create_file = lambda *args: calls.append(args) or ctypes.c_void_p(202).value

    def get_information(_handle, information) -> int:
        contents = ctypes.cast(
            information, ctypes.POINTER(api._ByHandleFileInformation)
        ).contents
        contents.FileAttributes = 0
        contents.NumberOfLinks = 1
        return 1

    api._get_file_information = get_information
    api._set_file_information = lambda *args: calls.append(("disposition", args)) or 1
    api._close_handle = lambda handle: calls.append(("close", handle)) or 1
    api._last_error = lambda: 5

    identity = api.delete_write_through(tmp_path / "control.json")

    create_arguments = calls[0]
    assert create_arguments[1] & api._DELETE
    assert create_arguments[5] == api._FILE_FLAG_OPEN_REPARSE_POINT
    disposition_call = next(call for call in calls if call[0] == "disposition")[1]
    assert disposition_call[1] == api._FILE_DISPOSITION_INFO
    assert ("close", ctypes.c_void_p(202).value) in calls
    assert identity == (0, 0, 0)


def test_native_windows_no_replace_move_requests_documented_write_through() -> None:
    """No-replace control publication asks MoveFileExW to acknowledge the move."""
    api = object.__new__(path_security._WindowsFileApi)
    calls: list[tuple[str, str, int]] = []
    api._move_file_ex = lambda source, destination, flags: calls.append(
        (source, destination, flags)
    ) or 1

    api.rename_no_replace(Path("candidate"), Path("final"))

    assert calls == [
        (
            "candidate",
            "final",
            path_security._WindowsFileApi._MOVEFILE_WRITE_THROUGH,
        )
    ]


def test_windows_fallback_promotes_exact_pending_control_after_process_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Windows fallback promotes only its digest-bound pending evidence."""
    root = tmp_path / "windows-pending-recovery"
    root.mkdir()
    file_ops = ManagedFileOps(root)
    file_ops._descriptor_mode = False
    final = root / "operations" / "operation.json"
    payload = b'{"exact":"control"}'
    digest = hashlib.sha256(payload).hexdigest()
    pending_name = f".{final.name}.pending.{digest}.{'a' * 32}.tmp"
    pending = final.parent / pending_name
    pending.parent.mkdir()
    pending.write_bytes(payload)
    flushed: list[Path] = []

    class FakeWindowsFileApi:
        def rename_no_replace(self, temporary: Path, destination: Path) -> None:
            if destination.exists():
                raise FileExistsError(destination)
            os.rename(temporary, destination)

        def flush_regular_file(self, locator: Path) -> None:
            flushed.append(locator)

    monkeypatch.setattr(path_security, "_platform_name", lambda: "nt")
    monkeypatch.setattr(path_security, "_windows_file_api", FakeWindowsFileApi)
    try:
        assert file_ops.promote_durable_pending_control(
            final, payload, pending_name=pending_name
        )
        assert final.read_bytes() == payload
        assert not pending.exists()
        assert flushed == [final]
    finally:
        file_ops.close()


def test_windows_fallback_leaves_substituted_pending_control_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A final-record race never authorizes pathname deletion of a new pending file."""
    root = tmp_path / "windows-pending-substitution"
    root.mkdir()
    file_ops = ManagedFileOps(root)
    file_ops._descriptor_mode = False
    final = root / "operations" / "operation.json"
    payload = b'{"exact":"control"}'
    digest = hashlib.sha256(payload).hexdigest()
    pending_name = f".{final.name}.pending.{digest}.{'b' * 32}.tmp"
    pending = final.parent / pending_name
    final.parent.mkdir()
    pending.write_bytes(payload)
    final.write_bytes(payload)
    victim = final.parent / "victim.json"
    victim.write_bytes(b"victim")

    class ExistingFinalWindowsFileApi:
        def rename_no_replace(self, _temporary: Path, _destination: Path) -> None:
            raise FileExistsError("final already exists")

    original_read = file_ops.read_bytes_bounded

    def swap_pending_after_final_read(locator: Path, *, max_bytes: int) -> bytes:
        observed = original_read(locator, max_bytes=max_bytes)
        if locator == final:
            os.replace(victim, pending)
        return observed

    monkeypatch.setattr(path_security, "_platform_name", lambda: "nt")
    monkeypatch.setattr(
        path_security, "_windows_file_api", ExistingFinalWindowsFileApi
    )
    monkeypatch.setattr(file_ops, "read_bytes_bounded", swap_pending_after_final_read)
    try:
        assert not file_ops.promote_durable_pending_control(
            final, payload, pending_name=pending_name
        )
        assert final.read_bytes() == payload
        assert pending.read_bytes() == b"victim"
    finally:
        file_ops.close()


@pytest.mark.parametrize("failure", ("create", "open", "identity"))
def test_failed_direct_json_repository_construction_closes_its_owned_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Every direct-construction failure releases the repository-owned descriptor."""
    root = tmp_path / f"owned-root-close-{failure}"
    root.mkdir()
    backend = JsonBackend(root / "metadata.json")
    file_ops = ManagedFileOps(root)
    closed: list[bool] = []
    original_close = file_ops.close

    def track_close() -> None:
        closed.append(True)
        original_close()

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError(f"injected {failure} failure")

    monkeypatch.setattr(file_ops, "close", track_close)
    monkeypatch.setattr(manifest_repository_module, "ManagedFileOps", lambda _root: file_ops)
    if failure == "create":
        monkeypatch.setattr(file_ops, "ensure_lifecycle_lock", fail)
    elif failure == "open":
        monkeypatch.setattr(file_ops, "open_verified_regular_file", fail)
    else:
        monkeypatch.setattr(file_ops, "ensure_lifecycle_lock", fail)
    try:
        with pytest.raises(OSError, match=failure):
            JsonManifestRepository(backend)
        assert closed == [True]
    finally:
        backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json", "sqlite"))
def test_remove_if_expected_cannot_retire_a_replaced_record(
    tmp_path: Path, backend_name: str
) -> None:
    """Stale reclamation preserves a record published after the observed bytes."""
    first_repo, second_repo, backends = _repository_pair(tmp_path, backend_name)
    original = _record("old")
    replacement = _record("new")
    old_expectation = ManifestExpectation.from_authenticated_record("old", original)
    new_expectation = ManifestExpectation.from_authenticated_record("new", replacement)
    try:
        first_repo.put_raw("key", original)
        first_repo.publish_if_expected("key", old_expectation, replacement)

        with pytest.raises(CacheBlobLifecycleConflictError):
            second_repo.remove_if_expected("key", old_expectation)

        assert first_repo.get_raw("key") == replacement
        second_repo.remove_if_expected("key", new_expectation)
        assert first_repo.get_raw("key") is None
    finally:
        for backend in backends:
            backend.close()


def test_sqlite_cas_rolls_back_compatibility_projection_with_raw_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw-row failure leaves both SQLite projections at their prior winner."""
    backend = SqliteBackend(tmp_path / "metadata.db")
    repository = SqliteManifestRepository(backend)
    original = _record("original")
    replacement = _record("replacement")
    original_entry = {"description": "original", "data_type": "object"}
    replacement_entry = {"description": "replacement", "data_type": "object"}
    repository.put_raw("key", original, entry_data=original_entry)
    expectation = ManifestExpectation.from_authenticated_record("generation", original)

    def fail_raw_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected raw record failure")

    monkeypatch.setattr(repository, "_write_raw_row", fail_raw_write)
    try:
        with pytest.raises(CacheBlobBackendError):
            repository.publish_if_expected(
                "key", expectation, replacement, entry_data=replacement_entry
            )

        assert repository.get_raw("key") == original
        assert backend.get_entry("key")["description"] == "original"
    finally:
        backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json", "sqlite"))
def test_manifest_pages_are_stable_bounded_and_retain_the_supplied_limits(
    tmp_path: Path, backend_name: str
) -> None:
    """Local adapters expose two-item opaque pages without a local policy copy."""
    repository, _other, backends = _repository_pair(tmp_path, backend_name)
    limits = LifecycleLimits(manifest_page_size=2)
    repository = type(repository)(repository.backend, lifecycle_limits=limits)
    records = {key: _record(key) for key in ("delta", "alpha", "charlie", "bravo")}
    try:
        for key, record in records.items():
            repository.put_raw(key, record)

        first = repository.list_page()
        assert repository.lifecycle_limits is limits
        assert [key for key, _raw in first.entries] == ["delta", "alpha"]
        assert [raw for _key, raw in first.entries] == [records["delta"], records["alpha"]]
        assert first.next_cursor is not None
        assert first.next_cursor.key == "alpha"
        assert first.next_cursor.snapshot_high_water == 4
        assert first.next_cursor.next_sequence == 3

        second = repository.list_page(first.next_cursor)
        assert [key for key, _raw in second.entries] == ["charlie", "bravo"]
        assert second.next_cursor is None
    finally:
        for backend in backends:
            backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json", "sqlite"))
def test_manifest_inventory_uses_a_high_water_snapshot_without_a_total_store_cap(
    tmp_path: Path, backend_name: str
) -> None:
    """A late publication cannot move backwards into an active page chain."""
    repository, _other, backends = _repository_pair(tmp_path, backend_name)
    limits = LifecycleLimits(manifest_page_size=2, max_inventory_items=2)
    repository = type(repository)(repository.backend, lifecycle_limits=limits)
    try:
        for key in ("a", "b", "c"):
            repository.put_raw(key, _record(key))
        first = repository.list_page()
        assert [key for key, _raw in first.entries] == ["a", "b"]
        assert first.next_cursor is not None

        # This sorts before the former lexical cursor but was born after the
        # snapshot high-water mark, so it belongs to a later chain.
        repository.put_raw("aa", _record("aa"))
        second = repository.list_page(first.next_cursor)
        assert [key for key, _raw in second.entries] == ["c"]
        assert second.next_cursor is None

    finally:
        for backend in backends:
            backend.close()


def test_memory_manifest_inventory_pages_a_store_above_the_former_hard_cap() -> None:
    """4,097 current records remain eligible under a two-name page budget."""
    backend = InMemoryBackend()
    repository = InMemoryManifestRepository(
        backend,
        lifecycle_limits=LifecycleLimits(manifest_page_size=2, max_inventory_items=2),
    )
    for index in range(4_097):
        key = f"large-{index:05d}"
        repository.put_raw(key, _record(key))
    page = repository.list_page()
    assert len(page.entries) == 2
    assert page.next_cursor is not None


def test_json_manifest_inventory_is_external_bounded_and_reopens(tmp_path: Path) -> None:
    """JSON authority never carries an append/rewrite history field on reopen."""
    metadata_path = tmp_path / "metadata.json"
    backend = JsonBackend(metadata_path)
    repository = JsonManifestRepository(
        backend, lifecycle_limits=LifecycleLimits(manifest_page_size=2)
    )
    try:
        for key in ("a", "b", "c"):
            repository.put_raw(key, _record(key))
        assert "_cacheness_manifest_inventory_v1" not in backend._metadata
        assert (
            tmp_path / ".metadata.json.manifest-inventory-v2-head.json"
        ).is_file()
        assert len(
            list(tmp_path.glob(".metadata.json.manifest-inventory-v2-event-*.json"))
        ) == 3
    finally:
        repository.close()
        backend.close()

    reopened_backend = JsonBackend(metadata_path)
    reopened = JsonManifestRepository(
        reopened_backend, lifecycle_limits=LifecycleLimits(manifest_page_size=2)
    )
    try:
        first = reopened.list_page()
        assert [key for key, _raw in first.entries] == ["a", "b"]
        assert first.next_cursor is not None
        second = reopened.list_page(first.next_cursor)
        assert [key for key, _raw in second.entries] == ["c"]
        assert second.next_cursor is None
    finally:
        reopened.close()
        reopened_backend.close()
