"""Deterministic ownership and admission contracts for ``BlobStore.close``."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from threading import Event, Thread

import pytest

from cacheness import CacheConfig
from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobCloseTimeoutError,
    CacheBlobLockReleaseError,
    CacheBlobStoreClosedError,
    CacheUnsafePathError,
)
from cacheness.metadata import InMemoryBackend
from cacheness.storage import BlobStore
from cacheness.storage import coordination
from cacheness.storage import path_security
from cacheness.storage.coordination import StoreAdmissionBarrier, interprocess_file_lock
from cacheness.storage.path_security import ManagedFileOps
from cacheness.storage.operation_repository import FileOperationRecordRepository


def _join(thread: Thread) -> None:
    """Join one event-driven worker without using sleep as an oracle."""
    thread.join(timeout=5)
    assert not thread.is_alive(), "worker did not finish within the bounded deadline"


def _configured_store(root: Path, *, close_wait_seconds: float = 0.02) -> BlobStore:
    """Create a store retaining one caller-owned finite close policy object."""
    limits = LifecycleLimits(close_wait_seconds=close_wait_seconds)
    store = BlobStore(root, backend="json", config=CacheConfig(lifecycle_limits=limits))
    assert store.lifecycle_limits is limits
    assert store._instance_admission.lifecycle_limits is limits
    return store


def test_close_admission_rejects_new_work_before_resource_access(tmp_path, monkeypatch):
    """Close drains one admitted call while every later call fails at admission."""
    store = _configured_store(tmp_path / "admission")
    entered = Event()
    release = Event()
    close_waiting = Event()
    operation_errors: list[BaseException] = []
    close_errors: list[BaseException] = []
    original_refresh = store._refresh_metadata_view_for_lifecycle

    def paused_refresh() -> None:
        entered.set()
        assert release.wait(timeout=5)
        original_refresh()

    def wait_with_signal(condition, timeout: float) -> None:
        close_waiting.set()
        condition.wait(timeout)

    monkeypatch.setattr(store, "_refresh_metadata_view_for_lifecycle", paused_refresh)
    monkeypatch.setattr(store._instance_admission, "_wait", wait_with_signal)

    def admitted_operation() -> None:
        try:
            assert store.get_metadata("missing") is None
        except BaseException as exc:  # pragma: no cover - asserted below.
            operation_errors.append(exc)

    def close_store() -> None:
        try:
            store.close()
        except BaseException as exc:  # pragma: no cover - asserted below.
            close_errors.append(exc)

    operation = Thread(target=admitted_operation)
    operation.start()
    assert entered.wait(timeout=5)
    closer = Thread(target=close_store)
    closer.start()
    assert close_waiting.wait(timeout=5)

    monkeypatch.setattr(
        store,
        "_refresh_metadata_view_for_lifecycle",
        lambda: (_ for _ in ()).throw(AssertionError("closed work reached resources")),
    )
    with pytest.raises(CacheBlobStoreClosedError):
        store.get_metadata("later")

    release.set()
    _join(operation)
    _join(closer)
    assert operation_errors == []
    assert close_errors == []


def test_close_timeout_keeps_resources_live_and_retry_converges(tmp_path, monkeypatch):
    """A finite injected deadline preserves CLOSING resources for later retry."""
    store = _configured_store(tmp_path / "timeout")
    entered = Event()
    release = Event()
    clock = [0.0]
    guarded_close_calls: list[None] = []
    backend_close_calls: list[None] = []
    original_guarded_close = store.guarded_handler_io.close
    original_backend_close = store.backend.close

    def advance_clock(_condition, timeout: float) -> None:
        clock[0] += timeout

    def guarded_close() -> None:
        guarded_close_calls.append(None)
        original_guarded_close()

    def backend_close() -> None:
        backend_close_calls.append(None)
        original_backend_close()

    monkeypatch.setattr(store._instance_admission, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(store._instance_admission, "_wait", advance_clock)
    monkeypatch.setattr(store.guarded_handler_io, "close", guarded_close)
    monkeypatch.setattr(store.backend, "close", backend_close)

    def held_operation() -> None:
        with store._instance_admission.operation():
            entered.set()
            assert release.wait(timeout=5)

    worker = Thread(target=held_operation)
    worker.start()
    assert entered.wait(timeout=5)
    with pytest.raises(CacheBlobCloseTimeoutError) as error:
        store.close()
    assert error.value.context["reason"] == "blob_close_timeout"
    assert clock[0] == pytest.approx(0.02)
    assert guarded_close_calls == []
    assert backend_close_calls == []

    release.set()
    _join(worker)
    store.close()
    assert guarded_close_calls == [None]
    assert backend_close_calls == [None]


def test_close_from_admitted_thread_is_typed_instead_of_self_deadlocking(tmp_path):
    """A public close called by in-flight work cannot wait for itself forever."""
    store = _configured_store(tmp_path / "reentrant")
    try:
        with store._instance_admission.operation():
            with pytest.raises(CacheBlobCloseTimeoutError) as error:
                store.close()
        assert error.value.context["reason"] == "blob_close_timeout"
        assert error.value.context["reentrant"] is True
    finally:
        store.close()


def test_close_releases_only_owned_resources_once_and_preserves_data(
    tmp_path, monkeypatch
):
    """Owned handles close once, injected handles survive, and close never clears."""
    root = tmp_path / "owned"
    owned = _configured_store(root)
    key = owned.put({"value": "persist"}, key="persisted")
    guarded_close_calls: list[None] = []
    backend_close_calls: list[None] = []
    original_guarded_close = owned.guarded_handler_io.close
    original_backend_close = owned.backend.close

    def guarded_close() -> None:
        guarded_close_calls.append(None)
        original_guarded_close()

    def backend_close() -> None:
        backend_close_calls.append(None)
        original_backend_close()

    monkeypatch.setattr(owned.guarded_handler_io, "close", guarded_close)
    monkeypatch.setattr(owned.backend, "close", backend_close)
    monkeypatch.setattr(
        owned,
        "clear",
        lambda: (_ for _ in ()).throw(AssertionError("close must not clear data")),
    )
    monkeypatch.setattr(
        owned,
        "delete",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("close must not delete data")
        ),
    )

    owned.close()
    owned.close()
    assert guarded_close_calls == [None]
    assert backend_close_calls == [None]

    reopened = BlobStore(root, backend="json")
    try:
        assert reopened.get(key) == {"value": "persist"}
    finally:
        reopened.close()

    injected_backend = InMemoryBackend()
    injected_close_calls: list[None] = []
    monkeypatch.setattr(injected_backend, "close", lambda: injected_close_calls.append(None))
    injected = BlobStore(tmp_path / "injected", backend=injected_backend)
    try:
        injected.close()
        injected.close()
        assert injected_close_calls == []
    finally:
        injected.close()


def test_partial_owned_resource_failure_is_typed_and_retries_without_double_close(
    tmp_path, monkeypatch
):
    """A backend failure leaves CLOSING state and retries only unreleased work."""
    store = _configured_store(tmp_path / "partial")
    guarded_close_calls: list[None] = []
    backend_close_calls: list[None] = []
    original_guarded_close = store.guarded_handler_io.close
    original_backend_close = store.backend.close

    def guarded_close() -> None:
        guarded_close_calls.append(None)
        original_guarded_close()

    def backend_close() -> None:
        backend_close_calls.append(None)
        if len(backend_close_calls) == 1:
            raise OSError("injected backend close failure")
        original_backend_close()

    monkeypatch.setattr(store.guarded_handler_io, "close", guarded_close)
    monkeypatch.setattr(store.backend, "close", backend_close)

    with pytest.raises(CacheBlobBackendError) as error:
        store.close()
    assert error.value.context["operation"] == "close"
    assert isinstance(error.value.__cause__, OSError)
    assert guarded_close_calls == [None]
    assert backend_close_calls == [None]

    store.close()
    assert guarded_close_calls == [None]
    assert backend_close_calls == [None, None]


@pytest.mark.parametrize("failing_index", (0, 1, 2))
def test_operation_repository_retains_every_failed_lock_close_for_retry(
    tmp_path: Path, failing_index: int
) -> None:
    """First, middle, and final retained evidence handles are not forgotten."""
    file_ops = ManagedFileOps(tmp_path)
    repository = FileOperationRecordRepository(
        file_ops, lifecycle_limits=LifecycleLimits()
    )

    class Handle:
        def __init__(self, index: int) -> None:
            self.index = index
            self.calls = 0

        def close(self) -> None:
            self.calls += 1
            if self.index == failing_index and self.calls == 1:
                raise OSError("injected retained lock close failure")

    handles = [Handle(index) for index in range(3)]
    repository._lock_handles = {
        f"lock-{index}": (tmp_path / f"lock-{index}", handle, (1, index))
        for index, handle in enumerate(handles)
    }
    try:
        with pytest.raises(OSError, match="retained lock close failure"):
            repository.close()
        assert set(repository._lock_handles) == {f"lock-{failing_index}"}
        repository.close()
        assert repository._lock_handles == {}
        assert handles[failing_index].calls == 2
    finally:
        repository.close()
        file_ops.close()


@pytest.mark.parametrize("resource", ("lock", "root"))
def test_final_barrier_release_keeps_its_registry_lease_until_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resource: str
) -> None:
    """A failed final barrier close cannot let a later store report CLOSED."""
    root = tmp_path / f"barrier-retry-{resource}"
    root.mkdir()
    barrier = StoreAdmissionBarrier.acquire(root)
    original_lock_close = barrier._lock_handle.close
    original_root_close = barrier._file_ops.close
    calls = 0

    def fail_once_lock() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected barrier lock close failure")
        original_lock_close()

    def fail_once_root() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected barrier root close failure")
        original_root_close()

    if resource == "lock":
        monkeypatch.setattr(barrier._lock_handle, "close", fail_once_lock)
    else:
        monkeypatch.setattr(barrier._file_ops, "close", fail_once_root)
    try:
        with pytest.raises(OSError):
            barrier.release()
        assert StoreAdmissionBarrier._instances[barrier._registry_identity] is barrier
        assert barrier._leases == 1
        barrier.release()
        assert barrier._registry_identity not in StoreAdmissionBarrier._instances
    finally:
        if barrier._registry_identity in StoreAdmissionBarrier._instances:
            barrier.release()


def test_concurrent_close_waiter_does_not_repeat_owned_release(tmp_path, monkeypatch):
    """A waiter observes CLOSED after another caller releases every owned handle."""
    store = _configured_store(tmp_path / "concurrent-close", close_wait_seconds=5)
    guarded_close_started = Event()
    release_resource = Event()
    waiting_close = Event()
    close_errors: list[BaseException] = []
    flush_calls: list[None] = []
    guarded_close_calls: list[None] = []
    original_guarded_close = store.guarded_handler_io.close
    original_wait = store._instance_admission._wait

    def flush() -> None:
        flush_calls.append(None)

    def guarded_close() -> None:
        guarded_close_calls.append(None)
        guarded_close_started.set()
        assert release_resource.wait(timeout=5)
        original_guarded_close()

    def wait_with_signal(condition, timeout: float) -> None:
        waiting_close.set()
        original_wait(condition, timeout)

    monkeypatch.setattr(
        store.lifecycle.operation_repository,
        "flush",
        flush,
        raising=False,
    )
    monkeypatch.setattr(store.guarded_handler_io, "close", guarded_close)
    monkeypatch.setattr(store._instance_admission, "_wait", wait_with_signal)

    def close_store() -> None:
        try:
            store.close()
        except BaseException as exc:  # pragma: no cover - asserted below.
            close_errors.append(exc)

    first_close = Thread(target=close_store)
    first_close.start()
    assert guarded_close_started.wait(timeout=5)

    second_close = Thread(target=close_store)
    second_close.start()
    assert waiting_close.wait(timeout=5)

    release_resource.set()
    _join(first_close)
    _join(second_close)
    assert close_errors == []
    assert flush_calls == [None]
    assert guarded_close_calls == [None]


def test_admission_barrier_releases_final_root_lease_and_recreates_identity(tmp_path):
    """Closed roots retain neither barrier descriptors nor stale inode identity."""
    roots = [tmp_path / f"leased-{index}" for index in range(24)]
    stores = [BlobStore(root, backend="json") for root in roots]
    barriers = [store._admission_barrier for store in stores]
    identities = [barrier._root_identity for barrier in barriers]
    for store in stores:
        store.close()

    assert all(
        StoreAdmissionBarrier._instances.get(identity) is None
        for identity in identities
    )
    assert all(barrier._file_ops._root_fd is None for barrier in barriers)

    root = tmp_path / "recreated-root"
    original = BlobStore(root, backend="json")
    original_barrier = original._admission_barrier
    original.close()
    shutil.rmtree(root)
    root.mkdir()

    recreated = BlobStore(root, backend="json")
    try:
        assert recreated._admission_barrier is not original_barrier
        recreated.put({"value": "new-root"}, key="new-root")
        assert recreated.get("new-root") == {"value": "new-root"}
    finally:
        recreated.close()


def test_windows_lock_path_keeps_canonical_blobstore_constructible(
    tmp_path, monkeypatch
):
    """The Win32 lock adapter serves shared admission and exact evidence CAS."""
    calls: list[tuple[str, bool | None]] = []

    class FakeWindowsLockApi:
        def lock(
            self, _descriptor: int, *, exclusive: bool, nonblocking: bool = False
        ) -> object:
            calls.append(("lock", exclusive))
            return object()

        def unlock(self, _descriptor: int, _token: object) -> object:
            calls.append(("unlock", None))
            return None

    monkeypatch.setattr(coordination, "_platform_name", lambda: "nt")
    monkeypatch.setattr(coordination, "_windows_lock_api", FakeWindowsLockApi)
    store = BlobStore(tmp_path / "windows-lock-path", backend="json")
    try:
        store.put({"value": "ordinary"}, key="ordinary")
        assert store.clear() == 1
        assert ("lock", False) in calls
        assert ("lock", True) in calls
    finally:
        store.close()


@pytest.mark.parametrize("backend_name", ("json", "sqlite"))
def test_default_blobstore_runs_the_native_windows_fallback_contract(
    tmp_path, monkeypatch, backend_name: str
):
    """Default stores exercise the one-user/session Windows fallback routing."""
    flushed_files: list[Path] = []
    authorities: dict[str, bytes] = {}

    class FakeWindowsFileApi:
        def rename_no_replace(self, temporary: Path, destination: Path) -> None:
            if destination.exists():
                raise FileExistsError(destination)
            os.rename(temporary, destination)

        def replace_write_through(self, temporary: Path, destination: Path) -> None:
            os.replace(temporary, destination)

        def delete_write_through(self, locator: Path) -> None:
            locator.unlink()

        def flush_regular_file(self, locator: Path) -> None:
            flushed_files.append(locator)

    class FakeWindowsRegistryAuthorityApi:
        def ensure(self, name: str, value: bytes) -> None:
            assert authorities.setdefault(name, value) == value

    class FakeWindowsLockApi:
        def lock(
            self, _descriptor: int, *, exclusive: bool, nonblocking: bool = False
        ) -> object:
            return (exclusive, object())

        def unlock(self, _descriptor: int, _token: object) -> object:
            return None

    monkeypatch.setattr(path_security, "_platform_name", lambda: "nt")
    monkeypatch.setattr(path_security, "_windows_file_api", FakeWindowsFileApi)
    monkeypatch.setattr(
        path_security,
        "_windows_registry_authority_api",
        FakeWindowsRegistryAuthorityApi,
    )
    monkeypatch.setattr(coordination, "_platform_name", lambda: "nt")
    monkeypatch.setattr(coordination, "_windows_lock_api", FakeWindowsLockApi)
    store = BlobStore(tmp_path / f"windows-default-{backend_name}", backend=backend_name)
    try:
        store.put({"value": backend_name}, key="entry")
        assert store.get("entry") == {"value": backend_name}
        assert store.delete("entry") is True
        store.put({"value": "clear"}, key="clear-entry")
        assert store.clear() == 1
        assert flushed_files
        assert authorities
    finally:
        store.close()


def test_barrier_registration_uses_its_descriptor_identity_after_root_replacement(
    tmp_path, monkeypatch
):
    """A candidate created before root replacement never registers under a stale path stat."""
    root = tmp_path / "replace-between-barrier-construction-and-registration"
    root.mkdir()
    retired = tmp_path / "retired-root"
    calls = 0

    def replace_root(_candidate: StoreAdmissionBarrier) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            root.rename(retired)
            root.mkdir()

    monkeypatch.setattr(StoreAdmissionBarrier, "_before_registry_insert", replace_root)
    first = StoreAdmissionBarrier.acquire(root)
    second = StoreAdmissionBarrier.acquire(root)
    try:
        assert first is not second
        assert first._registry_identity == first._root_identity
        assert second._registry_identity == second._root_identity
        assert first._root_identity != second._root_identity
    finally:
        first.release()
        second.release()
        StoreAdmissionBarrier._before_registry_insert = None
    assert first._file_ops._root_fd is None
    assert second._file_ops._root_fd is None


def test_lock_release_failure_never_masks_the_lifecycle_body_and_closes_handle(
    tmp_path, monkeypatch
):
    """Body failures win over unlock failures; standalone release errors stay typed."""
    file_ops = ManagedFileOps(tmp_path)
    locator = file_ops.root / "lock-release.lock"
    closed: list[bool] = []
    original_open_read = file_ops.open_read

    class TrackingHandle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __getattr__(self, name):
            return getattr(self._handle, name)

        def close(self) -> None:
            closed.append(True)
            self._handle.close()

    class FailingWindowsLockApi:
        def lock(
            self, _descriptor: int, *, exclusive: bool, nonblocking: bool = False
        ) -> object:
            assert exclusive
            return object()

        def unlock(self, _descriptor: int, _token: object) -> object:
            raise OSError("injected unlock failure")

    monkeypatch.setattr(file_ops, "open_read", lambda path: TrackingHandle(original_open_read(path)))
    monkeypatch.setattr(coordination, "_platform_name", lambda: "nt")
    monkeypatch.setattr(coordination, "_windows_lock_api", FailingWindowsLockApi)
    try:
        with pytest.raises(RuntimeError, match="body failure"):
            with interprocess_file_lock(
                file_ops, locator, exclusive=True, operation="release_body_test"
            ):
                raise RuntimeError("body failure")
        # Lock-authority validation also opens and closes its bounded marker;
        # the short-lived advisory descriptor still must be closed on the
        # body-failure path.
        first_close_count = len(closed)
        assert first_close_count >= 1

        with pytest.raises(CacheBlobLockReleaseError, match="could not be released") as error:
            with interprocess_file_lock(
                file_ops, locator, exclusive=True, operation="release_only_test"
            ):
                pass
        assert isinstance(error.value.__cause__, OSError)
        assert len(closed) > first_close_count
    finally:
        file_ops.close()


def test_short_lived_admission_lock_rejects_hard_links_and_later_inode_swaps(tmp_path):
    """Admission lock names retain one single-linked inode for the root lifetime."""
    root = tmp_path / "short-lived-lock-root"
    root.mkdir()
    outside = tmp_path / "outside-admission-lock"
    outside.write_bytes(b"lock\n")
    file_ops = ManagedFileOps(root)
    lock_locator = root / "admission.lock"
    try:
        os.link(outside, lock_locator)
        with pytest.raises(CacheUnsafePathError):
            with interprocess_file_lock(
                file_ops, lock_locator, exclusive=True, operation="hard_link"
            ):
                pass
        lock_locator.unlink()

        with interprocess_file_lock(
            file_ops, lock_locator, exclusive=True, operation="establish_identity"
        ):
            pass
        replacement = tmp_path / "replacement-admission-lock"
        replacement.write_bytes(b"lock\n")
        os.replace(replacement, lock_locator)
        with pytest.raises(CacheUnsafePathError):
            with interprocess_file_lock(
                file_ops, lock_locator, exclusive=True, operation="regular_swap"
            ):
                pass
    finally:
        file_ops.close()
