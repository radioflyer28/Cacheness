"""Deterministic ownership and admission contracts for ``BlobStore.close``."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import pytest

from cacheness import CacheConfig
from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobCloseTimeoutError,
    CacheBlobStoreClosedError,
)
from cacheness.metadata import InMemoryBackend
from cacheness.storage import BlobStore


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
