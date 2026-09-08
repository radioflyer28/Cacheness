"""Ownership and admission contracts for ``BlobStore.close``."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import pytest

from cacheness import CacheConfig
from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobCloseTimeoutError,
    CacheBlobStoreClosedError,
)
from cacheness.storage import BlobReceipt, BlobStore
from cacheness.storage.composition import BackendRef, StoreTopology


def _configured_store(root: Path, *, close_wait_seconds: float = 0.02) -> BlobStore:
    """Create a store with an explicit, finite close policy."""
    limits = LifecycleLimits(close_wait_seconds=close_wait_seconds)
    return BlobStore(
        StoreTopology(
            payload=BackendRef(name="filesystem", options={"base_dir": root}),
            authority=BackendRef(name="sqlite", options={"root": root}),
        ),
        cache_dir=root,
        config=CacheConfig(lifecycle_limits=limits),
    )


def _join(thread: Thread) -> None:
    thread.join(timeout=5)
    assert not thread.is_alive(), "worker did not finish within the bounded deadline"


def test_close_rejects_new_work_before_resource_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close drains admitted work while later work fails at public admission."""
    store = _configured_store(tmp_path / "admission")
    entered = Event()
    release = Event()
    close_waiting = Event()
    original_get_metadata = store._authority_lifecycle.get_metadata

    def paused_get_metadata(key: str):
        entered.set()
        assert release.wait(timeout=5)
        return original_get_metadata(key)

    def wait_with_signal(condition, timeout: float) -> None:
        close_waiting.set()
        condition.wait(timeout)

    monkeypatch.setattr(store._authority_lifecycle, "get_metadata", paused_get_metadata)
    monkeypatch.setattr(store._instance_admission, "_wait", wait_with_signal)
    operation = Thread(target=lambda: store.get_metadata("missing"))
    operation.start()
    assert entered.wait(timeout=5)
    closer = Thread(target=store.close)
    closer.start()
    assert close_waiting.wait(timeout=5)

    with pytest.raises(CacheBlobStoreClosedError):
        store.get_metadata("later")

    release.set()
    _join(operation)
    _join(closer)


def test_close_timeout_preserves_resources_for_a_safe_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bounded timeout leaves resources live until the in-flight call finishes."""
    store = _configured_store(tmp_path / "timeout")
    entered = Event()
    release = Event()
    clock = [0.0]

    def advance_clock(_condition, timeout: float) -> None:
        clock[0] += timeout

    monkeypatch.setattr(store._instance_admission, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(store._instance_admission, "_wait", advance_clock)

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

    release.set()
    _join(worker)
    store.close()


def test_owned_authority_close_preserves_data_for_reopen(tmp_path: Path) -> None:
    """Closing releases authority resources without turning close into clear."""
    root = tmp_path / "owned-authority"
    store = _configured_store(root)
    try:
        receipt = store.put_entry({"value": "persist"}, key="persisted")
        assert isinstance(receipt, BlobReceipt)
        assert receipt.key == "persisted"
    finally:
        store.close()

    reopened = _configured_store(root)
    try:
        assert reopened.get("persisted") == {"value": "persist"}
        assert reopened.lifecycle_authority.read_entry("persisted") is not None
    finally:
        reopened.close()
