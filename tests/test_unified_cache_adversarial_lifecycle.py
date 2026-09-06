"""Deterministic facade schedules for authority-owned lifecycle admission."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import pytest

from cacheness import CacheConfig, cacheness
from cacheness.error_handling import CacheBlobStoreClosedError
from cacheness.storage import BlobStore


def _cache(root: Path):
    """Construct one durable facade without eager compatibility cleanup."""
    return cacheness(
        CacheConfig(
            cache_dir=str(root),
            metadata_backend="sqlite",
            cleanup_on_init=False,
        )
    )


def _join(thread: Thread) -> None:
    """Join an adversarial worker without hiding a deadlock."""
    thread.join(timeout=5)
    assert not thread.is_alive(), "worker did not finish within the bounded wait"


def test_facade_put_admission_blocks_close_until_authority_promotion_exits(
    tmp_path: Path,
) -> None:
    """An admitted facade put owns the same close drain reference as BlobStore.put."""
    cache = _cache(tmp_path / "facade-put-close")
    entered = Event()
    release = Event()
    put_finished = Event()
    close_finished = Event()
    errors: list[BaseException] = []

    def pause_before_promotion(boundary: str) -> None:
        if boundary == "put.before_promotion":
            entered.set()
            assert release.wait(timeout=5)

    def put() -> None:
        try:
            cache.put({"generation": "m1"}, race_key="facade-close")
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)
        finally:
            put_finished.set()

    def close() -> None:
        try:
            cache._cache_blob_store.close()
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)
        finally:
            close_finished.set()

    cache._cache_blob_store.lifecycle.test_hook = pause_before_promotion
    writer = Thread(target=put)
    closer = Thread(target=close)
    try:
        writer.start()
        assert entered.wait(timeout=5)
        closer.start()
        assert not close_finished.wait(timeout=0.2)
        release.set()
        _join(writer)
        _join(closer)
        assert put_finished.is_set()
        assert errors == []
    finally:
        release.set()
        _join(writer)
        _join(closer)
        cache.close()


def test_clear_clear_queued_clear_reopens_after_snapshot_but_cannot_overtake_close(
    tmp_path: Path,
) -> None:
    """Only the active clear owns admission; its queued peer is never counted."""
    store = BlobStore(tmp_path / "clear-admission", backend="json")
    snapshot_entered = Event()
    release_snapshot = Event()
    cleanup_entered = Event()
    release_cleanup = Event()
    unrelated_done = Event()
    close_done = Event()
    first_results: list[int] = []
    second_errors: list[BaseException] = []
    errors: list[BaseException] = []

    def pause_active_clear(boundary: str) -> None:
        if boundary == "clear.snapshot_committed":
            snapshot_entered.set()
            assert release_snapshot.wait(timeout=5)
        if boundary == "clear.before_target_delete":
            cleanup_entered.set()
            assert release_cleanup.wait(timeout=5)

    def first_clear() -> None:
        try:
            first_results.append(store.clear())
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)

    def second_clear() -> None:
        try:
            store.clear()
        except BaseException as error:  # pragma: no cover - asserted below.
            second_errors.append(error)

    def unrelated_put() -> None:
        try:
            store.put("new", key="unrelated")
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)
        finally:
            unrelated_done.set()

    def close() -> None:
        try:
            store.close()
        except BaseException as error:  # pragma: no cover - asserted below.
            errors.append(error)
        finally:
            close_done.set()

    store.lifecycle.test_hook = pause_active_clear
    first = Thread(target=first_clear)
    second = Thread(target=second_clear)
    unrelated = Thread(target=unrelated_put)
    closer = Thread(target=close)
    try:
        store.put("old", key="target")
        first.start()
        assert snapshot_entered.wait(timeout=5)
        second.start()
        assert store._instance_admission.in_flight == 1
        assert len(store._instance_admission._clear_queue) == 1

        release_snapshot.set()
        assert cleanup_entered.wait(timeout=5)
        unrelated.start()
        assert unrelated_done.wait(timeout=5)
        assert store.get("unrelated") == "new"

        closer.start()
        assert not close_done.wait(timeout=0.2)
        with pytest.raises(CacheBlobStoreClosedError):
            store.get("later")
        release_cleanup.set()

        _join(first)
        _join(second)
        _join(unrelated)
        _join(closer)
        assert first_results == [1]
        assert errors == []
        assert len(second_errors) == 1
        assert isinstance(second_errors[0], CacheBlobStoreClosedError)
    finally:
        release_snapshot.set()
        release_cleanup.set()
        _join(first)
        _join(second)
        _join(unrelated)
        _join(closer)
        store.close()
