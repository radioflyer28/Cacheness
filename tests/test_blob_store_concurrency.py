"""Deterministic direct BlobStore coordination and read-race contracts."""

from __future__ import annotations

from contextlib import contextmanager
import threading

import pytest

from cacheness.error_handling import CacheBlobLifecycleConflictError
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.coordination import KeyCoordinatorRegistry


def _join(thread: threading.Thread) -> None:
    """Join one deterministic test worker without hiding a deadlock."""
    thread.join(timeout=5)
    assert not thread.is_alive(), "worker did not finish within the bounded wait"


def test_key_registry_retires_entries_after_exception_and_high_cardinality():
    """Refcounted entries are removed after every normal and exceptional holder."""
    registry = KeyCoordinatorRegistry()

    with pytest.raises(RuntimeError, match="intentional"):
        with registry.hold("exception-key"):
            assert registry.size == 1
            raise RuntimeError("intentional")
    assert registry.size == 0

    for number in range(128):
        with registry.hold(f"key-{number}"):
            assert registry.size == 1
    assert registry.size == 0


def test_key_registry_sorts_multi_key_acquisition_without_deadlock():
    """Inverted callers use one stable order and complete under bounded joins."""
    registry = KeyCoordinatorRegistry()
    first_entered = threading.Event()
    release_first = threading.Event()
    errors: list[BaseException] = []

    def first() -> None:
        try:
            with registry.hold_many(("z-key", "a-key")):
                first_entered.set()
                assert release_first.wait(timeout=5)
        except BaseException as exc:  # pragma: no cover - re-raised below.
            errors.append(exc)

    def second() -> None:
        try:
            with registry.hold_many(("a-key", "z-key")):
                pass
        except BaseException as exc:  # pragma: no cover - re-raised below.
            errors.append(exc)

    first_thread = threading.Thread(target=first)
    first_thread.start()
    assert first_entered.wait(timeout=5)
    second_thread = threading.Thread(target=second)
    second_thread.start()
    release_first.set()
    _join(first_thread)
    _join(second_thread)
    assert errors == []
    assert registry.size == 0


def test_independent_write_write_race_has_one_cas_winner(tmp_path):
    """Independent stores prove CAS, not the local lock, selects the winner."""
    root = tmp_path / "write-write"
    first = BlobStore(root, backend="json")
    second = BlobStore(root, backend="json")
    barrier = threading.Barrier(2)
    results: list[str] = []
    errors: list[BaseException] = []

    def pause_before_publish(seam, _record) -> None:
        if seam == "manifest_publish":
            barrier.wait(timeout=5)

    first.lifecycle.fault_hook = pause_before_publish
    second.lifecycle.fault_hook = pause_before_publish

    def write(store: BlobStore, value: str) -> None:
        try:
            results.append(store.put(value, key="same-key"))
        except BaseException as exc:  # pragma: no cover - asserted below.
            errors.append(exc)

    first_thread = threading.Thread(target=write, args=(first, "first"))
    second_thread = threading.Thread(target=write, args=(second, "second"))
    try:
        first_thread.start()
        second_thread.start()
        _join(first_thread)
        _join(second_thread)

        assert len(results) == 1
        assert len(errors) == 1
        assert isinstance(errors[0], CacheBlobLifecycleConflictError)
        assert first.get("same-key") in {"first", "second"}
    finally:
        second.close()
        first.close()


def test_distinct_key_put_completes_while_another_key_is_pre_cas(tmp_path):
    """A per-key holder never serializes an unrelated ordinary write."""
    store = BlobStore(tmp_path / "distinct", backend="json")
    key_a_entered = threading.Event()
    release_key_a = threading.Event()
    key_b_finished = threading.Event()
    errors: list[BaseException] = []

    def pause_key_a(seam, record) -> None:
        if seam == "manifest_publish" and record.key == "key-a":
            key_a_entered.set()
            assert release_key_a.wait(timeout=5)

    store.lifecycle.fault_hook = pause_key_a

    def put(key: str, value: str, completed: threading.Event | None = None) -> None:
        try:
            store.put(value, key=key)
        except BaseException as exc:  # pragma: no cover - asserted below.
            errors.append(exc)
        finally:
            if completed is not None:
                completed.set()

    key_a_thread = threading.Thread(target=put, args=("key-a", "a"))
    key_a_thread.start()
    assert key_a_entered.wait(timeout=5)
    key_b_thread = threading.Thread(
        target=put, args=("key-b", "b", key_b_finished)
    )
    key_b_thread.start()
    try:
        assert key_b_finished.wait(timeout=5)
        assert store.get("key-b") == "b"
    finally:
        release_key_a.set()
        _join(key_a_thread)
        _join(key_b_thread)
        store.close()
    assert errors == []


def test_read_retries_once_when_an_independent_writer_commits_new_generation(
    tmp_path,
):
    """A read discards its first snapshot when M2 proves a newer commit."""
    root = tmp_path / "read-write"
    reader = BlobStore(root, backend="json")
    writer = BlobStore(root, backend="json")
    key = "race-key"
    snapshots = 0
    try:
        reader.put("first", key=key)
        original_snapshot = reader.guarded_handler_io.open_snapshot

        @contextmanager
        def replace_before_first_snapshot(locator, metadata):
            nonlocal snapshots
            snapshots += 1
            if snapshots == 1:
                writer.put("second", key=key)
            with original_snapshot(locator, metadata) as snapshot:
                yield snapshot

        reader.guarded_handler_io.open_snapshot = replace_before_first_snapshot

        assert reader.get(key) == "second"
        assert snapshots == 2
    finally:
        writer.close()
        reader.close()


def test_read_delete_race_is_a_typed_lifecycle_conflict(tmp_path):
    """A delete after M1 cannot turn a committed read into a payload miss."""
    root = tmp_path / "read-delete"
    reader = BlobStore(root, backend="json")
    deleter = BlobStore(root, backend="json")
    key = "race-key"
    snapshots = 0
    try:
        reader.put("payload", key=key)
        original_snapshot = reader.guarded_handler_io.open_snapshot

        @contextmanager
        def delete_before_first_snapshot(locator, metadata):
            nonlocal snapshots
            snapshots += 1
            if snapshots == 1:
                assert deleter.delete(key) is True
            with original_snapshot(locator, metadata) as snapshot:
                yield snapshot

        reader.guarded_handler_io.open_snapshot = delete_before_first_snapshot

        with pytest.raises(CacheBlobLifecycleConflictError):
            reader.get(key)
        assert snapshots == 1
    finally:
        deleter.close()
        reader.close()
