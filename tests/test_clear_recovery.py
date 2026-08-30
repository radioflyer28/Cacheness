"""Durability and reopen contracts for the bounded BlobStore clear primitive."""

from copy import deepcopy
from pathlib import Path

import pytest

import cacheness.metadata as metadata_module
from cacheness.error_handling import CacheStorageError
from cacheness.metadata import JsonBackend
from cacheness.storage.blob_store import BlobStore


class _SimulatedClearInterruption(BaseException):
    """Represent loss of control between durable clear phases."""


def _entry_data(label: str) -> dict[str, object]:
    """Create a minimal current-format metadata entry for JsonBackend tests."""
    return {
        "description": label,
        "data_type": "object",
        "file_size": len(label),
        "metadata": {"label": label},
    }


def _put_json_payloads(store: BlobStore) -> tuple[list[str], dict[Path, bytes]]:
    """Store two public payloads and retain their managed bytes for comparison."""
    keys = [store.put("first", key="first"), store.put("second", key="second")]
    payload_bytes = {}
    for key in keys:
        entry = store.get_metadata(key)
        assert entry is not None
        payload_path = Path(entry["metadata"]["actual_path"])
        payload_bytes[payload_path] = payload_path.read_bytes()
    return keys, payload_bytes


def _reopen_json_store(root: Path) -> BlobStore:
    """Reopen the exact local JSON topology after a simulated interruption."""
    return BlobStore(root, backend="json")


def test_json_pre_replace_failure_preserves_live_disk_and_memory(
    tmp_path, monkeypatch
):
    """A failed candidate replacement is typed and leaves the prior document authoritative."""
    metadata_path = tmp_path / "cache_metadata.json"
    backend = JsonBackend(metadata_path)
    backend.put_entry("existing", _entry_data("existing"))
    before = deepcopy(backend.load_metadata())

    def fail_replace(source, destination):
        raise OSError("metadata replacement unavailable")

    monkeypatch.setattr(metadata_module.os, "replace", fail_replace)

    with pytest.raises(CacheStorageError, match="metadata"):
        backend.put_entry("candidate", _entry_data("candidate"))

    assert backend.load_metadata() == before
    assert JsonBackend(metadata_path).load_metadata() == before


def test_json_post_replace_directory_fsync_rolls_back_to_prior_document(
    tmp_path, monkeypatch
):
    """A post-replace barrier failure restores the backed-up JSON document durably."""
    metadata_path = tmp_path / "cache_metadata.json"
    backend = JsonBackend(metadata_path)
    backend.put_entry("existing", _entry_data("existing"))
    before = deepcopy(backend.load_metadata())
    directory_fsync_calls = 0

    def fail_only_post_replace_barrier() -> None:
        nonlocal directory_fsync_calls
        directory_fsync_calls += 1
        if directory_fsync_calls == 2:
            raise OSError("post-replace directory fsync unavailable")

    monkeypatch.setattr(
        backend,
        "_fsync_metadata_directory",
        fail_only_post_replace_barrier,
        raising=False,
    )

    with pytest.raises(CacheStorageError, match="metadata"):
        backend.put_entry("candidate", _entry_data("candidate"))

    assert directory_fsync_calls >= 3
    assert backend.load_metadata() == before
    assert JsonBackend(metadata_path).load_metadata() == before


def test_json_unacknowledged_rollback_rereads_live_document_before_typed_error(
    tmp_path, monkeypatch
):
    """An uncertain post-replace outcome never leaves memory claiming stale authority."""
    metadata_path = tmp_path / "cache_metadata.json"
    backend = JsonBackend(metadata_path)
    backend.put_entry("existing", _entry_data("existing"))
    directory_fsync_calls = 0

    def fail_post_replace_and_rollback_barriers() -> None:
        nonlocal directory_fsync_calls
        directory_fsync_calls += 1
        if directory_fsync_calls in (2, 3):
            raise OSError("directory fsync acknowledgement unavailable")

    monkeypatch.setattr(
        backend,
        "_fsync_metadata_directory",
        fail_post_replace_and_rollback_barriers,
        raising=False,
    )

    with pytest.raises(CacheStorageError, match="metadata"):
        backend.put_entry("candidate", _entry_data("candidate"))

    reopened = JsonBackend(metadata_path)
    assert backend.load_metadata() == reopened.load_metadata()


def test_json_prepared_clear_reopens_to_exact_payload_and_metadata_rollback(
    tmp_path, monkeypatch
):
    """A pre-commit interruption retains prepared evidence and rolls it back on reopen."""
    root = tmp_path / "blob-root"
    store = BlobStore(root, backend="json")
    keys, payload_bytes = _put_json_payloads(store)
    metadata_before = deepcopy(store.backend.load_metadata())

    def interrupt_metadata_clear() -> int:
        raise _SimulatedClearInterruption("interrupted while prepared")

    monkeypatch.setattr(store.backend, "clear_all", interrupt_metadata_clear)
    with pytest.raises(_SimulatedClearInterruption):
        store.clear()
    store.close()

    reopened = _reopen_json_store(root)
    try:
        assert reopened.backend.load_metadata() == metadata_before
        assert {path: path.read_bytes() for path in payload_bytes} == payload_bytes
        assert [reopened.get(key) for key in keys] == ["first", "second"]
    finally:
        reopened.close()


def test_json_prepared_clear_recovery_restores_entries_and_all_json_counters(
    tmp_path, monkeypatch
):
    """Rollback restores the complete JSON metadata document, not entries alone."""
    root = tmp_path / "blob-root"
    store = BlobStore(root, backend="json")
    _, payload_bytes = _put_json_payloads(store)
    store.backend.increment_hits()
    store.backend.increment_hits()
    store.backend.increment_misses()
    metadata_before = deepcopy(store.backend.load_metadata())
    clear_all = store.backend.clear_all

    def clear_metadata_then_interrupt() -> int:
        clear_all()
        raise _SimulatedClearInterruption("interrupted after metadata clear")

    monkeypatch.setattr(store.backend, "clear_all", clear_metadata_then_interrupt)
    with pytest.raises(_SimulatedClearInterruption):
        store.clear()
    store.close()

    reopened = _reopen_json_store(root)
    try:
        assert reopened.backend.load_metadata() == metadata_before
        assert {path: path.read_bytes() for path in payload_bytes} == payload_bytes
    finally:
        reopened.close()


def test_json_committed_clear_reopens_to_roll_forward_without_payload_residue(
    tmp_path, monkeypatch
):
    """A post-commit interruption reopens to complete payload erasure exactly once."""
    root = tmp_path / "blob-root"
    store = BlobStore(root, backend="json")
    keys, payload_bytes = _put_json_payloads(store)
    clear_all = store.backend.clear_all
    metadata_cleared = False
    delete = store.guarded_handler_io.file_ops.delete

    def clear_metadata_then_mark_committed() -> int:
        nonlocal metadata_cleared
        count = clear_all()
        metadata_cleared = True
        return count

    def interrupt_finalization(locator):
        if metadata_cleared:
            raise _SimulatedClearInterruption("interrupted after committed metadata clear")
        return delete(locator)

    monkeypatch.setattr(store.backend, "clear_all", clear_metadata_then_mark_committed)
    monkeypatch.setattr(store.guarded_handler_io.file_ops, "delete", interrupt_finalization)
    with pytest.raises(_SimulatedClearInterruption):
        store.clear()
    store.close()

    reopened = _reopen_json_store(root)
    try:
        assert reopened.backend.list_entries() == []
        assert [reopened.get(key) for key in keys] == [None, None]
        assert all(not payload_path.exists() for payload_path in payload_bytes)
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        reopened.close()


def test_successful_json_clear_returns_count_and_leaves_no_tombstones(tmp_path):
    """The completed clear returns its backend count with no recoverable residue."""
    root = tmp_path / "blob-root"
    store = BlobStore(root, backend="json")
    _, payload_bytes = _put_json_payloads(store)

    try:
        assert store.clear() == 2
        assert store.backend.list_entries() == []
        assert all(not payload_path.exists() for payload_path in payload_bytes)
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        store.close()
