"""Durability and reopen contracts for the bounded BlobStore clear primitive."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import threading

import numpy as np
import pytest

from cacheness import CacheConfig, cacheness
import cacheness.metadata as metadata_module
import cacheness.storage.clear_recovery as clear_recovery
from cacheness.error_handling import CacheBlobBackendError, CacheStorageError
from cacheness.metadata import (
    CachedMetadataBackend,
    InMemoryBackend,
    JsonBackend,
    SqliteBackend,
)
from cacheness.storage.backends.postgresql_backend import PostgresBackend
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.path_security import encode_physical_name


class _SimulatedClearInterruption(BaseException):
    """Represent loss of control between durable clear phases."""


class _NoopPostgresEngine:
    """Avoid real PostgreSQL construction while allowing sentinel cleanup."""

    def dispose(self) -> None:
        """Match the engine cleanup method without opening a connection."""


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


@pytest.mark.parametrize("writer", ("blob_store", "unified_cache"))
@pytest.mark.parametrize("retirement_fault", ("unlink", "directory_fsync"))
@pytest.mark.parametrize("case", ("first_write", "cross_format_overwrite"))
def test_json_backup_retirement_does_not_revoke_authoritative_candidate(
    tmp_path, monkeypatch, writer, retirement_fault, case
):
    """A post-authority backup fault keeps the candidate named by live metadata."""
    root = tmp_path / f"json-retirement-{writer}-{retirement_fault}-{case}"
    if writer == "blob_store":
        owner = BlobStore(root, backend="json")

        def put(value, key):
            return owner.put(value, key=key)

        def get(key):
            return owner.get(key)

        metadata_backend = owner.backend
    else:
        owner = _unified_cache(root, "json")

        def put(value, key):
            return owner.put(value, retirement_case=key)

        def get(key):
            return owner.get(retirement_case=key)

        metadata_backend = owner.metadata_backend

    target_key = "target"
    try:
        # A different established entry makes this the first write for target
        # while exercising the JSON backup-retirement region.
        put("metadata-already-exists", "anchor")
        if case == "cross_format_overwrite":
            put("old format", target_key)

        backup_path = metadata_backend.metadata_file.parent / (
            f".{metadata_backend.metadata_file.name}.backup"
        )
        if retirement_fault == "unlink":
            unlink = Path.unlink

            def fail_backup_retirement(path, *args, **kwargs):
                if path == backup_path:
                    raise OSError("backup retirement unavailable")
                return unlink(path, *args, **kwargs)

            monkeypatch.setattr(Path, "unlink", fail_backup_retirement)
        else:
            fsync_metadata_directory = metadata_backend._fsync_metadata_directory
            directory_syncs = 0

            def fail_post_unlink_acknowledgement():
                nonlocal directory_syncs
                directory_syncs += 1
                if directory_syncs == 3:
                    raise OSError("backup retirement acknowledgement unavailable")
                return fsync_metadata_directory()

            monkeypatch.setattr(
                metadata_backend,
                "_fsync_metadata_directory",
                fail_post_unlink_acknowledgement,
            )

        replacement = np.array([1, 2, 3]) if case == "cross_format_overwrite" else "new"
        put(replacement, target_key)

        if writer == "blob_store":
            entry = owner.get_metadata(target_key)
        else:
            cache_key = owner._create_cache_key({"retirement_case": target_key})
            entry = owner.metadata_backend.get_entry(cache_key)
        assert entry is not None
        candidate = Path(entry["metadata"]["actual_path"])
        assert candidate.exists()
        read_value = get(target_key)
        if case == "cross_format_overwrite":
            np.testing.assert_array_equal(read_value, replacement)
        else:
            assert read_value == replacement
    finally:
        owner.close()


def test_json_prepared_clear_reopens_to_exact_payload_and_metadata_rollback(
    tmp_path, monkeypatch
):
    """A pre-commit interruption retains prepared evidence and rolls it back on reopen."""
    root = tmp_path / "blob-root"
    store = BlobStore(root, backend="json")
    keys, payload_bytes = _put_json_payloads(store)
    metadata_before = deepcopy(store.backend.load_metadata())
    assert all("-candidate-" in path.name for path in payload_bytes)

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


def _put_payloads(store: BlobStore) -> tuple[list[str], dict[Path, bytes]]:
    """Populate any supported backend with two payloads and their exact bytes."""
    keys = [store.put("first", key="first"), store.put("second", key="second")]
    payloads = {}
    for key in keys:
        entry = store.get_metadata(key)
        assert entry is not None
        payload_path = Path(entry["metadata"]["actual_path"])
        payloads[payload_path] = payload_path.read_bytes()
    return keys, payloads


@pytest.mark.parametrize("backend_name", ("json", "sqlite", "memory"))
@pytest.mark.parametrize("boundary", ("serialization", "candidate_write", "replace"))
def test_committed_journal_prepublication_failure_rolls_back_exactly(
    tmp_path, monkeypatch, backend_name, boundary
):
    """A proven prepared journal is rolled back before the clear error returns."""
    root = tmp_path / f"commit-publication-{backend_name}-{boundary}"
    backend = InMemoryBackend() if backend_name == "memory" else backend_name
    store = BlobStore(root, backend=backend)
    try:
        keys, payloads = _put_payloads(store)
        store.backend.increment_hits()
        store.backend.increment_misses()
        before = _backend_snapshot(store.backend, keys)
        coordinator = store._clear_recovery
        assert coordinator is not None

        if boundary == "serialization":
            encode_journal = coordinator._encode_journal

            def fail_committed_serialization(journal):
                if journal["state"] == "committed":
                    raise RuntimeError("committed journal serialization unavailable")
                return encode_journal(journal)

            monkeypatch.setattr(
                coordinator,
                "_encode_journal",
                fail_committed_serialization,
            )
        elif boundary == "candidate_write":
            write_stream = store.guarded_handler_io.file_ops.write_stream_to_locator

            def fail_committed_candidate_write(locator, source):
                if Path(locator) == coordinator.journal_path:
                    raise RuntimeError("committed journal candidate write unavailable")
                return write_stream(locator, source)

            monkeypatch.setattr(
                store.guarded_handler_io.file_ops,
                "write_stream_to_locator",
                fail_committed_candidate_write,
            )
        else:
            replace_journal = coordinator._replace_journal

            def fail_before_committed_replace(journal):
                if journal["state"] == "committed":
                    raise RuntimeError("committed journal replacement unavailable")
                return replace_journal(journal)

            monkeypatch.setattr(
                coordinator,
                "_replace_journal",
                fail_before_committed_replace,
            )

        with pytest.raises((RuntimeError, CacheStorageError)):
            store.clear()

        assert _backend_snapshot(store.backend, keys) == before
        assert {path: path.read_bytes() for path in payloads} == payloads
        assert not coordinator.journal_path.exists()
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        store.close()


@pytest.mark.parametrize("backend_name", ("json", "sqlite", "memory"))
def test_unacknowledged_committed_journal_poison_rejects_intervening_operations(
    tmp_path, monkeypatch, backend_name
):
    """A post-replace journal barrier failure leaves evidence and rejects all work."""
    root = tmp_path / f"commit-publication-{backend_name}-directory-fsync"
    backend = InMemoryBackend() if backend_name == "memory" else backend_name
    store = BlobStore(root, backend=backend)
    try:
        _put_payloads(store)
        coordinator = store._clear_recovery
        assert coordinator is not None
        write_bytes_durable = store.guarded_handler_io.file_ops.write_bytes_durable

        def fail_postreplace_journal_acknowledgement(locator, data):
            written = write_bytes_durable(locator, data)
            if Path(locator) == coordinator.journal_path:
                raise OSError("committed journal directory fsync unavailable")
            return written

        monkeypatch.setattr(
            store.guarded_handler_io.file_ops,
            "write_bytes_durable",
            fail_postreplace_journal_acknowledgement,
        )

        with pytest.raises(CacheStorageError, match="unresolved recovery outcome"):
            store.clear()

        assert coordinator.journal_path.exists()
        with pytest.raises(CacheStorageError, match="terminal reconciliation"):
            store.put("must not publish", key="intervening")
        with pytest.raises(CacheStorageError, match="terminal reconciliation"):
            store.get("first")
        assert not list(root.glob("*intervening*candidate-*"))
    finally:
        store.close()


def test_base_exception_during_committed_publication_poison_rejects_live_store(
    tmp_path, monkeypatch
):
    """A caught interruption after metadata clear cannot leave the owner usable."""
    store = BlobStore(tmp_path / "base-exception-publication", backend="json")
    try:
        _put_payloads(store)
        coordinator = store._clear_recovery
        assert coordinator is not None

        def interrupt_committed_publication(_journal):
            raise _SimulatedClearInterruption("interrupted while publishing committed journal")

        monkeypatch.setattr(
            coordinator,
            "_replace_journal",
            interrupt_committed_publication,
        )
        with pytest.raises(_SimulatedClearInterruption):
            store.clear()
        with pytest.raises(CacheStorageError, match="terminal reconciliation"):
            store.put("must not publish", key="intervening")
    finally:
        store.close()


@pytest.mark.parametrize("backend_name", ("json", "sqlite"))
def test_blobstore_puts_are_rejected_before_candidate_creation_while_admitted(
    tmp_path, backend_name
):
    """Same- and two-instance writes cannot race a root's clear snapshot."""
    root = tmp_path / f"blob-put-admission-{backend_name}"
    owner = BlobStore(root, backend=backend_name)
    contender = BlobStore(root, backend=backend_name)
    try:
        keys, payloads = _put_payloads(owner)
        before = _backend_snapshot(owner.backend, keys)
        candidates_before = set(root.glob("*candidate-*"))
        coordinator = owner._clear_recovery
        assert coordinator is not None

        with coordinator.admission():
            with pytest.raises(CacheStorageError):
                owner.put("same instance", key="same-instance")
            with pytest.raises(CacheStorageError):
                contender.put("other instance", key="other-instance")

        assert _backend_snapshot(owner.backend, keys) == before
        assert {path: path.read_bytes() for path in payloads} == payloads
        assert set(root.glob("*candidate-*")) == candidates_before
    finally:
        contender.close()
        owner.close()


@pytest.mark.parametrize("backend_name", ("json", "sqlite"))
def test_unified_cache_puts_are_rejected_before_candidate_creation_while_admitted(
    tmp_path, backend_name
):
    """UnifiedCache shares the same root admission across cache instances."""
    root = tmp_path / f"unified-put-admission-{backend_name}"
    owner = _unified_cache(root, backend_name)
    contender = _unified_cache(root, backend_name)
    try:
        keys, payloads = _put_unified_payloads(owner)
        before = _backend_snapshot(owner.metadata_backend, keys)
        candidates_before = set(root.glob("*candidate-*"))
        coordinator = owner._clear_recovery
        assert coordinator is not None

        with coordinator.admission():
            with pytest.raises(CacheStorageError):
                owner.put({"value": "same"}, put_admission="same-instance")
            with pytest.raises(CacheStorageError):
                contender.put({"value": "other"}, put_admission="other-instance")

        assert _backend_snapshot(owner.metadata_backend, keys) == before
        assert {path: path.read_bytes() for path in payloads} == payloads
        assert set(root.glob("*candidate-*")) == candidates_before
    finally:
        contender.close()
        owner.close()


@pytest.mark.parametrize("backend_name", ("json", "sqlite"))
@pytest.mark.parametrize("writer_instance", ("same", "second"))
def test_blobstore_live_put_waits_for_clear_then_publishes_linearly(
    tmp_path, monkeypatch, backend_name, writer_instance
):
    """A live same- or second-instance put cannot be discarded by a clear."""
    root = tmp_path / f"blob-live-put-clear-{backend_name}-{writer_instance}"
    owner = BlobStore(root, backend=backend_name)
    contender = None
    try:
        old_keys, old_payloads = _put_payloads(owner)
        contender = (
            owner
            if writer_instance == "same"
            else BlobStore(root, backend=backend_name)
        )
        coordinator = owner._clear_recovery
        assert coordinator is not None
        entered = threading.Event()
        release = threading.Event()
        clear_errors: list[BaseException] = []
        put_errors: list[BaseException] = []
        put_complete = threading.Event()
        put_result: list[str] = []
        stage_mapping = coordinator._stage_mapping

        def pause_after_prepared_journal(mapping):
            entered.set()
            assert coordinator.journal_path.exists()
            assert release.wait(timeout=5)
            return stage_mapping(mapping)

        def run_clear():
            try:
                owner.clear()
            except BaseException as exc:
                clear_errors.append(exc)

        def run_put():
            try:
                put_result.append(contender.put("raced", key="raced"))
            except BaseException as exc:
                put_errors.append(exc)
            finally:
                put_complete.set()

        monkeypatch.setattr(coordinator, "_stage_mapping", pause_after_prepared_journal)
        clear_thread = threading.Thread(target=run_clear)
        clear_thread.start()
        assert entered.wait(timeout=5)
        put_thread = threading.Thread(target=run_put)
        put_thread.start()
        assert not put_complete.wait(timeout=0.2)

        release.set()
        clear_thread.join(timeout=5)
        put_thread.join(timeout=5)
        assert not clear_thread.is_alive()
        assert not put_thread.is_alive()
        assert clear_errors == []
        assert put_errors == []
        assert put_result == ["raced"]
        assert all(not path.exists() for path in old_payloads)
        assert contender.get("raced") == "raced"
        entry = contender.get_metadata("raced")
        assert entry is not None
        candidate = Path(entry["metadata"]["actual_path"])
        assert candidate.exists()
        assert {candidate} == set(root.glob("*candidate-*"))
        assert all(contender.get(key) is None for key in old_keys)
    finally:
        if contender is not None and contender is not owner:
            contender.close()
        owner.close()


@pytest.mark.parametrize("backend_name", ("json", "sqlite"))
@pytest.mark.parametrize("writer_instance", ("same", "second"))
def test_unified_cache_live_put_waits_for_clear_then_publishes_linearly(
    tmp_path, monkeypatch, backend_name, writer_instance
):
    """UnifiedCache serializes live same- and second-instance writers."""
    root = tmp_path / f"unified-live-put-clear-{backend_name}-{writer_instance}"
    owner = _unified_cache(root, backend_name)
    contender = None
    try:
        old_keys, old_payloads = _put_unified_payloads(owner)
        contender = (
            owner
            if writer_instance == "same"
            else _unified_cache(root, backend_name)
        )
        coordinator = owner._clear_recovery
        assert coordinator is not None
        entered = threading.Event()
        release = threading.Event()
        clear_errors: list[BaseException] = []
        put_errors: list[BaseException] = []
        put_complete = threading.Event()
        put_result: list[str] = []
        stage_mapping = coordinator._stage_mapping

        def pause_after_prepared_journal(mapping):
            entered.set()
            assert coordinator.journal_path.exists()
            assert release.wait(timeout=5)
            return stage_mapping(mapping)

        def run_clear():
            try:
                owner.clear_all()
            except BaseException as exc:
                clear_errors.append(exc)

        def run_put():
            try:
                put_result.append(contender.put({"value": "raced"}, race="raced"))
            except BaseException as exc:
                put_errors.append(exc)
            finally:
                put_complete.set()

        monkeypatch.setattr(coordinator, "_stage_mapping", pause_after_prepared_journal)
        clear_thread = threading.Thread(target=run_clear)
        clear_thread.start()
        assert entered.wait(timeout=5)
        put_thread = threading.Thread(target=run_put)
        put_thread.start()
        assert not put_complete.wait(timeout=0.2)

        release.set()
        clear_thread.join(timeout=5)
        put_thread.join(timeout=5)
        assert not clear_thread.is_alive()
        assert not put_thread.is_alive()
        assert clear_errors == []
        assert put_errors == []
        assert len(put_result) == 1
        assert all(not path.exists() for path in old_payloads)
        assert contender.get(race="raced") == {"value": "raced"}
        entry = contender.metadata_backend.get_entry(put_result[0])
        assert entry is not None
        candidate = Path(entry["metadata"]["actual_path"])
        assert candidate.exists()
        assert {candidate} == set(root.glob("*candidate-*"))
        assert all(contender.metadata_backend.get_entry(key) is None for key in old_keys)
    finally:
        if contender is not None and contender is not owner:
            contender.close()
        owner.close()


def test_preconstructed_json_blobstore_close_cannot_resurrect_cleared_metadata(
    tmp_path,
):
    """A non-mutating close cannot republish another store's stale JSON view."""
    root = tmp_path / "blob-close-after-clear"
    owner = BlobStore(root, backend="json")
    contender = None
    reopened = None
    try:
        key = owner.put("owned by the first store", key="before-clear")
        entry = owner.get_metadata(key)
        assert entry is not None
        payload_path = Path(entry["metadata"]["actual_path"])
        contender = BlobStore(root, backend="json")

        assert owner.clear() == 1
        assert not payload_path.exists()
        contender.close()
        contender = None

        reopened = BlobStore(root, backend="json")
        assert reopened.get_metadata(key) is None
        assert reopened.list() == []
    finally:
        if reopened is not None:
            reopened.close()
        if contender is not None:
            contender.close()
        owner.close()


def test_preconstructed_json_unified_cache_close_cannot_resurrect_cleared_metadata(
    tmp_path,
):
    """UnifiedCache close inherits the non-resurrecting JSON close contract."""
    root = tmp_path / "unified-close-after-clear"
    owner = _unified_cache(root, "json")
    contender = None
    reopened = None
    try:
        key = owner.put({"value": "owned"}, close_case="before-clear")
        entry = owner.metadata_backend.get_entry(key)
        assert entry is not None
        payload_path = Path(entry["metadata"]["actual_path"])
        contender = _unified_cache(root, "json")

        assert owner.clear_all() == 1
        assert not payload_path.exists()
        contender.close()
        contender = None

        reopened = _unified_cache(root, "json")
        assert reopened.metadata_backend.get_entry(key) is None
        assert reopened.list_entries() == []
    finally:
        if reopened is not None:
            reopened.close()
        if contender is not None:
            contender.close()
        owner.close()


@pytest.mark.parametrize(
    "reader",
    (
        "query_meta",
        "query_custom",
        "query_custom_metadata",
        "query_custom_session",
        "get_custom_metadata_for_entry",
    ),
)
def test_prepared_sqlite_clear_rejects_every_public_unified_query_read(
    tmp_path, monkeypatch, reader
):
    """A preconstructed SQLite reader cannot observe prepared-clear state."""
    root = tmp_path / f"prepared-query-read-{reader}"
    owner = _unified_cache(root, "sqlite", store_cache_key_params=True)
    contender = None
    try:
        keys, _ = _put_unified_payloads(owner)
        contender = _unified_cache(root, "sqlite", store_cache_key_params=True)
        _interrupted_clear(owner.metadata_backend, monkeypatch)

        with pytest.raises(_SimulatedClearInterruption):
            owner.clear_all()

        with pytest.raises(CacheStorageError, match="prepared clear journal"):
            if reader == "query_meta":
                contender.query_meta(clear_case="first")
            elif reader == "query_custom":
                contender.query_custom("not_reached")
            elif reader == "query_custom_metadata":
                contender.query_custom_metadata("not_reached")
            elif reader == "query_custom_session":
                with contender.query_custom_session("not_reached"):
                    pass
            else:
                contender.get_custom_metadata_for_entry(cache_key=keys[0])
    finally:
        if contender is not None:
            contender.close()
        owner.close()


@pytest.mark.parametrize(
    "reader",
    (
        "query_meta",
        "query_custom",
        "query_custom_metadata",
        "query_custom_session",
        "get_custom_metadata_for_entry",
    ),
)
def test_poisoned_sqlite_clear_rejects_every_public_unified_query_read(
    tmp_path, monkeypatch, reader
):
    """Publication uncertainty fails closed before each public query API runs."""
    root = tmp_path / f"poisoned-query-read-{reader}"
    cache = _unified_cache(root, "sqlite", store_cache_key_params=True)
    try:
        keys, _ = _put_unified_payloads(cache)
        coordinator = cache._clear_recovery
        assert coordinator is not None

        def interrupt_committed_publication(_journal):
            raise _SimulatedClearInterruption("interrupted committed publication")

        monkeypatch.setattr(
            coordinator,
            "_replace_journal",
            interrupt_committed_publication,
        )
        with pytest.raises(_SimulatedClearInterruption):
            cache.clear_all()

        with pytest.raises(CacheStorageError, match="terminal reconciliation"):
            if reader == "query_meta":
                cache.query_meta(clear_case="first")
            elif reader == "query_custom":
                cache.query_custom("not_reached")
            elif reader == "query_custom_metadata":
                cache.query_custom_metadata("not_reached")
            elif reader == "query_custom_session":
                with cache.query_custom_session("not_reached"):
                    pass
            else:
                cache.get_custom_metadata_for_entry(cache_key=keys[0])
    finally:
        cache.close()


def test_unified_query_meta_reads_committed_sqlite_clear_authority(
    tmp_path, monkeypatch
):
    """A committed clear exposes its empty authority even before cleanup retries."""
    cache = _unified_cache(
        tmp_path / "committed-query-read",
        "sqlite",
        store_cache_key_params=True,
    )
    try:
        _put_unified_payloads(cache)
        coordinator = cache._clear_recovery
        assert coordinator is not None

        def fail_tombstone_reclamation(_journal, *, wrap_errors):
            del wrap_errors
            raise RuntimeError("committed tombstone reclamation unavailable")

        monkeypatch.setattr(
            coordinator,
            "_roll_forward_committed",
            fail_tombstone_reclamation,
        )
        with pytest.raises(RuntimeError, match="tombstone reclamation"):
            cache.clear_all()

        assert coordinator.journal_path.exists()
        assert cache.query_meta(clear_case="first") == []
    finally:
        cache.close()


def test_query_custom_session_holds_read_admission_for_its_with_lifetime(tmp_path):
    """A clear waits until a public custom-query context has finished using it."""
    from uuid import uuid4

    from sqlalchemy import Column, String

    from cacheness.custom_metadata import (
        CustomMetadataBase,
        _reset_registry,
        custom_metadata_model,
    )
    from cacheness.metadata import Base

    suffix = uuid4().hex
    schema_name = f"clear_query_{suffix}"

    @custom_metadata_model(schema_name)
    class QueryMetadata(Base, CustomMetadataBase):
        __tablename__ = f"custom_clear_query_{suffix}"

        value = Column(String(20), nullable=False)

    cache = _unified_cache(tmp_path / "query-session-admission", "sqlite")
    clear_started = threading.Event()
    clear_finished = threading.Event()
    clear_errors: list[BaseException] = []
    try:
        _put_unified_payloads(cache)

        def run_clear():
            clear_started.set()
            try:
                cache.clear_all()
            except BaseException as exc:
                clear_errors.append(exc)
            finally:
                clear_finished.set()

        with cache.query_custom_session(schema_name):
            clear_thread = threading.Thread(target=run_clear)
            clear_thread.start()
            assert clear_started.wait(timeout=5)
            assert not clear_finished.wait(timeout=0.2)

        clear_thread.join(timeout=5)
        assert not clear_thread.is_alive()
        assert clear_errors == []
    finally:
        cache.close()
        _reset_registry()


def _backend_snapshot(backend, keys: list[str]) -> dict[str, object]:
    """Capture both entries and aggregate counters for exact recovery assertions."""
    return {
        "entries": [deepcopy(backend.get_entry(key)) for key in keys],
        "stats": deepcopy(backend.get_stats()),
    }


def _interrupted_clear(backend, monkeypatch) -> None:
    """Make the next metadata clear look like process loss after prepared staging."""
    def interrupt() -> int:
        raise _SimulatedClearInterruption("simulated process loss")

    monkeypatch.setattr(backend, "clear_all", interrupt)


def _journal_for(store: BlobStore, keys: list[str]) -> dict[str, object]:
    """Build a current-format attacker journal before mutating one field in tests."""
    coordinator = store._clear_recovery
    assert coordinator is not None
    mappings = []
    for key in keys:
        entry = store.get_metadata(key)
        assert entry is not None
        mappings.append((key, Path(entry["metadata"]["actual_path"])))
    journal = coordinator._new_prepared_journal(mappings)
    snapshot_entries = journal["metadata_snapshot"]["entries"]
    journal["metadata_snapshot"]["entries"] = {
        key: snapshot_entries[key] for key in keys
    }
    return journal


def _write_untrusted_journal(store: BlobStore, journal: dict[str, object]) -> None:
    """Bypass managed writes to model pre-existing storage-controlled journal bytes."""
    coordinator = store._clear_recovery
    assert coordinator is not None
    coordinator.journal_path.write_text(json.dumps(journal), encoding="utf-8")


def _unified_cache(
    root: Path, backend_name: str, *, store_cache_key_params: bool = False
):
    """Construct one production-selected UnifiedCache metadata topology."""
    return cacheness(
        CacheConfig(
            cache_dir=str(root),
            metadata_backend=backend_name,
            cleanup_on_init=False,
            store_cache_key_params=store_cache_key_params,
        )
    )


def _put_unified_payloads(cache) -> tuple[list[str], dict[Path, bytes]]:
    """Store two UnifiedCache values and retain their exact payload evidence."""
    keys = [
        cache.put({"value": "first"}, clear_case="first"),
        cache.put({"value": "second"}, clear_case="second"),
    ]
    payloads = {}
    for key in keys:
        entry = cache.metadata_backend.get_entry(key)
        assert entry is not None
        payload_path = Path(entry["metadata"]["actual_path"])
        payloads[payload_path] = payload_path.read_bytes()
    return keys, payloads


@pytest.mark.parametrize("backend_name", ("json", "sqlite", "memory"))
@pytest.mark.parametrize(
    "failure_point",
    (
        "tombstone_1",
        "tombstone_2",
        "original_1",
        "original_2",
        "metadata",
    ),
)
def test_unified_cache_clear_rolls_back_every_precommit_failure(
    tmp_path, monkeypatch, backend_name, failure_point
):
    """Every clear pre-commit fault restores exact entries, counters, and bytes."""
    root = tmp_path / f"unified-{backend_name}-{failure_point}"
    cache = _unified_cache(root, backend_name)

    try:
        keys, payloads = _put_unified_payloads(cache)
        cache.metadata_backend.increment_hits()
        cache.metadata_backend.increment_hits()
        cache.metadata_backend.increment_misses()
        before = _backend_snapshot(cache.metadata_backend, keys)

        if failure_point.startswith("tombstone"):
            expected_position = int(failure_point.rsplit("_", maxsplit=1)[1])
            write_stream = cache.guarded_handler_io.file_ops.write_stream_to_locator
            write_count = 0

            def fail_one_tombstone_stage(locator, source):
                nonlocal write_count
                if Path(locator).name.startswith("clear-tombstone-"):
                    write_count += 1
                    if write_count == expected_position:
                        raise RuntimeError("tombstone staging unavailable")
                return write_stream(locator, source)

            monkeypatch.setattr(
                cache.guarded_handler_io.file_ops,
                "write_stream_to_locator",
                fail_one_tombstone_stage,
            )
        elif failure_point.startswith("original"):
            expected_position = int(failure_point.rsplit("_", maxsplit=1)[1])
            delete = cache.guarded_handler_io.file_ops.delete
            delete_durable = cache.guarded_handler_io.file_ops.delete_durable
            delete_count = 0
            durable_delete_in_progress = False

            def fail_one_original_delete(locator, delete_operation):
                nonlocal delete_count
                if Path(locator) in payloads:
                    delete_count += 1
                    if delete_count == expected_position:
                        raise RuntimeError("original deletion unavailable")
                return delete_operation(locator)

            def fail_one_legacy_original_delete(locator):
                if durable_delete_in_progress:
                    return delete(locator)
                return fail_one_original_delete(locator, delete)

            def fail_one_durable_original_delete(locator):
                nonlocal durable_delete_in_progress
                durable_delete_in_progress = True
                try:
                    return fail_one_original_delete(locator, delete_durable)
                finally:
                    durable_delete_in_progress = False

            monkeypatch.setattr(
                cache.guarded_handler_io.file_ops,
                "delete",
                fail_one_legacy_original_delete,
            )
            monkeypatch.setattr(
                cache.guarded_handler_io.file_ops,
                "delete_durable",
                fail_one_durable_original_delete,
            )
        else:
            clear_metadata = cache.metadata_backend.clear_all

            def clear_metadata_then_raise():
                clear_metadata()
                raise RuntimeError("metadata clear unavailable")

            monkeypatch.setattr(
                cache.metadata_backend,
                "clear_all",
                clear_metadata_then_raise,
            )

        with pytest.raises(RuntimeError):
            cache.clear_all()

        if failure_point.startswith("original"):
            assert delete_count == expected_position
        assert _backend_snapshot(cache.metadata_backend, keys) == before
        assert {path: path.read_bytes() for path in payloads} == payloads
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        cache.close()


@pytest.mark.parametrize("backend_name", ("json", "sqlite"))
def test_unified_cache_prepared_base_exception_reopens_to_exact_rollback(
    tmp_path, monkeypatch, backend_name
):
    """A prepared BaseException rollback is exact and idempotent across reopens."""
    root = tmp_path / f"unified-prepared-{backend_name}"
    cache = _unified_cache(root, backend_name)
    keys, payloads = _put_unified_payloads(cache)
    cache.metadata_backend.increment_hits()
    cache.metadata_backend.increment_hits()
    cache.metadata_backend.increment_misses()
    before = _backend_snapshot(cache.metadata_backend, keys)
    clear_metadata = cache.metadata_backend.clear_all

    def clear_metadata_then_interrupt():
        clear_metadata()
        raise _SimulatedClearInterruption("prepared metadata interruption")

    monkeypatch.setattr(
        cache.metadata_backend,
        "clear_all",
        clear_metadata_then_interrupt,
    )
    with pytest.raises(_SimulatedClearInterruption):
        cache.clear_all()
    assert (root / ".cacheness-clear-journal-v1.json").exists()
    cache.close()

    reopened = _unified_cache(root, backend_name)
    try:
        assert _backend_snapshot(reopened.metadata_backend, keys) == before
        assert {path: path.read_bytes() for path in payloads} == payloads
        assert not (root / ".cacheness-clear-journal-v1.json").exists()
    finally:
        reopened.close()

    reopened_again = _unified_cache(root, backend_name)
    try:
        assert _backend_snapshot(reopened_again.metadata_backend, keys) == before
        assert {path: path.read_bytes() for path in payloads} == payloads
        assert not (root / ".cacheness-clear-journal-v1.json").exists()
    finally:
        reopened_again.close()


@pytest.mark.parametrize("backend_name", ("json", "sqlite", "memory"))
def test_unified_cache_successful_clear_returns_exact_backend_count(
    tmp_path, backend_name
):
    """Production-selected local backends clear through one exact return contract."""
    root = tmp_path / f"unified-success-{backend_name}"
    cache = _unified_cache(root, backend_name)

    try:
        keys, payloads = _put_unified_payloads(cache)

        assert cache.clear_all() == len(keys)
        assert all(cache.metadata_backend.get_entry(key) is None for key in keys)
        assert all(not path.exists() for path in payloads)
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        cache.close()


@pytest.mark.parametrize("backend_name", ("json", "sqlite"))
def test_unified_cache_committed_clear_recovers_after_two_persistent_reopens(
    tmp_path, monkeypatch, backend_name
):
    """Committed local clear evidence rolls forward once and then reopens idempotently."""
    root = tmp_path / f"unified-committed-{backend_name}"
    cache = _unified_cache(root, backend_name)
    keys, payloads = _put_unified_payloads(cache)
    clear_metadata = cache.metadata_backend.clear_all
    metadata_cleared = False
    delete = cache.guarded_handler_io.file_ops.delete

    def clear_then_mark_committed():
        nonlocal metadata_cleared
        count = clear_metadata()
        metadata_cleared = True
        return count

    def interrupt_committed_finalization(locator):
        if metadata_cleared:
            raise _SimulatedClearInterruption("committed final deletion interrupted")
        return delete(locator)

    monkeypatch.setattr(cache.metadata_backend, "clear_all", clear_then_mark_committed)
    monkeypatch.setattr(
        cache.guarded_handler_io.file_ops,
        "delete",
        interrupt_committed_finalization,
    )

    with pytest.raises(_SimulatedClearInterruption):
        cache.clear_all()
    assert (root / ".cacheness-clear-journal-v1.json").exists()
    cache.close()

    reopened = _unified_cache(root, backend_name)
    try:
        assert all(reopened.metadata_backend.get_entry(key) is None for key in keys)
        assert all(not path.exists() for path in payloads)
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        reopened.close()

    reopened_again = _unified_cache(root, backend_name)
    try:
        assert all(
            reopened_again.metadata_backend.get_entry(key) is None for key in keys
        )
        assert all(not path.exists() for path in payloads)
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        reopened_again.close()


class _UnifiedCustomMetadataBackend(InMemoryBackend):
    """An unsupported injection seam that must not acquire clear ownership."""


@pytest.mark.parametrize("topology", ("custom", "postgres"))
def test_unified_cache_rejects_unsupported_clear_topology_before_callbacks(
    tmp_path, topology
):
    """Custom and PostgreSQL clear requests fail before list, journal, or staging."""
    root = tmp_path / f"unified-unsupported-{topology}"
    cache = _unified_cache(root, "memory")
    original_backend = cache.metadata_backend
    original_actual_backend = cache.actual_backend
    callbacks: list[str] = []

    def forbidden_callback(*_args, **_kwargs):
        callbacks.append("callback")
        raise AssertionError("unsupported topology reached a clear callback")

    if topology == "custom":
        backend = _UnifiedCustomMetadataBackend()
        cache.actual_backend = "custom"
    else:
        backend = object.__new__(PostgresBackend)
        backend.engine = _NoopPostgresEngine()
        cache.actual_backend = "postgresql"
    backend.list_entries = forbidden_callback
    backend.load_metadata = forbidden_callback
    backend.clear_all = forbidden_callback
    cache.metadata_backend = backend

    try:
        with pytest.raises(CacheStorageError):
            cache.clear_all()

        assert callbacks == []
        assert not list(root.glob(".cacheness-clear-journal-*.json"))
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        cache.metadata_backend = original_backend
        cache.actual_backend = original_actual_backend
        cache.close()


def test_unified_cache_contender_fails_before_list_or_preflight_while_admitted(
    tmp_path, monkeypatch
):
    """A same-root contender cannot inspect entries while a clear owner holds admission."""
    cache = _unified_cache(tmp_path / "unified-admission", "memory")
    coordinator = cache._clear_recovery
    assert coordinator is not None
    callbacks: list[str] = []

    def forbidden_list_entries():
        callbacks.append("list")
        raise AssertionError("contender listed entries before admission")

    def forbidden_preflight(*_args, **_kwargs):
        callbacks.append("preflight")
        raise AssertionError("contender preflighted entries before admission")

    monkeypatch.setattr(cache.metadata_backend, "list_entries", forbidden_list_entries)
    monkeypatch.setattr(cache, "_preflight_entries", forbidden_preflight)
    try:
        with coordinator.admission():
            with pytest.raises(CacheStorageError):
                cache.clear_all()
        assert callbacks == []
    finally:
        cache.close()


def test_sqlite_prepared_recovery_uses_database_identity_not_wal_sidecars(
    tmp_path, monkeypatch
):
    """A matching SQLite store restores prepared work even when WAL sidecars exist."""
    root = tmp_path / "sqlite-root"
    store = BlobStore(root, backend="sqlite")
    keys, payloads = _put_payloads(store)
    store.backend.increment_hits()
    store.backend.increment_misses()
    before = _backend_snapshot(store.backend, keys)
    database_path = Path(store.backend.db_file)
    Path(f"{database_path}-wal").touch()
    _interrupted_clear(store.backend, monkeypatch)

    with pytest.raises(_SimulatedClearInterruption):
        store.clear()
    store.close()

    reopened = BlobStore(root, backend="sqlite")
    try:
        assert _backend_snapshot(reopened.backend, keys) == before
        assert {path: path.read_bytes() for path in payloads} == payloads
    finally:
        reopened.close()


def test_sqlite_committed_recovery_rolls_payloads_forward_after_reopen(
    tmp_path, monkeypatch
):
    """Committed SQLite clear evidence remains authoritative across a reopen."""
    root = tmp_path / "sqlite-root"
    store = BlobStore(root, backend="sqlite")
    keys, payloads = _put_payloads(store)
    clear_all = store.backend.clear_all
    metadata_cleared = False
    delete = store.guarded_handler_io.file_ops.delete

    def clear_then_mark_committed() -> int:
        nonlocal metadata_cleared
        result = clear_all()
        metadata_cleared = True
        return result

    def interrupt_final_delete(locator):
        if metadata_cleared:
            raise _SimulatedClearInterruption("interrupted after SQLite commit")
        return delete(locator)

    monkeypatch.setattr(store.backend, "clear_all", clear_then_mark_committed)
    monkeypatch.setattr(store.guarded_handler_io.file_ops, "delete", interrupt_final_delete)
    with pytest.raises(_SimulatedClearInterruption):
        store.clear()
    store.close()

    reopened = BlobStore(root, backend="sqlite")
    try:
        assert [reopened.get(key) for key in keys] == [None, None]
        assert all(not path.exists() for path in payloads)
    finally:
        reopened.close()


def test_memory_same_instance_prepared_recovery_restores_entries_and_counters(
    tmp_path, monkeypatch
):
    """A live in-memory identity can roll prepared work back exactly once."""
    root = tmp_path / "memory-root"
    backend = InMemoryBackend()
    store = BlobStore(root, backend=backend)
    keys, payloads = _put_payloads(store)
    backend.increment_hits()
    backend.increment_misses()
    before = _backend_snapshot(backend, keys)
    _interrupted_clear(backend, monkeypatch)

    with pytest.raises(_SimulatedClearInterruption):
        store.clear()
    store.close()

    reopened = BlobStore(root, backend=backend)
    try:
        assert _backend_snapshot(backend, keys) == before
        assert {path: path.read_bytes() for path in payloads} == payloads
    finally:
        reopened.close()


def test_memory_process_loss_rolls_prepared_journal_forward_without_snapshot_replay(
    tmp_path, monkeypatch
):
    """A new in-memory identity erases stale payloads instead of reviving lost metadata."""
    root = tmp_path / "memory-root"
    original_backend = InMemoryBackend()
    store = BlobStore(root, backend=original_backend)
    _, payloads = _put_payloads(store)
    _interrupted_clear(original_backend, monkeypatch)

    with pytest.raises(_SimulatedClearInterruption):
        store.clear()
    store.close()

    replacement_backend = InMemoryBackend()
    reopened = BlobStore(root, backend=replacement_backend)
    try:
        assert replacement_backend.list_entries() == []
        assert all(not path.exists() for path in payloads)
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        reopened.close()


def test_memory_process_loss_rolls_committed_journal_forward_without_snapshot_replay(
    tmp_path, monkeypatch
):
    """A new in-memory identity completes committed cleanup without snapshot replay."""
    root = tmp_path / "memory-root"
    original_backend = InMemoryBackend()
    store = BlobStore(root, backend=original_backend)
    _, payloads = _put_payloads(store)
    clear_all = original_backend.clear_all
    metadata_cleared = False
    delete = store.guarded_handler_io.file_ops.delete

    def clear_then_mark_committed() -> int:
        nonlocal metadata_cleared
        result = clear_all()
        metadata_cleared = True
        return result

    def interrupt_final_delete(locator):
        if metadata_cleared:
            raise _SimulatedClearInterruption("interrupted after memory commit")
        return delete(locator)

    monkeypatch.setattr(original_backend, "clear_all", clear_then_mark_committed)
    monkeypatch.setattr(
        store.guarded_handler_io.file_ops,
        "delete",
        interrupt_final_delete,
    )
    with pytest.raises(_SimulatedClearInterruption):
        store.clear()
    store.close()

    replacement_backend = InMemoryBackend()
    reopened = BlobStore(root, backend=replacement_backend)
    try:
        assert replacement_backend.list_entries() == []
        assert all(not path.exists() for path in payloads)
        assert not list(root.glob("clear-tombstone-*"))
    finally:
        reopened.close()


class _CustomMetadataBackend(InMemoryBackend):
    """Concrete unrecognized metadata topology for clear-admission tests."""


@pytest.mark.parametrize(
    ("name", "backend_factory"),
    (
        ("custom", lambda root: _CustomMetadataBackend()),
        ("sqlite_memory", lambda root: SqliteBackend(":memory:")),
        (
            "cached_wrapper",
            lambda root: CachedMetadataBackend(
                JsonBackend(root / "wrapped_metadata.json"),
                SimpleNamespace(enable_memory_cache=False),
            ),
        ),
    ),
)
def test_unsupported_manifest_topologies_fail_before_any_mutation(
    tmp_path, name, backend_factory
):
    """Unsupported backends fail before manifest or clear-recovery mutation."""
    root = tmp_path / name
    backend = backend_factory(root)
    store = None
    try:
        if name == "sqlite_memory":
            store = BlobStore(root, backend=backend)
            with pytest.raises(CacheBlobBackendError) as error:
                store.clear()
            assert {
                key: error.value.context[key]
                for key in ("operation", "backend", "supported_local_kinds")
            } == {
                "operation": "clear",
                "backend": "SqliteBackend",
                "supported_local_kinds": ["json", "sqlite", "memory"],
            }
            assert error.value.context["reason"] == "blob_backend_failure"
            assert not list(root.glob(".cacheness-clear-journal-*.json"))
            assert not list(root.glob("*candidate-*"))
            return

        with pytest.raises(CacheBlobBackendError) as error:
            BlobStore(root, backend=backend)

        assert error.value.context["operation"] == "create_manifest_repository"
        assert error.value.context["backend"] == type(backend).__name__
        assert error.value.context["reason"] == "blob_backend_failure"
        assert not list(root.glob(".cacheness-clear-journal-*.json"))
        assert not list(root.glob("*candidate-*"))
    finally:
        if store is not None:
            store.close()
        else:
            backend.close()


def test_postgres_backend_rejection_precedes_all_metadata_and_staging_callbacks(tmp_path):
    """A remote backend is refused before any callback or payload staging."""
    root = tmp_path / "postgres-root"
    backend = object.__new__(PostgresBackend)
    backend.engine = _NoopPostgresEngine()
    callbacks = []

    def forbidden_callback(*args, **kwargs):
        callbacks.append((args, kwargs))
        raise AssertionError("unsupported backend callback must not run")

    backend.list_entries = forbidden_callback
    backend.load_metadata = forbidden_callback
    backend.clear_all = forbidden_callback
    with pytest.raises(CacheBlobBackendError) as error:
        BlobStore(root, backend=backend)

    assert error.value.context["operation"] == "create_manifest_repository"
    assert error.value.context["backend"] == "PostgresBackend"
    assert error.value.context["reason"] == "blob_backend_failure"
    assert callbacks == []
    assert not list(root.glob(".cacheness-clear-journal-*.json"))
    assert not list(root.glob("clear-tombstone-*"))


def test_clear_journal_constants_match_the_declared_adversarial_bounds():
    """The recovery parser has fixed public audit bounds rather than magic values."""
    assert clear_recovery.MAX_JOURNAL_BYTES == 64 * 1024 * 1024
    assert clear_recovery.MAX_JOURNAL_ENTRIES == 100_000
    assert clear_recovery.MAX_JOURNAL_FIELD_BYTES == 8192


@pytest.mark.parametrize("offset", (-1, 0, 1))
def test_journal_total_encoded_byte_bound_precedes_json_deserialization(
    tmp_path, monkeypatch, offset
):
    """A journal one byte too large is rejected before JSON parsing or callbacks."""
    root = tmp_path / "journal-byte-bound"
    store = BlobStore(root, backend="json")
    try:
        _, _ = _put_payloads(store)
        journal = _journal_for(store, ["first"])
        coordinator = store._clear_recovery
        assert coordinator is not None
        encoded = coordinator._encode_journal(journal)
        monkeypatch.setattr(
            clear_recovery,
            "MAX_JOURNAL_BYTES",
            len(encoded) + offset,
            raising=False,
        )
        coordinator.journal_path.write_bytes(encoded)
        loads_calls = 0
        json_loads = clear_recovery.json_loads

        def counted_json_loads(document):
            nonlocal loads_calls
            loads_calls += 1
            return json_loads(document)

        monkeypatch.setattr(clear_recovery, "json_loads", counted_json_loads)
        if offset < 0:
            with pytest.raises(CacheStorageError):
                coordinator._read_journal()
            assert loads_calls == 0
        else:
            assert coordinator._read_journal() == journal
            assert loads_calls == 1
    finally:
        store.close()


@pytest.mark.parametrize("field", ("cache_key", "original", "tombstone"))
@pytest.mark.parametrize("offset", (-1, 0, 1))
def test_journal_field_bounds_are_checked_before_recovery_callbacks(
    tmp_path, field, offset
):
    """Raw bounds are checked before the stricter payload locator grammar."""
    root = tmp_path / "journal-bounds"
    store = BlobStore(root, backend="json")
    try:
        _, _ = _put_payloads(store)
        journal = _journal_for(store, ["first"])
        limit = clear_recovery.MAX_JOURNAL_FIELD_BYTES
        if field == "cache_key":
            value = "k" * (limit + offset)
        else:
            segment = "a/"
            value = (segment * ((limit + offset) // len(segment) + 1))[
                : limit + offset
            ]
        journal["mappings"][0][field] = value
        coordinator = store._clear_recovery
        assert coordinator is not None
        assert coordinator._within_field_bound(value) is (offset <= 0)
        if field == "cache_key":
            original = journal["mappings"][0]["original"]
            original_id = encode_physical_name("first", namespace="blob-store")
            suffix = original[len(original_id) :]
            journal["mappings"][0]["original"] = (
                encode_physical_name(value, namespace="blob-store") + suffix
            )
            entry = journal["metadata_snapshot"]["entries"].pop("first")
            journal["metadata_snapshot"]["entries"][value] = entry
            entry["metadata"]["actual_path"] = str(
                root / journal["mappings"][0]["original"]
            )

        if offset > 0 or field in {"original", "tombstone"}:
            with pytest.raises(CacheStorageError):
                coordinator._validate_journal(journal)
        else:
            coordinator._validate_journal(journal)
    finally:
        store.close()


@pytest.mark.parametrize("entry_count", (1, 2, 3))
def test_journal_entry_count_bounds_are_checked_before_recovery_callbacks(
    tmp_path, monkeypatch, entry_count
):
    """The configured entry limit is honored immediately below, at, and above it."""
    root = tmp_path / "journal-entry-count"
    store = BlobStore(root, backend="json")
    try:
        _, _ = _put_payloads(store)
        journal = _journal_for(store, ["first"])
        mapping = deepcopy(journal["mappings"][0])
        original_id = encode_physical_name("first", namespace="blob-store")
        suffix = mapping["original"][len(original_id) :]
        journal["mappings"] = []
        snapshot_entries = {}
        for index in range(entry_count):
            next_mapping = deepcopy(mapping)
            cache_key = f"key-{index}"
            next_mapping["cache_key"] = cache_key
            next_mapping["original"] = (
                encode_physical_name(cache_key, namespace="blob-store") + suffix
            )
            next_mapping["tombstone"] = (
                f"clear-tombstone-{journal['operation_id']}-{index}"
            )
            journal["mappings"].append(next_mapping)
        snapshot_entry = journal["metadata_snapshot"]["entries"]["first"]
        for mapping in journal["mappings"]:
            entry = deepcopy(snapshot_entry)
            entry["metadata"]["actual_path"] = str(root / mapping["original"])
            snapshot_entries[mapping["cache_key"]] = entry
        journal["metadata_snapshot"]["entries"] = snapshot_entries
        monkeypatch.setattr(clear_recovery, "MAX_JOURNAL_ENTRIES", 2, raising=False)

        coordinator = store._clear_recovery
        assert coordinator is not None
        if entry_count > 2:
            with pytest.raises(CacheStorageError):
                coordinator._validate_journal(journal)
        else:
            coordinator._validate_journal(journal)
    finally:
        store.close()


@pytest.mark.parametrize(
    "defect",
    (
        "extra_field",
        "missing_field",
        "unknown_version",
        "unknown_state",
        "unknown_owner",
        "topology_mismatch",
        "duplicate_cache_key",
        "duplicate_original",
        "duplicate_tombstone",
        "snapshot_cardinality_mismatch",
        "path_escape",
        "metadata_file_original",
        "admission_lock_original",
        "unrelated_payload_original",
        "long_handler_suffix",
        "malformed_candidate_marker",
        "malformed_candidate_uuid",
        "candidate_long_handler_suffix",
        "forged_tombstone",
        "snapshot_actual_path_mismatch",
        "snapshot_extra_field",
        "snapshot_non_dict_metadata",
        "negative_file_size",
        "negative_counter",
        "snapshot_bad_timestamp",
    ),
)
def test_hostile_journal_defects_fail_closed_without_recovery_mutation(
    tmp_path, defect
):
    """Malformed or hostile persisted evidence never reaches metadata or payload callbacks."""
    root = tmp_path / "hostile-journal"
    store = BlobStore(root, backend="json")
    keys, payloads = _put_payloads(store)
    metadata_before = deepcopy(store.backend.load_metadata())
    journal = _journal_for(store, keys)
    unrelated_payload = None
    other_payload = list(payloads)[1]

    if defect == "extra_field":
        journal["unexpected"] = True
    elif defect == "missing_field":
        del journal["metadata_snapshot"]
    elif defect == "unknown_version":
        journal["version"] = 2
    elif defect == "unknown_state":
        journal["state"] = "unknown"
    elif defect == "unknown_owner":
        journal["owner"] = "attacker"
    elif defect == "topology_mismatch":
        journal["topology"]["root_inode"] += 1
    elif defect == "duplicate_cache_key":
        journal["mappings"].append(deepcopy(journal["mappings"][0]))
    elif defect == "duplicate_original":
        duplicate = deepcopy(journal["mappings"][1])
        duplicate["original"] = journal["mappings"][0]["original"]
        journal["mappings"][1] = duplicate
    elif defect == "duplicate_tombstone":
        duplicate = deepcopy(journal["mappings"][1])
        duplicate["tombstone"] = journal["mappings"][0]["tombstone"]
        journal["mappings"][1] = duplicate
    elif defect == "snapshot_cardinality_mismatch":
        journal["metadata_snapshot"]["entries"].pop(keys[0])
    elif defect == "path_escape":
        journal["mappings"][0]["original"] = "../outside"
    elif defect == "metadata_file_original":
        journal["mappings"][0]["original"] = "cache_metadata.json"
        journal["metadata_snapshot"]["entries"][keys[0]]["metadata"][
            "actual_path"
        ] = str(root / "cache_metadata.json")
    elif defect == "admission_lock_original":
        journal["mappings"][0]["original"] = ".cacheness-clear-admission.lock"
        journal["metadata_snapshot"]["entries"][keys[0]]["metadata"][
            "actual_path"
        ] = str(root / ".cacheness-clear-admission.lock")
    elif defect == "unrelated_payload_original":
        unrelated_payload = root / "unrelated-payload.pkl"
        unrelated_payload.write_bytes(b"unrelated-payload")
        journal["mappings"][0]["original"] = unrelated_payload.name
        journal["metadata_snapshot"]["entries"][keys[0]]["metadata"][
            "actual_path"
        ] = str(unrelated_payload)
    elif defect == "long_handler_suffix":
        physical_name = encode_physical_name(keys[0], namespace="blob-store")
        forged_original = f"{physical_name}.{('a' * 97)}"
        journal["mappings"][0]["original"] = forged_original
        journal["metadata_snapshot"]["entries"][keys[0]]["metadata"][
            "actual_path"
        ] = str(root / forged_original)
    elif defect == "malformed_candidate_marker":
        physical_name = encode_physical_name(keys[0], namespace="blob-store")
        forged_original = f"{physical_name}-candidate-short.pkl"
        journal["mappings"][0]["original"] = forged_original
        journal["metadata_snapshot"]["entries"][keys[0]]["metadata"][
            "actual_path"
        ] = str(root / forged_original)
    elif defect == "malformed_candidate_uuid":
        physical_name = encode_physical_name(keys[0], namespace="blob-store")
        forged_original = f"{physical_name}-candidate-{'A' * 32}.pkl"
        journal["mappings"][0]["original"] = forged_original
        journal["metadata_snapshot"]["entries"][keys[0]]["metadata"][
            "actual_path"
        ] = str(root / forged_original)
    elif defect == "candidate_long_handler_suffix":
        physical_name = encode_physical_name(keys[0], namespace="blob-store")
        forged_original = f"{physical_name}-candidate-{'a' * 32}.{('b' * 97)}"
        journal["mappings"][0]["original"] = forged_original
        journal["metadata_snapshot"]["entries"][keys[0]]["metadata"][
            "actual_path"
        ] = str(root / forged_original)
    elif defect == "forged_tombstone":
        journal["mappings"][0]["tombstone"] = "cache_metadata.json"
    elif defect == "snapshot_actual_path_mismatch":
        journal["metadata_snapshot"]["entries"][keys[0]]["metadata"][
            "actual_path"
        ] = str(other_payload)
    elif defect == "snapshot_extra_field":
        journal["metadata_snapshot"]["entries"][keys[0]]["unexpected"] = True
    elif defect == "snapshot_non_dict_metadata":
        journal["metadata_snapshot"]["entries"][keys[0]]["metadata"] = []
    elif defect == "negative_file_size":
        journal["metadata_snapshot"]["entries"][keys[0]]["file_size"] = -1
    elif defect == "snapshot_bad_timestamp":
        journal["metadata_snapshot"]["entries"][keys[0]]["created_at"] = 0
    else:
        journal["metadata_snapshot"]["cache_hits"] = -1

    _write_untrusted_journal(store, journal)
    store.close()

    with pytest.raises(CacheStorageError):
        BlobStore(root, backend="json")

    assert JsonBackend(root / "cache_metadata.json").load_metadata() == metadata_before
    assert {path: path.read_bytes() for path in payloads} == payloads
    if unrelated_payload is not None:
        assert unrelated_payload.read_bytes() == b"unrelated-payload"


def test_advisory_lock_unavailability_fails_before_clear_mutation(tmp_path, monkeypatch):
    """A missing reliable OS advisory lock is a typed refusal, never a downgrade."""
    root = tmp_path / "advisory-lock"
    store = BlobStore(root, backend="json")
    try:
        keys, payloads = _put_payloads(store)
        before = _backend_snapshot(store.backend, keys)
        monkeypatch.setattr(
            clear_recovery,
            "_advisory_lock_available",
            lambda root_path: False,
            raising=False,
        )

        with pytest.raises(CacheStorageError):
            store.clear()

        assert _backend_snapshot(store.backend, keys) == before
        assert {path: path.read_bytes() for path in payloads} == payloads
        assert not list(root.glob(".cacheness-clear-journal-*.json"))
    finally:
        store.close()


def test_exclusive_journal_creation_preserves_existing_evidence(tmp_path):
    """A second clear cannot overwrite a fixed journal left by an earlier owner."""
    root = tmp_path / "exclusive-journal"
    store = BlobStore(root, backend="json")
    try:
        keys, payloads = _put_payloads(store)
        before = _backend_snapshot(store.backend, keys)
        coordinator = store._clear_recovery
        assert coordinator is not None
        evidence = b"unresolved-prior-clear-evidence"
        coordinator.journal_path.write_bytes(evidence)

        with pytest.raises(CacheStorageError):
            store.clear()

        assert coordinator.journal_path.read_bytes() == evidence
        assert _backend_snapshot(store.backend, keys) == before
        assert {path: path.read_bytes() for path in payloads} == payloads
    finally:
        store.close()


def test_same_root_thread_contender_fails_while_clear_owner_holds_admission(
    tmp_path, monkeypatch
):
    """Only one same-root caller may recover or clear while prepared work is active."""
    root = tmp_path / "thread-admission"
    owner = BlobStore(root, backend="json")
    _put_payloads(owner)
    coordinator = owner._clear_recovery
    assert coordinator is not None
    entered = threading.Event()
    release = threading.Event()
    owner_errors = []
    stage_mapping = coordinator._stage_mapping

    def hold_owner_admission(mapping):
        entered.set()
        assert release.wait(timeout=5)
        return stage_mapping(mapping)

    monkeypatch.setattr(coordinator, "_stage_mapping", hold_owner_admission)

    def run_owner_clear() -> None:
        try:
            owner.clear()
        except Exception as exc:
            owner_errors.append(exc)

    worker = threading.Thread(target=run_owner_clear)
    worker.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(CacheStorageError):
            BlobStore(root, backend="json")
    finally:
        release.set()
        worker.join(timeout=5)
        owner.close()
    assert not owner_errors


@pytest.mark.skipif(sys.platform.startswith("win"), reason="requires POSIX advisory locks")
@pytest.mark.parametrize("backend_name", ("json", "sqlite"))
def test_same_root_subprocess_contender_fails_while_clear_owner_holds_admission(
    tmp_path, monkeypatch, backend_name
):
    """The advisory lock rejects a separate process before it can recover or mutate."""
    root = tmp_path / f"subprocess-admission-{backend_name}"
    owner = BlobStore(root, backend=backend_name)
    _put_payloads(owner)
    coordinator = owner._clear_recovery
    assert coordinator is not None
    entered = threading.Event()
    release = threading.Event()
    stage_mapping = coordinator._stage_mapping

    def hold_owner_admission(mapping):
        entered.set()
        assert release.wait(timeout=5)
        return stage_mapping(mapping)

    monkeypatch.setattr(coordinator, "_stage_mapping", hold_owner_admission)
    worker = threading.Thread(target=owner.clear)
    worker.start()
    assert entered.wait(timeout=5)
    script = "\n".join(
        (
            "import sys",
            "from cacheness.error_handling import CacheStorageError",
            "from cacheness.storage.blob_store import BlobStore",
            "try:",
            "    store = BlobStore(sys.argv[1], backend=sys.argv[2])",
            "    store.put('subprocess contender', key='subprocess-contender')",
            "except CacheStorageError:",
            "    raise SystemExit(0)",
            "else:",
            "    store.close()",
            "    raise SystemExit(1)",
        )
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(root), backend_name],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
    finally:
        release.set()
        worker.join(timeout=5)
        owner.close()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="requires POSIX advisory locks")
@pytest.mark.parametrize("backend_name", ("json", "sqlite"))
def test_unified_cache_subprocess_put_is_rejected_while_clear_holds_admission(
    tmp_path, monkeypatch, backend_name
):
    """A separate UnifiedCache process cannot publish during a local clear."""
    root = tmp_path / f"unified-subprocess-admission-{backend_name}"
    owner = _unified_cache(root, backend_name)
    _put_unified_payloads(owner)
    coordinator = owner._clear_recovery
    assert coordinator is not None
    entered = threading.Event()
    release = threading.Event()
    stage_mapping = coordinator._stage_mapping

    def hold_owner_admission(mapping):
        entered.set()
        assert release.wait(timeout=5)
        return stage_mapping(mapping)

    monkeypatch.setattr(coordinator, "_stage_mapping", hold_owner_admission)
    worker = threading.Thread(target=owner.clear_all)
    worker.start()
    assert entered.wait(timeout=5)
    script = "\n".join(
        (
            "import sys",
            "from cacheness import CacheConfig, cacheness",
            "from cacheness.error_handling import CacheStorageError",
            "try:",
            "    cache = cacheness(CacheConfig(cache_dir=sys.argv[1], metadata_backend=sys.argv[2], cleanup_on_init=False))",
            "    cache.put({'value': 'subprocess contender'}, put_admission='subprocess')",
            "except CacheStorageError:",
            "    raise SystemExit(0)",
            "else:",
            "    cache.close()",
            "    raise SystemExit(1)",
        )
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(root), backend_name],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
    finally:
        release.set()
        worker.join(timeout=5)
        owner.close()
