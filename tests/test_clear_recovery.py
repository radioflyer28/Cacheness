"""Durability and reopen contracts for the bounded BlobStore clear primitive."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import threading

import pytest

import cacheness.metadata as metadata_module
import cacheness.storage.clear_recovery as clear_recovery
from cacheness.error_handling import CacheStorageError
from cacheness.metadata import (
    CachedMetadataBackend,
    InMemoryBackend,
    JsonBackend,
    SqliteBackend,
)
from cacheness.storage.backends.postgresql_backend import PostgresBackend
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
    return coordinator._new_prepared_journal(mappings)


def _write_untrusted_journal(store: BlobStore, journal: dict[str, object]) -> None:
    """Bypass managed writes to model pre-existing storage-controlled journal bytes."""
    coordinator = store._clear_recovery
    assert coordinator is not None
    coordinator.journal_path.write_text(json.dumps(journal), encoding="utf-8")


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
def test_unsupported_clear_topologies_fail_before_any_mutation(
    tmp_path, name, backend_factory
):
    """Remote, wrapped, custom, and SQLite-memory topologies have no local journal proof."""
    root = tmp_path / name
    store = BlobStore(root, backend=backend_factory(root))
    try:
        keys, payloads = _put_payloads(store)
        before = _backend_snapshot(store.backend, keys)

        with pytest.raises(CacheStorageError) as error:
            store.clear()

        assert error.value.context == {
            "operation": "clear",
            "backend": type(store.backend).__name__,
            "supported_local_kinds": ["json", "sqlite", "memory"],
        }
        assert _backend_snapshot(store.backend, keys) == before
        assert {path: path.read_bytes() for path in payloads} == payloads
        assert not list(root.glob(".cacheness-clear-journal-*.json"))
    finally:
        store.close()


def test_postgres_backend_rejection_precedes_all_metadata_and_staging_callbacks(tmp_path):
    """A remote backend is refused without connecting or calling any callback."""
    root = tmp_path / "postgres-root"
    backend = object.__new__(PostgresBackend)
    callbacks = []

    def forbidden_callback(*args, **kwargs):
        callbacks.append((args, kwargs))
        raise AssertionError("unsupported backend callback must not run")

    backend.list_entries = forbidden_callback
    backend.load_metadata = forbidden_callback
    backend.clear_all = forbidden_callback
    store = BlobStore(root, backend=backend)

    with pytest.raises(CacheStorageError) as error:
        store.clear()

    assert error.value.context == {
        "operation": "clear",
        "backend": "PostgresBackend",
        "supported_local_kinds": ["json", "sqlite", "memory"],
    }
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
    """Every string field accepts its boundary and rejects the next UTF-8 byte."""
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
        if offset > 0:
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
        journal["mappings"] = []
        for index in range(entry_count):
            next_mapping = deepcopy(mapping)
            next_mapping["cache_key"] = f"key-{index}"
            next_mapping["original"] = f"payload-{index}"
            next_mapping["tombstone"] = f"tombstone-{index}"
            journal["mappings"].append(next_mapping)
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
    else:
        journal["mappings"][0]["original"] = "../outside"

    _write_untrusted_journal(store, journal)
    store.close()

    with pytest.raises(CacheStorageError):
        BlobStore(root, backend="json")

    assert JsonBackend(root / "cache_metadata.json").load_metadata() == metadata_before
    assert {path: path.read_bytes() for path in payloads} == payloads


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
def test_same_root_subprocess_contender_fails_while_clear_owner_holds_admission(
    tmp_path, monkeypatch
):
    """The advisory lock rejects a separate process before it can recover or mutate."""
    root = tmp_path / "subprocess-admission"
    owner = BlobStore(root, backend="json")
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
            "    store = BlobStore(sys.argv[1], backend='json')",
            "except CacheStorageError:",
            "    raise SystemExit(0)",
            "else:",
            "    store.close()",
            "    raise SystemExit(1)",
        )
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(root)],
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
