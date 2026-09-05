"""Exact-record conditional publication contracts for local manifests."""

from __future__ import annotations

import builtins
import ctypes
import errno
import hashlib
import json
import multiprocessing
import os
from copy import deepcopy
from pathlib import Path
from threading import Barrier, RLock, Thread

import pytest

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobMigrationRequiredError,
    CacheManifestIntegrityError,
    CacheReason,
    CacheUnsafePathError,
)
from cacheness.metadata import InMemoryBackend, JsonBackend, SqliteBackend
from cacheness.storage.manifest_repository import (
    InMemoryManifestRepository,
    JsonProjectionExporter,
    JsonManifestRepository,
    ManifestExpectation,
    SqliteManifestRepository,
)
from cacheness.storage import coordination
from cacheness.storage import path_security
from cacheness.storage import manifest_repository as manifest_repository_module
from cacheness.storage.path_security import ManagedFileOps, resolve_managed_locator
from cacheness.storage.operation_repository import (
    FileOperationRecordRepository,
    OperationCursor,
    PendingControlCursor,
    ReconciliationCheckpointCursor,
)


def _record(label: str) -> bytes:
    """Return deliberately opaque canonical-record stand-ins for repository tests."""
    return f"canonical-manifest-record::{label}".encode("utf-8")


# =============================================================================
# Authority JSON projection (Plan 03-06)
# =============================================================================


def test_projection_export_streams_a_private_authority_backup_and_marks_revision_clean(
    tmp_path: Path,
) -> None:
    """Committed SQLite rows rebuild the public JSON shape without live reads."""
    from cacheness.storage import BlobStore

    root = tmp_path / "projection-store"
    store = BlobStore(root, backend="json")
    try:
        key = store.put("projection payload", key="projection-key", metadata={"tag": "v1"})
        authority = store.lifecycle_authority
        dirty_revision = authority.snapshot_state().revision
        projection_path = root / "cache_metadata.json"

        result = JsonProjectionExporter(authority, projection_path).export()

        assert result.value == dirty_revision
        assert authority.snapshot_state().projection_dirty is False
        document = json.loads(projection_path.read_text(encoding="utf-8"))
        assert document["entries"][key] == {
            "cache_key": key,
            "data_type": "object",
            "file_size": document["entries"][key]["file_size"],
            "created_at": document["entries"][key]["created_at"],
            "metadata": {
                "tag": "v1",
                "actual_path": document["entries"][key]["metadata"]["actual_path"],
            },
        }
        assert document["entries"][key]["metadata"]["actual_path"].startswith(
            "generations/"
        )
    finally:
        store.close()


def test_projection_export_failure_keeps_committed_authority_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A derived-output failure never rolls back or marks committed state clean."""
    from cacheness.storage import BlobStore
    from cacheness.error_handling import CacheBlobBackendError

    root = tmp_path / "projection-failure"
    store = BlobStore(root, backend="json")
    try:
        key = store.put("projection payload", key="projection-key")
        authority = store.lifecycle_authority
        original_replace = manifest_repository_module.os.replace

        def fail_projection_publish(source: object, destination: object) -> None:
            if Path(destination) == root / "cache_metadata.json":
                raise OSError("projection publish failed")
            original_replace(source, destination)

        monkeypatch.setattr(
            manifest_repository_module.os, "replace", fail_projection_publish
        )

        with pytest.raises(CacheBlobBackendError, match="projection"):
            JsonProjectionExporter(authority, root / "cache_metadata.json").export()

        assert authority.snapshot_state().projection_dirty is True
        assert store.get(key) == "projection payload"
    finally:
        store.close()


def test_pending_recovery_filters_unrelated_names_before_its_action_bound(
    tmp_path: Path,
) -> None:
    """Malformed and ordinary siblings cannot starve one valid pending record."""
    root = tmp_path / "pending-recovery-filter"
    root.mkdir()
    limits = LifecycleLimits(max_reconcile_actions=1)
    file_ops = ManagedFileOps(root)
    repository = FileOperationRecordRepository(
        file_ops,
        lifecycle_limits=limits,
        initialization_key_provider=lambda: b"k" * 32,
    )
    repository.initialize_new_store()
    operation_id = "f" * 32
    raw = b'{"pending":"exact"}'
    digest = hashlib.sha256(raw).hexdigest()
    operations = root / "operations"
    operations.mkdir(exist_ok=True)
    # These sort before the valid pending name but are not eligible controls.
    (operations / ".000-malformed.tmp").write_bytes(b"noise")
    (operations / ".111.json.pending.not-a-digest.tmp").write_bytes(b"noise")
    (operations / "clear-target-page-not-an-operation.json").write_bytes(b"noise")
    pending = operations / f".{operation_id}.json.pending.{digest}.{'a' * 32}.tmp"
    repository._append_inventory_event("pending", pending.name, raw)
    pending.write_bytes(raw)
    try:
        assert repository.recover_pending_operation_records() == (operation_id,)
        assert repository.get_raw(operation_id) == raw
        # Repeated bounded reopen/recovery calls do not get stuck on siblings.
        assert repository.recover_pending_operation_records() == ()
    finally:
        repository.close()
        file_ops.close()


def test_markerless_sibling_head_cannot_hide_raw_absent_family_evidence(
    tmp_path: Path,
) -> None:
    """A lazy v2 head is migration evidence, never absent-family provenance."""
    root = tmp_path / "mixed-lazy-v2"
    root.mkdir()
    key = b"k" * 32
    file_ops = ManagedFileOps(root)
    repository = FileOperationRecordRepository(
        file_ops,
        lifecycle_limits=LifecycleLimits(max_inventory_items=1),
        initialization_key_provider=lambda: key,
    )
    try:
        repository.initialize_new_store()
        # Model the old lazy-v2 crash shape: one valid sibling head survives,
        # v3 all-family provenance and the target-family head do not, and raw
        # evidence is present in that target family.
        file_ops.delete_durable(repository._inventory_initialization_locator())
        file_ops.delete_durable(repository._inventory_head_locator("sidecar"))
        file_ops.write_bytes_durable(
            repository.reconciliation_checkpoint_locator("a" * 32), b'{"raw":"legacy"}'
        )
    finally:
        repository.close()
        file_ops.close()

    reopened_ops = ManagedFileOps(root)
    reopened = FileOperationRecordRepository(
        reopened_ops,
        lifecycle_limits=LifecycleLimits(max_inventory_items=1),
        initialization_key_provider=lambda: key,
    )
    try:
        with pytest.raises(CacheBlobMigrationRequiredError):
            reopened.list_reconciliation_checkpoint_page()
    finally:
        reopened.close()
        reopened_ops.close()


def test_store_bound_inventory_provenance_rejects_cross_root_replay(
    tmp_path: Path,
) -> None:
    """A shared manifest key cannot authenticate another store's empty families."""
    key = b"k" * 32
    source_root = tmp_path / "inventory-source"
    target_root = tmp_path / "inventory-target"
    source_root.mkdir()
    target_root.mkdir()
    source_ops = ManagedFileOps(source_root)
    source = FileOperationRecordRepository(
        source_ops,
        lifecycle_limits=LifecycleLimits(max_inventory_items=1),
        initialization_key_provider=lambda: key,
    )
    try:
        source.initialize_new_store()
        marker = source_ops.read_bytes(source._inventory_initialization_locator())
    finally:
        source.close()
        source_ops.close()

    target_ops = ManagedFileOps(target_root)
    target = FileOperationRecordRepository(
        target_ops,
        lifecycle_limits=LifecycleLimits(max_inventory_items=1),
        initialization_key_provider=lambda: key,
    )
    try:
        target_ops.write_bytes_durable(target._inventory_initialization_locator(), marker)
        target_ops.write_bytes_durable(
            target.reconciliation_checkpoint_locator("a" * 32), b'{"legacy":true}'
        )
        with pytest.raises(CacheBlobBackendError, match="initialization marker is invalid"):
            target.list_reconciliation_checkpoint_page()
    finally:
        target.close()
        target_ops.close()


@pytest.mark.parametrize("family", ("primary", "sidecar", "pending"))
def test_operation_inventory_scheduling_substitution_fails_closed(
    tmp_path: Path, family: str
) -> None:
    """A valid-shaped unsigned scheduler record cannot suppress evidence."""
    root = tmp_path / f"signed-scheduler-{family}"
    root.mkdir()
    file_ops = ManagedFileOps(root)
    repository = FileOperationRecordRepository(
        file_ops,
        lifecycle_limits=LifecycleLimits(max_inventory_items=1),
        initialization_key_provider=lambda: b"k" * 32,
    )
    try:
        repository.initialize_new_store()
        raw = b'{"control":"current"}'
        operation_id = "a" * 32
        if family == "primary":
            name = operation_id
            locator = repository.locator_for(operation_id)
            read_page = repository.list_page
        elif family == "sidecar":
            name = f"reconcile-action-{operation_id}.json"
            locator = repository.reconciliation_checkpoint_locator(operation_id)
            read_page = repository.list_reconciliation_checkpoint_page
        else:
            digest = hashlib.sha256(raw).hexdigest()
            name = f".{operation_id}.json.pending.{digest}.{'b' * 32}.tmp"
            locator = root / "operations" / name

            def read_page():
                return repository.list_pending_control_page(None)

        repository._append_inventory_event(family, name, raw)
        file_ops.write_bytes_durable(locator, raw)
        # Receipt v1 makes the immutable sequence-specific head authoritative;
        # the former fixed head is only a maintenance checkpoint and may lag a
        # concurrent writer.  An invalid receipt must still fail closed.
        file_ops.write_bytes_durable(
            repository._inventory_head_receipt_locator(family, 1), b"{}"
        )
        with pytest.raises((CacheBlobBackendError, CacheManifestIntegrityError)):
            read_page()
    finally:
        repository.close()
        file_ops.close()


@pytest.mark.parametrize("backend_name", ("memory", "json"))
def test_manifest_scheduler_skip_proof_rejects_tampering_after_reopen(
    tmp_path: Path, backend_name: str
) -> None:
    """An old cursor cannot accept a substituted signed-run proof."""
    key = b"m" * 32
    limits = LifecycleLimits(manifest_page_size=1, max_inventory_items=1)
    if backend_name == "memory":
        backend = InMemoryBackend()
        repository = InMemoryManifestRepository(
            backend, lifecycle_limits=limits, inventory_key_provider=lambda: key
        )
    else:
        backend = JsonBackend(tmp_path / "manifest-scheduler.json")
        repository = JsonManifestRepository(
            backend, lifecycle_limits=limits, inventory_key_provider=lambda: key
        )
    try:
        repository.put_raw("anchor", _record("anchor"))
        repository.put_raw("churn", _record("churn-0"))
        old_page = repository.list_page()
        assert old_page.next_cursor is not None

        # Two replacements advance bounded maintenance past the stable anchor
        # and emit an authenticated sparse proof for the stale churn member.
        for generation in (1, 2):
            observed = repository.get_raw("churn")
            assert observed is not None
            repository.publish_if_expected(
                "churn",
                ManifestExpectation.from_authenticated_record(
                    f"generation-{generation}", observed
                ),
                _record(f"churn-{generation}"),
            )

        if backend_name == "json":
            repository.close()
            backend.close()
            backend = JsonBackend(tmp_path / "manifest-scheduler.json")
            repository = JsonManifestRepository(
                backend, lifecycle_limits=limits, inventory_key_provider=lambda: key
            )
        else:
            repository = InMemoryManifestRepository(
                backend, lifecycle_limits=limits, inventory_key_provider=lambda: key
            )

        # A cursor captured before the stale run remains a safe terminal view
        # after the reopened scheduler consumes that run.
        assert repository.list_page(old_page.next_cursor).entries == ()
        if backend_name == "json":
            marker_path = repository._json_inventory_skip_locator(2)  # type: ignore[union-attr]
            marker = json.loads(marker_path.read_bytes())
            marker["run_digest"] = "0" * 64
            marker_path.write_text(json.dumps(marker, sort_keys=True, separators=(",", ":")))
        else:
            repository._inventory_state()["skips"][2]["run_digest"] = "0" * 64

        with pytest.raises(CacheBlobBackendError, match="list_page failed"):
            repository.list_page(old_page.next_cursor)
    finally:
        repository.close()
        backend.close()


def test_manifest_scheduler_replay_from_another_root_fails_closed(tmp_path: Path) -> None:
    """A valid signed manifest head is scoped to one managed store identity."""
    key = b"m" * 32
    source_path = tmp_path / "manifest-source" / "metadata.json"
    source_path.parent.mkdir()
    source_backend = JsonBackend(source_path)
    source = JsonManifestRepository(
        source_backend, inventory_key_provider=lambda: key
    )
    try:
        source.put_raw("source", _record("source"))
        head = source._json_inventory_head_locator.read_bytes()  # type: ignore[union-attr]
    finally:
        source.close()
        source_backend.close()

    target_path = tmp_path / "manifest-target" / "metadata.json"
    target_path.parent.mkdir()
    target_backend = JsonBackend(target_path)
    target = JsonManifestRepository(
        target_backend, inventory_key_provider=lambda: key
    )
    try:
        target._json_inventory_head_locator.write_bytes(head)  # type: ignore[union-attr]
        with pytest.raises(CacheBlobBackendError, match="list_page failed"):
            target.list_page()
    finally:
        target.close()
        target_backend.close()


@pytest.mark.parametrize("family", ("primary", "sidecar", "pending"))
def test_operation_inventory_recovery_compacts_pinned_history_in_bounded_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    """A long-lived record does not strand later live work behind stale slots."""
    root = tmp_path / f"operation-maintenance-{family}"
    root.mkdir()
    limits = LifecycleLimits(operation_page_size=1, max_inventory_items=1)
    key = b"k" * 32
    file_ops = ManagedFileOps(root)
    repository = FileOperationRecordRepository(
        file_ops,
        lifecycle_limits=limits,
        initialization_key_provider=lambda: key,
    )
    pinned_id = "a" * 32
    later_id = "b" * 32
    pinned_raws = tuple(_record(f"{family}-pinned-{index}") for index in range(9))
    later_raw = _record(f"{family}-later")

    def pending_name(operation_id: str, raw: bytes) -> str:
        return (
            f".{operation_id}.json.pending.{hashlib.sha256(raw).hexdigest()}."
            f"{'f' * 32}.tmp"
        )

    if family == "primary":
        pinned_name, later_name = pinned_id, later_id
        locate = repository.locator_for
        list_page = repository.list_page
        old_cursor = OperationCursor(
            pinned_id, snapshot_high_water=10, next_sequence=9
        )
    elif family == "sidecar":
        pinned_name, later_name = pinned_id, later_id
        locate = repository.reconciliation_checkpoint_locator
        list_page = repository.list_reconciliation_checkpoint_page
        old_cursor = ReconciliationCheckpointCursor(
            pinned_id, snapshot_high_water=10, next_sequence=9
        )
    else:
        pinned_name = pending_name(pinned_id, pinned_raws[-1])
        later_name = pending_name(later_id, later_raw)

        def locate(name: str) -> Path:
            return root / "operations" / name

        list_page = repository.list_pending_control_page
        old_cursor = PendingControlCursor(
            pinned_name, snapshot_high_water=10, next_sequence=9
        )

    try:
        repository.initialize_new_store()
        for raw in pinned_raws:
            repository._append_inventory_event(family, pinned_name, raw)
            file_ops.write_bytes_durable(locate(pinned_name), raw)
        repository._append_inventory_event(family, later_name, later_raw)
        file_ops.write_bytes_durable(locate(later_name), later_raw)

        # A terminal boundary schedules all charged slots but may only inspect
        # one.  The remaining continuation is persisted across reopening.
        assert not repository._compact_inventory_after_retirement(family)
    finally:
        repository.close()
        file_ops.close()

    reopened_ops = ManagedFileOps(root)
    reopened = FileOperationRecordRepository(
        reopened_ops,
        lifecycle_limits=limits,
        initialization_key_provider=lambda: key,
    )
    try:
        if family == "primary":
            list_page = reopened.list_page
        elif family == "sidecar":
            list_page = reopened.list_reconciliation_checkpoint_page
        else:
            list_page = reopened.list_pending_control_page
        inspected: list[int] = []
        read_event = reopened._read_inventory_event

        def count_read(observed_family: str, sequence: int, **_kwargs: object):
            if observed_family == family:
                inspected.append(sequence)
            return read_event(observed_family, sequence)

        monkeypatch.setattr(reopened, "_read_inventory_event", count_read)
        passes = 0
        while not reopened.compact_inventory_for_recovery():
            passes += 1
            assert passes < 40
        # Receipt retirement now completes the final bounded window in the
        # same recovery invocation, rather than requiring a no-op confirmation
        # pass after the eighth pinned item.  The continuation remains bounded
        # and persisted across reopening.
        assert passes >= 8
        # Every accepted family head now authenticates its terminal tail member
        # in addition to the one bounded maintenance event.  The repeated
        # terminal read is fixed-size control verification, not an unbounded
        # maintenance scan.
        assert [sequence for sequence in inspected if sequence != 10] == list(range(2, 10))
        assert inspected.count(10) == 1

        # The exact preexisting high-water cursor still addresses the pinned
        # record, while a fresh one-item page chain reaches the later live
        # record without replaying the eight discarded primary/sidecar/pending
        # slots.
        old_page = list_page(old_cursor)
        assert [name for name, _raw in old_page.entries] == [pinned_name]
        first = list_page(None)
        assert [name for name, _raw in first.entries] == [pinned_name]
        assert first.next_cursor is not None
        second = list_page(first.next_cursor)
        assert [name for name, _raw in second.entries] == [later_name]
        assert second.next_cursor is None
        assert inspected[-2:] == [9, 10]
    finally:
        reopened.close()
        reopened_ops.close()


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
    # Scheduler-key initialization now correctly takes a second distinct
    # cross-process lock while the metadata lock is held.  Model the Windows
    # kernel primitive faithfully: separate lock handles remain reentrant for
    # their owning thread rather than deadlocking this deliberately simplified
    # fake on every nested descriptor.
    shared_lock = RLock()
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


@pytest.mark.parametrize("backend_name", ("memory", "json"))
def test_recovery_compacts_repeated_post_authority_failures_to_live_page_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend_name: str
) -> None:
    """Durable recovery skips stale runs without changing live event order."""
    limits = LifecycleLimits(manifest_page_size=1, max_inventory_items=1)
    if backend_name == "memory":
        backend = InMemoryBackend()
        repository = InMemoryManifestRepository(backend, lifecycle_limits=limits)
        metadata_path: Path | None = None
    else:
        metadata_path = tmp_path / "post-authority-maintenance.json"
        backend = JsonBackend(metadata_path)
        repository = JsonManifestRepository(backend, lifecycle_limits=limits)
    closed = False
    try:
        repository.put_raw("anchor", _record("anchor"))
        repository.put_raw("churn", _record("churn-0"))
        old_page = repository.list_page()
        assert [key for key, _raw in old_page.entries] == ["anchor"]
        assert old_page.next_cursor is not None

        real_compact = repository._compact_inventory_window

        def fail_after_authority() -> bool:
            raise OSError("injected post-authority compaction loss")

        monkeypatch.setattr(repository, "_compact_inventory_window", fail_after_authority)
        for generation in range(1, 9):
            observed = repository.get_raw("churn")
            assert observed is not None
            with pytest.raises(CacheBlobBackendError):
                repository.publish_if_expected(
                    "churn",
                    ManifestExpectation.from_authenticated_record(
                        f"generation-{generation}", observed
                    ),
                    _record(f"churn-{generation}"),
                )
        monkeypatch.setattr(repository, "_compact_inventory_window", real_compact)

        # Reopen the file-backed authority before consuming debt so the test
        # proves the exact cursor survives process loss, not just memory state.
        if metadata_path is not None:
            repository.close()
            backend.close()
            closed = True
            backend = JsonBackend(metadata_path)
            repository = JsonManifestRepository(backend, lifecycle_limits=limits)

        inspected: list[int] = []
        read_event = repository._read_inventory_event

        def count_read(sequence: int, **kwargs: object):
            inspected.append(sequence)
            return read_event(sequence, **kwargs)

        monkeypatch.setattr(repository, "_read_inventory_event", count_read)
        passes = 0
        while not repository.compact_inventory_for_recovery():
            passes += 1
            assert passes < 16
        assert passes == 9
        # Tail binding is fixed-size control verification at each accepted
        # head, while compaction still consumes exactly one historical slot per
        # pass under this limit-one configuration.
        assert [sequence for sequence in inspected if sequence != 10] == list(range(1, 10))
        assert inspected.count(10) >= 10

        # A cursor from the pre-compaction high-water remains a clean terminal
        # page.  It cannot resurrect a replaced record or force a new reader
        # to replay the discarded sparse prefix.
        assert repository.list_page(old_page.next_cursor).entries == ()
        first = repository.list_page()
        assert [key for key, _raw in first.entries] == ["anchor"]
        assert first.next_cursor is not None
        second = repository.list_page(first.next_cursor)
        assert [key for key, _raw in second.entries] == ["churn"]
        assert second.next_cursor is None
        # The two-page live snapshot only inspected its two original members.
        # The sparse-successor marker skips generations 2..9 without changing
        # the stable identity/order of the remaining sequence 10 member.
        assert 1 in inspected[-6:]
        assert inspected[-1] == 10
    finally:
        repository.close()
        if not closed:
            backend.close()


def test_json_manifest_concurrent_compaction_preserves_old_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent bounded recovery keeps authenticated old snapshots safe."""
    metadata_path = tmp_path / "concurrent-manifest-maintenance.json"
    limits = LifecycleLimits(manifest_page_size=1, max_inventory_items=1)
    key = b"m" * 32
    first_backend = JsonBackend(metadata_path)
    first = JsonManifestRepository(
        first_backend, lifecycle_limits=limits, inventory_key_provider=lambda: key
    )
    second_backend: JsonBackend | None = None
    second: JsonManifestRepository | None = None
    try:
        first.put_raw("anchor", _record("anchor"))
        first.put_raw("churn", _record("churn-0"))
        old_page = first.list_page()
        assert old_page.next_cursor is not None

        real_compact = first._compact_inventory_window

        def fail_after_authority() -> bool:
            raise OSError("post-authority loss")

        monkeypatch.setattr(first, "_compact_inventory_window", fail_after_authority)
        for generation in (1, 2, 3):
            observed = first.get_raw("churn")
            assert observed is not None
            with pytest.raises(CacheBlobBackendError):
                first.publish_if_expected(
                    "churn",
                    ManifestExpectation.from_authenticated_record(
                        f"generation-{generation}", observed
                    ),
                    _record(f"churn-{generation}"),
                )
        monkeypatch.setattr(first, "_compact_inventory_window", real_compact)

        # A second repository has independent in-process locks, so this proves
        # the file authority lock serializes two recoverers rather than merely
        # exercising the single-instance reentrant lock.
        second_backend = JsonBackend(metadata_path)
        second = JsonManifestRepository(
            second_backend, lifecycle_limits=limits, inventory_key_provider=lambda: key
        )
        gate = Barrier(2)
        outcomes: list[bool] = []
        failures: list[Exception] = []

        def compact(repository: JsonManifestRepository) -> None:
            try:
                gate.wait(timeout=5)
                outcomes.append(repository.compact_inventory_for_recovery())
            except Exception as exc:  # pragma: no cover - asserted below.
                failures.append(exc)

        threads = [
            Thread(target=compact, args=(first,)),
            Thread(target=compact, args=(second,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
        assert not failures
        assert len(outcomes) == 2

        while not first.compact_inventory_for_recovery():
            pass
        # A continuation captured before the concurrent skip compaction still
        # cannot revive the retired churn generation.
        assert first.list_page(old_page.next_cursor).entries == ()
    finally:
        first.close()
        first_backend.close()
        if second is not None:
            second.close()
        if second_backend is not None:
            second_backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json"))
def test_manifest_sparse_successor_marker_is_fail_closed(
    tmp_path: Path, backend_name: str
) -> None:
    """Malformed scheduling accelerators cannot hide a current manifest record."""
    limits = LifecycleLimits(manifest_page_size=1, max_inventory_items=1)
    if backend_name == "memory":
        backend = InMemoryBackend()
        repository = InMemoryManifestRepository(backend, lifecycle_limits=limits)
    else:
        backend = JsonBackend(tmp_path / "malformed-sparse-successor.json")
        repository = JsonManifestRepository(backend, lifecycle_limits=limits)
    try:
        repository.put_raw("current", _record("current"))
        if backend_name == "memory":
            state = repository._inventory_state()
            state["skips"][1] = {
                "version": 2,
                "start_sequence": 1,
                # A marker must move forward; a self-loop is not a benign
                # missing accelerator and must never become a page omission.
                "next_sequence": 1,
            }
        else:
            repository._json_lock_file_ops.write_bytes_durable(  # type: ignore[union-attr]
                repository._json_inventory_skip_locator(1), b"not-json"
            )

        with pytest.raises(CacheBlobBackendError, match="list_page failed"):
            repository.list_page()
        assert repository.get_raw("current") == _record("current")
    finally:
        repository.close()
        backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json"))
@pytest.mark.parametrize("replayed_control", ("head", "tail", "corrupt_tail"))
def test_manifest_append_tail_rejects_replayed_or_corrupt_control_member(
    tmp_path: Path, backend_name: str, replayed_control: str
) -> None:
    """A head/tail mismatch or corruption cannot hide immutable events."""
    metadata_path = tmp_path / "tail-control-replay.json"
    if backend_name == "memory":
        backend = InMemoryBackend()
        repository = InMemoryManifestRepository(backend)
        head_path: Path | None = None
        tail_path: Path | None = None
        saved_head: bytes | None = None
    else:
        backend = JsonBackend(metadata_path)
        repository = JsonManifestRepository(backend)
        head_path = repository._json_inventory_head_locator
        tail_path = repository._json_inventory_tail_locator_for_state()
        assert head_path is not None
        saved_head = None
    try:
        repository.put_raw("a", _record("a"))
        if backend_name == "memory":
            state = repository._inventory_state()
            saved_state = {
                name: state[name]
                for name in repository._inventory_head_field_names()
            }
            saved_tail = deepcopy(state["tail"])
        else:
            assert repository._json_lock_file_ops is not None
            saved_head = repository._json_lock_file_ops.read_bytes_bounded(
                head_path, max_bytes=4_096
            )
            saved_tail = repository._json_lock_file_ops.read_bytes_bounded(
                tail_path, max_bytes=4_096
            )
        repository.put_raw("b", _record("b"))
        repository.put_raw("c", _record("c"))

        if backend_name == "memory":
            if replayed_control == "head":
                for name, value in saved_state.items():
                    state[name] = value
            elif replayed_control == "tail":
                state["tail"] = saved_tail
            else:
                state["tail"] = {}
        else:
            assert repository._json_lock_file_ops is not None
            if replayed_control == "head":
                assert saved_head is not None
                repository._json_lock_file_ops.write_bytes_durable(head_path, saved_head)
            elif replayed_control == "tail":
                repository._json_lock_file_ops.write_bytes_durable(tail_path, saved_tail)
            else:
                repository._json_lock_file_ops.write_bytes_durable(tail_path, b"{}")
            # The durable control state must keep the same rejection on reopen,
            # not merely through cached state in the original repository.
            repository.close()
            backend.close()
            backend = JsonBackend(metadata_path)
            repository = JsonManifestRepository(backend)

        with pytest.raises(CacheBlobBackendError, match="list_page failed"):
            repository.list_page()
        # The canonical backend authority remains intact; only the scheduling
        # control plane is rejected as incomplete.
        assert repository.get_raw("c") == _record("c")
    finally:
        repository.close()
        backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json"))
@pytest.mark.parametrize("sequence", (1, 2, 3))
def test_manifest_missing_allocated_event_fails_closed_for_every_position(
    tmp_path: Path, backend_name: str, sequence: int
) -> None:
    """First, middle, and last allocated events need a direct skip proof."""
    if backend_name == "memory":
        backend = InMemoryBackend()
        repository = InMemoryManifestRepository(backend)
    else:
        backend = JsonBackend(tmp_path / "missing-event.json")
        repository = JsonManifestRepository(backend)
    try:
        for key in ("first", "middle", "last"):
            repository.put_raw(key, _record(key))
        first_page = repository.list_page(page_size=1)
        assert first_page.next_cursor is not None
        if backend_name == "memory":
            repository._inventory_state()["events"].pop(sequence)
        else:
            assert repository._json_lock_file_ops is not None
            repository._json_lock_file_ops.delete_durable(
                repository._json_inventory_event_locator(sequence)
            )
            metadata_path = backend.metadata_file
            repository.close()
            backend.close()
            backend = JsonBackend(metadata_path)
            repository = JsonManifestRepository(backend)

        with pytest.raises(CacheBlobBackendError, match="list_page failed"):
            if sequence == 1:
                repository.list_page(page_size=4)
            else:
                repository.list_page(first_page.next_cursor)
        assert repository.get_raw("last") == _record("last")
    finally:
        repository.close()
        backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json"))
@pytest.mark.parametrize("terminal", ("matching", "missing", "malformed", "different"))
def test_manifest_acknowledged_tail_terminal_binding_reopens_and_blocks_append(
    tmp_path: Path, backend_name: str, terminal: str
) -> None:
    """Every accepted manifest head retains an exact authenticated terminal."""
    metadata_path = tmp_path / f"manifest-terminal-{terminal}.json"
    if backend_name == "memory":
        backend = InMemoryBackend()
        repository = InMemoryManifestRepository(backend)
    else:
        backend = JsonBackend(metadata_path)
        repository = JsonManifestRepository(backend)
    try:
        repository.put_raw("a", _record("a"))
        repository.put_raw("b", _record("b"))
        state = repository._inventory_state()
        tail = repository._read_inventory_tail(state)
        assert tail is not None
        sequence = tail["terminal_sequence"]
        if terminal == "missing":
            if backend_name == "memory":
                state["events"].pop(sequence)
            else:
                assert repository._json_lock_file_ops is not None
                repository._json_lock_file_ops.delete_durable(
                    repository._json_inventory_event_locator(sequence)
                )
        elif terminal == "malformed":
            if backend_name == "memory":
                state["events"][sequence] = {}
            else:
                assert repository._json_lock_file_ops is not None
                repository._json_lock_file_ops.write_bytes_durable(
                    repository._json_inventory_event_locator(sequence), b"{}"
                )
        elif terminal == "different":
            event = repository._sign_inventory_value(
                {
                    "version": manifest_repository_module._MANIFEST_INVENTORY_SCHEMA_VERSION,
                    "store_id": repository._inventory_store_id(),
                    "epoch": state["epoch"],
                    "sequence": sequence,
                    "key": "different",
                    "digest": hashlib.sha256(_record("different")).hexdigest(),
                },
                initialize_new_store=True,
            )
            repository._validate_inventory_event(event, sequence)
            if backend_name == "memory":
                state["events"][sequence] = event
            else:
                assert repository._json_lock_file_ops is not None
                repository._json_lock_file_ops.write_bytes_durable(
                    repository._json_inventory_event_locator(sequence),
                    manifest_repository_module.json_dumps(event, default=str).encode("utf-8"),
                )
    finally:
        repository.close()
        if backend_name == "json":
            backend.close()

    if backend_name == "json":
        backend = JsonBackend(metadata_path)
    repository = (
        InMemoryManifestRepository(backend)
        if backend_name == "memory"
        else JsonManifestRepository(backend)
    )
    try:
        assert repository.get_raw("b") == _record("b")
        if terminal == "matching":
            assert [key for key, _raw in repository.list_page(page_size=4).entries] == ["a", "b"]
            repository.put_raw("c", _record("c"))
            assert repository.get_raw("c") == _record("c")
        else:
            with pytest.raises(CacheBlobBackendError, match="list_page failed"):
                repository.list_page(page_size=4)
            with pytest.raises(CacheBlobBackendError, match="put_raw failed"):
                repository.put_raw("c", _record("c"))
            assert repository.get_raw("b") == _record("b")
    finally:
        repository.close()
        backend.close()


@pytest.mark.parametrize("backend_name", ("memory", "json"))
def test_manifest_tail_ahead_crash_window_is_page_safe_and_append_resumable(
    tmp_path: Path, backend_name: str
) -> None:
    """Only a writer resumes an event/tail commit lacking its head acknowledgement."""
    if backend_name == "memory":
        backend = InMemoryBackend()
        repository = InMemoryManifestRepository(backend)
    else:
        backend = JsonBackend(tmp_path / "tail-ahead.json")
        repository = JsonManifestRepository(backend)
    try:
        repository.put_raw("a", _record("a"))
        state = repository._inventory_state()
        sequence = state["next_sequence"]
        event = repository._sign_inventory_value(
            {
                "version": manifest_repository_module._MANIFEST_INVENTORY_SCHEMA_VERSION,
                "store_id": repository._inventory_store_id(),
                "epoch": state["epoch"],
                "sequence": sequence,
                "key": "b",
                "digest": hashlib.sha256(_record("b")).hexdigest(),
            },
            initialize_new_store=True,
        )
        repository._validate_inventory_event(event, sequence)
        if backend_name == "memory":
            state["events"][sequence] = event
        else:
            assert repository._json_lock_file_ops is not None
            repository._json_lock_file_ops.create_bytes_durable_exclusive(
                repository._json_inventory_event_locator(sequence),
                manifest_repository_module.json_dumps(event, default=str).encode("utf-8"),
            )
        repository._write_inventory_tail(
            state,
            sequence + 1,
            terminal_event=(event["key"], event["digest"]),
        )

        with pytest.raises(CacheBlobBackendError, match="list_page failed"):
            repository.list_page()

        # The exact matching publication may acknowledge the sole durable
        # crash window.  Reads and reconciliation never make this mutation.
        repository.put_raw("b", _record("b"))
        page = repository.list_page(page_size=4)
        assert [key for key, _raw in page.entries] == ["a", "b"]
    finally:
        repository.close()
        backend.close()


def test_json_manifest_tail_ahead_missing_terminal_fails_before_replacement(
    tmp_path: Path,
) -> None:
    """A replayed head cannot reuse a tail-bound slot after its event is lost."""
    metadata_path = tmp_path / "tail-ahead-missing-terminal.json"
    backend = JsonBackend(metadata_path)
    repository = JsonManifestRepository(backend)
    try:
        repository.put_raw("first", _record("first"))
        assert repository._json_lock_file_ops is not None
        assert repository._json_inventory_head_locator is not None
        saved_head = repository._json_lock_file_ops.read_bytes_bounded(
            repository._json_inventory_head_locator, max_bytes=4_096
        )
        repository.put_raw("lost", _record("lost"))
        lost_sequence = repository._inventory_state()["next_sequence"] - 1
        repository._json_lock_file_ops.write_bytes_durable(
            repository._json_inventory_head_locator, saved_head
        )
        repository._json_lock_file_ops.delete_durable(
            repository._json_inventory_event_locator(lost_sequence)
        )
    finally:
        repository.close()
        backend.close()

    backend = JsonBackend(metadata_path)
    repository = JsonManifestRepository(backend)
    try:
        with pytest.raises(CacheBlobBackendError, match="put_raw failed"):
            repository.put_raw("replacement", _record("replacement"))
        assert repository.get_raw("lost") == _record("lost")
        assert repository.get_raw("replacement") is None
        assert not repository._json_inventory_event_locator(lost_sequence).exists()
    finally:
        repository.close()
        backend.close()


@pytest.mark.parametrize(
    ("collision_keys", "expected_offset"),
    (
        (("b",), 0),
        (("other",), 1),
        (("other", "b"), 1),
        (("other", "other-two"), 2),
    ),
)
def test_json_manifest_collision_rebuilds_sequence_bound_event_bytes(
    tmp_path: Path, collision_keys: tuple[str, ...], expected_offset: int
) -> None:
    """Every collision retry writes bytes signed for its own sequence."""
    backend = JsonBackend(tmp_path / f"collision-{len(collision_keys)}.json")
    repository = JsonManifestRepository(backend)
    try:
        repository.put_raw("a", _record("a"))
        state = repository._inventory_state()
        sequence = state["next_sequence"]
        assert repository._json_lock_file_ops is not None
        for offset, existing_key in enumerate(collision_keys):
            event_sequence = sequence + offset
            event = repository._sign_inventory_value(
                {
                    "version": manifest_repository_module._MANIFEST_INVENTORY_SCHEMA_VERSION,
                    "store_id": repository._inventory_store_id(),
                    "epoch": state["epoch"],
                    "sequence": event_sequence,
                    "key": existing_key,
                    "digest": hashlib.sha256(_record(existing_key)).hexdigest(),
                },
                initialize_new_store=True,
            )
            repository._json_lock_file_ops.create_bytes_durable_exclusive(
                repository._json_inventory_event_locator(event_sequence),
                manifest_repository_module.json_dumps(event, default=str).encode("utf-8"),
            )

        repository.put_raw("b", _record("b"))
        expected_sequence = sequence + expected_offset
        event_raw = repository._json_lock_file_ops.read_bytes_bounded(
            repository._json_inventory_event_locator(expected_sequence),
            max_bytes=repository._inventory_event_max_bytes(),
        )
        persisted = json.loads(event_raw)
        assert persisted["sequence"] == expected_sequence
        assert persisted["key"] == "b"
        assert repository.get_raw("b") == _record("b")
        page = repository.list_page(page_size=4)
        assert [key for key, _raw in page.entries] == ["a", "b"]
    finally:
        repository.close()
        backend.close()


@pytest.mark.parametrize("family", ("primary", "sidecar", "pending"))
@pytest.mark.parametrize("replayed_control", ("head", "tail", "corrupt_tail"))
def test_operation_append_tail_rejects_replayed_or_corrupt_control_member(
    tmp_path: Path, family: str, replayed_control: str
) -> None:
    """Each family rejects a stale or corrupt high-water control member."""
    root = tmp_path / f"operation-tail-{family}"
    root.mkdir()
    file_ops = ManagedFileOps(root)
    repository = FileOperationRecordRepository(
        file_ops,
        lifecycle_limits=LifecycleLimits(),
        initialization_key_provider=lambda: b"t" * 32,
    )
    first_id = "a" * 32
    second_id = "b" * 32
    first_raw = b"first-indexed-control"
    second_raw = b"second-indexed-control"
    if family == "primary":
        first_name, second_name = first_id, second_id
        first_locator = repository.locator_for(first_id)
        second_locator = repository.locator_for(second_id)
        read_page = repository.list_page
    elif family == "sidecar":
        first_name, second_name = first_id, second_id
        first_locator = repository.reconciliation_checkpoint_locator(first_id)
        second_locator = repository.reconciliation_checkpoint_locator(second_id)
        read_page = repository.list_reconciliation_checkpoint_page
    else:
        first_name = (
            f".{first_id}.json.pending.{hashlib.sha256(first_raw).hexdigest()}."
            f"{'1' * 32}.tmp"
        )
        second_name = (
            f".{second_id}.json.pending.{hashlib.sha256(second_raw).hexdigest()}."
            f"{'2' * 32}.tmp"
        )
        first_locator = root / "operations" / first_name
        second_locator = root / "operations" / second_name

        def read_page():
            return repository.list_pending_control_page(None)
    try:
        repository._append_inventory_event(family, first_name, first_raw)
        file_ops.write_bytes_durable(first_locator, first_raw)
        saved_head = file_ops.read_bytes_bounded(
            repository._inventory_head_receipt_locator(family, 1), max_bytes=4_096
        )
        saved_tail = file_ops.read_bytes_bounded(
            repository._inventory_tail_receipt_locator(family, 1), max_bytes=4_096
        )
        repository._append_inventory_event(family, second_name, second_raw)
        file_ops.write_bytes_durable(second_locator, second_raw)
        if replayed_control == "head":
            file_ops.write_bytes_durable(
                repository._inventory_head_receipt_locator(family, 2), saved_head
            )
        elif replayed_control == "tail":
            file_ops.write_bytes_durable(
                repository._inventory_tail_receipt_locator(family, 2), saved_tail
            )
        else:
            file_ops.write_bytes_durable(
                repository._inventory_tail_receipt_locator(family, 2), b"{}"
            )

        with pytest.raises(
            (CacheBlobBackendError, CacheManifestIntegrityError)
        ):
            read_page()
        assert file_ops.read_bytes_bounded(
            second_locator, max_bytes=1_048_576
        ) == second_raw
    finally:
        repository.close()
        file_ops.close()

@pytest.mark.parametrize("family", ("primary", "sidecar", "pending"))
def test_operation_tail_ahead_missing_terminal_fails_before_slot_replacement(
    tmp_path: Path, family: str
) -> None:
    """Every operation family preserves its tail-bound terminal slot on reopen."""
    root = tmp_path / f"operation-tail-ahead-missing-{family}"
    root.mkdir()
    key = b"q" * 32
    file_ops = ManagedFileOps(root)
    repository = FileOperationRecordRepository(
        file_ops,
        lifecycle_limits=LifecycleLimits(),
        initialization_key_provider=lambda: key,
    )
    first_id, lost_id, replacement_id = "a" * 32, "b" * 32, "c" * 32
    first_raw, lost_raw, replacement_raw = b"first", b"lost", b"replacement"
    if family == "primary":
        first_name, lost_name, replacement_name = first_id, lost_id, replacement_id
        first_locator = repository.locator_for(first_id)
        lost_locator = repository.locator_for(lost_id)
    elif family == "sidecar":
        first_name, lost_name, replacement_name = first_id, lost_id, replacement_id
        first_locator = repository.reconciliation_checkpoint_locator(first_id)
        lost_locator = repository.reconciliation_checkpoint_locator(lost_id)
    else:
        def pending_name(operation_id: str, raw: bytes, token: str) -> str:
            return (
                f".{operation_id}.json.pending.{hashlib.sha256(raw).hexdigest()}."
                f"{token * 32}.tmp"
            )

        first_name = pending_name(first_id, first_raw, "1")
        lost_name = pending_name(lost_id, lost_raw, "2")
        replacement_name = pending_name(replacement_id, replacement_raw, "3")
        first_locator = root / "operations" / first_name
        lost_locator = root / "operations" / lost_name
    try:
        repository._append_inventory_event(family, first_name, first_raw)
        file_ops.write_bytes_durable(first_locator, first_raw)
        saved_head = file_ops.read_bytes_bounded(
            repository._inventory_head_locator(family), max_bytes=4_096
        )
        repository._append_inventory_event(family, lost_name, lost_raw)
        file_ops.write_bytes_durable(lost_locator, lost_raw)
        lost_sequence = repository._read_inventory(family)["next_sequence"] - 1
        file_ops.write_bytes_durable(repository._inventory_head_locator(family), saved_head)
        file_ops.delete_durable(repository._inventory_event_locator(family, lost_sequence))
    finally:
        repository.close()
        file_ops.close()

    file_ops = ManagedFileOps(root)
    repository = FileOperationRecordRepository(
        file_ops,
        lifecycle_limits=LifecycleLimits(),
        initialization_key_provider=lambda: key,
    )
    try:
        with pytest.raises(CacheManifestIntegrityError, match="terminal event is missing"):
            repository._append_inventory_event(
                family, replacement_name, replacement_raw
            )
        assert file_ops.read_bytes_bounded(lost_locator, max_bytes=4_096) == lost_raw
        assert not repository._inventory_event_locator(family, lost_sequence).exists()
    finally:
        repository.close()
        file_ops.close()


@pytest.mark.parametrize("family", ("primary", "sidecar", "pending"))
@pytest.mark.parametrize("terminal", ("matching", "missing", "malformed", "different"))
def test_operation_receipt_terminal_binding_reopens_and_blocks_bad_append(
    tmp_path: Path, family: str, terminal: str
) -> None:
    """Accepted receipt heads bind an exact terminal across every family."""
    root = tmp_path / f"receipt-terminal-{family}-{terminal}"
    root.mkdir()
    key = b"r" * 32
    first_id, second_id = "a" * 32, "b" * 32
    first_raw, second_raw = b"first-receipt-control", b"second-receipt-control"

    def pending_name(operation_id: str, raw: bytes, token: str) -> str:
        return (
            f".{operation_id}.json.pending.{hashlib.sha256(raw).hexdigest()}."
            f"{token * 32}.tmp"
        )

    if family == "primary":
        first_name, second_name = first_id, second_id

        def write_current(repository: FileOperationRecordRepository, name: str, raw: bytes) -> None:
            repository.file_ops.write_bytes_durable(repository.locator_for(name), raw)

        def read_current(repository: FileOperationRecordRepository, name: str) -> bytes | None:
            return repository.get_raw(name)

        def read_page(repository: FileOperationRecordRepository):
            return repository.list_page(None)
    elif family == "sidecar":
        first_name, second_name = first_id, second_id

        def write_current(repository: FileOperationRecordRepository, name: str, raw: bytes) -> None:
            repository.file_ops.write_bytes_durable(
                repository.reconciliation_checkpoint_locator(name), raw
            )

        def read_current(repository: FileOperationRecordRepository, name: str) -> bytes | None:
            return repository.get_reconciliation_checkpoint_raw(name)

        def read_page(repository: FileOperationRecordRepository):
            return repository.list_reconciliation_checkpoint_page(None)
    else:
        first_name = pending_name(first_id, first_raw, "1")
        second_name = pending_name(second_id, second_raw, "2")

        def write_current(repository: FileOperationRecordRepository, name: str, raw: bytes) -> None:
            repository.file_ops.write_bytes_durable(root / "operations" / name, raw)

        def read_current(repository: FileOperationRecordRepository, name: str) -> bytes | None:
            return repository._get_pending_control_raw(name)

        def read_page(repository: FileOperationRecordRepository):
            return repository.list_pending_control_page(None)

    file_ops = ManagedFileOps(root)
    repository = FileOperationRecordRepository(
        file_ops, lifecycle_limits=LifecycleLimits(), initialization_key_provider=lambda: key
    )
    try:
        repository._append_inventory_event(family, first_name, first_raw)
        write_current(repository, first_name, first_raw)
        tail_locator = repository._inventory_tail_receipt_locator(family, 1)
        if terminal == "missing":
            file_ops.delete_durable(tail_locator)
        elif terminal == "malformed":
            file_ops.write_bytes_durable(tail_locator, b"{}")
        elif terminal == "different":
            tail = json.loads(file_ops.read_bytes_bounded(tail_locator, max_bytes=4_096))
            tail["name"] = "d" * 32
            tail["digest"] = hashlib.sha256(b"different-terminal").hexdigest()
            tail.pop("signature")
            signed = repository._sign_inventory_scheduling(tail)
            file_ops.write_bytes_durable(
                tail_locator,
                json.dumps(signed, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            )
    finally:
        repository.close()
        file_ops.close()

    reopened_ops = ManagedFileOps(root)
    reopened = FileOperationRecordRepository(
        reopened_ops,
        lifecycle_limits=LifecycleLimits(),
        initialization_key_provider=lambda: key,
    )
    try:
        assert read_current(reopened, first_name) == first_raw
        if terminal == "matching":
            assert read_page(reopened).entries
            reopened._append_inventory_event(family, second_name, second_raw)
            write_current(reopened, second_name, second_raw)
            assert read_current(reopened, second_name) == second_raw
        else:
            with pytest.raises((CacheBlobBackendError, CacheManifestIntegrityError)):
                read_page(reopened)
            with pytest.raises((CacheBlobBackendError, CacheManifestIntegrityError)):
                reopened._append_inventory_event(family, second_name, second_raw)
            assert read_current(reopened, first_name) == first_raw
    finally:
        reopened.close()
        reopened_ops.close()

@pytest.mark.parametrize("family", ("primary", "sidecar", "pending"))
@pytest.mark.parametrize("sequence", (1, 2, 3))
def test_operation_missing_allocated_event_fails_closed_for_every_position(
    tmp_path: Path, family: str, sequence: int
) -> None:
    """No family treats a missing first, middle, or last member as stale."""
    root = tmp_path / f"operation-missing-{family}"
    root.mkdir()
    file_ops = ManagedFileOps(root)
    repository = FileOperationRecordRepository(
        file_ops,
        lifecycle_limits=LifecycleLimits(),
        initialization_key_provider=lambda: b"m" * 32,
    )
    if family == "primary":
        read_page = repository.list_page
    elif family == "sidecar":
        read_page = repository.list_reconciliation_checkpoint_page
    else:
        def read_page():
            return repository.list_pending_control_page(None)

    try:
        stored_records: list[tuple[Path, bytes]] = []
        for index, character in enumerate(("a", "b", "c"), start=1):
            operation_id = character * 32
            raw = f"current-indexed-control-{index}".encode("ascii")
            if family == "primary":
                name = operation_id
                locator = repository.locator_for(operation_id)
            elif family == "sidecar":
                name = operation_id
                locator = repository.reconciliation_checkpoint_locator(operation_id)
            else:
                name = (
                    f".{operation_id}.json.pending.{hashlib.sha256(raw).hexdigest()}."
                    f"{character * 32}.tmp"
                )
                locator = root / "operations" / name
            repository._append_inventory_event(family, name, raw)
            file_ops.write_bytes_durable(locator, raw)
            stored_records.append((locator, raw))
        file_ops.delete_durable(repository._inventory_event_locator(family, sequence))

        with pytest.raises(CacheManifestIntegrityError, match="event is missing"):
            read_page()
        for locator, raw in stored_records:
            assert file_ops.read_bytes_bounded(locator, max_bytes=1_048_576) == raw
    finally:
        repository.close()
        file_ops.close()
