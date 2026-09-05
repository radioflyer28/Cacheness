"""End-to-end tracer coverage for BlobStore's immutable write lifecycle."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from cacheness import CacheConfig
from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobRecoverableCleanupError,
    CacheUnsafePathError,
)
from cacheness.metadata import InMemoryBackend
from cacheness.storage import BlobStore
from cacheness.storage.manifest import BlobManifestV1
from cacheness.storage.manifest_repository import ManifestCursor, ManifestPage
from cacheness.storage.operation_record import (
    ClearTarget,
    ClearTargetCheckpoint,
    ClearTargetPage,
)
from cacheness.storage.operation_repository import FileOperationRecordRepository
from cacheness.storage.path_security import ManagedFileOps, resolve_managed_locator


class _NativeJsonHandler:
    """Small handler whose bytes can be read by a native JSON reader."""

    data_type = "native_json"
    payload_format = "json"
    payload_format_version = 1

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def put(self, data: Any, file_path: Path, _config: Any) -> dict[str, Any]:
        self.events.append("private_serialization")
        payload_path = file_path.with_suffix(".json")
        payload_path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        return {
            "actual_path": str(payload_path),
            "file_size": payload_path.stat().st_size,
            "metadata": {"storage_format": "json"},
            "storage_format": "json",
        }

    def get(self, file_path: Path, _metadata: dict[str, Any]) -> Any:
        self.events.append("handler_read")
        return json.loads(file_path.read_text(encoding="utf-8"))


class _SingleHandlerRegistry:
    """Keep the tracer's payload format and handler resolution observable."""

    def __init__(self, handler: _NativeJsonHandler) -> None:
        self.handler = handler

    def get_handler(self, _data: Any) -> _NativeJsonHandler:
        return self.handler

    def get_handler_by_type(self, data_type: str) -> _NativeJsonHandler:
        assert data_type == self.handler.data_type
        return self.handler

    def resolve_payload_contract(
        self,
        data_type: str,
        payload_format: str,
        payload_format_version: int,
    ) -> _NativeJsonHandler:
        assert data_type == self.handler.data_type
        assert payload_format == self.handler.payload_format
        assert payload_format_version == self.handler.payload_format_version
        return self.handler


class _SimulatedProcessLoss(BaseException):
    """Model a crash that skips the ordinary-exception cleanup path."""


def _exit_during_control_durability_step(
    root: str, step: str, control_prefix: str
) -> None:
    """Stop a fresh process at one real durable-control boundary."""
    store = BlobStore(root, backend="json")
    file_ops = store.lifecycle.operation_repository.file_ops

    if control_prefix:
        store.put({"survivor": True}, key="control-process-survivor")

    def stop(observed_step: str, locator: Path) -> None:
        if observed_step == step and locator.name.startswith(control_prefix):
            os._exit(23)

    file_ops.after_control_durability_step = stop
    if control_prefix:
        store.clear()
    else:
        store.put({"interrupted": step}, key="interrupted-control-record")


def _exit_during_fixed_lock_creation(root: str, relative_locator: str, step: str) -> None:
    """Model process loss around lock and root-authority publication."""
    file_ops = ManagedFileOps(root)
    locator = resolve_managed_locator(
        file_ops.root,
        relative_locator,
        operation="fixed_lock_process_loss",
        allow_missing_leaf=True,
    )

    def stop(observed_step: str, _locator: Path) -> None:
        if observed_step == step:
            os._exit(23)

    file_ops.after_control_durability_step = stop
    file_ops.ensure_lifecycle_lock(locator)


class _FailingSerializationHandler(_NativeJsonHandler):
    """Prove private handler serialization precedes lifecycle evidence."""

    def put(self, data: Any, file_path: Path, config: Any) -> dict[str, Any]:
        del data, file_path, config
        self.events.append("private_serialization")
        raise RuntimeError("native serialization failed")


def test_tracer_json_put_uses_immutable_generation_cas_and_native_bytes(
    tmp_path: Path,
) -> None:
    """A direct write publishes one complete generation through the lifecycle."""
    events: list[str] = []
    store = BlobStore(tmp_path / "store", backend="json")
    store.handlers = _SingleHandlerRegistry(_NativeJsonHandler(events))
    lifecycle_events: list[str] = []
    store.lifecycle.test_hook = lifecycle_events.append

    try:
        key = store.put({"generation": 1}, key="tracer-key")
        first_metadata = store.get_metadata(key)
        assert first_metadata is not None
        first_locator = store.cache_dir / first_metadata["metadata"]["actual_path"]
        assert json.loads(first_locator.read_text(encoding="utf-8")) == {"generation": 1}

        assert store.put({"generation": 2}, key=key) == key
        second_metadata = store.get_metadata(key)
        assert second_metadata is not None
        second_locator = store.cache_dir / second_metadata["metadata"]["actual_path"]
        assert second_locator != first_locator
        assert second_locator.parent.parent.name == "generations"
        assert not first_locator.exists()
        assert store.get(key) == {"generation": 2}
        assert not list((store.cache_dir / "operations").glob("*.json"))
        assert lifecycle_events == [
            "put.intent_prepared",
            "put.before_candidate_publish",
            "put.candidate_published",
            "put.candidate_verified",
            "put.before_promotion",
            "put.promoted",
            "put.cleanup_retired",
            "put.intent_prepared",
            "put.before_candidate_publish",
            "put.candidate_published",
            "put.candidate_verified",
            "put.before_promotion",
            "put.promoted",
            "cleanup.before_payload_delete",
            "cleanup.after_payload_delete",
            "put.cleanup_retired",
        ]
        assert events == [
            "private_serialization",
            "private_serialization",
            "handler_read",
        ]
    finally:
        store.close()


def test_default_store_promotes_authority_backed_metadata_without_scheduler_artifacts(
    tmp_path: Path,
) -> None:
    """Public writes keep old/new native generations under one authority."""
    events: list[str] = []
    root = tmp_path / "authority-backed-store"
    store = BlobStore(root, backend="json")
    store.handlers = _SingleHandlerRegistry(_NativeJsonHandler(events))

    try:
        assert store.put({"generation": 1}, key="authority-key", metadata={"phase": 1}) == "authority-key"
        assert store.get("authority-key") == {"generation": 1}

        assert store.update_metadata("authority-key", {"phase": 2, "owner": "tests"})
        metadata = store.get_metadata("authority-key")
        assert metadata is not None
        assert metadata["metadata"]["phase"] == 2
        assert metadata["metadata"]["owner"] == "tests"

        assert store.put({"generation": 2}, key="authority-key") == "authority-key"
        assert store.get("authority-key") == {"generation": 2}
        assert (root / ".cacheness" / "lifecycle-authority-v1.sqlite3").is_file()
        assert not (root / "operations").exists()
    finally:
        store.close()


def test_clear_snapshot_preserves_post_snapshot_create_and_overwrite(
    tmp_path: Path,
) -> None:
    """Clear removes only the exact generations captured in its authority snapshot."""
    root = tmp_path / "clear-snapshot-post-snapshot-writes"
    store = BlobStore(root, backend="json")
    try:
        store.put({"generation": "old"}, key="existing")

        def create_after_snapshot(boundary: str) -> None:
            if boundary == "clear.snapshot_committed":
                store.put({"generation": "new"}, key="existing")
                store.put({"generation": "post-snapshot"}, key="late")

        store.lifecycle.test_hook = create_after_snapshot
        store.clear()

        assert store.get("existing") == {"generation": "new"}
        assert store.get("late") == {"generation": "post-snapshot"}
    finally:
        store.close()


@pytest.mark.skipif(os.name != "posix", reason="special-node substitution fixture")
@pytest.mark.parametrize("replacement_kind", ("symlink", "hard_link", "fifo", "inode"))
def _retired_scheduler_candidate_verification_rejects_every_substituted_inode_before_manifest_cas(
    tmp_path: Path, replacement_kind: str
) -> None:
    """Candidate verification never follows or commits a substituted payload inode."""
    root = tmp_path / f"candidate-substitution-{replacement_kind}"
    outside = tmp_path / f"outside-{replacement_kind}"
    outside.write_bytes(b"outside bytes must never be hashed")
    store = BlobStore(root, backend="json")
    key = "candidate-key"

    def replace_candidate(seam: str, record: object) -> None:
        if seam != "candidate_verification":
            return
        candidate = root / getattr(record, "candidate_locator")
        candidate.unlink()
        if replacement_kind == "symlink":
            candidate.symlink_to(outside)
        elif replacement_kind == "hard_link":
            os.link(outside, candidate)
        elif replacement_kind == "fifo":
            os.mkfifo(candidate)
        else:
            candidate.write_bytes(b"replacement inode")

    store.lifecycle.fault_hook = replace_candidate
    try:
        with pytest.raises(CacheUnsafePathError):
            store.put({"payload": replacement_kind}, key=key)
        assert store.manifest_repository.get_raw(key) is None
        assert outside.read_bytes() == b"outside bytes must never be hashed"
    finally:
        store.close()


def test_serialization_failure_leaves_no_operation_evidence_or_candidate(
    tmp_path: Path,
) -> None:
    """The durable lifecycle starts only after native serialization succeeds."""
    root = tmp_path / "serialization-failure"
    events: list[str] = []
    store = BlobStore(root, backend="json")
    store.handlers = _SingleHandlerRegistry(_FailingSerializationHandler(events))

    try:
        with pytest.raises(RuntimeError, match="native serialization failed"):
            store.put({"value": "never-published"}, key="failure-key")

        assert events == ["private_serialization"]
        assert not list((root / "operations").glob("*.json"))
        assert not list(root.glob("*generation-*"))
        assert store.get_metadata("failure-key") is None
    finally:
        store.close()


def test_delete_retries_exact_tombstone_cleanup_before_absent_retirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A promoted tombstone survives cleanup failure and resumes only its debt."""
    root = tmp_path / "tombstone-cleanup"
    store = BlobStore(root, backend="json")
    try:
        key = store.put({"generation": "old"}, key="delete-key")
        metadata = store.get_metadata(key)
        assert metadata is not None
        old_locator = root / metadata["metadata"]["actual_path"]
        original_delete = store._delete_or_prove_absent

        monkeypatch.setattr(
            store,
            "_delete_or_prove_absent",
            lambda _locator: (_ for _ in ()).throw(OSError("defer exact cleanup")),
        )
        with pytest.raises(CacheBlobRecoverableCleanupError):
            store.delete(key)

        assert store.get(key) is None
        assert old_locator.exists()

        monkeypatch.setattr(store, "_delete_or_prove_absent", original_delete)
        assert store.delete(key) is True
        assert not old_locator.exists()
        assert store.get_metadata(key) is None
        assert store.delete(key) is False
    finally:
        store.close()


def test_reconciliation_reclaims_only_old_tombstone_debt_after_new_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup from a tombstoned generation cannot revoke a later promotion."""
    root = tmp_path / "tombstone-new-winner"
    store = BlobStore(root, backend="json")
    try:
        key = store.put({"generation": "old"}, key="winner-key")
        metadata = store.get_metadata(key)
        assert metadata is not None
        old_locator = root / metadata["metadata"]["actual_path"]
        original_delete = store._delete_or_prove_absent
        monkeypatch.setattr(
            store,
            "_delete_or_prove_absent",
            lambda _locator: (_ for _ in ()).throw(OSError("defer exact cleanup")),
        )
        with pytest.raises(CacheBlobRecoverableCleanupError):
            store.delete(key)

        monkeypatch.setattr(store, "_delete_or_prove_absent", original_delete)
        assert store.put({"generation": "new"}, key=key) == key
        assert store.get(key) == {"generation": "new"}
        assert old_locator.exists()

        assert store.reconcile(apply=True).applied == 1
        assert not old_locator.exists()
        assert store.get(key) == {"generation": "new"}
    finally:
        store.close()


@pytest.mark.parametrize(
    "fault_step",
    (
        "evidence_created",
        "candidate_published",
        "candidate_verified",
        "authority_published",
        "cleanup_completed",
        "evidence_retired",
    ),
)
@pytest.mark.parametrize("failure_kind", ("ordinary", "crash"))
def _retired_scheduler_lifecycle_failure_boundary_reopens_to_one_complete_generation(
    tmp_path: Path,
    fault_step: str,
    failure_kind: str,
) -> None:
    """Every persisted boundary converges after exceptions or process loss."""
    root = tmp_path / f"reopen-{fault_step}-{failure_kind}"
    store = BlobStore(root, backend="json")
    previous_path: Path | None = None
    try:
        key = store.put({"generation": "old"}, key="failure-key")
        existing = store._load_authenticated_manifest(
            key,
            operation="test",
            require_locator=True,
        )
        assert existing is not None
        previous_path = existing[2]
        assert previous_path is not None

        def interrupt(step: str, _record: Any) -> None:
            if step != fault_step:
                return
            if failure_kind == "crash":
                raise _SimulatedProcessLoss(f"process lost at {step}")
            raise RuntimeError(f"ordinary failure at {step}")

        store.lifecycle.test_hook = interrupt
        expected_exception: type[BaseException]
        if failure_kind == "crash":
            expected_exception = _SimulatedProcessLoss
        else:
            expected_exception = Exception
        with pytest.raises(expected_exception):
            store.put({"generation": "new"}, key=key)
    finally:
        store.close()

    assert previous_path is not None
    reopened = BlobStore(root, backend="json")
    try:
        new_authority = fault_step in {
            "authority_published",
            "cleanup_completed",
            "evidence_retired",
        }
        expected = {"generation": "new"} if new_authority else {"generation": "old"}
        assert reopened.get("failure-key") == expected
        assert not list((root / "operations").glob("*.json"))
        if new_authority:
            assert not previous_path.exists()
        else:
            assert previous_path.exists()
    finally:
        reopened.close()


def _retired_scheduler_cleanup_failure_keeps_new_authority_and_resumes_after_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-authority cleanup debt carries stable context and remains resumable."""
    root = tmp_path / "cleanup-failure"
    store = BlobStore(root, backend="json")
    previous_path: Path | None = None
    try:
        key = store.put({"generation": "old"}, key="cleanup-key")
        existing = store._load_authenticated_manifest(
            key,
            operation="test",
            require_locator=True,
        )
        assert existing is not None
        previous_path = existing[2]
        assert previous_path is not None

        def reject_cleanup(_locator: Path) -> None:
            raise OSError("simulated cleanup failure")

        monkeypatch.setattr(store, "_delete_or_prove_absent", reject_cleanup)
        with pytest.raises(CacheBlobRecoverableCleanupError) as error:
            store.put({"generation": "new"}, key=key)

        assert error.value.context["operation_id"]
        assert error.value.context["generation"]
        assert error.value.context["key"] == key
        assert store.get(key) == {"generation": "new"}
        assert previous_path.exists()
        assert list((root / "operations").glob("*.json"))
    finally:
        store.close()

    assert previous_path is not None
    reopened = BlobStore(root, backend="json")
    try:
        assert reopened.get("cleanup-key") == {"generation": "new"}
        assert not previous_path.exists()
        assert not list((root / "operations").glob("*.json"))
    finally:
        reopened.close()


def _retired_scheduler_new_clear_uses_only_current_lifecycle_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new clear cannot revive the predecessor journal coordinator."""
    root = tmp_path / "new-clear-current-evidence"
    store = BlobStore(root, backend="json")
    try:
        store.put({"state": "present"}, key="clear-key")
        legacy_adapter = store._legacy_clear_evidence
        assert legacy_adapter is not None
        legacy_coordinator = store._clear_recovery
        assert legacy_coordinator is not None

        def reject_predecessor_clear(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("new clear must not use predecessor coordination")

        monkeypatch.setattr(legacy_coordinator, "clear", reject_predecessor_clear)
        observed: list[str] = []
        store.lifecycle.test_hook = lambda step, _record: observed.append(step)

        assert store.clear() == 1

        assert not legacy_adapter.has_evidence()
        assert "evidence_created" in observed
        assert not (root / ".cacheness-clear-journal-v1.json").exists()
    finally:
        store.close()


def _retired_scheduler_clear_pages_spill_large_valid_manifests_and_retire_control_evidence(
    tmp_path: Path,
) -> None:
    """Clear bounds encoded pages without rejecting valid large manifests."""
    root = tmp_path / "large-clear-pages"
    store = BlobStore(root, backend="json")
    try:
        for index in range(4):
            store.put(
                {"index": index},
                key=f"large-{index}",
                metadata={"large": "x" * 200_000},
            )

        assert store.clear() == 4
        assert store.list() == []
        operations = root / "operations"
        assert not list(operations.glob("clear-target-page-*.json"))
        assert not list(operations.glob("clear-target-checkpoint-*.json"))
        assert not list(operations.glob("clear-target-reference-*.json"))
    finally:
        store.close()

    reopened = BlobStore(root, backend="json")
    try:
        assert reopened.list() == []
        assert not list((root / "operations").glob("clear-target-*.json"))
    finally:
        reopened.close()


def _retired_scheduler_terminal_clear_retires_pages_resumably_before_main_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash during page retirement reopens and finishes exact cleanup."""
    root = tmp_path / "resumable-clear-retirement"
    store = BlobStore(root, backend="json")
    try:
        store.put(
            {"large": True},
            key="large",
            metadata={"large": "x" * 200_000},
        )
        original_retire = store.lifecycle.operation_repository.retire_clear_target_page_if_exact
        calls = 0

        def interrupt_once(*args: object, **kwargs: object) -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("simulated interruption after sidecar retirement")
            return original_retire(*args, **kwargs)

        monkeypatch.setattr(
            store.lifecycle.operation_repository,
            "retire_clear_target_page_if_exact",
            interrupt_once,
        )
        with pytest.raises(RuntimeError, match="simulated interruption"):
            store.clear()
        assert list((root / "operations").glob("clear-target-page-*.json"))
    finally:
        store.close()

    reopened = BlobStore(root, backend="json")
    try:
        assert reopened.list() == []
        assert not list((root / "operations").glob("clear-target-*.json"))
        assert not list((root / "operations").glob("[0-9a-f]" * 32 + ".json"))
    finally:
        reopened.close()


def test_stale_overwrite_conflict_reclaims_only_loser_candidate(
    tmp_path: Path,
) -> None:
    """A stale writer cannot revoke a winner and leaves no owned candidate behind."""
    root = tmp_path / "stale-overwrite"
    contender = BlobStore(root, backend="json")
    winner = BlobStore(root, backend="json")
    try:
        key = contender.put({"generation": "old"}, key="shared-key")
        published_winner = False

        def publish_winner(step: str, _record: Any) -> None:
            nonlocal published_winner
            if step == "candidate_verified" and not published_winner:
                published_winner = True
                winner.put({"generation": "winner"}, key=key)

        contender.lifecycle.test_hook = publish_winner
        with pytest.raises(CacheBlobLifecycleConflictError):
            contender.put({"generation": "loser"}, key=key)

        assert winner.get(key) == {"generation": "winner"}
        assert contender.get(key) == {"generation": "winner"}
        assert not list((root / "operations").glob("*.json"))
        generation_payloads = [
            path for path in (root / "generations").rglob("*") if path.is_file()
        ]
        assert len(generation_payloads) == 1
    finally:
        contender.close()
        winner.close()


def test_overwrite_snapshot_cas_never_adopts_a_winner_published_after_load(
    tmp_path: Path,
) -> None:
    """An overwrite CAS remains bound to the exact authenticated first read."""
    root = tmp_path / "overwrite-snapshot-cas"
    contender = BlobStore(root, backend="json")
    winner = BlobStore(root, backend="json")
    try:
        key = contender.put({"generation": "old"}, key="snapshot-key")
        published_winner = False

        def publish_after_candidate(step: str, _record: object) -> None:
            nonlocal published_winner
            if step == "candidate_verified" and not published_winner:
                published_winner = True
                winner.put({"generation": "winner"}, key=key)

        contender.lifecycle.test_hook = publish_after_candidate
        with pytest.raises(CacheBlobLifecycleConflictError):
            contender.put({"generation": "stale"}, key=key)

        assert contender.get(key) == {"generation": "winner"}
        assert winner.get(key) == {"generation": "winner"}
    finally:
        contender.close()
        winner.close()


def _retired_scheduler_delete_publishes_signed_tombstone_before_payload_reclamation(
    tmp_path: Path,
) -> None:
    """Delete preserves signed absence intent if reclamation is interrupted."""
    root = tmp_path / "tombstone-first"
    store = BlobStore(root, backend="json")
    payload_path: Path | None = None
    try:
        key = store.put({"state": "present"}, key="delete-key")
        existing = store._load_authenticated_manifest(
            key,
            operation="test",
            require_locator=True,
        )
        assert existing is not None
        payload_path = existing[2]
        assert payload_path is not None

        def interrupt_reclamation(seam: str, _record: Any) -> None:
            if seam != "payload_cleanup":
                return
            raw_tombstone = store.manifest_repository.get_raw(key)
            assert raw_tombstone is not None
            tombstone = BlobManifestV1.from_canonical_bytes(raw_tombstone)
            assert tombstone.state == "tombstoned"
            assert tombstone.signature
            assert payload_path.exists()
            raise RuntimeError("interrupted after tombstone authority")

        store.lifecycle.fault_hook = interrupt_reclamation
        with pytest.raises(CacheBlobRecoverableCleanupError) as error:
            store.delete(key)
        assert error.value.__cause__ is not None
    finally:
        store.close()

    assert payload_path is not None
    reopened = BlobStore(root, backend="json")
    try:
        assert reopened.get(key) is None
        assert not payload_path.exists()
        assert not list((root / "operations").glob("*.json"))
    finally:
        reopened.close()


def _retired_scheduler_repeated_delete_resumes_the_same_signed_tombstone(
    tmp_path: Path,
) -> None:
    """A repeated delete completes its own retained tombstone rather than guessing."""
    root = tmp_path / "repeated-delete"
    store = BlobStore(root, backend="json")
    try:
        key = store.put({"state": "present"}, key="repeat-key")

        def interrupt_reclamation(seam: str, _record: Any) -> None:
            if seam == "payload_cleanup":
                raise RuntimeError("pause tombstone cleanup")

        store.lifecycle.fault_hook = interrupt_reclamation
        with pytest.raises(CacheBlobRecoverableCleanupError):
            store.delete(key)

        store.lifecycle.fault_hook = None
        assert store.delete(key) is True
        assert store.delete(key) is False
        assert store.get(key) is None
        assert not list((root / "operations").glob("*.json"))
    finally:
        store.close()


def _retired_scheduler_repeated_delete_uses_the_signed_tombstone_operation_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unrelated inventory history cannot block a tombstone's own cleanup."""
    store = BlobStore(
        tmp_path / "direct-tombstone-reference",
        backend="json",
        config=CacheConfig(
            lifecycle_limits=LifecycleLimits(
                operation_page_size=1,
                max_inventory_items=1,
                max_reconcile_actions=1,
            )
        ),
    )
    try:
        key = store.put({"state": "present"}, key="repeat-key")

        def interrupt_reclamation(seam: str, _record: Any) -> None:
            if seam == "payload_cleanup":
                raise RuntimeError("pause tombstone cleanup")

        store.lifecycle.fault_hook = interrupt_reclamation
        with pytest.raises(CacheBlobRecoverableCleanupError):
            store.delete(key)

        raw_tombstone = store.manifest_repository.get_raw(key)
        assert raw_tombstone is not None
        tombstone = BlobManifestV1.from_canonical_bytes(raw_tombstone)
        operation_id = tombstone.handler_metadata[
            "_cacheness_tombstone_operation_id"
        ]
        assert isinstance(operation_id, str)

        def inventory_must_not_be_scanned(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("tombstone recovery scanned unrelated inventory")

        monkeypatch.setattr(
            store.lifecycle.operation_repository,
            "list_page",
            inventory_must_not_be_scanned,
        )
        store.lifecycle.fault_hook = None
        assert store.delete(key) is True
        assert store.get(key) is None
    finally:
        store.close()


def _retired_scheduler_stale_delete_conflict_preserves_newer_committed_generation(
    tmp_path: Path,
) -> None:
    """A delete that loses its tombstone CAS cannot revoke a newer winner."""
    root = tmp_path / "stale-delete"
    contender = BlobStore(root, backend="json")
    winner = BlobStore(root, backend="json")
    try:
        key = contender.put({"generation": "old"}, key="shared-delete-key")
        published_winner = False

        def publish_winner(seam: str, _record: Any) -> None:
            nonlocal published_winner
            if seam == "tombstone_publish" and not published_winner:
                published_winner = True
                winner.put({"generation": "winner"}, key=key)

        contender.lifecycle.fault_hook = publish_winner
        with pytest.raises(CacheBlobLifecycleConflictError):
            contender.delete(key)

        assert contender.get(key) == {"generation": "winner"}
        assert winner.get(key) == {"generation": "winner"}
        assert not list((root / "operations").glob("*.json"))
    finally:
        contender.close()
        winner.close()


def _retired_scheduler_delete_snapshot_cas_never_adopts_a_winner_published_after_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delete tombstone CAS remains bound to its authenticated snapshot."""
    root = tmp_path / "delete-snapshot-cas"
    contender = BlobStore(root, backend="json")
    winner = BlobStore(root, backend="json")
    try:
        key = contender.put({"generation": "old"}, key="snapshot-delete-key")
        original_load = contender._load_authenticated_manifest_with_raw
        published_winner = False

        def load_then_publish(*args: object, **kwargs: object):
            nonlocal published_winner
            snapshot = original_load(*args, **kwargs)
            if kwargs.get("operation") == "delete" and not published_winner:
                published_winner = True
                winner.put({"generation": "winner"}, key=key)
            return snapshot

        monkeypatch.setattr(
            contender, "_load_authenticated_manifest_with_raw", load_then_publish
        )
        with pytest.raises(CacheBlobLifecycleConflictError):
            contender.delete(key)

        assert contender.get(key) == {"generation": "winner"}
        assert winner.get(key) == {"generation": "winner"}
    finally:
        contender.close()
        winner.close()


def _retired_scheduler_tombstone_publication_conflict_reports_recoverable_evidence_debt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A loser-retirement failure retains the winner-preserving conflict as cause."""
    root = tmp_path / "tombstone-conflict-retirement"
    contender = BlobStore(root, backend="json")
    winner = BlobStore(root, backend="json")
    try:
        key = contender.put({"generation": "old"}, key="conflict-key")

        def publish_winner(seam: str, _record: Any) -> None:
            if seam == "tombstone_publish":
                winner.put({"generation": "winner"}, key=key)

        contender.lifecycle.fault_hook = publish_winner
        monkeypatch.setattr(
            contender.lifecycle,
            "_retire",
            lambda _record: (_ for _ in ()).throw(OSError("retirement failed")),
        )
        with pytest.raises(CacheBlobRecoverableCleanupError) as error:
            contender.delete(key)

        assert isinstance(error.value.__cause__, CacheBlobLifecycleConflictError)
        assert error.value.context["operation"] == "tombstone_publish_conflict"
        assert contender.get(key) == {"generation": "winner"}
    finally:
        contender.close()
        winner.close()


def _retired_scheduler_post_authority_tombstone_conflict_reports_recoverable_evidence_debt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-authority terminal debt cannot be masked by evidence retirement."""
    store = BlobStore(tmp_path / "tombstone-final-conflict", backend="json")
    try:
        key = store.put({"generation": "old"}, key="post-authority-key")
        monkeypatch.setattr(
            store.manifest_repository,
            "remove_if_expected",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                CacheBlobLifecycleConflictError("later authority won")
            ),
        )
        monkeypatch.setattr(
            store.lifecycle,
            "_retire",
            lambda _record: (_ for _ in ()).throw(OSError("retirement failed")),
        )
        with pytest.raises(CacheBlobRecoverableCleanupError) as error:
            store.delete(key)

        assert isinstance(error.value.__cause__, CacheBlobLifecycleConflictError)
        assert error.value.context["operation"] == "tombstone_retire_conflict"
    finally:
        store.close()


@pytest.mark.parametrize("replacement", [False, True])
def test_pre_cas_put_residue_converges_after_a_later_distinct_winner(
    tmp_path: Path, replacement: bool
) -> None:
    """Signed create/replace candidates do not remain blocked after losing CAS."""
    root = tmp_path / f"pre-cas-residue-{replacement}"
    contender = BlobStore(root, backend="json")
    winner = BlobStore(root, backend="json")
    candidate_locator: Path | None = None
    try:
        key = "replace-key" if replacement else "create-key"
        if replacement:
            contender.put({"generation": "old"}, key=key)

        def interrupt_before_cas(seam: str) -> None:
            if seam == "put.before_promotion":
                raise _SimulatedProcessLoss("interrupted before authority CAS")

        contender.lifecycle.fault_hook = interrupt_before_cas
        with pytest.raises(_SimulatedProcessLoss, match="interrupted before authority CAS"):
            contender.put({"generation": "interrupted"}, key=key)
        candidates = [path for path in (root / "generations").rglob("*") if path.is_file()]
        candidate_locator = candidates[-1]

        winner.put({"generation": "winner"}, key=key)
        assert winner.get(key) == {"generation": "winner"}
    finally:
        contender.close()
        winner.close()

    reopened = BlobStore(root, backend="json")
    try:
        assert reopened.get(key) == {"generation": "winner"}
        reopened.reconcile(apply=True)
        assert candidate_locator is not None and not candidate_locator.exists()
        assert not (root / "operations").exists()
    finally:
        reopened.close()


def _retired_scheduler_clear_target_page_and_checkpoint_preserve_exact_progress_after_reopen(
    tmp_path: Path,
) -> None:
    """Clear evidence persists exact targets and rejects stale progress writers."""
    root = tmp_path / "clear-target-evidence"
    store = BlobStore(root, backend="json")
    operation_id = "a" * 32
    page_id = "b" * 32
    raw_first = b'{"first":"exact"}'
    raw_second = b'{"second":"exact"}'
    # Direct control-fixture writes must establish the same authenticated
    # all-family inventory provenance that a public lifecycle operation does.
    store._manifest_key(initialize_new_store=True)
    page = ClearTargetPage(
        operation_id=operation_id,
        page_id=page_id,
        source_cursor=None,
        next_cursor="second",
        targets=(
            ClearTarget.from_raw("first", "c" * 32, raw_first),
            ClearTarget.from_raw("second", "d" * 32, raw_second),
        ),
        # Existing pages remain parseable as the v1 cursor-only form.
        schema_version=1,
    )
    repository = FileOperationRecordRepository(
        store.guarded_handler_io.file_ops,
        lifecycle_limits=store.lifecycle_limits,
        initialization_key_provider=store._initialize_inventory_provenance_key,
    )
    try:
        raw_page = page.canonical_bytes()
        repository.create_clear_target_page_exclusive(
            operation_id, page_id, raw_page
        )
        checkpoint = ClearTargetCheckpoint.initial_for(page)
        raw_initial = checkpoint.canonical_bytes()
        repository.create_clear_target_checkpoint_exclusive(
            operation_id, page_id, raw_initial
        )
        advanced = checkpoint.with_completed_target(0)
        raw_advanced = advanced.canonical_bytes()
        repository.checkpoint_clear_target_if_exact(
            operation_id,
            page_id,
            expected_raw=raw_initial,
            raw_record=raw_advanced,
        )
        with pytest.raises(CacheBlobLifecycleConflictError):
            repository.checkpoint_clear_target_if_exact(
                operation_id,
                page_id,
                expected_raw=raw_initial,
                raw_record=checkpoint.with_completed_target(1).canonical_bytes(),
            )
    finally:
        store.close()

    reopened = BlobStore(root, backend="json")
    try:
        repository = reopened.lifecycle.operation_repository
        persisted_page = repository.get_clear_target_page_raw(operation_id, page_id)
        persisted_checkpoint = repository.get_clear_target_checkpoint_raw(
            operation_id, page_id
        )
        assert persisted_page == raw_page
        assert persisted_checkpoint == raw_advanced
        assert ClearTargetPage.from_canonical_bytes(persisted_page) == page
        assert ClearTargetCheckpoint.from_canonical_bytes(
            persisted_checkpoint
        ).completed_target_indices == (0,)
    finally:
        reopened.close()


def _retired_scheduler_clear_snapshot_barrier_preserves_a_later_key_after_bounded_admission(
    tmp_path: Path,
) -> None:
    """Clear only admits its finite target snapshot before ordinary work resumes."""
    root = tmp_path / "clear-snapshot-barrier"
    limits = LifecycleLimits(manifest_page_size=1)
    config = CacheConfig(cache_dir=str(root), lifecycle_limits=limits)
    owner = BlobStore(root, backend="json", config=config)
    contender = BlobStore(root, backend="json", config=config)
    snapshot_complete = threading.Event()
    release_snapshot = threading.Event()
    later_complete = threading.Event()
    clear_errors: list[BaseException] = []

    try:
        owner.put({"generation": "before"}, key="before")
        assert owner.manifest_repository.lifecycle_limits is limits
        assert owner._admission_barrier is contender._admission_barrier

        def pause_at_snapshot(seam: str, _record: Any) -> None:
            if seam == "clear_snapshot_complete":
                snapshot_complete.set()
                assert release_snapshot.wait(timeout=5)

        def clear_owner() -> None:
            try:
                owner.clear()
            except BaseException as exc:
                clear_errors.append(exc)

        def publish_later() -> None:
            contender.put({"generation": "later"}, key="later")
            later_complete.set()

        owner.lifecycle.fault_hook = pause_at_snapshot
        clear_thread = threading.Thread(target=clear_owner)
        clear_thread.start()
        assert snapshot_complete.wait(timeout=5)

        put_thread = threading.Thread(target=publish_later)
        put_thread.start()
        assert not later_complete.wait(timeout=0.2)

        release_snapshot.set()
        clear_thread.join(timeout=5)
        put_thread.join(timeout=5)
        assert not clear_thread.is_alive()
        assert not put_thread.is_alive()
        assert clear_errors == []
        assert owner.get("before") is None
        assert contender.get("later") == {"generation": "later"}
    finally:
        contender.close()
        owner.close()


def _retired_scheduler_clear_conflict_does_not_revoke_a_later_generation(tmp_path: Path) -> None:
    """A snapshot target whose authority changed is retained as a conflict."""
    root = tmp_path / "clear-changed-target"
    owner = BlobStore(root, backend="json")
    winner = BlobStore(root, backend="json")
    overwritten = False

    try:
        owner.put({"generation": "before"}, key="shared")

        def overwrite_after_snapshot(seam: str, _record: Any) -> None:
            nonlocal overwritten
            if seam == "clear_target_delete" and not overwritten:
                overwritten = True
                winner.put({"generation": "later"}, key="shared")

        owner.lifecycle.fault_hook = overwrite_after_snapshot
        assert owner.clear() == 0
        assert overwritten
        assert owner.get("shared") == {"generation": "later"}
        assert winner.get("shared") == {"generation": "later"}
    finally:
        winner.close()
        owner.close()


def _retired_scheduler_clear_resume_does_not_repeat_completed_targets_after_reopen(
    tmp_path: Path,
) -> None:
    """Durable page checkpoints let recovery resume only the unfinished target."""
    root = tmp_path / "clear-resume"
    limits = LifecycleLimits(manifest_page_size=1)
    store = BlobStore(
        root,
        backend="json",
        config=CacheConfig(cache_dir=str(root), lifecycle_limits=limits),
    )
    completed_once = False
    observed_cursors: list[tuple[str | None, int | None, int | None]] = []
    try:
        store.put({"generation": "first"}, key="first")
        store.put({"generation": "second"}, key="second")

        original_list_page = store.manifest_repository.list_page

        def bounded_list_page(
            cursor: ManifestCursor | None = None,
            *,
            page_size: int | None = None,
        ) -> ManifestPage:
            observed_cursors.append(
                (
                    None if cursor is None else cursor.key,
                    None if cursor is None else cursor.snapshot_high_water,
                    None if cursor is None else cursor.next_sequence,
                )
            )
            # The initial key-provider existence check is followed by exactly
            # two snapshot pages. A reset to the first high-water page exceeds
            # this budget and fails deterministically.
            assert len(observed_cursors) <= 3
            return original_list_page(cursor, page_size=page_size)

        store.manifest_repository.list_page = bounded_list_page  # type: ignore[method-assign]

        def interrupt_after_checkpoint(seam: str, _record: Any) -> None:
            nonlocal completed_once
            if seam == "clear_target_checkpoint" and not completed_once:
                completed_once = True
                raise _SimulatedProcessLoss("interrupted after durable clear progress")

        store.lifecycle.fault_hook = interrupt_after_checkpoint
        with pytest.raises(_SimulatedProcessLoss):
            store.clear()
        assert observed_cursors == [
            (None, None, None),
            (None, None, None),
            ("first", 2, 2),
        ]
    finally:
        store.close()

    reopened = BlobStore(root, backend="json")
    try:
        assert completed_once
        assert reopened.get("first") is None
        assert reopened.get("second") is None
    finally:
        reopened.close()


def _retired_scheduler_prepared_clear_inventory_is_aborted_after_process_loss_before_next_page(
    tmp_path: Path,
) -> None:
    """Recovery never resumes a lexical inventory after its admission epoch ends."""
    root = tmp_path / "prepared-clear-process-loss"
    limits = LifecycleLimits(manifest_page_size=1)
    config = CacheConfig(cache_dir=str(root), lifecycle_limits=limits)
    owner = BlobStore(root, backend="json", config=config)
    independent_writer = BlobStore(root, backend="json", config=config)
    page_persisted = False
    observed_cursors: list[tuple[str | None, int | None, int | None]] = []
    try:
        owner.put({"generation": "a"}, key="a")
        owner.put({"generation": "z"}, key="z")

        original_list_page = owner.manifest_repository.list_page

        def record_first_high_water_page(
            cursor: ManifestCursor | None = None,
            *,
            page_size: int | None = None,
        ) -> ManifestPage:
            observed_cursors.append(
                (
                    None if cursor is None else cursor.key,
                    None if cursor is None else cursor.snapshot_high_water,
                    None if cursor is None else cursor.next_sequence,
                )
            )
            # One key-provider existence read precedes the first snapshot
            # page. The injected process loss must prevent a second page.
            assert len(observed_cursors) <= 2
            return original_list_page(cursor, page_size=page_size)

        owner.manifest_repository.list_page = record_first_high_water_page  # type: ignore[method-assign]

        def interrupt_after_first_page(seam: str, _record: Any) -> None:
            nonlocal page_persisted
            if seam == "clear_target_page_persisted" and not page_persisted:
                page_persisted = True
                raise _SimulatedProcessLoss("lost clear admission after first page")

        owner.lifecycle.fault_hook = interrupt_after_first_page
        with pytest.raises(_SimulatedProcessLoss):
            owner.clear()
        assert page_persisted
        assert observed_cursors == [(None, None, None), (None, None, None)]

        # This store existed before the interrupted clear and only gains its
        # ordinary admission after the simulated owner process releases it.
        independent_writer.put({"generation": "m"}, key="m")
    finally:
        owner.close()

    reopened = BlobStore(root, backend="json", config=config)
    try:
        assert reopened.get("a") == {"generation": "a"}
        assert reopened.get("m") == {"generation": "m"}
        assert reopened.get("z") == {"generation": "z"}
        operations = root / "operations"
        assert not list(operations.glob("clear-target-*.json"))
        assert not list(operations.glob("[0-9a-f]" * 32 + ".json"))
    finally:
        reopened.close()
        independent_writer.close()


def _retired_scheduler_chunked_clear_reference_survives_crash_and_retires_under_small_limit(
    tmp_path: Path,
) -> None:
    """A valid manifest larger than record policy remains recoverable evidence."""
    root = tmp_path / "chunked-clear-reference"
    limits = LifecycleLimits(
        max_operation_record_bytes=10_000,
        max_operation_field_bytes=8_192,
        manifest_page_size=1,
    )
    config = CacheConfig(cache_dir=str(root), lifecycle_limits=limits)
    store = BlobStore(root, backend="json", config=config)
    try:
        store.put(
            {"value": "large manifest"},
            key="large",
            metadata={"large": "x" * 15_000},
        )
        raw_manifest = store.manifest_repository.get_raw("large")
        assert raw_manifest is not None
        assert len(raw_manifest) > limits.max_operation_record_bytes

        def interrupt_after_snapshot(seam: str, _record: Any) -> None:
            if seam == "clear_snapshot_complete":
                raise _SimulatedProcessLoss("crash after chunked clear snapshot")

        store.lifecycle.fault_hook = interrupt_after_snapshot
        with pytest.raises(_SimulatedProcessLoss):
            store.clear()

        chunks = list(
            (root / "operations").glob("clear-target-reference-*-part-*.json")
        )
        assert len(chunks) > 1
        assert all(path.stat().st_size <= limits.max_operation_record_bytes for path in chunks)
    finally:
        store.close()

    reopened = BlobStore(root, backend="json", config=config)
    try:
        assert reopened.list() == []
        assert not list((root / "operations").glob("clear-target-*.json"))
        assert not list((root / "operations").glob("[0-9a-f]" * 32 + ".json"))
    finally:
        reopened.close()


def _retired_scheduler_clear_empty_store_initializes_authenticated_control_evidence(
    tmp_path: Path,
) -> None:
    """An empty store still clears through authenticated bounded control state."""
    store = BlobStore(tmp_path / "empty-clear", backend="json")
    try:
        assert store.clear() == 0
        assert store.list() == []
    finally:
        store.close()


@pytest.mark.parametrize("backend_name", ("memory", "json", "sqlite"))
def _retired_scheduler_clear_traverses_durable_empty_inventory_bridges(
    tmp_path: Path, backend_name: str
) -> None:
    """Clear reaches current targets after stale-only high-water windows.

    The first page deliberately contains only superseded manifest events.  A
    zero-target signed page must bridge it to the later current targets rather
    than letting continuation from ``None`` mistake the absent first target
    page for a terminal clear.
    """
    root = tmp_path / f"clear-empty-bridge-{backend_name}"
    limits = LifecycleLimits(manifest_page_size=2, max_inventory_items=2)
    config = CacheConfig(cache_dir=str(root), lifecycle_limits=limits)
    backend = InMemoryBackend() if backend_name == "memory" else backend_name
    store = BlobStore(root, backend=backend, config=config)
    try:
        store.put({"generation": 1}, key="k")
        store.put({"generation": 2}, key="k")
        store.put({"generation": 3}, key="k")
        store.put({"generation": "x"}, key="x")

        assert store.clear() == 2
        assert store.get("k") is None
        assert store.get("x") is None
        assert not list((root / "operations").glob("clear-target-*.json"))
    finally:
        store.close()


@pytest.mark.parametrize(
    ("control_name", "chunk_index", "payload"),
    (
        ("clear-target-page-", None, {"value": "page"}),
        ("clear-target-checkpoint-", None, {"value": "checkpoint"}),
        ("clear-target-reference-", 0, {"value": "chunk-zero"}),
        ("clear-target-reference-", 1, {"value": "chunk-one"}),
    ),
)
def _retired_scheduler_partial_clear_control_publish_never_installs_a_poisoned_final_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_name: str,
    chunk_index: int | None,
    payload: dict[str, str],
) -> None:
    """A process-loss write leaves no partial page, checkpoint, or chunk final."""
    root = tmp_path / control_name.rstrip("-")
    limits = LifecycleLimits(
        max_operation_record_bytes=10_000,
        max_operation_field_bytes=8_192,
        manifest_page_size=1,
    )
    config = CacheConfig(cache_dir=str(root), lifecycle_limits=limits)
    store = BlobStore(root, backend="json", config=config)
    file_ops = store.lifecycle.operation_repository.file_ops
    if not file_ops.descriptor_mode:
        pytest.skip("partial-write injection requires descriptor-backed managed I/O")

    armed = False
    original_write_all = file_ops._write_all

    def arm_on_control_create(operation: str, locator: Path) -> None:
        nonlocal armed
        is_target_chunk = (
            chunk_index is None
            or locator.name.endswith(f"-part-{chunk_index:04x}.json")
        )
        if (
            operation == "exclusive_create"
            and control_name in locator.name
            and is_target_chunk
        ):
            armed = True

    def write_prefix_then_lose_process(descriptor: int, data: bytes) -> None:
        if armed:
            os.write(descriptor, data[:1])
            raise _SimulatedProcessLoss(f"lost while writing {control_name}")
        original_write_all(descriptor, data)

    monkeypatch.setattr(file_ops, "before_operation", arm_on_control_create)
    monkeypatch.setattr(file_ops, "_write_all", write_prefix_then_lose_process)
    try:
        store.put(
            payload,
            key="survivor",
            metadata=(
                {"large": "x" * 15_000}
                if control_name == "clear-target-reference-"
                else None
            ),
        )
        with pytest.raises(_SimulatedProcessLoss):
            store.clear()
        assert armed
        expected_partial = (
            f"{control_name}*-part-{chunk_index:04x}.json"
            if chunk_index is not None
            else f"{control_name}*.json"
        )
        assert not list((root / "operations").glob(expected_partial))
    finally:
        store.close()

    reopened = BlobStore(root, backend="json", config=config)
    try:
        assert reopened.get("survivor") == payload
        assert not list((root / "operations").glob("clear-target-*.json"))
        assert not list((root / "operations").glob("[0-9a-f]" * 32 + ".json"))
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "step",
    (
        "control_temp_fsynced",
        "control_rename_completed",
        "control_directory_fsynced",
    ),
)
@pytest.mark.parametrize("control_prefix", ("", "clear-target-"))
def _retired_scheduler_process_loss_at_control_publish_boundaries_leaves_no_operation_residue(
    tmp_path: Path, step: str, control_prefix: str
) -> None:
    """Native control publication recovers operation and clear-sidecar crashes."""
    label = "operation" if not control_prefix else "clear-sidecar"
    root = tmp_path / f"control-process-loss-{label}-{step}"
    context = multiprocessing.get_context("spawn")
    worker = context.Process(
        target=_exit_during_control_durability_step,
        args=(str(root), step, control_prefix),
    )
    worker.start()
    worker.join(timeout=15)
    assert worker.exitcode == 23

    reopened = BlobStore(root, backend="json")
    try:
        operations = root / "operations"
        assert not list(operations.glob("*.tmp"))
        assert not list(operations.glob("[0-9a-f]" * 32 + ".json"))
        assert not list(operations.glob("clear-target-*.json"))
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "relative_locator",
    (
        ".cacheness-lifecycle-admission.lock",
        ".cache_metadata.json.manifest-cas.lock",
        "operations/.clear-resume.lock",
        "operations/.conditional-locks/00.lock",
    ),
)
@pytest.mark.parametrize(
    "step",
    (
        "lock_file_bytes_fsynced",
        "lock_file_directory_fsynced",
        "lock_authority_before_publish",
        "lock_authority_published",
    ),
)
def _retired_scheduler_process_loss_during_lock_authority_publication_converges_without_sidecars(
    tmp_path: Path, relative_locator: str, step: str
) -> None:
    """Lock authority is root-bound and remains reopenable at every crash seam.

    ``os._exit`` models process loss only; it does not claim physical power-loss
    durability. Reopening must accept the fixed lock and either create or read
    its complete root-object binding without direct-written authority sidecars
    or pending control residue.
    """
    root = tmp_path / f"lock-authority-loss-{relative_locator.replace('/', '-')}-{step}"
    root.mkdir()
    context = multiprocessing.get_context("spawn")
    worker = context.Process(
        target=_exit_during_fixed_lock_creation,
        args=(str(root), relative_locator, step),
    )
    worker.start()
    worker.join(timeout=15)
    assert worker.exitcode == 23

    file_ops = ManagedFileOps(root)
    try:
        locator = resolve_managed_locator(
            file_ops.root,
            relative_locator,
            operation="fixed_lock_reopen",
            allow_missing_leaf=True,
        )
        identity = file_ops.ensure_lifecycle_lock(locator)
        assert file_ops.retain_lock_identity(locator) == identity
        assert not list(root.rglob("*.pending.*.tmp"))
        assert not (root / ".cacheness-lock-authorities").exists()
    finally:
        file_ops.close()


def _retired_scheduler_truncated_legacy_lock_sidecar_never_bricks_root_bound_authority(
    tmp_path: Path,
) -> None:
    """Ambiguous legacy sidecar bytes are ignored, not deleted or trusted."""
    root = tmp_path / "legacy-lock-sidecar"
    root.mkdir()
    legacy = root / ".cacheness-lock-authorities" / "truncated.authority"
    legacy.parent.mkdir()
    legacy.write_bytes(b"partial")
    file_ops = ManagedFileOps(root)
    try:
        locator = resolve_managed_locator(
            file_ops.root,
            "authority.lock",
            operation="legacy_lock_authority",
            allow_missing_leaf=True,
        )
        identity = file_ops.ensure_lifecycle_lock(locator)
        assert file_ops.retain_lock_identity(locator) == identity
        assert legacy.read_bytes() == b"partial"
    finally:
        file_ops.close()


def _retired_scheduler_lifecycle_payload_cleanup_uses_durable_delete_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Overwrite and delete reach the same durable cleanup authority path."""
    store = BlobStore(tmp_path / "durable-lifecycle-delete", backend="json")
    try:
        store.put({"generation": 1}, key="durable-key")
        file_ops = store.guarded_handler_io.file_ops
        original = file_ops.delete_durable
        calls: list[Path] = []

        def durable_delete(locator: Path) -> bool:
            calls.append(locator)
            return original(locator)

        monkeypatch.setattr(file_ops, "delete_durable", durable_delete)
        store.put({"generation": 2}, key="durable-key")
        assert store.delete("durable-key")
        assert len(calls) >= 2
    finally:
        store.close()


def _retired_scheduler_clear_lifecycle_payload_cleanup_uses_durable_delete_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clear delegates each tombstone payload reclamation to durable deletion."""
    store = BlobStore(tmp_path / "durable-clear-delete", backend="json")
    try:
        store.put({"generation": 1}, key="clear-one")
        store.put({"generation": 2}, key="clear-two")
        file_ops = store.guarded_handler_io.file_ops
        original = file_ops.delete_durable
        calls: list[Path] = []

        def durable_delete(locator: Path) -> bool:
            calls.append(locator)
            return original(locator)

        monkeypatch.setattr(file_ops, "delete_durable", durable_delete)
        assert store.clear() == 2
        assert len(calls) >= 2
    finally:
        store.close()


def _retired_scheduler_recovered_tombstone_preserves_conflict_when_evidence_retirement_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later winner remains explicit when reopen cleanup cannot retire debt."""
    store = BlobStore(tmp_path / "recover-tombstone-conflict", backend="json")
    try:
        store.put({"value": "delete"}, key="tombstone-key")
        original_cleanup = store._delete_or_prove_absent

        def defer_cleanup(_locator: Path) -> None:
            raise OSError("defer")

        monkeypatch.setattr(store, "_delete_or_prove_absent", defer_cleanup)
        with pytest.raises(CacheBlobRecoverableCleanupError):
            store.delete("tombstone-key")
        monkeypatch.setattr(store, "_delete_or_prove_absent", original_cleanup)

        operation_id, raw = next(iter(store.lifecycle.operation_repository.list_page().entries))
        recovered = store.lifecycle._recoverable_record(operation_id, raw)
        assert recovered is not None
        record, candidate, _previous = recovered
        conflict = CacheBlobLifecycleConflictError("later winner")
        monkeypatch.setattr(
            store.manifest_repository,
            "remove_if_expected",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(conflict),
        )
        monkeypatch.setattr(
            store.lifecycle,
            "_retire",
            lambda _record: (_ for _ in ()).throw(OSError("retire failed")),
        )

        with pytest.raises(CacheBlobRecoverableCleanupError) as error:
            store.lifecycle._recover_tombstone(record, candidate)
        assert error.value.context["post_authority"] is True
        assert error.value.context["later_winner_preserved"] is True
        assert error.value.context["conflict_type"] == "CacheBlobLifecycleConflictError"
        assert error.value.__cause__ is conflict
    finally:
        store.close()
