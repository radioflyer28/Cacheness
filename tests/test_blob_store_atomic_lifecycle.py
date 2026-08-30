"""End-to-end tracer coverage for BlobStore's immutable write lifecycle."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from cacheness import CacheConfig
from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobRecoverableCleanupError,
)
from cacheness.storage import BlobStore
from cacheness.storage.manifest import BlobManifestV1
from cacheness.storage.operation_record import (
    ClearTarget,
    ClearTargetCheckpoint,
    ClearTargetPage,
)
from cacheness.storage.operation_repository import FileOperationRecordRepository


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
    store.lifecycle.test_hook = lambda step, _record: lifecycle_events.append(step)

    try:
        key = store.put({"generation": 1}, key="tracer-key")
        first_manifest = store._load_authenticated_manifest(
            key,
            operation="test",
            require_locator=True,
        )
        assert first_manifest is not None
        first_locator = first_manifest[2]
        assert first_locator is not None
        assert json.loads(first_locator.read_text(encoding="utf-8")) == {"generation": 1}

        assert store.put({"generation": 2}, key=key) == key
        second_manifest = store._load_authenticated_manifest(
            key,
            operation="test",
            require_locator=True,
        )
        assert second_manifest is not None
        second_locator = second_manifest[2]
        assert second_locator is not None
        assert second_locator != first_locator
        assert second_manifest[0].state == "committed"
        assert "generation" in second_locator.name
        assert not first_locator.exists()
        assert store.get(key) == {"generation": 2}
        assert not list((store.cache_dir / "operations").glob("*.json"))
        assert lifecycle_events == [
            "evidence_created",
            "candidate_published",
            "candidate_verified",
            "authority_published",
            "cleanup_completed",
            "evidence_retired",
            "evidence_created",
            "candidate_published",
            "candidate_verified",
            "authority_published",
            "cleanup_completed",
            "evidence_retired",
        ]
        assert events == [
            "private_serialization",
            "private_serialization",
            "handler_read",
        ]
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
def test_lifecycle_failure_boundary_reopens_to_one_complete_generation(
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


def test_cleanup_failure_keeps_new_authority_and_resumes_after_reopen(
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
        generation_payloads = list(root.glob("*generation-*"))
        assert len(generation_payloads) == 1
    finally:
        contender.close()
        winner.close()


def test_delete_publishes_signed_tombstone_before_payload_reclamation(
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


def test_repeated_delete_resumes_the_same_signed_tombstone(
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


def test_stale_delete_conflict_preserves_newer_committed_generation(
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


def test_clear_target_page_and_checkpoint_preserve_exact_progress_after_reopen(
    tmp_path: Path,
) -> None:
    """Clear evidence persists exact targets and rejects stale progress writers."""
    root = tmp_path / "clear-target-evidence"
    store = BlobStore(root, backend="json")
    operation_id = "a" * 32
    page_id = "b" * 32
    raw_first = b'{"first":"exact"}'
    raw_second = b'{"second":"exact"}'
    page = ClearTargetPage(
        operation_id=operation_id,
        page_id=page_id,
        source_cursor=None,
        next_cursor="second",
        targets=(
            ClearTarget.from_raw("first", "c" * 32, raw_first),
            ClearTarget.from_raw("second", "d" * 32, raw_second),
        ),
    )
    repository = FileOperationRecordRepository(
        store.guarded_handler_io.file_ops,
        lifecycle_limits=store.lifecycle_limits,
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


def test_clear_snapshot_barrier_preserves_a_later_key_after_bounded_admission(
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


def test_clear_conflict_does_not_revoke_a_later_generation(tmp_path: Path) -> None:
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


def test_clear_resume_does_not_repeat_completed_targets_after_reopen(
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
    try:
        store.put({"generation": "first"}, key="first")
        store.put({"generation": "second"}, key="second")

        def interrupt_after_checkpoint(seam: str, _record: Any) -> None:
            nonlocal completed_once
            if seam == "clear_target_checkpoint" and not completed_once:
                completed_once = True
                raise _SimulatedProcessLoss("interrupted after durable clear progress")

        store.lifecycle.fault_hook = interrupt_after_checkpoint
        with pytest.raises(_SimulatedProcessLoss):
            store.clear()
    finally:
        store.close()

    reopened = BlobStore(root, backend="json")
    try:
        assert completed_once
        assert reopened.get("first") is None
        assert reopened.get("second") is None
    finally:
        reopened.close()
