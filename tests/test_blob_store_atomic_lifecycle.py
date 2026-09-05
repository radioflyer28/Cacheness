"""Public BlobStore tests for the authority-owned immutable lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cacheness.error_handling import CacheBlobRecoverableCleanupError
from cacheness.storage import BlobStore


class _NativeJsonHandler:
    """Small handler whose payload bytes remain independently observable."""

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
    """Make selection of the public test handler deterministic."""

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
    """Model a crash that skips ordinary exception cleanup."""


class _FailingSerializationHandler(_NativeJsonHandler):
    """Prove native serialization precedes authority mutation evidence."""

    def put(self, data: Any, file_path: Path, config: Any) -> dict[str, Any]:
        del data, file_path, config
        self.events.append("private_serialization")
        raise RuntimeError("native serialization failed")


def test_put_promotes_immutable_generation_through_lifecycle_authority(
    tmp_path: Path,
) -> None:
    """Writes replace a generation through the one durable authority."""
    events: list[str] = []
    store = BlobStore(tmp_path / "store", backend="json")
    store.handlers = _SingleHandlerRegistry(_NativeJsonHandler(events))
    lifecycle_events: list[str] = []
    store.lifecycle.test_hook = lifecycle_events.append
    try:
        key = store.put({"generation": 1}, key="authority-key")
        first = store.get_metadata(key)
        assert first is not None
        first_locator = store.cache_dir / first["metadata"]["actual_path"]

        assert store.put({"generation": 2}, key=key) == key
        second = store.get_metadata(key)
        assert second is not None
        second_locator = store.cache_dir / second["metadata"]["actual_path"]
        assert second_locator != first_locator
        assert second_locator.parent.parent.name == "generations"
        assert not first_locator.exists()
        assert store.get(key) == {"generation": 2}
        assert (store.cache_dir / ".cacheness" / "lifecycle-authority-v1.sqlite3").is_file()
        assert not (store.cache_dir / "operations").exists()
        assert lifecycle_events[:6] == [
            "put.intent_prepared",
            "put.before_candidate_publish",
            "put.candidate_published",
            "put.candidate_verified",
            "put.before_promotion",
            "put.promoted",
        ]
        assert events == ["private_serialization", "private_serialization", "handler_read"]
    finally:
        store.close()


def test_clear_preserves_post_snapshot_writes(tmp_path: Path) -> None:
    """Clear removes only exact generations captured by the authority snapshot."""
    store = BlobStore(tmp_path / "clear-snapshot", backend="json")
    try:
        store.put({"generation": "old"}, key="existing")

        def write_after_snapshot(boundary: str) -> None:
            if boundary == "clear.snapshot_committed":
                store.put({"generation": "new"}, key="existing")
                store.put({"generation": "late"}, key="late")

        store.lifecycle.test_hook = write_after_snapshot
        store.clear()
        assert store.get("existing") == {"generation": "new"}
        assert store.get("late") == {"generation": "late"}
    finally:
        store.close()


def test_clear_resumes_after_payload_delete_before_progress_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LifecycleAuthority proves an already-deleted clear target on restart."""
    store = BlobStore(tmp_path / "clear-resume", backend="json")
    deletions: list[Path] = []
    original_delete = store._delete_or_prove_absent

    def observe_delete(locator: Path) -> None:
        deletions.append(locator)
        original_delete(locator)

    monkeypatch.setattr(store, "_delete_or_prove_absent", observe_delete)
    try:
        store.put({"generation": "only"}, key="resume-key")

        def interrupt_after_delete(boundary: str) -> None:
            if boundary == "clear.after_target_delete":
                raise _SimulatedProcessLoss("interrupted after exact target deletion")

        store.lifecycle.test_hook = interrupt_after_delete
        with pytest.raises(_SimulatedProcessLoss):
            store.clear()
        assert len(deletions) == 1

        store.lifecycle.test_hook = None
        assert store.clear() == 0
        assert len(deletions) == 1
        assert store.get("resume-key") is None
    finally:
        store.close()


def test_failed_serialization_leaves_no_authority_entry_or_candidate(tmp_path: Path) -> None:
    """A handler failure cannot create a committed authority entry."""
    events: list[str] = []
    root = tmp_path / "serialization-failure"
    store = BlobStore(root, backend="json")
    store.handlers = _SingleHandlerRegistry(_FailingSerializationHandler(events))
    try:
        with pytest.raises(RuntimeError, match="native serialization failed"):
            store.put({"value": "never-published"}, key="failure-key")
        assert events == ["private_serialization"]
        assert not list(root.glob("*generation-*"))
        assert store.lifecycle_authority.read_entry("failure-key") is None
        assert store.get_metadata("failure-key") is None
    finally:
        store.close()


def test_reconciliation_reclaims_only_old_cleanup_debt_after_new_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reconciliation cannot let old cleanup debt revoke a newer generation."""
    root = tmp_path / "new-winner"
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

        assert store.reconcile(apply=True).applied is True
        assert not old_locator.exists()
        assert store.get(key) == {"generation": "new"}
    finally:
        store.close()
