"""Tracer coverage for BlobStore's canonical committed read contract."""

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

from cacheness.storage import BlobStore


class _TracingHandler:
    """Small native handler that makes read ordering observable."""

    data_type = "tracing_object"

    def __init__(self, events: list[str]):
        self.events = events

    def put(self, data: Any, file_path: Path, _config: Any) -> dict[str, Any]:
        payload_path = file_path.with_suffix(".trace")
        payload_path.write_text(str(data), encoding="utf-8")
        return {
            "storage_format": "trace",
            "file_size": payload_path.stat().st_size,
            "actual_path": str(payload_path),
            "metadata": {"storage_format": "trace"},
        }

    def get(self, file_path: Path, _metadata: dict[str, Any]) -> str:
        self.events.append("handler")
        return file_path.read_text(encoding="utf-8")


class _SingleHandlerRegistry:
    """Use one simple handler for the tracer's real storage path."""

    def __init__(self, handler: _TracingHandler):
        self.handler = handler

    def get_handler(self, _data: Any) -> _TracingHandler:
        return self.handler

    def get_handler_by_type(self, data_type: str) -> _TracingHandler:
        assert data_type == self.handler.data_type
        return self.handler


def test_tracer_direct_blob_store_uses_one_authenticated_committed_manifest(
    tmp_path, monkeypatch
):
    """A direct put/get signs the raw record and verifies one snapshot first."""
    events: list[str] = []
    store = BlobStore(tmp_path / "tracer", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler(events))

    try:
        key = store.put("tracer payload", key="tracer-key")

        raw_record = store.manifest_repository.get_raw(key)
        assert raw_record is not None
        manifest = json.loads(raw_record)
        assert manifest["schema_version"] == 1
        assert manifest["payload_format_version"] == 1
        assert manifest["state"] == "committed"
        assert manifest["key"] == key
        assert manifest["signature_algorithm"] == "hmac-sha256"
        assert manifest["signature"]

        original_get_raw = store.manifest_repository.get_raw

        def get_raw_with_event(blob_key: str):
            events.append("repository")
            return original_get_raw(blob_key)

        monkeypatch.setattr(store.manifest_repository, "get_raw", get_raw_with_event)

        original_snapshot = store.guarded_handler_io.open_snapshot

        @contextmanager
        def snapshot_with_event(locator, metadata):
            events.append("snapshot")
            with original_snapshot(locator, metadata) as snapshot:
                yield snapshot

        monkeypatch.setattr(
            store.guarded_handler_io, "open_snapshot", snapshot_with_event
        )

        from cacheness.storage import blob_store as blob_store_module

        original_digest = blob_store_module.sha256_and_size

        def digest_with_event(path):
            events.append("digest")
            return original_digest(path)

        monkeypatch.setattr(blob_store_module, "sha256_and_size", digest_with_event)

        events.clear()
        assert store.get(key) == "tracer payload"
        assert events == ["repository", "snapshot", "digest", "handler"]
    finally:
        store.close()


def test_absent_raw_record_returns_none_without_snapshot_or_handler(tmp_path, monkeypatch):
    """Repository absence is the only direct-read outcome represented as None."""
    events: list[str] = []
    store = BlobStore(tmp_path / "absent", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler(events))

    try:
        monkeypatch.setattr(
            store.guarded_handler_io,
            "open_snapshot",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("an absent record must not open a snapshot")
            ),
        )
        assert store.get("absent-key") is None
        assert events == []
    finally:
        store.close()
