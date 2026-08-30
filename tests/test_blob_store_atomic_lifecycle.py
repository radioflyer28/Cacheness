"""End-to-end tracer coverage for BlobStore's immutable write lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cacheness.storage import BlobStore


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
