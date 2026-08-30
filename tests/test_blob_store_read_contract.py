"""Tracer coverage for BlobStore's canonical committed read contract."""

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from cacheness.error_handling import (
    CacheBlobLifecycleConflictError,
    CacheBlobManifestUnauthenticatedError,
    CacheBlobPayloadTamperedError,
)
from cacheness.storage import BlobStore
from cacheness.storage.integrity import sign_hmac_sha256
from cacheness.storage.manifest import BlobManifestV1


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


def _replace_signed_manifest(
    store: BlobStore, key: str, **changes: Any
) -> BlobManifestV1:
    """Replace one test manifest while preserving its canonical signature."""
    raw_manifest = store.manifest_repository.get_raw(key)
    assert raw_manifest is not None
    manifest = BlobManifestV1.from_canonical_bytes(raw_manifest)
    changed_manifest = replace(manifest, **changes)
    signed_manifest = changed_manifest.with_signature(
        sign_hmac_sha256(changed_manifest.signing_bytes(), store._manifest_key())
    )
    store.manifest_repository.put_raw(key, signed_manifest.canonical_bytes())
    return signed_manifest


@pytest.mark.parametrize("backend_name", ("json", "sqlite"))
def test_tracer_direct_blob_store_uses_one_authenticated_committed_manifest(
    tmp_path, monkeypatch, backend_name
):
    """A direct put/get signs the raw record and verifies one snapshot first."""
    events: list[str] = []
    store = BlobStore(tmp_path / "tracer", backend=backend_name)
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


def test_get_metadata_authenticates_committed_manifest_without_payload_io(
    tmp_path, monkeypatch
):
    """Metadata reads use signed canonical truth but never open a payload."""
    events: list[str] = []
    store = BlobStore(tmp_path / "metadata", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler(events))

    try:
        key = store.put("metadata payload", key="metadata-key", metadata={"tag": "v1"})
        monkeypatch.setattr(
            store.guarded_handler_io,
            "open_snapshot",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("get_metadata must not open a payload snapshot")
            ),
        )
        monkeypatch.setattr(
            store.backend,
            "update_access_time",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("get_metadata must not update access state")
            ),
        )

        metadata = store.get_metadata(key)

        assert metadata is not None
        assert metadata["cache_key"] == key
        assert metadata["data_type"] == "tracing_object"
        assert metadata["metadata"]["tag"] == "v1"
        assert events == []

        raw_manifest = store.manifest_repository.get_raw(key)
        assert raw_manifest is not None
        tampered = json.loads(raw_manifest)
        tampered["signature"] = "0" * 64
        store.manifest_repository.put_raw(
            key,
            json.dumps(
                tampered, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8"),
        )

        with pytest.raises(CacheBlobManifestUnauthenticatedError):
            store.get_metadata(key)
        assert events == []
    finally:
        store.close()


def test_exists_verifies_one_snapshot_without_deserializing(tmp_path, monkeypatch):
    """Existence means a valid authenticated manifest and intact payload."""
    events: list[str] = []
    store = BlobStore(tmp_path / "exists", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler(events))

    try:
        key = store.put("exists payload", key="exists-key")
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

        assert store.exists(key) is True
        assert events == ["snapshot", "digest"]

        manifest = BlobManifestV1.from_canonical_bytes(
            store.manifest_repository.get_raw(key) or b""
        )
        Path(manifest.locator).write_text("tamper! payload", encoding="utf-8")
        events.clear()

        with pytest.raises(CacheBlobPayloadTamperedError):
            store.exists(key)
        assert events == ["snapshot", "digest"]
        assert "handler" not in events
    finally:
        store.close()


def test_list_authenticates_every_selected_manifest_before_returning(tmp_path):
    """A conflicted selected record cannot be silently omitted from listing."""
    events: list[str] = []
    store = BlobStore(tmp_path / "list", backend="json")
    store.handlers = _SingleHandlerRegistry(_TracingHandler(events))

    try:
        first = store.put("first", key="selected-first", metadata={"group": "one"})
        second = store.put("second", key="selected-second", metadata={"group": "one"})
        _replace_signed_manifest(store, second, state="prepared")

        with pytest.raises(CacheBlobLifecycleConflictError):
            store.list(prefix="selected-", metadata_filter={"group": "one"})

        assert events == []
        assert first == "selected-first"
    finally:
        store.close()
