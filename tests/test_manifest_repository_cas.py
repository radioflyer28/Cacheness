"""Derived JSON-projection tests for authority-owned metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cacheness.error_handling import CacheBlobLifecycleConflictError
from cacheness.storage import BlobStore
from cacheness.storage import manifest_repository as manifest_repository_module
from cacheness.storage.manifest_repository import JsonProjectionExporter


def test_projection_export_streams_authority_backup_and_marks_revision_clean(
    tmp_path: Path,
) -> None:
    """Committed authority rows rebuild the JSON projection without live reads."""
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
        assert document["entries"][key] == store.get_metadata(key)
        assert document["entries"][key]["metadata"]["tag"] == "v1"
        assert document["entries"][key]["metadata"]["actual_path"].startswith(
            "generations/"
        )
    finally:
        store.close()


def test_projection_export_failure_keeps_committed_authority_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A derived-output failure cannot roll back committed authority state."""
    root = tmp_path / "projection-failure"
    store = BlobStore(root, backend="json")
    try:
        authority = store.lifecycle_authority
        original_replace = manifest_repository_module.os.replace

        def fail_projection_publish(source: object, destination: object) -> None:
            if Path(destination) == root / "cache_metadata.json":
                raise OSError("projection publish failed")
            original_replace(source, destination)

        monkeypatch.setattr(
            manifest_repository_module.os, "replace", fail_projection_publish
        )
        key = store.put("projection payload", key="projection-key")
        assert authority.snapshot_state().projection_dirty is True
        assert store.get(key) == "projection payload"
    finally:
        store.close()


def test_stale_projection_export_cannot_mark_newer_authority_revision_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale exporter does not make a later authority revision appear clean."""
    root = tmp_path / "projection-race"
    store = BlobStore(root, backend="json")
    try:
        store.put("first", key="first")
        authority = store.lifecycle_authority
        captured_revision = authority.snapshot_state().revision
        exporter = JsonProjectionExporter(authority, root / "cache_metadata.json")
        original_write = exporter._write_snapshot

        def advance_after_snapshot(snapshot_path: Path, revision: object) -> Path:
            candidate = original_write(snapshot_path, revision)
            assert authority.open_write_transactions == 0
            store.put("second", key="second")
            return candidate

        monkeypatch.setattr(exporter, "_write_snapshot", advance_after_snapshot)
        with pytest.raises(CacheBlobLifecycleConflictError, match="Projection revision changed"):
            exporter.export()

        document = json.loads((root / "cache_metadata.json").read_text(encoding="utf-8"))
        assert document["_cacheness_authority_revision"] == captured_revision + 1
        assert set(document["entries"]) == {"first", "second"}
        assert authority.snapshot_state().projection_dirty is False
        assert store.get("second") == "second"
    finally:
        store.close()
