"""Tests for write intent journal (crash-safe blob writes)."""

import json
import os
import time
from pathlib import Path

import pytest

from cacheness.config import (
    CacheConfig,
    CacheMetadataConfig,
    CacheStorageConfig,
    CompressionConfig,
)
from cacheness.core import UnifiedCache as cacheness
from cacheness.write_intent import WriteIntentJournal


def _make_cache(tmp_path, **storage_kwargs):
    """Create a cache instance with optional storage config overrides."""
    storage = CacheStorageConfig(cache_dir=str(tmp_path), **storage_kwargs)
    config = CacheConfig(
        storage=storage,
        metadata=CacheMetadataConfig(metadata_backend="sqlite"),
        compression=CompressionConfig(use_blosc2_arrays=False),
    )
    return cacheness(config)


class TestWriteIntentJournal:
    """Unit tests for the WriteIntentJournal class."""

    def test_record_and_clear_intent(self, tmp_path):
        journal = WriteIntentJournal(tmp_path)
        intent_path = journal.record_intent("key1", "/some/blob/path.pkl")
        assert intent_path.exists()

        journal.clear_intent("key1")
        assert not intent_path.exists()

    def test_clear_nonexistent_intent(self, tmp_path):
        journal = WriteIntentJournal(tmp_path)
        # Should not raise
        journal.clear_intent("never_recorded_key")

    def test_cleanup_stale_intents(self, tmp_path):
        journal = WriteIntentJournal(tmp_path, stale_threshold_seconds=0)

        # Create a fake blob file
        blob_file = tmp_path / "fake_blob.pkl"
        blob_file.write_bytes(b"fake data")

        # Record intent pointing to the fake blob
        journal.record_intent("stale_key", str(blob_file))

        # Small delay so the intent ages past the 0-second threshold
        time.sleep(0.05)

        # Cleanup should find and remove the stale intent + orphaned blob
        cleaned = journal.cleanup_stale_intents()
        assert cleaned == 1
        assert not blob_file.exists()

    def test_cleanup_skips_fresh_intents(self, tmp_path):
        journal = WriteIntentJournal(tmp_path, stale_threshold_seconds=3600)
        intent_path = journal.record_intent("fresh_key", "/some/path.pkl")

        cleaned = journal.cleanup_stale_intents()
        assert cleaned == 0
        assert intent_path.exists()

    def test_intent_dir_created_lazily(self, tmp_path):
        journal = WriteIntentJournal(tmp_path)
        intents_dir = tmp_path / ".intents"
        assert not intents_dir.exists()

        journal.record_intent("key1", "/some/path")
        assert intents_dir.exists()

    def test_record_intent_fsyncs_when_enabled(self, tmp_path, monkeypatch):
        calls = []

        def record_fsync(fd):
            calls.append(fd)

        monkeypatch.setattr(os, "fsync", record_fsync)
        journal = WriteIntentJournal(tmp_path, fsync_on_write=True)

        intent_path = journal.record_intent("key1", "/some/blob/path.pkl")

        assert intent_path.exists()
        assert calls

    def test_record_intent_does_not_fsync_by_default(self, tmp_path, monkeypatch):
        calls = []

        def record_fsync(fd):
            calls.append(fd)

        monkeypatch.setattr(os, "fsync", record_fsync)
        journal = WriteIntentJournal(tmp_path)

        journal.record_intent("key1", "/some/blob/path.pkl")

        assert calls == []

    def test_cleanup_handles_missing_blob(self, tmp_path):
        """Stale intent whose blob was already deleted — should still clean intent file."""
        journal = WriteIntentJournal(tmp_path, stale_threshold_seconds=0)
        journal.record_intent("orphan_key", "/nonexistent/path.pkl")
        time.sleep(0.05)

        cleaned = journal.cleanup_stale_intents()
        assert cleaned == 1

    def test_cleanup_resolves_relative_blob_path_against_cache_dir(
        self, tmp_path, monkeypatch
    ):
        """Relative intent blob paths are cache-dir relative, not CWD relative."""
        journal = WriteIntentJournal(tmp_path, stale_threshold_seconds=0)
        blob_file = tmp_path / "default" / "relative_blob.pkl"
        blob_file.parent.mkdir()
        blob_file.write_bytes(b"orphaned data")

        intent_path = journal.record_intent("relative_key", "default/relative_blob.pkl")
        monkeypatch.chdir(tmp_path.parent)
        time.sleep(0.05)

        cleaned = journal.cleanup_stale_intents()

        assert cleaned == 1
        assert not blob_file.exists()
        assert not intent_path.exists()

    def test_cleanup_preserves_blob_when_entry_exists(self, tmp_path):
        """Crash after metadata commit should clear intent without deleting the blob."""
        journal = WriteIntentJournal(tmp_path, stale_threshold_seconds=0)
        blob_file = tmp_path / "default" / "committed_blob.pkl"
        blob_file.parent.mkdir()
        blob_file.write_bytes(b"committed data")

        intent_path = journal.record_intent("committed_key", str(blob_file))
        time.sleep(0.05)

        cleaned = journal.cleanup_stale_intents(
            entry_exists=lambda key: key == "committed_key"
        )

        assert cleaned == 1
        assert blob_file.exists()
        assert not intent_path.exists()

    def test_cleanup_preserves_committed_blob_when_planned_blob_missing(self, tmp_path):
        """Failure before blob creation clears only the stale intent for committed keys."""
        journal = WriteIntentJournal(tmp_path, stale_threshold_seconds=0)
        committed_blob = tmp_path / "default" / "committed_blob.pkl"
        committed_blob.parent.mkdir()
        committed_blob.write_bytes(b"committed data")

        missing_planned_blob = tmp_path / "default" / "planned_blob.pkl"
        intent_path = journal.record_intent(
            "committed_key",
            str(missing_planned_blob.relative_to(tmp_path)),
        )
        time.sleep(0.05)

        cleaned = journal.cleanup_stale_intents(
            entry_exists=lambda key: key == "committed_key"
        )

        assert cleaned == 1
        assert committed_blob.exists()
        assert not missing_planned_blob.exists()
        assert not intent_path.exists()


class TestWriteIntentIntegration:
    """Integration tests with UnifiedCache."""

    def test_normal_put_leaves_no_intents(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache.put("hello world", cache_key="test-key")

        intents_dir = tmp_path / ".intents"
        if intents_dir.exists():
            intent_files = list(intents_dir.glob("*.intent"))
            assert len(intent_files) == 0, f"Residual intent files: {intent_files}"

    def test_multiple_puts_leave_no_intents(self, tmp_path):
        cache = _make_cache(tmp_path)
        for i in range(10):
            cache.put(f"value_{i}", cache_key=f"key_{i}")

        intents_dir = tmp_path / ".intents"
        if intents_dir.exists():
            intent_files = list(intents_dir.glob("*.intent"))
            assert len(intent_files) == 0

    def test_simulated_crash_leaves_intent(self, tmp_path):
        """Simulate crash: leave an intent + orphan blob, then re-init to clean up."""
        cache = _make_cache(tmp_path, stale_intent_threshold_seconds=0)

        # Create a fake orphaned blob
        blob_file = tmp_path / "orphan_blob.pkl"
        blob_file.write_bytes(b"orphaned data from crashed write")

        # Manually record a stale intent (simulating crash between blob write and metadata commit)
        cache._write_journal.record_intent("crashed_key", str(blob_file))
        time.sleep(0.05)

        # Re-init cache — cleanup_on_init should clean up the stale intent
        cache2 = _make_cache(tmp_path, stale_intent_threshold_seconds=0)

        assert not blob_file.exists(), "Orphaned blob should have been deleted"
        intents_dir = tmp_path / ".intents"
        intent_files = list(intents_dir.glob("*.intent"))
        assert len(intent_files) == 0, "Stale intent file should have been removed"

    def test_configurable_threshold(self, tmp_path):
        """Threshold=0 means immediate cleanup; threshold=3600 means skip fresh intents."""
        cache = _make_cache(tmp_path, stale_intent_threshold_seconds=3600)

        blob_file = tmp_path / "fresh_blob.pkl"
        blob_file.write_bytes(b"still fresh")

        cache._write_journal.record_intent("fresh_key", str(blob_file))

        # Re-init with high threshold — intent should survive
        cache2 = _make_cache(tmp_path, stale_intent_threshold_seconds=3600)
        assert blob_file.exists(), "Fresh blob should NOT be deleted"

        intents_dir = tmp_path / ".intents"
        intent_files = list(intents_dir.glob("*.intent"))
        assert len(intent_files) == 1, "Fresh intent should survive cleanup"

    def test_inline_put_skips_intent(self, tmp_path):
        """Inline writes produce no blob on disk, so no intent is needed."""
        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            metadata=CacheMetadataConfig(metadata_backend="sqlite"),
            compression=CompressionConfig(use_blosc2_arrays=False),
        )
        cache = cacheness(config)

        # Small string value — should be inlined
        cache.put("tiny", cache_key="inline-test")

        intents_dir = tmp_path / ".intents"
        if intents_dir.exists():
            intent_files = list(intents_dir.glob("*.intent"))
            assert len(intent_files) == 0

    def test_cache_mode_records_intent_before_blob_write(self, tmp_path, monkeypatch):
        """Non-inline cache writes record an intent before blob I/O starts."""
        cache = _make_cache(tmp_path)
        intents_dir = tmp_path / ".intents"
        data = {"payload": "not-inline"}
        cache_key = "pre-write-cache"
        handler = cache._blob_store.handlers.get_handler(data)
        expected_blob_path = cache._get_cache_file_path(cache_key).with_suffix(
            handler.get_file_extension(cache.config)
        )
        expected_relative_path = str(expected_blob_path.relative_to(tmp_path))

        def fail_after_check(*args, **kwargs):
            intent_files = list(intents_dir.glob("*.intent"))
            assert len(intent_files) == 1
            payload = json.loads(intent_files[0].read_text(encoding="utf-8"))
            assert payload["cache_key"] == cache_key
            assert payload["blob_path"] == expected_relative_path
            assert not expected_blob_path.exists()
            raise RuntimeError("blob write interrupted")

        monkeypatch.setattr(cache._blob_store, "_write_blob", fail_after_check)

        with pytest.raises(RuntimeError, match="blob write interrupted"):
            cache.put(data, cache_key=cache_key)

        assert not list(intents_dir.glob("*.intent"))

    def test_overwrite_clears_intent(self, tmp_path):
        """Overwriting an existing key should not leave stale intents."""
        cache = _make_cache(tmp_path)
        cache.put("original", cache_key="overwrite-key")
        cache.put("updated", cache_key="overwrite-key")

        intents_dir = tmp_path / ".intents"
        if intents_dir.exists():
            intent_files = list(intents_dir.glob("*.intent"))
            assert len(intent_files) == 0


class TestStorageModeWriteIntentCleanup:
    """Storage mode still performs conservative stale-intent cleanup."""

    def _make_storage_cache(self, tmp_path):
        config = CacheConfig(
            storage=CacheStorageConfig(
                cache_dir=str(tmp_path),
                stale_intent_threshold_seconds=0,
            ),
            metadata=CacheMetadataConfig(metadata_backend="sqlite"),
            compression=CompressionConfig(use_blosc2_arrays=False),
            storage_mode=True,
        )
        return cacheness(config)

    def test_storage_mode_preserves_committed_entry_and_removes_intent(self, tmp_path):
        cache = self._make_storage_cache(tmp_path)
        cache.put("durable", cache_key="committed-key")

        entry = cache.metadata_backend.get_entry("committed-key")
        assert entry is not None
        blob_path = cache._resolve_actual_path(entry["metadata"]["actual_path"])
        intent_path = cache._write_journal.record_intent(
            "committed-key", entry["metadata"]["actual_path"]
        )
        time.sleep(0.05)

        reopened = self._make_storage_cache(tmp_path)

        assert reopened.get("committed-key") == "durable"
        assert blob_path.exists()
        assert not intent_path.exists()

    def test_storage_mode_removes_uncommitted_orphan_blob_and_intent(self, tmp_path):
        cache = self._make_storage_cache(tmp_path)
        blob_file = tmp_path / "default" / "orphan.pkl"
        blob_file.parent.mkdir(exist_ok=True)
        blob_file.write_bytes(b"orphaned")
        intent_path = cache._write_journal.record_intent(
            "orphan-key", "default/orphan.pkl"
        )
        time.sleep(0.05)

        self._make_storage_cache(tmp_path)

        assert not blob_file.exists()
        assert not intent_path.exists()

    def test_storage_mode_records_intent_before_blob_write(self, tmp_path, monkeypatch):
        cache = self._make_storage_cache(tmp_path)
        intents_dir = tmp_path / ".intents"
        data = {"payload": "not-inline"}
        cache_key = "pre-write-storage"
        handler = cache._blob_store.handlers.get_handler(data)
        expected_blob_path = cache._get_cache_file_path(cache_key).with_suffix(
            handler.get_file_extension(cache.config)
        )
        expected_relative_path = str(expected_blob_path.relative_to(tmp_path))

        def fail_after_check(*args, **kwargs):
            intent_files = list(intents_dir.glob("*.intent"))
            assert len(intent_files) == 1
            payload = json.loads(intent_files[0].read_text(encoding="utf-8"))
            assert payload["cache_key"] == cache_key
            assert payload["blob_path"] == expected_relative_path
            assert not expected_blob_path.exists()
            raise RuntimeError("blob write interrupted")

        monkeypatch.setattr(cache._blob_store, "_write_blob", fail_after_check)

        with pytest.raises(RuntimeError, match="blob write interrupted"):
            cache.put(data, cache_key=cache_key)

        assert not list(intents_dir.glob("*.intent"))

    def test_storage_mode_failed_pre_blob_overwrite_keeps_committed_data(
        self, tmp_path, monkeypatch
    ):
        cache = self._make_storage_cache(tmp_path)
        cache_key = "pre-blob-overwrite"
        cache.put({"payload": "committed"}, cache_key=cache_key)
        entry = cache.metadata_backend.get_entry(cache_key)
        assert entry is not None
        committed_blob = cache._resolve_actual_path(entry["metadata"]["actual_path"])
        assert committed_blob.exists()

        def fail_before_blob_created(*args, **kwargs):
            assert list((tmp_path / ".intents").glob("*.intent"))
            raise RuntimeError("handler failed before blob creation")

        monkeypatch.setattr(cache._blob_store, "_write_blob", fail_before_blob_created)

        with pytest.raises(RuntimeError, match="handler failed before blob creation"):
            cache.put({"payload": "replacement"}, cache_key=cache_key)

        assert cache.get(cache_key=cache_key) == {"payload": "committed"}
        assert committed_blob.exists()
        assert not list((tmp_path / ".intents").glob("*.intent"))
