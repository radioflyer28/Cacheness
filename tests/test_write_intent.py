"""Tests for write intent journal (crash-safe blob writes)."""

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

    def test_cleanup_handles_missing_blob(self, tmp_path):
        """Stale intent whose blob was already deleted — should still clean intent file."""
        journal = WriteIntentJournal(tmp_path, stale_threshold_seconds=0)
        journal.record_intent("orphan_key", "/nonexistent/path.pkl")
        time.sleep(0.05)

        cleaned = journal.cleanup_stale_intents()
        assert cleaned == 1


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

    def test_overwrite_clears_intent(self, tmp_path):
        """Overwriting an existing key should not leave stale intents."""
        cache = _make_cache(tmp_path)
        cache.put("original", cache_key="overwrite-key")
        cache.put("updated", cache_key="overwrite-key")

        intents_dir = tmp_path / ".intents"
        if intents_dir.exists():
            intent_files = list(intents_dir.glob("*.intent"))
            assert len(intent_files) == 0
