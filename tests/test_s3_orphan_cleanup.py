"""
S3 Orphan Cleanup Tests
========================

Verifies that when a blob is uploaded to S3 but the subsequent metadata
write fails, the S3 object is deleted (rolled back) so no orphaned
blobs accumulate.

Covers all three write paths in ``UnifiedCache``:
- ``put()`` — normal cache write
- ``_storage_mode_put()`` — storage-mode write (no eviction)
- ``update_data()`` — in-place data update

Issue: CACHE-yxf
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cacheness.core import UnifiedCache, _PutCleanup
from cacheness.config import (
    CacheConfig,
    CacheStorageConfig,
    CacheMetadataConfig,
    CompressionConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

S3_URI = "s3://test-bucket/blobs/abc123.pkl"


def _make_cache(tmp_dir, storage_mode=False):
    """Create a UnifiedCache for testing."""
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_dir)),
        metadata=CacheMetadataConfig(metadata_backend="json"),
        compression=CompressionConfig(use_blosc2_arrays=False),
        storage_mode=storage_mode,
    )
    return UnifiedCache(config)


def _fake_write_blob_result():
    """Return a minimal handler + result tuple that looks like S3 upload."""
    handler = MagicMock()
    handler.data_type = "pickle"
    handler.serializer = "pickle"

    result = {
        "actual_path": S3_URI,
        "file_size": 42,
        "content_hash": "abc123",
        "file_hash": "def456",
        "storage_format": "pickle",
        "metadata": {"actual_path": S3_URI},
        "s3_etag": '"etag123"',
    }
    file_hash = "def456"
    return handler, result, file_hash


# ===========================================================================
# Unit tests for _PutCleanup
# ===========================================================================


class TestPutCleanup:
    """Unit tests for the _PutCleanup rollback helper."""

    def test_commit_prevents_rollback(self, tmp_path):
        """After commit(), rollback() should not delete anything."""
        blob = tmp_path / "test.pkl"
        blob.write_bytes(b"data")

        cleanup = _PutCleanup()
        cleanup.blob_path = blob
        cleanup.commit()
        cleanup.rollback()

        assert blob.exists(), "commit() should prevent rollback from deleting blob"

    def test_rollback_deletes_local_blob(self, tmp_path):
        """rollback() should delete a tracked local blob file."""
        blob = tmp_path / "test.pkl"
        blob.write_bytes(b"data")

        cleanup = _PutCleanup()
        cleanup.blob_path = blob
        cleanup.rollback()

        assert not blob.exists(), "rollback() should delete the local blob"

    def test_rollback_deletes_remote_blob(self):
        """rollback() should call delete_blob() on a tracked remote backend."""
        mock_backend = MagicMock()
        cleanup = _PutCleanup()
        cleanup.set_remote(mock_backend, S3_URI)
        cleanup.rollback()

        mock_backend.delete_blob.assert_called_once_with(S3_URI)

    def test_rollback_deletes_both_local_and_remote(self, tmp_path):
        """rollback() should clean up both local and remote blobs."""
        blob = tmp_path / "test.pkl"
        blob.write_bytes(b"data")
        mock_backend = MagicMock()

        cleanup = _PutCleanup()
        cleanup.blob_path = blob
        cleanup.set_remote(mock_backend, S3_URI)
        cleanup.rollback()

        assert not blob.exists()
        mock_backend.delete_blob.assert_called_once_with(S3_URI)

    def test_double_rollback_is_safe(self):
        """Calling rollback() twice should not raise."""
        mock_backend = MagicMock()
        cleanup = _PutCleanup()
        cleanup.set_remote(mock_backend, S3_URI)
        cleanup.rollback()
        cleanup.rollback()  # should not raise

        # delete_blob only called once (first rollback)
        mock_backend.delete_blob.assert_called_once()

    def test_rollback_handles_missing_local_file(self, tmp_path):
        """rollback() should not raise if local blob is already gone."""
        cleanup = _PutCleanup()
        cleanup.blob_path = tmp_path / "nonexistent.pkl"
        cleanup.rollback()  # should not raise

    def test_rollback_logs_remote_failure(self):
        """rollback() should log but not raise on remote delete failure."""
        mock_backend = MagicMock()
        mock_backend.delete_blob.side_effect = Exception("S3 unreachable")

        cleanup = _PutCleanup()
        cleanup.set_remote(mock_backend, S3_URI)
        cleanup.rollback()  # should not raise

    def test_no_resources_tracked(self):
        """rollback() with nothing tracked should be a no-op."""
        cleanup = _PutCleanup()
        cleanup.rollback()  # should not raise


# ===========================================================================
# S3 orphan cleanup on put()
# ===========================================================================


class TestS3OrphanCleanupPut:
    """Verify S3 blobs are cleaned up when put() fails after blob upload."""

    def test_s3_blob_deleted_on_metadata_failure(self, tmp_path):
        """put(): S3 blob should be deleted when put_entry() raises."""
        cache = _make_cache(tmp_path)
        mock_backend = MagicMock()

        with (
            patch.object(
                cache._blob_store,
                "_write_blob",
                return_value=_fake_write_blob_result(),
            ),
            patch.object(
                cache._blob_store,
                "blob_backend",
                mock_backend,
                create=True,
            ),
            patch.object(
                cache.metadata_backend,
                "put_entry",
                side_effect=RuntimeError("Simulated metadata failure"),
            ),
        ):
            with pytest.raises(RuntimeError, match="Simulated metadata failure"):
                cache.put("test data", cache_key="orphan-test")

        mock_backend.delete_blob.assert_called_once_with(S3_URI)

    def test_s3_blob_deleted_on_ioerror(self, tmp_path):
        """put(): S3 blob should be deleted on IOError during metadata write."""
        cache = _make_cache(tmp_path)
        mock_backend = MagicMock()

        with (
            patch.object(
                cache._blob_store,
                "_write_blob",
                return_value=_fake_write_blob_result(),
            ),
            patch.object(
                cache._blob_store,
                "blob_backend",
                mock_backend,
                create=True,
            ),
            patch.object(
                cache.metadata_backend,
                "put_entry",
                side_effect=OSError("Disk I/O failure"),
            ),
        ):
            with pytest.raises(OSError, match="Disk I/O failure"):
                cache.put("test data", cache_key="io-error-test")

        mock_backend.delete_blob.assert_called_once_with(S3_URI)


# ===========================================================================
# S3 orphan cleanup on _storage_mode_put()
# ===========================================================================


class TestS3OrphanCleanupStorageMode:
    """Verify S3 blobs are cleaned up when _storage_mode_put() fails."""

    def test_s3_blob_deleted_on_metadata_failure(self, tmp_path):
        """_storage_mode_put(): S3 blob deleted on put_entry() failure."""
        cache = _make_cache(tmp_path, storage_mode=True)
        mock_backend = MagicMock()

        with (
            patch.object(
                cache._blob_store,
                "_write_blob",
                return_value=_fake_write_blob_result(),
            ),
            patch.object(
                cache._blob_store,
                "blob_backend",
                mock_backend,
                create=True,
            ),
            patch.object(
                cache.metadata_backend,
                "put_entry",
                side_effect=RuntimeError("Simulated metadata failure"),
            ),
        ):
            with pytest.raises(RuntimeError, match="Simulated metadata failure"):
                cache.put("test data", cache_key="storage-mode-orphan")

        mock_backend.delete_blob.assert_called_once_with(S3_URI)


# ===========================================================================
# S3 orphan cleanup on update_data()
# ===========================================================================


class TestS3OrphanCleanupUpdateData:
    """Verify S3 blobs are cleaned up when update_data() fails."""

    def test_s3_blob_deleted_on_metadata_failure(self, tmp_path):
        """update_data(): S3 blob deleted when update_entry_metadata() raises."""
        cache = _make_cache(tmp_path)

        # First, create a real entry so update_data() finds it
        cache.put("original data", cache_key="update-target")

        mock_backend = MagicMock()

        with (
            patch.object(
                cache._blob_store,
                "_write_blob",
                return_value=_fake_write_blob_result(),
            ),
            patch.object(
                cache._blob_store,
                "blob_backend",
                mock_backend,
                create=True,
            ),
            patch.object(
                cache.metadata_backend,
                "update_entry_metadata",
                side_effect=RuntimeError("Simulated metadata failure"),
            ),
        ):
            with pytest.raises(RuntimeError, match="Simulated metadata failure"):
                cache.update_data("new data", cache_key="update-target")

        mock_backend.delete_blob.assert_called_once_with(S3_URI)

    def test_no_cleanup_when_entry_not_found(self, tmp_path):
        """update_data(): returns False without cleanup when entry missing."""
        cache = _make_cache(tmp_path)
        result = cache.update_data("data", cache_key="nonexistent")
        assert result is False


# ===========================================================================
# Successful writes should NOT trigger cleanup
# ===========================================================================


class TestS3NoCleanupOnSuccess:
    """Verify successful writes do NOT trigger delete_blob()."""

    def test_put_success_no_delete(self, tmp_path):
        """put(): successful write should not call delete_blob()."""
        cache = _make_cache(tmp_path)
        mock_backend = MagicMock()

        with (
            patch.object(
                cache._blob_store,
                "_write_blob",
                return_value=_fake_write_blob_result(),
            ),
            patch.object(
                cache._blob_store,
                "blob_backend",
                mock_backend,
                create=True,
            ),
        ):
            cache.put("test data", cache_key="success-test")

        mock_backend.delete_blob.assert_not_called()

    def test_storage_mode_put_success_no_delete(self, tmp_path):
        """_storage_mode_put(): successful write should not call delete_blob()."""
        cache = _make_cache(tmp_path, storage_mode=True)
        mock_backend = MagicMock()

        with (
            patch.object(
                cache._blob_store,
                "_write_blob",
                return_value=_fake_write_blob_result(),
            ),
            patch.object(
                cache._blob_store,
                "blob_backend",
                mock_backend,
                create=True,
            ),
        ):
            cache.put("test data", cache_key="success-storage")

        mock_backend.delete_blob.assert_not_called()
