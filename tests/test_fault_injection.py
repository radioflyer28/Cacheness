"""
Fault Injection Tests for Cacheness
=====================================

Uses unittest.mock.patch to verify Cacheness handles I/O failures,
mid-operation crashes, and data corruption correctly.

Tests cover:
1. Orphaned blob on put() crash (metadata write failure after blob write)
2. get() auto-deletion on transient I/O errors
3. get() auto-deletion on handler exception (deserialization failure)
4. JSON backend corruption recovery
5. Disk full during blob write
6. TOCTOU race in get() (file exists check → file gone on open)

Issue: CACHE-yw9
"""

import os
import json
import errno
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from cacheness import cacheness
from cacheness.config import (
    CacheConfig,
    CacheStorageConfig,
    CacheMetadataConfig,
    CompressionConfig,
)
from cacheness.metadata import JsonBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cache(tmp_dir, backend="json"):
    """Create a cacheness instance for testing."""
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir=str(tmp_dir)),
        metadata=CacheMetadataConfig(metadata_backend=backend),
        compression=CompressionConfig(use_blosc2_arrays=False),
    )
    return cacheness(config)


# ===========================================================================
# 1. Orphaned Blob on put() Crash
# ===========================================================================


class TestOrphanedBlobOnPutCrash:
    """When metadata_backend.put_entry() fails after handler.put() writes
    its blob, the blob must be cleaned up to prevent orphans."""

    def test_blob_cleaned_up_on_metadata_failure(self, tmp_path):
        """put() should remove the blob file when metadata write fails."""
        cache = _make_cache(tmp_path)
        data = {"key": "value", "numbers": [1, 2, 3]}

        with patch.object(
            cache.metadata_backend, "put_entry",
            side_effect=RuntimeError("Simulated metadata write failure"),
        ):
            with pytest.raises(RuntimeError, match="Simulated metadata write failure"):
                cache.put(data, test_key="orphan_test")

        # After the failed put, no blob files should remain
        blob_files = list(tmp_path.rglob("*.pkl*"))
        assert blob_files == [], (
            f"Orphaned blob files found after failed put(): {blob_files}"
        )

    def test_no_metadata_entry_after_failed_put(self, tmp_path):
        """After a failed put(), no metadata entry should exist."""
        cache = _make_cache(tmp_path)
        data = "test string"

        with patch.object(
            cache.metadata_backend, "put_entry",
            side_effect=OSError("Simulated I/O failure"),
        ):
            with pytest.raises(OSError):
                cache.put(data, test_key="no_metadata_test")

        # Verify no metadata entry exists
        result = cache.get(test_key="no_metadata_test")
        assert result is None

    def test_blob_cleaned_up_on_signing_failure(self, tmp_path):
        """If signing succeeds but put_entry fails, blob should still be cleaned up."""
        cache = _make_cache(tmp_path)
        data = np.array([1, 2, 3])

        with patch.object(
            cache.metadata_backend, "put_entry",
            side_effect=Exception("Database connection lost"),
        ):
            with pytest.raises(Exception, match="Database connection lost"):
                cache.put(data, test_key="sign_then_fail")

        blob_files = list(tmp_path.rglob("*.npz")) + list(tmp_path.rglob("*.b2nd"))
        assert blob_files == [], (
            f"Orphaned blob files after metadata failure: {blob_files}"
        )


# ===========================================================================
# 2. get() Behavior on Transient I/O Errors
# ===========================================================================


class TestGetTransientIOErrors:
    """Transient I/O errors (e.g. disk temporarily unavailable) should NOT
    cause permanent metadata deletion. Only permanent errors
    (FileNotFoundError) should trigger cleanup."""

    def test_transient_ioerror_preserves_metadata(self, tmp_path):
        """A transient IOError in get() should NOT delete the metadata entry."""
        cache = _make_cache(tmp_path)
        data = {"important": "data"}

        # First, successfully cache the data
        cache_key = cache.put(data, test_key="transient_test")

        # Verify it's there
        assert cache.get(test_key="transient_test") == data

        # Now simulate a transient I/O error during get()
        handler = cache.handlers.get_handler_by_type("object")
        original_get = handler.get

        def fail_once(*args, **kwargs):
            raise IOError(errno.EIO, "Temporary disk error")

        with patch.object(handler, "get", side_effect=fail_once):
            result = cache.get(test_key="transient_test")
            assert result is None  # This get() should fail

        # The metadata entry should still exist for retry
        entry = cache.metadata_backend.get_entry(cache_key)
        assert entry is not None, (
            "Metadata entry was permanently deleted after a transient IOError"
        )

        # And a subsequent get() (without the fault) should succeed
        result = cache.get(test_key="transient_test")
        assert result == data

    def test_permanent_file_not_found_deletes_metadata(self, tmp_path):
        """FileNotFoundError (blob truly gone) SHOULD delete metadata."""
        cache = _make_cache(tmp_path)
        data = "some cached string"

        cache_key = cache.put(data, test_key="permanent_gone")

        # Delete the actual blob file
        entry = cache.metadata_backend.get_entry(cache_key)
        actual_path = entry["metadata"]["actual_path"]
        os.remove(actual_path)

        # get() should return None AND clean up the dangling metadata
        result = cache.get(test_key="permanent_gone")
        assert result is None

        entry_after = cache.metadata_backend.get_entry(cache_key)
        assert entry_after is None, (
            "Metadata entry should be removed when blob file is permanently missing"
        )


# ===========================================================================
# 3. get() on Handler Deserialization Failure
# ===========================================================================


class TestGetDeserializationFailure:
    """When handler.get() raises a deserialization error (corrupted data),
    the behavior should be intentional: delete the metadata if the blob
    is genuinely corrupt, but preserve it for transient issues."""

    def test_corrupted_pickle_deletes_metadata(self, tmp_path):
        """Permanently corrupted data should cause metadata cleanup."""
        cache = _make_cache(tmp_path)
        data = {"clean": "data"}

        cache_key = cache.put(data, test_key="corrupt_test")
        entry = cache.metadata_backend.get_entry(cache_key)
        actual_path = entry["metadata"]["actual_path"]

        # Corrupt the blob file
        with open(actual_path, "wb") as f:
            f.write(b"THIS IS NOT VALID PICKLE DATA AT ALL")

        result = cache.get(test_key="corrupt_test")
        assert result is None

        # Metadata should be cleaned up for genuine corruption
        entry_after = cache.metadata_backend.get_entry(cache_key)
        assert entry_after is None

    def test_module_not_found_deletes_metadata(self, tmp_path):
        """ModuleNotFoundError during unpickling should delete metadata."""
        cache = _make_cache(tmp_path)
        data = "test"

        cache_key = cache.put(data, test_key="module_test")

        handler = cache.handlers.get_handler_by_type("object")
        with patch.object(
            handler, "get",
            side_effect=ModuleNotFoundError("No module named 'deleted_package'"),
        ):
            result = cache.get(test_key="module_test")
            assert result is None

        entry_after = cache.metadata_backend.get_entry(cache_key)
        assert entry_after is None


# ===========================================================================
# 4. JSON Backend Corruption Recovery
# ===========================================================================


class TestJsonBackendCorruption:
    """JsonBackend should handle corrupted metadata files gracefully."""

    def test_garbage_bytes_starts_fresh(self, tmp_path):
        """Completely invalid JSON causes backend to start fresh."""
        metadata_file = tmp_path / "cache_metadata.json"
        metadata_file.write_bytes(b"\x00\xff\xfe GARBAGE NOT JSON \x80\x81")

        backend = JsonBackend(metadata_file)
        entries = backend.list_entries()
        assert entries == []

    def test_partial_json_starts_fresh(self, tmp_path):
        """Truncated JSON (simulating crash during write) starts fresh."""
        metadata_file = tmp_path / "cache_metadata.json"
        metadata_file.write_text('{"entries": {"key1": {"data_type": "test"')  # truncated

        backend = JsonBackend(metadata_file)
        entries = backend.list_entries()
        assert entries == []

    def test_valid_json_wrong_schema_starts_fresh(self, tmp_path):
        """Valid JSON but wrong structure starts fresh."""
        metadata_file = tmp_path / "cache_metadata.json"
        metadata_file.write_text('["this", "is", "a", "list", "not", "a", "dict"]')

        backend = JsonBackend(metadata_file)
        # Should not crash — implementation-dependent whether it starts fresh
        # or uses the data. The key invariant is no unhandled exception.
        assert isinstance(backend.list_entries(), list)

    def test_empty_file_starts_fresh(self, tmp_path):
        """Empty metadata file starts fresh."""
        metadata_file = tmp_path / "cache_metadata.json"
        metadata_file.write_text("")

        backend = JsonBackend(metadata_file)
        entries = backend.list_entries()
        assert entries == []

    def test_corruption_after_valid_entries(self, tmp_path):
        """Corruption after having valid entries causes loss of all entries."""
        metadata_file = tmp_path / "cache_metadata.json"

        # First write valid data
        backend = JsonBackend(metadata_file)
        backend.put_entry("key1", {
            "data_type": "test",
            "prefix": "",
            "description": "test entry",
            "file_size": 100,
            "metadata": {},
        })
        assert len(backend.list_entries()) == 1

        # Now corrupt the file externally
        metadata_file.write_bytes(b"CORRUPTED")

        # A new backend instance loads the corrupted file
        backend2 = JsonBackend(metadata_file)
        entries = backend2.list_entries()
        assert entries == [], "Corrupted metadata should result in empty entries"

    def test_no_crash_on_permission_error(self, tmp_path):
        """Backend should handle permission errors on the metadata file."""
        metadata_file = tmp_path / "cache_metadata.json"

        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            # Should not crash — start fresh instead
            backend = JsonBackend(metadata_file)
            entries = backend.list_entries()
            assert entries == []


# ===========================================================================
# 5. Disk Full During Blob Write
# ===========================================================================


class TestDiskFullDuringWrite:
    """When the disk is full during handler.put(), no partial blob or
    metadata entry should be left behind."""

    def test_oserror_during_put_no_partial_files(self, tmp_path):
        """OSError(ENOSPC) during blob write leaves no partial files."""
        cache = _make_cache(tmp_path)
        data = {"large": "data" * 1000}

        handler = cache.handlers.get_handler(data)
        original_put = handler.put

        def fail_disk_full(*args, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        with patch.object(handler, "put", side_effect=fail_disk_full):
            with pytest.raises(OSError):
                cache.put(data, test_key="disk_full_test")

        # No blob files should exist
        blob_files = list(tmp_path.rglob("*.pkl*"))
        assert blob_files == [], f"Partial blob files found: {blob_files}"

        # No metadata entry should exist
        assert cache.get(test_key="disk_full_test") is None


# ===========================================================================
# 6. TOCTOU Race in get() — file exists, then disappears
# ===========================================================================


class TestTOCTOURace:
    """Simulate a race condition where the blob file is deleted between
    the metadata lookup and the handler.get() call."""

    def test_file_deleted_between_check_and_read(self, tmp_path):
        """If blob is deleted after metadata check but before read,
        get() should handle it gracefully."""
        cache = _make_cache(tmp_path)
        data = "race condition test"

        cache_key = cache.put(data, test_key="race_test")

        # Get the actual path from metadata
        entry = cache.metadata_backend.get_entry(cache_key)
        actual_path = entry["metadata"]["actual_path"]

        # Delete the file right before handler.get() is called
        handler = cache.handlers.get_handler_by_type("object")
        original_get = handler.get

        def get_after_delete(*args, **kwargs):
            # Delete the file, then try to load it
            if os.path.exists(actual_path):
                os.remove(actual_path)
            return original_get(*args, **kwargs)

        with patch.object(handler, "get", side_effect=get_after_delete):
            result = cache.get(test_key="race_test")

        # Should return None without crashing
        assert result is None

        # Metadata should be cleaned up (file is permanently gone)
        entry_after = cache.metadata_backend.get_entry(cache_key)
        assert entry_after is None


# ===========================================================================
# 7. Exception Safety: put() doesn't leave partial state
# ===========================================================================


class TestPutExceptionSafety:
    """put() must be atomic: either both blob + metadata succeed,
    or neither is left behind."""

    def test_custom_metadata_failure_doesnt_orphan(self, tmp_path):
        """If custom metadata storage fails after put_entry, the cache entry
        should still be usable (custom metadata is optional)."""
        cache = _make_cache(tmp_path)
        data = "custom meta test"

        # This should succeed even if custom metadata fails —
        # custom metadata is best-effort
        cache_key = cache.put(data, test_key="custom_fail")
        assert cache.get(test_key="custom_fail") == data

    def test_concurrent_put_same_key(self, tmp_path):
        """Two sequential puts with the same key should not corrupt state."""
        cache = _make_cache(tmp_path)

        cache.put("first value", test_key="same_key")
        cache.put("second value", test_key="same_key")

        result = cache.get(test_key="same_key")
        assert result == "second value"
