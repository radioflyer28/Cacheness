"""
Tests for S3 ETag-based integrity verification optimization.

Issue: CACHE-6x5

When verify_integrity(verify_hashes=True) is called on a cache backed by S3,
it should use stored s3_etag + cheap HEAD check to skip full blob downloads
when the ETag confirms the blob hasn't changed.  Falls back to full xxhash
when no ETag is stored or the ETag mismatches.
"""

from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from cacheness.metadata import SqliteBackend
from cacheness.storage.blob_store import BlobStore
from cacheness.storage.backends.blob_backends import BlobBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_HASH = "a1b2c3d4e5f60708"
FAKE_ETAG = "d41d8cd98f00b204e9800998ecf8427e"
FAKE_S3_PATH = "s3://bucket/ab/abc123.blob"


def _make_s3_blob_store(tmp_path, entries):
    """Create a BlobStore with a mocked S3 blob backend and pre-populated entries.

    Args:
        tmp_path: Temporary directory for the SQLite database.
        entries: List of dicts with keys: cache_key, actual_path (**must**
            contain ``://`` for the S3 ETag path to trigger), file_hash,
            s3_etag (optional).

    Returns:
        (blob_store, mock_blob_backend) tuple.
    """
    db_path = tmp_path / "meta.db"
    backend = SqliteBackend(str(db_path))

    # Insert entries into SQLite metadata
    for e in entries:
        meta = {"actual_path": e["actual_path"], "file_hash": e.get("file_hash")}
        if e.get("s3_etag"):
            meta["s3_etag"] = e["s3_etag"]
        backend.put_entry(
            e["cache_key"],
            {
                "cache_key": e["cache_key"],
                "data_type": "object",
                "file_size": e.get("file_size", 1024),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": meta,
            },
        )

    # Build a mock S3 blob backend with verify_etag capability
    mock_bb = MagicMock(spec=BlobBackend)
    # Add S3-specific methods not on the base ABC
    mock_bb.verify_etag = MagicMock(return_value=True)
    mock_bb.list_blobs.return_value = [e["actual_path"] for e in entries]
    mock_bb.exists.return_value = True
    mock_bb.get_size.side_effect = lambda p: next(
        (e.get("file_size", 1024) for e in entries if e["actual_path"] == p), -1
    )
    # Default: verify_etag returns True (ETag matches)
    mock_bb.verify_etag.return_value = True
    # Default: read_blob_stream returns empty stream (hash = known value)
    mock_bb.read_blob_stream.return_value = BytesIO(b"")

    store = BlobStore(
        cache_dir=str(tmp_path / "blobs"),
        backend=backend,
        blob_backend=mock_bb,
    )
    return store, mock_bb


# ===========================================================================
# ETag Match — Skip Full Download
# ===========================================================================


class TestETagMatchSkipsDownload:
    """When stored ETag matches the live HEAD ETag, no full download needed."""

    def test_etag_match_skips_blob_download(self, tmp_path):
        """verify_integrity should NOT call read_blob_stream when ETag matches."""
        store, mock_bb = _make_s3_blob_store(
            tmp_path,
            [
                {
                    "cache_key": "k1",
                    "actual_path": FAKE_S3_PATH,
                    "file_hash": FAKE_HASH,
                    "s3_etag": FAKE_ETAG,
                }
            ],
        )
        mock_bb.verify_etag.return_value = True

        report = store.verify_integrity(verify_hashes=True)

        # verify_etag was called (cheap HEAD check)
        mock_bb.verify_etag.assert_called_once_with(FAKE_S3_PATH, FAKE_ETAG)
        # read_blob_stream was NOT called (no full download)
        mock_bb.read_blob_stream.assert_not_called()
        # No hash mismatches reported
        assert report["hash_mismatches"] == []

    def test_etag_match_multiple_entries(self, tmp_path):
        """All S3 entries with matching ETags skip downloads."""
        entries = [
            {
                "cache_key": f"k{i}",
                "actual_path": f"s3://bucket/{i:02x}/blob{i}.pkl",
                "file_hash": f"hash{i}",
                "s3_etag": f"etag{i}",
            }
            for i in range(5)
        ]
        store, mock_bb = _make_s3_blob_store(tmp_path, entries)
        mock_bb.verify_etag.return_value = True

        report = store.verify_integrity(verify_hashes=True)

        assert mock_bb.verify_etag.call_count == 5
        mock_bb.read_blob_stream.assert_not_called()
        assert report["hash_mismatches"] == []


# ===========================================================================
# ETag Mismatch — Definitive Integrity Failure (No Download)
# ===========================================================================


class TestETagMismatchReportsFailure:
    """When ETag mismatches, report immediately without downloading."""

    def test_etag_mismatch_reports_without_download(self, tmp_path):
        """verify_integrity should NOT download when ETag mismatches."""
        store, mock_bb = _make_s3_blob_store(
            tmp_path,
            [
                {
                    "cache_key": "k1",
                    "actual_path": FAKE_S3_PATH,
                    "file_hash": FAKE_HASH,
                    "s3_etag": FAKE_ETAG,
                }
            ],
        )
        mock_bb.verify_etag.return_value = False
        mock_bb.get_etag = MagicMock(return_value="different_etag_from_s3")

        report = store.verify_integrity(verify_hashes=True)

        # verify_etag was called
        mock_bb.verify_etag.assert_called_once()
        # get_etag called to report actual value
        mock_bb.get_etag.assert_called_once_with(FAKE_S3_PATH)
        # NO full download
        mock_bb.read_blob_stream.assert_not_called()
        # Mismatch reported
        assert len(report["hash_mismatches"]) == 1
        assert report["hash_mismatches"][0]["cache_key"] == "k1"
        assert report["hash_mismatches"][0]["expected_hash"] == FAKE_HASH
        assert "etag-mismatch:" in report["hash_mismatches"][0]["actual_hash"]

    def test_etag_mismatch_includes_actual_etag(self, tmp_path):
        """The reported actual_hash should include the live S3 ETag."""
        store, mock_bb = _make_s3_blob_store(
            tmp_path,
            [
                {
                    "cache_key": "k1",
                    "actual_path": FAKE_S3_PATH,
                    "file_hash": FAKE_HASH,
                    "s3_etag": FAKE_ETAG,
                }
            ],
        )
        mock_bb.verify_etag.return_value = False
        mock_bb.get_etag = MagicMock(return_value="abc123newetag")

        report = store.verify_integrity(verify_hashes=True)

        assert (
            report["hash_mismatches"][0]["actual_hash"] == "etag-mismatch:abc123newetag"
        )


# ===========================================================================
# No Stored ETag — Full Hash Only
# ===========================================================================


class TestNoStoredETag:
    """When no s3_etag is stored, always do full download + xxhash."""

    def test_no_etag_skips_verify_etag(self, tmp_path):
        """Without s3_etag, verify_etag should NOT be called."""
        store, mock_bb = _make_s3_blob_store(
            tmp_path,
            [
                {
                    "cache_key": "k1",
                    "actual_path": FAKE_S3_PATH,
                    "file_hash": FAKE_HASH,
                    # No s3_etag
                }
            ],
        )
        mock_bb.read_blob_stream.return_value = BytesIO(b"")

        report = store.verify_integrity(verify_hashes=True)

        mock_bb.verify_etag.assert_not_called()
        mock_bb.read_blob_stream.assert_called_once()


# ===========================================================================
# Non-Remote Blobs (local filesystem) — Full Hash Only
# ===========================================================================


class TestLocalBlobAlwaysFullHash:
    """Local filesystem blobs always use full xxhash, never ETag."""

    def test_local_path_skips_etag_check(self, tmp_path):
        """Local paths (no '://') should not trigger ETag check."""
        db_path = tmp_path / "meta.db"
        backend = SqliteBackend(str(db_path))

        local_path = str(tmp_path / "blobs" / "local.pkl")

        backend.put_entry(
            "local_k",
            {
                "cache_key": "local_k",
                "data_type": "object",
                "file_size": 100,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "actual_path": local_path,
                    "file_hash": FAKE_HASH,
                },
            },
        )

        mock_bb = MagicMock(spec=BlobBackend)
        # Add S3-specific method — should NOT be called for local paths
        mock_bb.verify_etag = MagicMock()
        mock_bb.list_blobs.return_value = [local_path]
        mock_bb.exists.return_value = True
        mock_bb.get_size.return_value = 100
        mock_bb.read_blob_stream.return_value = BytesIO(b"")

        store = BlobStore(
            cache_dir=str(tmp_path / "blobs"),
            backend=backend,
            blob_backend=mock_bb,
        )

        report = store.verify_integrity(verify_hashes=True)

        # verify_etag should NOT be called since this is a local path
        mock_bb.verify_etag.assert_not_called()
        # Full hash IS performed
        mock_bb.read_blob_stream.assert_called_once()


# ===========================================================================
# Backend Without verify_etag — Full Hash Only
# ===========================================================================


class TestBackendWithoutVerifyEtag:
    """Blob backends without verify_etag method always use full hash."""

    def test_backend_without_verify_etag_skips_etag(self, tmp_path):
        """When blob_backend has no verify_etag, ETag optimization is skipped."""
        db_path = tmp_path / "meta.db"
        backend = SqliteBackend(str(db_path))

        backend.put_entry(
            "k1",
            {
                "cache_key": "k1",
                "data_type": "object",
                "file_size": 1024,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "actual_path": FAKE_S3_PATH,
                    "file_hash": FAKE_HASH,
                    "s3_etag": FAKE_ETAG,
                },
            },
        )

        # Create mock WITHOUT verify_etag (simulates non-S3 blob backend)
        mock_bb = MagicMock(spec=BlobBackend)
        mock_bb.list_blobs.return_value = [FAKE_S3_PATH]
        mock_bb.exists.return_value = True
        mock_bb.get_size.return_value = 1024
        mock_bb.read_blob_stream.return_value = BytesIO(b"")
        # BlobBackend ABC doesn't have verify_etag, and MagicMock(spec=...)
        # restricts attribute access to the spec's interface, so hasattr
        # will return False for verify_etag.

        store = BlobStore(
            cache_dir=str(tmp_path / "blobs"),
            backend=backend,
            blob_backend=mock_bb,
        )

        report = store.verify_integrity(verify_hashes=True)

        # No verify_etag on spec=BlobBackend → hasattr returns False
        assert not hasattr(mock_bb, "verify_etag")
        # Full hash IS performed
        mock_bb.read_blob_stream.assert_called_once()


# ===========================================================================
# iter_entry_summaries Includes s3_etag
# ===========================================================================


class TestIterEntrySummariesIncludesETag:
    """Verify that iter_entry_summaries returns s3_etag when stored."""

    def test_s3_etag_in_summary(self, tmp_path):
        """iter_entry_summaries should include s3_etag when present."""
        db_path = tmp_path / "meta.db"
        backend = SqliteBackend(str(db_path))

        backend.put_entry(
            "k1",
            {
                "cache_key": "k1",
                "data_type": "array",
                "file_size": 2048,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "s3_etag": FAKE_ETAG,
                    "actual_path": FAKE_S3_PATH,
                    "file_hash": FAKE_HASH,
                },
            },
        )

        summaries = backend.iter_entry_summaries()
        assert len(summaries) == 1
        assert summaries[0]["s3_etag"] == FAKE_ETAG
        assert summaries[0]["actual_path"] == FAKE_S3_PATH

        backend.close()

    def test_s3_etag_absent_when_not_stored(self, tmp_path):
        """iter_entry_summaries should NOT include s3_etag if not stored."""
        db_path = tmp_path / "meta.db"
        backend = SqliteBackend(str(db_path))

        backend.put_entry(
            "k_local",
            {
                "cache_key": "k_local",
                "data_type": "object",
                "file_size": 512,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "actual_path": "/local/path/blob.pkl",
                    "file_hash": FAKE_HASH,
                },
            },
        )

        summaries = backend.iter_entry_summaries()
        assert len(summaries) == 1
        assert "s3_etag" not in summaries[0]

        backend.close()


# ===========================================================================
# Mixed Entries — S3 + Local
# ===========================================================================


class TestMixedEntries:
    """Verify correct behavior with a mix of S3 (with/without ETag) and local entries."""

    def test_mixed_entries_selective_etag_check(self, tmp_path):
        """Only S3 entries with stored ETags use ETag check; rest do full hash."""
        entries = [
            {
                "cache_key": "s3_with_etag",
                "actual_path": "s3://bucket/a.blob",
                "file_hash": "hash_a",
                "s3_etag": "etag_a",
                "file_size": 100,
            },
            {
                "cache_key": "s3_no_etag",
                "actual_path": "s3://bucket/b.blob",
                "file_hash": "hash_b",
                "file_size": 200,
                # No s3_etag
            },
            {
                "cache_key": "local_entry",
                "actual_path": "/local/c.pkl",
                "file_hash": "hash_c",
                "file_size": 300,
            },
        ]
        store, mock_bb = _make_s3_blob_store(tmp_path, entries)
        mock_bb.verify_etag.return_value = True
        mock_bb.read_blob_stream.return_value = BytesIO(b"")

        report = store.verify_integrity(verify_hashes=True)

        # Only the S3 entry WITH etag should get verify_etag called
        mock_bb.verify_etag.assert_called_once_with("s3://bucket/a.blob", "etag_a")
        # The other two (S3 without etag + local) should trigger full download
        assert mock_bb.read_blob_stream.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
