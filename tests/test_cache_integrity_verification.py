"""
Tests for Cache Integrity Verification (fsck)
===============================================

Tests for UnifiedCache.verify_integrity() which detects and optionally
repairs inconsistencies between blob files and metadata.

Issue: CACHE-8fu
"""

import os

import numpy as np

from cacheness import cacheness
from cacheness.config import (
    CacheConfig,
    CacheStorageConfig,
    CacheMetadataConfig,
    CompressionConfig,
)


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
# Clean Cache — No Issues
# ===========================================================================


class TestVerifyIntegrityClean:
    """verify_integrity() on a healthy cache should find no issues."""

    def test_empty_cache_is_clean(self, tmp_path):
        cache = _make_cache(tmp_path)
        report = cache.verify_integrity()

        assert report["orphaned_blobs"] == []
        assert report["dangling_entries"] == []
        assert report["size_mismatches"] == []

    def test_single_entry_is_clean(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache.put("hello world", test_key="clean")

        report = cache.verify_integrity()

        assert report["orphaned_blobs"] == []
        assert report["dangling_entries"] == []
        assert report["size_mismatches"] == []

    def test_multiple_types_clean(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache.put("string data", test_key="str")
        cache.put({"dict": "data"}, test_key="dict")
        cache.put([1, 2, 3], test_key="list")
        cache.put(np.array([1, 2, 3]), test_key="numpy")

        report = cache.verify_integrity()

        assert report["orphaned_blobs"] == []
        assert report["dangling_entries"] == []
        assert report["size_mismatches"] == []


# ===========================================================================
# Orphaned Blobs Detection
# ===========================================================================


class TestOrphanedBlobDetection:
    """Detect files in cache_dir with no corresponding metadata entry."""

    def test_detects_orphaned_pkl(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache.put("real data", test_key="real")

        # Create an orphaned .pkl file in the namespace blob directory
        ns_dir = tmp_path / "default"
        ns_dir.mkdir(exist_ok=True)
        orphan = ns_dir / "orphaned_file.pkl"
        orphan.write_bytes(b"fake pickle data")

        report = cache.verify_integrity()

        assert len(report["orphaned_blobs"]) == 1
        assert os.path.normpath(str(orphan)) in report["orphaned_blobs"]
        assert report["dangling_entries"] == []

    def test_detects_orphaned_npz(self, tmp_path):
        cache = _make_cache(tmp_path)

        ns_dir = tmp_path / "default"
        ns_dir.mkdir(exist_ok=True)
        orphan = ns_dir / "orphaned_array.npz"
        orphan.write_bytes(b"fake npz data")

        report = cache.verify_integrity()

        assert len(report["orphaned_blobs"]) == 1

    def test_detects_orphaned_compressed_pkl(self, tmp_path):
        cache = _make_cache(tmp_path)

        ns_dir = tmp_path / "default"
        ns_dir.mkdir(exist_ok=True)
        orphan = ns_dir / "orphaned.pkl.zstd"
        orphan.write_bytes(b"fake compressed data")

        report = cache.verify_integrity()

        assert len(report["orphaned_blobs"]) == 1

    def test_detects_custom_extension_orphan_and_ignores_temp_file(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache.put("real data", test_key="real")

        ns_dir = tmp_path / "default"
        ns_dir.mkdir(exist_ok=True)
        orphan = ns_dir / "orphan.custom"
        orphan.write_bytes(b"custom handler data")
        temp_artifact = ns_dir / "ignore.tmp"
        temp_artifact.write_bytes(b"in-progress temp data")

        report = cache.verify_integrity()

        assert os.path.normpath(str(orphan)) in report["orphaned_blobs"]
        assert os.path.normpath(str(temp_artifact)) not in report["orphaned_blobs"]
        assert report["dangling_entries"] == []

    def test_detects_multiple_orphans(self, tmp_path):
        cache = _make_cache(tmp_path)

        ns_dir = tmp_path / "default"
        ns_dir.mkdir(exist_ok=True)
        (ns_dir / "orphan1.pkl").write_bytes(b"data1")
        (ns_dir / "orphan2.npz").write_bytes(b"data2")
        (ns_dir / "orphan3.parquet").write_bytes(b"data3")

        report = cache.verify_integrity()

        assert len(report["orphaned_blobs"]) == 3

    def test_repair_deletes_orphaned_blobs(self, tmp_path):
        cache = _make_cache(tmp_path)

        ns_dir = tmp_path / "default"
        ns_dir.mkdir(exist_ok=True)
        orphan = ns_dir / "orphaned.pkl"
        orphan.write_bytes(b"garbage")

        report = cache.verify_integrity(repair=True)

        assert report["repaired"]["orphans_deleted"] == 1
        assert not orphan.exists()

    def test_repair_preserves_valid_entries(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache.put("keep me", test_key="valid")

        # Add an orphan
        ns_dir = tmp_path / "default"
        ns_dir.mkdir(exist_ok=True)
        (ns_dir / "orphan.pkl").write_bytes(b"delete me")

        report = cache.verify_integrity(repair=True)

        assert report["repaired"]["orphans_deleted"] == 1
        # Valid entry should still work
        assert cache.get(test_key="valid") == "keep me"


# ===========================================================================
# Dangling Metadata Detection
# ===========================================================================


class TestDanglingMetadataDetection:
    """Detect metadata entries pointing to missing blob files."""

    def test_detects_dangling_after_file_deletion(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache_key = cache.put("will be deleted", test_key="dangling")

        # Delete the blob file but keep metadata
        entry = cache.metadata_backend.get_entry(cache_key)
        actual_path = cache._resolve_actual_path(entry["metadata"]["actual_path"])
        os.remove(actual_path)

        report = cache.verify_integrity()

        assert len(report["dangling_entries"]) == 1
        assert report["dangling_entries"][0]["cache_key"] == cache_key

    def test_repair_removes_dangling_metadata(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache_key = cache.put("disappearing", test_key="gone")

        entry = cache.metadata_backend.get_entry(cache_key)
        os.remove(cache._resolve_actual_path(entry["metadata"]["actual_path"]))

        report = cache.verify_integrity(repair=True)

        assert report["repaired"]["dangling_removed"] == 1
        assert cache.metadata_backend.get_entry(cache_key) is None

    def test_no_false_positives_on_valid_entries(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache.put("valid", test_key="exists")

        report = cache.verify_integrity()
        assert report["dangling_entries"] == []


# ===========================================================================
# Size Mismatch Detection
# ===========================================================================


class TestSizeMismatchDetection:
    """Detect when metadata file_size doesn't match actual file size."""

    def test_detects_size_mismatch_after_external_modification(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache_key = cache.put("original data", test_key="modified")

        entry = cache.metadata_backend.get_entry(cache_key)
        actual_path = cache._resolve_actual_path(entry["metadata"]["actual_path"])

        # Modify the file externally (change its size)
        with open(actual_path, "ab") as f:
            f.write(b"extra bytes appended externally")

        report = cache.verify_integrity()

        assert len(report["size_mismatches"]) == 1
        mismatch = report["size_mismatches"][0]
        assert mismatch["cache_key"] == cache_key
        assert mismatch["actual_size"] > mismatch["expected_size"]

    def test_detects_truncated_file(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache_key = cache.put("some data to truncate", test_key="truncated")

        entry = cache.metadata_backend.get_entry(cache_key)
        actual_path = cache._resolve_actual_path(entry["metadata"]["actual_path"])

        # Truncate the file
        with open(actual_path, "wb") as f:
            f.write(b"short")

        report = cache.verify_integrity()

        assert len(report["size_mismatches"]) == 1
        mismatch = report["size_mismatches"][0]
        assert mismatch["actual_size"] < mismatch["expected_size"]

    def test_no_size_mismatch_on_clean_cache(self, tmp_path):
        cache = _make_cache(tmp_path)
        cache.put("data", test_key="clean")

        report = cache.verify_integrity()
        assert report["size_mismatches"] == []


# ===========================================================================
# Hash Verification
# ===========================================================================


class TestHashVerification:
    """Optional hash verification catches silent corruption."""

    def test_detects_hash_mismatch(self, tmp_path):
        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            metadata=CacheMetadataConfig(
                metadata_backend="json",
                verify_cache_integrity=True,  # Enable hash storage
            ),
            compression=CompressionConfig(use_blosc2_arrays=False),
        )
        cache = cacheness(config)
        cache_key = cache.put("hash test data", test_key="hash_test")

        entry = cache.metadata_backend.get_entry(cache_key)
        actual_path = cache._resolve_actual_path(entry["metadata"]["actual_path"])

        # Corrupt the file without changing its size (overwrite some bytes)
        with open(actual_path, "r+b") as f:
            f.seek(0)
            f.write(b"\x00\x00\x00\x00")

        report = cache.verify_integrity(verify_hashes=True)

        assert len(report["hash_mismatches"]) == 1
        assert report["hash_mismatches"][0]["cache_key"] == cache_key

    def test_no_hash_check_without_flag(self, tmp_path):
        """verify_hashes=False should skip hash verification."""
        cache = _make_cache(tmp_path)
        cache.put("data", test_key="no_hash")

        report = cache.verify_integrity(verify_hashes=False)

        assert "hash_mismatches" not in report

    def test_hash_check_clean(self, tmp_path):
        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_path)),
            metadata=CacheMetadataConfig(
                metadata_backend="json",
                verify_cache_integrity=True,
            ),
            compression=CompressionConfig(use_blosc2_arrays=False),
        )
        cache = cacheness(config)
        cache.put("good data", test_key="clean_hash")

        report = cache.verify_integrity(verify_hashes=True)
        assert report["hash_mismatches"] == []


# ===========================================================================
# Combined Scenarios
# ===========================================================================


class TestVerifyIntegrityCombined:
    """Combined scenarios with multiple issue types."""

    def test_detects_all_issues_at_once(self, tmp_path):
        cache = _make_cache(tmp_path)

        # 1. Valid entry
        cache.put("valid", test_key="ok")

        # 2. Create orphaned blob
        ns_dir = tmp_path / "default"
        ns_dir.mkdir(exist_ok=True)
        (ns_dir / "orphan.pkl").write_bytes(b"orphan data")

        # 3. Create dangling metadata (put then delete blob)
        cache_key = cache.put("will dangle", test_key="dangle")
        entry = cache.metadata_backend.get_entry(cache_key)
        os.remove(cache._resolve_actual_path(entry["metadata"]["actual_path"]))

        report = cache.verify_integrity()

        assert len(report["orphaned_blobs"]) == 1
        assert len(report["dangling_entries"]) == 1
        assert report["size_mismatches"] == []

    def test_repair_fixes_all_issues(self, tmp_path):
        cache = _make_cache(tmp_path)

        # Valid entry should survive
        cache.put("survivor", test_key="keep")

        # Orphan
        ns_dir = tmp_path / "default"
        ns_dir.mkdir(exist_ok=True)
        orphan = ns_dir / "orphan.pkl"
        orphan.write_bytes(b"delete me")

        # Dangling
        cache_key = cache.put("ghost", test_key="ghost")
        entry = cache.metadata_backend.get_entry(cache_key)
        os.remove(cache._resolve_actual_path(entry["metadata"]["actual_path"]))

        report = cache.verify_integrity(repair=True)

        assert report["repaired"]["orphans_deleted"] == 1
        assert report["repaired"]["dangling_removed"] == 1
        assert not orphan.exists()
        assert cache.metadata_backend.get_entry(cache_key) is None

        # Valid entry still works
        assert cache.get(test_key="keep") == "survivor"

        # Second run should be clean
        report2 = cache.verify_integrity()
        assert report2["orphaned_blobs"] == []
        assert report2["dangling_entries"] == []

    def test_report_structure(self, tmp_path):
        """Verify the report dict has the expected keys."""
        cache = _make_cache(tmp_path)

        # Without repair or hash check
        report = cache.verify_integrity(verify_hashes=False)
        assert "orphaned_blobs" in report
        assert "dangling_entries" in report
        assert "size_mismatches" in report
        assert "repaired" not in report
        assert "hash_mismatches" not in report

        # With repair
        report = cache.verify_integrity(repair=True)
        assert "repaired" in report

        # Default includes hash check (verify_hashes=True by default)
        report = cache.verify_integrity()
        assert "hash_mismatches" in report


# ===========================================================================
# Signature Verification via verify_integrity
# ===========================================================================


class TestVerifyIntegritySignatures:
    """Tests for HMAC signature verification in verify_integrity()."""

    def _make_signed_cache(self, tmp_dir):
        """Create a cacheness instance with entry signing enabled."""
        from cacheness.config import SecurityConfig

        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_dir)),
            metadata=CacheMetadataConfig(metadata_backend="json"),
            compression=CompressionConfig(use_blosc2_arrays=False),
            security=SecurityConfig(
                enable_entry_signing=True,
                allow_unsigned_entries=True,
            ),
        )
        return cacheness(config)

    def test_file_hash_in_signed_fields(self):
        """Confirmatory: file_hash must be in HMAC signed fields for all versions."""
        from cacheness.security import CacheEntrySigner

        for version, fields in CacheEntrySigner.SIGNED_FIELDS_BY_VERSION.items():
            assert "file_hash" in fields, (
                f"file_hash missing from signed fields v{version}"
            )

    def test_verify_integrity_detects_signature_failure(self, tmp_path):
        """Tampered signature should be detected by verify_signatures=True."""
        cache = self._make_signed_cache(tmp_path)
        cache_key = cache.put("signed data", test_key="sig_test")

        # Tamper with the stored signature in metadata
        entry = cache.metadata_backend.get_entry(cache_key)
        entry["metadata"]["entry_signature"] = "tampered_signature_value"
        entry["entry_signature"] = "tampered_signature_value"
        cache.metadata_backend.put_entry(cache_key, entry)

        report = cache.verify_integrity(verify_signatures=True)

        assert "signature_failures" in report
        failures = report["signature_failures"]
        assert len(failures) >= 1
        tampered = [f for f in failures if f["cache_key"] == cache_key]
        assert len(tampered) == 1
        assert tampered[0]["reason"] == "invalid_signature"

    def test_verify_integrity_signature_clean(self, tmp_path):
        """Properly signed cache should have no signature failures."""
        cache = self._make_signed_cache(tmp_path)
        cache.put("good data", test_key="clean_sig")

        report = cache.verify_integrity(verify_signatures=True)

        assert "signature_failures" in report
        assert report["signature_failures"] == []

    def test_verify_integrity_no_signatures_by_default(self, tmp_path):
        """verify_integrity() without verify_signatures should not include signature_failures."""
        cache = self._make_signed_cache(tmp_path)
        cache.put("data", test_key="default_sig")

        report = cache.verify_integrity()

        assert "signature_failures" not in report

    def test_verify_integrity_blob_tampering_end_to_end(self, tmp_path):
        """Modified blob content should be caught by both hash and signature checks."""
        cache = self._make_signed_cache(tmp_path)
        cache_key = cache.put("important data", test_key="e2e_test")

        entry = cache.metadata_backend.get_entry(cache_key)
        actual_path = cache._resolve_actual_path(entry["metadata"]["actual_path"])

        # Corrupt the blob on disk
        with open(actual_path, "r+b") as f:
            f.seek(0)
            f.write(b"\x00\x00\x00\x00")

        report = cache.verify_integrity(verify_hashes=True, verify_signatures=True)

        # Hash mismatch should be detected
        assert len(report["hash_mismatches"]) >= 1
        # Signature should still be valid (metadata wasn't tampered)
        sig_failures = [
            f for f in report["signature_failures"] if f["cache_key"] == cache_key
        ]
        assert len(sig_failures) == 0
