"""
Atomic Write Verification Tests
================================

Verifies cross-platform atomic rename correctness and concurrent write safety.
Tests both Path.replace() (used by FilesystemBlobBackend) and shutil.move()
(used by JSON metadata backend) atomic write patterns.

Covers TEST-02 (cross-platform atomic write verification).
"""

import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache


class TestAtomicRename:
    """Verify atomic rename correctness for both Path.replace() and shutil.move()."""

    def test_path_replace_overwrites_existing_file(self, tmp_path):
        """Path.replace() atomically overwrites an existing target file."""
        target = tmp_path / "data.bin"
        target.write_bytes(b"original")
        source = tmp_path / "data.bin.tmp"
        source.write_bytes(b"updated")

        source.replace(target)

        assert target.read_bytes() == b"updated"
        assert not source.exists()

    def test_path_replace_creates_new_file(self, tmp_path):
        """Path.replace() creates target when it doesn't exist."""
        source = tmp_path / "data.bin.tmp"
        source.write_bytes(b"new")
        target = tmp_path / "data.bin"

        source.replace(target)

        assert target.read_bytes() == b"new"
        assert not source.exists()

    def test_shutil_move_overwrites_existing_file(self, tmp_path):
        """shutil.move() atomically overwrites an existing target file."""
        target = tmp_path / "metadata.json"
        target.write_text("original")
        source = tmp_path / "metadata.json.tmp"
        source.write_text("updated")

        shutil.move(str(source), str(target))

        assert target.read_text() == "updated"
        assert not source.exists()

    def test_shutil_move_same_volume(self, tmp_path):
        """shutil.move() within the same volume completes without error (same-fs rename)."""
        source = tmp_path / "src.json"
        source.write_text("same-volume-data")
        target = tmp_path / "dst.json"

        shutil.move(str(source), str(target))

        assert target.read_text() == "same-volume-data"
        assert not source.exists()


class TestConcurrentAtomicWrites:
    """Verify concurrent write safety through the cache abstraction layer."""

    def test_concurrent_blob_writes_no_corruption(self, tmp_path):
        """8 threads writing to same key — last-writer-wins, no corruption."""
        config = CacheConfig(cache_dir=str(tmp_path), metadata_backend="sqlite")
        cache = UnifiedCache(config=config)
        errors = []

        def writer(tid):
            try:
                cache.put(f"data-{tid}", key="same-key")
            except Exception as e:
                errors.append(f"Writer {tid}: {e}")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(writer, tid) for tid in range(8)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Write errors: {errors}"
        val = cache.get(key="same-key")
        assert val is not None, "Entry missing after concurrent writes"
        assert val.startswith("data-"), f"Unexpected value: {val!r}"
        cache.close()

    def test_concurrent_json_metadata_writes(self, tmp_path):
        """4 threads putting to distinct keys via JSON backend — all entries survive."""
        config = CacheConfig(cache_dir=str(tmp_path), metadata_backend="json")
        cache = UnifiedCache(config=config)
        errors = []

        def writer(tid):
            try:
                cache.put(f"data-{tid}", key=f"key-{tid}")
            except Exception as e:
                errors.append(f"Writer {tid}: {e}")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(writer, tid) for tid in range(4)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Write errors: {errors}"
        assert len(cache.list_entries()) == 4, (
            f"Expected 4 entries, got {len(cache.list_entries())}"
        )
        for tid in range(4):
            val = cache.get(key=f"key-{tid}")
            assert val == f"data-{tid}", f"key-{tid}: expected 'data-{tid}', got {val!r}"
        cache.close()

    def test_no_temp_file_residue_after_concurrent_writes(self, tmp_path):
        """Concurrent writes leave no .tmp files in the blob directory."""
        config = CacheConfig(cache_dir=str(tmp_path), metadata_backend="sqlite")
        cache = UnifiedCache(config=config)
        errors = []

        def writer(tid):
            try:
                for i in range(5):
                    cache.put(f"data-{tid}-{i}", key=f"entry-{tid}-{i}")
            except Exception as e:
                errors.append(f"Writer {tid}: {e}")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(writer, tid) for tid in range(4)]
            for f in as_completed(futures):
                f.result()

        assert not errors, f"Write errors: {errors}"

        # Scan for residual .tmp files in the cache directory
        tmp_files = list(Path(tmp_path).rglob("*.tmp"))
        assert not tmp_files, f"Residual .tmp files found: {tmp_files}"
        cache.close()
