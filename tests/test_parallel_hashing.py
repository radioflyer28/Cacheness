"""
Tests for parallel hashing utilities in the utils module.
"""

import pytest
import tempfile
from pathlib import Path

from cacheness.utils import (
    hash_directory_parallel,
    hash_file_content,
    _hash_single_file,
)


class TestParallelHashing:
    """Test cases for parallel directory hashing functionality."""

    def test_hash_single_file_success(self):
        """Test that _hash_single_file correctly hashes a file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test.txt"
            test_file.write_text("Hello, World!")

            result = _hash_single_file((test_file, tmp_path))
            rel_path, content_hash = result

            assert rel_path == "test.txt"
            assert isinstance(content_hash, str)
            assert len(content_hash) > 0

    def test_hash_single_file_unreadable(self):
        """Test that _hash_single_file handles unreadable files gracefully."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create a file path that doesn't exist
            nonexistent_file = tmp_path / "nonexistent.txt"

            result = _hash_single_file((nonexistent_file, tmp_path))
            rel_path, content_hash = result

            assert rel_path == "nonexistent.txt"
            assert content_hash.startswith("unreadable:")

    def test_hash_file_content(self):
        """Test single file content hashing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test.txt"
            test_file.write_text("Test content for hashing")

            hash_result = hash_file_content(test_file)

            assert isinstance(hash_result, str)
            assert len(hash_result) > 0

            # Same content should produce same hash
            hash_result2 = hash_file_content(test_file)
            assert hash_result == hash_result2

            # Different content should produce different hash
            test_file.write_text("Different content")
            hash_result3 = hash_file_content(test_file)
            assert hash_result != hash_result3

    def test_hash_file_content_missing_file(self):
        """Test hashing of missing file."""
        nonexistent = Path("/nonexistent/file.txt")
        result = hash_file_content(nonexistent)
        assert result.startswith("missing_file:")

    def test_hash_file_content_directory(self):
        """Test hashing when path is a directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result = hash_file_content(tmp_path)
            assert result.startswith("not_a_file:")

    def test_hash_directory_parallel_small(self):
        """Test parallel directory hashing with small directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create a few test files
            (tmp_path / "file1.txt").write_text("Content 1")
            (tmp_path / "file2.txt").write_text("Content 2")
            (tmp_path / "subdir").mkdir()
            (tmp_path / "subdir" / "file3.txt").write_text("Content 3")

            hash_result = hash_directory_parallel(tmp_path)

            assert isinstance(hash_result, str)
            assert len(hash_result) > 0

            # Same directory should produce same hash
            hash_result2 = hash_directory_parallel(tmp_path)
            assert hash_result == hash_result2

    def test_hash_directory_parallel_large(self):
        """Test parallel directory hashing with larger directory to trigger parallel processing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create many files to trigger parallel processing
            for i in range(20):
                (tmp_path / f"file_{i:03d}.txt").write_text(f"Content for file {i}")
                if i % 5 == 0:
                    subdir = tmp_path / f"subdir_{i}"
                    subdir.mkdir()
                    (subdir / f"nested_{i}.txt").write_text(f"Nested content {i}")

            hash_result = hash_directory_parallel(tmp_path, max_workers=4)

            assert isinstance(hash_result, str)
            assert len(hash_result) > 0

            # Same directory should produce same hash
            hash_result2 = hash_directory_parallel(tmp_path, max_workers=4)
            assert hash_result == hash_result2

            # Adding a file should change the hash
            (tmp_path / "new_file.txt").write_text("New content")
            hash_result3 = hash_directory_parallel(tmp_path, max_workers=4)
            assert hash_result != hash_result3

    def test_hash_directory_parallel_empty(self):
        """Test parallel directory hashing with empty directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            hash_result = hash_directory_parallel(tmp_path)

            assert isinstance(hash_result, str)
            assert len(hash_result) > 0
            assert (
                "empty_directory" in hash_result or len(hash_result) == 16
            )  # xxhash hex length

    def test_hash_directory_parallel_missing(self):
        """Test parallel directory hashing with missing directory."""
        nonexistent = Path("/nonexistent/directory")
        result = hash_directory_parallel(nonexistent)
        assert result.startswith("missing_directory:")

    def test_hash_directory_parallel_single_worker(self):
        """Test parallel directory hashing with single worker (sequential fallback)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create a few test files
            (tmp_path / "file1.txt").write_text("Content 1")
            (tmp_path / "file2.txt").write_text("Content 2")

            hash_result = hash_directory_parallel(tmp_path, max_workers=1)

            assert isinstance(hash_result, str)
            assert len(hash_result) > 0

            # Should produce same result as multi-worker
            hash_result_multi = hash_directory_parallel(tmp_path, max_workers=4)
            assert hash_result == hash_result_multi

    def test_hash_consistency_across_methods(self):
        """Test that both parallel and sequential methods produce consistent results."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create test files
            (tmp_path / "file1.txt").write_text("Content 1")
            (tmp_path / "file2.txt").write_text("Content 2")
            (tmp_path / "subdir").mkdir()
            (tmp_path / "subdir" / "file3.txt").write_text("Content 3")

            # Test with different worker counts
            hash_1_worker = hash_directory_parallel(tmp_path, max_workers=1)
            hash_4_workers = hash_directory_parallel(tmp_path, max_workers=4)

            # Should be identical regardless of worker count
            assert hash_1_worker == hash_4_workers

    def test_smart_threshold_logic(self):
        """Test that the smart threshold correctly chooses sequential vs parallel processing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Test case 1: Small directory (should use sequential)
            for i in range(5):
                (tmp_path / f"small_{i}.txt").write_text("small content")

            # This should complete very quickly (sequential)
            import time

            start = time.time()
            result1 = hash_directory_parallel(tmp_path)
            time1 = time.time() - start

            # Should be fast (< 0.1s) indicating sequential processing
            assert time1 < 0.1, (
                f"Small directory took {time1:.3f}s, expected sequential processing"
            )
            assert isinstance(result1, str) and len(result1) > 0

            # Test case 2: Many files (should use parallel)
            # Add more files to trigger parallel processing
            for i in range(25):  # Total of 30 files
                (tmp_path / f"many_{i}.txt").write_text(f"content {i}")

            result2 = hash_directory_parallel(tmp_path)

            # This might take longer due to parallel overhead, but should still work
            assert isinstance(result2, str) and len(result2) > 0
            assert result1 != result2  # Should be different due to additional files
