"""
Tests for the file_hashing module.

This module tests file hashing utility functions including parallel directory hashing,
file content hashing, multiprocessing logic, error handling, and edge cases.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import xxhash

from cacheness.file_hashing import (
    _hash_single_file,
    hash_directory_parallel,
    _hash_directory_sequential,
    hash_file_content,
)


class TestHashSingleFile:
    """Test the _hash_single_file function."""

    def test_successful_file_hashing(self):
        """Test successful hashing of a regular file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.txt"
            test_content = b"Hello, World!"
            test_file.write_bytes(test_content)

            # Test the function
            rel_path, content_hash = _hash_single_file((test_file, temp_path))

            # Verify relative path
            assert rel_path == "test.txt"

            # Verify hash is correct
            expected_hash = xxhash.xxh3_64(test_content).hexdigest()
            assert content_hash == expected_hash

    def test_nested_file_hashing(self):
        """Test hashing of files in nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            nested_dir = temp_path / "subdir" / "nested"
            nested_dir.mkdir(parents=True)

            test_file = nested_dir / "nested_file.txt"
            test_content = b"Nested content"
            test_file.write_bytes(test_content)

            rel_path, content_hash = _hash_single_file((test_file, temp_path))

            # Check relative path includes subdirectories
            assert rel_path == "subdir/nested/nested_file.txt"

            # Verify hash
            expected_hash = xxhash.xxh3_64(test_content).hexdigest()
            assert content_hash == expected_hash

    def test_large_file_chunked_reading(self):
        """Test that large files are read in chunks correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            large_file = temp_path / "large.txt"

            # Create a file larger than the chunk size (8192 bytes)
            large_content = b"A" * 10000  # 10KB file
            large_file.write_bytes(large_content)

            rel_path, content_hash = _hash_single_file((large_file, temp_path))

            assert rel_path == "large.txt"

            # Verify hash matches direct hashing
            expected_hash = xxhash.xxh3_64(large_content).hexdigest()
            assert content_hash == expected_hash

    def test_empty_file_hashing(self):
        """Test hashing of empty files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            empty_file = temp_path / "empty.txt"
            empty_file.write_bytes(b"")

            rel_path, content_hash = _hash_single_file((empty_file, temp_path))

            assert rel_path == "empty.txt"

            # Empty file should have the hash of empty bytes
            expected_hash = xxhash.xxh3_64(b"").hexdigest()
            assert content_hash == expected_hash

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_unreadable_file_fallback(self, mock_open_func):
        """Test fallback behavior when file cannot be read."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "unreadable.txt"
            test_file.write_text("content")  # Create the file first

            # Mock pathlib.Path.stat to return file size
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 1024

                rel_path, content_hash = _hash_single_file((test_file, temp_path))

                assert rel_path == "unreadable.txt"
                assert content_hash == "unreadable:1024"

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_unreadable_file_no_stat(self, mock_open_func):
        """Test fallback when both file reading and stat fail."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "unreadable.txt"
            test_file.write_text("content")  # Create the file first

            # Mock pathlib.Path.stat to also fail
            with patch("pathlib.Path.stat", side_effect=OSError("Stat failed")):
                rel_path, content_hash = _hash_single_file((test_file, temp_path))

                assert rel_path == "unreadable.txt"
                assert content_hash == "unreadable:unknown"


class TestHashDirectoryParallel:
    """Test the hash_directory_parallel function."""

    def test_empty_directory(self):
        """Test hashing of an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            result = hash_directory_parallel(temp_path)

            # Should return a hash based on the empty directory identifier
            expected = xxhash.xxh3_64(
                f"empty_directory:{str(temp_path)}".encode()
            ).hexdigest()
            assert result == expected

    def test_missing_directory(self):
        """Test behavior with non-existent directory."""
        non_existent = Path("/non/existent/directory")

        result = hash_directory_parallel(non_existent)

        assert result == f"missing_directory:{str(non_existent)}"

    def test_not_a_directory(self):
        """Test behavior when path is not a directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            file_path = Path(temp_file.name)

            result = hash_directory_parallel(file_path)

            assert result == f"missing_directory:{str(file_path)}"

    def test_single_file_directory(self):
        """Test directory with a single file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "single.txt"
            test_content = b"Single file content"
            test_file.write_bytes(test_content)

            result = hash_directory_parallel(temp_path)

            # Should be a valid hash string
            assert isinstance(result, str)
            assert len(result) == 16  # xxhash.xxh3_64 produces 16-character hex strings

            # Result should be consistent
            result2 = hash_directory_parallel(temp_path)
            assert result == result2

    def test_multiple_files_directory(self):
        """Test directory with multiple files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple files
            files_data = [
                ("file1.txt", b"Content of file 1"),
                ("file2.txt", b"Content of file 2"),
                ("file3.txt", b"Content of file 3"),
            ]

            for filename, content in files_data:
                (temp_path / filename).write_bytes(content)

            result = hash_directory_parallel(temp_path)

            assert isinstance(result, str)
            assert len(result) == 16

            # Should be deterministic
            result2 = hash_directory_parallel(temp_path)
            assert result == result2

    def test_nested_directory_structure(self):
        """Test directory with nested subdirectories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create nested structure
            (temp_path / "dir1").mkdir()
            (temp_path / "dir1" / "file1.txt").write_bytes(b"Nested file 1")
            (temp_path / "dir2").mkdir()
            (temp_path / "dir2" / "subdir").mkdir()
            (temp_path / "dir2" / "subdir" / "file2.txt").write_bytes(
                b"Deeply nested file"
            )
            (temp_path / "root_file.txt").write_bytes(b"Root level file")

            result = hash_directory_parallel(temp_path)

            assert isinstance(result, str)
            assert len(result) == 16

            # Should be deterministic
            result2 = hash_directory_parallel(temp_path)
            assert result == result2

    def test_sequential_vs_parallel_consistency(self):
        """Test that sequential and parallel methods produce the same result."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a directory structure
            for i in range(5):
                (temp_path / f"file_{i}.txt").write_bytes(f"Content {i}".encode())

            # Get file paths for sequential method
            file_paths = [(f, temp_path) for f in temp_path.rglob("*") if f.is_file()]

            # Compare results
            parallel_result = hash_directory_parallel(temp_path)
            sequential_result = _hash_directory_sequential(temp_path, file_paths)

            assert parallel_result == sequential_result

    def test_max_workers_parameter(self):
        """Test that max_workers parameter is respected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some files
            for i in range(3):
                (temp_path / f"file_{i}.txt").write_bytes(f"Content {i}".encode())

            # Test with different worker counts
            result1 = hash_directory_parallel(temp_path, max_workers=1)
            result2 = hash_directory_parallel(temp_path, max_workers=2)

            # Results should be the same regardless of worker count
            assert result1 == result2

    def test_small_directory_uses_sequential(self):
        """Test that small directories use sequential processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a small directory (below thresholds)
            (temp_path / "small.txt").write_bytes(b"Small content")

            with patch(
                "cacheness.file_hashing._hash_directory_sequential"
            ) as mock_sequential:
                mock_sequential.return_value = "mocked_hash"

                result = hash_directory_parallel(temp_path)

                # Should use sequential method
                mock_sequential.assert_called_once()
                assert result == "mocked_hash"

    def test_large_directory_uses_parallel(self):
        """Test that large directories trigger parallel processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create many files to trigger parallel processing (above the 80 file threshold)
            for i in range(100):
                (temp_path / f"file_{i:03d}.txt").write_bytes(f"Content {i}".encode())

            # Mock ProcessPoolExecutor at the module level where it's imported
            with patch(
                "cacheness.file_hashing.ProcessPoolExecutor"
            ) as mock_executor_class:
                # Create a mock executor instance
                mock_executor = Mock()
                mock_executor.__enter__ = Mock(return_value=mock_executor)
                mock_executor.__exit__ = Mock(return_value=None)
                mock_executor.submit = Mock()
                mock_executor_class.return_value = mock_executor

                # Create mock futures that will return valid file hash results
                mock_futures = []
                for i in range(100):
                    mock_future = Mock()
                    mock_future.result.return_value = (f"file_{i:03d}.txt", f"hash_{i}")
                    mock_futures.append(mock_future)

                # Mock as_completed to return our mock futures
                with patch(
                    "cacheness.file_hashing.as_completed", return_value=mock_futures
                ):
                    result = hash_directory_parallel(temp_path)

                    # Should have attempted to create ProcessPoolExecutor
                    mock_executor_class.assert_called_once()
                    # Should return a hash (not checking exact value since it's based on mock data)
                    assert isinstance(result, str)
                    assert len(result) == 16  # xxhash produces 16-character hex strings

    def test_parallel_processing_fallback(self):
        """Test fallback to sequential when parallel processing fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files that would trigger parallel processing (above 80 file threshold)
            for i in range(100):
                (temp_path / f"file_{i:03d}.txt").write_bytes(f"Content {i}".encode())

            # Mock ProcessPoolExecutor at the module level to fail
            with patch(
                "cacheness.file_hashing.ProcessPoolExecutor",
                side_effect=Exception("Process pool failed"),
            ):
                with patch(
                    "cacheness.file_hashing._hash_directory_sequential"
                ) as mock_sequential:
                    mock_sequential.return_value = "fallback_hash"

                    result = hash_directory_parallel(temp_path)

                    # Should fall back to sequential
                    mock_sequential.assert_called_once()
                    assert result == "fallback_hash"

    def test_file_hashing_error_handling(self):
        """Test handling of file hashing errors in parallel processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create many files to ensure parallel processing
            for i in range(100):
                (temp_path / f"file_{i:03d}.txt").write_bytes(f"Content {i}".encode())

            # Mock _hash_single_file to fail for some files
            original_hash_single_file = _hash_single_file

            def mock_hash_with_errors(file_info):
                file_path, base_path = file_info
                if "050" in file_path.name:  # Fail for file_050.txt
                    raise Exception("Simulated file error")
                return original_hash_single_file(file_info)

            with patch(
                "cacheness.file_hashing._hash_single_file",
                side_effect=mock_hash_with_errors,
            ):
                result = hash_directory_parallel(temp_path)

                # Should still return a valid hash
                assert isinstance(result, str)
                assert len(result) == 16

    def test_deterministic_ordering(self):
        """Test that file ordering is deterministic regardless of creation order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files in one order
            files1 = ["zebra.txt", "alpha.txt", "beta.txt"]
            for filename in files1:
                (temp_path / filename).write_bytes(f"Content of {filename}".encode())

            result1 = hash_directory_parallel(temp_path)

            # Clear directory
            for f in temp_path.iterdir():
                f.unlink()

            # Create same files in different order
            files2 = ["alpha.txt", "zebra.txt", "beta.txt"]
            for filename in files2:
                (temp_path / filename).write_bytes(f"Content of {filename}".encode())

            result2 = hash_directory_parallel(temp_path)

            # Results should be identical
            assert result1 == result2


class TestHashDirectorySequential:
    """Test the _hash_directory_sequential function."""

    def test_empty_file_list(self):
        """Test sequential hashing with empty file list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            result = _hash_directory_sequential(temp_path, [])

            # Should return hash of empty content
            expected = xxhash.xxh3_64(b"").hexdigest()
            assert result == expected

    def test_single_file_sequential(self):
        """Test sequential hashing with single file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.txt"
            test_content = b"Test content"
            test_file.write_bytes(test_content)

            file_paths = [(test_file, temp_path)]
            result = _hash_directory_sequential(temp_path, file_paths)

            assert isinstance(result, str)
            assert len(result) == 16

    def test_multiple_files_sequential(self):
        """Test sequential hashing with multiple files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            files_data = [
                ("file1.txt", b"Content 1"),
                ("file2.txt", b"Content 2"),
                ("file3.txt", b"Content 3"),
            ]

            file_paths = []
            for filename, content in files_data:
                file_path = temp_path / filename
                file_path.write_bytes(content)
                file_paths.append((file_path, temp_path))

            result = _hash_directory_sequential(temp_path, file_paths)

            assert isinstance(result, str)
            assert len(result) == 16

    def test_unreadable_file_sequential(self):
        """Test sequential handling of unreadable files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.txt"
            test_file.write_bytes(b"content")

            file_paths = [(test_file, temp_path)]

            # Mock file opening to fail
            with patch("builtins.open", side_effect=OSError("Permission denied")):
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 1024

                    result = _hash_directory_sequential(temp_path, file_paths)

                    # Should still return a valid hash
                    assert isinstance(result, str)
                    assert len(result) == 16

    def test_file_order_independence(self):
        """Test that file order doesn't affect the result due to internal sorting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files
            file1 = temp_path / "zzz.txt"
            file2 = temp_path / "aaa.txt"
            file1.write_bytes(b"Content Z")
            file2.write_bytes(b"Content A")

            # Test with different input orders
            file_paths1 = [(file1, temp_path), (file2, temp_path)]
            file_paths2 = [(file2, temp_path), (file1, temp_path)]

            result1 = _hash_directory_sequential(temp_path, file_paths1)
            result2 = _hash_directory_sequential(temp_path, file_paths2)

            # Results should be identical due to internal sorting
            assert result1 == result2


class TestHashFileContent:
    """Test the hash_file_content function."""

    def test_regular_file_hashing(self):
        """Test hashing of a regular file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        content = b"Hello, file content hashing!"
        temp_path.write_bytes(content)

        try:
            result = hash_file_content(temp_path)

            # Should return hash of the content
            expected = xxhash.xxh3_64(content).hexdigest()
            assert result == expected
        finally:
            temp_path.unlink()

    def test_empty_file_hashing(self):
        """Test hashing of an empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        temp_path.write_bytes(b"")

        try:
            result = hash_file_content(temp_path)

            # Should return hash of empty content
            expected = xxhash.xxh3_64(b"").hexdigest()
            assert result == expected
        finally:
            temp_path.unlink()

    def test_large_file_hashing(self):
        """Test hashing of a large file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        # Create content larger than chunk size
        large_content = b"A" * 20000  # 20KB
        temp_path.write_bytes(large_content)

        try:
            result = hash_file_content(temp_path)

            # Should return correct hash
            expected = xxhash.xxh3_64(large_content).hexdigest()
            assert result == expected
        finally:
            temp_path.unlink()

    def test_missing_file(self):
        """Test behavior with non-existent file."""
        non_existent = Path("/non/existent/file.txt")

        result = hash_file_content(non_existent)

        assert result == f"missing_file:{str(non_existent)}"

    def test_directory_instead_of_file(self):
        """Test behavior when path points to a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)

            result = hash_file_content(dir_path)

            assert result == f"not_a_file:{str(dir_path)}"

    @patch("builtins.open", side_effect=OSError("Permission denied"))
    def test_unreadable_file(self, mock_open_func):
        """Test behavior when file cannot be read."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            result = hash_file_content(temp_path)

            # Should return error indicator
            assert result.startswith(f"error_reading:{str(temp_path)}")
            assert "Permission denied" in result
        finally:
            temp_path.unlink()

    def test_chunk_reading_consistency(self):
        """Test that chunked reading produces consistent results."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        # Create content that spans multiple chunks
        content = b"0123456789" * 1000  # 10KB content
        temp_path.write_bytes(content)

        try:
            result1 = hash_file_content(temp_path)
            result2 = hash_file_content(temp_path)

            # Should be consistent
            assert result1 == result2

            # Should match direct hashing
            expected = xxhash.xxh3_64(content).hexdigest()
            assert result1 == expected
        finally:
            temp_path.unlink()


class TestFileHashingIntegration:
    """Integration tests for file_hashing module functions."""

    def test_parallel_vs_sequential_large_directory(self):
        """Test that parallel and sequential methods agree on large directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a large directory structure
            for i in range(20):
                subdir = temp_path / f"subdir_{i:02d}"
                subdir.mkdir()
                for j in range(5):
                    file_path = subdir / f"file_{j}.txt"
                    file_path.write_bytes(f"Content {i}-{j}".encode())

            # Force sequential processing by mocking the threshold check
            with patch("cacheness.file_hashing.cpu_count", return_value=1):
                sequential_result = hash_directory_parallel(temp_path, max_workers=1)

            # Use parallel processing
            parallel_result = hash_directory_parallel(temp_path, max_workers=4)

            # Results should be identical
            assert sequential_result == parallel_result

    def test_directory_modification_affects_hash(self):
        """Test that directory changes affect the hash."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create initial file
            (temp_path / "file1.txt").write_bytes(b"Initial content")
            initial_hash = hash_directory_parallel(temp_path)

            # Add another file
            (temp_path / "file2.txt").write_bytes(b"Additional content")
            modified_hash = hash_directory_parallel(temp_path)

            # Hashes should be different
            assert initial_hash != modified_hash

            # Modify existing file
            (temp_path / "file1.txt").write_bytes(b"Modified content")
            final_hash = hash_directory_parallel(temp_path)

            # Hash should change again
            assert final_hash != modified_hash
            assert final_hash != initial_hash

    def test_error_resilience(self):
        """Test that the system is resilient to various errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some normal files
            (temp_path / "normal1.txt").write_bytes(b"Normal content 1")
            (temp_path / "normal2.txt").write_bytes(b"Normal content 2")

            # Create a file that will be problematic
            problematic_file = temp_path / "problematic.txt"
            problematic_file.write_bytes(b"This will cause issues")

            # Mock file operations to simulate various failures
            original_open = open

            def selective_mock_open(file_path, *args, **kwargs):
                if "problematic" in str(file_path):
                    raise OSError("Simulated file error")
                return original_open(file_path, *args, **kwargs)

            with patch("builtins.open", side_effect=selective_mock_open):
                # Should still work despite errors
                result = hash_directory_parallel(temp_path)

                assert isinstance(result, str)
                assert len(result) == 16

    def test_memory_usage_with_large_files(self):
        """Test memory efficiency with large files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a moderately large file (1MB)
            large_content = b"X" * (1024 * 1024)  # 1MB
            large_file = temp_path / "large_file.dat"
            large_file.write_bytes(large_content)

            # Hash using both file hashing and directory hashing
            file_hash = hash_file_content(large_file)
            dir_hash = hash_directory_parallel(temp_path)

            # Both should work without memory issues
            assert isinstance(file_hash, str)
            assert len(file_hash) == 16
            assert isinstance(dir_hash, str)
            assert len(dir_hash) == 16

    def test_concurrent_access_simulation(self):
        """Test behavior under simulated concurrent access."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            for i in range(10):
                (temp_path / f"file_{i}.txt").write_bytes(f"Content {i}".encode())

            # Simulate multiple concurrent hash operations
            results = []
            for _ in range(5):
                result = hash_directory_parallel(temp_path)
                results.append(result)

            # All results should be identical
            assert all(r == results[0] for r in results)

    def test_performance_threshold_logic(self):
        """Test the logic for choosing between parallel and sequential processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test with exactly threshold number of files (80)
            for i in range(80):
                (temp_path / f"file_{i:03d}.txt").write_bytes(f"Content {i}".encode())

            # Should use sequential (below threshold)
            with patch(
                "cacheness.file_hashing._hash_directory_sequential"
            ) as mock_sequential:
                mock_sequential.return_value = "sequential_hash"
                result = hash_directory_parallel(temp_path)
                mock_sequential.assert_called_once()

            # Add one more file to exceed threshold
            (temp_path / "file_080.txt").write_bytes(b"Content 80")

            # Should now attempt parallel processing (81 files > 80 threshold)
            with patch(
                "cacheness.file_hashing.ProcessPoolExecutor"
            ) as mock_executor_class:
                # Create a mock executor instance
                mock_executor = Mock()
                mock_executor.__enter__ = Mock(return_value=mock_executor)
                mock_executor.__exit__ = Mock(return_value=None)
                mock_executor.submit = Mock()
                mock_executor_class.return_value = mock_executor

                with patch("cacheness.file_hashing.as_completed", return_value=[]):
                    with patch(
                        "cacheness.file_hashing._hash_directory_sequential"
                    ) as mock_sequential:
                        mock_sequential.return_value = "fallback_hash"
                        result = hash_directory_parallel(temp_path)
                        # Should have attempted to use ProcessPoolExecutor
                        mock_executor_class.assert_called_once()


class TestFileHashingErrorHandling:
    """Test error handling and edge cases in file_hashing functions."""

    def test_invalid_input_types(self):
        """Test behavior with invalid input types."""
        # Test with None
        result = hash_directory_parallel(None)
        assert "missing_directory:None" in result

        result = hash_file_content(None)
        assert "missing_file:None" in result

    def test_unicode_file_paths(self):
        """Test handling of unicode characters in file paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files with unicode names
            unicode_file = temp_path / "测试文件.txt"
            unicode_file.write_bytes("Unicode content".encode("utf-8"))

            emoji_file = temp_path / "📁_emoji_file.txt"
            emoji_file.write_bytes("Emoji content".encode("utf-8"))

            # Should handle unicode paths correctly
            dir_result = hash_directory_parallel(temp_path)
            file_result = hash_file_content(unicode_file)

            assert isinstance(dir_result, str)
            assert len(dir_result) == 16
            assert isinstance(file_result, str)
            assert len(file_result) == 16

    def test_symlink_handling(self):
        """Test handling of symbolic links."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a regular file
            original_file = temp_path / "original.txt"
            original_file.write_bytes(b"Original content")

            # Create a symlink (if supported by the OS)
            symlink_file = temp_path / "symlink.txt"
            try:
                symlink_file.symlink_to(original_file)

                # Hash the directory containing the symlink
                result = hash_directory_parallel(temp_path)

                assert isinstance(result, str)
                assert len(result) == 16

            except OSError:
                # Skip test if symlinks not supported
                pytest.skip("Symbolic links not supported on this system")

    def test_very_long_file_paths(self):
        """Test handling of very long file paths."""
        import platform

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # On Windows, limit path depth to avoid MAX_PATH issues
            max_depth = 5 if platform.system() == "Windows" else 10

            # Create nested directories to build a long path
            current_path = temp_path
            try:
                for i in range(max_depth):  # Create deep nesting
                    current_path = current_path / f"very_long_directory_name_{i:02d}"
                    current_path.mkdir()

                # Create a file with a long name in the deep directory
                # On Windows, use shorter filename
                filename_length = 50 if platform.system() == "Windows" else 100
                long_filename = "very_long_filename_" + "x" * filename_length + ".txt"
                long_file = current_path / long_filename
                long_file.write_bytes(b"Content in deeply nested file")

                # Should handle long paths correctly
                result = hash_directory_parallel(temp_path)

                assert isinstance(result, str)
                assert len(result) == 16
            except (OSError, FileNotFoundError) as e:
                # Skip test if path is too long for the system
                pytest.skip(f"Path too long for system: {e}")

    def test_special_characters_in_content(self):
        """Test handling of special characters in file content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create files with various special content
            binary_file = temp_path / "binary.dat"
            binary_file.write_bytes(bytes(range(256)))  # All byte values

            null_file = temp_path / "nulls.dat"
            null_file.write_bytes(b"\x00" * 1000)  # Null bytes

            mixed_file = temp_path / "mixed.txt"
            mixed_file.write_bytes("Mixed: \x00\xff\n\r\t".encode())

            # Should handle all content types
            dir_result = hash_directory_parallel(temp_path)
            binary_result = hash_file_content(binary_file)

            assert isinstance(dir_result, str)
            assert len(dir_result) == 16
            assert isinstance(binary_result, str)
            assert len(binary_result) == 16
