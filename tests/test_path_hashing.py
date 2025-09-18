#!/usr/bin/env python3
"""
Test Path content hashing functionality.
"""

import tempfile
import pytest
from pathlib import Path

from cacheness import cacheness, CacheConfig


class TestPathHashing:
    """Test that Path objects are hashed by content, not path string."""

    def test_same_content_different_paths_retrieval(self):
        """Test that files with same content in different paths can retrieve cached data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir + "/cache", metadata_backend="json")
            cache = cacheness(config)

            # Create files with same content in different locations
            file1_path = Path(temp_dir) / "file1.txt"
            file2_path = Path(temp_dir) / "subdir" / "file2.txt"
            file2_path.parent.mkdir(exist_ok=True)

            # Same content for both files
            content = "This is test content for hashing"
            file1_path.write_text(content)
            file2_path.write_text(content)

            test_data = {"message": "test data"}

            # Cache data with file1 as key
            cache.put(test_data, description="Test with file1", input_file=file1_path)

            # Should be able to retrieve with file2 (same content, different path)
            result = cache.get(input_file=file2_path)
            assert result is not None
            assert result == test_data

    def test_different_content_cache_miss(self):
        """Test that files with different content produce cache misses."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir + "/cache", metadata_backend="json")
            cache = cacheness(config)

            # Create files with different content
            file1_path = Path(temp_dir) / "file1.txt"
            file2_path = Path(temp_dir) / "file2.txt"

            file1_path.write_text("Content 1")
            file2_path.write_text("Content 2")

            test_data = {"message": "test data"}

            # Cache data with file1
            cache.put(test_data, description="Test with file1", input_file=file1_path)

            # Should NOT be able to retrieve with file2 (different content)
            result = cache.get(input_file=file2_path)
            assert result is None

    def test_file_modification_invalidates_cache(self):
        """Test that cache invalidates when file content changes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir + "/cache", metadata_backend="json")
            cache = cacheness(config)

            test_file = Path(temp_dir) / "mutable_file.txt"

            # Initial content
            test_file.write_text("Initial content")
            test_data1 = {"version": 1, "data": "first"}

            cache.put(test_data1, description="Version 1", input_file=test_file)
            result1 = cache.get(input_file=test_file)
            assert result1 is not None

            # Modify file content
            test_file.write_text("Modified content")

            # Should not find cached data (different content hash)
            result2 = cache.get(input_file=test_file)
            assert result2 is None

            # Cache new data with modified file
            test_data2 = {"version": 2, "data": "second"}
            cache.put(test_data2, description="Version 2", input_file=test_file)
            result3 = cache.get(input_file=test_file)
            assert result3 is not None
            assert result3 == test_data2

    def test_directory_content_hashing(self):
        """Test that directories are hashed based on their contents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir + "/cache", metadata_backend="json")
            cache = cacheness(config)

            # Create directories with identical content structure
            dir1 = Path(temp_dir) / "dir1"
            dir2 = Path(temp_dir) / "dir2"
            dir1.mkdir()
            dir2.mkdir()

            # Same directory structure and content
            (dir1 / "file1.txt").write_text("Content 1")
            (dir1 / "file2.txt").write_text("Content 2")
            (dir2 / "file1.txt").write_text("Content 1")
            (dir2 / "file2.txt").write_text("Content 2")

            test_data = {"directory_data": "test"}

            # Cache with dir1
            cache.put(test_data, description="Directory test", input_dir=dir1)

            # Should find with dir2 (same content structure)
            result = cache.get(input_dir=dir2)
            assert result is not None
            assert result == test_data

    def test_missing_file_handling(self):
        """Test handling of missing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir + "/cache", metadata_backend="json")
            cache = cacheness(config)

            missing_file = Path(temp_dir) / "missing.txt"
            test_data = {"missing": "file test"}

            # Cache with missing file - should work (uses path string)
            cache.put(
                test_data, description="Missing file test", input_file=missing_file
            )
            result = cache.get(input_file=missing_file)
            assert result is not None
            assert result == test_data


if __name__ == "__main__":
    pytest.main([__file__])
