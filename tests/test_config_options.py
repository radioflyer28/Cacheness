#!/usr/bin/env python3
"""
Test the hash_path_content configuration option.
"""

import tempfile
import unittest
from pathlib import Path

from cacheness import cacheness, CacheConfig


class TestPathHashingConfig(unittest.TestCase):
    """Test that the hash_path_content configuration option works correctly."""

    def test_hash_path_content_enabled(self):
        """Test that with hash_path_content=True, files with same content can be retrieved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=temp_dir + "/cache",
                metadata_backend="json",
                hash_path_content=True  # Enable content hashing
            )
            cache = cacheness(config)
            
            # Create files with same content in different locations
            file1_path = Path(temp_dir) / "file1.txt"
            file2_path = Path(temp_dir) / "subdir" / "file2.txt"
            file2_path.parent.mkdir(exist_ok=True)
            
            # Same content for both files
            content = "This is test content for hashing"
            file1_path.write_text(content)
            file2_path.write_text(content)
            
            test_data = {"message": "content hashing test"}
            
            # Cache data with file1 as key
            cache.put(test_data, description="Test with content hashing", input_file=file1_path)
            
            # Should be able to retrieve with file2 (same content, different path)
            result = cache.get(input_file=file2_path)
            self.assertIsNotNone(result)
            self.assertEqual(result, test_data)

    def test_hash_path_content_disabled(self):
        """Test that with hash_path_content=False, files with same content but different paths produce cache misses."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=temp_dir + "/cache",
                metadata_backend="json",
                hash_path_content=False  # Disable content hashing
            )
            cache = cacheness(config)
            
            # Create files with same content in different locations
            file1_path = Path(temp_dir) / "file1.txt"
            file2_path = Path(temp_dir) / "subdir" / "file2.txt"
            file2_path.parent.mkdir(exist_ok=True)
            
            # Same content for both files
            content = "This is test content for hashing"
            file1_path.write_text(content)
            file2_path.write_text(content)
            
            test_data = {"message": "filename hashing test"}
            
            # Cache data with file1 as key
            cache.put(test_data, description="Test with filename hashing", input_file=file1_path)
            
            # Should NOT be able to retrieve with file2 (different path, filename-based hashing)
            result = cache.get(input_file=file2_path)
            self.assertIsNone(result)

    def test_hash_path_content_disabled_same_path_works(self):
        """Test that with hash_path_content=False, the same path still works."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=temp_dir + "/cache",
                metadata_backend="json",
                hash_path_content=False  # Disable content hashing
            )
            cache = cacheness(config)
            
            file_path = Path(temp_dir) / "file.txt"
            file_path.write_text("Some content")
            
            test_data = {"message": "same path test"}
            
            # Cache and retrieve with the same path
            cache.put(test_data, description="Test same path", input_file=file_path)
            result = cache.get(input_file=file_path)
            
            self.assertIsNotNone(result)
            self.assertEqual(result, test_data)

    def test_hash_path_content_disabled_content_change_not_detected(self):
        """Test that with hash_path_content=False, content changes are not detected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                cache_dir=temp_dir + "/cache",
                metadata_backend="json",
                hash_path_content=False  # Disable content hashing
            )
            cache = cacheness(config)
            
            file_path = Path(temp_dir) / "mutable_file.txt"
            
            # Initial content and caching
            file_path.write_text("Initial content")
            test_data1 = {"version": 1, "data": "first"}
            cache.put(test_data1, description="Version 1", input_file=file_path)
            
            # Modify file content
            file_path.write_text("Modified content")
            
            # Should still find cached data (content change not detected with filename-based hashing)
            result = cache.get(input_file=file_path)
            self.assertIsNotNone(result)
            self.assertEqual(result, test_data1)  # Original data should be returned

    def test_default_config_enables_content_hashing(self):
        """Test that the default configuration enables content hashing."""
        config = CacheConfig()
        self.assertTrue(config.hash_path_content)


if __name__ == "__main__":
    unittest.main()
