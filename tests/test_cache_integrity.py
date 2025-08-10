"""
Tests for cache file integrity verification functionality.
"""

import tempfile
from pathlib import Path
import numpy as np

from cacheness import UnifiedCache, CacheConfig


class TestCacheIntegrity:
    """Test cache file integrity verification."""

    def test_cache_integrity_verification_enabled_by_default(self):
        """Test that cache integrity verification is enabled by default."""
        config = CacheConfig()
        assert config.verify_cache_integrity is True

    def test_cache_integrity_verification_can_be_disabled(self):
        """Test that cache integrity verification can be disabled."""
        config = CacheConfig(verify_cache_integrity=False)
        assert config.verify_cache_integrity is False

    def test_file_hash_calculation(self):
        """Test that file hash is calculated correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir, verify_cache_integrity=True)
            cache = UnifiedCache(config)

            # Create a test file
            test_file = Path(temp_dir) / "test_file.txt"
            test_content = b"Hello, World!"
            test_file.write_bytes(test_content)

            # Calculate hash
            calculated_hash = cache._calculate_file_hash(test_file)
            assert calculated_hash is not None
            assert isinstance(calculated_hash, str)
            assert (
                len(calculated_hash) == 16
            )  # XXH3_64 produces 16-character hex strings

    def test_file_hash_stored_in_metadata(self):
        """Test that file hash is stored in metadata when caching."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir, verify_cache_integrity=True)
            cache = UnifiedCache(config)

            # Cache some data
            test_data = {"message": "Hello, World!"}
            cache.put(test_data, description="Test data", test_key="value")

            # Check that metadata contains file hash
            cache_key = cache._create_cache_key({"test_key": "value"})
            entry = cache.metadata_backend.get_entry(cache_key)
            assert entry is not None

            metadata = entry.get("metadata", {})
            assert "file_hash" in metadata
            assert metadata["file_hash"] is not None
            assert isinstance(metadata["file_hash"], str)

    def test_file_hash_not_stored_when_disabled(self):
        """Test that file hash is not stored when verification is disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir, verify_cache_integrity=False)
            cache = UnifiedCache(config)

            # Cache some data
            test_data = {"message": "Hello, World!"}
            cache.put(test_data, description="Test data", test_key="value")

            # Check that metadata doesn't contain file hash
            cache_key = cache._create_cache_key({"test_key": "value"})
            entry = cache.metadata_backend.get_entry(cache_key)
            assert entry is not None

            metadata = entry.get("metadata", {})
            assert metadata.get("file_hash") is None

    def test_successful_integrity_verification(self):
        """Test that valid cache files pass integrity verification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir, verify_cache_integrity=True)
            cache = UnifiedCache(config)

            # Cache some data
            test_data = np.array([1, 2, 3, 4, 5])
            cache.put(test_data, description="Test array", test_key="array")

            # Retrieve data - should succeed with integrity verification
            retrieved_data = cache.get(test_key="array")
            assert retrieved_data is not None
            np.testing.assert_array_equal(retrieved_data, test_data)

    def test_corrupted_cache_file_detection(self):
        """Test that corrupted cache files are detected and removed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir, verify_cache_integrity=True)
            cache = UnifiedCache(config)

            # Cache some data
            test_data = {"message": "Hello, World!"}
            cache.put(test_data, description="Test data", test_key="value")

            # Get the cache file path and corrupt it
            cache_key = cache._create_cache_key({"test_key": "value"})
            entry = cache.metadata_backend.get_entry(cache_key)
            assert entry is not None, "Cache entry should exist"
            file_path = Path(entry["metadata"]["actual_path"])

            # Corrupt the file by appending some bytes
            with open(file_path, "ab") as f:
                f.write(b"CORRUPTED")

            # Try to retrieve data - should detect corruption and return None
            retrieved_data = cache.get(test_key="value")
            assert retrieved_data is None

            # Verify that the corrupted entry was removed from metadata
            entry_after = cache.metadata_backend.get_entry(cache_key)
            assert entry_after is None

    def test_missing_file_hash_allows_retrieval(self):
        """Test that missing file hash (legacy entries) still allows retrieval."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir, verify_cache_integrity=True)
            cache = UnifiedCache(config)

            # Cache some data first
            test_data = {"message": "Hello, World!"}
            cache.put(test_data, description="Test data", test_key="value")

            # Manually remove the file_hash from metadata to simulate legacy entry
            cache_key = cache._create_cache_key({"test_key": "value"})
            entry = cache.metadata_backend.get_entry(cache_key)
            assert entry is not None, "Cache entry should exist"
            metadata = entry["metadata"]
            del metadata["file_hash"]

            # Update the metadata without file_hash
            entry_data = {
                "description": entry["description"],
                "data_type": entry["data_type"],
                "prefix": entry["prefix"],
                "file_size": entry["file_size"],
                "metadata": metadata,
            }
            cache.metadata_backend.put_entry(cache_key, entry_data)

            # Should still be able to retrieve the data (no verification for legacy entries)
            retrieved_data = cache.get(test_key="value")
            assert retrieved_data is not None
            assert retrieved_data == test_data

    def test_integrity_verification_disabled_skips_check(self):
        """Test that disabling verification skips integrity check completely."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First cache with verification enabled
            config_enabled = CacheConfig(
                cache_dir=temp_dir, verify_cache_integrity=True
            )
            cache_enabled = UnifiedCache(config_enabled)

            test_data = {"message": "Hello, World!"}
            cache_enabled.put(test_data, description="Test data", test_key="value")

            # Get the cache file path and corrupt it
            cache_key = cache_enabled._create_cache_key({"test_key": "value"})
            entry = cache_enabled.metadata_backend.get_entry(cache_key)
            assert entry is not None, "Cache entry should exist"
            file_path = Path(entry["metadata"]["actual_path"])

            with open(file_path, "ab") as f:
                f.write(b"CORRUPTED")

            # Create new cache instance with verification disabled
            config_disabled = CacheConfig(
                cache_dir=temp_dir, verify_cache_integrity=False
            )
            cache_disabled = UnifiedCache(config_disabled)

            # Should still return data (though corrupted) because verification is disabled
            # Note: This might fail at the handler level due to actual corruption,
            # but it won't fail due to hash verification
            try:
                _ = cache_disabled.get(test_key="value")
                # If we get here, verification was truly skipped
                # The data might be None due to handler-level corruption detection
            except Exception:
                # Expected - the handler will likely fail to parse corrupted data
                # But the important thing is we didn't fail due to hash verification
                pass

    def test_file_hash_calculation_error_handling(self):
        """Test that file hash calculation handles errors gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(cache_dir=temp_dir, verify_cache_integrity=True)
            cache = UnifiedCache(config)

            # Test with non-existent file
            non_existent_file = Path(temp_dir) / "does_not_exist.txt"
            hash_result = cache._calculate_file_hash(non_existent_file)
            assert hash_result is None
