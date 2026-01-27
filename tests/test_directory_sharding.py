"""
Tests for Phase 2.10: Git-Style Directory Sharding

Tests the directory sharding feature for blob storage backends.
"""

import pytest
import tempfile
from pathlib import Path

from cacheness.storage.backends.blob_backends import (
    FilesystemBlobBackend,
    InMemoryBlobBackend,
)
from cacheness.config import CacheBlobConfig


# =============================================================================
# CacheBlobConfig shard_chars Tests
# =============================================================================

class TestCacheBlobConfigShardChars:
    """Test shard_chars configuration in CacheBlobConfig."""
    
    def test_default_shard_chars_is_two(self):
        """Default shard_chars should be 2 (matching Git)."""
        config = CacheBlobConfig()
        assert config.shard_chars == 2
    
    def test_shard_chars_can_be_zero(self):
        """shard_chars=0 disables sharding."""
        config = CacheBlobConfig(shard_chars=0)
        assert config.shard_chars == 0
    
    def test_shard_chars_can_be_custom(self):
        """shard_chars can be set to custom values."""
        config = CacheBlobConfig(shard_chars=3)
        assert config.shard_chars == 3
        
        config = CacheBlobConfig(shard_chars=4)
        assert config.shard_chars == 4
    
    def test_shard_chars_negative_raises_error(self):
        """Negative shard_chars should raise ValueError."""
        with pytest.raises(ValueError, match="shard_chars must be non-negative"):
            CacheBlobConfig(shard_chars=-1)
    
    def test_shard_chars_excessive_raises_error(self):
        """Excessive shard_chars (>8) should raise ValueError."""
        with pytest.raises(ValueError, match="shard_chars must be <= 8"):
            CacheBlobConfig(shard_chars=9)
        
        with pytest.raises(ValueError, match="shard_chars must be <= 8"):
            CacheBlobConfig(shard_chars=16)
    
    def test_shard_chars_max_allowed(self):
        """shard_chars=8 should be allowed."""
        config = CacheBlobConfig(shard_chars=8)
        assert config.shard_chars == 8


# =============================================================================
# FilesystemBlobBackend Sharding Tests
# =============================================================================

class TestFilesystemSharding:
    """Test Git-style directory sharding in FilesystemBlobBackend."""
    
    @pytest.fixture
    def temp_dir(self):
        """Provide a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_default_shard_chars_is_two(self, temp_dir):
        """Default shard_chars should be 2."""
        backend = FilesystemBlobBackend(temp_dir)
        assert backend.shard_chars == 2
    
    def test_shard_chars_zero_disables_sharding(self, temp_dir):
        """With shard_chars=0, blobs are stored directly in base_dir."""
        backend = FilesystemBlobBackend(temp_dir, shard_chars=0)
        
        blob_id = "abc123def456"
        data = b"test data"
        blob_path = backend.write_blob(blob_id, data)
        
        # Should be directly in base_dir, not in a subdirectory
        path = Path(blob_path)
        assert path.parent == temp_dir
        assert path.name == blob_id
    
    def test_shard_chars_two_creates_subdirectory(self, temp_dir):
        """With shard_chars=2, blobs are stored in 2-char subdirectory."""
        backend = FilesystemBlobBackend(temp_dir, shard_chars=2)
        
        blob_id = "abc123def456"
        data = b"test data"
        blob_path = backend.write_blob(blob_id, data)
        
        # Should be in ab/ subdirectory
        path = Path(blob_path)
        assert path.parent.name == "ab"
        assert path.parent.parent == temp_dir
        assert path.name == blob_id
    
    def test_shard_chars_three_creates_subdirectory(self, temp_dir):
        """With shard_chars=3, blobs are stored in 3-char subdirectory."""
        backend = FilesystemBlobBackend(temp_dir, shard_chars=3)
        
        blob_id = "abc123def456"
        data = b"test data"
        blob_path = backend.write_blob(blob_id, data)
        
        # Should be in abc/ subdirectory
        path = Path(blob_path)
        assert path.parent.name == "abc"
        assert path.name == blob_id
    
    def test_read_sharded_blob(self, temp_dir):
        """Reading sharded blobs should work correctly."""
        backend = FilesystemBlobBackend(temp_dir, shard_chars=2)
        
        blob_id = "xyz789abc123"
        data = b"some test data"
        blob_path = backend.write_blob(blob_id, data)
        
        # Read back
        read_data = backend.read_blob(blob_path)
        assert read_data == data
    
    def test_delete_sharded_blob(self, temp_dir):
        """Deleting sharded blobs should work correctly."""
        backend = FilesystemBlobBackend(temp_dir, shard_chars=2)
        
        blob_id = "delete_me_123"
        data = b"data to delete"
        blob_path = backend.write_blob(blob_id, data)
        
        assert backend.exists(blob_path)
        assert backend.delete_blob(blob_path)
        assert not backend.exists(blob_path)
    
    def test_exists_sharded_blob(self, temp_dir):
        """exists() should work correctly with sharded blobs."""
        backend = FilesystemBlobBackend(temp_dir, shard_chars=2)
        
        blob_id = "check_exists_123"
        data = b"check this"
        blob_path = backend.write_blob(blob_id, data)
        
        assert backend.exists(blob_path)
        
        # Non-existent path
        fake_path = str(temp_dir / "ch" / "nonexistent")
        assert not backend.exists(fake_path)
    
    def test_multiple_blobs_same_shard(self, temp_dir):
        """Multiple blobs with same prefix should go in same shard directory."""
        backend = FilesystemBlobBackend(temp_dir, shard_chars=2)
        
        # All start with "ab"
        blob_ids = ["ab001", "ab002", "ab003", "abcdef"]
        
        for blob_id in blob_ids:
            backend.write_blob(blob_id, f"data_{blob_id}".encode())
        
        # Check all are in the "ab" directory
        ab_dir = temp_dir / "ab"
        assert ab_dir.exists()
        files_in_ab = list(ab_dir.iterdir())
        assert len(files_in_ab) == 4
    
    def test_multiple_blobs_different_shards(self, temp_dir):
        """Blobs with different prefixes should go in different shard directories."""
        backend = FilesystemBlobBackend(temp_dir, shard_chars=2)
        
        blob_ids = ["aa001", "bb002", "cc003", "dd004"]
        
        for blob_id in blob_ids:
            backend.write_blob(blob_id, f"data_{blob_id}".encode())
        
        # Check each shard directory exists
        for prefix in ["aa", "bb", "cc", "dd"]:
            shard_dir = temp_dir / prefix
            assert shard_dir.exists()
            files = list(shard_dir.iterdir())
            assert len(files) == 1
    
    def test_short_blob_id_no_sharding(self, temp_dir):
        """Blob IDs shorter than shard_chars should not be sharded."""
        backend = FilesystemBlobBackend(temp_dir, shard_chars=2)
        
        blob_id = "a"  # Only 1 char
        data = b"short id data"
        blob_path = backend.write_blob(blob_id, data)
        
        # Should be directly in base_dir
        path = Path(blob_path)
        assert path.parent == temp_dir
        assert path.name == blob_id
    
    def test_blob_id_equal_to_shard_chars(self, temp_dir):
        """Blob ID with exactly shard_chars length should be sharded."""
        backend = FilesystemBlobBackend(temp_dir, shard_chars=2)
        
        blob_id = "ab"  # Exactly 2 chars
        data = b"exact length data"
        blob_path = backend.write_blob(blob_id, data)
        
        # Should be sharded
        path = Path(blob_path)
        assert path.parent.name == "ab"
        assert path.name == blob_id
    
    def test_streaming_with_sharding(self, temp_dir):
        """Streaming writes should respect sharding."""
        backend = FilesystemBlobBackend(temp_dir, shard_chars=2)
        
        from io import BytesIO
        
        blob_id = "stream_abc123"
        data = b"streamed data content"
        stream = BytesIO(data)
        
        blob_path = backend.write_blob_stream(blob_id, stream)
        
        # Should be sharded
        path = Path(blob_path)
        assert path.parent.name == "st"
        assert path.name == blob_id
        
        # Verify content
        assert backend.read_blob(blob_path) == data
    
    def test_hex_like_blob_ids(self, temp_dir):
        """Test with hex-like blob IDs (realistic cache key hashes)."""
        backend = FilesystemBlobBackend(temp_dir, shard_chars=2)
        
        # Simulate hex hashes
        blob_ids = [
            "a1b2c3d4e5f6a7b8c9d0e1f2",
            "a1ffee00112233445566778899",
            "00deadbeef001122334455667788",
            "ff99887766554433221100aabbcc",
        ]
        
        for blob_id in blob_ids:
            backend.write_blob(blob_id, b"data")
        
        # First two share "a1" shard
        a1_dir = temp_dir / "a1"
        assert a1_dir.exists()
        assert len(list(a1_dir.iterdir())) == 2
        
        # Check other shards
        assert (temp_dir / "00").exists()
        assert (temp_dir / "ff").exists()


# =============================================================================
# Sharding Edge Cases
# =============================================================================

class TestShardingEdgeCases:
    """Test edge cases in directory sharding."""
    
    @pytest.fixture
    def temp_dir(self):
        """Provide a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_path_traversal_safe_with_sharding(self, temp_dir):
        """Path traversal attempts should be sanitized even with sharding."""
        backend = FilesystemBlobBackend(temp_dir, shard_chars=2)
        
        # Attempt path traversal
        blob_id = "../escape"
        data = b"attempt escape"
        blob_path = backend.write_blob(blob_id, data)
        
        # Should be safely contained in temp_dir
        path = Path(blob_path)
        # The ".." should be sanitized to "__"
        assert temp_dir in path.parents or path.parent == temp_dir or path.parent.parent == temp_dir
        # Should not escape temp_dir
        assert str(blob_path).startswith(str(temp_dir))
    
    def test_various_shard_chars_values(self, temp_dir):
        """Test various shard_chars values work correctly."""
        blob_id = "abcdefghij123456"
        data = b"test"
        
        for shard_chars in [0, 1, 2, 3, 4, 5]:
            backend = FilesystemBlobBackend(temp_dir / f"shard_{shard_chars}", shard_chars=shard_chars)
            blob_path = backend.write_blob(blob_id, data)
            path = Path(blob_path)
            
            if shard_chars == 0:
                # No sharding
                assert path.parent == backend.base_dir
            else:
                # Check shard directory name
                assert path.parent.name == blob_id[:shard_chars]
    
    def test_get_size_with_sharding(self, temp_dir):
        """get_size should work correctly with sharded paths."""
        backend = FilesystemBlobBackend(temp_dir, shard_chars=2)
        
        blob_id = "size_test_123"
        data = b"x" * 1000
        blob_path = backend.write_blob(blob_id, data)
        
        assert backend.get_size(blob_path) == 1000


# =============================================================================
# InMemoryBlobBackend (sharding not applicable but test config)
# =============================================================================

class TestInMemorySharding:
    """Test that InMemoryBlobBackend doesn't break with sharding config."""
    
    def test_inmemory_ignores_sharding(self):
        """InMemoryBlobBackend doesn't use filesystem paths, so no sharding."""
        backend = InMemoryBlobBackend()
        
        blob_id = "memory_blob_123"
        data = b"in memory data"
        blob_path = backend.write_blob(blob_id, data)
        
        # Memory backend uses memory:// prefix
        assert blob_path.startswith("memory://")
        assert backend.read_blob(blob_path) == data
