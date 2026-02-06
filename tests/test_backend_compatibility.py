"""
Tests for Phase 2.10: Backend Compatibility Validation

Tests that incompatible metadata + blob backend combinations are rejected.
"""

import pytest

from cacheness.config import (
    CacheConfig,
    CacheMetadataConfig,
    CacheBlobConfig,
)


# =============================================================================
# Backend Compatibility Validation Tests
# =============================================================================

class TestBackendCompatibility:
    """Test backend compatibility validation in CacheConfig."""

    def test_local_metadata_with_local_blob_allowed(self):
        """Local metadata + local blobs is allowed."""
        config = CacheConfig(
            metadata=CacheMetadataConfig(metadata_backend="sqlite"),
            blob=CacheBlobConfig(blob_backend="filesystem"),
        )
        assert config.metadata.metadata_backend == "sqlite"
        assert config.blob.blob_backend == "filesystem"

    def test_distributed_metadata_with_remote_blob_allowed(self):
        """PostgreSQL metadata + S3 blobs is allowed (both distributed)."""
        config = CacheConfig(
            metadata=CacheMetadataConfig(metadata_backend="postgresql"),
            blob=CacheBlobConfig(blob_backend="s3"),
        )
        assert config.metadata.metadata_backend == "postgresql"
        assert config.blob.blob_backend == "s3"

    def test_local_metadata_with_remote_blob_rejected(self):
        """Local metadata + S3 blobs is NOT allowed."""
        with pytest.raises(ValueError, match="Incompatible backend combination"):
            CacheConfig(
                metadata=CacheMetadataConfig(metadata_backend="sqlite"),
                blob=CacheBlobConfig(blob_backend="s3"),
            )

    def test_default_config_valid(self):
        """Default CacheConfig should be valid."""
        config = CacheConfig()
        assert config.metadata.metadata_backend == "auto"
        assert config.blob.blob_backend == "filesystem"
