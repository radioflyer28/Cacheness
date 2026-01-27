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
    
    # =========================================================================
    # Valid Combinations
    # =========================================================================
    
    def test_sqlite_with_filesystem_allowed(self):
        """SQLite metadata + Filesystem blobs is allowed (both local)."""
        config = CacheConfig(
            metadata=CacheMetadataConfig(metadata_backend="sqlite"),
            blob=CacheBlobConfig(blob_backend="filesystem"),
        )
        assert config.metadata.metadata_backend == "sqlite"
        assert config.blob.blob_backend == "filesystem"
    
    def test_json_with_filesystem_allowed(self):
        """JSON metadata + Filesystem blobs is allowed (both local)."""
        config = CacheConfig(
            metadata=CacheMetadataConfig(metadata_backend="json"),
            blob=CacheBlobConfig(blob_backend="filesystem"),
        )
        assert config.metadata.metadata_backend == "json"
        assert config.blob.blob_backend == "filesystem"
    
    def test_auto_with_filesystem_allowed(self):
        """Auto metadata + Filesystem blobs is allowed."""
        config = CacheConfig(
            metadata=CacheMetadataConfig(metadata_backend="auto"),
            blob=CacheBlobConfig(blob_backend="filesystem"),
        )
        assert config.metadata.metadata_backend == "auto"
    
    def test_memory_with_memory_allowed(self):
        """Memory metadata + Memory blobs is allowed (both ephemeral, for testing)."""
        config = CacheConfig(
            metadata=CacheMetadataConfig(metadata_backend="memory"),
            blob=CacheBlobConfig(blob_backend="memory"),
        )
        assert config.metadata.metadata_backend == "memory"
        assert config.blob.blob_backend == "memory"
    
    def test_postgresql_with_s3_allowed(self):
        """PostgreSQL metadata + S3 blobs is allowed (both distributed)."""
        config = CacheConfig(
            metadata=CacheMetadataConfig(metadata_backend="postgresql"),
            blob=CacheBlobConfig(blob_backend="s3"),
        )
        assert config.metadata.metadata_backend == "postgresql"
        assert config.blob.blob_backend == "s3"
    
    def test_postgresql_with_filesystem_allowed(self):
        """PostgreSQL metadata + Filesystem blobs is allowed."""
        config = CacheConfig(
            metadata=CacheMetadataConfig(metadata_backend="postgresql"),
            blob=CacheBlobConfig(blob_backend="filesystem"),
        )
        assert config.metadata.metadata_backend == "postgresql"
        assert config.blob.blob_backend == "filesystem"
    
    def test_memory_with_filesystem_allowed(self):
        """Memory metadata + Filesystem blobs is allowed (testing with persistent blobs)."""
        config = CacheConfig(
            metadata=CacheMetadataConfig(metadata_backend="memory"),
            blob=CacheBlobConfig(blob_backend="filesystem"),
        )
        assert config.metadata.metadata_backend == "memory"
    
    # =========================================================================
    # Invalid Combinations - Local Metadata + Remote Blobs
    # =========================================================================
    
    def test_sqlite_with_s3_rejected(self):
        """SQLite metadata + S3 blobs is NOT allowed."""
        with pytest.raises(ValueError) as exc_info:
            CacheConfig(
                metadata=CacheMetadataConfig(metadata_backend="sqlite"),
                blob=CacheBlobConfig(blob_backend="s3"),
            )
        
        assert "Incompatible backend combination" in str(exc_info.value)
        assert "sqlite" in str(exc_info.value)
        assert "s3" in str(exc_info.value)
        assert "postgresql" in str(exc_info.value)  # Suggests PostgreSQL
    
    def test_json_with_s3_rejected(self):
        """JSON metadata + S3 blobs is NOT allowed."""
        with pytest.raises(ValueError) as exc_info:
            CacheConfig(
                metadata=CacheMetadataConfig(metadata_backend="json"),
                blob=CacheBlobConfig(blob_backend="s3"),
            )
        
        assert "Incompatible backend combination" in str(exc_info.value)
        assert "json" in str(exc_info.value)
        assert "s3" in str(exc_info.value)
    
    def test_auto_with_s3_rejected(self):
        """Auto metadata + S3 blobs is NOT allowed (auto defaults to local)."""
        with pytest.raises(ValueError) as exc_info:
            CacheConfig(
                metadata=CacheMetadataConfig(metadata_backend="auto"),
                blob=CacheBlobConfig(blob_backend="s3"),
            )
        
        assert "Incompatible backend combination" in str(exc_info.value)
    
    def test_sqlite_memory_with_s3_rejected(self):
        """SQLite memory + S3 blobs is NOT allowed."""
        with pytest.raises(ValueError) as exc_info:
            CacheConfig(
                metadata=CacheMetadataConfig(metadata_backend="sqlite_memory"),
                blob=CacheBlobConfig(blob_backend="s3"),
            )
        
        assert "Incompatible backend combination" in str(exc_info.value)
    
    # =========================================================================
    # Invalid Combinations - Ephemeral Metadata + Remote Blobs
    # =========================================================================
    
    def test_memory_with_s3_rejected(self):
        """Memory metadata + S3 blobs is NOT allowed."""
        with pytest.raises(ValueError) as exc_info:
            CacheConfig(
                metadata=CacheMetadataConfig(metadata_backend="memory"),
                blob=CacheBlobConfig(blob_backend="s3"),
            )
        
        assert "Incompatible backend combination" in str(exc_info.value)
        assert "ephemeral" in str(exc_info.value).lower()
        assert "memory" in str(exc_info.value)
        assert "s3" in str(exc_info.value)
    
    # =========================================================================
    # Future Cloud Backends (verify compatibility logic extends)
    # =========================================================================
    
    def test_sqlite_with_azure_rejected(self):
        """SQLite metadata + Azure blobs would be rejected."""
        with pytest.raises(ValueError) as exc_info:
            CacheConfig(
                metadata=CacheMetadataConfig(metadata_backend="sqlite"),
                blob=CacheBlobConfig(blob_backend="azure"),
            )
        
        assert "Incompatible backend combination" in str(exc_info.value)
    
    def test_sqlite_with_gcs_rejected(self):
        """SQLite metadata + GCS blobs would be rejected."""
        with pytest.raises(ValueError) as exc_info:
            CacheConfig(
                metadata=CacheMetadataConfig(metadata_backend="sqlite"),
                blob=CacheBlobConfig(blob_backend="gcs"),
            )
        
        assert "Incompatible backend combination" in str(exc_info.value)
    
    # =========================================================================
    # Default Configuration (should work)
    # =========================================================================
    
    def test_default_config_valid(self):
        """Default CacheConfig should be valid."""
        config = CacheConfig()
        # Should not raise
        assert config.metadata.metadata_backend == "auto"
        assert config.blob.blob_backend == "filesystem"
    
    # =========================================================================
    # Backwards Compatibility Parameters
    # =========================================================================
    
    def test_backwards_compat_params_validated(self):
        """Backwards compatibility params should also trigger validation."""
        with pytest.raises(ValueError) as exc_info:
            CacheConfig(
                metadata_backend="sqlite",
                blob_backend="s3",
            )
        
        assert "Incompatible backend combination" in str(exc_info.value)


# =============================================================================
# Compatibility Matrix Documentation Test
# =============================================================================

class TestCompatibilityMatrix:
    """
    Document the full compatibility matrix.
    
    | Metadata Backend | Blob Backend | Allowed | Reason |
    |------------------|--------------|---------|--------|
    | postgresql       | s3           | ✅      | Both distributed |
    | postgresql       | filesystem   | ✅      | Shared metadata, local blobs |
    | postgresql       | memory       | ✅      | Testing with persistent metadata |
    | sqlite           | filesystem   | ✅      | Both local |
    | sqlite           | memory       | ✅      | Testing |
    | sqlite           | s3           | ❌      | Local metadata + remote blobs |
    | json             | filesystem   | ✅      | Both local |
    | json             | s3           | ❌      | Local metadata + remote blobs |
    | memory           | memory       | ✅      | Both ephemeral (testing) |
    | memory           | filesystem   | ✅      | Ephemeral metadata, local blobs |
    | memory           | s3           | ❌      | Ephemeral + persistent mismatch |
    | auto             | filesystem   | ✅      | Auto is local |
    | auto             | s3           | ❌      | Auto is local |
    """
    
    def test_matrix_documented(self):
        """This test documents the compatibility matrix in docstring."""
        # The matrix is documented above
        pass
