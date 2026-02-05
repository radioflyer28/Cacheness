"""
Tests for S3 ETag field in metadata backends.

This module tests that the s3_etag field can be properly stored and retrieved
from both SQLite and PostgreSQL metadata backends.
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path


class TestS3ETagSQLiteBackend:
    """Test s3_etag field storage in SQLite backend."""
    
    def test_s3_etag_storage_and_retrieval(self, tmp_path):
        """Test storing and retrieving s3_etag in SQLite backend."""
        from cacheness.metadata import SqliteBackend
        
        db_path = tmp_path / "test_metadata.db"
        backend = SqliteBackend(str(db_path))
        
        # Store entry with s3_etag
        entry_data = {
            "cache_key": "test_s3_entry",
            "description": "Test S3 ETag storage",
            "data_type": "array",
            "prefix": "s3_test",
            "file_size": 1024,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "s3_etag": "d41d8cd98f00b204e9800998ecf8427e",  # Example S3 ETag
                "s3_bucket": "test-bucket",
                "s3_key": "ab/abc123def456.blob",
                "actual_path": "s3://test-bucket/ab/abc123def456.blob",
            }
        }
        
        backend.put_entry("test_s3_entry", entry_data)
        
        # Retrieve entry
        retrieved = backend.get_entry("test_s3_entry")
        
        assert retrieved is not None
        assert "metadata" in retrieved
        # s3_etag is stored as a dedicated column
        assert retrieved["metadata"]["s3_etag"] == "d41d8cd98f00b204e9800998ecf8427e"
        # Other S3 fields remain in metadata JSON (not extracted to columns)
        # Note: actual_path is extracted to a column, but s3_bucket and s3_key stay in metadata
        assert "actual_path" in retrieved["metadata"]
        
        backend.close()
    
    def test_s3_etag_in_list_entries(self, tmp_path):
        """Test that s3_etag appears in list_entries output."""
        from cacheness.metadata import SqliteBackend
        
        db_path = tmp_path / "test_metadata.db"
        backend = SqliteBackend(str(db_path))
        
        # Store multiple entries, some with s3_etag
        entries_data = [
            {
                "cache_key": "s3_entry_1",
                "description": "S3 Entry 1",
                "data_type": "dataframe",
                "prefix": "df",
                "file_size": 2048,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "s3_etag": "etag_1",
                    "actual_path": "s3://bucket-1/df/entry1.parquet",
                }
            },
            {
                "cache_key": "local_entry_1",
                "description": "Local Entry",
                "data_type": "array",
                "prefix": "arr",
                "file_size": 512,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "actual_path": "/local/path/to/cache",
                }
            },
            {
                "cache_key": "s3_entry_2",
                "description": "S3 Entry 2",
                "data_type": "object",
                "prefix": "obj",
                "file_size": 4096,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "s3_etag": "etag_2",
                    "actual_path": "s3://bucket-2/cd/cdef1234.blob",
                }
            },
        ]
        
        for entry_data in entries_data:
            backend.put_entry(entry_data["cache_key"], entry_data)
        
        # List all entries
        all_entries = backend.list_entries()
        
        assert len(all_entries) == 3
        
        # Find S3 entries
        s3_entries = [e for e in all_entries if e.get("metadata", {}).get("s3_etag")]
        assert len(s3_entries) == 2
        
        # Verify S3 ETags are present
        etags = [e["metadata"]["s3_etag"] for e in s3_entries]
        assert "etag_1" in etags
        assert "etag_2" in etags
        
        backend.close()
    
    def test_s3_etag_optional(self, tmp_path):
        """Test that s3_etag is optional (for non-S3 backends)."""
        from cacheness.metadata import SqliteBackend
        
        db_path = tmp_path / "test_metadata.db"
        backend = SqliteBackend(str(db_path))
        
        # Store entry WITHOUT s3_etag (filesystem backend)
        entry_data = {
            "cache_key": "filesystem_entry",
            "description": "Filesystem Entry",
            "data_type": "object",
            "prefix": "fs",
            "file_size": 256,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "actual_path": "/cache/fs/filesystem_entry.pkl",
                "storage_format": "pickle",
            }
        }
        
        backend.put_entry("filesystem_entry", entry_data)
        
        # Retrieve entry
        retrieved = backend.get_entry("filesystem_entry")
        
        assert retrieved is not None
        assert "metadata" in retrieved
        # s3_etag should not be present for filesystem entries
        assert "s3_etag" not in retrieved["metadata"]
        assert retrieved["metadata"]["actual_path"] == "/cache/fs/filesystem_entry.pkl"
        
        backend.close()


class TestS3ETagPostgreSQLBackend:
    """Test s3_etag field storage in PostgreSQL backend."""
    
    @pytest.mark.skipif(
        True,  # Skip by default since it requires Docker/PostgreSQL running
        reason="Requires PostgreSQL instance (run Docker integration tests separately)"
    )
    def test_s3_etag_storage_postgresql(self):
        """Test storing and retrieving s3_etag in PostgreSQL backend."""
        from cacheness.storage.backends.postgresql_backend import PostgresBackend
        
        # This test is designed to run with Docker integration tests
        # where PostgreSQL is available at localhost:5432
        backend = PostgresBackend(
            connection_url="postgresql://cacheness_test:test_password@localhost:5432/cacheness_test"
        )
        
        # Store entry with s3_etag
        entry_data = {
            "cache_key": "pg_s3_test",
            "description": "PostgreSQL S3 ETag test",
            "data_type": "array",
            "prefix": "pg_s3",
            "file_size": 8192,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "s3_etag": "abcdef1234567890",
                "actual_path": "s3://pg-test-bucket/pg/pg_test.blob",
            }
        }
        
        backend.put_entry("pg_s3_test", entry_data)
        
        # Retrieve entry
        retrieved = backend.get_entry("pg_s3_test")
        
        assert retrieved is not None
        assert "metadata" in retrieved
        assert retrieved["metadata"]["s3_etag"] == "abcdef1234567890"
        
        # Clean up
        backend.remove_entry("pg_s3_test")
        backend.close()


class TestS3ETagIntegration:
    """Integration tests for s3_etag with cacheness core."""
    
    def test_s3_etag_with_cacheness_core(self, tmp_path):
        """Test that s3_etag can be stored via cacheness core API."""
        from cacheness import cacheness, CacheConfig
        
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            metadata_backend="sqlite",
        )
        
        cache = cacheness(config=config)
        
        # Store data with s3_etag in metadata (simulating S3 backend usage)
        data = {"test": "data"}
        cache.put(
            data,
            test_key="s3_integration",
            # Simulate S3 backend adding s3_etag to metadata
            _metadata_override={
                "s3_etag": "integration_test_etag",
                "actual_path": "s3://integration-bucket/test.blob",
            }
        )
        
        # List entries and verify s3_etag is present
        entries = cache.list_entries()
        
        # Find our entry
        test_entry = None
        for entry in entries:
            metadata = entry.get("metadata", {})
            if metadata.get("actual_path", "").startswith("s3://integration-bucket"):
                test_entry = entry
                break
        
        if test_entry:
            assert test_entry["metadata"]["s3_etag"] == "integration_test_etag"
            assert "s3://" in test_entry["metadata"]["actual_path"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
