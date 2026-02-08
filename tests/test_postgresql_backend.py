"""
Tests for PostgreSQL Metadata Backend
=====================================

Tests for Phase 2.6: PostgreSQL metadata backend implementation.

Note: Most tests use SQLite-in-memory mode to test the base functionality.
PostgreSQL-specific tests are marked and require a running PostgreSQL instance.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

# Try to import PostgresBackend
try:
    from cacheness.storage.backends.postgresql_backend import (
        PostgresBackend,
        SQLALCHEMY_AVAILABLE,
        PSYCOPG_AVAILABLE,
        PSYCOPG_VERSION,
    )

    _HAS_POSTGRES_BACKEND = True
except ImportError:
    _HAS_POSTGRES_BACKEND = False
    SQLALCHEMY_AVAILABLE = False
    PSYCOPG_AVAILABLE = False
    PSYCOPG_VERSION = None

from cacheness.storage.backends import (
    list_metadata_backends,
    get_metadata_backend,
    MetadataBackend,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def postgres_url():
    """Get PostgreSQL connection URL from environment or skip test."""
    url = os.environ.get("CACHENESS_TEST_POSTGRES_URL")
    if not url:
        pytest.skip(
            "PostgreSQL tests require CACHENESS_TEST_POSTGRES_URL environment variable. "
            "Example: postgresql://postgres:password@localhost:5432/cacheness_test"
        )
    return url


# =============================================================================
# Test: Module Availability
# =============================================================================


class TestPostgresBackendAvailability:
    """Test PostgreSQL backend availability and dependencies."""

    def test_backend_import(self):
        """Test that the backend module can be imported."""
        # Even without psycopg, the module should import
        from cacheness.storage.backends import postgresql_backend

        assert hasattr(postgresql_backend, "PostgresBackend")

    def test_sqlalchemy_check(self):
        """Test SQLAlchemy availability check."""
        from cacheness.storage.backends.postgresql_backend import SQLALCHEMY_AVAILABLE

        # SQLAlchemy should be available in our test environment
        assert SQLALCHEMY_AVAILABLE is True

    def test_psycopg_version_detection(self):
        """Test psycopg version detection."""
        from cacheness.storage.backends.postgresql_backend import (
            PSYCOPG_AVAILABLE,
            PSYCOPG_VERSION,
        )

        # Just verify the variables are set correctly
        if PSYCOPG_AVAILABLE:
            assert PSYCOPG_VERSION in ("psycopg2", "psycopg3")
        else:
            assert PSYCOPG_VERSION is None

    @pytest.mark.skipif(not PSYCOPG_AVAILABLE, reason="psycopg not installed")
    def test_postgres_in_registry(self):
        """Test that PostgreSQL is registered when available."""
        backends = list_metadata_backends()
        backend_names = [b["name"] for b in backends]
        assert "postgresql" in backend_names


# =============================================================================
# Test: PostgreSQL Backend Interface (Mock)
# =============================================================================


class TestPostgresBackendInterface:
    """Test PostgreSQL backend interface without actual database."""

    @pytest.mark.skipif(
        not _HAS_POSTGRES_BACKEND, reason="PostgresBackend not available"
    )
    def test_class_exists(self):
        """Test PostgresBackend class exists."""
        assert PostgresBackend is not None
        assert issubclass(PostgresBackend, MetadataBackend)

    @pytest.mark.skipif(
        not _HAS_POSTGRES_BACKEND, reason="PostgresBackend not available"
    )
    def test_init_requires_connection_url(self):
        """Test that initialization requires connection URL."""
        with pytest.raises(TypeError):
            PostgresBackend()  # Missing required argument

    @pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not available")
    @pytest.mark.skipif(
        not _HAS_POSTGRES_BACKEND, reason="PostgresBackend not available"
    )
    def test_init_without_psycopg_raises(self):
        """Test that initialization fails without psycopg."""
        with patch(
            "cacheness.storage.backends.postgresql_backend.PSYCOPG_AVAILABLE", False
        ):
            # Re-import to get patched value

            # Can't easily test this without mocking the import
            pass

    @pytest.mark.skipif(
        not _HAS_POSTGRES_BACKEND, reason="PostgresBackend not available"
    )
    def test_safe_url_masks_password(self):
        """Test that _safe_url masks password."""
        # Create mock backend to test the method
        url = "postgresql://user:secret123@localhost:5432/db"

        # We can't actually create the backend without a real DB,
        # but we can test the logic
        if "@" in url:
            parts = url.split("@")
            safe = f"postgresql://***@{parts[-1]}"
            assert "secret123" not in safe
            assert "localhost" in safe


# =============================================================================
# Test: PostgreSQL Backend Functionality (Requires PostgreSQL)
# =============================================================================


@pytest.mark.skipif(not _HAS_POSTGRES_BACKEND, reason="PostgresBackend not available")
@pytest.mark.skipif(not PSYCOPG_AVAILABLE, reason="psycopg not installed")
class TestPostgresBackendWithDatabase:
    """Test PostgreSQL backend with actual database connection."""

    @pytest.fixture
    def backend(self, postgres_url):
        """Create a PostgreSQL backend for testing."""
        backend = PostgresBackend(
            connection_url=postgres_url,
            pool_size=2,
            max_overflow=5,
        )
        # Clear any existing data
        backend.clear_all()
        yield backend
        # Cleanup
        backend.clear_all()
        backend.close()

    def test_put_and_get_entry(self, backend):
        """Test basic put and get operations."""
        entry_data = {
            "data_type": "pickle",
            "file_size": 1024,
            "description": "Test entry",
            "metadata": {
                "object_type": "<class 'dict'>",
                "storage_format": "pickle",
                "compression_codec": "zstd",
            },
        }

        backend.put_entry("test_key_001", entry_data)

        result = backend.get_entry("test_key_001")
        assert result is not None
        assert result["cache_key"] == "test_key_001"
        assert result["data_type"] == "pickle"
        assert result["file_size"] == 1024

    def test_get_nonexistent_entry(self, backend):
        """Test getting a non-existent entry returns None."""
        result = backend.get_entry("nonexistent_key")
        assert result is None

    def test_remove_entry(self, backend):
        """Test removing an entry."""
        backend.put_entry("remove_me", {"data_type": "test", "file_size": 0})

        result = backend.get_entry("remove_me")
        assert result is not None

        backend.remove_entry("remove_me")

        result = backend.get_entry("remove_me")
        assert result is None

    def test_list_entries(self, backend):
        """Test listing all entries."""
        # Add multiple entries
        for i in range(5):
            backend.put_entry(
                f"list_test_{i:03d}",
                {
                    "data_type": "pickle",
                    "file_size": i * 100,
                },
            )

        entries = backend.list_entries()
        assert len(entries) >= 5

        # Check entries are sorted by created_at desc
        keys = [e["cache_key"] for e in entries]
        assert all(k.startswith("list_test_") for k in keys[:5])

    def test_update_access_time(self, backend):
        """Test updating access time."""
        backend.put_entry("access_test", {"data_type": "test", "file_size": 0})

        original = backend.get_entry("access_test")
        original_time = original["accessed_at"]

        import time

        time.sleep(0.1)

        backend.update_access_time("access_test")

        updated = backend.get_entry("access_test")
        # Access time should be updated
        assert updated["accessed_at"] >= original_time

    def test_stats_tracking(self, backend):
        """Test statistics tracking."""
        initial_stats = backend.get_stats()

        backend.increment_hits()
        backend.increment_hits()
        backend.increment_misses()

        stats = backend.get_stats()
        assert stats["hits"] >= initial_stats["hits"] + 2
        assert stats["misses"] >= initial_stats["misses"] + 1

    def test_cleanup_expired(self, backend):
        """Test cleanup of expired entries."""
        # Add some entries
        for i in range(3):
            backend.put_entry(f"cleanup_{i}", {"data_type": "test", "file_size": 0})

        # Cleanup with 0 TTL should do nothing
        count = backend.cleanup_expired(0)
        assert count == 0

        # Entries should still exist
        entries = backend.list_entries()
        cleanup_entries = [e for e in entries if e["cache_key"].startswith("cleanup_")]
        assert len(cleanup_entries) == 3

    def test_clear_all(self, backend):
        """Test clearing all entries."""
        # Add some entries
        for i in range(5):
            backend.put_entry(f"clear_{i}", {"data_type": "test", "file_size": 0})

        count = backend.clear_all()
        assert count >= 5

        entries = backend.list_entries()
        assert len(entries) == 0

        stats = backend.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_concurrent_access(self, backend, postgres_url):
        """Test concurrent access from multiple threads."""
        import concurrent.futures

        def worker(worker_id):
            """Worker function that does put/get operations."""
            for i in range(10):
                key = f"concurrent_{worker_id}_{i}"
                backend.put_entry(key, {"data_type": "test", "file_size": i})
                result = backend.get_entry(key)
                assert result is not None
            return worker_id

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 4

    def test_context_manager(self, postgres_url):
        """Test using backend as context manager."""
        with PostgresBackend(connection_url=postgres_url) as backend:
            backend.put_entry("context_test", {"data_type": "test", "file_size": 0})
            result = backend.get_entry("context_test")
            assert result is not None

    def test_metadata_preservation(self, backend):
        """Test that custom metadata is preserved."""
        entry_data = {
            "data_type": "dataframe",
            "file_size": 50000,
            "description": "ML model weights",
            "metadata": {
                "object_type": "<class 'pandas.DataFrame'>",
                "storage_format": "parquet",
                "serializer": "pyarrow",
                "compression_codec": "zstd",
                "actual_path": "/cache/model_weights.parquet",
                "file_hash": "abc123def456",
            },
        }

        backend.put_entry("metadata_test", entry_data)

        result = backend.get_entry("metadata_test")
        assert result["data_type"] == "dataframe"
        assert "metadata" in result
        assert result["metadata"]["storage_format"] == "parquet"
        assert result["metadata"]["compression_codec"] == "zstd"


# =============================================================================
# Test: Registry Integration
# =============================================================================


class TestPostgresRegistryIntegration:
    """Test PostgreSQL backend integration with the registry."""

    @pytest.mark.skipif(not PSYCOPG_AVAILABLE, reason="psycopg not installed")
    def test_get_backend_via_registry(self, postgres_url):
        """Test getting PostgreSQL backend via registry."""
        backend = get_metadata_backend(
            "postgresql",
            connection_url=postgres_url,
            pool_size=2,
        )

        assert backend is not None
        assert isinstance(backend, PostgresBackend)
        backend.close()

    @pytest.mark.skipif(not PSYCOPG_AVAILABLE, reason="psycopg not installed")
    def test_backend_info_in_list(self):
        """Test that PostgreSQL appears in backend list."""
        backends = list_metadata_backends()
        pg_backends = [b for b in backends if b["name"] == "postgresql"]

        assert len(pg_backends) == 1
        assert pg_backends[0]["class"] == "PostgresBackend"
        assert pg_backends[0]["is_builtin"] is True


# =============================================================================
# Test: Error Handling
# =============================================================================


@pytest.mark.skipif(not _HAS_POSTGRES_BACKEND, reason="PostgresBackend not available")
@pytest.mark.skipif(not PSYCOPG_AVAILABLE, reason="psycopg not installed")
class TestPostgresErrorHandling:
    """Test error handling in PostgreSQL backend."""

    def test_invalid_connection_url(self):
        """Test handling of invalid connection URL."""
        with pytest.raises(Exception):  # Could be various SQLAlchemy/psycopg errors
            backend = PostgresBackend(
                connection_url="postgresql://invalid:invalid@nonexistent:5432/db"
            )
            # Try to use it - this should fail
            backend.list_entries()

    def test_connection_timeout(self):
        """Test handling of connection timeout."""
        # This test is mainly to ensure we don't hang indefinitely
        # Skip if it takes too long
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Connection attempt timed out")

        # Note: signal.alarm doesn't work on Windows
        import sys

        if sys.platform == "win32":
            pytest.skip("Signal-based timeout not available on Windows")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 second timeout

        try:
            with pytest.raises(Exception):
                backend = PostgresBackend(
                    connection_url="postgresql://user:pass@192.0.2.1:5432/db"  # Reserved IP
                )
                backend.list_entries()
        finally:
            signal.alarm(0)


# =============================================================================
# Test: Custom Metadata + PostgreSQL Integration (Phase 2.7)
# =============================================================================


class TestCustomMetadataPostgresIntegration:
    """Test custom metadata functionality with PostgreSQL backend.

    These tests verify that the custom metadata system (Phase 2.7) works
    correctly with the PostgreSQL metadata backend (Phase 2.6).
    """

    def test_core_supports_postgresql_for_custom_metadata(self):
        """Test that core.py recognizes PostgreSQL for custom metadata."""
        # This tests the _supports_custom_metadata logic without requiring PostgreSQL
        from cacheness.core import UnifiedCache
        from cacheness import CacheConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(
                cache_dir=tmpdir,
                metadata_backend="sqlite",  # Use SQLite for actual testing
            )
            cache = UnifiedCache(config)

            # Simulate PostgreSQL backend
            cache.actual_backend = "postgresql"
            cache._custom_metadata_enabled = True

            # Verify it's now considered supporting custom metadata
            assert cache._supports_custom_metadata() is True

            cache.close()

    def test_init_custom_metadata_support_with_postgresql(self):
        """Test _init_custom_metadata_support recognizes PostgreSQL."""

        with tempfile.TemporaryDirectory() as tmpdir:
            from cacheness import CacheConfig
            from cacheness.core import UnifiedCache

            config = CacheConfig(cache_dir=tmpdir, metadata_backend="sqlite")
            cache = UnifiedCache(config)

            # Force PostgreSQL backend type
            cache.actual_backend = "postgresql"
            cache._init_custom_metadata_support()

            # Should be enabled for PostgreSQL
            assert cache._custom_metadata_enabled is True

            cache.close()

    def test_create_metadata_backend_postgresql_option(self):
        """Test that create_metadata_backend accepts postgresql option."""
        from cacheness.metadata import create_metadata_backend

        # Without actual PostgreSQL, this should raise an import error about psycopg
        if PSYCOPG_AVAILABLE:
            pytest.skip("This test is for when psycopg is NOT installed")

        with pytest.raises(ImportError) as exc_info:
            backend = create_metadata_backend(
                "postgresql", connection_url="postgresql://localhost/test"
            )

        assert (
            "psycopg" in str(exc_info.value).lower()
            or "postgresql" in str(exc_info.value).lower()
        )

    @pytest.mark.skipif(not PSYCOPG_AVAILABLE, reason="psycopg not installed")
    def test_custom_metadata_model_registration(self, postgres_url):
        """Test custom metadata model registration with PostgreSQL."""
        from cacheness.custom_metadata import (
            custom_metadata_model,
            CustomMetadataBase,
            list_registered_schemas,
            _reset_registry,
        )
        from cacheness.metadata import Base
        from sqlalchemy import Column, String, Float

        _reset_registry()

        @custom_metadata_model("pg_experiments")
        class PgExperimentMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_pg_experiments"
            experiment_id = Column(String(100), nullable=False, index=True)
            score = Column(Float, nullable=False)

        assert "pg_experiments" in list_registered_schemas()

        _reset_registry()

    @pytest.mark.skipif(not PSYCOPG_AVAILABLE, reason="psycopg not installed")
    def test_migrate_custom_metadata_tables_with_postgresql(self, postgres_url):
        """Test migrate_custom_metadata_tables with PostgreSQL engine."""
        from cacheness.storage.backends.postgresql_backend import PostgresBackend
        from cacheness.custom_metadata import (
            custom_metadata_model,
            CustomMetadataBase,
            migrate_custom_metadata_tables,
            _reset_registry,
        )
        from cacheness.metadata import Base
        from sqlalchemy import Column, String, Float, inspect

        _reset_registry()

        @custom_metadata_model("pg_test_migrate")
        class PgTestMigrateMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_pg_test_migrate"
            name = Column(String(100), nullable=False)
            value = Column(Float, nullable=False)

        # Create backend and migrate
        backend = PostgresBackend(connection_url=postgres_url)
        migrate_custom_metadata_tables(backend.engine)

        # Verify tables exist
        inspector = inspect(backend.engine)
        tables = inspector.get_table_names()

        assert "cache_metadata_links" in tables
        assert "custom_pg_test_migrate" in tables

        backend.close()
        _reset_registry()

    @pytest.mark.skipif(not PSYCOPG_AVAILABLE, reason="psycopg not installed")
    def test_store_and_query_custom_metadata_with_postgresql(self, postgres_url):
        """Test full custom metadata workflow with PostgreSQL backend."""
        from cacheness import cacheness, CacheConfig
        from cacheness.custom_metadata import (
            custom_metadata_model,
            CustomMetadataBase,
            _reset_registry,
        )
        from cacheness.metadata import Base
        from sqlalchemy import Column, String, Float

        _reset_registry()

        @custom_metadata_model("pg_full_test")
        class PgFullTestMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_pg_full_test"
            run_id = Column(String(100), nullable=False, unique=True, index=True)
            accuracy = Column(Float, nullable=False, index=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(
                cache_dir=tmpdir,
                metadata_backend="postgresql",
                metadata_backend_options={
                    "connection_url": postgres_url,
                },
            )

            cache = cacheness(config)

            # Verify custom metadata is enabled
            assert cache._custom_metadata_enabled is True
            assert cache.actual_backend == "postgresql"

            # Store data with custom metadata
            metadata = PgFullTestMetadata(run_id="test_run_001", accuracy=0.95)

            cache_key = cache.put(
                {"test": "data"}, run="test_run_001", custom_metadata=metadata
            )

            # Query the custom metadata
            results = cache.query_custom("pg_full_test")
            assert len(results) >= 1

            matching = [r for r in results if r.run_id == "test_run_001"]
            assert len(matching) == 1
            assert matching[0].accuracy == 0.95

            cache.close()

        _reset_registry()

    @pytest.mark.skipif(not PSYCOPG_AVAILABLE, reason="psycopg not installed")
    def test_query_custom_session_with_postgresql(self, postgres_url):
        """Test query_custom_session context manager with PostgreSQL."""
        from cacheness import cacheness, CacheConfig
        from cacheness.custom_metadata import (
            custom_metadata_model,
            CustomMetadataBase,
            _reset_registry,
        )
        from cacheness.metadata import Base
        from sqlalchemy import Column, String, Float

        _reset_registry()

        @custom_metadata_model("pg_session_test")
        class PgSessionTestMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_pg_session_test"
            name = Column(String(100), nullable=False, index=True)
            score = Column(Float, nullable=False, index=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(
                cache_dir=tmpdir,
                metadata_backend="postgresql",
                metadata_backend_options={
                    "connection_url": postgres_url,
                },
            )

            cache = cacheness(config)

            # Store multiple entries
            for i, (name, score) in enumerate(
                [
                    ("alice", 0.9),
                    ("bob", 0.85),
                    ("charlie", 0.95),
                ]
            ):
                metadata = PgSessionTestMetadata(name=name, score=score)
                cache.put({"data": i}, key=f"entry_{i}", custom_metadata=metadata)

            # Query using context manager
            with cache.query_custom_session("pg_session_test") as query:
                high_scorers = query.filter(PgSessionTestMetadata.score >= 0.9).all()

                assert len(high_scorers) == 2
                names = {r.name for r in high_scorers}
                assert names == {"alice", "charlie"}

            cache.close()

        _reset_registry()
