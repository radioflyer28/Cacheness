"""Tests for SQLite backend schema versioning and namespace registry."""

import os
import tempfile

import pytest

from cacheness.metadata import (
    SqliteBackend,
    CacheNamespace,
    NamespaceInfo,
    DEFAULT_NAMESPACE,
    validate_namespace_id,
)


@pytest.fixture
def sqlite_backend(tmp_path):
    """Create a fresh SQLite backend in a temp directory."""
    db_file = str(tmp_path / "test_cache.db")
    backend = SqliteBackend(db_file)
    yield backend
    backend.close()


@pytest.fixture
def sqlite_backend_path(tmp_path):
    """Return a db path for manual backend construction."""
    return str(tmp_path / "test_cache.db")


class TestSqliteSchemaVersioning:
    """Test schema versioning on SqliteBackend."""

    def test_default_namespace_registered_on_init(self, sqlite_backend):
        """Creating a backend should register the 'default' namespace."""
        ns = sqlite_backend.get_namespace(DEFAULT_NAMESPACE)
        assert ns is not None
        assert ns.namespace_id == DEFAULT_NAMESPACE
        assert ns.display_name == "Default"
        assert ns.schema_version >= 1

    def test_get_schema_version_default(self, sqlite_backend):
        """Schema version should be >= 1 after init (migrations ran)."""
        version = sqlite_backend.get_schema_version(DEFAULT_NAMESPACE)
        assert version >= 1

    def test_set_schema_version(self, sqlite_backend):
        """set_schema_version should persist."""
        sqlite_backend.set_schema_version(DEFAULT_NAMESPACE, 99)
        assert sqlite_backend.get_schema_version(DEFAULT_NAMESPACE) == 99

    def test_get_schema_version_unknown_namespace(self, sqlite_backend):
        """Unknown namespace returns 0."""
        assert sqlite_backend.get_schema_version("nonexistent") == 0

    def test_migrations_run_on_fresh_db(self, sqlite_backend):
        """Fresh database should have all migrations applied."""
        version = sqlite_backend.get_schema_version(DEFAULT_NAMESPACE)
        # v0→v1 is the only migration currently
        assert version == 1

    def test_migrations_idempotent(self, sqlite_backend_path):
        """Opening the same database twice should not fail or re-run migrations."""
        backend1 = SqliteBackend(sqlite_backend_path)
        v1 = backend1.get_schema_version(DEFAULT_NAMESPACE)
        backend1.close()

        backend2 = SqliteBackend(sqlite_backend_path)
        v2 = backend2.get_schema_version(DEFAULT_NAMESPACE)
        backend2.close()

        assert v1 == v2

    def test_cacheness_namespaces_table_exists(self, sqlite_backend):
        """The cacheness_namespaces table should exist after init."""
        from sqlalchemy import inspect

        inspector = inspect(sqlite_backend.engine)
        tables = inspector.get_table_names()
        assert "cacheness_namespaces" in tables

    def test_legacy_run_migrations_is_noop(self, sqlite_backend):
        """The old _run_migrations() method should be a harmless no-op."""
        # Should not raise
        sqlite_backend._run_migrations()


class TestSqliteNamespaceRegistry:
    """Test namespace registry on SqliteBackend."""

    def test_list_namespaces_has_default(self, sqlite_backend):
        """list_namespaces should include 'default' after init."""
        namespaces = sqlite_backend.list_namespaces()
        ids = [ns.namespace_id for ns in namespaces]
        assert DEFAULT_NAMESPACE in ids

    def test_create_namespace(self, sqlite_backend):
        """create_namespace should register and create tables."""
        ns = sqlite_backend.create_namespace("project_alpha", "Project Alpha")
        assert ns.namespace_id == "project_alpha"
        assert ns.display_name == "Project Alpha"
        assert ns.schema_version == 1

        # Verify tables were created
        from sqlalchemy import inspect

        inspector = inspect(sqlite_backend.engine)
        tables = inspector.get_table_names()
        assert "cache_entries_project_alpha" in tables
        assert "cache_stats_project_alpha" in tables

    def test_create_namespace_appears_in_list(self, sqlite_backend):
        """Created namespace should appear in list_namespaces."""
        sqlite_backend.create_namespace("my_ns")
        namespaces = sqlite_backend.list_namespaces()
        ids = [ns.namespace_id for ns in namespaces]
        assert "my_ns" in ids
        assert DEFAULT_NAMESPACE in ids

    def test_create_namespace_duplicate_raises(self, sqlite_backend):
        """Creating a namespace that already exists should raise ValueError."""
        sqlite_backend.create_namespace("dup_ns")
        with pytest.raises(ValueError, match="already exists"):
            sqlite_backend.create_namespace("dup_ns")

    def test_create_default_raises(self, sqlite_backend):
        """Cannot re-create the 'default' namespace."""
        with pytest.raises(ValueError, match="pre-registered"):
            sqlite_backend.create_namespace(DEFAULT_NAMESPACE)

    def test_create_namespace_invalid_id(self, sqlite_backend):
        """Invalid namespace IDs should be rejected."""
        with pytest.raises(ValueError, match="must match"):
            sqlite_backend.create_namespace("Invalid-Name")

    def test_drop_namespace(self, sqlite_backend):
        """drop_namespace should remove tables and registry entry."""
        sqlite_backend.create_namespace("temp_ns")

        from sqlalchemy import inspect

        # Verify tables created
        inspector = inspect(sqlite_backend.engine)
        assert "cache_entries_temp_ns" in inspector.get_table_names()

        # Drop
        result = sqlite_backend.drop_namespace("temp_ns")
        assert result is True

        # Verify tables gone
        inspector = inspect(sqlite_backend.engine)
        tables = inspector.get_table_names()
        assert "cache_entries_temp_ns" not in tables
        assert "cache_stats_temp_ns" not in tables

        # Verify registry entry gone
        assert sqlite_backend.get_namespace("temp_ns") is None

    def test_drop_nonexistent_namespace(self, sqlite_backend):
        """Dropping a nonexistent namespace returns False."""
        assert sqlite_backend.drop_namespace("no_such_ns") is False

    def test_drop_default_raises(self, sqlite_backend):
        """Cannot drop the 'default' namespace."""
        with pytest.raises(ValueError, match="Cannot drop"):
            sqlite_backend.drop_namespace(DEFAULT_NAMESPACE)

    def test_get_namespace(self, sqlite_backend):
        """get_namespace should return NamespaceInfo for existing namespace."""
        sqlite_backend.create_namespace("lookup_ns", "Lookup Test")
        ns = sqlite_backend.get_namespace("lookup_ns")
        assert ns is not None
        assert ns.namespace_id == "lookup_ns"
        assert ns.display_name == "Lookup Test"

    def test_get_namespace_nonexistent(self, sqlite_backend):
        """get_namespace returns None for nonexistent namespace."""
        assert sqlite_backend.get_namespace("nope") is None

    def test_namespace_exists(self, sqlite_backend):
        """namespace_exists should return True/False correctly."""
        assert sqlite_backend.namespace_exists(DEFAULT_NAMESPACE) is True
        assert sqlite_backend.namespace_exists("nope") is False
        sqlite_backend.create_namespace("exists_ns")
        assert sqlite_backend.namespace_exists("exists_ns") is True

    def test_create_namespace_with_indexes(self, sqlite_backend):
        """Created namespace tables should have indexes."""
        sqlite_backend.create_namespace("idx_ns")

        from sqlalchemy import inspect

        inspector = inspect(sqlite_backend.engine)
        indexes = inspector.get_indexes("cache_entries_idx_ns")
        index_names = [idx["name"] for idx in indexes]
        assert "idx_idx_ns_list_entries" in index_names
        assert "idx_idx_ns_cleanup" in index_names
        assert "idx_idx_ns_size_mgmt" in index_names

    def test_create_namespace_stats_initialized(self, sqlite_backend):
        """New namespace should have an initialized stats row."""
        sqlite_backend.create_namespace("stats_ns")

        from sqlalchemy import text

        with sqlite_backend.SessionLocal() as session:
            row = session.execute(
                text(
                    "SELECT cache_hits, cache_misses FROM cache_stats_stats_ns WHERE id = 1"
                )
            ).one()
            assert row[0] == 0  # cache_hits
            assert row[1] == 0  # cache_misses

    def test_multiple_namespaces(self, sqlite_backend):
        """Can create multiple namespaces, all coexist."""
        sqlite_backend.create_namespace("ns_a", "Namespace A")
        sqlite_backend.create_namespace("ns_b", "Namespace B")
        sqlite_backend.create_namespace("ns_c")

        namespaces = sqlite_backend.list_namespaces()
        ids = sorted(ns.namespace_id for ns in namespaces)
        assert ids == [DEFAULT_NAMESPACE, "ns_a", "ns_b", "ns_c"]

    def test_drop_one_doesnt_affect_others(self, sqlite_backend):
        """Dropping one namespace shouldn't affect others."""
        sqlite_backend.create_namespace("keep_ns")
        sqlite_backend.create_namespace("drop_ns")

        sqlite_backend.drop_namespace("drop_ns")

        # 'keep_ns' and 'default' should still exist
        namespaces = sqlite_backend.list_namespaces()
        ids = [ns.namespace_id for ns in namespaces]
        assert DEFAULT_NAMESPACE in ids
        assert "keep_ns" in ids
        assert "drop_ns" not in ids


class TestSqliteBackwardCompatibility:
    """Test that existing functionality is not broken."""

    def test_existing_operations_still_work(self, sqlite_backend):
        """Basic put/get/remove should still work on default tables."""
        sqlite_backend.put_entry(
            "test_key",
            {
                "description": "test",
                "data_type": "pickle",
                "prefix": "",
                "file_size": 100,
            },
        )

        entry = sqlite_backend.get_entry("test_key")
        assert entry is not None
        assert entry["data_type"] == "pickle"

        removed = sqlite_backend.remove_entry("test_key")
        assert removed is True

    def test_stats_still_work(self, sqlite_backend):
        """Stats should still work on the default tables."""
        sqlite_backend.increment_hits()
        sqlite_backend.increment_misses()
        stats = sqlite_backend.get_stats()
        assert stats["cache_hits"] >= 1
        assert stats["cache_misses"] >= 1

    def test_clear_all_still_works(self, sqlite_backend):
        """clear_all should still work."""
        sqlite_backend.put_entry(
            "key1",
            {
                "data_type": "pickle",
                "file_size": 50,
            },
        )
        count = sqlite_backend.clear_all()
        assert count >= 1

    def test_pre_existing_db_without_registry(self, tmp_path):
        """Opening a pre-existing DB that lacks cacheness_namespaces should work.

        Simulates upgrading from a database created before namespace support.
        """
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker

        db_file = str(tmp_path / "legacy.db")

        # Create a legacy database without the registry table
        engine = create_engine(f"sqlite:///{db_file}")
        Session = sessionmaker(bind=engine)

        with Session() as session:
            session.execute(
                text("""
                CREATE TABLE cache_entries (
                    cache_key VARCHAR(16) PRIMARY KEY,
                    description VARCHAR(500) NOT NULL DEFAULT '',
                    data_type VARCHAR(20) NOT NULL,
                    prefix VARCHAR(100) NOT NULL DEFAULT '',
                    created_at DATETIME NOT NULL,
                    accessed_at DATETIME NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    file_hash VARCHAR(16),
                    entry_signature VARCHAR(64),
                    s3_etag VARCHAR(100),
                    object_type VARCHAR(100),
                    storage_format VARCHAR(20),
                    serializer VARCHAR(20),
                    compression_codec VARCHAR(20),
                    actual_path VARCHAR(500),
                    cache_key_params TEXT,
                    metadata_dict TEXT
                )
            """)
            )
            session.execute(
                text("""
                CREATE TABLE cache_stats (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    cache_hits INTEGER NOT NULL DEFAULT 0,
                    cache_misses INTEGER NOT NULL DEFAULT 0,
                    last_updated DATETIME NOT NULL
                )
            """)
            )
            session.execute(
                text("""
                INSERT INTO cache_entries (cache_key, data_type, file_size, created_at, accessed_at)
                VALUES ('legacy_key', 'pickle', 42, datetime('now'), datetime('now'))
            """)
            )
            session.execute(
                text("""
                INSERT INTO cache_stats (id, cache_hits, cache_misses, last_updated)
                VALUES (1, 10, 5, datetime('now'))
            """)
            )
            session.commit()
        engine.dispose()

        # Now open with SqliteBackend — should auto-create registry
        backend = SqliteBackend(db_file)

        # Registry should exist with 'default' namespace
        ns = backend.get_namespace(DEFAULT_NAMESPACE)
        assert ns is not None
        assert ns.schema_version >= 1

        # Legacy data should still be accessible
        entry = backend.get_entry("legacy_key")
        assert entry is not None
        assert entry["data_type"] == "pickle"

        stats = backend.get_stats()
        assert stats["cache_hits"] == 10

        backend.close()
