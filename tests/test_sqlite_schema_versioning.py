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
        # v1 baseline + v1→v2 partial index + v2→v3 access_count/ttl/expires_at + v3→v4 encryption
        assert version == 4

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


class TestSqliteLifecyclePragmas:
    """Test SQLite connection and close lifecycle PRAGMAs."""

    class _Connection:
        def __init__(self, calls, fail_optimize=False):
            self.calls = calls
            self.fail_optimize = fail_optimize

        def __enter__(self):
            self.calls.append("enter")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.calls.append("exit")
            return False

        def execute(self, statement):
            self.calls.append(f"execute:{statement}")
            if self.fail_optimize:
                raise RuntimeError("simulated optimize failure")

    class _Engine:
        def __init__(self, fail_optimize=False):
            self.calls = []
            self.fail_optimize = fail_optimize

        def connect(self):
            self.calls.append("connect")
            return TestSqliteLifecyclePragmas._Connection(
                self.calls, fail_optimize=self.fail_optimize
            )

        def dispose(self):
            self.calls.append("dispose")

    def test_close_runs_optimize_before_dispose(self):
        """close() should run PRAGMA optimize before disposing the engine."""
        backend = SqliteBackend.__new__(SqliteBackend)
        engine = self._Engine()
        backend.engine = engine

        backend.close()

        assert "execute:PRAGMA optimize" in engine.calls
        assert engine.calls.index("execute:PRAGMA optimize") < engine.calls.index(
            "dispose"
        )

    def test_close_disposes_engine_when_optimize_fails(self):
        """close() should still dispose the engine if optimize is best-effort."""
        backend = SqliteBackend.__new__(SqliteBackend)
        engine = self._Engine(fail_optimize=True)
        backend.engine = engine

        backend.close()

        assert "execute:PRAGMA optimize" in engine.calls
        assert "dispose" in engine.calls


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
        assert ns.schema_version == 4

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
                    entry_signature VARCHAR(100),
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


class TestSqlitePartialIndex:
    """Test the v2 partial index on metadata_dict IS NOT NULL."""

    def test_schema_version_is_v4(self, sqlite_backend):
        """Fresh database should be at schema version 4."""
        version = sqlite_backend.get_schema_version(DEFAULT_NAMESPACE)
        assert version == 4

    def test_partial_index_exists_default_table(self, sqlite_backend):
        """Default table should have idx_metadata_notnull partial index."""
        from sqlalchemy import inspect

        inspector = inspect(sqlite_backend.engine)
        indexes = inspector.get_indexes("cache_entries")
        index_names = [idx["name"] for idx in indexes]
        assert "idx_metadata_notnull" in index_names

    def test_partial_index_has_correct_columns(self, sqlite_backend):
        """The partial index should be on created_at."""
        from sqlalchemy import inspect

        inspector = inspect(sqlite_backend.engine)
        indexes = inspector.get_indexes("cache_entries")
        meta_idx = next(idx for idx in indexes if idx["name"] == "idx_metadata_notnull")
        assert "created_at" in meta_idx["column_names"]

    def test_partial_index_exists_on_namespace_table(self, sqlite_backend):
        """Namespace tables should also have the partial index."""
        sqlite_backend.create_namespace("idx_test_ns")

        from sqlalchemy import inspect

        inspector = inspect(sqlite_backend.engine)
        indexes = inspector.get_indexes("cache_entries_idx_test_ns")
        index_names = [idx["name"] for idx in indexes]
        assert "idx_idx_test_ns_metadata_notnull" in index_names

    def test_migration_creates_index_on_legacy_db(self, tmp_path):
        """Opening a v1 database should auto-migrate to v2 and add the index."""
        from sqlalchemy import create_engine, text, inspect
        from sqlalchemy.orm import sessionmaker

        db_file = str(tmp_path / "legacy_v1.db")

        # Create a v1 database manually (tables but NO partial index)
        engine = create_engine(f"sqlite:///{db_file}")
        Session = sessionmaker(bind=engine)

        with Session() as session:
            session.execute(
                text("""
                CREATE TABLE cacheness_namespaces (
                    namespace_id VARCHAR(100) PRIMARY KEY,
                    display_name VARCHAR(200) NOT NULL DEFAULT '',
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    created_at DATETIME NOT NULL DEFAULT (datetime('now')),
                    signature VARCHAR(200)
                )
            """)
            )
            session.execute(
                text("""
                INSERT INTO cacheness_namespaces
                    (namespace_id, display_name, schema_version, created_at)
                VALUES ('default', 'Default', 1, datetime('now'))
            """)
            )
            session.execute(
                text("""
                CREATE TABLE cache_entries (
                    cache_key VARCHAR(16) PRIMARY KEY,
                    description VARCHAR(500) NOT NULL DEFAULT '',
                    data_type VARCHAR(20) NOT NULL,
                    created_at DATETIME NOT NULL,
                    accessed_at DATETIME NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    file_hash VARCHAR(16),
                    entry_signature VARCHAR(100),
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
            # Create v1 indexes (no partial index)
            session.execute(
                text("CREATE INDEX idx_list_entries ON cache_entries (created_at DESC)")
            )
            session.execute(
                text("CREATE INDEX idx_cleanup ON cache_entries (created_at)")
            )
            session.execute(
                text(
                    "CREATE INDEX idx_size_mgmt ON cache_entries (file_size, created_at)"
                )
            )
            session.execute(
                text("CREATE INDEX idx_data_type ON cache_entries (data_type)")
            )
            session.commit()
        engine.dispose()

        # Verify NO partial index before opening with SqliteBackend
        engine2 = create_engine(f"sqlite:///{db_file}")
        inspector = inspect(engine2)
        index_names = [idx["name"] for idx in inspector.get_indexes("cache_entries")]
        assert "idx_metadata_notnull" not in index_names
        engine2.dispose()

        # Open with SqliteBackend — should auto-migrate v1→v2→v3→v4
        backend = SqliteBackend(db_file)

        # Schema should now be v4
        assert backend.get_schema_version(DEFAULT_NAMESPACE) == 4

        # Partial index should exist
        inspector = inspect(backend.engine)
        index_names = [idx["name"] for idx in inspector.get_indexes("cache_entries")]
        assert "idx_metadata_notnull" in index_names

        backend.close()

    def test_migration_is_idempotent(self, sqlite_backend_path):
        """Opening the same database twice should not fail on duplicate index."""
        backend1 = SqliteBackend(sqlite_backend_path)
        v1 = backend1.get_schema_version(DEFAULT_NAMESPACE)
        backend1.close()

        backend2 = SqliteBackend(sqlite_backend_path)
        v2 = backend2.get_schema_version(DEFAULT_NAMESPACE)
        backend2.close()

        assert v1 == v2 == 4

    def test_query_plan_uses_partial_index(self, sqlite_backend):
        """EXPLAIN QUERY PLAN should reference the partial index."""
        from sqlalchemy import text

        with sqlite_backend.SessionLocal() as session:
            plan = session.execute(
                text(
                    "EXPLAIN QUERY PLAN "
                    "SELECT cache_key, metadata_dict FROM cache_entries "
                    "WHERE metadata_dict IS NOT NULL "
                    "ORDER BY created_at DESC"
                )
            ).all()
            plan_text = " ".join(str(row) for row in plan)
            # SQLite should mention the partial index in the plan
            assert "idx_metadata_notnull" in plan_text


class TestSqliteV3Columns:
    """Test v3 schema columns: access_count, ttl_seconds, expires_at."""

    def test_v3_columns_exist_on_fresh_db(self, sqlite_backend):
        """Fresh database should have the v3 columns."""
        from sqlalchemy import inspect

        inspector = inspect(sqlite_backend.engine)
        cols = {c["name"] for c in inspector.get_columns("cache_entries")}
        assert "access_count" in cols
        assert "ttl_seconds" in cols
        assert "expires_at" in cols

    def test_v3_indexes_exist_on_fresh_db(self, sqlite_backend):
        """Fresh database should have idx_expires_at and idx_access_count."""
        from sqlalchemy import inspect

        inspector = inspect(sqlite_backend.engine)
        indexes = inspector.get_indexes("cache_entries")
        idx_names = {idx["name"] for idx in indexes}
        assert "idx_expires_at" in idx_names
        assert "idx_access_count" in idx_names

    def test_v3_columns_on_namespace_table(self, sqlite_backend):
        """Namespace tables created via create_namespace should have v3 columns."""
        sqlite_backend.create_namespace("v3_test_ns")
        from sqlalchemy import inspect

        inspector = inspect(sqlite_backend.engine)
        cols = {c["name"] for c in inspector.get_columns("cache_entries_v3_test_ns")}
        assert "access_count" in cols
        assert "ttl_seconds" in cols
        assert "expires_at" in cols

    def test_v3_indexes_on_namespace_table(self, sqlite_backend):
        """Namespace tables should have v3 indexes."""
        sqlite_backend.create_namespace("v3_idx_ns")
        from sqlalchemy import inspect

        inspector = inspect(sqlite_backend.engine)
        indexes = inspector.get_indexes("cache_entries_v3_idx_ns")
        idx_names = {idx["name"] for idx in indexes}
        assert "idx_v3_idx_ns_expires_at" in idx_names
        assert "idx_v3_idx_ns_access_count" in idx_names

    def test_access_count_starts_at_zero(self, sqlite_backend):
        """New entries should have access_count = 0."""
        sqlite_backend.put_entry(
            "ac_test_001",
            {
                "data_type": "pickle",
                "description": "test access count",
                "file_size": 100,
                "metadata": {},
            },
        )
        entry = sqlite_backend.get_entry("ac_test_001")
        assert entry is not None
        assert entry["access_count"] == 0

    def test_access_count_incremented_on_access(self, sqlite_backend):
        """update_access_time should increment access_count."""
        sqlite_backend.put_entry(
            "ac_test_002",
            {
                "data_type": "pickle",
                "description": "test increment",
                "file_size": 100,
                "metadata": {},
            },
        )

        sqlite_backend.update_access_time("ac_test_002")
        entry = sqlite_backend.get_entry("ac_test_002")
        assert entry["access_count"] == 1

        sqlite_backend.update_access_time("ac_test_002")
        entry = sqlite_backend.get_entry("ac_test_002")
        assert entry["access_count"] == 2

    def test_ttl_seconds_stored_on_put(self, sqlite_backend):
        """put_entry with ttl_seconds should store it and compute expires_at."""
        sqlite_backend.put_entry(
            "ttl_test_001",
            {
                "data_type": "pickle",
                "description": "test ttl",
                "file_size": 100,
                "ttl_seconds": 3600,
                "metadata": {},
            },
        )
        entry = sqlite_backend.get_entry("ttl_test_001")
        assert entry is not None
        assert entry["ttl_seconds"] == 3600
        assert entry["expires_at"] is not None

    def test_ttl_null_when_not_provided(self, sqlite_backend):
        """put_entry without ttl_seconds should leave ttl/expires_at as NULL."""
        sqlite_backend.put_entry(
            "ttl_test_002",
            {
                "data_type": "pickle",
                "description": "no ttl",
                "file_size": 100,
                "metadata": {},
            },
        )
        entry = sqlite_backend.get_entry("ttl_test_002")
        assert entry is not None
        assert entry["ttl_seconds"] is None
        assert entry["expires_at"] is None

    def test_iter_entry_summaries_includes_v3_fields(self, sqlite_backend):
        """iter_entry_summaries should include access_count and TTL fields."""
        sqlite_backend.put_entry(
            "sum_test_001",
            {
                "data_type": "pickle",
                "description": "summary test",
                "file_size": 100,
                "ttl_seconds": 7200,
                "metadata": {},
            },
        )
        sqlite_backend.update_access_time("sum_test_001")

        summaries = sqlite_backend.iter_entry_summaries()
        entry = next(s for s in summaries if s["cache_key"] == "sum_test_001")
        assert entry["access_count"] == 1
        assert entry["ttl_seconds"] == 7200
        assert "expires_at" in entry

    def test_migration_from_v1_adds_v3_columns(self, tmp_path):
        """Opening a v1 database should migrate through v2 and v3."""
        from sqlalchemy import create_engine, text, inspect
        from sqlalchemy.orm import sessionmaker

        db_file = str(tmp_path / "legacy_v1_to_v3.db")

        # Create a v1 database manually (no partial index, no v3 columns)
        engine = create_engine(f"sqlite:///{db_file}")
        Session = sessionmaker(bind=engine)

        with Session() as session:
            session.execute(
                text("""
                CREATE TABLE cacheness_namespaces (
                    namespace_id VARCHAR(100) PRIMARY KEY,
                    display_name VARCHAR(200) NOT NULL DEFAULT '',
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    created_at DATETIME NOT NULL DEFAULT (datetime('now')),
                    signature VARCHAR(200)
                )
            """)
            )
            session.execute(
                text("""
                INSERT INTO cacheness_namespaces
                    (namespace_id, display_name, schema_version, created_at)
                VALUES ('default', 'Default', 1, datetime('now'))
            """)
            )
            session.execute(
                text("""
                CREATE TABLE cache_entries (
                    cache_key VARCHAR(16) PRIMARY KEY,
                    description VARCHAR(500) NOT NULL DEFAULT '',
                    data_type VARCHAR(20) NOT NULL,
                    created_at DATETIME NOT NULL,
                    accessed_at DATETIME NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    file_hash VARCHAR(16),
                    entry_signature VARCHAR(100),
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
                text("CREATE INDEX idx_list_entries ON cache_entries (created_at DESC)")
            )
            session.execute(
                text("CREATE INDEX idx_cleanup ON cache_entries (created_at)")
            )
            session.execute(
                text(
                    "CREATE INDEX idx_size_mgmt ON cache_entries (file_size, created_at)"
                )
            )
            session.execute(
                text("CREATE INDEX idx_data_type ON cache_entries (data_type)")
            )
            # Insert a test entry at v1
            session.execute(
                text("""
                INSERT INTO cache_entries (cache_key, data_type, created_at, accessed_at, file_size)
                VALUES ('migrated_001', 'pickle', datetime('now'), datetime('now'), 42)
            """)
            )
            session.commit()
        engine.dispose()

        # Open with SqliteBackend — should auto-migrate v1→v2→v3→v4
        backend = SqliteBackend(db_file)
        assert backend.get_schema_version(DEFAULT_NAMESPACE) == 4

        # v3 columns should exist
        inspector = inspect(backend.engine)
        cols = {c["name"] for c in inspector.get_columns("cache_entries")}
        assert "access_count" in cols
        assert "ttl_seconds" in cols
        assert "expires_at" in cols

        # v3 indexes should exist
        idx_names = {idx["name"] for idx in inspector.get_indexes("cache_entries")}
        assert "idx_expires_at" in idx_names
        assert "idx_access_count" in idx_names

        # Pre-existing entry should have defaults (access_count=0, ttl/expires=NULL)
        entry = backend.get_entry("migrated_001")
        assert entry is not None
        assert entry["access_count"] == 0
        assert entry["ttl_seconds"] is None
        assert entry["expires_at"] is None

        backend.close()


class TestSqliteV3ToV4Migration:
    """Test v3→v4 migration adds encryption columns."""

    def test_v3_to_v4_migration_adds_columns(self, sqlite_backend):
        """Verify migration adds encryption_algorithm, encryption_iv, cacheness_version."""
        from sqlalchemy import text

        with sqlite_backend.SessionLocal() as session:
            cols = {
                row[1]
                for row in session.execute(
                    text('PRAGMA table_info("cache_entries")')
                ).fetchall()
            }
        assert "encryption_algorithm" in cols
        assert "encryption_iv" in cols
        assert "cacheness_version" in cols

    def test_encryption_metadata_roundtrip_sqlite(self, sqlite_backend):
        """Verify put_entry preserves encryption fields and get_entry returns them."""
        entry_data = {
            "description": "encrypted entry",
            "data_type": "bytes",
            "file_size": 100,
            "metadata": {
                "object_type": "bytes",
                "storage_format": "raw",
                "encryption_algorithm": "AES-256-GCM",
                "encryption_iv": "abcdef0123456789",
                "file_hash": "somehash",
            },
        }
        sqlite_backend.put_entry("test_enc_key", entry_data)
        result = sqlite_backend.get_entry("test_enc_key")
        assert result is not None
        meta = result["metadata"]
        assert meta["encryption_algorithm"] == "AES-256-GCM"
        assert meta["encryption_iv"] == "abcdef0123456789"

    def test_cacheness_version_roundtrip_sqlite(self, sqlite_backend):
        """Verify cacheness_version column preserved through put/get."""
        entry_data = {
            "description": "versioned entry",
            "data_type": "bytes",
            "file_size": 50,
            "metadata": {
                "cacheness_version": "0.11.0",
            },
        }
        sqlite_backend.put_entry("test_ver_key", entry_data)
        result = sqlite_backend.get_entry("test_ver_key")
        assert result is not None
        assert result["metadata"]["cacheness_version"] == "0.11.0"

    def test_iter_entry_summaries_includes_encryption_fields(self, sqlite_backend):
        """Verify iter_entry_summaries includes encryption metadata."""
        entry_data = {
            "description": "enc summary",
            "data_type": "bytes",
            "file_size": 100,
            "metadata": {
                "encryption_algorithm": "AES-256-GCM",
                "encryption_iv": "deadbeef",
            },
        }
        sqlite_backend.put_entry("sum_enc_key", entry_data)
        summaries = sqlite_backend.iter_entry_summaries()
        enc_summary = [s for s in summaries if s["cache_key"] == "sum_enc_key"]
        assert len(enc_summary) == 1
        assert enc_summary[0]["encryption_algorithm"] == "AES-256-GCM"
        assert enc_summary[0]["encryption_iv"] == "deadbeef"

    def test_update_entry_metadata_encryption_fields(self, sqlite_backend):
        """Verify update_entry_metadata can update encryption fields."""
        entry_data = {
            "description": "update enc",
            "data_type": "bytes",
            "file_size": 100,
            "metadata": {
                "encryption_algorithm": "AES-256-GCM",
                "encryption_iv": "old_iv",
            },
        }
        sqlite_backend.put_entry("upd_enc_key", entry_data)
        sqlite_backend.update_entry_metadata(
            "upd_enc_key", {"encryption_iv": "new_iv"}
        )
        result = sqlite_backend.get_entry("upd_enc_key")
        assert result["metadata"]["encryption_iv"] == "new_iv"


class TestSqliteV3ToV4RealDataMigration:
    """Test v3->v4 migration on a database with existing cached entries (HARD-02)."""

    def test_v3_to_v4_migration_preserves_existing_data(self, tmp_path):
        """Opening a v3 database with entries should migrate to v4 and preserve all data."""
        import json
        from sqlalchemy import create_engine, text, inspect as sa_inspect
        from sqlalchemy.orm import sessionmaker

        db_file = str(tmp_path / "v3_with_data.db")
        engine = create_engine(f"sqlite:///{db_file}")
        Session = sessionmaker(bind=engine)

        with Session() as session:
            session.execute(text("""
                CREATE TABLE cacheness_namespaces (
                    namespace_id VARCHAR(100) PRIMARY KEY,
                    display_name VARCHAR(200) NOT NULL DEFAULT '',
                    schema_version INTEGER NOT NULL DEFAULT 3,
                    created_at DATETIME NOT NULL DEFAULT (datetime('now')),
                    signature VARCHAR(200)
                )
            """))
            session.execute(text("""
                INSERT INTO cacheness_namespaces
                    (namespace_id, display_name, schema_version)
                VALUES ('default', 'Default', 3)
            """))

            session.execute(text("""
                CREATE TABLE cache_entries (
                    cache_key VARCHAR(16) PRIMARY KEY,
                    description VARCHAR(500) NOT NULL DEFAULT '',
                    data_type VARCHAR(20) NOT NULL,
                    created_at DATETIME NOT NULL,
                    accessed_at DATETIME NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    file_hash VARCHAR(16),
                    entry_signature VARCHAR(100),
                    s3_etag VARCHAR(100),
                    object_type VARCHAR(100),
                    storage_format VARCHAR(20),
                    serializer VARCHAR(20),
                    compression_codec VARCHAR(20),
                    actual_path VARCHAR(500),
                    cache_key_params TEXT,
                    metadata_dict TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    ttl_seconds INTEGER,
                    expires_at DATETIME,
                    blob_data BLOB,
                    is_inline INTEGER NOT NULL DEFAULT 0,
                    inline_ext TEXT
                )
            """))

            session.execute(text("""
                CREATE TABLE cache_stats (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    cache_hits INTEGER NOT NULL DEFAULT 0,
                    cache_misses INTEGER NOT NULL DEFAULT 0,
                    last_updated DATETIME NOT NULL DEFAULT (datetime('now'))
                )
            """))
            session.execute(text("""
                INSERT INTO cache_stats (id, cache_hits, cache_misses)
                VALUES (1, 10, 5)
            """))

            session.execute(text(
                "CREATE INDEX idx_list_entries ON cache_entries (created_at DESC)"
            ))
            session.execute(text(
                "CREATE INDEX idx_cleanup ON cache_entries (created_at)"
            ))
            session.execute(text(
                "CREATE INDEX idx_size_mgmt ON cache_entries (file_size, created_at)"
            ))
            session.execute(text(
                "CREATE INDEX idx_data_type ON cache_entries (data_type)"
            ))
            session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_metadata_notnull "
                "ON cache_entries (metadata_dict, created_at DESC) "
                "WHERE metadata_dict IS NOT NULL"
            ))
            session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_expires_at "
                "ON cache_entries (expires_at) WHERE expires_at IS NOT NULL"
            ))
            session.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_access_count "
                "ON cache_entries (access_count, accessed_at)"
            ))

            # Insert plain cached entry
            session.execute(text("""
                INSERT INTO cache_entries
                    (cache_key, description, data_type, created_at, accessed_at,
                     file_size, file_hash, object_type, storage_format, metadata_dict)
                VALUES
                    ('plain_entry_01', 'test plain entry', 'pickle',
                     datetime('now'), datetime('now'), 256, 'hash_abc123',
                     'dict', 'pickle', :metadata)
            """), {"metadata": json.dumps({
                "object_type": "dict",
                "storage_format": "pickle",
            })})

            # Insert entry with custom metadata
            session.execute(text("""
                INSERT INTO cache_entries
                    (cache_key, description, data_type, created_at, accessed_at,
                     file_size, file_hash, object_type, storage_format, metadata_dict)
                VALUES
                    ('custom_meta_01', 'entry with custom metadata', 'pickle',
                     datetime('now'), datetime('now'), 512, 'hash_def456',
                     'list', 'pickle', :metadata)
            """), {"metadata": json.dumps({
                "object_type": "list",
                "storage_format": "pickle",
                "user_tag": "experiment_42",
                "model_version": "v2.1",
            })})

            session.commit()
        engine.dispose()

        # Verify v3 schema (no v4 columns)
        engine2 = create_engine(f"sqlite:///{db_file}")
        with engine2.connect() as conn:
            cols = {
                row[1]
                for row in conn.execute(
                    text('PRAGMA table_info("cache_entries")')
                ).fetchall()
            }
        assert "encryption_algorithm" not in cols
        engine2.dispose()

        # Open with SqliteBackend -- should auto-migrate v3->v4
        backend = SqliteBackend(db_file)

        # Schema should now be v4
        assert backend.get_schema_version(DEFAULT_NAMESPACE) == 4

        # Verify v4 columns exist
        inspector = sa_inspect(backend.engine)
        col_names = {c["name"] for c in inspector.get_columns("cache_entries")}
        assert "encryption_algorithm" in col_names
        assert "encryption_iv" in col_names
        assert "cacheness_version" in col_names

        # Verify plain entry survived migration
        plain = backend.get_entry("plain_entry_01")
        assert plain is not None
        assert plain["description"] == "test plain entry"
        assert plain["data_type"] == "pickle"
        assert plain["file_size"] == 256

        # Verify custom metadata entry survived migration
        custom = backend.get_entry("custom_meta_01")
        assert custom is not None
        assert custom["description"] == "entry with custom metadata"
        meta = custom.get("metadata", {})
        metadata_value = meta.get("metadata_dict", {})
        metadata_dict = (
            metadata_value
            if isinstance(metadata_value, dict)
            else json.loads(metadata_value)
        )
        assert metadata_dict.get("user_tag") == "experiment_42"
        assert metadata_dict.get("model_version") == "v2.1"

        backend.close()

    def test_v3_to_v4_migration_preserves_custom_metadata(self, tmp_path):
        """Custom metadata stored in metadata_dict survives v3->v4 migration."""
        import json
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker

        db_file = str(tmp_path / "v3_custom_meta.db")
        engine = create_engine(f"sqlite:///{db_file}")
        Session = sessionmaker(bind=engine)

        with Session() as session:
            session.execute(text("""
                CREATE TABLE cacheness_namespaces (
                    namespace_id VARCHAR(100) PRIMARY KEY,
                    display_name VARCHAR(200) NOT NULL DEFAULT '',
                    schema_version INTEGER NOT NULL DEFAULT 3,
                    created_at DATETIME NOT NULL DEFAULT (datetime('now')),
                    signature VARCHAR(200)
                )
            """))
            session.execute(text(
                "INSERT INTO cacheness_namespaces "
                "(namespace_id, display_name, schema_version) "
                "VALUES ('default', 'Default', 3)"
            ))

            session.execute(text("""
                CREATE TABLE cache_entries (
                    cache_key VARCHAR(16) PRIMARY KEY,
                    description VARCHAR(500) NOT NULL DEFAULT '',
                    data_type VARCHAR(20) NOT NULL,
                    created_at DATETIME NOT NULL,
                    accessed_at DATETIME NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    file_hash VARCHAR(16),
                    entry_signature VARCHAR(100),
                    s3_etag VARCHAR(100),
                    object_type VARCHAR(100),
                    storage_format VARCHAR(20),
                    serializer VARCHAR(20),
                    compression_codec VARCHAR(20),
                    actual_path VARCHAR(500),
                    cache_key_params TEXT,
                    metadata_dict TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    ttl_seconds INTEGER,
                    expires_at DATETIME,
                    blob_data BLOB,
                    is_inline INTEGER NOT NULL DEFAULT 0,
                    inline_ext TEXT
                )
            """))

            session.execute(text("""
                CREATE TABLE cache_stats (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    cache_hits INTEGER NOT NULL DEFAULT 0,
                    cache_misses INTEGER NOT NULL DEFAULT 0,
                    last_updated DATETIME NOT NULL DEFAULT (datetime('now'))
                )
            """))

            for i in range(5):
                session.execute(text("""
                    INSERT INTO cache_entries
                        (cache_key, description, data_type, created_at,
                         accessed_at, file_size, object_type, storage_format,
                         metadata_dict)
                    VALUES
                        (:key, :desc, 'pickle', datetime('now'),
                         datetime('now'), :size, 'dict', 'pickle', :metadata)
                """), {
                    "key": f"batch_{i:03d}",
                    "desc": f"batch entry {i}",
                    "size": 100 + i * 50,
                    "metadata": json.dumps({
                        "object_type": "dict",
                        "storage_format": "pickle",
                        "batch_id": i,
                        "run_name": f"run_{i}",
                    }),
                })
            session.commit()
        engine.dispose()

        # Migrate via SqliteBackend
        backend = SqliteBackend(db_file)
        assert backend.get_schema_version(DEFAULT_NAMESPACE) == 4

        # Verify all 5 entries survived
        for i in range(5):
            entry = backend.get_entry(f"batch_{i:03d}")
            assert entry is not None, f"Entry batch_{i:03d} lost during migration"
            assert entry["description"] == f"batch entry {i}"
            meta = entry.get("metadata", {})
            metadata_value = meta.get("metadata_dict", {})
            metadata_dict = (
                metadata_value
                if isinstance(metadata_value, dict)
                else json.loads(metadata_value)
            )
            assert metadata_dict.get("batch_id") == i
            assert metadata_dict.get("run_name") == f"run_{i}"

        backend.close()
