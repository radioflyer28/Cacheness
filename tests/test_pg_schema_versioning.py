"""Tests for PostgreSQL backend schema versioning and namespace registry.

These tests use a live PostgreSQL database when available (via
CACHENESS_TEST_POSTGRES_URL), otherwise they are skipped.
"""

import os

import pytest

from cacheness.metadata import (
    NamespaceInfo,
    DEFAULT_NAMESPACE,
    validate_namespace_id,
)

# Try to import PG backend
try:
    from cacheness.storage.backends.postgresql_backend import (
        PostgresBackend,
        PgCacheNamespace,
        SQLALCHEMY_AVAILABLE,
        PSYCOPG_AVAILABLE,
    )

    _HAS_PG = SQLALCHEMY_AVAILABLE and PSYCOPG_AVAILABLE
except ImportError:
    _HAS_PG = False


def _get_pg_url():
    """Get PostgreSQL URL from environment or return None."""
    return os.environ.get("CACHENESS_TEST_POSTGRES_URL")


# Shared skip condition
requires_postgres = pytest.mark.skipif(
    not _HAS_PG or not _get_pg_url(),
    reason="Requires PostgreSQL (set CACHENESS_TEST_POSTGRES_URL)",
)


@pytest.fixture
def pg_backend():
    """Create a fresh PostgreSQL backend, cleaning up namespace tables after."""
    url = _get_pg_url()
    if not url:
        pytest.skip("CACHENESS_TEST_POSTGRES_URL not set")
    backend = PostgresBackend(connection_url=url)
    yield backend

    # Cleanup: drop any test namespaces we created
    for ns in backend.list_namespaces():
        if ns.namespace_id != DEFAULT_NAMESPACE:
            try:
                backend.drop_namespace(ns.namespace_id)
            except Exception:
                pass
    backend.close()


# ── Schema versioning ─────────────────────────────────────────────────


@requires_postgres
class TestPgSchemaVersioning:
    """Test schema versioning on PostgresBackend."""

    def test_default_namespace_registered_on_init(self, pg_backend):
        """Creating a backend should register the 'default' namespace."""
        ns = pg_backend.get_namespace(DEFAULT_NAMESPACE)
        assert ns is not None
        assert ns.namespace_id == DEFAULT_NAMESPACE
        assert ns.display_name == "Default"
        assert ns.schema_version >= 1

    def test_get_schema_version_default(self, pg_backend):
        """Schema version should be >= 1 after init (migrations ran)."""
        version = pg_backend.get_schema_version(DEFAULT_NAMESPACE)
        assert version >= 1

    def test_set_schema_version(self, pg_backend):
        """set_schema_version should persist."""
        pg_backend.set_schema_version(DEFAULT_NAMESPACE, 99)
        assert pg_backend.get_schema_version(DEFAULT_NAMESPACE) == 99
        # Restore
        pg_backend.set_schema_version(DEFAULT_NAMESPACE, 1)

    def test_get_schema_version_unknown_namespace(self, pg_backend):
        """Unknown namespace returns 0."""
        assert pg_backend.get_schema_version("nonexistent") == 0

    def test_migrations_run_on_fresh_db(self, pg_backend):
        """Fresh database should have all migrations applied."""
        version = pg_backend.get_schema_version(DEFAULT_NAMESPACE)
        assert version == 1

    def test_migrations_idempotent(self):
        """Opening the same database twice should not fail or re-run migrations."""
        url = _get_pg_url()
        backend1 = PostgresBackend(connection_url=url)
        v1 = backend1.get_schema_version(DEFAULT_NAMESPACE)
        backend1.close()

        backend2 = PostgresBackend(connection_url=url)
        v2 = backend2.get_schema_version(DEFAULT_NAMESPACE)
        backend2.close()

        assert v1 == v2

    def test_cacheness_namespaces_table_exists(self, pg_backend):
        """The cacheness_namespaces table should exist after init."""
        from sqlalchemy import inspect

        inspector = inspect(pg_backend.engine)
        tables = inspector.get_table_names()
        assert "cacheness_namespaces" in tables

    def test_legacy_run_migrations_is_noop(self, pg_backend):
        """The old _run_migrations() method should be a harmless no-op."""
        pg_backend._run_migrations()  # Should not raise


# ── Namespace registry ─────────────────────────────────────────────────


@requires_postgres
class TestPgNamespaceRegistry:
    """Test namespace registry on PostgresBackend."""

    def test_list_namespaces_has_default(self, pg_backend):
        """list_namespaces should include 'default' after init."""
        namespaces = pg_backend.list_namespaces()
        ids = [ns.namespace_id for ns in namespaces]
        assert DEFAULT_NAMESPACE in ids

    def test_create_namespace(self, pg_backend):
        """create_namespace should register and create tables."""
        ns = pg_backend.create_namespace("project_alpha", "Project Alpha")
        assert ns.namespace_id == "project_alpha"
        assert ns.display_name == "Project Alpha"
        assert ns.schema_version == 1

        # Per-namespace tables should exist
        from sqlalchemy import inspect

        inspector = inspect(pg_backend.engine)
        tables = inspector.get_table_names()
        assert "cache_entries_project_alpha" in tables
        assert "cache_stats_project_alpha" in tables

        # Should appear in list
        ids = [n.namespace_id for n in pg_backend.list_namespaces()]
        assert "project_alpha" in ids

    def test_create_namespace_duplicate_raises(self, pg_backend):
        """Creating a namespace that already exists should raise ValueError."""
        pg_backend.create_namespace("dup_ns")
        with pytest.raises(ValueError, match="already exists"):
            pg_backend.create_namespace("dup_ns")

    def test_create_default_namespace_raises(self, pg_backend):
        """Cannot re-create the default namespace."""
        with pytest.raises(ValueError, match="pre-registered"):
            pg_backend.create_namespace(DEFAULT_NAMESPACE)

    def test_create_namespace_invalid_id(self, pg_backend):
        """Invalid namespace IDs should be rejected."""
        with pytest.raises(ValueError, match="Invalid namespace ID"):
            pg_backend.create_namespace("UPPERCASE")
        with pytest.raises(ValueError, match="Invalid namespace ID"):
            pg_backend.create_namespace("has-dash")
        with pytest.raises(ValueError, match="Invalid namespace ID"):
            pg_backend.create_namespace("")

    def test_drop_namespace(self, pg_backend):
        """drop_namespace should remove tables and registry entry."""
        pg_backend.create_namespace("to_drop")
        from sqlalchemy import inspect

        inspector = inspect(pg_backend.engine)
        assert "cache_entries_to_drop" in inspector.get_table_names()

        result = pg_backend.drop_namespace("to_drop")
        assert result is True

        # Refresh inspector
        inspector = inspect(pg_backend.engine)
        assert "cache_entries_to_drop" not in inspector.get_table_names()
        assert not pg_backend.namespace_exists("to_drop")

    def test_drop_nonexistent_namespace(self, pg_backend):
        """Dropping a namespace that doesn't exist returns False."""
        assert pg_backend.drop_namespace("nonexistent") is False

    def test_drop_default_raises(self, pg_backend):
        """Cannot drop the 'default' namespace."""
        with pytest.raises(ValueError, match="Cannot drop"):
            pg_backend.drop_namespace(DEFAULT_NAMESPACE)

    def test_get_namespace(self, pg_backend):
        """get_namespace should return NamespaceInfo or None."""
        pg_backend.create_namespace("lookup_ns", "Lookup Test")
        ns = pg_backend.get_namespace("lookup_ns")
        assert ns is not None
        assert ns.namespace_id == "lookup_ns"
        assert ns.display_name == "Lookup Test"
        assert ns.schema_version == 1

        assert pg_backend.get_namespace("nonexistent") is None

    def test_namespace_exists(self, pg_backend):
        """namespace_exists should return correct boolean."""
        assert pg_backend.namespace_exists(DEFAULT_NAMESPACE) is True
        assert pg_backend.namespace_exists("nope") is False

        pg_backend.create_namespace("check_me")
        assert pg_backend.namespace_exists("check_me") is True

    def test_multiple_namespaces(self, pg_backend):
        """Can create and list multiple namespaces."""
        pg_backend.create_namespace("ns_one")
        pg_backend.create_namespace("ns_two")
        pg_backend.create_namespace("ns_three")

        namespaces = pg_backend.list_namespaces()
        ids = {ns.namespace_id for ns in namespaces}
        assert {"ns_one", "ns_two", "ns_three"}.issubset(ids)
        assert DEFAULT_NAMESPACE in ids

    def test_namespace_indexes_created(self, pg_backend):
        """Per-namespace tables should have indexes."""
        pg_backend.create_namespace("idx_test")
        from sqlalchemy import inspect

        inspector = inspect(pg_backend.engine)
        indexes = inspector.get_indexes("cache_entries_idx_test")
        idx_names = {idx["name"] for idx in indexes}
        assert "idx_idx_test_list_entries" in idx_names
        assert "idx_idx_test_cleanup" in idx_names
        assert "idx_idx_test_size_mgmt" in idx_names

    def test_drop_namespace_then_recreate(self, pg_backend):
        """Can drop and re-create the same namespace."""
        pg_backend.create_namespace("recyclable")
        pg_backend.drop_namespace("recyclable")
        assert not pg_backend.namespace_exists("recyclable")

        ns = pg_backend.create_namespace("recyclable", "Second time")
        assert ns.display_name == "Second time"
        assert pg_backend.namespace_exists("recyclable")

    def test_set_schema_version_on_created_namespace(self, pg_backend):
        """Schema version can be set on namespaces created via create_namespace."""
        pg_backend.create_namespace("versioned_ns")
        assert pg_backend.get_schema_version("versioned_ns") == 1

        pg_backend.set_schema_version("versioned_ns", 5)
        assert pg_backend.get_schema_version("versioned_ns") == 5


# ── Backward compatibility ─────────────────────────────────────────────


@requires_postgres
class TestPgBackwardCompatibility:
    """Ensure existing PostgresBackend operations still work correctly."""

    def test_put_and_get_entry(self, pg_backend):
        """Basic put/get should still work after namespace additions."""
        pg_backend.put_entry(
            "pg_test_001",
            {
                "description": "test entry",
                "data_type": "pickle",
                "metadata": {"key": "value"},
            },
        )
        entry = pg_backend.get_entry("pg_test_001")
        assert entry is not None
        assert entry["description"] == "test entry"
        assert entry["data_type"] == "pickle"

        # Cleanup
        pg_backend.remove_entry("pg_test_001")

    def test_list_entries(self, pg_backend):
        """list_entries should still work."""
        pg_backend.put_entry(
            "pg_test_002",
            {"data_type": "pickle", "file_size": 1024},
        )
        entries = pg_backend.list_entries()
        keys = [e["cache_key"] for e in entries]
        assert "pg_test_002" in keys

        # Cleanup
        pg_backend.remove_entry("pg_test_002")

    def test_stats(self, pg_backend):
        """get_stats should still work."""
        stats = pg_backend.get_stats()
        assert "hits" in stats or "cache_hits" in stats

    def test_clear_all(self, pg_backend):
        """clear_all should still work."""
        pg_backend.put_entry("pg_test_003", {"data_type": "pickle"})
        count = pg_backend.clear_all()
        assert count >= 1
        assert pg_backend.get_entry("pg_test_003") is None
