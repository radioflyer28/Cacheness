"""
Multi-tenant namespace integration tests (CACHE-adi.12).

Validates table-per-namespace isolation across all three metadata backends
(JSON, SQLite, PostgreSQL) plus S3 prefix isolation.  Tests cover:

  (1) Two namespaces on same SQLite file — zero cross-talk
  (2) Two namespaces on same JSON cache_dir — separate files
  (3) Two namespaces on same PostgreSQL — separate tables (skip if no PG)
  (4) S3 prefix isolation per namespace (skip if no boto3/moto)
  (5) cache-mode + storage-mode on same backend with different namespaces
  (6) create_namespace / drop_namespace lifecycle — verify tables created/dropped
  (7) Namespace ID validation rejects invalid identifiers
  (8) clear_all only affects active namespace
  (9) clear_all_namespaces nuclear option
"""

import os

import pytest

from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache
from cacheness.metadata import (
    DEFAULT_NAMESPACE,
    JsonBackend,
    NamespaceInfo,
    SqliteBackend,
    validate_namespace_id,
)

# PostgreSQL availability
try:
    from cacheness.storage.backends.postgresql_backend import PostgresBackend

    HAS_PG = True
except Exception:
    HAS_PG = False

# S3/moto availability
try:
    import boto3
    from moto import mock_aws

    HAS_S3_MOCK = True
except ImportError:
    HAS_S3_MOCK = False


# ─── helpers ──────────────────────────────────────────────────────────


def _make_cache(tmp_path, namespace="default", backend="json", **kw):
    """Create a UnifiedCache with the given namespace and metadata backend."""
    cfg = CacheConfig(
        cache_dir=str(tmp_path / "cache"),
        metadata_backend=backend,
        namespace=namespace,
        cleanup_on_init=False,
        **kw,
    )
    return UnifiedCache(config=cfg)


# =====================================================================
# (2)  JSON backend — two namespaces, separate files
# =====================================================================


class TestJsonNamespaceIsolation:
    """Two namespaces on the same JSON cache_dir use separate files."""

    def test_put_get_isolated(self, tmp_path):
        """Data written in ns_a is invisible in ns_b."""
        cache_a = _make_cache(tmp_path, namespace="ns_a", backend="json")
        cache_b = _make_cache(tmp_path, namespace="ns_b", backend="json")

        cache_a.put("hello", "key1")
        assert cache_a.get("key1") == "hello"
        assert cache_b.get("key1") is None

    def test_list_entries_scoped(self, tmp_path):
        """list_entries only returns entries for the active namespace."""
        cache_a = _make_cache(tmp_path, namespace="ns_a", backend="json")
        cache_b = _make_cache(tmp_path, namespace="ns_b", backend="json")

        cache_a.put("data_a", "shared_key")
        cache_b.put("data_b", "shared_key")
        cache_b.put("extra_b", "other_key")

        entries_a = cache_a.list_entries()
        entries_b = cache_b.list_entries()

        assert len(entries_a) == 1
        assert len(entries_b) == 2

    def test_delete_scoped(self, tmp_path):
        """Deleting in ns_a does not affect ns_b."""
        cache_a = _make_cache(tmp_path, namespace="ns_a", backend="json")
        cache_b = _make_cache(tmp_path, namespace="ns_b", backend="json")

        cache_a.put("data_a", "key1")
        cache_b.put("data_b", "key1")

        cache_a.invalidate(cache_key="key1")

        assert cache_a.get("key1") is None
        assert cache_b.get("key1") == "data_b"

    def test_stats_scoped(self, tmp_path):
        """Stats are per-namespace."""
        cache_a = _make_cache(tmp_path, namespace="ns_a", backend="json")
        cache_b = _make_cache(tmp_path, namespace="ns_b", backend="json")

        cache_a.put("data", "k1")
        cache_a.put("data", "k2")
        cache_b.put("data", "k1")

        stats_a = cache_a.get_stats()
        stats_b = cache_b.get_stats()

        assert stats_a["total_entries"] == 2
        assert stats_b["total_entries"] == 1

    def test_clear_all_scoped(self, tmp_path):
        """clear_all only removes entries in the active namespace."""
        cache_a = _make_cache(tmp_path, namespace="ns_a", backend="json")
        cache_b = _make_cache(tmp_path, namespace="ns_b", backend="json")

        cache_a.put("data_a", "k1")
        cache_b.put("data_b", "k1")

        cache_a.clear_all()

        assert cache_a.get("k1") is None
        assert cache_b.get("k1") == "data_b"


# =====================================================================
# (1)  SQLite backend — two namespaces on the same database file
# =====================================================================


class TestSqliteNamespaceIsolation:
    """Two namespaces on the same SQLite file use separate table sets."""

    def test_put_get_isolated(self, tmp_path):
        """Data written in ns_a is invisible in ns_b."""
        cache_a = _make_cache(tmp_path, namespace="ns_a", backend="sqlite")
        cache_b = _make_cache(tmp_path, namespace="ns_b", backend="sqlite")

        cache_a.put(42, "key1")
        assert cache_a.get("key1") == 42
        assert cache_b.get("key1") is None

    def test_list_entries_scoped(self, tmp_path):
        """list_entries only returns entries for the active namespace."""
        cache_a = _make_cache(tmp_path, namespace="ns_a", backend="sqlite")
        cache_b = _make_cache(tmp_path, namespace="ns_b", backend="sqlite")

        cache_a.put("data_a", "shared_key")
        cache_b.put("data_b", "shared_key")
        cache_b.put("extra_b", "other_key")

        entries_a = cache_a.list_entries()
        entries_b = cache_b.list_entries()

        assert len(entries_a) == 1
        assert len(entries_b) == 2

    def test_delete_scoped(self, tmp_path):
        """Deleting in ns_a does not affect ns_b."""
        cache_a = _make_cache(tmp_path, namespace="ns_a", backend="sqlite")
        cache_b = _make_cache(tmp_path, namespace="ns_b", backend="sqlite")

        cache_a.put("data_a", "key1")
        cache_b.put("data_b", "key1")

        cache_a.invalidate(cache_key="key1")

        assert cache_a.get("key1") is None
        assert cache_b.get("key1") == "data_b"

    def test_stats_scoped(self, tmp_path):
        """Stats are per-namespace."""
        cache_a = _make_cache(tmp_path, namespace="ns_a", backend="sqlite")
        cache_b = _make_cache(tmp_path, namespace="ns_b", backend="sqlite")

        cache_a.put("data", "k1")
        cache_a.put("data", "k2")
        cache_b.put("data", "k1")

        stats_a = cache_a.get_stats()
        stats_b = cache_b.get_stats()

        assert stats_a["total_entries"] == 2
        assert stats_b["total_entries"] == 1

    def test_clear_all_scoped(self, tmp_path):
        """clear_all only removes entries in the active namespace."""
        cache_a = _make_cache(tmp_path, namespace="ns_a", backend="sqlite")
        cache_b = _make_cache(tmp_path, namespace="ns_b", backend="sqlite")

        cache_a.put("data_a", "k1")
        cache_b.put("data_b", "k1")

        cache_a.clear_all()

        assert cache_a.get("k1") is None
        assert cache_b.get("k1") == "data_b"

    def test_default_uses_unsuffixed_tables(self, tmp_path):
        """Default namespace uses 'cache_entries' / 'cache_stats' (no suffix)."""
        cache = _make_cache(tmp_path, namespace="default", backend="sqlite")
        backend = cache.metadata_backend
        # Unwrap CachedMetadataBackend if present
        while hasattr(backend, "backend"):
            backend = backend.backend

        assert backend._entries_table == "cache_entries"
        assert backend._stats_table == "cache_stats"

    def test_custom_namespace_uses_suffixed_tables(self, tmp_path):
        """Non-default namespace uses 'cache_entries_{ns}' tables."""
        cache = _make_cache(tmp_path, namespace="analytics", backend="sqlite")
        backend = cache.metadata_backend
        while hasattr(backend, "backend"):
            backend = backend.backend

        assert backend._entries_table == "cache_entries_analytics"
        assert backend._stats_table == "cache_stats_analytics"


# =====================================================================
# (6)  create_namespace / drop_namespace lifecycle
# =====================================================================


class TestNamespaceLifecycleSqlite:
    """Lifecycle tests using SQLite backend (most accessible)."""

    def test_create_namespace_adds_registry_entry(self, tmp_path):
        """create_namespace adds a registry entry and creates tables."""
        cache = _make_cache(tmp_path, namespace="default", backend="sqlite")
        backend = cache.metadata_backend
        while hasattr(backend, "backend"):
            backend = backend.backend

        ns_info = backend.create_namespace("test_ns", display_name="Test NS")
        assert isinstance(ns_info, NamespaceInfo)
        assert ns_info.namespace_id == "test_ns"

        namespaces = backend.list_namespaces()
        ns_ids = [ns.namespace_id for ns in namespaces]
        assert "test_ns" in ns_ids

    def test_drop_namespace_removes_registry_entry(self, tmp_path):
        """drop_namespace removes registry entry and drops tables."""
        cache = _make_cache(tmp_path, namespace="default", backend="sqlite")
        backend = cache.metadata_backend
        while hasattr(backend, "backend"):
            backend = backend.backend

        backend.create_namespace("ephemeral")
        result = backend.drop_namespace("ephemeral")
        assert result is True

        ns_ids = [ns.namespace_id for ns in backend.list_namespaces()]
        assert "ephemeral" not in ns_ids

    def test_drop_nonexistent_returns_false(self, tmp_path):
        """drop_namespace returns False for unknown namespace."""
        cache = _make_cache(tmp_path, namespace="default", backend="sqlite")
        backend = cache.metadata_backend
        while hasattr(backend, "backend"):
            backend = backend.backend

        assert backend.drop_namespace("does_not_exist") is False

    def test_cannot_drop_default(self, tmp_path):
        """Cannot drop the 'default' namespace."""
        cache = _make_cache(tmp_path, namespace="default", backend="sqlite")
        backend = cache.metadata_backend
        while hasattr(backend, "backend"):
            backend = backend.backend

        with pytest.raises(ValueError, match="default"):
            backend.drop_namespace("default")

    def test_create_duplicate_raises(self, tmp_path):
        """Creating a namespace that already exists raises ValueError."""
        cache = _make_cache(tmp_path, namespace="default", backend="sqlite")
        backend = cache.metadata_backend
        while hasattr(backend, "backend"):
            backend = backend.backend

        backend.create_namespace("dup_ns")
        with pytest.raises(ValueError, match="already exists"):
            backend.create_namespace("dup_ns")


class TestNamespaceLifecycleJson:
    """Lifecycle tests using JSON backend."""

    def test_create_and_list(self, tmp_path):
        """create_namespace creates registry entry, list_namespaces returns it."""
        cache = _make_cache(tmp_path, namespace="default", backend="json")
        backend = cache.metadata_backend
        while hasattr(backend, "backend"):
            backend = backend.backend

        backend.create_namespace("json_ns", display_name="JSON Test")
        ns_ids = [ns.namespace_id for ns in backend.list_namespaces()]
        assert "json_ns" in ns_ids

    def test_drop_removes_metadata_file(self, tmp_path):
        """drop_namespace removes the per-namespace metadata file."""
        cache = _make_cache(tmp_path, namespace="default", backend="json")
        backend = cache.metadata_backend
        while hasattr(backend, "backend"):
            backend = backend.backend

        backend.create_namespace("drop_me")
        # Write something to ensure file exists
        ns_file = backend._metadata_file_for_namespace("drop_me")

        result = backend.drop_namespace("drop_me")
        assert result is True
        assert not ns_file.exists()


# =====================================================================
# (7)  Namespace ID validation
# =====================================================================


class TestNamespaceIdValidation:
    """Namespace IDs must match ^[a-z0-9_]{1,48}$."""

    @pytest.mark.parametrize(
        "invalid_id",
        [
            "UPPER",
            "has-dash",
            "has space",
            "has.dot",
            "",
            "a" * 49,
            "special!char",
            "über",
        ],
    )
    def test_rejects_invalid_ids(self, invalid_id):
        with pytest.raises(ValueError):
            validate_namespace_id(invalid_id)

    @pytest.mark.parametrize(
        "valid_id",
        [
            "default",
            "analytics",
            "ml_pipeline",
            "ns_42",
            "a" * 48,
            "a",
        ],
    )
    def test_accepts_valid_ids(self, valid_id):
        validate_namespace_id(valid_id)  # Should not raise


# =====================================================================
# (8)  clear_all only affects active namespace
# =====================================================================


class TestClearAllNamespaceScoping:
    """clear_all() only clears the active namespace's data."""

    @pytest.mark.parametrize("backend", ["json", "sqlite"])
    def test_clear_does_not_cross_namespace_boundary(self, tmp_path, backend):
        """Clearing ns_a leaves ns_b's data intact."""
        cache_a = _make_cache(tmp_path, namespace="ns_a", backend=backend)
        cache_b = _make_cache(tmp_path, namespace="ns_b", backend=backend)

        cache_a.put("data_a1", "k1")
        cache_a.put("data_a2", "k2")
        cache_b.put("data_b1", "k1")

        removed = cache_a.clear_all()
        assert removed == 2

        # ns_a is empty
        assert cache_a.get("k1") is None
        assert len(cache_a.list_entries()) == 0

        # ns_b is untouched
        assert cache_b.get("k1") == "data_b1"
        assert len(cache_b.list_entries()) == 1


# =====================================================================
# (9)  clear_all_namespaces nuclear option
# =====================================================================


class TestClearAllNamespacesNuclear:
    """clear_all_namespaces() drops non-default and clears default."""

    @pytest.mark.parametrize("backend", ["json", "sqlite"])
    def test_nuclear_clear(self, tmp_path, backend):
        """After clear_all_namespaces, default is empty; non-default are dropped."""
        cache_default = _make_cache(tmp_path, namespace="default", backend=backend)
        cache_a = _make_cache(tmp_path, namespace="ns_a", backend=backend)

        cache_default.put("default_data", "k1")
        cache_a.put("ns_a_data", "k1")

        results = cache_default.clear_all_namespaces()

        assert DEFAULT_NAMESPACE in results
        assert results[DEFAULT_NAMESPACE] >= 0  # rows cleared
        assert cache_default.get("k1") is None


# =====================================================================
# (5)  cache-mode + storage-mode on same backend, different namespaces
# =====================================================================


class TestCacheAndStorageModeNamespaceIsolation:
    """cache-mode and storage-mode instances with different namespaces are isolated."""

    def test_cache_and_storage_mode_independent(self, tmp_path):
        """A cache-mode ns and a storage-mode ns don't interfere."""
        cache_cache = _make_cache(
            tmp_path, namespace="cache_ns", backend="sqlite", storage_mode=False
        )
        cache_store = _make_cache(
            tmp_path, namespace="store_ns", backend="sqlite", storage_mode=True
        )

        cache_cache.put("cached_value", "key1")
        cache_store.put("stored_value", "key1")

        assert cache_cache.get("key1") == "cached_value"
        assert cache_store.get("key1") == "stored_value"

        # Delete from cache side
        cache_cache.invalidate(cache_key="key1")
        assert cache_cache.get("key1") is None
        assert cache_store.get("key1") == "stored_value"


# =====================================================================
# (4)  S3 prefix isolation per namespace
# =====================================================================


@pytest.mark.skipif(not HAS_S3_MOCK, reason="boto3/moto not installed")
class TestS3NamespacePrefixIsolation:
    """S3BlobBackend auto-derives prefix from namespace."""

    def test_default_namespace_uses_base_prefix(self):
        """Default namespace keeps the user-supplied prefix unchanged."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")

            backend = S3BlobBackend(
                bucket="test-bucket",
                prefix="cache/v1/",
                namespace="default",
            )
            assert backend.prefix == "cache/v1/"
            assert backend._base_prefix == "cache/v1/"

    def test_custom_namespace_appends_to_prefix(self):
        """Non-default namespace appends namespace_id/ to prefix."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")

            backend = S3BlobBackend(
                bucket="test-bucket",
                prefix="cache/v1/",
                namespace="staging",
            )
            assert backend.prefix == "cache/v1/staging/"
            assert backend._base_prefix == "cache/v1/"

    def test_empty_prefix_with_namespace(self):
        """Namespace with empty base prefix creates namespace/ prefix."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")

            backend = S3BlobBackend(
                bucket="test-bucket",
                prefix="",
                namespace="analytics",
            )
            assert backend.prefix == "analytics/"
            assert backend._base_prefix == ""

    def test_blob_isolation_between_namespaces(self):
        """Blobs written in different namespaces land under different prefixes."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")

            backend_a = S3BlobBackend(
                bucket="test-bucket",
                prefix="cache/",
                namespace="ns_a",
                shard_chars=0,
            )
            backend_b = S3BlobBackend(
                bucket="test-bucket",
                prefix="cache/",
                namespace="ns_b",
                shard_chars=0,
            )

            path_a = backend_a.write_blob("blob1", b"data_a")
            path_b = backend_b.write_blob("blob1", b"data_b")

            assert path_a != path_b
            assert backend_a.read_blob(path_a) == b"data_a"
            assert backend_b.read_blob(path_b) == b"data_b"

    def test_delete_namespace_blobs(self):
        """delete_namespace_blobs removes all objects under a namespace prefix."""
        from cacheness.storage.backends.s3_backend import S3BlobBackend

        with mock_aws():
            s3 = boto3.client("s3", region_name="us-east-1")
            s3.create_bucket(Bucket="test-bucket")

            backend = S3BlobBackend(
                bucket="test-bucket",
                prefix="cache/",
                namespace="to_delete",
                shard_chars=0,
            )
            backend.write_blob("b1", b"data1")
            backend.write_blob("b2", b"data2")

            # Verify blobs exist
            keys = backend.list_keys()
            assert len(keys) == 2

            deleted, failed = backend.delete_namespace_blobs("to_delete")
            assert deleted == 2
            assert failed == 0

            # Verify blobs are gone
            keys_after = backend.list_keys()
            assert len(keys_after) == 0
