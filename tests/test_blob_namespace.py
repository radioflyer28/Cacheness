"""Tests for filesystem blob namespace isolation (CACHE-adi.14).

Verifies that:
- Default namespace stores blobs directly under base_dir (backward compat)
- Non-default namespaces store blobs under base_dir/{namespace}/
- Blobs are physically isolated between namespaces
- BlobStore passes namespace through to FilesystemBlobBackend
- UnifiedCache wires namespace to BlobStore
"""

import pytest
from pathlib import Path

from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache
from cacheness.metadata import DEFAULT_NAMESPACE
from cacheness.storage.backends.blob_backends import FilesystemBlobBackend
from cacheness.storage.blob_store import BlobStore


# =============================================================================
# FilesystemBlobBackend namespace tests
# =============================================================================


class TestFilesystemBlobBackendNamespace:
    """Test FilesystemBlobBackend namespace isolation."""

    def test_default_namespace_uses_base_dir(self, tmp_path):
        """Default namespace stores blobs directly under base_dir."""
        backend = FilesystemBlobBackend(tmp_path / "blobs", shard_chars=0)
        assert backend.base_dir == tmp_path / "blobs"
        assert backend._namespace == "default"

    def test_default_namespace_explicit(self, tmp_path):
        """Explicitly passing 'default' behaves same as omitting."""
        backend = FilesystemBlobBackend(
            tmp_path / "blobs", shard_chars=0, namespace="default"
        )
        assert backend.base_dir == tmp_path / "blobs"
        assert backend._namespace == "default"

    def test_custom_namespace_uses_subdirectory(self, tmp_path):
        """Non-default namespace stores blobs under base_dir/{namespace}/."""
        backend = FilesystemBlobBackend(
            tmp_path / "blobs", shard_chars=0, namespace="analytics"
        )
        assert backend.base_dir == tmp_path / "blobs" / "analytics"
        assert backend._namespace == "analytics"
        assert backend.base_dir.exists()

    def test_namespace_subdirectory_created(self, tmp_path):
        """Namespace subdirectory is created on init."""
        backend = FilesystemBlobBackend(
            tmp_path / "blobs", shard_chars=0, namespace="ml_models"
        )
        assert (tmp_path / "blobs" / "ml_models").is_dir()

    def test_blob_write_read_default_namespace(self, tmp_path):
        """Write and read a blob in the default namespace."""
        backend = FilesystemBlobBackend(tmp_path / "blobs", shard_chars=0)
        path = backend.write_blob("test_blob", b"hello default")
        assert Path(path).parent == tmp_path / "blobs"
        assert backend.read_blob(path) == b"hello default"

    def test_blob_write_read_custom_namespace(self, tmp_path):
        """Write and read a blob in a custom namespace."""
        backend = FilesystemBlobBackend(
            tmp_path / "blobs", shard_chars=0, namespace="staging"
        )
        path = backend.write_blob("test_blob", b"hello staging")
        assert Path(path).parent == tmp_path / "blobs" / "staging"
        assert backend.read_blob(path) == b"hello staging"

    def test_blob_isolation_between_namespaces(self, tmp_path):
        """Blobs with same ID in different namespaces are independent."""
        default_backend = FilesystemBlobBackend(
            tmp_path / "blobs", shard_chars=0, namespace="default"
        )
        custom_backend = FilesystemBlobBackend(
            tmp_path / "blobs", shard_chars=0, namespace="analytics"
        )

        default_path = default_backend.write_blob("shared_id", b"default_data")
        custom_path = custom_backend.write_blob("shared_id", b"analytics_data")

        # Paths are different
        assert default_path != custom_path

        # Data is isolated
        assert default_backend.read_blob(default_path) == b"default_data"
        assert custom_backend.read_blob(custom_path) == b"analytics_data"

    def test_delete_in_one_namespace_doesnt_affect_other(self, tmp_path):
        """Deleting a blob in one namespace leaves the other intact."""
        ns_a = FilesystemBlobBackend(
            tmp_path / "blobs", shard_chars=0, namespace="ns_a"
        )
        ns_b = FilesystemBlobBackend(
            tmp_path / "blobs", shard_chars=0, namespace="ns_b"
        )

        path_a = ns_a.write_blob("blob1", b"data_a")
        path_b = ns_b.write_blob("blob1", b"data_b")

        ns_a.delete_blob(path_a)

        assert not ns_a.exists(path_a)
        assert ns_b.exists(path_b)
        assert ns_b.read_blob(path_b) == b"data_b"

    def test_sharding_with_namespace(self, tmp_path):
        """Git-style sharding works correctly within namespace subdirectory."""
        backend = FilesystemBlobBackend(
            tmp_path / "blobs", shard_chars=2, namespace="sharded_ns"
        )
        path = backend.write_blob("abcdef123456", b"sharded data")
        blob_path = Path(path)
        # Should be base_dir/ab/abcdef123456  where base_dir includes namespace
        assert blob_path.parent.name == "ab"
        assert blob_path.parent.parent == tmp_path / "blobs" / "sharded_ns"

    def test_multiple_namespaces_coexist(self, tmp_path):
        """Multiple namespaces can coexist under the same base_dir."""
        namespaces = ["alpha", "beta", "gamma"]
        backends = {}
        paths = {}

        for ns in namespaces:
            backends[ns] = FilesystemBlobBackend(
                tmp_path / "blobs", shard_chars=0, namespace=ns
            )
            paths[ns] = backends[ns].write_blob("blob1", f"data_{ns}".encode())

        # All directories exist
        for ns in namespaces:
            assert (tmp_path / "blobs" / ns).is_dir()

        # All data is correct
        for ns in namespaces:
            assert backends[ns].read_blob(paths[ns]) == f"data_{ns}".encode()


# =============================================================================
# BlobStore namespace plumbing tests
# =============================================================================


class TestBlobStoreNamespacePlumbing:
    """Test that BlobStore passes namespace to FilesystemBlobBackend."""

    def test_blobstore_default_namespace(self, tmp_path):
        """BlobStore with default namespace uses base_dir directly."""
        store = BlobStore(cache_dir=tmp_path / "store")
        assert store._namespace == "default"
        assert store.blob_backend._namespace == "default"
        assert store.blob_backend.base_dir == tmp_path / "store"

    def test_blobstore_custom_namespace(self, tmp_path):
        """BlobStore with custom namespace creates namespaced blob backend."""
        store = BlobStore(cache_dir=tmp_path / "store", namespace="analytics")
        assert store._namespace == "analytics"
        assert store.blob_backend._namespace == "analytics"
        assert store.blob_backend.base_dir == tmp_path / "store" / "analytics"

    def test_blobstore_namespace_isolation(self, tmp_path):
        """Two BlobStores with different namespaces store blobs independently."""
        store_a = BlobStore(cache_dir=tmp_path / "store", namespace="ns_a")
        store_b = BlobStore(cache_dir=tmp_path / "store", namespace="ns_b")

        # Write same blob_id through both blob_backends
        path_a = store_a.blob_backend.write_blob("blob1", b"data_a")
        path_b = store_b.blob_backend.write_blob("blob1", b"data_b")

        assert path_a != path_b
        assert store_a.blob_backend.read_blob(path_a) == b"data_a"
        assert store_b.blob_backend.read_blob(path_b) == b"data_b"

    def test_blobstore_memory_backend_ignores_namespace(self, tmp_path):
        """In-memory blob backend doesn't use namespace (no filesystem)."""
        store = BlobStore(
            cache_dir=tmp_path / "store",
            blob_backend="memory",
            namespace="custom_ns",
        )
        assert store._namespace == "custom_ns"
        # InMemoryBlobBackend doesn't have _namespace attribute
        assert not hasattr(store.blob_backend, "_namespace")


# =============================================================================
# UnifiedCache → BlobStore namespace wiring tests
# =============================================================================


class TestUnifiedCacheBlobNamespaceWiring:
    """Test that UnifiedCache wires namespace to BlobStore."""

    def test_default_namespace_blob_path(self, tmp_path):
        """UnifiedCache with default namespace stores blobs in cache_dir."""
        config = CacheConfig(cache_dir=str(tmp_path / "cache"))
        cache = UnifiedCache(config=config)
        assert cache._blob_store._namespace == DEFAULT_NAMESPACE
        assert cache._blob_store.blob_backend.base_dir == tmp_path / "cache"

    def test_custom_namespace_blob_path(self, tmp_path):
        """UnifiedCache with custom namespace stores blobs in subdirectory."""
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            namespace="ml_pipeline",
        )
        cache = UnifiedCache(config=config)
        assert cache._blob_store._namespace == "ml_pipeline"
        assert (
            cache._blob_store.blob_backend.base_dir
            == tmp_path / "cache" / "ml_pipeline"
        )

    def test_namespace_matches_between_cache_and_blobstore(self, tmp_path):
        """UnifiedCache namespace propagates consistently to BlobStore."""
        ns = "experiment_42"
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            namespace=ns,
        )
        cache = UnifiedCache(config=config)
        assert cache.namespace == ns
        assert cache._blob_store._namespace == ns
        assert cache._blob_store.blob_backend._namespace == ns

    def test_two_caches_different_namespaces_isolated(self, tmp_path):
        """Two UnifiedCache instances with different namespaces have isolated blobs."""
        config_a = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            namespace="ns_alpha",
        )
        config_b = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            namespace="ns_beta",
        )
        cache_a = UnifiedCache(config=config_a)
        cache_b = UnifiedCache(config=config_b)

        blob_dir_a = cache_a._blob_store.blob_backend.base_dir
        blob_dir_b = cache_b._blob_store.blob_backend.base_dir

        assert blob_dir_a != blob_dir_b
        assert blob_dir_a == tmp_path / "cache" / "ns_alpha"
        assert blob_dir_b == tmp_path / "cache" / "ns_beta"
