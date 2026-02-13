"""Tests for namespace plumbing: UnifiedCache → factory → backend init (CACHE-adi.13).

Verifies that namespace flows from CacheConfig through create_metadata_backend
to each concrete backend's _active_namespace attribute.
"""

import pytest
from pathlib import Path

from cacheness.config import CacheConfig
from cacheness.core import UnifiedCache
from cacheness.metadata import (
    DEFAULT_NAMESPACE,
    MetadataBackend,
    JsonBackend,
    SqliteBackend,
    create_metadata_backend,
)


class TestFactoryNamespacePlumbing:
    """Test that create_metadata_backend passes namespace to backends."""

    def test_json_default_namespace(self, tmp_path):
        """JSON backend gets DEFAULT_NAMESPACE when none specified."""
        backend = create_metadata_backend("json", metadata_file=tmp_path / "meta.json")
        assert backend.active_namespace == DEFAULT_NAMESPACE
        assert backend._active_namespace == DEFAULT_NAMESPACE

    def test_json_custom_namespace(self, tmp_path):
        """JSON backend receives custom namespace from factory."""
        backend = create_metadata_backend(
            "json", metadata_file=tmp_path / "meta.json", namespace="analytics"
        )
        assert backend.active_namespace == "analytics"
        assert backend._active_namespace == "analytics"

    def test_sqlite_default_namespace(self, tmp_path):
        """SQLite backend gets DEFAULT_NAMESPACE when none specified."""
        backend = create_metadata_backend("sqlite", db_file=str(tmp_path / "meta.db"))
        assert backend.active_namespace == DEFAULT_NAMESPACE
        assert backend._active_namespace == DEFAULT_NAMESPACE
        backend.close()

    def test_sqlite_custom_namespace(self, tmp_path):
        """SQLite backend receives custom namespace from factory."""
        backend = create_metadata_backend(
            "sqlite", db_file=str(tmp_path / "meta.db"), namespace="ml_models"
        )
        assert backend.active_namespace == "ml_models"
        assert backend._active_namespace == "ml_models"
        backend.close()

    def test_sqlite_memory_custom_namespace(self):
        """In-memory SQLite backend receives custom namespace."""
        backend = create_metadata_backend("sqlite_memory", namespace="ephemeral")
        assert backend.active_namespace == "ephemeral"
        backend.close()

    def test_auto_backend_default_namespace(self, tmp_path):
        """Auto-selected backend gets DEFAULT_NAMESPACE."""
        backend = create_metadata_backend("auto", db_file=str(tmp_path / "auto.db"))
        assert backend.active_namespace == DEFAULT_NAMESPACE
        backend.close()

    def test_auto_backend_custom_namespace(self, tmp_path):
        """Auto-selected backend receives custom namespace."""
        backend = create_metadata_backend(
            "auto", db_file=str(tmp_path / "auto.db"), namespace="team_data"
        )
        assert backend.active_namespace == "team_data"
        backend.close()

    def test_invalid_namespace_rejected(self, tmp_path):
        """Factory rejects invalid namespace IDs."""
        with pytest.raises(ValueError, match="Invalid namespace ID"):
            create_metadata_backend(
                "json",
                metadata_file=tmp_path / "meta.json",
                namespace="BAD-NS!",
            )

    def test_invalid_namespace_rejected_sqlite(self, tmp_path):
        """Factory rejects invalid namespace IDs for SQLite."""
        with pytest.raises(ValueError, match="Invalid namespace ID"):
            create_metadata_backend(
                "sqlite",
                db_file=str(tmp_path / "meta.db"),
                namespace="",
            )


class TestCachedBackendNamespaceProxy:
    """Test that CachedMetadataBackend proxies namespace from wrapped backend."""

    def test_cached_wrapper_proxies_namespace(self, tmp_path):
        """CachedMetadataBackend should expose wrapped backend's namespace."""
        from cacheness.config import CacheMetadataConfig

        config = CacheMetadataConfig(enable_memory_cache=True)
        backend = create_metadata_backend(
            "sqlite",
            db_file=str(tmp_path / "meta.db"),
            namespace="cached_ns",
            config=config,
        )
        # If memory cache is enabled, we get a CachedMetadataBackend wrapper
        assert backend.active_namespace == "cached_ns"
        backend.close()

    def test_cached_wrapper_delegates_list_namespaces(self, tmp_path):
        """CachedMetadataBackend should delegate namespace registry methods."""
        from cacheness.config import CacheMetadataConfig

        config = CacheMetadataConfig(enable_memory_cache=True)
        backend = create_metadata_backend(
            "sqlite",
            db_file=str(tmp_path / "meta.db"),
            config=config,
        )
        namespaces = backend.list_namespaces()
        # Should delegate to wrapped backend, which has 'default' namespace
        ns_ids = [ns.namespace_id for ns in namespaces]
        assert DEFAULT_NAMESPACE in ns_ids
        backend.close()


class TestUnifiedCacheNamespacePlumbing:
    """Test namespace flows from CacheConfig through UnifiedCache to backend."""

    def test_backend_receives_default_namespace(self, tmp_path):
        """Backend should have DEFAULT_NAMESPACE when no namespace configured."""
        config = CacheConfig(cache_dir=str(tmp_path / "cache"))
        cache = UnifiedCache(config=config)
        assert cache.metadata_backend.active_namespace == DEFAULT_NAMESPACE
        cache.close()

    def test_backend_receives_custom_namespace(self, tmp_path):
        """Backend should receive custom namespace from CacheConfig."""
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            namespace="my_project",
        )
        cache = UnifiedCache(config=config)
        assert cache.namespace == "my_project"
        assert cache.metadata_backend.active_namespace == "my_project"
        cache.close()

    def test_namespace_consistent_across_stack(self, tmp_path):
        """config.namespace == cache.namespace == backend.active_namespace."""
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            namespace="full_stack",
        )
        cache = UnifiedCache(config=config)
        assert config.namespace == "full_stack"
        assert cache.namespace == "full_stack"
        assert cache.metadata_backend.active_namespace == "full_stack"
        cache.close()

    def test_json_backend_receives_namespace(self, tmp_path):
        """JSON backend receives namespace when explicitly configured."""
        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            metadata_backend="json",
            namespace="json_ns",
        )
        cache = UnifiedCache(config=config)
        assert cache.metadata_backend.active_namespace == "json_ns"
        cache.close()


class TestMetadataBackendABCProperty:
    """Test the active_namespace property on MetadataBackend ABC."""

    def test_property_returns_default_when_not_set(self):
        """A backend without _active_namespace returns DEFAULT_NAMESPACE."""

        class MinimalBackend(MetadataBackend):
            def load_metadata(self):
                return {}

            def save_metadata(self, metadata):
                pass

            def get_entry(self, cache_key):
                return None

            def put_entry(self, cache_key, entry_data):
                pass

            def remove_entry(self, cache_key):
                return False

            def list_entries(self, prefix=None, data_type=None):
                return []

            def get_stats(self):
                return {}

            def increment_hits(self):
                pass

            def increment_misses(self):
                pass

            def cleanup_expired(self, ttl_seconds):
                return 0

            def cleanup_by_size(self, target_size_mb):
                return {}

            def clear_all(self):
                pass

            def update_access_time(self, cache_key):
                pass

            def update_entry_metadata(self, cache_key, updates):
                return False

            def close(self):
                pass

        backend = MinimalBackend()
        assert backend.active_namespace == DEFAULT_NAMESPACE

    def test_property_returns_set_value(self):
        """A backend with _active_namespace set returns that value."""

        class NamespacedBackend(MetadataBackend):
            def __init__(self):
                self._active_namespace = "custom"

            def load_metadata(self):
                return {}

            def save_metadata(self, metadata):
                pass

            def get_entry(self, cache_key):
                return None

            def put_entry(self, cache_key, entry_data):
                pass

            def remove_entry(self, cache_key):
                return False

            def list_entries(self, prefix=None, data_type=None):
                return []

            def get_stats(self):
                return {}

            def increment_hits(self):
                pass

            def increment_misses(self):
                pass

            def cleanup_expired(self, ttl_seconds):
                return 0

            def cleanup_by_size(self, target_size_mb):
                return {}

            def clear_all(self):
                pass

            def update_access_time(self, cache_key):
                pass

            def update_entry_metadata(self, cache_key, updates):
                return False

            def close(self):
                pass

        backend = NamespacedBackend()
        assert backend.active_namespace == "custom"
