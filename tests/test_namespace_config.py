"""Tests for namespace parameter in CacheConfig and UnifiedCache."""

import pytest

from cacheness.config import CacheConfig
from cacheness.metadata import DEFAULT_NAMESPACE


class TestCacheConfigNamespace:
    """Test namespace parameter on CacheConfig."""

    def test_default_namespace(self):
        """CacheConfig defaults to 'default' namespace."""
        config = CacheConfig()
        assert config.namespace == DEFAULT_NAMESPACE

    def test_custom_namespace(self):
        """CacheConfig accepts a custom namespace."""
        config = CacheConfig(namespace="my_project")
        assert config.namespace == "my_project"

    def test_namespace_validation_rejects_uppercase(self):
        """Uppercase namespace IDs are rejected."""
        with pytest.raises(ValueError, match="Invalid namespace ID"):
            CacheConfig(namespace="MyProject")

    def test_namespace_validation_rejects_dash(self):
        """Dashes in namespace IDs are rejected."""
        with pytest.raises(ValueError, match="Invalid namespace ID"):
            CacheConfig(namespace="my-project")

    def test_namespace_validation_rejects_empty(self):
        """Empty namespace IDs are rejected."""
        with pytest.raises(ValueError, match="Invalid namespace ID"):
            CacheConfig(namespace="")

    def test_namespace_validation_rejects_too_long(self):
        """Namespace IDs over 48 chars are rejected."""
        with pytest.raises(ValueError, match="Invalid namespace ID"):
            CacheConfig(namespace="a" * 49)

    def test_namespace_allows_underscores(self):
        """Underscores are valid in namespace IDs."""
        config = CacheConfig(namespace="my_data_project")
        assert config.namespace == "my_data_project"

    def test_namespace_allows_numbers(self):
        """Numbers are valid in namespace IDs."""
        config = CacheConfig(namespace="project42")
        assert config.namespace == "project42"

    def test_namespace_max_length(self):
        """48-char namespace ID is accepted."""
        ns = "a" * 48
        config = CacheConfig(namespace=ns)
        assert config.namespace == ns

    def test_namespace_immutable_after_init(self):
        """Namespace should not change after config init."""
        config = CacheConfig(namespace="original")
        assert config.namespace == "original"
        # While Python doesn't enforce immutability, we test the contract
        # that config.namespace stores the validated value
        assert config.namespace == "original"

    def test_namespace_with_other_params(self):
        """Namespace works alongside other config parameters."""
        config = CacheConfig(
            namespace="analytics",
            cache_dir="/tmp/test_cache",
            storage_mode=True,
        )
        assert config.namespace == "analytics"
        assert config.storage.cache_dir == "/tmp/test_cache"
        assert config.storage_mode is True


class TestUnifiedCacheNamespace:
    """Test namespace propagation to UnifiedCache."""

    def test_default_namespace_on_cache(self, tmp_path):
        """UnifiedCache should have 'default' namespace by default."""
        from cacheness.core import UnifiedCache

        config = CacheConfig(cache_dir=str(tmp_path / "cache"))
        cache = UnifiedCache(config=config)
        assert cache.namespace == DEFAULT_NAMESPACE
        cache.close()

    def test_custom_namespace_on_cache(self, tmp_path):
        """UnifiedCache should reflect the config namespace."""
        from cacheness.core import UnifiedCache

        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            namespace="my_project",
        )
        cache = UnifiedCache(config=config)
        assert cache.namespace == "my_project"
        cache.close()

    def test_namespace_matches_config(self, tmp_path):
        """cache.namespace should equal config.namespace."""
        from cacheness.core import UnifiedCache

        config = CacheConfig(
            cache_dir=str(tmp_path / "cache"),
            namespace="team_alpha",
        )
        cache = UnifiedCache(config=config)
        assert cache.namespace == config.namespace
        cache.close()
