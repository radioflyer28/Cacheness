#!/usr/bin/env python3
"""
Test for store_cache_key_params configuration option
"""

import tempfile
from pathlib import Path
from cacheness.core import UnifiedCache
from cacheness.config import CacheConfig


class TestStoreCacheKeyParamsConfig:
    """Test the store_cache_key_params configuration option."""

    def test_default_config_doesnt_store_params(self):
        """Test that default configuration does NOT store cache_key_params for performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache_default"
            config = CacheConfig(cache_dir=str(cache_dir), metadata_backend="sqlite")
            cache = UnifiedCache(config)

            # Verify default is False for performance
            assert config.metadata.store_cache_key_params is False

            # Store data with parameters
            test_params = {"model": "gpt-4", "temperature": 0.7}
            cache.put("test data", description="Test", **test_params)

            # Check metadata does NOT include cache_key_params (default behavior)
            entries = cache.list_entries()
            assert len(entries) == 1
            assert "cache_key_params" not in entries[0]["metadata"]

    def test_explicit_enable_stores_params(self):
        """Test that explicitly enabling storage works."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache_enabled"
            config = CacheConfig(
                cache_dir=str(cache_dir),
                metadata_backend="sqlite",
                store_cache_key_params=True,
            )
            cache = UnifiedCache(config)

            # Verify configuration
            assert config.metadata.store_cache_key_params is True

            # Store data with parameters
            test_params = {"model": "llama", "temperature": 0.3}
            cache.put("test data", description="Test", **test_params)

            # Check metadata includes cache_key_params
            entries = cache.list_entries()
            assert len(entries) == 1
            assert "cache_key_params" in entries[0]["metadata"]
            assert entries[0]["metadata"]["cache_key_params"]["model"] == "str:llama"

    def test_disable_doesnt_store_params(self):
        """Test that disabling storage prevents cache_key_params from being stored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache_disabled"
            config = CacheConfig(
                cache_dir=str(cache_dir),
                metadata_backend="sqlite",
                store_cache_key_params=False,
            )
            cache = UnifiedCache(config)

            # Verify configuration
            assert config.metadata.store_cache_key_params is False

            # Store data with parameters
            test_params = {"model": "claude", "temperature": 0.5}
            cache.put("test data", description="Test", **test_params)

            # Check metadata does NOT include cache_key_params
            entries = cache.list_entries()
            assert len(entries) == 1
            assert "cache_key_params" not in entries[0]["metadata"]

    def test_json_backend_respects_config(self):
        """Test that JSON backend also respects the configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test enabled
            cache_dir1 = Path(temp_dir) / "json_enabled"
            config1 = CacheConfig(
                cache_dir=str(cache_dir1),
                metadata_backend="json",
                store_cache_key_params=True,
            )
            cache1 = UnifiedCache(config1)

            test_params = {"model": "gpt-3.5", "temperature": 0.2}
            cache1.put("test data", description="Test", **test_params)

            entries1 = cache1.list_entries()
            assert "cache_key_params" in entries1[0]["metadata"]

            # Test disabled
            cache_dir2 = Path(temp_dir) / "json_disabled"
            config2 = CacheConfig(
                cache_dir=str(cache_dir2),
                metadata_backend="json",
                store_cache_key_params=False,
            )
            cache2 = UnifiedCache(config2)

            cache2.put("test data", description="Test", **test_params)

            entries2 = cache2.list_entries()
            assert "cache_key_params" not in entries2[0]["metadata"]

    def test_sub_config_access(self):
        """Test that the configuration can be accessed through sub-configuration objects."""
        from cacheness.config import CacheMetadataConfig

        # Test through sub-configuration
        metadata_config = CacheMetadataConfig(store_cache_key_params=False)
        config = CacheConfig(metadata=metadata_config)

        assert config.metadata.store_cache_key_params is False

        # Test backwards compatibility parameter
        config2 = CacheConfig(store_cache_key_params=False)
        assert config2.metadata.store_cache_key_params is False

    def test_cache_functionality_unchanged(self):
        """Test that disabling cache_key_params storage doesn't affect core cache functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "functionality_test"
            config = CacheConfig(
                cache_dir=str(cache_dir),
                metadata_backend="sqlite",
                store_cache_key_params=False,
            )
            cache = UnifiedCache(config)

            # Store and retrieve data
            test_data = {"key": "value", "number": 42}
            test_params = {"experiment": "test", "version": "1.0"}

            cache.put(test_data, description="Test data", **test_params)
            retrieved = cache.get(**test_params)

            # Core functionality should work the same
            assert retrieved == test_data

            # But metadata shouldn't include cache_key_params
            entries = cache.list_entries()
            assert "cache_key_params" not in entries[0]["metadata"]

    def test_complex_params_handling(self):
        """Test that complex parameters are handled correctly when storage is enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "complex_params"
            config = CacheConfig(
                cache_dir=str(cache_dir),
                metadata_backend="sqlite",
                store_cache_key_params=True,
            )
            cache = UnifiedCache(config)

            # Use complex parameters that require serialization
            model_path = Path("/tmp/model.pkl")

            test_params = {
                "model_path": model_path,
                "config": {"lr": 0.001, "epochs": 100},
                "temperature": 0.7,
            }

            cache.put("test data", description="Complex params test", **test_params)
            entries = cache.list_entries()

            # Check that complex parameters are properly serialized
            stored_params = entries[0]["metadata"]["cache_key_params"]
            assert "model_path" in stored_params
            assert "config" in stored_params
            assert "temperature" in stored_params

            # Verify Path object is serialized properly
            assert (
                "path_missing:" in stored_params["model_path"]
                or "Path:" in stored_params["model_path"]
            )
