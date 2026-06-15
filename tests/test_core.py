"""
Unit tests for the core cache system.

Tests the main cacheness class and CacheConfig functionality.
"""

import pytest
import tempfile
import time
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from cacheness import CacheConfig, cacheness


class TestCacheConfig:
    """Test CacheConfig class functionality."""

    def test_unified_cache_root_export_preserves_alias(self):
        """Root package exports both UnifiedCache and the legacy alias."""
        from cacheness import UnifiedCache, cacheness

        assert UnifiedCache is cacheness
        assert "UnifiedCache" in __import__("cacheness").__all__
        assert "cacheness" in __import__("cacheness").__all__

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()

        assert config.storage.cache_dir == "./cache"
        assert config.metadata.default_ttl_seconds == 86400  # 24 hours in seconds
        assert config.storage.max_cache_size_mb == 2000
        assert config.compression.parquet_compression == "lz4"
        assert config.compression.npz_compression is True
        assert config.metadata.enable_metadata is True
        assert config.storage.cleanup_on_init is True
        assert config.metadata.metadata_backend == "auto"
        assert config.metadata.sqlite_db_file == "cache_metadata.db"
        assert config.compression.pickle_compression_codec == "zstd"
        assert config.compression.pickle_compression_level == 5
        assert config.compression.use_blosc2_arrays is True
        assert config.compression.blosc2_array_codec == "lz4"
        assert config.compression.blosc2_array_clevel == 5
        # store_cache_key_params was removed in Section 2.13 (unified metadata)

    def test_custom_config(self):
        """Test custom configuration values."""
        from cacheness.config import (
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )

        # Use a platform-agnostic path for testing
        import tempfile

        test_cache_dir = Path(tempfile.gettempdir()) / "test_cache"
        storage_config = CacheStorageConfig(
            cache_dir=str(test_cache_dir), max_cache_size_mb=1000, cleanup_on_init=False
        )
        metadata_config = CacheMetadataConfig(
            default_ttl_seconds=172800, enable_metadata=False, metadata_backend="json"
        )
        compression_config = CompressionConfig(
            parquet_compression="gzip", npz_compression=False
        )
        serialization_config = SerializationConfig()
        handler_config = HandlerConfig()

        config = CacheConfig(
            storage=storage_config,
            metadata=metadata_config,
            compression=compression_config,
            serialization=serialization_config,
            handlers=handler_config,
        )

        # Compare using as_posix() for cross-platform compatibility
        assert Path(config.storage.cache_dir).as_posix() == test_cache_dir.as_posix()
        assert config.metadata.default_ttl_seconds == 172800  # 48 hours in seconds
        assert config.storage.max_cache_size_mb == 1000
        assert config.compression.parquet_compression == "gzip"
        assert config.compression.npz_compression is False
        assert config.metadata.enable_metadata is False
        assert config.storage.cleanup_on_init is False
        assert config.metadata.metadata_backend == "json"
        # store_cache_key_params was removed in Section 2.13 - now using store_full_metadata


class TestCacheness:
    """Test cacheness class functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "test_cache"

    @pytest.fixture
    def test_config(self, temp_cache_dir):
        """Create a test configuration."""
        from cacheness.config import (
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )

        storage_config = CacheStorageConfig(cache_dir=str(temp_cache_dir))
        metadata_config = CacheMetadataConfig(
            metadata_backend="json"
        )  # Use JSON to avoid SQLite dependencies
        compression_config = CompressionConfig(
            use_blosc2_arrays=False
        )  # Disable blosc2 for testing
        serialization_config = SerializationConfig()
        handler_config = HandlerConfig()

        return CacheConfig(
            storage=storage_config,
            metadata=metadata_config,
            compression=compression_config,
            serialization=serialization_config,
            handlers=handler_config,
        )

    @pytest.fixture
    def cache(self, test_config):
        """Create a cache instance for testing."""
        return cacheness(test_config)

    def test_cache_initialization(self, cache, test_config):
        """Test cache initialization."""
        assert cache.config == test_config
        assert cache.handlers is not None
        assert cache.metadata_backend is not None
        assert cache.config.storage.cache_dir in str(cache.config.storage.cache_dir)

    def test_put_and_get_simple_object(self, cache):
        """Test caching simple objects."""
        test_data = {"key": "value", "number": 42}
        cache_key_params = {"test": "simple_object"}

        # Put data
        cache.put(test_data, description="Test object", **cache_key_params)

        # Get data
        retrieved = cache.get(**cache_key_params)

        assert retrieved == test_data

    def test_get_cache_hit_reads_metadata_entry_once(self, cache):
        """Cache hit expiry checks reuse the entry already fetched by get()."""
        test_data = {"key": "value", "number": 42}
        cache_key_params = {"test": "single_metadata_read"}

        cache.put(test_data, **cache_key_params)

        with patch.object(
            cache.metadata_backend,
            "get_entry",
            wraps=cache.metadata_backend.get_entry,
        ) as get_entry:
            retrieved = cache.get(**cache_key_params)

        assert retrieved == test_data
        assert get_entry.call_count == 1

    def test_put_and_get_numpy_array(self, cache):
        """Test caching numpy arrays."""
        test_array = np.random.rand(10, 5)
        cache_key_params = {"test": "numpy_array", "shape": "10x5"}

        # Put array
        cache.put(test_array, description="Test array", **cache_key_params)

        # Get array
        retrieved = cache.get(**cache_key_params)

        assert isinstance(retrieved, np.ndarray)
        assert retrieved.shape == test_array.shape
        assert np.allclose(retrieved, test_array)

    def test_put_and_get_array_dict(self, cache):
        """Test caching dictionary of arrays."""
        test_dict = {
            "array1": np.random.rand(5, 3),
            "array2": np.random.randint(0, 10, (4, 2)),
            "labels": np.array(["a", "b", "c"]),
        }
        cache_key_params = {"test": "array_dict"}

        # Put dictionary
        cache.put(test_dict, description="Test array dict", **cache_key_params)

        # Get dictionary
        retrieved = cache.get(**cache_key_params)

        assert isinstance(retrieved, dict)
        assert set(retrieved.keys()) == set(test_dict.keys())
        for key in test_dict.keys():
            assert np.array_equal(retrieved[key], test_dict[key])

    def _has_dataframe_library(self):
        """Check if polars or pandas is available."""
        import importlib.util

        return (
            importlib.util.find_spec("polars") is not None
            or importlib.util.find_spec("pandas") is not None
        )

    @pytest.mark.skipif(
        not _has_dataframe_library(None),
        reason="DataFrame tests require polars or pandas",
    )
    def test_put_and_get_dataframe(self, cache):
        """Test caching dataframes (requires polars or pandas)."""
        try:
            import polars as pl

            test_df = pl.DataFrame(
                {
                    "id": range(10),
                    "value": np.random.rand(10),
                    "category": ["A", "B"] * 5,
                }
            )
            data_type = "polars"
        except ImportError:
            try:
                import pandas as pd

                test_df = pd.DataFrame(
                    {
                        "id": range(10),
                        "value": np.random.rand(10),
                        "category": ["A", "B"] * 5,
                    }
                )
                data_type = "pandas"
            except ImportError:
                pytest.skip("Neither polars nor pandas available")

        cache_key_params = {"test": "dataframe", "type": data_type}

        # Put dataframe
        cache.put(test_df, description="Test dataframe", **cache_key_params)

        # Get dataframe
        retrieved = cache.get(**cache_key_params)

        assert isinstance(retrieved, type(test_df))
        assert retrieved.shape == test_df.shape

        if data_type == "polars":
            # For polars, compare values as numpy arrays
            assert np.array_equal(test_df.to_numpy(), retrieved.to_numpy())
        else:  # pandas
            # For pandas, use DataFrame.equals()
            import pandas as pd

            assert isinstance(test_df, pd.DataFrame)
            assert isinstance(retrieved, pd.DataFrame)
            assert test_df.equals(retrieved)

    def test_put_and_get_series(self, cache):
        """Test caching Series (requires polars or pandas)."""
        try:
            import polars as pl

            test_series = pl.Series("test_data", [1.1, 2.2, 3.3, 4.4, 5.5])
            data_type = "polars"
        except ImportError:
            try:
                import pandas as pd

                test_series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5], name="test_data")
                data_type = "pandas"
            except ImportError:
                pytest.skip("Neither polars nor pandas available")

        cache_key_params = {"test": "series", "type": data_type}

        # Put series
        cache.put(test_series, description="Test series", **cache_key_params)

        # Get series
        retrieved = cache.get(**cache_key_params)

        assert isinstance(retrieved, type(test_series))
        assert retrieved.shape == test_series.shape

        if data_type == "polars":
            # For polars, compare values
            assert retrieved.equals(test_series)
            assert retrieved.name == test_series.name
        else:  # pandas
            # For pandas, use Series.equals()
            import pandas as pd

            assert isinstance(test_series, pd.Series)
            assert isinstance(retrieved, pd.Series)
            assert test_series.equals(retrieved)

    def test_cache_miss(self, cache):
        """Test cache miss scenario."""
        result = cache.get(nonexistent="key")
        assert result is None

    def test_get_with_metadata_returns_both(self, cache):
        """Test that get_with_metadata() returns both data and metadata."""
        # Create a cache entry
        test_data = np.array([1, 2, 3, 4, 5])
        cache.put(test_data, experiment="test_123")

        # Retrieve with metadata
        result = cache.get_with_metadata(experiment="test_123")

        assert result is not None
        data, metadata = result

        # Verify data
        assert isinstance(data, np.ndarray)
        np.testing.assert_array_equal(data, test_data)

        # Verify metadata dict contains expected fields
        assert isinstance(metadata, dict)
        assert "cache_key" in metadata
        assert "data_type" in metadata
        assert "created_at" in metadata
        assert metadata["data_type"] == "array"

    def test_get_with_metadata_returns_none_for_missing(self, cache):
        """Test that get_with_metadata() returns None for missing entries."""
        result = cache.get_with_metadata(nonexistent="entry")
        assert result is None

    def test_get_with_metadata_single_lookup(self, cache):
        """Test that get_with_metadata() performs single metadata lookup."""
        # Create entry
        test_data = {"value": 42, "name": "test"}
        cache.put(test_data, single_lookup_test="data")

        # Get with metadata
        result = cache.get_with_metadata(single_lookup_test="data")
        assert result is not None

        data, metadata = result
        assert data == test_data

        # Verify metadata includes cache_key (proves single lookup)
        assert "cache_key" in metadata
        cache_key = cache._create_cache_key({"single_lookup_test": "data"})
        assert metadata["cache_key"] == cache_key

    def test_get_with_metadata_respects_ttl(self):
        """Test that get_with_metadata() respects TTL expiration."""
        import time
        from cacheness import cacheness
        from cacheness.config import (
            CacheConfig,
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with short TTL
            storage_config = CacheStorageConfig(cache_dir=temp_dir)
            metadata_config = CacheMetadataConfig(
                metadata_backend="json", default_ttl_seconds=0.1
            )
            compression_config = CompressionConfig()
            serialization_config = SerializationConfig()
            handler_config = HandlerConfig()

            config = CacheConfig(
                storage=storage_config,
                metadata=metadata_config,
                compression=compression_config,
                serialization=serialization_config,
                handlers=handler_config,
            )

            cache = cacheness(config)

            # Create entry
            test_data = {"expired": "data"}
            cache.put(test_data, ttl_test="entry")

            # Verify exists initially
            result = cache.get_with_metadata(ttl_test="entry")
            assert result is not None

            # Wait for expiration
            time.sleep(0.15)

            # Should return None after expiration
            result = cache.get_with_metadata(ttl_test="entry")
            assert result is None

    def test_get_with_metadata_with_custom_metadata(self, cache):
        """Test that get_with_metadata() returns all metadata fields including nested metadata."""
        # Create entry
        test_data = np.array([10, 20, 30])
        cache.put(test_data, custom_test="data_with_metadata")

        # Retrieve with metadata
        result = cache.get_with_metadata(custom_test="data_with_metadata")
        assert result is not None

        data, metadata = result
        np.testing.assert_array_equal(data, test_data)

        # Verify metadata contains expected fields
        assert "cache_key" in metadata
        assert "data_type" in metadata
        assert "created_at" in metadata
        assert "metadata" in metadata  # Nested metadata dict

        # Verify nested metadata structure exists and is a dict
        meta_dict = metadata["metadata"]
        assert isinstance(meta_dict, dict)
        assert len(meta_dict) > 0  # Contains handler-specific fields

    def test_exists_returns_true_for_cached_entry(self, cache):
        """Test that exists() returns True for entries that exist."""
        # Create a cache entry
        test_data = {"value": 42}
        cache.put(test_data, test_key="exists_test")

        # Check existence using kwargs
        assert cache.exists(test_key="exists_test") is True

        # Check existence using direct cache_key
        cache_key = cache._create_cache_key({"test_key": "exists_test"})
        assert cache.exists(cache_key=cache_key) is True

    def test_exists_returns_false_for_missing_entry(self, cache):
        """Test that exists() returns False for entries that don't exist."""
        # Check for non-existent entry using kwargs
        assert cache.exists(nonexistent_key="missing") is False

        # Check for non-existent entry using direct cache_key
        assert cache.exists(cache_key="fake_cache_key_123") is False

    def test_exists_returns_false_for_expired_entry(self, temp_cache_dir):
        """Test that exists() returns False for expired entries."""
        from cacheness.config import (
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )
        import time

        # Create cache with a long TTL, then make stored expiry deterministic.
        storage_config = CacheStorageConfig(
            cache_dir=str(temp_cache_dir), cleanup_on_init=False
        )
        metadata_config = CacheMetadataConfig(
            metadata_backend="json", default_ttl_seconds=3600
        )
        compression_config = CompressionConfig()
        serialization_config = SerializationConfig()
        handler_config = HandlerConfig()

        config = CacheConfig(
            storage=storage_config,
            metadata=metadata_config,
            compression=compression_config,
            serialization=serialization_config,
            handlers=handler_config,
        )

        cache = cacheness(config)

        # Create entry
        test_data = {"value": 99}
        cache.put(test_data, expired_test="entry")

        # Verify it exists initially
        assert cache.exists(expired_test="entry") is True

        cache_key = cache._create_cache_key({"expired_test": "entry"})
        entry = cache.metadata_backend.get_entry(cache_key)
        entry["ttl_seconds"] = -1
        entry["expires_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
        cache.metadata_backend.put_entry(cache_key, entry)

        # Verify it no longer exists
        assert cache.exists(expired_test="entry") is False

    def test_exists_check_expiration_parameter(self, temp_cache_dir):
        """Test that exists() respects the check_expiration parameter."""
        from cacheness.config import (
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )
        import time

        # Create cache with very short TTL
        storage_config = CacheStorageConfig(
            cache_dir=str(temp_cache_dir), cleanup_on_init=False
        )
        metadata_config = CacheMetadataConfig(
            metadata_backend="json", default_ttl_seconds=0.1
        )
        compression_config = CompressionConfig()
        serialization_config = SerializationConfig()
        handler_config = HandlerConfig()

        config = CacheConfig(
            storage=storage_config,
            metadata=metadata_config,
            compression=compression_config,
            serialization=serialization_config,
            handlers=handler_config,
        )

        cache = cacheness(config)

        # Create entry
        test_data = {"value": 123}
        cache.put(test_data, expiration_param_test="entry")

        # Wait for expiration
        time.sleep(0.15)

        # With check_expiration=True (default), should return False
        assert cache.exists(expiration_param_test="entry") is False

        # With check_expiration=False, should return True (entry still in metadata)
        assert (
            cache.exists(expiration_param_test="entry", check_expiration=False) is True
        )

    def test_exists_is_metadata_only(self, cache, tmp_path):
        """Test that exists() does not load the blob file."""
        # Create a large entry
        large_data = np.random.rand(1000, 1000)  # ~8MB array
        cache.put(large_data, large_test="data")

        # Get the cache entry to find the blob file path
        cache_key = cache._create_cache_key({"large_test": "data"})
        entry = cache.metadata_backend.get_entry(cache_key)
        blob_path = cache._resolve_actual_path(
            entry.get("metadata", {}).get("actual_path")
        )

        # Delete the blob file (but keep metadata)
        if blob_path.exists():
            blob_path.unlink()

        # exists() should still return True (metadata exists)
        # This proves it doesn't access the blob file
        assert cache.exists(large_test="data") is True

        # But get() should return None (blob missing)
        assert cache.get(large_test="data") is None

    def test_clear_alias_clears_cache(self, cache):
        """Test that clear() alias works like clear_all()."""
        # Add some entries
        cache.put(np.array([1, 2, 3]), key="a")
        cache.put(np.array([4, 5, 6]), key="b")
        cache.put(np.array([7, 8, 9]), key="c")

        # Verify entries exist
        assert cache.exists(key="a")
        assert cache.exists(key="b")
        assert cache.exists(key="c")

        # Clear cache using clear() alias
        removed_count = cache.clear()

        # Verify all entries are gone
        assert not cache.exists(key="a")
        assert not cache.exists(key="b")
        assert not cache.exists(key="c")
        assert removed_count == 3

    def test_clear_alias_returns_count(self, cache):
        """Test that clear() returns the correct count of removed entries."""
        # Empty cache should return 0
        count = cache.clear()
        assert count == 0

        # Add 5 entries
        for i in range(5):
            cache.put({"value": i}, key=f"entry_{i}")

        # Clear should return 5
        count = cache.clear()
        assert count == 5

    def test_clear_all_removes_namespace_blob_files_and_preserves_reserved_files(
        self, tmp_path
    ):
        """clear_all() removes namespace blobs without deleting reserved files."""
        from cacheness.config import (
            CacheMetadataConfig,
            CacheStorageConfig,
            CompressionConfig,
            HandlerConfig,
            SecurityConfig,
            SerializationConfig,
        )

        cache_dir = tmp_path / "cache"

        def make_config(namespace="default"):
            return CacheConfig(
                storage=CacheStorageConfig(cache_dir=str(cache_dir)),
                metadata=CacheMetadataConfig(metadata_backend="json"),
                compression=CompressionConfig(use_blosc2_arrays=False),
                serialization=SerializationConfig(),
                handlers=HandlerConfig(),
                security=SecurityConfig(
                    enable_entry_signing=True,
                    signing_key_file="cache_signing_key.bin",
                ),
                namespace=namespace,
            )

        cache = cacheness(make_config())
        cache.metadata_backend.create_namespace("custom_ns", "Custom")
        custom_cache = cacheness(make_config(namespace="custom_ns"))

        cache.put({"default": 1}, cache_key="default-one")
        cache.put({"default": 2}, cache_key="default-two")
        custom_cache.put({"custom": 3}, cache_key="custom-one")

        blob_files_before = [
            p
            for p in cache_dir.rglob("*")
            if p.is_file()
            and ".intents" not in p.parts
            and not p.name.startswith("cache_metadata.json")
            and not p.name.endswith("_metadata.json")
            and not p.name.startswith("cache_signing_key")
            and p.name != "cacheness_namespaces.json"
        ]
        assert blob_files_before

        removed_count = cache.clear_all()

        default_blob_files_after = [
            p for p in (cache_dir / "default").rglob("*") if p.is_file()
        ]
        custom_blob_files_after = [
            p for p in (cache_dir / "custom_ns").rglob("*") if p.is_file()
        ]
        assert default_blob_files_after == []
        assert custom_blob_files_after
        assert removed_count == 2
        assert custom_cache.get("custom-one") == {"custom": 3}
        assert (cache_dir / "cache_metadata.json").exists()
        assert (cache_dir / "custom_ns_metadata.json").exists()
        assert (cache_dir / "cacheness_namespaces.json").exists()
        assert (cache_dir / "cache_signing_key.bin").exists()

    def test_clear_all_namespaces_removes_namespace_blob_directories(self, tmp_path):
        """clear_all_namespaces() removes non-default namespace blob directories."""
        from cacheness.config import (
            CacheMetadataConfig,
            CacheStorageConfig,
            CompressionConfig,
            HandlerConfig,
            SerializationConfig,
        )

        cache_dir = tmp_path / "cache"

        def make_config(namespace="default"):
            return CacheConfig(
                storage=CacheStorageConfig(cache_dir=str(cache_dir)),
                metadata=CacheMetadataConfig(metadata_backend="json"),
                compression=CompressionConfig(use_blosc2_arrays=False),
                serialization=SerializationConfig(),
                handlers=HandlerConfig(),
                namespace=namespace,
            )

        cache = cacheness(make_config())
        cache.metadata_backend.create_namespace("custom_ns", "Custom")
        custom_cache = cacheness(make_config(namespace="custom_ns"))

        cache.put({"default": 1}, cache_key="default-one")
        custom_cache.put({"custom": 2}, cache_key="custom-one")
        assert any((cache_dir / "default").rglob("*"))
        assert any((cache_dir / "custom_ns").rglob("*"))

        results = cache.clear_all_namespaces()

        assert results["default"] == 1
        assert results["custom_ns"] == -1
        assert not any(p.is_file() for p in (cache_dir / "default").rglob("*"))
        assert not (cache_dir / "custom_ns").exists()
        assert (cache_dir / "cache_metadata.json").exists()

    def test_cleanup_expired_removes_entries(self):
        """Test that cleanup_expired() removes expired entries."""
        from cacheness import cacheness
        from cacheness.config import (
            CacheConfig,
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with long TTL, then make stored expiry deterministic.
            storage_config = CacheStorageConfig(cache_dir=temp_dir)
            metadata_config = CacheMetadataConfig(
                metadata_backend="json", default_ttl_seconds=3600
            )
            compression_config = CompressionConfig()
            serialization_config = SerializationConfig()
            handler_config = HandlerConfig()

            config = CacheConfig(
                storage=storage_config,
                metadata=metadata_config,
                compression=compression_config,
                serialization=serialization_config,
                handlers=handler_config,
            )

            cache = cacheness(config)

            # Create 3 entries
            cache.put({"value": 1}, key="entry_1")
            cache.put({"value": 2}, key="entry_2")
            cache.put({"value": 3}, key="entry_3")

            # Verify all exist
            assert cache.exists(key="entry_1")
            assert cache.exists(key="entry_2")
            assert cache.exists(key="entry_3")

            now = datetime.now(timezone.utc)
            for key in ("entry_1", "entry_2", "entry_3"):
                cache_key = cache._create_cache_key({"key": key})
                entry = cache.metadata_backend.get_entry(cache_key)
                entry["created_at"] = now.isoformat()
                entry["ttl_seconds"] = -1
                entry["expires_at"] = (now - timedelta(seconds=1)).isoformat()
                cache.metadata_backend.put_entry(cache_key, entry)

            # Cleanup expired entries
            removed = cache.cleanup_expired()

            # All should be removed
            assert removed == 3
            assert not cache.exists(key="entry_1")
            assert not cache.exists(key="entry_2")
            assert not cache.exists(key="entry_3")

    def test_cleanup_expired_with_custom_ttl(self):
        """Test cleanup_expired() with custom TTL parameter."""
        from cacheness import cacheness
        from cacheness.config import (
            CacheConfig,
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with long default TTL (won't expire on its own)
            storage_config = CacheStorageConfig(cache_dir=temp_dir)
            metadata_config = CacheMetadataConfig(
                metadata_backend="json",
                default_ttl_seconds=3600,  # 1 hour
            )
            compression_config = CompressionConfig()
            serialization_config = SerializationConfig()
            handler_config = HandlerConfig()

            config = CacheConfig(
                storage=storage_config,
                metadata=metadata_config,
                compression=compression_config,
                serialization=serialization_config,
                handlers=handler_config,
            )

            cache = cacheness(config)

            # Create entries
            cache.put({"value": 1}, key="test_1")
            cache.put({"value": 2}, key="test_2")

            # Custom TTL is a fallback only when no stored expires_at exists.
            old_created_at = (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat()
            for key in ("test_1", "test_2"):
                cache_key = cache._create_cache_key({"key": key})
                entry = cache.metadata_backend.get_entry(cache_key)
                entry.pop("expires_at", None)
                entry.pop("ttl_seconds", None)
                entry["created_at"] = old_created_at
                cache.metadata_backend.put_entry(cache_key, entry)

            # Cleanup with very short custom TTL of 0.1 seconds
            # (shorter than default, so entries should be expired)
            removed = cache.cleanup_expired(ttl_seconds=0.1)

            # Should remove both entries
            assert removed == 2
            assert not cache.exists(key="test_1")
            assert not cache.exists(key="test_2")

    def test_cleanup_expired_honors_stored_expires_at_for_blob_cleanup(self):
        """Public cleanup uses stored expires_at before fallback TTL."""
        from datetime import datetime, timedelta, timezone
        from cacheness import cacheness
        from cacheness.config import (
            CacheConfig,
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            now = datetime.now(timezone.utc)
            storage_config = CacheStorageConfig(
                cache_dir=temp_dir, cleanup_on_init=False
            )
            metadata_config = CacheMetadataConfig(
                metadata_backend="json",
                default_ttl_seconds=99999,
            )
            config = CacheConfig(
                storage=storage_config,
                metadata=metadata_config,
                compression=CompressionConfig(),
                serialization=SerializationConfig(),
                handlers=HandlerConfig(),
            )
            cache = cacheness(config)

            cache.put({"expired": True}, key="stored-past")
            cache.put({"expired": False}, key="stored-future")

            past_key = cache._create_cache_key({"key": "stored-past"})
            future_key = cache._create_cache_key({"key": "stored-future"})
            past_entry = cache.metadata_backend.get_entry(past_key)
            future_entry = cache.metadata_backend.get_entry(future_key)

            past_path = cache._resolve_actual_path(
                past_entry.get("actual_path")
                or past_entry.get("metadata", {}).get("actual_path")
            )
            future_path = cache._resolve_actual_path(
                future_entry.get("actual_path")
                or future_entry.get("metadata", {}).get("actual_path")
            )
            assert past_path.exists()
            assert future_path.exists()

            past_entry["created_at"] = (now - timedelta(minutes=1)).isoformat()
            past_entry["ttl_seconds"] = -1
            past_entry["expires_at"] = (now - timedelta(seconds=1)).isoformat()
            future_entry["created_at"] = (now - timedelta(days=10)).isoformat()
            future_entry["ttl_seconds"] = 999999
            future_entry["expires_at"] = (now + timedelta(days=1)).isoformat()
            cache.metadata_backend.put_entry(past_key, past_entry)
            cache.metadata_backend.put_entry(future_key, future_entry)

            removed = cache.cleanup_expired(ttl_seconds=99999)

            assert removed == 1
            assert cache.metadata_backend.get_entry(past_key) is None
            assert not past_path.exists()
            assert cache.metadata_backend.get_entry(future_key) is not None
            assert future_path.exists()

    def test_cleanup_expired_deletes_blob_files(self):
        import time

        """Test that cleanup_expired() deletes blob files, not just metadata."""
        from pathlib import Path
        from cacheness import cacheness
        from cacheness.config import (
            CacheConfig,
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = CacheStorageConfig(cache_dir=temp_dir)
            metadata_config = CacheMetadataConfig(
                metadata_backend="json", default_ttl_seconds=0.1
            )
            compression_config = CompressionConfig()
            serialization_config = SerializationConfig()
            handler_config = HandlerConfig()

            config = CacheConfig(
                storage=storage_config,
                metadata=metadata_config,
                compression=compression_config,
                serialization=serialization_config,
                handlers=handler_config,
            )

            cache = cacheness(config)

            # Create entry and get blob path
            cache.put(np.array([1, 2, 3]), key="array_test")
            cache_key = cache._create_cache_key({"key": "array_test"})
            entry = cache.metadata_backend.get_entry(cache_key)

            # Get actual_path from metadata (may be nested)
            metadata = entry.get("metadata", {})
            actual_path = metadata.get("actual_path") or entry.get("actual_path")
            blob_path = cache._resolve_actual_path(actual_path)

            # Verify blob file exists
            assert blob_path.exists()

            # Wait for expiration
            time.sleep(0.15)

            # Cleanup
            removed = cache.cleanup_expired()
            assert removed == 1

            # Blob file should be gone
            assert not blob_path.exists()

    def test_init_cleanup_expired_deletes_blob_files_and_invokes_hook(self):
        """Init-time cleanup uses the public cleanup path."""
        from cacheness.config import (
            CacheConfig,
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
            HooksConfig,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            evicted: list[tuple[str, str]] = []

            def on_evict(cache_key: str, reason: str) -> None:
                evicted.append((cache_key, reason))

            base_config = CacheConfig(
                storage=CacheStorageConfig(
                    cache_dir=temp_dir,
                    cleanup_on_init=False,
                ),
                metadata=CacheMetadataConfig(
                    metadata_backend="json",
                    default_ttl_seconds=3600,
                ),
                compression=CompressionConfig(use_blosc2_arrays=False),
                serialization=SerializationConfig(),
                handlers=HandlerConfig(),
                hooks=HooksConfig(on_evict=on_evict),
            )
            cache = cacheness(base_config)

            cache.put(np.array([1, 2, 3]), key="init_expired")
            cache_key = cache._create_cache_key({"key": "init_expired"})
            entry = cache.metadata_backend.get_entry(cache_key)
            assert entry is not None
            actual_path = entry.get("actual_path") or entry.get("metadata", {}).get(
                "actual_path"
            )
            assert actual_path is not None
            blob_path = cache._resolve_actual_path(actual_path)
            assert isinstance(blob_path, Path)
            assert blob_path.exists()

            created_at = datetime.now(timezone.utc) - timedelta(hours=2)
            entry["created_at"] = created_at.isoformat()
            entry["ttl_seconds"] = 1
            entry["expires_at"] = (created_at + timedelta(seconds=1)).isoformat()
            cache.metadata_backend.put_entry(cache_key, entry)

            cleanup_config = CacheConfig(
                storage=CacheStorageConfig(
                    cache_dir=temp_dir,
                    cleanup_on_init=True,
                ),
                metadata=CacheMetadataConfig(
                    metadata_backend="json",
                    default_ttl_seconds=3600,
                ),
                compression=CompressionConfig(use_blosc2_arrays=False),
                serialization=SerializationConfig(),
                handlers=HandlerConfig(),
                hooks=HooksConfig(on_evict=on_evict),
            )
            reopened = cacheness(cleanup_config)

            assert reopened.metadata_backend.get_entry(cache_key) is None
            assert not blob_path.exists()
            assert evicted == [(cache_key, "expired")]

    def test_cleanup_expired_empty_cache(self):
        """Test cleanup_expired() on empty cache returns 0."""
        from cacheness import cacheness
        from cacheness.config import (
            CacheConfig,
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            storage_config = CacheStorageConfig(cache_dir=temp_dir)
            metadata_config = CacheMetadataConfig(
                metadata_backend="json", default_ttl_seconds=3600
            )
            compression_config = CompressionConfig()
            serialization_config = SerializationConfig()
            handler_config = HandlerConfig()

            config = CacheConfig(
                storage=storage_config,
                metadata=metadata_config,
                compression=compression_config,
                serialization=serialization_config,
                handlers=handler_config,
            )

            cache = cacheness(config)

            # Empty cache cleanup should return 0
            removed = cache.cleanup_expired()
            assert removed == 0

    def test_cache_key_generation(self, cache):
        """Test cache key generation from parameters."""
        params1 = {"a": 1, "b": 2, "c": 3}
        params2 = {"c": 3, "b": 2, "a": 1}  # Same params, different order
        params3 = {"a": 1, "b": 2, "c": 4}  # Different values

        key1 = cache._create_cache_key(params1)
        key2 = cache._create_cache_key(params2)
        key3 = cache._create_cache_key(params3)

        # Same parameters should generate same key regardless of order
        assert key1 == key2
        # Different parameters should generate different keys
        assert key1 != key3
        # Keys should be strings of expected length (xxhash truncated to 16 chars)
        assert isinstance(key1, str)
        assert len(key1) == 16  # xxhash truncated to 16 characters

    def test_cache_file_path_generation(self, cache):
        """Test cache file path generation."""
        cache_key = "test_key_123"

        file_path = cache._get_cache_file_path(cache_key)

        assert isinstance(file_path, Path)
        assert cache_key in str(file_path)
        assert str(cache.config.storage.cache_dir) in str(file_path)

    def test_ttl_expiration(self, cache):
        """Test TTL-based expiration."""
        test_data = {"test": "data"}
        cache_key_params = {"test": "ttl_expiration"}

        # Put data
        cache.put(test_data, description="TTL test", **cache_key_params)

        # Should be retrievable immediately
        assert cache.get(**cache_key_params) == test_data

        # Mock expiration check to return True
        with patch.object(cache, "_is_expired", return_value=True):
            result = cache.get(**cache_key_params)
            assert result is None

    def test_cache_stats(self, cache):
        """Test cache statistics functionality."""
        # Initial stats
        stats = cache.get_stats()
        initial_entries = stats.get("total_entries", 0)

        # Add some data
        cache.put({"test": 1}, test_id="1")
        cache.put(np.array([1, 2, 3]), test_id="2")

        # Check updated stats
        stats = cache.get_stats()
        assert stats["total_entries"] == initial_entries + 2
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "hit_rate" in stats
        assert "total_size_mb" in stats

    def test_list_entries(self, cache):
        """Test listing cache entries."""
        # Add some test data
        cache.put({"test": 1}, description="Test 1", test_id="1")
        cache.put({"test": 2}, description="Test 2", test_id="2")

        entries = cache.list_entries()

        assert isinstance(entries, list)
        assert len(entries) >= 2

        # Check entry structure
        for entry in entries:
            assert "cache_key" in entry
            assert "description" in entry
            assert "data_type" in entry
            assert "created" in entry
            assert "last_accessed" in entry
            assert "size_mb" in entry

    def test_error_handling_invalid_data(self, cache):
        """Test error handling for invalid data."""
        # Test with a truly unpickleable object (generators cannot be pickled or dilled)
        unpickleable_object = (x for x in range(10))  # Generator object

        with pytest.raises((ValueError, TypeError)):
            cache.put(unpickleable_object, test="unpickleable")

    def test_concurrent_access(self, cache):
        """Test thread safety of cache operations."""
        import threading
        import time

        results = []
        errors = []

        def cache_worker(worker_id):
            try:
                # Put data
                data = {"worker": worker_id, "data": list(range(100))}
                cache.put(data, worker_id=worker_id, test="concurrent")

                # Small delay
                time.sleep(0.01)

                # Get data
                retrieved = cache.get(worker_id=worker_id, test="concurrent")
                results.append((worker_id, retrieved == data))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Create and start threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert all(success for _, success in results)

    def test_cleanup_initialization(self, temp_cache_dir):
        """Test cleanup on initialization."""
        from cacheness.config import (
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )

        storage_config = CacheStorageConfig(
            cache_dir=str(temp_cache_dir), cleanup_on_init=True
        )
        metadata_config = CacheMetadataConfig(metadata_backend="json")
        compression_config = CompressionConfig()
        serialization_config = SerializationConfig()
        handler_config = HandlerConfig()

        config = CacheConfig(
            storage=storage_config,
            metadata=metadata_config,
            compression=compression_config,
            serialization=serialization_config,
            handlers=handler_config,
        )

        # Create some dummy files
        temp_cache_dir.mkdir(parents=True, exist_ok=True)
        (temp_cache_dir / "old_file.pkl").touch()

        # Initialize cache with cleanup
        cache = cacheness(config)

        # Cache should be initialized properly
        assert cache.config.storage.cleanup_on_init is True

    def test_ttl_expiration_with_utc_timestamps(self, cache):
        """Test that TTL expiration handles UTC timestamps correctly"""
        from datetime import datetime, timezone
        from unittest.mock import patch

        test_data = {"timezone": "test"}
        cache_key_params = {"test": "timezone_ttl"}

        # Store data to create a real cache entry
        cache.put(test_data, **cache_key_params)

        # Find the actual cache key that was created
        entries = cache.list_entries()
        assert len(entries) > 0, "Should have at least one cache entry"

        # Get the first entry's cache key
        cache_key = entries[0]["cache_key"]

        # Test that _is_expired uses UTC time correctly
        with patch("cacheness.core.datetime") as mock_datetime:
            # Mock current UTC time
            current_utc = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = current_utc
            mock_datetime.fromisoformat = datetime.fromisoformat

            # Should not be expired with fresh timestamp (large TTL)
            assert cache._is_expired(cache_key, ttl_seconds=86400) is False

            # Verify datetime.now was called with timezone.utc
            mock_datetime.now.assert_called_with(timezone.utc)

    def test_timezone_aware_expiration_calculation(self, cache):
        """Test expiration calculation with timezone-aware datetime strings"""
        from datetime import datetime, timezone, timedelta
        from unittest.mock import patch

        # Create a mock metadata entry with a timezone-aware timestamp
        utc_time = datetime.now(timezone.utc) - timedelta(hours=2)  # 2 hours ago
        timestamp_iso = utc_time.isoformat()

        # Mock the metadata backend to return our test entry
        mock_entry = {"created_at": timestamp_iso, "description": "timezone test"}

        with patch.object(cache.metadata_backend, "get_entry", return_value=mock_entry):
            # Should be expired (created 2 hours ago, checking with 1 hour TTL = 3600 seconds)
            assert cache._is_expired("test_key", ttl_seconds=3600) is True

            # Should not be expired with longer TTL: 3 hours = 10800 seconds
            assert cache._is_expired("test_key", ttl_seconds=10800) is False

    def test_is_expired_honors_stored_expires_at_before_fallback_ttl(self, cache):
        """Stored expires_at is authoritative over caller fallback TTL."""
        from datetime import datetime, timedelta, timezone
        from unittest.mock import patch

        now = datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc)
        recent_created = (now - timedelta(minutes=1)).isoformat()
        old_created = (now - timedelta(days=10)).isoformat()

        past_expiry_entry = {
            "created_at": recent_created,
            "expires_at": (now - timedelta(seconds=1)).isoformat(),
        }
        future_expiry_entry = {
            "created_at": old_created,
            "expires_at": (now + timedelta(days=1)).isoformat(),
        }

        with patch("cacheness.core.datetime") as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.fromisoformat = datetime.fromisoformat

            with patch.object(
                cache.metadata_backend, "get_entry", return_value=past_expiry_entry
            ):
                assert cache._is_expired("past-expiry", ttl_seconds=99999) is True

            with patch.object(
                cache.metadata_backend, "get_entry", return_value=future_expiry_entry
            ):
                assert cache._is_expired("future-expiry", ttl_seconds=1) is False

    def test_timezone_consistency_across_cache_operations(self, cache):
        """Test that all cache operations use consistent UTC timezone handling"""
        from datetime import datetime, timezone

        test_data = {"consistent": "timezone"}
        cache_key_params = {"consistency": "test"}

        # Store data - this should create a UTC timestamp
        cache.put(test_data, **cache_key_params)

        # The timestamp in metadata should be UTC
        # We can't easily access the internal cache key generation,
        # but we can verify the data is retrievable (indicating consistent timezone handling)
        retrieved = cache.get(**cache_key_params)
        assert retrieved == test_data

        # Test with current time for comparison
        current_utc = datetime.now(timezone.utc)
        assert current_utc.tzinfo == timezone.utc  # Verify we're using UTC

    def test_size_limit_enforcement(self, temp_cache_dir):
        """Test that cache enforces size limits with LRU eviction."""
        from cacheness.config import (
            CacheStorageConfig,
            CacheMetadataConfig,
            CompressionConfig,
            SerializationConfig,
            HandlerConfig,
        )
        import time

        # Create cache with very small size limit (0.005 MB = ~5KB)
        storage_config = CacheStorageConfig(
            cache_dir=str(temp_cache_dir), max_cache_size_mb=0.005
        )
        metadata_config = CacheMetadataConfig(metadata_backend="json")
        compression_config = CompressionConfig()
        serialization_config = SerializationConfig()
        handler_config = HandlerConfig()

        config = CacheConfig(
            storage=storage_config,
            metadata=metadata_config,
            compression=compression_config,
            serialization=serialization_config,
            handlers=handler_config,
        )

        cache = cacheness(config)

        # Put several entries that exceed the size limit
        # Each entry is ~500 bytes, so 20 entries = ~10KB, exceeding 5KB limit
        for i in range(20):
            data = {"index": i, "data": "x" * 500}
            cache.put(data, entry=f"entry_{i}")
            time.sleep(0.02)  # Delay to ensure different accessed_at timestamps

        # After enforcing 80% of 5KB limit = 4KB, only ~8 most recent entries should remain
        # Check that oldest entries were evicted
        first_entry = cache.get(entry="entry_0")
        assert first_entry is None, "Oldest entry should have been evicted"

        second_entry = cache.get(entry="entry_1")
        assert second_entry is None, "Second entry should have been evicted"

        # More recent entries should still exist
        last_entry = cache.get(entry="entry_19")
        assert last_entry is not None, "Most recent entry should still exist"
        assert last_entry["index"] == 19

        # Check stats to verify size is under target
        stats = cache.metadata_backend.get_stats()
        total_size_mb = stats.get("total_size_mb", 0)
        assert total_size_mb <= 0.005, (
            f"Cache size {total_size_mb}MB should be <= 0.005MB after enforcement"
        )

    def test_size_limit_eviction_deletes_memory_uri_blob(self, temp_cache_dir):
        """D-19/D-22: size eviction deletes URI blobs via the blob backend."""
        from cacheness.config import (
            CacheMetadataConfig,
            CacheStorageConfig,
            CompressionConfig,
            HandlerConfig,
            SerializationConfig,
        )
        from cacheness.storage.backends.blob_backends import InMemoryBlobBackend

        config = CacheConfig(
            storage=CacheStorageConfig(
                cache_dir=str(temp_cache_dir),
                max_cache_size=None,
                max_cache_size_mb=None,
            ),
            metadata=CacheMetadataConfig(
                metadata_backend="json", verify_cache_integrity=False
            ),
            compression=CompressionConfig(use_blosc2_arrays=False),
            serialization=SerializationConfig(),
            handlers=HandlerConfig(),
        )
        cache = cacheness(config)
        cache._blob_store.blob_backend = InMemoryBlobBackend()

        cache.put({"payload": bytes(range(256)) * 8}, cache_key="old-uri-blob")
        old_entry = cache.metadata_backend.get_entry("old-uri-blob")
        assert old_entry is not None
        old_path = old_entry["metadata"]["actual_path"]
        assert isinstance(old_path, str)
        assert old_path.startswith("memory://")
        assert cache._blob_store.blob_backend.exists(old_path)

        time.sleep(0.01)
        cache.put(
            {"payload": bytes(reversed(range(256))) * 8}, cache_key="new-uri-blob"
        )
        new_entry = cache.metadata_backend.get_entry("new-uri-blob")
        assert new_entry is not None
        new_path = new_entry["metadata"]["actual_path"]
        assert isinstance(new_path, str)
        assert cache._blob_store.blob_backend.exists(new_path)

        total_size_bytes = cache.metadata_backend.get_stats()["total_size_bytes"]
        cache.config.storage.max_cache_size = int(total_size_bytes * 0.75)
        assert total_size_bytes > cache.config.storage.max_cache_size_bytes
        cache._enforce_size_limit()

        assert cache.metadata_backend.get_entry("old-uri-blob") is None
        assert not cache._blob_store.blob_backend.exists(old_path)
        assert cache.metadata_backend.get_entry("new-uri-blob") is not None
        assert cache._blob_store.blob_backend.exists(new_path)

    def test_size_limit_remote_delete_failure_logs_warning(
        self, temp_cache_dir, caplog
    ):
        """D-21: remote deletion failures warn and do not crash enforcement."""
        from cacheness.config import (
            CacheBlobConfig,
            CacheMetadataConfig,
            CacheStorageConfig,
            CompressionConfig,
            HandlerConfig,
            SerializationConfig,
        )

        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(temp_cache_dir), max_cache_size=1),
            metadata=CacheMetadataConfig(metadata_backend="json"),
            compression=CompressionConfig(use_blosc2_arrays=False),
            serialization=SerializationConfig(),
            handlers=HandlerConfig(),
            blob=CacheBlobConfig(blob_backend="memory"),
        )
        cache = cacheness(config)

        with (
            patch.object(
                cache.metadata_backend,
                "get_stats",
                return_value={"total_size_bytes": 2},
            ),
            patch.object(
                cache.metadata_backend,
                "cleanup_by_size",
                return_value={
                    "count": 1,
                    "removed_entries": [
                        {"cache_key": "remote-key", "actual_path": "memory://boom"}
                    ],
                },
            ),
            patch.object(
                cache._blob_store.blob_backend,
                "delete_blob",
                side_effect=RuntimeError("backend unavailable"),
            ),
            caplog.at_level("WARNING", logger="cacheness.core"),
        ):
            cache._enforce_size_limit()

        assert (
            "Failed to delete remote blob memory://boom during size enforcement"
            in caplog.text
        )
        assert "backend unavailable" in caplog.text

    def test_storage_mode_size_enforcement_remains_disabled(self, temp_cache_dir):
        """D-23: storage mode does not gain size-eviction behavior."""
        config = CacheConfig(
            cache_dir=str(temp_cache_dir),
            metadata_backend="json",
            blob_backend="memory",
            storage_mode=True,
            max_cache_size=1,
        )
        cache = cacheness(config)

        cache.put({"payload": "durable-" + ("z" * 2000)}, cache_key="durable-key")
        cache._enforce_size_limit()

        assert cache.config.storage.max_cache_size_bytes is None
        assert cache.get(cache_key="durable-key") == {
            "payload": "durable-" + ("z" * 2000)
        }


class TestFactoryMethods:
    """Test factory methods for creating specialized cache instances."""

    def test_for_api_factory(self):
        """Test cacheness.for_api() factory method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = cacheness.for_api(cache_dir=temp_dir, ttl_seconds=28800)

            # Should be a UnifiedCache instance
            assert isinstance(cache, cacheness)

            # Should have the specified configuration
            assert cache.config.storage.cache_dir == temp_dir
            assert (
                cache.config.metadata.default_ttl_seconds == 28800
            )  # 8 hours in seconds
            assert cache.config.compression.pickle_compression_codec == "zstd"

            # Test basic functionality
            test_data = {"api": "response", "data": [1, 2, 3]}
            cache.put(test_data, endpoint="test")

            retrieved = cache.get(endpoint="test")
            assert retrieved == test_data

            cache.close()

    def test_for_api_default_values(self):
        """Test default values for for_api factory."""
        cache = cacheness.for_api()

        # Check default values
        assert cache.config.storage.cache_dir == "./cache"
        assert cache.config.metadata.default_ttl_seconds == 21600  # 6 hours in seconds
        assert cache.config.compression.pickle_compression_codec == "zstd"


class TestTimezoneHandling:
    """Test timezone handling in core cache functionality"""


class TestMemoryCacheConfig:
    """Test memory cache layer configuration and functionality."""

    def test_memory_cache_config_new_naming(self):
        """Test memory cache configuration with parameter names."""
        config = CacheConfig(
            cache_dir="/tmp/test_memory_cache",
            metadata_backend="sqlite",
            enable_memory_cache=True,
            memory_cache_type="lru",
            memory_cache_maxsize=500,
            memory_cache_ttl_seconds=300,
            memory_cache_stats=True,
        )

        # Verify parameters are set correctly
        assert config.metadata.enable_memory_cache is True
        assert config.metadata.memory_cache_type == "lru"
        assert config.metadata.memory_cache_maxsize == 500
        assert config.metadata.memory_cache_ttl_seconds == 300
        assert config.metadata.memory_cache_stats is True

    def test_memory_cache_config_defaults(self):
        """Test memory cache defaults when not specified."""
        config = CacheConfig(cache_dir="/tmp/test_defaults")

        # Verify defaults
        assert config.metadata.enable_memory_cache is False
        assert config.metadata.memory_cache_type == "lru"
        assert config.metadata.memory_cache_maxsize == 1000
        assert config.metadata.memory_cache_ttl_seconds == 300.0
        assert config.metadata.memory_cache_stats is False

    def test_memory_cache_functional_test(self):
        """Test that memory cache layer actually works."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with memory cache layer enabled
            config = CacheConfig(
                cache_dir=temp_dir,
                metadata_backend="sqlite",
                enable_memory_cache=True,
                memory_cache_type="lru",
                memory_cache_maxsize=10,
                memory_cache_ttl_seconds=2,  # Short TTL for testing
                memory_cache_stats=True,
            )

            cache = cacheness(config)

            # Store some test data
            test_data = {"test": "memory_cache_data", "value": 42}
            cache.put(test_data, test_key="memory_cache_test")

            # Verify data is cached
            result = cache.get(test_key="memory_cache_test")
            assert result == test_data

            # Check if memory cache statistics are available (should be CachedMetadataBackend)
            from cacheness.metadata import CachedMetadataBackend

            if isinstance(cache.metadata_backend, CachedMetadataBackend):
                stats = cache.metadata_backend.get_cache_stats()
                assert "memory_cache_enabled" in stats
                assert stats["memory_cache_enabled"] is True
                assert stats["memory_cache_type"] == "lru"
                assert stats["memory_cache_maxsize"] == 10

            cache.close()

    def test_memory_cache_cache_types(self):
        """Test different memory cache types."""
        cache_types = ["lru", "lfu", "fifo", "rr"]

        for cache_type in cache_types:
            config = CacheConfig(
                cache_dir="/tmp/test_cache_types",
                enable_memory_cache=True,
                memory_cache_type=cache_type,
                memory_cache_maxsize=50,
            )

            assert config.metadata.memory_cache_type == cache_type

    def test_memory_cache_validation(self):
        """Test validation of memory cache parameters."""
        # Test invalid cache type
        config = CacheConfig(enable_memory_cache=True, memory_cache_type="invalid_type")

        # Should accept any string (validation happens at cache creation)
        assert config.metadata.memory_cache_type == "invalid_type"

        # Test edge case values
        config = CacheConfig(
            enable_memory_cache=True,
            memory_cache_maxsize=1,  # Minimum
            memory_cache_ttl_seconds=0.1,  # Very short
        )

        assert config.metadata.memory_cache_maxsize == 1
        assert config.metadata.memory_cache_ttl_seconds == 0.1


class TestDeleteByPrefix:
    """Tests for UnifiedCache.delete_by_prefix()."""

    def _make_cache(self, tmp_dir, backend="json"):
        from cacheness.config import CacheStorageConfig, CacheMetadataConfig

        config = CacheConfig(
            storage=CacheStorageConfig(cache_dir=str(tmp_dir)),
            metadata=CacheMetadataConfig(metadata_backend=backend),
        )
        return cacheness(config)

    def test_delete_by_prefix_basic(self, tmp_path):
        """Delete entries matching a prefix, leaving others intact."""
        cache = self._make_cache(tmp_path)
        cache.put("a1", cache_key="models_bert")
        cache.put("a2", cache_key="models_gpt")
        cache.put("b1", cache_key="data_train")

        deleted = cache.delete_by_prefix("models_")

        assert deleted == 2
        assert cache.get(cache_key="models_bert") is None
        assert cache.get(cache_key="models_gpt") is None
        assert cache.get(cache_key="data_train") == "b1"

    def test_delete_by_prefix_removes_blobs(self, tmp_path):
        """Blob files must be removed alongside metadata."""
        cache = self._make_cache(tmp_path)
        cache.put("payload", cache_key="rm_blob")
        entry = cache.metadata_backend.get_entry("rm_blob")
        actual_path = cache._resolve_actual_path(entry["metadata"]["actual_path"])
        assert Path(actual_path).exists()

        cache.delete_by_prefix("rm_")

        assert not Path(actual_path).exists()
        assert cache.metadata_backend.get_entry("rm_blob") is None

    def test_delete_by_prefix_no_match(self, tmp_path):
        """Non-matching prefix returns 0 without error."""
        cache = self._make_cache(tmp_path)
        cache.put("data", cache_key="keep_this")

        assert cache.delete_by_prefix("nonexistent_") == 0
        assert cache.get(cache_key="keep_this") == "data"

    def test_delete_by_prefix_empty_string(self, tmp_path):
        """Empty prefix matches all entries."""
        cache = self._make_cache(tmp_path)
        cache.put("a", cache_key="x")
        cache.put("b", cache_key="y")

        deleted = cache.delete_by_prefix("")

        assert deleted == 2

    def test_delete_by_prefix_sqlite(self, tmp_path):
        """Same behavior with SQLite backend (SQL LIKE path)."""
        cache = self._make_cache(tmp_path, backend="sqlite")
        cache.put("s1", cache_key="prefix_one")
        cache.put("s2", cache_key="prefix_two")
        cache.put("s3", cache_key="other_three")

        deleted = cache.delete_by_prefix("prefix_")

        assert deleted == 2
        assert cache.get(cache_key="other_three") == "s3"


class TestConfigValidation:
    """Tests for config validation of known-bad combinations (HARD-01)."""

    def test_encryption_no_key_file_raises(self):
        """Combo 1: encryption + empty key file raises."""
        from cacheness.config import SecurityConfig
        from cacheness.error_handling import CacheConfigurationError

        with pytest.raises(CacheConfigurationError, match="no key file"):
            SecurityConfig(
                enable_content_encryption=True,
                encryption_key_file="",
                use_in_memory_key=False,
            )

    def test_encryption_signing_disabled_raises(self):
        """Combo 2: encryption + signing disabled raises."""
        from cacheness.config import SecurityConfig
        from cacheness.error_handling import CacheConfigurationError

        with pytest.raises(CacheConfigurationError, match="signing disabled"):
            SecurityConfig(
                enable_content_encryption=True,
                enable_entry_signing=False,
            )

    def test_in_memory_key_encryption_raises(self):
        """Combo 3: in-memory key + encryption raises."""
        from cacheness.config import SecurityConfig
        from cacheness.error_handling import CacheConfigurationError

        with pytest.raises(CacheConfigurationError, match="[Ii]n-memory"):
            SecurityConfig(
                use_in_memory_key=True,
                enable_content_encryption=True,
            )

    def test_json_backend_inline_blobs_raises(self, tmp_path):
        """Combo 5: JSON + inline blobs raises."""
        from cacheness.error_handling import CacheConfigurationError

        with pytest.raises(CacheConfigurationError, match="JSON"):
            CacheConfig(
                metadata_backend="json",
                max_inline_size=4000,
                cache_dir=str(tmp_path),
            )

    def test_valid_encryption_config_no_error(self):
        """Valid encryption config does not raise."""
        from cacheness.config import SecurityConfig

        sec = SecurityConfig()
        assert sec.enable_content_encryption is False

    def test_error_messages_are_actionable(self):
        """Every validation error contains a fix suggestion."""
        from cacheness.config import SecurityConfig
        from cacheness.error_handling import CacheConfigurationError

        try:
            SecurityConfig(
                enable_content_encryption=True,
                encryption_key_file="",
                use_in_memory_key=False,
            )
        except CacheConfigurationError as e:
            msg = str(e)
            assert any(word in msg for word in ("Set", "set", "disable", "switch"))
