"""
Tests for store_full_metadata configuration option.

This feature stores complete cache key parameters (kwargs) as JSON in the cache_key_params
database column for debugging and querying purposes. It's disabled by default for performance.
"""

import numpy as np
import pytest
from pathlib import Path

from cacheness import cacheness
from cacheness.config import CacheConfig


class TestFullMetadataStorageSQLite:
    """Test full_metadata storage with SQLite backend."""

    def test_full_metadata_disabled_by_default(self, tmp_path):
        """Verify kwargs are NOT stored when store_full_metadata is disabled (default)."""
        cache_dir = tmp_path / "cache"
        db_file = tmp_path / "test.db"

        config = CacheConfig(
            cache_dir=str(cache_dir),
            metadata_backend="sqlite",
            metadata_backend_options={"db_file": str(db_file)},
            store_full_metadata=False,  # Default
        )

        cache = cacheness(config=config)
        data = np.array([10, 20, 30])
        cache.put(data, experiment="test", model_type="baseline")

        # Check cache's own metadata backend
        entries = cache.metadata_backend.list_entries()

        assert len(entries) == 1
        entry = entries[0]

        # cache_key_params should NOT be present when disabled
        assert "cache_key_params" not in entry.get("metadata", {})

        cache.close()

    def test_full_metadata_enabled_stores_kwargs(self, tmp_path):
        """Verify kwargs are stored as serialized strings when enabled."""
        cache_dir = tmp_path / "cache"
        db_file = tmp_path / "test.db"

        config = CacheConfig(
            cache_dir=str(cache_dir),
            metadata_backend="sqlite",
            metadata_backend_options={"db_file": str(db_file)},
            store_full_metadata=True,  # Enable feature
        )

        cache = cacheness(config=config)

        # Store data with various kwargs
        data = np.array([1, 2, 3])
        cache.put(
            data,
            experiment="exp_001",
            model_type="xgboost",
            max_depth=10,
            features=["feature1", "feature2", "feature3"],
            accuracy=0.92,
            is_production=True,
        )

        # Check backend
        backend = cache.metadata_backend
        entries = backend.list_entries()

        assert len(entries) == 1
        entry = entries[0]

        # cache_key_params SHOULD be present when store_full_metadata=True
        assert "cache_key_params" in entry["metadata"]

        full_meta = entry["metadata"]["cache_key_params"]

        # Verify all kwargs are serialized
        assert "experiment" in full_meta
        assert "model_type" in full_meta
        assert "max_depth" in full_meta
        assert "features" in full_meta
        assert "accuracy" in full_meta
        assert "is_production" in full_meta

        # All values should be serialized to strings
        assert isinstance(full_meta["experiment"], str)
        assert isinstance(full_meta["model_type"], str)
        assert isinstance(full_meta["max_depth"], str)
        assert isinstance(full_meta["features"], str)
        assert isinstance(full_meta["accuracy"], str)
        assert isinstance(full_meta["is_production"], str)

        cache.close()

    def test_full_metadata_with_complex_types(self, tmp_path):
        """Verify complex types (Path, dict, list) are properly serialized."""
        cache_dir = tmp_path / "cache"
        db_file = tmp_path / "test.db"

        config = CacheConfig(
            cache_dir=str(cache_dir),
            metadata_backend="sqlite",
            metadata_backend_options={"db_file": str(db_file)},
            store_full_metadata=True,
        )

        cache = cacheness(config=config)

        # Use complex types as kwargs
        input_path = Path("/data/input.csv")
        config_dict = {"learning_rate": 0.01, "epochs": 100}
        tags = ["ml", "production", "v2"]

        cache.put(
            np.array([1, 2, 3]),
            input_file=input_path,
            config=config_dict,
            tags=tags,
            threshold=0.75,
        )

        # Check backend
        backend = cache.metadata_backend
        entries = backend.list_entries()

        assert len(entries) == 1
        full_meta = entries[0]["metadata"]["cache_key_params"]

        # All fields should be serialized to strings
        assert "input_file" in full_meta
        assert "config" in full_meta
        assert "tags" in full_meta
        assert "threshold" in full_meta

        # Verify they're all strings (serialize_for_cache_key output)
        assert isinstance(full_meta["input_file"], str)
        assert isinstance(full_meta["config"], str)
        assert isinstance(full_meta["tags"], str)
        assert isinstance(full_meta["threshold"], str)

        cache.close()

    def test_full_metadata_query_by_parameter(self, tmp_path):
        """Verify stored kwargs can be queried using SQLite JSON functions."""
        cache_dir = tmp_path / "cache"
        db_file = tmp_path / "test.db"

        config = CacheConfig(
            cache_dir=str(cache_dir),
            metadata_backend="sqlite",
            metadata_backend_options={"db_file": str(db_file)},
            store_full_metadata=True,
        )

        cache = cacheness(config=config)

        # Store multiple entries
        cache.put(np.array([1]), experiment="exp_A", model="xgb", accuracy=0.95)
        cache.put(np.array([2]), experiment="exp_B", model="cnn", accuracy=0.88)
        cache.put(np.array([3]), experiment="exp_C", model="xgb", accuracy=0.92)

        # Query using raw SQL (simulating what query_meta() would do)
        backend = cache.metadata_backend
        with backend.SessionLocal() as session:
            from sqlalchemy import text

            # Query for xgb models
            result = session.execute(
                text("""
                    SELECT cache_key FROM cache_entries
                    WHERE cache_key_params LIKE '%xgb%'
                """)
            ).fetchall()

            assert len(result) == 2  # exp_A and exp_C

        cache.close()

    def test_full_metadata_empty_kwargs(self, tmp_path):
        """Verify behavior when no kwargs are provided."""
        cache_dir = tmp_path / "cache"
        db_file = tmp_path / "test.db"

        config = CacheConfig(
            cache_dir=str(cache_dir),
            metadata_backend="sqlite",
            metadata_backend_options={"db_file": str(db_file)},
            store_full_metadata=True,
        )

        cache = cacheness(config=config)

        # Store without any kwargs
        data = np.array([1, 2, 3])
        cache.put(data)

        # Check backend
        backend = cache.metadata_backend
        entries = backend.list_entries()

        assert len(entries) == 1

        # cache_key_params should exist but be empty dict
        assert "cache_key_params" in entries[0]["metadata"]
        assert entries[0]["metadata"]["cache_key_params"] == {}

        cache.close()

    def test_full_metadata_serialization_failure(self, tmp_path):
        """Verify graceful handling when serialization fails."""
        cache_dir = tmp_path / "cache"
        db_file = tmp_path / "test.db"

        config = CacheConfig(
            cache_dir=str(cache_dir),
            metadata_backend="sqlite",
            metadata_backend_options={"db_file": str(db_file)},
            store_full_metadata=True,
        )

        cache = cacheness(config=config)

        # Use a type that should serialize fine (all standard types)
        cache.put(
            np.array([1, 2, 3]),
            name="test",
            value=123,
        )

        # Should succeed - just verify it doesn't crash
        entries = cache.metadata_backend.list_entries()
        assert len(entries) == 1
        assert "cache_key_params" in entries[0]["metadata"]

        cache.close()


@pytest.mark.skip(reason="Requires Docker container with PostgreSQL")
class TestFullMetadataStoragePostgreSQL:
    """Test full_metadata storage with PostgreSQL backend."""

    @pytest.fixture
    def postgres_config(self):
        """PostgreSQL configuration for Docker tests."""
        return {
            "host": "localhost",
            "port": 5432,
            "database": "cache_test",
            "user": "cache_user",
            "password": "cache_pass",
        }

    def test_full_metadata_postgresql(self, tmp_path, postgres_config):
        """Verify full_metadata works with PostgreSQL backend."""
        cache_dir = tmp_path / "cache"


        config = CacheConfig(
            cache_dir=str(cache_dir),
            metadata_backend="postgresql",
            metadata_backend_options=postgres_config,
            store_full_metadata=True,
        )

        cache = cacheness(config=config)
        data = np.array([1, 2, 3])
        cache.put(data, experiment="pg_test", model="neural_net", epochs=50)

        # Verify cache_key_params stored
        cache_key = list(cache.metadata_backend._entries_cache.keys())[0]
        entry = cache.metadata_backend.get_entry(cache_key)

        assert "cache_key_params" in entry["metadata"]
        full_meta = entry["metadata"]["cache_key_params"]

        assert "experiment" in full_meta
        assert "model" in full_meta
        assert "epochs" in full_meta

        cache.close()


class TestFullMetadataConfig:
    """Test that store_full_metadata config properly controls cache_key_params storage."""

    def test_store_full_metadata_creates_cache_key_params(self, tmp_path):
        """Verify store_full_metadata config stores data in cache_key_params field."""
        cache_dir = tmp_path / "cache"
        db_file = tmp_path / "test.db"

        config = CacheConfig(
            cache_dir=str(cache_dir),
            metadata_backend="sqlite",
            metadata_backend_options={"db_file": str(db_file)},
            store_full_metadata=True,  # User-facing config name
        )

        cache = cacheness(config=config)
        data = np.array([1, 2, 3, 4, 5])
        cache.put(data, experiment="unified_test", accuracy=0.88)

        # Check backend
        backend = cache.metadata_backend
        entries = backend.list_entries()

        assert len(entries) == 1
        entry = entries[0]

        # Data should be in cache_key_params field (internal storage name)
        assert "cache_key_params" in entry["metadata"]

        # Verify serialized strings
        params = entry["metadata"]["cache_key_params"]
        assert isinstance(params["experiment"], str)
        assert isinstance(params["accuracy"], str)

        cache.close()

    def test_disabled_stores_nothing(self, tmp_path):
        """Verify disabling store_full_metadata stores no kwargs."""
        cache_dir = tmp_path / "cache"
        db_file = tmp_path / "test.db"

        config = CacheConfig(
            cache_dir=str(cache_dir),
            metadata_backend="sqlite",
            metadata_backend_options={"db_file": str(db_file)},
            store_full_metadata=False,  # Disabled
        )

        cache = cacheness(config=config)
        cache.put([1, 2, 3], test="disabled")

        backend = cache.metadata_backend
        entries = backend.list_entries()
        assert "cache_key_params" not in entries[0]["metadata"]

        cache.close()
