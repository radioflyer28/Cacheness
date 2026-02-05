"""
Example integration test using configuration files.

This demonstrates how to use the pre-configured YAML/JSON config files
for integration testing with PostgreSQL and S3/MinIO.
"""

import pytest
from cacheness import cached


# ==================== Tests Using Config Fixtures ====================


def test_cache_with_yaml_config(cacheness_cache_from_yaml):
    """Test cache operations using YAML configuration."""
    cache = cacheness_cache_from_yaml
    
    # Store data
    test_data = {"message": "Hello from YAML config", "value": 42}
    cache.put(test_data, key="yaml_test", version=1)
    
    # Retrieve data
    result = cache.get(key="yaml_test", version=1)
    assert result == test_data
    
    # Cache hit (should be instant)
    result2 = cache.get(key="yaml_test", version=1)
    assert result2 == test_data


def test_cache_with_json_config(cacheness_cache_from_json):
    """Test cache operations using JSON configuration."""
    cache = cacheness_cache_from_json
    
    # Store data
    test_data = {"message": "Hello from JSON config", "items": [1, 2, 3]}
    cache.put(test_data, key="json_test", category="integration")
    
    # Retrieve data
    result = cache.get(key="json_test", category="integration")
    assert result == test_data


def test_signing_key_from_config(cacheness_cache_from_yaml):
    """Test that signing key from config is applied."""
    cache = cacheness_cache_from_yaml
    
    # Verify config has signing key configured
    assert cache.config.security.signing_key_file is not None
    assert "dev_test_key" in cache.config.security.signing_key_file
    
    # Store and retrieve - should be signed
    cache.put({"signed": True}, key="signature_test")
    result = cache.get(key="signature_test")
    assert result == {"signed": True}


def test_backend_config_from_yaml(cacheness_config_from_yaml):
    """Test that backend configuration is loaded correctly."""
    config = cacheness_config_from_yaml
    
    # Verify PostgreSQL metadata backend
    assert config.metadata.metadata_backend == "postgresql"
    assert "postgresql" in config.metadata.metadata_backend_options["connection_url"]
    
    # Verify S3 blob backend
    assert config.blob.blob_backend == "s3"
    assert config.blob.blob_backend_options["bucket"] == "cache-bucket"
    assert config.blob.blob_backend_options["endpoint_url"] == "http://localhost:9000"
    
    # Verify sharding config
    assert config.blob.shard_chars == 2


def test_compression_config_from_yaml(cacheness_config_from_yaml):
    """Test that compression settings are loaded."""
    config = cacheness_config_from_yaml
    
    assert config.compression.use_blosc2_arrays is True
    assert config.compression.blosc2_array_clevel == 5


# ==================== Tests Using Local SQLite Config ====================


def test_local_sqlite_fs_config(cacheness_local_sqlite_fs):
    """Test using local SQLite+Filesystem config (no containers needed)."""
    cache = cacheness_local_sqlite_fs
    
    # Store data
    test_data = {"mode": "local", "backend": "sqlite+filesystem"}
    cache.put(test_data, key="local_test")
    
    # Retrieve data
    result = cache.get(key="local_test")
    assert result == test_data


# ==================== Manual Config Loading Tests ====================


def test_load_config_manually(config_dir):
    """Test loading config files manually."""
    from cacheness.config import load_config_from_yaml, load_config_from_json
    
    # Load YAML
    yaml_config = load_config_from_yaml(str(config_dir / "test_config.yaml"))
    assert yaml_config.security.signing_key_file is not None
    assert yaml_config.metadata.metadata_backend == "postgresql"
    assert yaml_config.blob.blob_backend == "s3"
    
    # Load JSON
    json_config = load_config_from_json(str(config_dir / "test_config.json"))
    assert json_config.security.signing_key_file is not None
    assert json_config.metadata.metadata_backend == "postgresql"
    assert json_config.blob.blob_backend == "s3"


def test_config_override_from_env(config_dir, monkeypatch):
    """Test that environment variables can override config values."""
    import os
    from cacheness.config import load_config_from_yaml
    
    # Set environment variable
    monkeypatch.setenv("S3_BUCKET", "my-custom-bucket")
    
    # Load config
    config = load_config_from_yaml(str(config_dir / "test_config.yaml"))
    
    # Override from environment
    config.blob.blob_backend_options["bucket"] = os.getenv("S3_BUCKET")
    assert config.blob.blob_backend_options["bucket"] == "my-custom-bucket"


# ==================== Decorator Tests with Config ====================


def test_decorator_with_config(cacheness_cache_from_yaml):
    """Test function decorator with configured cache."""
    cache = cacheness_cache_from_yaml
    
    call_count = 0
    
    @cached(cache_instance=cache)
    def expensive_function(x, y):
        nonlocal call_count
        call_count += 1
        return x + y
    
    # First call - should execute
    result1 = expensive_function(10, 20)
    assert result1 == 30
    assert call_count == 1
    
    # Second call - should use cache
    result2 = expensive_function(10, 20)
    assert result2 == 30
    assert call_count == 1  # Not incremented
    
    # Different args - should execute
    result3 = expensive_function(5, 15)
    assert result3 == 20
    assert call_count == 2


# ==================== Complex Data Types with Config ====================


@pytest.mark.parametrize("cache_fixture", [
    "cacheness_cache_from_yaml",
    "cacheness_cache_from_json",
])
def test_complex_data_types(cache_fixture, request):
    """Test caching complex data types with config."""
    cache = request.getfixturevalue(cache_fixture)
    
    # Dictionary with nested structures
    complex_data = {
        "nested": {
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2},
        },
        "values": [{"id": i, "val": i*2} for i in range(5)],
    }
    
    cache.put(complex_data, key="complex", test_type="nested")
    result = cache.get(key="complex", test_type="nested")
    assert result == complex_data


# ==================== Notes ====================

"""
To run these tests:

1. Start Docker containers:
   docker-compose up -d

2. Run integration tests:
   pytest tests/test_config_integration_example.py -v

3. Or use Makefile:
   make test-integration

Configuration files used:
- config/test_config.yaml (PostgreSQL + MinIO)
- config/test_config.json (same as YAML)
- config/local_sqlite_fs.yaml (no Docker needed)

Fixtures available:
- cacheness_cache_from_yaml: Cache from test_config.yaml
- cacheness_cache_from_json: Cache from test_config.json
- cacheness_config_from_yaml: CacheConfig object from YAML
- cacheness_config_from_json: CacheConfig object from JSON
- cacheness_local_sqlite_fs: Local SQLite+FS cache
- config_dir: Path to config directory
"""
