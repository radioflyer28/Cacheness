"""
Tests for Phase 2.4: Configuration Schema & Validation

Tests the configuration validation system and file loading capabilities.
"""

import pytest
import tempfile
import json
from pathlib import Path

# Import configuration classes and functions
from cacheness.config import (
    CacheConfig,
    CacheMetadataConfig,
    CacheBlobConfig,
    ConfigValidationError,
    validate_config,
    validate_config_strict,
    load_config_from_dict,
    load_config_from_json,
    save_config_to_json,
    create_cache_config,
)

# Import module-level API
import cacheness


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_config():
    """Provide a valid configuration instance."""
    return CacheConfig()


# =============================================================================
# Test CacheBlobConfig
# =============================================================================


class TestCacheBlobConfig:
    """Test the new CacheBlobConfig dataclass."""

    def test_default_values(self):
        """Test default blob config values."""
        config = CacheBlobConfig()
        assert config.blob_backend == "filesystem"
        assert config.blob_backend_options is None
        assert config.use_atomic_writes is True
        assert config.create_subdirectories is True
        assert config.stream_threshold_bytes == 10 * 1024 * 1024

    def test_custom_blob_backend(self):
        """Test setting custom blob backend."""
        config = CacheBlobConfig(
            blob_backend="s3",
            blob_backend_options={"bucket": "my-cache", "region": "us-west-2"},
        )
        assert config.blob_backend == "s3"
        assert config.blob_backend_options == {
            "bucket": "my-cache",
            "region": "us-west-2",
        }

    def test_invalid_blob_backend_options_type(self):
        """Test that non-dict blob_backend_options raises error."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            CacheBlobConfig(blob_backend_options="not a dict")

    def test_invalid_stream_threshold(self):
        """Test that negative stream_threshold_bytes raises error."""
        with pytest.raises(ValueError, match="must be non-negative"):
            CacheBlobConfig(stream_threshold_bytes=-1)


class TestCacheMetadataConfigExtensions:
    """Test the extended CacheMetadataConfig with metadata_backend_options."""

    def test_default_backend_options(self):
        """Test default metadata_backend_options is None."""
        config = CacheMetadataConfig()
        assert config.metadata_backend_options is None

    def test_custom_backend_options(self):
        """Test setting custom backend options."""
        config = CacheMetadataConfig(
            metadata_backend="postgresql",
            metadata_backend_options={
                "connection_url": "postgresql://localhost/cache",
                "pool_size": 10,
            },
        )
        assert config.metadata_backend == "postgresql"
        assert (
            config.metadata_backend_options["connection_url"]
            == "postgresql://localhost/cache"
        )
        assert config.metadata_backend_options["pool_size"] == 10

    def test_invalid_backend_options_type(self):
        """Test that non-dict metadata_backend_options raises error."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            CacheMetadataConfig(metadata_backend_options="not a dict")

    def test_custom_backend_name_allowed(self):
        """Test that custom backend names are allowed (validated at runtime)."""
        # Custom backends are validated when get_metadata_backend() is called
        config = CacheMetadataConfig(metadata_backend="my_custom_backend")
        assert config.metadata_backend == "my_custom_backend"


class TestCacheConfigBlobIntegration:
    """Test CacheConfig integration with blob configuration."""

    def test_blob_config_accessible(self):
        """Test that blob config is accessible from CacheConfig."""
        config = CacheConfig()
        assert hasattr(config, "blob")
        assert isinstance(config.blob, CacheBlobConfig)

    def test_blob_backend_parameter(self):
        """Test blob_backend top-level parameter."""
        config = CacheConfig(blob_backend="memory")
        assert config.blob.blob_backend == "memory"
        assert config.blob_backend == "memory"  # Via property

    def test_blob_backend_options_parameter(self):
        """Test blob_backend_options top-level parameter."""
        options = {"bucket": "test-bucket"}
        config = CacheConfig(blob_backend_options=options)
        assert config.blob.blob_backend_options == options
        assert config.blob_backend_options == options  # Via property

    def test_metadata_backend_options_parameter(self):
        """Test metadata_backend_options top-level parameter."""
        options = {"connection_url": "postgresql://..."}
        config = CacheConfig(metadata_backend_options=options)
        assert config.metadata.metadata_backend_options == options
        assert config.metadata_backend_options == options  # Via property

    def test_full_config_with_backends(self):
        """Test full configuration with both backend options."""
        config = CacheConfig(
            cache_dir="./test_cache",
            metadata_backend="postgresql",
            metadata_backend_options={"connection_url": "postgresql://localhost/cache"},
            blob_backend="s3",
            blob_backend_options={"bucket": "my-cache"},
        )

        assert config.storage.cache_dir == "./test_cache"
        assert config.metadata.metadata_backend == "postgresql"
        assert (
            config.metadata.metadata_backend_options["connection_url"]
            == "postgresql://localhost/cache"
        )
        assert config.blob.blob_backend == "s3"
        assert config.blob.blob_backend_options["bucket"] == "my-cache"


# =============================================================================
# Test Configuration Validation
# =============================================================================


class TestValidateConfig:
    """Test the validate_config function."""

    def test_valid_config_no_errors(self, valid_config):
        """Test that valid config returns no errors."""
        errors = validate_config(valid_config)
        assert errors == []

    def test_invalid_storage_cache_dir_type(self):
        """Test validation of cache_dir type."""
        config = CacheConfig()
        config.storage.cache_dir = 123  # Invalid type

        errors = validate_config(config)
        assert len(errors) == 1
        assert errors[0].field == "storage.cache_dir"
        assert "must be a string" in errors[0].message

    def test_invalid_max_cache_size(self):
        """Test validation of max_cache_size_mb."""
        config = CacheConfig()
        config.storage.max_cache_size_mb = -100

        errors = validate_config(config)
        assert any(e.field == "storage.max_cache_size_mb" for e in errors)

    def test_invalid_default_ttl(self):
        """Test validation of default_ttl_seconds."""
        config = CacheConfig()
        config.metadata.default_ttl_seconds = -5

        errors = validate_config(config)
        assert any(e.field == "metadata.default_ttl_seconds" for e in errors)

    def test_invalid_blob_backend_options_type(self):
        """Test validation of blob_backend_options type."""
        config = CacheConfig()
        config.blob.blob_backend_options = "not a dict"

        errors = validate_config(config)
        assert any(e.field == "blob.blob_backend_options" for e in errors)

    def test_invalid_compression_codec(self):
        """Test validation of compression codec."""
        config = CacheConfig()
        config.compression.pickle_compression_codec = "invalid"

        errors = validate_config(config)
        assert any(e.field == "compression.pickle_compression_codec" for e in errors)

    def test_invalid_compression_level(self):
        """Test validation of compression level range."""
        config = CacheConfig()
        config.compression.pickle_compression_level = 100

        errors = validate_config(config)
        assert any(e.field == "compression.pickle_compression_level" for e in errors)

    def test_invalid_serialization_depth(self):
        """Test validation of max_collection_depth."""
        config = CacheConfig()
        config.serialization.max_collection_depth = 0

        errors = validate_config(config)
        assert any(e.field == "serialization.max_collection_depth" for e in errors)

    def test_invalid_memory_cache_type(self):
        """Test validation of memory_cache_type."""
        config = CacheConfig()
        config.metadata.memory_cache_type = "invalid"

        errors = validate_config(config)
        assert any(e.field == "metadata.memory_cache_type" for e in errors)

    def test_multiple_errors(self):
        """Test that multiple errors are collected."""
        config = CacheConfig()
        config.storage.max_cache_size_mb = -100
        config.metadata.default_ttl_seconds = -5
        config.compression.pickle_compression_level = 100

        errors = validate_config(config)
        assert len(errors) >= 3


class TestValidateConfigStrict:
    """Test the validate_config_strict function."""

    def test_valid_config_no_exception(self, valid_config):
        """Test that valid config doesn't raise."""
        validate_config_strict(valid_config)  # Should not raise

    def test_invalid_config_raises_valueerror(self):
        """Test that invalid config raises ValueError."""
        config = CacheConfig()
        config.storage.max_cache_size_mb = -100

        with pytest.raises(ValueError, match="Invalid configuration"):
            validate_config_strict(config)

    def test_error_message_contains_details(self):
        """Test that error message contains field details."""
        config = CacheConfig()
        config.storage.max_cache_size_mb = -100

        with pytest.raises(ValueError) as exc_info:
            validate_config_strict(config)

        assert "storage.max_cache_size_mb" in str(exc_info.value)


class TestConfigValidationError:
    """Test the ConfigValidationError class."""

    def test_error_repr(self):
        """Test error representation."""
        error = ConfigValidationError("field.name", "must be positive", -5)
        repr_str = repr(error)

        assert "field.name" in repr_str
        assert "must be positive" in repr_str
        assert "-5" in repr_str

    def test_error_str(self):
        """Test error string representation."""
        error = ConfigValidationError("field.name", "must be positive", -5)
        str_repr = str(error)

        assert "field.name" in str_repr
        assert "must be positive" in str_repr

    def test_error_without_value(self):
        """Test error without value."""
        error = ConfigValidationError("field.name", "is required")

        assert error.value is None
        assert "is required" in str(error)


# =============================================================================
# Test Configuration Loading
# =============================================================================


class TestLoadConfigFromDict:
    """Test the load_config_from_dict function."""

    def test_flat_format(self):
        """Test loading flat dictionary format."""
        data = {
            "cache_dir": "./cache",  # Use "./cache" which is preserved as-is
            "metadata_backend": "sqlite",
            "blob_backend": "memory",
        }

        config = load_config_from_dict(data)

        assert config.storage.cache_dir == "./cache"
        assert config.metadata.metadata_backend == "sqlite"
        assert config.blob.blob_backend == "memory"

    def test_nested_format(self):
        """Test loading nested dictionary format."""
        data = {
            "storage": {
                "cache_dir": "./cache"
            },  # Use "./cache" which is preserved as-is
            "metadata": {"metadata_backend": "sqlite"},
            "blob": {"blob_backend": "memory"},
        }

        config = load_config_from_dict(data)

        # Note: CacheStorageConfig converts non-./cache relative paths to absolute
        assert config.storage.cache_dir == "./cache"
        assert config.metadata.metadata_backend == "sqlite"
        assert config.blob.blob_backend == "memory"

    def test_nested_with_options(self):
        """Test nested format with backend options."""
        data = {
            "metadata": {
                "metadata_backend": "postgresql",
                "metadata_backend_options": {"connection_url": "postgresql://..."},
            },
            "blob": {
                "blob_backend": "s3",
                "blob_backend_options": {"bucket": "my-cache"},
            },
        }

        config = load_config_from_dict(data)

        assert config.metadata.metadata_backend == "postgresql"
        assert (
            config.metadata.metadata_backend_options["connection_url"]
            == "postgresql://..."
        )
        assert config.blob.blob_backend == "s3"
        assert config.blob.blob_backend_options["bucket"] == "my-cache"

    def test_partial_nested_format(self):
        """Test partial nested format with defaults."""
        data = {
            "storage": {
                "cache_dir": "./cache"
            }  # Use "./cache" which is preserved as-is
        }

        config = load_config_from_dict(data)

        assert config.storage.cache_dir == "./cache"
        # Other configs should have defaults
        assert config.metadata.metadata_backend == "auto"
        assert config.blob.blob_backend == "filesystem"


class TestLoadConfigFromJson:
    """Test the load_config_from_json function."""

    def test_load_flat_json(self, temp_dir):
        """Test loading flat JSON config."""
        config_file = temp_dir / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "cache_dir": "./cache",  # Use "./cache" which is preserved as-is
                    "metadata_backend": "sqlite",
                }
            )
        )

        config = load_config_from_json(config_file)

        assert config.storage.cache_dir == "./cache"
        assert config.metadata.metadata_backend == "sqlite"

    def test_load_nested_json(self, temp_dir):
        """Test loading nested JSON config."""
        config_file = temp_dir / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "storage": {
                        "cache_dir": "./cache"
                    },  # Use "./cache" which is preserved as-is
                    "blob": {"blob_backend": "memory"},
                }
            )
        )

        config = load_config_from_json(config_file)

        assert config.storage.cache_dir == "./cache"
        assert config.blob.blob_backend == "memory"

    def test_file_not_found(self, temp_dir):
        """Test loading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_config_from_json(temp_dir / "nonexistent.json")


class TestSaveConfigToJson:
    """Test the save_config_to_json function."""

    def test_save_and_reload(self, temp_dir):
        """Test saving and reloading config."""
        original = CacheConfig(
            cache_dir="./cache",  # Use "./cache" which is preserved as-is
            metadata_backend="sqlite",
            blob_backend="memory",
        )

        config_file = temp_dir / "saved_config.json"
        save_config_to_json(original, config_file)

        loaded = load_config_from_json(config_file)

        # Both should have the same cache_dir
        assert loaded.storage.cache_dir == original.storage.cache_dir
        assert loaded.metadata.metadata_backend == original.metadata.metadata_backend
        assert loaded.blob.blob_backend == original.blob.blob_backend

    def test_json_is_valid(self, temp_dir):
        """Test that saved JSON is valid."""
        config = CacheConfig()
        config_file = temp_dir / "config.json"

        save_config_to_json(config, config_file)

        # Should parse without error
        with config_file.open() as f:
            data = json.load(f)

        assert "storage" in data
        assert "metadata" in data
        assert "blob" in data


# =============================================================================
# Test Module-Level API
# =============================================================================


class TestModuleLevelConfigAPI:
    """Test module-level configuration API exports."""

    def test_config_classes_exported(self):
        """Test that config classes are exported."""
        assert hasattr(cacheness, "CacheConfig")
        assert hasattr(cacheness, "CacheBlobConfig")
        assert hasattr(cacheness, "CacheMetadataConfig")
        assert hasattr(cacheness, "CacheStorageConfig")
        assert hasattr(cacheness, "CompressionConfig")
        assert hasattr(cacheness, "SerializationConfig")
        assert hasattr(cacheness, "HandlerConfig")
        assert hasattr(cacheness, "SecurityConfig")

    def test_validation_functions_exported(self):
        """Test that validation functions are exported."""
        assert hasattr(cacheness, "ConfigValidationError")
        assert hasattr(cacheness, "validate_config")
        assert hasattr(cacheness, "validate_config_strict")

    def test_loading_functions_exported(self):
        """Test that loading functions are exported."""
        assert hasattr(cacheness, "load_config_from_dict")
        assert hasattr(cacheness, "load_config_from_json")
        assert hasattr(cacheness, "save_config_to_json")
        assert hasattr(cacheness, "create_cache_config")

    def test_module_level_validation_works(self, valid_config):
        """Test module-level validate_config works."""
        errors = cacheness.validate_config(valid_config)
        assert errors == []

    def test_module_level_loading_works(self):
        """Test module-level load_config_from_dict works."""
        config = cacheness.load_config_from_dict({"cache_dir": "./test"})
        assert config.storage.cache_dir == "./test"


# =============================================================================
# Test create_cache_config Factory
# =============================================================================


class TestCreateCacheConfig:
    """Test the create_cache_config factory function."""

    def test_basic_creation(self):
        """Test basic config creation."""
        config = create_cache_config()
        assert isinstance(config, CacheConfig)

    def test_with_cache_dir(self):
        """Test creation with cache_dir."""
        config = create_cache_config(cache_dir="./factory_cache")
        assert config.storage.cache_dir == "./factory_cache"

    def test_performance_mode(self):
        """Test performance mode."""
        config = create_cache_config(performance_mode=True)
        assert config.compression.pickle_compression_codec == "lz4"
        assert config.compression.pickle_compression_level == 1

    def test_size_mode(self):
        """Test size optimization mode."""
        config = create_cache_config(size_mode=True)
        assert config.compression.pickle_compression_codec == "zstd"
        assert config.compression.pickle_compression_level == 9

    def test_cannot_combine_modes(self):
        """Test that performance_mode and size_mode cannot be combined."""
        with pytest.raises(ValueError, match="Cannot enable both"):
            create_cache_config(performance_mode=True, size_mode=True)

    def test_with_overrides(self):
        """Test creation with overrides."""
        config = create_cache_config(
            cache_dir="./override_cache", metadata_backend="json", blob_backend="memory"
        )
        assert config.storage.cache_dir == "./override_cache"
        assert config.metadata.metadata_backend == "json"
        assert config.blob.blob_backend == "memory"


# =============================================================================
# Integration Tests
# =============================================================================


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_full_workflow(self, temp_dir):
        """Test complete workflow: create, validate, save, load."""
        # Create config
        config = CacheConfig(
            cache_dir=str(temp_dir / "cache"),
            metadata_backend="sqlite",
            metadata_backend_options={"sqlite_db_file": "meta.db"},
            blob_backend="filesystem",
            blob_backend_options={"use_atomic_writes": True},
        )

        # Validate
        errors = validate_config(config)
        assert errors == []

        # Save
        config_file = temp_dir / "config.json"
        save_config_to_json(config, config_file)

        # Load
        loaded = load_config_from_json(config_file)

        # Verify
        assert loaded.storage.cache_dir == str(temp_dir / "cache")
        assert loaded.metadata.metadata_backend == "sqlite"
        assert loaded.blob.blob_backend == "filesystem"

    def test_invalid_config_caught_by_validation(self):
        """Test that invalid config is caught by validation."""
        config = CacheConfig()

        # Make it invalid
        config.compression.pickle_compression_level = 999
        config.metadata.memory_cache_maxsize = -1

        errors = validate_config(config)
        assert len(errors) >= 2

        # strict validation should raise
        with pytest.raises(ValueError):
            validate_config_strict(config)


# =============================================================================
# YAML Tests (optional)
# =============================================================================


class TestYamlConfig:
    """Test YAML configuration loading (requires PyYAML)."""

    @pytest.fixture
    def yaml_available(self):
        """Check if PyYAML is available."""
        try:
            import yaml

            return True
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_load_yaml_config(self, temp_dir, yaml_available):
        """Test loading YAML config."""
        from cacheness.config import load_config_from_yaml

        config_file = temp_dir / "config.yaml"
        config_file.write_text("""
storage:
  cache_dir: ./yaml_cache
metadata:
  metadata_backend: sqlite
blob:
  blob_backend: memory
""")

        config = load_config_from_yaml(config_file)

        assert config.storage.cache_dir == "./yaml_cache"
        assert config.metadata.metadata_backend == "sqlite"
        assert config.blob.blob_backend == "memory"

    def test_save_yaml_config(self, temp_dir, yaml_available):
        """Test saving YAML config."""
        from cacheness.config import save_config_to_yaml, load_config_from_yaml

        original = CacheConfig(
            cache_dir="./saved_yaml", metadata_backend="json", blob_backend="filesystem"
        )

        config_file = temp_dir / "config.yaml"
        save_config_to_yaml(original, config_file)

        loaded = load_config_from_yaml(config_file)

        # Paths should both end with 'saved_yaml' (loaded will be absolute)
        assert loaded.storage.cache_dir.endswith("saved_yaml")
        assert (
            original.storage.cache_dir == "./saved_yaml"
            or original.storage.cache_dir.endswith("saved_yaml")
        )
        assert loaded.metadata.metadata_backend == original.metadata.metadata_backend
