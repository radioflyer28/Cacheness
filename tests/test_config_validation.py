"""Nested configuration validation for the post-cutover public API."""

from __future__ import annotations

import json

import pytest

from cacheness.config import (
    CacheConfig,
    CacheMetadataConfig,
    CachePolicyConfig,
    CacheStorageConfig,
    CompressionConfig,
    ConfigValidationError,
    HandlerConfig,
    LifecycleLimits,
    SecurityConfig,
    SerializationConfig,
    load_config_from_dict,
    load_config_from_json,
    save_config_to_json,
    validate_config,
    validate_config_strict,
)


def test_default_nested_configuration_is_valid() -> None:
    """The current constructor provides only ownership-aligned sections."""

    config = CacheConfig()

    assert validate_config(config) == []
    validate_config_strict(config)
    assert isinstance(config.storage, CacheStorageConfig)
    assert isinstance(config.policy, CachePolicyConfig)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: CachePolicyConfig(catalog_page_size=0),
        lambda: CompressionConfig(pickle_compression_level=20),
        lambda: SerializationConfig(max_collection_depth=0),
        lambda: SecurityConfig(signature_version=0),
        lambda: LifecycleLimits(authority_busy_timeout_seconds=0),
        lambda: HandlerConfig(handler_priority=["not-a-handler"]),
    ),
)
def test_invalid_nested_values_are_rejected_at_their_owner(factory) -> None:
    """Each policy section rejects malformed values before storage composition."""

    with pytest.raises(ValueError):
        factory()


def test_validate_config_reports_mutated_storage_and_compression_errors() -> None:
    """Validation remains useful for configurations changed after construction."""

    config = CacheConfig(metadata=CacheMetadataConfig())
    config.storage.cache_dir = 42
    config.compression.pickle_compression_level = 20

    errors = validate_config(config)

    assert {error.field for error in errors} >= {
        "storage.cache_dir",
        "compression.pickle_compression_level",
    }
    with pytest.raises(ValueError, match="storage.cache_dir"):
        validate_config_strict(config)


def test_config_validation_error_has_actionable_text() -> None:
    """Individual validation facts retain their field and rejected value."""

    error = ConfigValidationError("policy.catalog_page_size", "must be positive", 0)

    assert error.field == "policy.catalog_page_size"
    assert "must be positive" in str(error)
    assert "0" in repr(error)


def test_nested_mapping_loader_rejects_retired_flat_configuration() -> None:
    """The cutover fails explicitly instead of rebuilding removed flat options."""

    with pytest.raises(ValueError, match="nested ownership sections"):
        load_config_from_dict({"cache_dir": "./retired"})

    loaded = load_config_from_dict(
        {
            "storage": {"cache_dir": "./current"},
            "policy": {"default_ttl_hours": 4.0},
            "handlers": {"enable_object_pickle": False},
        }
    )

    assert loaded.storage.cache_dir == "./current"
    assert loaded.policy.default_ttl_hours == 4.0
    assert loaded.handlers.enable_object_pickle is False


def test_json_round_trip_preserves_nested_configuration(tmp_path) -> None:
    """JSON persistence preserves the authored nested cache configuration."""

    path = tmp_path / "config.json"
    config = CacheConfig(
        storage=CacheStorageConfig(cache_dir="relative-cache"),
        policy=CachePolicyConfig(default_ttl_hours=None),
        handlers=HandlerConfig(enable_numpy_arrays=False),
    )

    save_config_to_json(config, path)
    loaded = load_config_from_json(path)

    assert json.loads(path.read_text(encoding="utf-8"))["storage"]["cache_dir"] == (
        "relative-cache"
    )
    assert loaded.storage.cache_dir == "relative-cache"
    assert loaded.policy.default_ttl_hours is None
    assert loaded.handlers.enable_numpy_arrays is False
