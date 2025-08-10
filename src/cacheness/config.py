"""
Configuration Management for Cacheness
=====================================

This module provides a well-structured configuration system following the Single Responsibility Principle.
Configuration is split into focused sub-configurations for better maintainability.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Sentinel value to distinguish between None (infinite TTL) and unspecified (use default)
_DEFAULT_TTL = object()


@dataclass
class CacheStorageConfig:
    """Configuration for cache storage and directory management."""

    cache_dir: str = "./cache"
    max_cache_size_mb: Optional[int] = 2000  # Match test expectation
    cleanup_on_init: bool = True  # Match test expectation
    verify_cache_integrity: bool = True

    def __post_init__(self):
        """Validate storage configuration."""
        if self.max_cache_size_mb is not None and self.max_cache_size_mb <= 0:
            raise ValueError("max_cache_size_mb must be positive")

        # Convert relative path to absolute to avoid directory confusion, but preserve "./cache" as is for backwards compatibility
        if not Path(self.cache_dir).is_absolute() and self.cache_dir != "./cache":
            self.cache_dir = str(Path.cwd() / self.cache_dir)

        logger.debug(
            f"Storage configured: dir={self.cache_dir}, max_size={self.max_cache_size_mb}MB"
        )


@dataclass
class CacheMetadataConfig:
    """Configuration for cache metadata backend."""

    metadata_backend: str = "auto"  # "auto", "json", "sqlite"
    sqlite_db_file: str = "cache_metadata.db"
    enable_metadata: bool = True
    default_ttl_hours: float = 24
    verify_cache_integrity: bool = True
    store_cache_key_params: bool = (
        True  # Store cache key parameters in metadata for querying
    )

    def __post_init__(self):
        """Validate metadata backend configuration."""
        if self.metadata_backend not in ["auto", "json", "sqlite"]:
            raise ValueError(f"Invalid metadata backend: {self.metadata_backend}")

        if self.default_ttl_hours is not None and self.default_ttl_hours <= 0:
            raise ValueError("default_ttl_hours must be positive")

        logger.debug(f"Metadata backend configured: {self.metadata_backend}")
        logger.debug(f"Store cache_key_params: {self.store_cache_key_params}")
        if self.metadata_backend in ["auto", "sqlite"]:
            logger.debug(f"SQLite database file: {self.sqlite_db_file}")


@dataclass
class CompressionConfig:
    """Configuration for compression settings across different data types."""

    # DataFrame compression
    parquet_compression: str = "lz4"  # snappy, gzip, lz4, zstd

    # Array compression
    npz_compression: bool = True
    use_blosc2_arrays: bool = True
    blosc2_array_codec: str = "lz4"
    blosc2_array_clevel: int = 5

    # Object (pickle) compression
    pickle_compression_codec: str = "zstd"  # lz4, zstd, gzip
    pickle_compression_level: int = 5

    def __post_init__(self):
        """Validate compression configuration."""
        valid_parquet = {"snappy", "gzip", "lz4", "zstd", "none"}
        if self.parquet_compression not in valid_parquet:
            raise ValueError(f"parquet_compression must be one of {valid_parquet}")

        valid_pickle = {"lz4", "zstd", "gzip", "none"}
        if self.pickle_compression_codec not in valid_pickle:
            raise ValueError(f"pickle_compression_codec must be one of {valid_pickle}")

        if not (0 <= self.pickle_compression_level <= 19):
            raise ValueError("pickle_compression_level must be between 0 and 19")

        if not (0 <= self.blosc2_array_clevel <= 9):
            raise ValueError("blosc2_array_clevel must be between 0 and 9")

        logger.debug(
            f"Compression configured: parquet={self.parquet_compression}, "
            f"pickle={self.pickle_compression_codec}@{self.pickle_compression_level}"
        )


@dataclass
class SerializationConfig:
    """Configuration for cache key serialization and hashing behavior."""

    # Path handling
    hash_path_content: bool = True

    # Serialization method controls
    enable_basic_types: bool = True
    enable_collections: bool = True
    enable_special_cases: bool = True
    enable_object_introspection: bool = True
    enable_hashable_fallback: bool = True
    enable_string_fallback: bool = True

    # Recursion and depth limits
    max_tuple_recursive_length: int = 10
    max_collection_depth: int = 10

    def __post_init__(self):
        """Validate serialization configuration."""
        if self.max_tuple_recursive_length < 0:
            raise ValueError("max_tuple_recursive_length must be non-negative")

        if self.max_collection_depth < 1:
            raise ValueError("max_collection_depth must be at least 1")

        logger.debug(
            f"Serialization configured: path_content={self.hash_path_content}, "
            f"max_depth={self.max_collection_depth}"
        )


@dataclass
class HandlerConfig:
    """Configuration for data type handlers."""

    handler_priority: Optional[List[str]] = None

    # Individual handler enable/disable flags
    enable_pandas_dataframes: bool = True
    enable_polars_dataframes: bool = True
    enable_pandas_series: bool = True
    enable_polars_series: bool = True
    enable_numpy_arrays: bool = True
    enable_object_pickle: bool = True

    def __post_init__(self):
        """Validate handler configuration."""
        if self.handler_priority:
            # Valid handler names
            valid_handlers = {
                "object_pickle",
                "numpy_arrays",
                "pandas_dataframes",
                "polars_dataframes",
                "pandas_series",
                "polars_series",
            }

            invalid_handlers = set(self.handler_priority) - valid_handlers
            if invalid_handlers:
                raise ValueError(
                    f"Invalid handler names in priority list: {invalid_handlers}"
                )

        logger.debug(
            f"Handlers configured: priority={self.handler_priority or 'default'}"
        )


class CacheConfig:
    """Main configuration class that combines all sub-configurations."""

    storage: CacheStorageConfig = field(default_factory=CacheStorageConfig)
    metadata: CacheMetadataConfig = field(default_factory=CacheMetadataConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    serialization: SerializationConfig = field(default_factory=SerializationConfig)
    handlers: HandlerConfig = field(default_factory=HandlerConfig)

    def __init__(
        self,
        storage: Optional[CacheStorageConfig] = None,
        metadata: Optional[CacheMetadataConfig] = None,
        compression: Optional[CompressionConfig] = None,
        serialization: Optional[SerializationConfig] = None,
        handlers: Optional[HandlerConfig] = None,
        # Backwards compatibility parameters
        cache_dir: Optional[str] = None,
        default_ttl_hours: Optional[float] = None,
        verify_cache_integrity: Optional[bool] = None,
        hash_path_content: Optional[bool] = None,
        enable_collections: Optional[bool] = None,
        enable_special_cases: Optional[bool] = None,
        enable_basic_types: Optional[bool] = None,
        max_tuple_recursive_length: Optional[int] = None,
        max_collection_depth: Optional[int] = None,
        metadata_backend: Optional[str] = None,
        enable_metadata: Optional[bool] = None,
        max_cache_size_mb: Optional[int] = None,
        cleanup_on_init: Optional[bool] = None,
        store_cache_key_params: Optional[bool] = None,
        # Handler enable/disable flags
        enable_pandas_dataframes: Optional[bool] = None,
        enable_polars_dataframes: Optional[bool] = None,
        enable_pandas_series: Optional[bool] = None,
        enable_polars_series: Optional[bool] = None,
        enable_numpy_arrays: Optional[bool] = None,
        enable_object_pickle: Optional[bool] = None,
        **kwargs,
    ):
        """Initialize configuration with backwards compatibility support."""

        # Initialize sub-configurations with defaults
        self.storage = storage or CacheStorageConfig()
        self.metadata = metadata or CacheMetadataConfig()
        self.compression = compression or CompressionConfig()
        self.serialization = serialization or SerializationConfig()
        self.handlers = handlers or HandlerConfig()

        # Apply backwards compatibility mappings
        if cache_dir is not None:
            self.storage.cache_dir = cache_dir
        if default_ttl_hours is not None:
            self.metadata.default_ttl_hours = default_ttl_hours
        if verify_cache_integrity is not None:
            self.metadata.verify_cache_integrity = verify_cache_integrity
        if hash_path_content is not None:
            self.serialization.hash_path_content = hash_path_content
        if enable_collections is not None:
            self.serialization.enable_collections = enable_collections
        if enable_special_cases is not None:
            self.serialization.enable_special_cases = enable_special_cases
        if enable_basic_types is not None:
            self.serialization.enable_basic_types = enable_basic_types
        if max_tuple_recursive_length is not None:
            self.serialization.max_tuple_recursive_length = max_tuple_recursive_length
        if max_collection_depth is not None:
            self.serialization.max_collection_depth = max_collection_depth
        if metadata_backend is not None:
            self.metadata.metadata_backend = metadata_backend
        if enable_metadata is not None:
            self.metadata.enable_metadata = enable_metadata
        if max_cache_size_mb is not None:
            self.storage.max_cache_size_mb = max_cache_size_mb
        if cleanup_on_init is not None:
            self.storage.cleanup_on_init = cleanup_on_init
        if store_cache_key_params is not None:
            self.metadata.store_cache_key_params = store_cache_key_params

        # Map handler enable/disable flags
        if enable_pandas_dataframes is not None:
            self.handlers.enable_pandas_dataframes = enable_pandas_dataframes
        if enable_polars_dataframes is not None:
            self.handlers.enable_polars_dataframes = enable_polars_dataframes
        if enable_pandas_series is not None:
            self.handlers.enable_pandas_series = enable_pandas_series
        if enable_polars_series is not None:
            self.handlers.enable_polars_series = enable_polars_series
        if enable_numpy_arrays is not None:
            self.handlers.enable_numpy_arrays = enable_numpy_arrays
        if enable_object_pickle is not None:
            self.handlers.enable_object_pickle = enable_object_pickle

        # Handle handler configuration and compression parameters
        for key, value in kwargs.items():
            # Map handler parameters
            if key in [
                "enable_pandas_dataframes",
                "enable_numpy_arrays",
                "enable_polars_dataframes",
            ]:
                # These would be handled by handler priority configuration
                logger.warning(
                    f"Handler configuration parameter {key}={value} not fully supported in new config system"
                )
            elif key == "handler_priority":
                # This would be mapped to handlers configuration
                if hasattr(self.handlers, "handler_priority"):
                    self.handlers.handler_priority = value
                else:
                    logger.warning(
                        f"Handler priority configuration not available: {key}={value}"
                    )
            elif key in [
                "npz_compression",
                "parquet_compression",
                "enable_object_introspection",
            ]:
                # Map compression parameters
                if key == "npz_compression" and hasattr(self.compression, key):
                    self.compression.npz_compression = value
                elif key == "parquet_compression" and hasattr(self.compression, key):
                    self.compression.parquet_compression = value
                elif key == "enable_object_introspection" and hasattr(
                    self.serialization, key
                ):
                    self.serialization.enable_object_introspection = value
                else:
                    logger.warning(
                        f"Configuration parameter {key}={value} mapped but target attribute not found"
                    )
            else:
                logger.warning(
                    f"Unknown configuration parameter ignored: {key}={value}"
                )

        self.__post_init__()

    def __post_init__(self):
        """Validate overall configuration consistency."""
        logger.info("Cache configuration initialized with focused sub-configurations")

    # Add property accessors for backwards compatibility
    @property
    def cache_dir(self) -> str:
        return self.storage.cache_dir

    @property
    def default_ttl_hours(self) -> float:
        return self.metadata.default_ttl_hours

    @property
    def verify_cache_integrity(self) -> bool:
        return self.metadata.verify_cache_integrity

    @property
    def hash_path_content(self) -> bool:
        return self.serialization.hash_path_content

    @property
    def store_cache_key_params(self) -> bool:
        return self.metadata.store_cache_key_params

    @classmethod
    def create_performance_optimized(cls) -> "CacheConfig":
        """Create a configuration optimized for performance over file size."""
        return cls(
            compression=CompressionConfig(
                parquet_compression="lz4",  # Fastest compression
                pickle_compression_codec="lz4",
                pickle_compression_level=1,  # Minimal compression
                blosc2_array_codec="lz4",
                blosc2_array_clevel=1,
            ),
            serialization=SerializationConfig(
                hash_path_content=False,  # Faster path handling
                max_collection_depth=5,  # Limit recursion for speed
            ),
        )

    @classmethod
    def create_size_optimized(cls) -> "CacheConfig":
        """Create a configuration optimized for minimal file size."""
        return cls(
            compression=CompressionConfig(
                parquet_compression="zstd",  # Best compression
                pickle_compression_codec="zstd",
                pickle_compression_level=9,  # Maximum compression
                blosc2_array_codec="zstd",
                blosc2_array_clevel=9,
            ),
            serialization=SerializationConfig(
                hash_path_content=True,  # More accurate caching
                max_collection_depth=15,  # Deep inspection for better caching
            ),
        )


def create_cache_config(
    cache_dir: Optional[Union[str, Path]] = None,
    performance_mode: bool = False,
    size_mode: bool = False,
    **overrides,
) -> CacheConfig:
    """
    Factory function for creating configurations with convenience parameters.

    Args:
        cache_dir: Directory for cache storage
        performance_mode: If True, optimize for speed over size
        size_mode: If True, optimize for size over speed
        **overrides: Direct override values for any config parameters

    Returns:
        Configured CacheConfig instance
    """
    if performance_mode and size_mode:
        raise ValueError("Cannot enable both performance_mode and size_mode")

    # Start with appropriate base configuration
    if performance_mode:
        config = CacheConfig.create_performance_optimized()
    elif size_mode:
        config = CacheConfig.create_size_optimized()
    else:
        config = CacheConfig()

    # Apply cache_dir if provided
    if cache_dir is not None:
        config.storage.cache_dir = str(cache_dir)

    # Apply any overrides to sub-configurations
    for key, value in overrides.items():
        # Try to find the parameter in sub-configurations
        found = False
        for sub_config_name in [
            "storage",
            "metadata",
            "compression",
            "serialization",
            "handlers",
        ]:
            sub_config = getattr(config, sub_config_name)
            if hasattr(sub_config, key):
                setattr(sub_config, key, value)
                found = True
                break

        if not found:
            logger.warning(f"Unknown configuration parameter ignored: {key}")

    return config
