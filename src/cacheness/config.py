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
    create_cache_dir: bool = True  # Automatically create cache directory if it doesn't exist
    temp_dir: Optional[str] = None  # Temporary directory for atomic writes (None = use cache_dir/tmp)

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
        False  # Store cache key parameters in metadata for querying - DISABLED by default for performance
    )
    enable_cache_stats: bool = True  # Track cache hit/miss statistics
    auto_cleanup_expired: bool = True  # Automatically clean up expired entries
    
    # Memory cache layer (sits between application and disk-persistent backends)
    enable_memory_cache: bool = False  # Enable in-memory caching of disk-stored metadata entries
    memory_cache_type: str = "lru"  # "lru", "lfu", "fifo", "rr" (random replacement)
    memory_cache_maxsize: int = 1000  # Maximum number of metadata entries to cache in memory
    memory_cache_ttl_seconds: float = 300  # 5 minutes TTL for memory-cached entries
    memory_cache_stats: bool = False  # Enable cache hit/miss statistics for memory cache layer

    def __post_init__(self):
        """Validate metadata backend configuration."""
        if self.metadata_backend not in ["auto", "memory", "json", "sqlite", "sqlite_memory"]:
            raise ValueError(f"Invalid metadata backend: {self.metadata_backend}")

        if self.default_ttl_hours is not None and self.default_ttl_hours <= 0:
            raise ValueError("default_ttl_hours must be positive")
            
        # Validate memory cache layer configuration
        if self.memory_cache_type not in ["lru", "lfu", "fifo", "rr"]:
            raise ValueError(f"Invalid memory_cache_type: {self.memory_cache_type}")
            
        if self.memory_cache_maxsize <= 0:
            raise ValueError("memory_cache_maxsize must be positive")
            
        if self.memory_cache_ttl_seconds <= 0:
            raise ValueError("memory_cache_ttl_seconds must be positive")

        logger.debug(f"Metadata backend configured: {self.metadata_backend}")
        logger.debug(f"Store cache_key_params: {self.store_cache_key_params}")
        if self.memory_cache_type and self.enable_memory_cache:
            logger.debug(f"Memory cache layer: {self.memory_cache_type} (maxsize={self.memory_cache_maxsize}, ttl={self.memory_cache_ttl_seconds}s)")
        if self.metadata_backend in ["auto", "sqlite", "sqlite_memory"]:
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
    
    # Performance and safety options
    enable_parallel_compression: bool = True  # Use multiple threads for compression when available
    compression_threshold_bytes: int = 1024  # Only compress objects larger than this threshold

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

        if self.compression_threshold_bytes < 0:
            raise ValueError("compression_threshold_bytes must be non-negative")

        logger.debug(
            f"Compression configured: parquet={self.parquet_compression}, "
            f"pickle={self.pickle_compression_codec}@{self.pickle_compression_level}, "
            f"threshold={self.compression_threshold_bytes}B"
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
    
    # Performance and safety options
    enable_type_validation: bool = True  # Validate types during key generation

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
    enable_tensorflow_tensors: bool = False  # Disabled by default due to import issues
    
    # Advanced serialization options
    enable_dill_fallback: bool = True  # Use dill for objects that pickle can't handle

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
                "tensorflow_tensors",
            }

            invalid_handlers = set(self.handler_priority) - valid_handlers
            if invalid_handlers:
                raise ValueError(
                    f"Invalid handler names in priority list: {invalid_handlers}"
                )

        logger.debug(
            f"Handlers configured: priority={self.handler_priority or 'default'}"
        )


@dataclass
class SecurityConfig:
    """Configuration for cache security and integrity."""

    # Valid fields that can be signed (kept in sync with CacheEntrySigner.DEFAULT_SIGNED_FIELDS)
    VALID_SIGNED_FIELDS = {
        "cache_key", "file_hash", "data_type", "file_size", 
        "created_at", "prefix", "description", "actual_path",
        "object_type", "storage_format", "serializer", "compression_codec",
        "cache_key_params"  # Optional field when store_cache_key_params=True
    }

    # Entry signing for metadata integrity
    enable_entry_signing: bool = True
    signing_key_file: str = "cache_signing_key.bin"
    
    # Custom field selection (if not provided, uses default enhanced fields)
    custom_signed_fields: Optional[List[str]] = None
    
    # Key management options
    use_in_memory_key: bool = False  # Use in-memory key (not persisted to disk)
    
    # Backward compatibility and key rotation
    allow_unsigned_entries: bool = True  # Allow entries without signatures
    signature_version: int = 1  # For future algorithm changes
    
    # Cleanup behavior for invalid signatures
    delete_invalid_signatures: bool = True  # Automatically delete entries with invalid signatures

    def __post_init__(self):
        """Validate security configuration."""
        if self.signature_version < 1:
            raise ValueError("signature_version must be at least 1")
        
        # Validate custom_signed_fields if provided
        if self.custom_signed_fields:
            invalid_fields = set(self.custom_signed_fields) - self.VALID_SIGNED_FIELDS
            if invalid_fields:
                raise ValueError(
                    f"Invalid signed fields: {invalid_fields}. "
                    f"Valid fields are: {sorted(self.VALID_SIGNED_FIELDS)}"
                )

        logger.debug(
            f"Security configured: signing={self.enable_entry_signing}, "
            f"custom_fields={self.custom_signed_fields}, "
            f"in_memory_key={self.use_in_memory_key}, "
            f"allow_unsigned={self.allow_unsigned_entries}, "
            f"delete_invalid={self.delete_invalid_signatures}"
        )


class CacheConfig:
    """Main configuration class that combines all sub-configurations."""

    storage: CacheStorageConfig = field(default_factory=CacheStorageConfig)
    metadata: CacheMetadataConfig = field(default_factory=CacheMetadataConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    serialization: SerializationConfig = field(default_factory=SerializationConfig)
    handlers: HandlerConfig = field(default_factory=HandlerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def __init__(
        self,
        storage: Optional[CacheStorageConfig] = None,
        metadata: Optional[CacheMetadataConfig] = None,
        compression: Optional[CompressionConfig] = None,
        serialization: Optional[SerializationConfig] = None,
        handlers: Optional[HandlerConfig] = None,
        security: Optional[SecurityConfig] = None,
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
        enable_tensorflow_tensors: Optional[bool] = None,
        enable_dill_fallback: Optional[bool] = None,
        # Additional useful parameters
        enable_cache_stats: Optional[bool] = None,
        auto_cleanup_expired: Optional[bool] = None,
        compression_threshold_bytes: Optional[int] = None,
        enable_parallel_compression: Optional[bool] = None,
        # Security parameters
        delete_invalid_signatures: Optional[bool] = None,
        use_in_memory_key: Optional[bool] = None,
        # Memory cache layer parameters (sits between application and disk backends)
        enable_memory_cache: Optional[bool] = None,
        memory_cache_type: Optional[str] = None,
        memory_cache_maxsize: Optional[int] = None,
        memory_cache_ttl_seconds: Optional[float] = None,
        memory_cache_stats: Optional[bool] = None,
        **kwargs,
    ):
        """Initialize configuration with backwards compatibility support."""

        # Initialize sub-configurations with defaults
        self.storage = storage or CacheStorageConfig()
        self.metadata = metadata or CacheMetadataConfig()
        self.compression = compression or CompressionConfig()
        self.serialization = serialization or SerializationConfig()
        self.handlers = handlers or HandlerConfig()
        self.security = security or SecurityConfig()

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
        if enable_tensorflow_tensors is not None:
            self.handlers.enable_tensorflow_tensors = enable_tensorflow_tensors
        if enable_dill_fallback is not None:
            self.handlers.enable_dill_fallback = enable_dill_fallback

        # Map additional configuration parameters
        if enable_cache_stats is not None:
            self.metadata.enable_cache_stats = enable_cache_stats
        if auto_cleanup_expired is not None:
            self.metadata.auto_cleanup_expired = auto_cleanup_expired
        if compression_threshold_bytes is not None:
            self.compression.compression_threshold_bytes = compression_threshold_bytes
        if enable_parallel_compression is not None:
            self.compression.enable_parallel_compression = enable_parallel_compression

        # Map security configuration parameters
        if delete_invalid_signatures is not None:
            self.security.delete_invalid_signatures = delete_invalid_signatures
        if use_in_memory_key is not None:
            self.security.use_in_memory_key = use_in_memory_key

        # Map memory cache layer configuration parameters
        if enable_memory_cache is not None:
            self.metadata.enable_memory_cache = enable_memory_cache
        if memory_cache_type is not None:
            self.metadata.memory_cache_type = memory_cache_type
        if memory_cache_maxsize is not None:
            self.metadata.memory_cache_maxsize = memory_cache_maxsize
        if memory_cache_ttl_seconds is not None:
            self.metadata.memory_cache_ttl_seconds = memory_cache_ttl_seconds
        if memory_cache_stats is not None:
            self.metadata.memory_cache_stats = memory_cache_stats

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
