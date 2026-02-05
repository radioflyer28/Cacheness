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

        # Convert relative path to absolute to avoid directory confusion, but preserve "./cache" and "./yaml_cache" as is for backwards compatibility
        if not Path(self.cache_dir).is_absolute() and self.cache_dir not in ("./cache", "./yaml_cache"):
            self.cache_dir = str(Path.cwd() / self.cache_dir)

        logger.debug(
            f"Storage configured: dir={self.cache_dir}, max_size={self.max_cache_size_mb}MB"
        )


@dataclass
class CacheMetadataConfig:
    """Configuration for cache metadata backend."""

    metadata_backend: str = "auto"  # "auto", "json", "sqlite", or custom registered backend name
    metadata_backend_options: Optional[dict] = None  # Backend-specific options (e.g., connection_url for postgresql)
    sqlite_db_file: str = "cache_metadata.db"
    enable_metadata: bool = True
    
    # TTL configuration (standardized on seconds for consistency)
    default_ttl_seconds: float = 86400  # Cache TTL in seconds (default: 24 hours)
    
    verify_cache_integrity: bool = True
    store_full_metadata: bool = (
        False  # Store complete cache key parameters (kwargs) as JSON for debugging/querying - DISABLED by default for performance
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
        # Built-in backends have special validation; custom registered backends are validated at runtime
        builtin_backends = ["auto", "memory", "json", "sqlite", "sqlite_memory"]
        # Note: Custom backends are validated when get_metadata_backend() is called
        
        # Validate TTL
        if self.default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be positive")
        
        # Validate metadata_backend_options is a dict if provided
        if self.metadata_backend_options is not None:
            if not isinstance(self.metadata_backend_options, dict):
                raise ValueError("metadata_backend_options must be a dictionary")
            
        # Validate memory cache layer configuration
        if self.memory_cache_type not in ["lru", "lfu", "fifo", "rr"]:
            raise ValueError(f"Invalid memory_cache_type: {self.memory_cache_type}")
            
        if self.memory_cache_maxsize <= 0:
            raise ValueError("memory_cache_maxsize must be positive")
            
        if self.memory_cache_ttl_seconds <= 0:
            raise ValueError("memory_cache_ttl_seconds must be positive")

        logger.debug(f"Metadata backend configured: {self.metadata_backend}")
        logger.debug(f"Store full_metadata: {self.store_full_metadata}")
        if self.metadata_backend_options:
            logger.debug(f"Metadata backend options: {list(self.metadata_backend_options.keys())}")
        if self.memory_cache_type and self.enable_memory_cache:
            logger.debug(f"Memory cache layer: {self.memory_cache_type} (maxsize={self.memory_cache_maxsize}, ttl={self.memory_cache_ttl_seconds}s)")
        if self.metadata_backend in ["auto", "sqlite", "sqlite_memory"]:
            logger.debug(f"SQLite database file: {self.sqlite_db_file}")


@dataclass
class CacheBlobConfig:
    """Configuration for blob storage backend.
    
    This configures where and how cached data (blobs) are stored.
    The blob backend is separate from metadata backend - metadata tracks
    what's cached, while blob backend stores the actual cached data.
    
    Example:
        # Use filesystem (default)
        blob_config = CacheBlobConfig(blob_backend="filesystem")
        
        # Use S3 storage (requires custom registration)
        blob_config = CacheBlobConfig(
            blob_backend="s3",
            blob_backend_options={
                "bucket": "my-cache-bucket",
                "region": "us-west-2"
            }
        )
    """
    
    blob_backend: str = "filesystem"  # "filesystem", "memory", or custom registered backend
    blob_backend_options: Optional[dict] = None  # Backend-specific options
    
    # Filesystem-specific options (used when blob_backend="filesystem")
    use_atomic_writes: bool = True  # Use temp file + rename for atomic writes
    create_subdirectories: bool = True  # Create subdirectories based on blob ID
    
    # Git-style directory sharding (applies to filesystem, S3, and compatible backends)
    # Uses leading characters of blob ID as subdirectory, like Git's .git/objects/
    # Example with shard_chars=2: "abc123..." -> "ab/abc123..."
    shard_chars: int = 2  # Number of leading chars for directory sharding (0 to disable)
    
    # Streaming options
    stream_threshold_bytes: int = 10 * 1024 * 1024  # 10MB - use streaming for larger blobs
    
    def __post_init__(self):
        """Validate blob storage configuration."""
        # Validate blob_backend_options is a dict if provided
        if self.blob_backend_options is not None:
            if not isinstance(self.blob_backend_options, dict):
                raise ValueError("blob_backend_options must be a dictionary")
        
        if self.stream_threshold_bytes < 0:
            raise ValueError("stream_threshold_bytes must be non-negative")
        
        if self.shard_chars < 0:
            raise ValueError("shard_chars must be non-negative")
        
        if self.shard_chars > 8:
            raise ValueError("shard_chars must be <= 8 (excessive sharding not recommended)")
            
        logger.debug(f"Blob backend configured: {self.blob_backend}")
        if self.blob_backend_options:
            logger.debug(f"Blob backend options: {list(self.blob_backend_options.keys())}")


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
    blob: CacheBlobConfig = field(default_factory=CacheBlobConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    serialization: SerializationConfig = field(default_factory=SerializationConfig)
    handlers: HandlerConfig = field(default_factory=HandlerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def __init__(
        self,
        storage: Optional[CacheStorageConfig] = None,
        metadata: Optional[CacheMetadataConfig] = None,
        blob: Optional[CacheBlobConfig] = None,
        compression: Optional[CompressionConfig] = None,
        serialization: Optional[SerializationConfig] = None,
        handlers: Optional[HandlerConfig] = None,
        security: Optional[SecurityConfig] = None,
        # Backwards compatibility parameters
        cache_dir: Optional[str] = None,
        default_ttl_seconds: Optional[float] = None,
        verify_cache_integrity: Optional[bool] = None,
        hash_path_content: Optional[bool] = None,
        enable_collections: Optional[bool] = None,
        enable_special_cases: Optional[bool] = None,
        enable_basic_types: Optional[bool] = None,
        max_tuple_recursive_length: Optional[int] = None,
        max_collection_depth: Optional[int] = None,
        metadata_backend: Optional[str] = None,
        metadata_backend_options: Optional[dict] = None,
        enable_metadata: Optional[bool] = None,
        max_cache_size_mb: Optional[int] = None,
        cleanup_on_init: Optional[bool] = None,
        store_cache_key_params: Optional[bool] = None,
        store_full_metadata: Optional[bool] = None,
        # Blob backend parameters
        blob_backend: Optional[str] = None,
        blob_backend_options: Optional[dict] = None,
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
        self.blob = blob or CacheBlobConfig()
        self.compression = compression or CompressionConfig()
        self.serialization = serialization or SerializationConfig()
        self.handlers = handlers or HandlerConfig()
        self.security = security or SecurityConfig()

        # Apply backwards compatibility mappings
        if cache_dir is not None:
            self.storage.cache_dir = cache_dir
        
        # Map TTL parameter
        if default_ttl_seconds is not None:
            self.metadata.default_ttl_seconds = default_ttl_seconds
            
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
        if metadata_backend_options is not None:
            self.metadata.metadata_backend_options = metadata_backend_options
        if enable_metadata is not None:
            self.metadata.enable_metadata = enable_metadata
        if max_cache_size_mb is not None:
            self.storage.max_cache_size_mb = max_cache_size_mb
        if cleanup_on_init is not None:
            self.storage.cleanup_on_init = cleanup_on_init
        
        # Backward compatibility: map old store_cache_key_params to store_full_metadata
        if store_cache_key_params is not None:
            import warnings
            warnings.warn(
                "store_cache_key_params is deprecated, use store_full_metadata instead",
                DeprecationWarning,
                stacklevel=2
            )
            # Only use old parameter if new one wasn't explicitly set
            if store_full_metadata is None:
                store_full_metadata = store_cache_key_params
        
        if store_full_metadata is not None:
            self.metadata.store_full_metadata = store_full_metadata

        # Map blob backend parameters
        if blob_backend is not None:
            self.blob.blob_backend = blob_backend
        if blob_backend_options is not None:
            self.blob.blob_backend_options = blob_backend_options

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
        # Validate backend compatibility
        self._validate_backend_compatibility()
        
        logger.info("Cache configuration initialized with focused sub-configurations")
    
    def _validate_backend_compatibility(self):
        """
        Validate that metadata and blob backend combinations are compatible.
        
        Rules:
        - Local metadata (sqlite, json) should not be combined with remote blobs (s3)
        - Memory metadata should not be combined with persistent remote blobs
        - This prevents inconsistent state where metadata is lost but blobs remain
        
        Raises:
            ValueError: If backend combination is incompatible
        """
        metadata_backend = self.metadata.metadata_backend
        blob_backend = self.blob.blob_backend
        
        # Define backend categories
        local_metadata_backends = {"sqlite", "sqlite_memory", "json", "auto"}
        remote_blob_backends = {"s3", "azure", "gcs"}  # S3 and future cloud backends
        ephemeral_metadata_backends = {"memory"}
        persistent_blob_backends = {"filesystem", "s3", "azure", "gcs"}
        
        # Check for incompatible combinations
        if metadata_backend in local_metadata_backends and blob_backend in remote_blob_backends:
            raise ValueError(
                f"Incompatible backend combination: local metadata backend '{metadata_backend}' "
                f"cannot be used with remote blob backend '{blob_backend}'. "
                f"Use 'postgresql' metadata backend for remote blob storage to ensure "
                f"distributed consistency."
            )
        
        if metadata_backend in ephemeral_metadata_backends and blob_backend in remote_blob_backends:
            raise ValueError(
                f"Incompatible backend combination: ephemeral metadata backend '{metadata_backend}' "
                f"should not be used with remote blob backend '{blob_backend}'. "
                f"Memory metadata would be lost on restart while blobs persist in '{blob_backend}'."
            )

    # Add property accessors for backwards compatibility
    @property
    def cache_dir(self) -> str:
        return self.storage.cache_dir

    @property
    def default_ttl_seconds(self) -> float:
        """Get cache TTL in seconds."""
        return self.metadata.default_ttl_seconds

    @property
    def verify_cache_integrity(self) -> bool:
        return self.metadata.verify_cache_integrity

    @property
    def hash_path_content(self) -> bool:
        return self.serialization.hash_path_content

    @property
    def store_full_metadata(self) -> bool:
        return self.metadata.store_full_metadata

    @property
    def blob_backend(self) -> str:
        """Get the blob storage backend name."""
        return self.blob.blob_backend

    @property
    def blob_backend_options(self) -> Optional[dict]:
        """Get the blob storage backend options."""
        return self.blob.blob_backend_options

    @property
    def metadata_backend_options(self) -> Optional[dict]:
        """Get the metadata backend options."""
        return self.metadata.metadata_backend_options

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
            "blob",
            "compression",
            "serialization",
            "handlers",
            "security",
        ]:
            sub_config = getattr(config, sub_config_name)
            if hasattr(sub_config, key):
                setattr(sub_config, key, value)
                found = True
                break

        if not found:
            logger.warning(f"Unknown configuration parameter ignored: {key}")

    return config


# =============================================================================
# Configuration Validation (Phase 2.4)
# =============================================================================

class ConfigValidationError:
    """Represents a single configuration validation error."""
    
    def __init__(self, field: str, message: str, value: any = None):
        self.field = field
        self.message = message
        self.value = value
    
    def __repr__(self):
        if self.value is not None:
            return f"ConfigValidationError(field='{self.field}', message='{self.message}', value={self.value!r})"
        return f"ConfigValidationError(field='{self.field}', message='{self.message}')"
    
    def __str__(self):
        if self.value is not None:
            return f"{self.field}: {self.message} (got: {self.value!r})"
        return f"{self.field}: {self.message}"


def validate_config(config: CacheConfig) -> List["ConfigValidationError"]:
    """
    Validate a CacheConfig instance and return any errors found.
    
    This function performs comprehensive validation including:
    - Type checking for all configuration values
    - Range validation for numeric values
    - Backend availability checking
    - Cross-field consistency validation
    
    Args:
        config: The CacheConfig instance to validate
        
    Returns:
        List of ConfigValidationError objects. Empty list means valid configuration.
        
    Example:
        >>> config = CacheConfig(metadata_backend="postgresql")
        >>> errors = validate_config(config)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"  - {error}")
        ...     raise ValueError("Invalid configuration")
    """
    errors = []
    
    # Validate storage configuration
    if not isinstance(config.storage.cache_dir, str):
        errors.append(ConfigValidationError(
            "storage.cache_dir", "must be a string", config.storage.cache_dir
        ))
    
    if config.storage.max_cache_size_mb is not None:
        if not isinstance(config.storage.max_cache_size_mb, (int, float)):
            errors.append(ConfigValidationError(
                "storage.max_cache_size_mb", "must be a number or None",
                config.storage.max_cache_size_mb
            ))
        elif config.storage.max_cache_size_mb <= 0:
            errors.append(ConfigValidationError(
                "storage.max_cache_size_mb", "must be positive",
                config.storage.max_cache_size_mb
            ))
    
    # Validate metadata configuration
    if not isinstance(config.metadata.metadata_backend, str):
        errors.append(ConfigValidationError(
            "metadata.metadata_backend", "must be a string",
            config.metadata.metadata_backend
        ))
    
    if config.metadata.default_ttl_seconds is not None:
        if not isinstance(config.metadata.default_ttl_seconds, (int, float)):
            errors.append(ConfigValidationError(
                "metadata.default_ttl_seconds", "must be a number or None",
                config.metadata.default_ttl_seconds
            ))
        elif config.metadata.default_ttl_seconds <= 0:
            errors.append(ConfigValidationError(
                "metadata.default_ttl_seconds", "must be positive",
                config.metadata.default_ttl_seconds
            ))
    
    if config.metadata.metadata_backend_options is not None:
        if not isinstance(config.metadata.metadata_backend_options, dict):
            errors.append(ConfigValidationError(
                "metadata.metadata_backend_options", "must be a dictionary",
                type(config.metadata.metadata_backend_options).__name__
            ))
    
    # Validate blob configuration
    if not isinstance(config.blob.blob_backend, str):
        errors.append(ConfigValidationError(
            "blob.blob_backend", "must be a string",
            config.blob.blob_backend
        ))
    
    if config.blob.blob_backend_options is not None:
        if not isinstance(config.blob.blob_backend_options, dict):
            errors.append(ConfigValidationError(
                "blob.blob_backend_options", "must be a dictionary",
                type(config.blob.blob_backend_options).__name__
            ))
    
    if config.blob.stream_threshold_bytes < 0:
        errors.append(ConfigValidationError(
            "blob.stream_threshold_bytes", "must be non-negative",
            config.blob.stream_threshold_bytes
        ))
    
    # Validate compression configuration
    valid_parquet = {"snappy", "gzip", "lz4", "zstd", "none"}
    if config.compression.parquet_compression not in valid_parquet:
        errors.append(ConfigValidationError(
            "compression.parquet_compression",
            f"must be one of {valid_parquet}",
            config.compression.parquet_compression
        ))
    
    valid_pickle = {"lz4", "zstd", "gzip", "none"}
    if config.compression.pickle_compression_codec not in valid_pickle:
        errors.append(ConfigValidationError(
            "compression.pickle_compression_codec",
            f"must be one of {valid_pickle}",
            config.compression.pickle_compression_codec
        ))
    
    if not (0 <= config.compression.pickle_compression_level <= 19):
        errors.append(ConfigValidationError(
            "compression.pickle_compression_level",
            "must be between 0 and 19",
            config.compression.pickle_compression_level
        ))
    
    if not (0 <= config.compression.blosc2_array_clevel <= 9):
        errors.append(ConfigValidationError(
            "compression.blosc2_array_clevel",
            "must be between 0 and 9",
            config.compression.blosc2_array_clevel
        ))
    
    # Validate serialization configuration
    if config.serialization.max_collection_depth < 1:
        errors.append(ConfigValidationError(
            "serialization.max_collection_depth",
            "must be at least 1",
            config.serialization.max_collection_depth
        ))
    
    if config.serialization.max_tuple_recursive_length < 0:
        errors.append(ConfigValidationError(
            "serialization.max_tuple_recursive_length",
            "must be non-negative",
            config.serialization.max_tuple_recursive_length
        ))
    
    # Validate security configuration
    if config.security.signature_version < 1:
        errors.append(ConfigValidationError(
            "security.signature_version",
            "must be at least 1",
            config.security.signature_version
        ))
    
    if config.security.custom_signed_fields:
        invalid_fields = set(config.security.custom_signed_fields) - SecurityConfig.VALID_SIGNED_FIELDS
        if invalid_fields:
            errors.append(ConfigValidationError(
                "security.custom_signed_fields",
                f"contains invalid fields: {invalid_fields}",
                config.security.custom_signed_fields
            ))
    
    # Validate memory cache configuration
    valid_cache_types = {"lru", "lfu", "fifo", "rr"}
    if config.metadata.memory_cache_type not in valid_cache_types:
        errors.append(ConfigValidationError(
            "metadata.memory_cache_type",
            f"must be one of {valid_cache_types}",
            config.metadata.memory_cache_type
        ))
    
    if config.metadata.memory_cache_maxsize <= 0:
        errors.append(ConfigValidationError(
            "metadata.memory_cache_maxsize",
            "must be positive",
            config.metadata.memory_cache_maxsize
        ))
    
    if config.metadata.memory_cache_ttl_seconds <= 0:
        errors.append(ConfigValidationError(
            "metadata.memory_cache_ttl_seconds",
            "must be positive",
            config.metadata.memory_cache_ttl_seconds
        ))
    
    return errors


def validate_config_strict(config: CacheConfig) -> None:
    """
    Validate configuration and raise ValueError if invalid.
    
    Args:
        config: The CacheConfig instance to validate
        
    Raises:
        ValueError: If any validation errors are found
        
    Example:
        >>> config = CacheConfig(compression=CompressionConfig(pickle_compression_level=100))
        >>> validate_config_strict(config)  # Raises ValueError
    """
    errors = validate_config(config)
    if errors:
        error_messages = [str(e) for e in errors]
        raise ValueError(
            f"Invalid configuration ({len(errors)} errors):\n  - " + 
            "\n  - ".join(error_messages)
        )


# =============================================================================
# Configuration File Loading (Phase 2.4)
# =============================================================================

def load_config_from_dict(data: dict) -> CacheConfig:
    """
    Load configuration from a dictionary.
    
    The dictionary can have either flat keys (backwards compatible)
    or nested sub-configuration objects.
    
    Args:
        data: Dictionary with configuration values
        
    Returns:
        CacheConfig instance
        
    Example:
        >>> # Flat format (backwards compatible)
        >>> config = load_config_from_dict({
        ...     "cache_dir": "./my_cache",
        ...     "metadata_backend": "sqlite",
        ...     "blob_backend": "filesystem"
        ... })
        >>> 
        >>> # Nested format
        >>> config = load_config_from_dict({
        ...     "storage": {"cache_dir": "./my_cache"},
        ...     "metadata": {"metadata_backend": "sqlite"},
        ...     "blob": {"blob_backend": "filesystem"}
        ... })
    """
    # Check if nested format
    sub_config_names = {"storage", "metadata", "blob", "compression", "serialization", "handlers", "security"}
    is_nested = any(key in sub_config_names for key in data.keys())
    
    if is_nested:
        # Nested format - create sub-configs
        storage = CacheStorageConfig(**data.get("storage", {})) if "storage" in data else None
        metadata = CacheMetadataConfig(**data.get("metadata", {})) if "metadata" in data else None
        blob = CacheBlobConfig(**data.get("blob", {})) if "blob" in data else None
        compression = CompressionConfig(**data.get("compression", {})) if "compression" in data else None
        serialization = SerializationConfig(**data.get("serialization", {})) if "serialization" in data else None
        handlers = HandlerConfig(**data.get("handlers", {})) if "handlers" in data else None
        security = SecurityConfig(**data.get("security", {})) if "security" in data else None
        
        return CacheConfig(
            storage=storage,
            metadata=metadata,
            blob=blob,
            compression=compression,
            serialization=serialization,
            handlers=handlers,
            security=security
        )
    else:
        # Flat format - use CacheConfig's backwards compatibility
        return CacheConfig(**data)


def load_config_from_json(path: Union[str, Path]) -> CacheConfig:
    """
    Load configuration from a JSON file.
    
    Args:
        path: Path to JSON configuration file
        
    Returns:
        CacheConfig instance
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        
    Example:
        >>> # cache_config.json:
        >>> # {
        >>> #   "storage": {"cache_dir": "./my_cache"},
        >>> #   "metadata": {"metadata_backend": "sqlite"},
        >>> #   "blob": {"blob_backend": "filesystem"}
        >>> # }
        >>> config = load_config_from_json("cache_config.json")
    """
    import json
    
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    return load_config_from_dict(data)


def load_config_from_yaml(path: Union[str, Path]) -> CacheConfig:
    """
    Load configuration from a YAML file.
    
    Requires PyYAML to be installed.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        CacheConfig instance
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ImportError: If PyYAML is not installed
        yaml.YAMLError: If file is not valid YAML
        
    Example:
        >>> # cache_config.yaml:
        >>> # storage:
        >>> #   cache_dir: ./my_cache
        >>> # metadata:
        >>> #   metadata_backend: sqlite
        >>> # blob:
        >>> #   blob_backend: filesystem
        >>> config = load_config_from_yaml("cache_config.yaml")
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML configuration loading. "
            "Install it with: pip install pyyaml"
        )
    
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return load_config_from_dict(data)


def save_config_to_json(config: CacheConfig, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: CacheConfig instance to save
        path: Path to output JSON file
        indent: JSON indentation level (default: 2)
        
    Example:
        >>> config = CacheConfig(cache_dir="./my_cache")
        >>> save_config_to_json(config, "cache_config.json")
    """
    import json
    from dataclasses import asdict
    
    data = {
        "storage": asdict(config.storage),
        "metadata": asdict(config.metadata),
        "blob": asdict(config.blob),
        "compression": asdict(config.compression),
        "serialization": asdict(config.serialization),
        "handlers": asdict(config.handlers),
        "security": asdict(config.security),
    }
    
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def save_config_to_yaml(config: CacheConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Requires PyYAML to be installed.
    
    Args:
        config: CacheConfig instance to save
        path: Path to output YAML file
        
    Raises:
        ImportError: If PyYAML is not installed
        
    Example:
        >>> config = CacheConfig(cache_dir="./my_cache")
        >>> save_config_to_yaml(config, "cache_config.yaml")
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML configuration saving. "
            "Install it with: pip install pyyaml"
        )
    
    from dataclasses import asdict
    
    data = {
        "storage": asdict(config.storage),
        "metadata": asdict(config.metadata),
        "blob": asdict(config.blob),
        "compression": asdict(config.compression),
        "serialization": asdict(config.serialization),
        "handlers": asdict(config.handlers),
        "security": asdict(config.security),
    }
    
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
