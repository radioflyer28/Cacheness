"""
Configuration Management for Cacheness
=====================================

This module provides a well-structured configuration system following the Single Responsibility Principle.
Configuration is split into focused sub-configurations for better maintainability.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, List, Union
from pathlib import Path

from cacheness.error_handling import CacheConfigurationError

logger = logging.getLogger(__name__)


_TRUSTED_OBJECT_ARRAY_SIGNED_FIELDS = frozenset(
    {
        "cache_key",
        "file_hash",
        "data_type",
        "actual_path",
        "storage_format",
        "serializer",
        "compression_codec",
    }
)

# Sentinel value to distinguish between None (infinite TTL) and unspecified (use default)
_DEFAULT_TTL = object()


@dataclass
class CacheStorageConfig:
    """Configuration for cache storage and directory management."""

    cache_dir: str = "./cache"
    verify_cache_integrity: bool = True

    def __post_init__(self):
        """Validate storage configuration."""
        # Preserve the authored path for configuration serialization. Filesystem
        # boundaries resolve it when performing runtime storage operations.

        logger.debug(f"Storage configured: dir={self.cache_dir}")


@dataclass
class CacheMetadataConfig:
    """Metadata observer settings stored alongside canonical BlobStore entries."""

    verify_cache_integrity: bool = True
    store_cache_key_params: bool = (
        False  # Store cache key parameters in metadata for querying - DISABLED by default for performance
    )
    enable_cache_stats: bool = True  # Track cache hit/miss statistics

    def __post_init__(self):
        """Validate metadata observer configuration."""
        logger.debug(f"Store cache_key_params: {self.store_cache_key_params}")


@dataclass(frozen=True)
class CachePolicyConfig:
    """Finite cache-policy limits independent of BlobStore composition.

    These values control cache semantics only.  They neither select a payload
    backend nor authorize lifecycle transitions; ``BlobStore`` continues to
    own every payload and authoritative catalog mutation.
    """

    default_ttl_hours: float | None = 24.0
    max_authoritative_bytes: int = 2_000 * 1024 * 1024
    catalog_page_size: int = 100
    maintenance_work_cap: int = 100
    max_maintenance_state_bytes: int = 16_384

    def __post_init__(self) -> None:
        """Reject unbounded maintenance policy before any storage access."""

        if self.default_ttl_hours is not None and (
            isinstance(self.default_ttl_hours, bool)
            or not isinstance(self.default_ttl_hours, (int, float))
            or not math.isfinite(self.default_ttl_hours)
            or self.default_ttl_hours <= 0
        ):
            raise ValueError("default_ttl_hours must be a positive finite number or None")
        if (
            not isinstance(self.max_authoritative_bytes, int)
            or isinstance(self.max_authoritative_bytes, bool)
            or self.max_authoritative_bytes < 0
        ):
            raise ValueError("max_authoritative_bytes must be a non-negative integer")
        if (
            not isinstance(self.catalog_page_size, int)
            or isinstance(self.catalog_page_size, bool)
            or self.catalog_page_size <= 0
        ):
            raise ValueError("catalog_page_size must be a positive integer")
        if (
            not isinstance(self.maintenance_work_cap, int)
            or isinstance(self.maintenance_work_cap, bool)
            or self.maintenance_work_cap <= 0
        ):
            raise ValueError("maintenance_work_cap must be a positive integer")
        if self.catalog_page_size > self.maintenance_work_cap:
            raise ValueError("catalog_page_size cannot exceed maintenance_work_cap")
        if (
            not isinstance(self.max_maintenance_state_bytes, int)
            or isinstance(self.max_maintenance_state_bytes, bool)
            or not 512 <= self.max_maintenance_state_bytes <= 65_536
        ):
            raise ValueError(
                "max_maintenance_state_bytes must be an integer between 512 and 65536"
            )

        # Keep policy limits inside the portable catalog contract without
        # importing storage modules while this configuration module initializes.
        if self.catalog_page_size > 256:
            raise ValueError("catalog_page_size exceeds the portable catalog bound")
        if self.maintenance_work_cap > 4_096:
            raise ValueError("maintenance_work_cap exceeds the portable catalog bound")


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
    allow_trusted_object_arrays: bool = False

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

    # Entry-signing policy retained for handler-level trusted payload decisions.
    enable_entry_signing: bool = True

    # Custom field selection (if not provided, uses default enhanced fields)
    custom_signed_fields: Optional[List[str]] = None

    # Handler-level compatibility policy for signed cache entries.
    allow_unsigned_entries: bool = True  # Allow entries without signatures

    def __post_init__(self):
        """Validate security configuration."""
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
            f"allow_unsigned={self.allow_unsigned_entries}"
        )


@dataclass(frozen=True)
class LifecycleAuthorityTopology:
    """Explicit local-authority deployment capability requested by a caller.

    The only supported durable SQLite topology is one current OS user in one
    current interactive or service session on a local filesystem.  Broader
    requests are rejected during configuration rather than downgraded later.
    """

    filesystem: str = "local"
    principal_scope: str = "current_user_current_session"
    durable: bool = True
    multiprocess: bool = True
    exact_cas: bool = True
    indexed_paging: bool = True
    projection: bool = True

    def __post_init__(self) -> None:
        if self.filesystem != "local":
            raise CacheConfigurationError(
                "Lifecycle authority requires a local filesystem",
                context={"filesystem": self.filesystem},
            )
        if self.principal_scope != "current_user_current_session":
            raise CacheConfigurationError(
                "Lifecycle authority supports only the current user and session",
                context={"principal_scope": self.principal_scope},
            )
        for field_name in (
            "durable",
            "multiprocess",
            "exact_cas",
            "indexed_paging",
            "projection",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise CacheConfigurationError(
                    "Lifecycle authority capability requirements must be booleans",
                    context={"field": field_name},
                )


@dataclass(frozen=True)
class LifecycleLimits:
    """Explicit caller-owned bounds for BlobStore lifecycle operations.

    This value contains policy only. Runtime lifecycle components consume the
    instance supplied by :class:`CacheConfig`; they must not recreate it or
    perform repository, recovery, or payload work here.
    """

    # Caller-owned operational policy; benchmark evidence remains independent.
    max_operation_record_bytes: int = 131_072
    max_operation_field_bytes: int = 8_192
    manifest_page_size: int = 256
    operation_page_size: int = 32
    max_inventory_items: int = 4_096
    max_reconcile_actions: int = 32
    orphan_grace_seconds: float = 300.0
    close_wait_seconds: float = 30.0
    key_initialization_timeout_seconds: float = 5.0
    key_initialization_retry_seconds: float = 0.01
    authority_busy_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        """Reject invalid operational bounds rather than silently normalizing them."""
        integer_fields = (
            "max_operation_record_bytes",
            "max_operation_field_bytes",
            "manifest_page_size",
            "operation_page_size",
            "max_inventory_items",
            "max_reconcile_actions",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        for field_name in (
            "orphan_grace_seconds",
            "close_wait_seconds",
            "key_initialization_timeout_seconds",
            "key_initialization_retry_seconds",
            "authority_busy_timeout_seconds",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive finite number")


class CacheConfig:
    """Main configuration class that combines all sub-configurations."""

    storage: CacheStorageConfig = field(default_factory=CacheStorageConfig)
    metadata: CacheMetadataConfig = field(default_factory=CacheMetadataConfig)
    policy: CachePolicyConfig = field(default_factory=CachePolicyConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    serialization: SerializationConfig = field(default_factory=SerializationConfig)
    handlers: HandlerConfig = field(default_factory=HandlerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    lifecycle_limits: LifecycleLimits = field(default_factory=LifecycleLimits)
    lifecycle_topology: LifecycleAuthorityTopology = field(
        default_factory=LifecycleAuthorityTopology
    )

    def __init__(
        self,
        storage: Optional[CacheStorageConfig] = None,
        metadata: Optional[CacheMetadataConfig] = None,
        policy: Optional[CachePolicyConfig] = None,
        compression: Optional[CompressionConfig] = None,
        serialization: Optional[SerializationConfig] = None,
        handlers: Optional[HandlerConfig] = None,
        security: Optional[SecurityConfig] = None,
        lifecycle_limits: Optional[LifecycleLimits] = None,
        lifecycle_topology: Optional[LifecycleAuthorityTopology] = None,
    ):
        """Initialize one ownership-aligned nested cache configuration."""

        # Initialize sub-configurations with defaults
        self.storage = storage or CacheStorageConfig()
        self.metadata = metadata or CacheMetadataConfig()
        if policy is not None and not isinstance(policy, CachePolicyConfig):
            raise ValueError("policy must be a CachePolicyConfig instance")
        self.policy = CachePolicyConfig() if policy is None else policy
        self.compression = compression or CompressionConfig()
        self.serialization = serialization or SerializationConfig()
        self.handlers = handlers or HandlerConfig()
        self.security = security or SecurityConfig()
        if lifecycle_limits is not None and not isinstance(lifecycle_limits, LifecycleLimits):
            raise ValueError("lifecycle_limits must be a LifecycleLimits instance")
        self.lifecycle_limits = (
            LifecycleLimits() if lifecycle_limits is None else lifecycle_limits
        )
        if lifecycle_topology is not None and not isinstance(
            lifecycle_topology, LifecycleAuthorityTopology
        ):
            raise ValueError(
                "lifecycle_topology must be a LifecycleAuthorityTopology instance"
            )
        self.lifecycle_topology = (
            LifecycleAuthorityTopology()
            if lifecycle_topology is None
            else lifecycle_topology
        )

        self.__post_init__()

    def __post_init__(self):
        """Validate overall configuration consistency."""
        if not isinstance(self.policy, CachePolicyConfig):
            raise ValueError("policy must be a CachePolicyConfig instance")
        self._validate_trusted_object_array_configuration()
        
        logger.info("Cache configuration initialized with focused sub-configurations")
    
    def _validate_trusted_object_array_configuration(self) -> None:
        """Require complete authenticity policy before enabling object arrays.

        Object-dtype NumPy arrays require pickle semantics. They may therefore
        cross the executable ObjectHandler boundary only after their payload and
        metadata are authenticated by the normal cache read gate.
        """
        if not self.handlers.allow_trusted_object_arrays:
            return

        if not self.handlers.enable_object_pickle:
            raise ValueError(
                "allow_trusted_object_arrays requires enable_object_pickle=True"
            )
        if not self.security.enable_entry_signing:
            raise ValueError(
                "allow_trusted_object_arrays requires enable_entry_signing=True"
            )
        if not self.metadata.verify_cache_integrity:
            raise ValueError(
                "allow_trusted_object_arrays requires verify_cache_integrity=True"
            )
        if self.security.allow_unsigned_entries:
            raise ValueError(
                "allow_trusted_object_arrays requires allow_unsigned_entries=False"
            )
        if self.security.custom_signed_fields is not None:
            missing_fields = _TRUSTED_OBJECT_ARRAY_SIGNED_FIELDS - set(
                self.security.custom_signed_fields
            )
            if missing_fields:
                raise ValueError(
                    "allow_trusted_object_arrays requires custom_signed_fields "
                    "to include every payload identity field; missing "
                    f"{sorted(missing_fields)}"
                )

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
    - Topology and optional-feature availability checking
    - Cross-field consistency validation
    
    Args:
        config: The CacheConfig instance to validate
        
    Returns:
        List of ConfigValidationError objects. Empty list means valid configuration.
        
    Example:
        >>> config = CacheConfig(
        ...     storage=CacheStorageConfig(cache_dir="./my_cache")
        ... )
        >>> errors = validate_config(config)
        >>> if errors:
        ...     for error in errors:
        ...         print(f"  - {error}")
        ...     raise ValueError("Invalid configuration")
    """
    errors = []
    
    # Validate storage configuration
    if not isinstance(config.storage.cache_dir, (str, Path)):
        errors.append(ConfigValidationError(
            "storage.cache_dir", "must be a string or Path", config.storage.cache_dir
        ))
    
    if not isinstance(config.policy, CachePolicyConfig):
        errors.append(ConfigValidationError("policy", "must be a CachePolicyConfig"))
    else:
        try:
            CachePolicyConfig(
                default_ttl_hours=config.policy.default_ttl_hours,
                max_authoritative_bytes=config.policy.max_authoritative_bytes,
                catalog_page_size=config.policy.catalog_page_size,
                maintenance_work_cap=config.policy.maintenance_work_cap,
                max_maintenance_state_bytes=config.policy.max_maintenance_state_bytes,
            )
        except ValueError as error:
            field = str(error).split(" ", 1)[0]
            errors.append(
                ConfigValidationError(
                    f"policy.{field}", str(error), getattr(config.policy, field, None)
                )
            )
    
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
    if config.security.custom_signed_fields:
        invalid_fields = set(config.security.custom_signed_fields) - SecurityConfig.VALID_SIGNED_FIELDS
        if invalid_fields:
            errors.append(ConfigValidationError(
                "security.custom_signed_fields",
                f"contains invalid fields: {invalid_fields}",
                config.security.custom_signed_fields
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
    
    The dictionary must use ownership-aligned nested sub-configuration objects.
    
    Args:
        data: Dictionary with configuration values
        
    Returns:
        CacheConfig instance
        
    Example:
        >>> # Nested format
        >>> config = load_config_from_dict({
        ...     "storage": {"cache_dir": "./my_cache"},
        ... })
    """
    # Check if nested format
    sub_config_names = {
        "storage",
        "metadata",
        "policy",
        "compression",
        "serialization",
        "handlers",
        "security",
        "lifecycle_limits",
        "lifecycle_topology",
    }
    is_nested = any(key in sub_config_names for key in data.keys())
    
    if is_nested:
        # Nested format - create sub-configs
        storage = CacheStorageConfig(**data.get("storage", {})) if "storage" in data else None
        metadata = CacheMetadataConfig(**data.get("metadata", {})) if "metadata" in data else None
        policy = CachePolicyConfig(**data.get("policy", {})) if "policy" in data else None
        compression = CompressionConfig(**data.get("compression", {})) if "compression" in data else None
        serialization = SerializationConfig(**data.get("serialization", {})) if "serialization" in data else None
        handlers = HandlerConfig(**data.get("handlers", {})) if "handlers" in data else None
        security = SecurityConfig(**data.get("security", {})) if "security" in data else None
        lifecycle_limits = (
            LifecycleLimits(**data["lifecycle_limits"])
            if "lifecycle_limits" in data
            else None
        )
        lifecycle_topology = (
            LifecycleAuthorityTopology(**data["lifecycle_topology"])
            if "lifecycle_topology" in data
            else None
        )
        
        return CacheConfig(
            storage=storage,
            metadata=metadata,
            policy=policy,
            compression=compression,
            serialization=serialization,
            handlers=handlers,
            security=security,
            lifecycle_limits=lifecycle_limits,
            lifecycle_topology=lifecycle_topology,
        )
    raise ValueError("Cache configuration must use nested ownership sections")


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
        >>> config = CacheConfig(storage=CacheStorageConfig(cache_dir="./my_cache"))
        >>> save_config_to_json(config, "cache_config.json")
    """
    import json
    from dataclasses import asdict
    
    data = {
        "storage": asdict(config.storage),
        "metadata": asdict(config.metadata),
        "policy": asdict(config.policy),
        "compression": asdict(config.compression),
        "serialization": asdict(config.serialization),
        "handlers": asdict(config.handlers),
        "security": asdict(config.security),
        "lifecycle_limits": asdict(config.lifecycle_limits),
        "lifecycle_topology": asdict(config.lifecycle_topology),
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
        >>> config = CacheConfig(storage=CacheStorageConfig(cache_dir="./my_cache"))
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
        "policy": asdict(config.policy),
        "compression": asdict(config.compression),
        "serialization": asdict(config.serialization),
        "handlers": asdict(config.handlers),
        "security": asdict(config.security),
        "lifecycle_limits": asdict(config.lifecycle_limits),
        "lifecycle_topology": asdict(config.lifecycle_topology),
    }
    
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
