"""
Unified Serialization for Cache Keys - CONFIGURABLE ORDER
========================================================

This module provides consistent serialization for cache key generation
across both UnifiedCache and decorator usage with configurable ordering strategy.
"""

from typing import Any, Optional
import xxhash


def _serialize_path_object(obj: Any, config: Optional[Any] = None) -> Optional[str]:
    """Serialize Path objects with configurable content hashing using existing utilities."""
    try:
        # Import Path and utilities here to avoid circular imports
        from pathlib import Path
        from .file_hashing import hash_file_content, hash_directory_parallel

        if isinstance(obj, Path):
            # Check if we should hash content or just use path string
            should_hash_content = True  # Default behavior
            if config:
                should_hash_content = getattr(
                    config.serialization, "hash_path_content", True
                )

            if should_hash_content and obj.exists():
                if obj.is_file():
                    # Use existing utility for file content hashing
                    content_hash = hash_file_content(obj)
                    # Extract just the hash part if it's a full hash, or use as-is if it's an error
                    if content_hash.startswith(
                        ("missing_file:", "not_a_file:", "error_reading:")
                    ):
                        return f"path_file_error:{content_hash}"
                    else:
                        # Normal hash - for content-based caching, only use the hash (not filename)
                        return f"path_file_content:{content_hash[:16]}"
                elif obj.is_dir():
                    # Use existing utility for directory hashing
                    content_hash = hash_directory_parallel(obj)
                    if content_hash.startswith("missing_directory:"):
                        return f"path_dir_error:{content_hash}"
                    else:
                        # Normal hash - for content-based caching, only use the hash
                        return f"path_dir_content:{content_hash[:16]}"
            else:
                # Either content hashing is disabled or file doesn't exist
                if obj.exists():
                    # Use path string for faster but less robust hashing
                    return f"path_string:{str(obj)}"
                else:
                    return f"path_missing:{str(obj)}"
        else:
            # Generic path-like object
            return f"path_like:{str(obj)}"
    except Exception:
        # Fall back to string representation if path handling fails
        return f"path_error:{str(obj)}"


def _serialize_dataframe(obj: Any) -> Optional[str]:
    """Serialize Pandas or Polars DataFrames."""
    try:
        # Pandas DataFrame
        if str(type(obj).__module__).startswith("pandas"):
            # Hash the values and column info
            values_hash = xxhash.xxh3_64(obj.values.tobytes()).hexdigest()[:16]
            cols_hash = xxhash.xxh3_64(
                ",".join(obj.columns.astype(str)).encode()
            ).hexdigest()[:8]
            return f"pandas_df:{obj.shape}:{cols_hash}:{values_hash}"
        # Polars DataFrame
        elif str(type(obj).__module__).startswith("polars"):
            # For Polars, convert to pandas-compatible format for hashing
            values_hash = xxhash.xxh3_64(
                str(obj.to_pandas().values).encode()
            ).hexdigest()[:16]
            cols_hash = xxhash.xxh3_64(",".join(obj.columns).encode()).hexdigest()[:8]
            return f"polars_df:{obj.shape}:{cols_hash}:{values_hash}"
    except Exception:
        # Fall back to basic info if DataFrame hashing fails
        try:
            return f"dataframe:{obj.shape}:{type(obj).__name__}"
        except Exception:
            pass
    return None


def _serialize_series(obj: Any) -> Optional[str]:
    """Serialize Pandas or Polars Series."""
    try:
        if str(type(obj).__module__).startswith("pandas"):
            # Pandas Series
            values_hash = xxhash.xxh3_64(obj.values.tobytes()).hexdigest()[:16]
            return f"pandas_series:{len(obj)}:{obj.dtype}:{values_hash}"
        elif str(type(obj).__module__).startswith("polars"):
            # Polars Series
            values_hash = xxhash.xxh3_64(
                str(obj.to_pandas().values).encode()
            ).hexdigest()[:16]
            return f"polars_series:{len(obj)}:{obj.dtype}:{values_hash}"
    except Exception:
        # Fall back to basic info
        try:
            return f"series:{len(obj)}:{type(obj).__name__}"
        except Exception:
            pass
    return None


def _serialize_numpy_array(obj: Any) -> Optional[str]:
    """Serialize NumPy arrays and other array-like objects."""
    try:
        content_hash = xxhash.xxh3_64(obj.tobytes()).hexdigest()[:16]
        return f"array:{obj.shape}:{obj.dtype}:{content_hash}"
    except Exception:
        pass  # Fall through to other methods
    return None


def serialize_for_cache_key(obj: Any, config: Optional[Any] = None) -> str:
    """
    Serialize an object for cache key generation with configurable ordering.

    Default Prioritization Strategy:
    1. Basic immutable types (direct representation)
    2. Special cases with custom handling (highest quality)
    3. Collections (recursive - good introspection)
    4. Objects with __dict__ (introspection)
    5. Hashable objects (performance fallback)
    6. String representation (last resort)

    Args:
        obj: Object to serialize
        config: Optional CacheConfig to control serialization behavior

    Returns:
        String representation suitable for cache key generation
    """
    # If no config provided, use defaults
    if config is None:
        # Use default behavior (all methods enabled)
        enable_basic = enable_special = enable_collections = True
        enable_introspection = enable_hashable = enable_string = True
        max_tuple_length = 10
        max_depth = 10
    else:
        # Use nested configuration structure
        enable_basic = getattr(config.serialization, "enable_basic_types", True)
        enable_special = getattr(config.serialization, "enable_special_cases", True)
        enable_collections = getattr(config.serialization, "enable_collections", True)
        enable_introspection = getattr(
            config.serialization, "enable_object_introspection", True
        )
        enable_hashable = getattr(
            config.serialization, "enable_hashable_fallback", True
        )
        enable_string = getattr(config.serialization, "enable_string_fallback", True)
        max_tuple_length = getattr(
            config.serialization, "max_tuple_recursive_length", 10
        )
        max_depth = getattr(config.serialization, "max_collection_depth", 10)

    return _serialize_with_config(
        obj,
        config,
        enable_basic,
        enable_special,
        enable_collections,
        enable_introspection,
        enable_hashable,
        enable_string,
        max_tuple_length,
        max_depth,
        depth=0,
    )


def _serialize_with_config(
    obj: Any,
    config: Optional[Any],
    enable_basic: bool,
    enable_special: bool,
    enable_collections: bool,
    enable_introspection: bool,
    enable_hashable: bool,
    enable_string: bool,
    max_tuple_length: int,
    max_depth: int,
    depth: int = 0,
) -> str:
    """Internal serialization with configuration parameters."""

    # Prevent infinite recursion
    if depth > max_depth:
        return f"max_depth_exceeded:{type(obj).__name__}"

    # 1. Handle basic immutable types directly
    if enable_basic:
        if obj is None:
            return "None"

        if isinstance(obj, (str, int, float, bool, bytes)):
            return f"{type(obj).__name__}:{obj}"

    # 2. Special cases with custom handling (highest quality)
    if enable_special:
        # NumPy arrays and other scientific types
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            array_result = _serialize_numpy_array(obj)
            if array_result:
                return array_result

        # Path object handling - hash the content or path string
        if hasattr(obj, "__fspath__") or str(type(obj).__name__) in (
            "Path",
            "PosixPath",
            "WindowsPath",
        ):
            path_result = _serialize_path_object(obj, config)
            if path_result:
                return path_result

        # DataFrame handling (Pandas and Polars)
        if hasattr(obj, "shape") and hasattr(obj, "columns"):
            df_result = _serialize_dataframe(obj)
            if df_result:
                return df_result

        # Series handling (Pandas and Polars)
        if (
            hasattr(obj, "dtype")
            and hasattr(obj, "__len__")
            and not hasattr(obj, "shape")
        ):
            series_result = _serialize_series(obj)
            if series_result:
                return series_result

    # 3. Collections (recursive - provides good introspection)
    if enable_collections:
        if isinstance(obj, list):
            items = [
                _serialize_with_config(
                    item,
                    config,
                    enable_basic,
                    enable_special,
                    enable_collections,
                    enable_introspection,
                    enable_hashable,
                    enable_string,
                    max_tuple_length,
                    max_depth,
                    depth + 1,
                )
                for item in obj
            ]
            return f"list:[{','.join(items)}]"

        if isinstance(obj, dict):
            # Sort by keys for deterministic ordering
            items = [
                f"{
                    _serialize_with_config(
                        k,
                        config,
                        enable_basic,
                        enable_special,
                        enable_collections,
                        enable_introspection,
                        enable_hashable,
                        enable_string,
                        max_tuple_length,
                        max_depth,
                        depth + 1,
                    )
                }:"
                f"{
                    _serialize_with_config(
                        v,
                        config,
                        enable_basic,
                        enable_special,
                        enable_collections,
                        enable_introspection,
                        enable_hashable,
                        enable_string,
                        max_tuple_length,
                        max_depth,
                        depth + 1,
                    )
                }"
                for k, v in sorted(obj.items(), key=lambda x: str(x[0]))
            ]
            return f"dict:[{','.join(items)}]"

        if isinstance(obj, set):
            # Sort for deterministic ordering
            items = [
                _serialize_with_config(
                    item,
                    config,
                    enable_basic,
                    enable_special,
                    enable_collections,
                    enable_introspection,
                    enable_hashable,
                    enable_string,
                    max_tuple_length,
                    max_depth,
                    depth + 1,
                )
                for item in sorted(obj, key=str)
            ]
            return f"set:[{','.join(items)}]"

        # Special handling for tuples (common case)
        if isinstance(obj, tuple):
            # For small tuples, use recursive; for large ones, use hash
            if len(obj) <= max_tuple_length:
                items = [
                    _serialize_with_config(
                        item,
                        config,
                        enable_basic,
                        enable_special,
                        enable_collections,
                        enable_introspection,
                        enable_hashable,
                        enable_string,
                        max_tuple_length,
                        max_depth,
                        depth + 1,
                    )
                    for item in obj
                ]
                return f"tuple:[{','.join(items)}]"
            else:
                # Large tuple - fall through to hashable handling
                pass

    # 4. Objects with __dict__ (good introspection for custom classes)
    if enable_introspection:
        if hasattr(obj, "__dict__") and obj.__dict__:
            dict_repr = _serialize_with_config(
                obj.__dict__,
                config,
                enable_basic,
                enable_special,
                enable_collections,
                enable_introspection,
                enable_hashable,
                enable_string,
                max_tuple_length,
                max_depth,
                depth + 1,
            )
            return f"{type(obj).__name__}:{dict_repr}"

    # 5. Hashable objects (performance fallback when introspection isn't useful)
    if enable_hashable:
        if hasattr(obj, "__hash__") and obj.__hash__ is not None:
            try:
                # Test if the object is actually hashable
                hash_value = hash(obj)
                return f"hashed:{type(obj).__name__}:{hash_value}"
            except (TypeError, ValueError):
                pass  # Fall through to string representation

    # 6. Final fallback: string representation
    if enable_string:
        return f"{type(obj).__name__}:{str(obj)}"

    # If all methods are disabled, this is an error case
    return f"no_serialization_method:{type(obj).__name__}"


def create_unified_cache_key(params: dict, config: Optional[Any] = None) -> str:
    """
    Create a cache key from parameters using unified serialization.

    Args:
        params: Dictionary of parameters
        config: Optional CacheConfig to control serialization behavior

    Returns:
        16-character hex string cache key
    """
    # Serialize all parameters
    param_strings = []
    for key, value in sorted(params.items()):
        serialized_value = serialize_for_cache_key(value, config)
        param_strings.append(f"{key}:{serialized_value}")

    # Join and hash
    combined = "|".join(param_strings)
    hash_value = xxhash.xxh3_64(combined.encode()).hexdigest()
    return hash_value[:16]
