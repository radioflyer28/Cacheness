"""
Unified Serialization for Cache Keys - CONFIGURABLE ORDER
========================================================

This module provides consistent serialization for cache key generation
across both UnifiedCache and decorator usage with configurable ordering strategy.
"""

from typing import Any, Optional
import xxhash


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
        # Use config settings
        enable_basic = getattr(config, 'enable_basic_types', True)
        enable_special = getattr(config, 'enable_special_cases', True)
        enable_collections = getattr(config, 'enable_collections', True)
        enable_introspection = getattr(config, 'enable_object_introspection', True)
        enable_hashable = getattr(config, 'enable_hashable_fallback', True)
        enable_string = getattr(config, 'enable_string_fallback', True)
        max_tuple_length = getattr(config, 'max_tuple_recursive_length', 10)
        max_depth = getattr(config, 'max_collection_depth', 10)
    
    return _serialize_with_config(obj, config, enable_basic, enable_special, 
                                 enable_collections, enable_introspection,
                                 enable_hashable, enable_string, 
                                 max_tuple_length, max_depth, depth=0)


def _serialize_with_config(obj: Any, config: Optional[Any], 
                          enable_basic: bool, enable_special: bool,
                          enable_collections: bool, enable_introspection: bool,
                          enable_hashable: bool, enable_string: bool,
                          max_tuple_length: int, max_depth: int, depth: int = 0) -> str:
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
        if hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
            try:
                content_hash = xxhash.xxh3_64(obj.tobytes()).hexdigest()[:16]
                return f"array:{obj.shape}:{obj.dtype}:{content_hash}"
            except Exception:
                pass  # Fall through to other methods
        
        # TODO: Add Path object handling here if needed
        # TODO: Add DataFrame handling here if needed
    
    # 3. Collections (recursive - provides good introspection)
    if enable_collections:
        if isinstance(obj, list):
            items = [_serialize_with_config(item, config, enable_basic, enable_special,
                                          enable_collections, enable_introspection,
                                          enable_hashable, enable_string,
                                          max_tuple_length, max_depth, depth + 1) 
                    for item in obj]
            return f"list:[{','.join(items)}]"
        
        if isinstance(obj, dict):
            # Sort by keys for deterministic ordering
            items = [
                f"{_serialize_with_config(k, config, enable_basic, enable_special,
                                        enable_collections, enable_introspection,
                                        enable_hashable, enable_string,
                                        max_tuple_length, max_depth, depth + 1)}:"
                f"{_serialize_with_config(v, config, enable_basic, enable_special,
                                        enable_collections, enable_introspection,
                                        enable_hashable, enable_string,
                                        max_tuple_length, max_depth, depth + 1)}"
                for k, v in sorted(obj.items(), key=lambda x: str(x[0]))
            ]
            return f"dict:[{','.join(items)}]"
        
        if isinstance(obj, set):
            # Sort for deterministic ordering
            items = [_serialize_with_config(item, config, enable_basic, enable_special,
                                          enable_collections, enable_introspection,
                                          enable_hashable, enable_string,
                                          max_tuple_length, max_depth, depth + 1)
                    for item in sorted(obj, key=str)]
            return f"set:[{','.join(items)}]"
        
        # Special handling for tuples (common case)
        if isinstance(obj, tuple):
            # For small tuples, use recursive; for large ones, use hash
            if len(obj) <= max_tuple_length:
                items = [_serialize_with_config(item, config, enable_basic, enable_special,
                                              enable_collections, enable_introspection,
                                              enable_hashable, enable_string,
                                              max_tuple_length, max_depth, depth + 1)
                        for item in obj]
                return f"tuple:[{','.join(items)}]"
            else:
                # Large tuple - fall through to hashable handling
                pass
    
    # 4. Objects with __dict__ (good introspection for custom classes)
    if enable_introspection:
        if hasattr(obj, "__dict__") and obj.__dict__:
            dict_repr = _serialize_with_config(obj.__dict__, config, enable_basic, enable_special,
                                             enable_collections, enable_introspection,
                                             enable_hashable, enable_string,
                                             max_tuple_length, max_depth, depth + 1)
            return f"{type(obj).__name__}:{dict_repr}"
    
    # 5. Hashable objects (performance fallback when introspection isn't useful)
    if enable_hashable:
        if hasattr(obj, '__hash__') and obj.__hash__ is not None:
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
