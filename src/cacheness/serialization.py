"""
Unified Serialization for Cache Keys - IMPROVED ORDER
=====================================================

This module provides consistent serialization for cache key generation
across both UnifiedCache and decorator usage with optimal ordering strategy.
"""

from typing import Any
import xxhash


def serialize_for_cache_key(obj: Any) -> str:
    """
    Serialize an object for cache key generation with optimal ordering.
    
    Prioritization Strategy:
    1. Basic immutable types (direct representation)
    2. Special cases with custom handling (highest quality)
    3. Collections (recursive - good introspection)  
    4. Objects with __dict__ (introspection)
    5. Hashable objects (performance fallback)
    6. String representation (last resort)
    
    Args:
        obj: Object to serialize
        
    Returns:
        String representation suitable for cache key generation
    """
    # 1. Handle basic immutable types directly
    if obj is None:
        return "None"
    
    if isinstance(obj, (str, int, float, bool, bytes)):
        return f"{type(obj).__name__}:{obj}"
    
    # 2. Special cases with custom handling (highest quality)
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
    if isinstance(obj, list):
        items = [serialize_for_cache_key(item) for item in obj]
        return f"list:[{','.join(items)}]"
    
    if isinstance(obj, dict):
        # Sort by keys for deterministic ordering
        items = [
            f"{serialize_for_cache_key(k)}:{serialize_for_cache_key(v)}"
            for k, v in sorted(obj.items(), key=lambda x: str(x[0]))
        ]
        return f"dict:[{','.join(items)}]"
    
    if isinstance(obj, set):
        # Sort for deterministic ordering
        items = [serialize_for_cache_key(item) for item in sorted(obj, key=str)]
        return f"set:[{','.join(items)}]"
    
    # 4. Objects with __dict__ (good introspection for custom classes)
    if hasattr(obj, "__dict__") and obj.__dict__:
        return f"{type(obj).__name__}:{serialize_for_cache_key(obj.__dict__)}"
    
    # 5. Hashable objects (performance fallback when introspection isn't useful)
    # Special handling for tuples (common case)
    if isinstance(obj, tuple):
        # For small tuples, use recursive; for large ones, use hash
        if len(obj) <= 10:  # Configurable threshold
            items = [serialize_for_cache_key(item) for item in obj]
            return f"tuple:[{','.join(items)}]"
        else:
            # Large tuple - use hash for performance
            try:
                return f"hashed:tuple:{hash(obj)}"
            except (TypeError, ValueError):
                pass  # Fall through
    
    # General hashable objects (when introspection isn't available/useful)
    if hasattr(obj, '__hash__') and obj.__hash__ is not None:
        try:
            # Test if the object is actually hashable
            hash_value = hash(obj)
            return f"hashed:{type(obj).__name__}:{hash_value}"
        except (TypeError, ValueError):
            pass  # Fall through to string representation
    
    # 6. Final fallback: string representation
    return f"{type(obj).__name__}:{str(obj)}"


def create_unified_cache_key(params: dict) -> str:
    """
    Create a cache key from parameters using unified serialization.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        16-character hex string cache key
    """
    # Serialize all parameters
    param_strings = []
    for key, value in sorted(params.items()):
        serialized_value = serialize_for_cache_key(value)
        param_strings.append(f"{key}:{serialized_value}")
    
    # Join and hash
    combined = "|".join(param_strings)
    hash_value = xxhash.xxh3_64(combined.encode()).hexdigest()
    return hash_value[:16]
