"""
High-Performance JSON Utilities
===============================

Provides optimized JSON serialization using orjson when available,
falling back to the built-in json library for compatibility.

orjson provides 2-5x performance improvements over built-in json
and handles datetime objects natively.
"""

import logging
from typing import Any, Union

logger = logging.getLogger(__name__)

try:
    import orjson

    def dumps(obj: Any, sort_keys: bool = False, default: Any = None) -> str:
        """
        Serialize object to JSON string using orjson.

        Args:
            obj: Object to serialize
            sort_keys: Whether to sort dictionary keys (for consistent hashing)
            default: Function to handle non-serializable objects (orjson handles most natively)

        Returns:
            JSON string
        """
        option = 0
        if sort_keys:
            option |= orjson.OPT_SORT_KEYS

        # orjson returns bytes, decode to string for compatibility
        return orjson.dumps(obj, option=option, default=default).decode("utf-8")

    def loads(s: Union[str, bytes]) -> Any:
        """
        Deserialize JSON string using orjson.

        Args:
            s: JSON string or bytes to deserialize

        Returns:
            Deserialized object
        """
        return orjson.loads(s)

    JSON_BACKEND = "orjson"
    logger.debug("🚀 Using orjson for high-performance JSON operations")

except ImportError:
    # Fallback to built-in json
    import json

    def dumps(obj: Any, sort_keys: bool = False, default: Any = None) -> str:
        """
        Serialize object to JSON string using built-in json.

        Args:
            obj: Object to serialize
            sort_keys: Whether to sort dictionary keys
            default: Function to handle non-serializable objects

        Returns:
            JSON string
        """
        return json.dumps(obj, sort_keys=sort_keys, default=default)

    def loads(s: Union[str, bytes]) -> Any:
        """
        Deserialize JSON string using built-in json.

        Args:
            s: JSON string to deserialize

        Returns:
            Deserialized object
        """
        return json.loads(s)

    JSON_BACKEND = "builtin"
    logger.debug("📝 Using built-in json library")


def get_json_backend() -> str:
    """Get the currently active JSON backend name."""
    return JSON_BACKEND
