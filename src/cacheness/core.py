"""
Simplified Unified Cache with Handler-based Architecture
=======================================================

This module provides a cleaner, more maintainable cache system using the Strategy pattern.
The main UnifiedCache class is now focused on coordination and delegates format-specific
operations to specialized handlers.
"""

import xxhash
import threading
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List

from .config import CacheConfig, _DEFAULT_TTL, create_cache_config
from .handlers import HandlerRegistry
from .serialization import create_unified_cache_key

logger = logging.getLogger(__name__)


class UnifiedCache:
    """
    Simplified unified caching system using the Strategy pattern.

    This class focuses on coordination and delegates format-specific operations
    to specialized handlers for better maintainability and extensibility.
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        metadata_backend=None,
    ):
        """
        Initialize the unified cache system.

        Args:
            config: CacheConfig object (uses defaults if None)
            metadata_backend: Optional metadata backend instance (if None, creates based on config)
        """
        # Use provided config or create default
        self.config = config or CacheConfig()

        self.cache_dir = Path(self.config.storage.cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Thread safety
        self._lock = threading.Lock()

        # Initialize handler registry with config
        self.handlers = HandlerRegistry(self.config)

        # Initialize metadata backend
        self._init_metadata_backend(metadata_backend)

        # Clean up expired entries on initialization
        if self.config.storage.cleanup_on_init:
            self._cleanup_expired()

        logger.info(
            f"✅ Unified cache initialized: {self.cache_dir} (backend: {self.actual_backend})"
        )

    def _init_metadata_backend(self, metadata_backend):
        """Initialize the metadata backend."""
        actual_backend = "unknown"  # Default value
        if metadata_backend is not None:
            self.metadata_backend = metadata_backend
            actual_backend = "custom"
        else:
            # Import here to avoid circular imports
            from .metadata import create_metadata_backend, SQLALCHEMY_AVAILABLE

        # Determine backend based on config and availability
        if self.config.metadata.metadata_backend == "json":
            # Explicitly requested JSON
            self.metadata_backend = create_metadata_backend(
                "json", metadata_file=self.cache_dir / "cache_metadata.json"
            )
            actual_backend = "json"
        elif self.config.metadata.metadata_backend == "sqlite":
            # Explicitly requested SQLite
            if not SQLALCHEMY_AVAILABLE:
                raise ImportError(
                    "SQLAlchemy is required for SQLite backend but is not available. Install with: uv add sqlalchemy"
                )
            self.metadata_backend = create_metadata_backend(
                "sqlite",
                db_file=str(self.cache_dir / self.config.metadata.sqlite_db_file),
            )
            actual_backend = "sqlite"
        else:
            # Auto mode: prefer SQLite, fallback to JSON
            if SQLALCHEMY_AVAILABLE:
                try:
                    self.metadata_backend = create_metadata_backend(
                        "sqlite",
                        db_file=str(
                            self.cache_dir / self.config.metadata.sqlite_db_file
                        ),
                    )
                    actual_backend = "sqlite"
                    logger.info(
                        "🗄️  Using SQLite backend (auto-selected for better performance)"
                    )
                except Exception as e:
                    logger.warning(f"SQLite backend failed, falling back to JSON: {e}")
                    self.metadata_backend = create_metadata_backend(
                        "json", metadata_file=self.cache_dir / "cache_metadata.json"
                    )
                    actual_backend = "json"
            else:
                logger.info("📝 SQLModel not available, using JSON backend")
                self.metadata_backend = create_metadata_backend(
                    "json", metadata_file=self.cache_dir / "cache_metadata.json"
                )
                actual_backend = "json"  # Store the actual backend used for reporting
        self.actual_backend = actual_backend

    def _create_cache_key(self, params: Dict) -> str:
        """
        Create cache key using unified serialization approach.

        Uses the unified serialization system that:
        - Handles Path objects with content hashing based on config
        - Leverages __hash__() when available for hashable objects
        - Provides consistent behavior with decorators
        - Falls back gracefully for complex objects

        Args:
            params: Dictionary of parameters to hash

        Returns:
            16-character hex string cache key
        """
        # Use unified cache key generation with config
        # Path objects will be handled by the serialization system
        return create_unified_cache_key(params, self.config)

    def _get_cache_file_path(self, cache_key: str, prefix: str = "") -> Path:
        """Get base cache file path (without extension)."""
        if prefix:
            filename_base = f"{prefix}_{cache_key}"
        else:
            filename_base = cache_key

        return self.cache_dir / filename_base

    def _is_expired(self, cache_key: str, ttl_hours=_DEFAULT_TTL) -> bool:
        """Check if cache entry is expired."""
        entry = self.metadata_backend.get_entry(cache_key)
        if not entry:
            return True

        # Handle infinite TTL: if ttl_hours is explicitly None, never expire
        if ttl_hours is None:
            return False  # Never expires
        elif ttl_hours is _DEFAULT_TTL:
            ttl = self.config.metadata.default_ttl_hours
        else:
            ttl = ttl_hours

        # Type guard to ensure ttl is numeric
        assert isinstance(ttl, (int, float)), f"TTL must be numeric, got {type(ttl)}"

        creation_time_str = entry["created_at"]

        # Handle timezone-aware datetime strings
        if isinstance(creation_time_str, str):
            creation_time = datetime.fromisoformat(creation_time_str)
        else:
            creation_time = creation_time_str

        # Ensure both datetimes are timezone-aware
        if creation_time.tzinfo is None:
            creation_time = creation_time.replace(tzinfo=timezone.utc)

        expiry_time = creation_time + timedelta(hours=ttl)
        current_time = datetime.now(timezone.utc)

        return current_time > expiry_time

    def _calculate_file_hash(self, file_path: Path) -> Optional[str]:
        """
        Calculate XXH3_64 hash of a cache file for integrity verification.

        Args:
            file_path: Path to the cache file

        Returns:
            Hex string of the file hash, or None if file doesn't exist or error
        """
        try:
            if not file_path.exists():
                return None

            hasher = xxhash.xxh3_64()
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate hash for {file_path}: {e}")
            return None

    def _cleanup_expired(self):
        """Remove expired cache entries."""
        ttl = self.config.metadata.default_ttl_hours
        removed_count = self.metadata_backend.cleanup_expired(ttl)

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache entries")

    def put(self, data: Any, prefix: str = "", description: str = "", **kwargs):
        """
        Store any supported data type in cache.

        Args:
            data: Data to cache (DataFrame, array, or general object)
            prefix: Descriptive prefix prepended to the cache filename
            description: Human-readable description
            **kwargs: Parameters identifying this data
        """
        # Get appropriate handler
        handler = self.handlers.get_handler(data)
        cache_key = self._create_cache_key(kwargs)
        base_file_path = self._get_cache_file_path(cache_key, prefix)

        try:
            # Use handler to store the data
            result = handler.put(data, base_file_path, self.config)

            # Calculate file hash for integrity verification (if enabled)
            file_hash = None
            if self.config.metadata.verify_cache_integrity:
                actual_path = result.get("actual_path", str(base_file_path))
                file_hash = self._calculate_file_hash(Path(actual_path))

            # Update metadata
            metadata_dict = {
                **result["metadata"],
                "prefix": prefix,
                "actual_path": result.get("actual_path", str(base_file_path)),
                "file_hash": file_hash,  # Store file hash for verification
            }

            # Only store cache_key_params if enabled in configuration
            if self.config.metadata.store_cache_key_params:
                metadata_dict["cache_key_params"] = (
                    kwargs  # Store original key-value parameters for efficient querying
                )

            entry_data = {
                "data_type": handler.data_type,
                "prefix": prefix,
                "description": description,
                "file_size": result["file_size"],
                "metadata": metadata_dict,
            }

            self.metadata_backend.put_entry(cache_key, entry_data)
            self._enforce_size_limit()

            file_size_mb = result["file_size"] / (1024 * 1024)
            format_info = f"({result['storage_format']} format)"
            logger.info(
                f"Cached {handler.data_type} {cache_key} ({file_size_mb:.3f}MB) {format_info}: {description}"
            )

        except Exception as e:
            logger.error(f"Failed to cache {handler.data_type}: {e}")
            raise

    def get(
        self,
        cache_key: Optional[str] = None,
        ttl_hours: Optional[int] = None,
        prefix: str = "",
        **kwargs,
    ) -> Optional[Any]:
        """
        Retrieve any supported data type from cache.

        Args:
            cache_key: Direct cache key (if provided, **kwargs are ignored)
            ttl_hours: Custom TTL (overrides default)
            prefix: Descriptive prefix prepended to the cache filename
            **kwargs: Parameters identifying the cached data (used if cache_key is None)

        Returns:
            Cached data or None if not found/expired
        """
        if cache_key is None:
            cache_key = self._create_cache_key(kwargs)

        # Check if entry exists and is not expired
        entry = self.metadata_backend.get_entry(cache_key)
        if not entry or self._is_expired(cache_key, ttl_hours):
            self.metadata_backend.increment_misses()
            return None

        # Get appropriate handler
        data_type = entry.get("data_type")
        if not data_type:
            self.metadata_backend.increment_misses()
            return None

        try:
            handler = self.handlers.get_handler_by_type(data_type)
            base_file_path = self._get_cache_file_path(cache_key, prefix)

            # Use actual path from metadata if available, otherwise use base path
            metadata = entry.get("metadata", {})
            actual_path = metadata.get("actual_path")
            if actual_path:
                file_path = Path(actual_path)
            else:
                file_path = base_file_path

            # Verify cache file integrity if enabled and hash is available
            if self.config.metadata.verify_cache_integrity:
                stored_hash = metadata.get("file_hash")
                if stored_hash is not None:
                    current_hash = self._calculate_file_hash(file_path)
                    if current_hash != stored_hash:
                        logger.warning(
                            f"Cache integrity verification failed for {cache_key}: "
                            f"stored hash {stored_hash} != current hash {current_hash}. "
                            f"Removing corrupted cache entry."
                        )
                        self.metadata_backend.remove_entry(cache_key)
                        self.metadata_backend.increment_misses()
                        return None

            # Use handler to load the data
            data = handler.get(file_path, metadata)

            # Update access time
            self.metadata_backend.update_access_time(cache_key)
            self.metadata_backend.increment_hits()

            logger.debug(f"Cache hit ({data_type}): {cache_key}")
            return data

        except Exception as e:
            logger.warning(f"Failed to load cached {data_type} {cache_key}: {e}")
            self.metadata_backend.remove_entry(cache_key)
            self.metadata_backend.increment_misses()
            return None

    def _enforce_size_limit(self):
        """Enforce cache size limits using LRU eviction."""
        # Get current total size from metadata backend
        stats = self.metadata_backend.get_stats()
        total_size_mb = stats.get("total_size_mb", 0)

        if total_size_mb <= self.config.storage.max_cache_size_mb:
            return

        # Use metadata backend's cleanup functionality
        target_size = (
            self.config.storage.max_cache_size_mb * 0.8
        )  # Clean to 80% of limit
        removed_count = self.metadata_backend.cleanup_by_size(target_size)

        if removed_count > 0:
            logger.info(f"Cache size enforcement: removed {removed_count} entries")

    def invalidate(self, cache_key: Optional[str] = None, prefix: str = "", **kwargs):
        """
        Invalidate (remove) specific cache entries.

        Args:
            cache_key: Direct cache key (if provided, **kwargs are ignored)
            prefix: Descriptive prefix of the cache filename
            **kwargs: Parameters identifying the cached data (used if cache_key is None)
        """
        if cache_key is None:
            cache_key = self._create_cache_key(kwargs)

        # Remove from metadata backend (handles file cleanup)
        if self.metadata_backend.remove_entry(cache_key):
            logger.info(f"Invalidated cache entry {cache_key}")
        else:
            logger.debug(f"Cache entry {cache_key} not found for invalidation")

    def clear_all(self):
        """Clear all cache entries."""
        removed_count = self.metadata_backend.clear_all()
        logger.info(f"Cleared {removed_count} cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.metadata_backend.get_stats()

        # Add cache-specific information
        stats.update(
            {
                "cache_dir": str(self.cache_dir),
                "max_size_mb": self.config.storage.max_cache_size_mb,
                "default_ttl_hours": self.config.metadata.default_ttl_hours,
                "backend_type": self.actual_backend,  # Report actual backend used
            }
        )

        return stats

    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries with metadata."""
        entries = self.metadata_backend.list_entries()

        # Add expiration status for each entry
        for entry in entries:
            entry["expired"] = self._is_expired(entry["cache_key"])

        return entries


# Global cache instance for convenience
_global_cache: Optional[UnifiedCache] = None


def get_cache(
    config: Optional[CacheConfig] = None, metadata_backend=None
) -> UnifiedCache:
    """Get the global cache instance, creating it if necessary."""
    global _global_cache
    if _global_cache is None:
        _global_cache = UnifiedCache(config, metadata_backend)
    return _global_cache


def reset_cache(config: Optional[CacheConfig] = None, metadata_backend=None):
    """Reset the global cache instance."""
    global _global_cache
    _global_cache = UnifiedCache(config, metadata_backend)
