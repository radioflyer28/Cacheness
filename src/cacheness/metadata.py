"""
Cache Metadata Backend System
=============================

This module provides pluggable metadata backends for the unified cache system,
supporting both JSON file storage and SQLite database storage with SQLAlchemy ORM.

Features:
- Abstract base class for metadata backends
- JSON file backend (original implementation)
- SQLite backend with SQLAlchemy ORM
- Dependency injection for backend selection
- Thread-safe operations
- Migration utilities

Usage:
    # Use auto backend (defaults to SQLite if available, falls back to JSON)
    cache = UnifiedCache()

    # Force SQLite backend
    from shared.cache.metadata import SqliteBackend
    backend = SqliteBackend("cache.db")
    cache = UnifiedCache(metadata_backend=backend)

    # Force JSON backend
    from shared.cache.metadata import JsonBackend
    backend = JsonBackend(Path("cache_metadata.json"))
    cache = UnifiedCache(metadata_backend=backend)
"""

import os
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable

import logging

from .json_utils import dumps as json_dumps, loads as json_loads

logger = logging.getLogger(__name__)

# --- Namespace validation ---

#: Regex pattern for valid namespace identifiers.
#: Only lowercase alphanumeric and underscores, 1-48 chars.
#: These IDs are used as table/file suffixes, so must be SQL-identifier-safe.
NAMESPACE_ID_PATTERN = re.compile(r"^[a-z0-9_]{1,48}$")

#: Default namespace ID. Maps to existing unsuffixed tables for backward compat.
DEFAULT_NAMESPACE = "default"


def validate_namespace_id(namespace_id: str) -> str:
    """Validate and return a namespace identifier.

    Namespace IDs are used as suffixes for table names and file names, so they
    must be safe SQL identifiers.  Only lowercase alphanumeric characters and
    underscores are allowed, 1-48 characters long.

    Args:
        namespace_id: The identifier to validate.

    Returns:
        The validated namespace_id (unchanged).

    Raises:
        ValueError: If the namespace_id doesn't match the required pattern.
    """
    if not isinstance(namespace_id, str):
        raise ValueError(
            f"Namespace ID must be a string, got {type(namespace_id).__name__}"
        )
    if not NAMESPACE_ID_PATTERN.match(namespace_id):
        raise ValueError(
            f"Invalid namespace ID {namespace_id!r}: must match "
            f"{NAMESPACE_ID_PATTERN.pattern} (lowercase alphanumeric + "
            f"underscore, 1-48 chars)"
        )
    return namespace_id


@dataclass
class NamespaceInfo:
    """Information about a registered namespace.

    Attributes:
        namespace_id: Unique identifier used as table/file suffix.
        display_name: Optional human-readable name.
        schema_version: Current schema version for this namespace's tables.
        created_at: When the namespace was registered.
        signature: HMAC signature for integrity verification (optional).
    """

    namespace_id: str
    display_name: str = ""
    schema_version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signature: Optional[str] = None


#: Type alias for a schema migration step.
#: Each migration is a tuple of (from_version, to_version, callable).
#: The callable receives the backend instance and namespace_id.
Migration = Tuple[int, int, Callable[["MetadataBackend", str], None]]

# Try to import cachetools for entry caching
try:
    from cachetools import TTLCache

    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    logger.debug("cachetools not available, entry caching disabled")

try:
    from sqlalchemy import (
        create_engine,
        Column,
        String,
        Integer,
        DateTime,
        Text,
        Index,
        select,
        update,
        delete,
        desc,
        func,
        text,
        inspect,
        case,
    )
    from sqlalchemy.orm import sessionmaker, declarative_base

    SQLALCHEMY_AVAILABLE = True

    # Create declarative base
    Base = declarative_base()

    class CacheEntry(Base):
        """SQLAlchemy model for cache entry metadata."""

        __tablename__ = "cache_entries"

        cache_key = Column(String(16), primary_key=True)
        description = Column(String(500), default="", nullable=False)
        # Remove individual indexes - will be covered by composite indexes
        data_type = Column(String(20), nullable=False)  # Removed index=True
        prefix = Column(String(100), default="", nullable=False)  # Removed index=True

        # Timestamps - no individual indexes, use composite indexes only
        created_at = Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        )  # Removed index=True
        accessed_at = Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        )  # Removed index=True

        # File info - no individual index, use composite
        file_size = Column(Integer, default=0, nullable=False)  # Removed index=True

        # Cache integrity verification
        file_hash = Column(String(16), nullable=True)  # XXH3_64 hash (16 hex chars)

        # Entry signature for metadata integrity protection
        entry_signature = Column(
            String(64), nullable=True
        )  # HMAC-SHA256 hex (64 chars)

        # S3 ETag for S3-backed blob storage (separate from file_hash)
        s3_etag = Column(String(100), nullable=True)  # S3 ETag (MD5 or multipart hash)

        # Backend technical metadata (previously stored in metadata_json)
        object_type = Column(String(100), nullable=True)  # e.g., "<class 'int'>"
        storage_format = Column(String(20), nullable=True)  # e.g., "pickle", "parquet"
        serializer = Column(String(20), nullable=True)  # e.g., "pickle", "dill"
        compression_codec = Column(String(20), nullable=True)  # e.g., "zstd", "lz4"
        actual_path = Column(String(500), nullable=True)  # Full path to cache file

        # Original cache key parameters - only store, don't index (rarely queried)
        # Only populated when store_full_metadata=True config option is enabled
        cache_key_params = Column(
            Text, nullable=True
        )  # JSON-serialized kwargs used for cache key

        # User metadata dict - simple key-value metadata for filtering/querying
        # Raw values stored for easy JSON_EXTRACT queries (not hashed like cache_key_params)
        metadata_dict = Column(
            Text, nullable=True
        )  # JSON dict for query_meta() filtering

        # Optimized composite indexes for actual query patterns only
        __table_args__ = (
            # PRIMARY: List entries by creation time (most common - list_entries)
            Index("idx_list_entries", desc("created_at")),
            # CLEANUP: TTL-based cleanup operations
            Index("idx_cleanup", "created_at"),
            # SIZE MANAGEMENT: Combined file size + time for cache management
            Index("idx_size_mgmt", "file_size", "created_at"),
            # STATS: Data type for statistics (if needed)
            Index("idx_data_type", "data_type"),
        )

    class CacheStats(Base):
        """SQLAlchemy model for cache statistics with minimal indexing."""

        __tablename__ = "cache_stats"

        id = Column(Integer, primary_key=True, default=1)
        # Remove individual indexes - stats table is tiny and rarely queried
        cache_hits = Column(Integer, default=0, nullable=False)  # Removed index=True
        cache_misses = Column(Integer, default=0, nullable=False)  # Removed index=True
        last_updated = Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        )  # Removed index=True

        # No composite indexes needed - single row table

    class CacheNamespace(Base):
        """SQLAlchemy model for the namespace registry.

        Tracks registered namespaces with per-namespace schema versioning.
        Each namespace maps to its own set of tables:
        - ``'default'`` → existing ``cache_entries`` / ``cache_stats``
        - other → ``cache_entries_{namespace_id}`` / ``cache_stats_{namespace_id}``
        """

        __tablename__ = "cacheness_namespaces"

        namespace_id = Column(String(48), primary_key=True)
        display_name = Column(String(200), default="", nullable=False)
        schema_version = Column(Integer, default=1, nullable=False)
        created_at = Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        )
        # HMAC signature for integrity verification (optional)
        signature = Column(String(64), nullable=True)

except ImportError:
    # SQLAlchemy not available
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not available, SQLite backend will not work")

    # Define dummy classes to avoid runtime errors
    class CacheEntry:
        pass

    class CacheStats:
        pass

    class CacheNamespace:  # type: ignore[no-redef]
        pass

    Base = None


class MetadataBackend(ABC):
    """Abstract base class for cache metadata backends.

    This is the canonical MetadataBackend interface.  Every metadata backend
    (JSON, SQLite, PostgreSQL, custom) must extend this ABC.

    The class is re-exported by ``cacheness.storage.backends.base`` so that
    ``from cacheness.storage.backends import MetadataBackend`` and
    ``from cacheness.metadata import MetadataBackend`` both resolve to the
    same type.
    """

    @abstractmethod
    def load_metadata(self) -> Dict[str, Any]:
        """Load complete metadata structure."""
        pass

    @abstractmethod
    def save_metadata(self, metadata: Dict[str, Any]):
        """Save complete metadata structure."""
        pass

    @abstractmethod
    def get_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get specific cache entry metadata (internal storage format).

        Returns a dict with internal field names:
            description, data_type, prefix, created_at, accessed_at,
            file_size (bytes), metadata (nested dict)

        See also: list_entries() returns user-facing format with different keys.
        """
        pass

    @abstractmethod
    def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
        """Store cache entry metadata."""
        pass

    @abstractmethod
    def remove_entry(self, cache_key: str) -> bool:
        """Remove cache entry metadata.

        Returns:
            bool: True if the entry existed and was removed, False if not found.
        """
        pass

    @abstractmethod
    def update_entry_metadata(self, cache_key: str, updates: Dict[str, Any]) -> bool:
        """Update metadata fields for an existing cache entry.

        Updates derived metadata fields (file_size, content_hash, timestamps,
        data_type, etc.) after blob data has been written by the cache layer.
        The cache_key remains immutable.

        This method only updates metadata — blob I/O is handled by
        UnifiedCache.update_data() before calling this method.

        Args:
            cache_key: The unique identifier for the cache entry to update
            updates: Dict of metadata fields to update. Expected keys include:
                - file_size (int): New file size in bytes
                - file_hash (str): New file hash
                - content_hash (str): New content hash
                - actual_path (str): New actual file path
                - storage_format (str): New storage format
                - data_type (str): New data type identifier
                - serializer (str): Serializer used
                - compression_codec (str): Compression codec used
                - object_type (str): Object type identifier

        Returns:
            bool: True if entry was updated, False if entry doesn't exist
        """
        pass

    @abstractmethod
    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries with metadata (user-facing format).

        Returns a list of dicts with user-facing field names:
            cache_key, data_type, description, metadata (nested dict),
            created (ISO string), last_accessed (ISO string), size_mb (float)

        Note: Field names differ from get_entry() — this format is used by
        delete_where(), delete_matching(), and UnifiedCache.list_entries().
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass

    @abstractmethod
    def update_access_time(self, cache_key: str):
        """Update last access time for cache entry."""
        pass

    @abstractmethod
    def increment_hits(self):
        """Increment cache hits counter."""
        pass

    @abstractmethod
    def increment_misses(self):
        """Increment cache misses counter."""
        pass

    @abstractmethod
    def cleanup_expired(self, ttl_seconds: float) -> int:
        """Remove expired entries and return count removed."""
        pass

    @abstractmethod
    def cleanup_by_size(self, target_size_mb: float) -> Dict[str, Any]:
        """Remove least-recently-accessed entries until cache size drops to or below target.

        Args:
            target_size_mb: Target cache size in megabytes

        Returns:
            Dict with 'count' (int) and 'removed_entries' (list of dicts with 'cache_key' and 'actual_path')
        """
        pass

    @abstractmethod
    def clear_all(self) -> int:
        """Remove all cache entries and return count removed."""
        pass

    def iter_entry_summaries(self) -> List[Dict[str, Any]]:
        """Return lightweight entry summaries for internal filtering.

        Each dict contains flat, unprocessed column values:
            cache_key, data_type, description, created_at (raw),
            accessed_at (raw), file_size (bytes int), plus any
            backend-specific technical fields (object_type,
            storage_format, serializer, compression_codec,
            actual_path, file_hash, entry_signature, prefix).

        Unlike list_entries(), this method:
        - Skips ORM hydration on SQL backends (uses raw SELECT)
        - Skips nested metadata dict construction
        - Skips isoformat() conversion on timestamps
        - Skips cache_key_params JSON parsing
        - Skips size_mb calculation

        The returned dicts are flat (no nested 'metadata' key) so callers
        can match against any field with simple ``entry.get(k) == v``.

        Default implementation falls back to list_entries() for custom
        backends that haven't overridden this method.
        """
        # Fallback: flatten list_entries() output for custom backends
        result = []
        for entry in self.list_entries():
            flat = {k: v for k, v in entry.items() if k != "metadata"}
            flat.update(entry.get("metadata", {}))
            result.append(flat)
        return result

    # --- Schema versioning ---

    def get_schema_version(self, namespace_id: str = DEFAULT_NAMESPACE) -> int:
        """Get the current schema version for a namespace.

        Backends that support schema versioning should override this to read
        the version from their persistent registry.  The default returns 0,
        meaning "no version tracked yet" (pre-versioning database).

        Args:
            namespace_id: The namespace to query.

        Returns:
            The current schema version integer, or 0 if untracked.
        """
        return 0

    def set_schema_version(self, namespace_id: str, version: int) -> None:
        """Set the schema version for a namespace.

        Backends that support schema versioning should override this to
        persist the version in their registry.  The default is a no-op.

        Args:
            namespace_id: The namespace to update.
            version: The new schema version.
        """
        pass

    def get_migrations(self) -> List[Migration]:
        """Return the ordered list of schema migrations for this backend.

        Each migration is a ``(from_version, to_version, callable)`` tuple.
        Migrations are applied sequentially: only migrations whose
        ``from_version`` matches the current version are executed.

        Subclasses should override this to provide backend-specific
        migrations.  The default returns an empty list (no migrations).

        Returns:
            List of ``(from_version, to_version, callable)`` tuples.
        """
        return []

    def run_migrations(self, namespace_id: str = DEFAULT_NAMESPACE) -> int:
        """Run pending schema migrations for a namespace.

        Finds the current schema version, then applies each migration whose
        ``from_version`` matches, in order.  After each successful migration
        the version is updated.

        Args:
            namespace_id: The namespace to migrate.

        Returns:
            The final schema version after all migrations.
        """
        current = self.get_schema_version(namespace_id)
        for from_ver, to_ver, migrate_fn in self.get_migrations():
            if current == from_ver:
                logger.info(
                    "Migrating namespace %r schema v%d -> v%d",
                    namespace_id,
                    from_ver,
                    to_ver,
                )
                migrate_fn(self, namespace_id)
                self.set_schema_version(namespace_id, to_ver)
                current = to_ver
        return current

    # --- Namespace registry ---

    def create_namespace(
        self,
        namespace_id: str,
        display_name: str = "",
    ) -> NamespaceInfo:
        """Register a new namespace and create its backing tables/files.

        The ``namespace_id`` must pass :func:`validate_namespace_id`.  The
        ``'default'`` namespace is pre-registered (maps to existing unsuffixed
        tables) and cannot be created again.

        Subclasses that support namespaces must override this method to
        create per-namespace tables/files and insert a registry row.

        Args:
            namespace_id: Unique identifier (``^[a-z0-9_]{1,48}$``).
            display_name: Optional human-readable name.

        Returns:
            A :class:`NamespaceInfo` describing the new namespace.

        Raises:
            ValueError: If *namespace_id* is invalid or already exists.
            NotImplementedError: If the backend does not support namespaces.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support namespaces")

    def drop_namespace(self, namespace_id: str) -> bool:
        """Remove a namespace and all its data (tables, files, entries).

        The ``'default'`` namespace cannot be dropped.

        Subclasses that support namespaces must override this method to
        drop per-namespace tables/files and remove the registry row.

        Args:
            namespace_id: The namespace to remove.

        Returns:
            True if the namespace existed and was removed.

        Raises:
            ValueError: If attempting to drop the ``'default'`` namespace.
            NotImplementedError: If the backend does not support namespaces.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support namespaces")

    def list_namespaces(self) -> List[NamespaceInfo]:
        """List all registered namespaces.

        Subclasses that support namespaces should override this to read
        from the registry.  The default returns a single ``'default'``
        namespace to preserve backward compatibility.

        Returns:
            List of :class:`NamespaceInfo` objects.
        """
        return [
            NamespaceInfo(
                namespace_id=DEFAULT_NAMESPACE,
                display_name="Default",
                schema_version=self.get_schema_version(DEFAULT_NAMESPACE),
            )
        ]

    def get_namespace(self, namespace_id: str) -> Optional[NamespaceInfo]:
        """Get info for a specific namespace.

        Default implementation searches :meth:`list_namespaces`.

        Args:
            namespace_id: The namespace to look up.

        Returns:
            :class:`NamespaceInfo` if found, else ``None``.
        """
        for ns in self.list_namespaces():
            if ns.namespace_id == namespace_id:
                return ns
        return None

    def namespace_exists(self, namespace_id: str) -> bool:
        """Check whether a namespace is registered.

        Default implementation delegates to :meth:`get_namespace`.
        """
        return self.get_namespace(namespace_id) is not None

    def close(self):
        """Close and clean up any resources (default implementation does nothing)."""
        pass


def create_entry_cache(cache_type: str, maxsize: int, ttl_seconds: float):
    """Create a TTLCache for the memory cache layer.

    Uses cachetools.TTLCache which provides LRU eviction combined with
    time-based expiration. The ``cache_type`` parameter is accepted for
    forward-compatibility but currently only ``"lru"`` (the default) is
    meaningfully distinct — all types create a TTLCache with LRU eviction.

    Args:
        cache_type: Cache eviction strategy name (currently all map to TTLCache).
        maxsize: Maximum number of entries in the memory cache.
        ttl_seconds: Time-to-live for cached entries in seconds.

    Returns:
        TTLCache instance, or None if cachetools is not installed.
    """
    if not CACHETOOLS_AVAILABLE:
        return None

    if cache_type not in ("lru", "lfu", "fifo", "rr"):
        logger.warning(f"Unknown cache type '{cache_type}', using TTLCache (LRU)")

    # All types currently use TTLCache (LRU + TTL). cachetools does not
    # provide LFU/FIFO/RR variants with built-in TTL support.
    return TTLCache(maxsize=maxsize, ttl=ttl_seconds)


class CachedMetadataBackend(MetadataBackend):
    """Wrapper that adds memory caching layer to disk-persistent metadata backends.

    This wrapper adds an in-memory cache layer between the application and disk-persistent
    backends (JSON and SQLite) to avoid repeated disk I/O operations. The memory cache
    is completely separate from the in-memory backend - it's a caching layer on top
    of disk storage.

    Architecture:
        Application → Memory Cache Layer → Disk Backend (JSON/SQLite)

    Features:
    - Configurable cache type (LRU, LFU, FIFO, RR)
    - TTL-based expiration for cached metadata entries
    - Cache invalidation on mutations (put, remove, clear)
    - Optional cache statistics tracking
    - Only applies to disk backends, never to pure in-memory backend
    """

    def __init__(self, wrapped_backend: MetadataBackend, config):
        """Initialize memory cache layer for disk-persistent backend.

        Args:
            wrapped_backend: The underlying disk-persistent metadata backend to wrap
            config: CacheMetadataConfig with memory cache settings
        """
        self.backend = wrapped_backend
        self.config = config
        self._lock = threading.RLock()

        # Initialize memory cache layer if cachetools is available and enabled
        if CACHETOOLS_AVAILABLE and config.enable_memory_cache:
            self._memory_cache = create_entry_cache(
                config.memory_cache_type,
                config.memory_cache_maxsize,
                config.memory_cache_ttl_seconds,
            )

            # Optional cache statistics
            if config.memory_cache_stats:
                self._cache_hits = 0
                self._cache_misses = 0
            else:
                self._cache_hits = None
                self._cache_misses = None

            logger.info(
                f"🚀 Memory cache layer enabled: {config.memory_cache_type} "
                f"(maxsize={config.memory_cache_maxsize}, ttl={config.memory_cache_ttl_seconds}s)"
            )
        else:
            self._memory_cache = None
            self._cache_hits = None
            self._cache_misses = None

            if not CACHETOOLS_AVAILABLE:
                logger.warning(
                    "cachetools not available, memory cache layer disabled. Install with: pip install cachetools"
                )

    def _cache_key_for_entry(self, cache_key: str) -> str:
        """Create cache key for memory cache layer."""
        return f"entry:{cache_key}"

    def get_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get specific cache entry metadata with memory cache layer."""
        if self._memory_cache is None:
            return self.backend.get_entry(cache_key)

        with self._lock:
            entry_cache_key = self._cache_key_for_entry(cache_key)

            # Try memory cache first
            cached_entry = self._memory_cache.get(entry_cache_key)
            if cached_entry is not None:
                if self._cache_hits is not None:
                    self._cache_hits += 1
                logger.debug(f"Memory cache hit: {cache_key}")
                return cached_entry

            # Cache miss - load from disk backend
            if self._cache_misses is not None:
                self._cache_misses += 1

            entry = self.backend.get_entry(cache_key)
            if entry is not None:
                # Cache the result in memory
                self._memory_cache[entry_cache_key] = entry
                logger.debug(f"Entry cached in memory: {cache_key}")

            return entry

    def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
        """Store cache entry metadata and update memory cache."""
        # Always call disk backend first
        self.backend.put_entry(cache_key, entry_data)

        # Update memory cache if enabled
        if self._memory_cache is not None:
            with self._lock:
                entry_cache_key = self._cache_key_for_entry(cache_key)
                # Store the new entry in memory cache (get_entry format)
                formatted_entry = self.backend.get_entry(cache_key)
                if formatted_entry is not None:
                    self._memory_cache[entry_cache_key] = formatted_entry
                    logger.debug(f"Memory cache updated: {cache_key}")

    def remove_entry(self, cache_key: str) -> bool:
        """Remove cache entry metadata and invalidate memory cache."""
        # Remove from disk backend first
        removed = self.backend.remove_entry(cache_key)

        # Remove from memory cache if enabled
        if self._memory_cache is not None:
            with self._lock:
                entry_cache_key = self._cache_key_for_entry(cache_key)
                self._memory_cache.pop(entry_cache_key, None)
                logger.debug(f"Memory cache invalidated: {cache_key}")

        return removed

    def update_entry_metadata(self, cache_key: str, updates: Dict[str, Any]) -> bool:
        """Update entry metadata and invalidate memory cache entry."""
        result = self.backend.update_entry_metadata(cache_key, updates)

        # Invalidate memory cache entry so next get_entry() fetches fresh metadata
        if result and self._memory_cache is not None:
            with self._lock:
                entry_cache_key = self._cache_key_for_entry(cache_key)
                self._memory_cache.pop(entry_cache_key, None)
                logger.debug(
                    f"Memory cache invalidated after metadata update: {cache_key}"
                )

        return result

    def clear_all(self) -> int:
        """Remove all cache entries and clear memory cache."""
        count = self.backend.clear_all()

        # Clear memory cache if enabled
        if self._memory_cache is not None:
            with self._lock:
                self._memory_cache.clear()
                logger.debug("Memory cache cleared")

        return count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get memory cache layer statistics."""
        if (
            self._memory_cache is None
            or self._cache_hits is None
            or self._cache_misses is None
        ):
            return {}

        with self._lock:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

            return {
                "memory_cache_enabled": True,
                "memory_cache_type": self.config.memory_cache_type,
                "memory_cache_size": len(self._memory_cache),
                "memory_cache_maxsize": self.config.memory_cache_maxsize,
                "memory_cache_hits": self._cache_hits,
                "memory_cache_misses": self._cache_misses,
                "memory_cache_hit_rate": round(hit_rate, 3),
            }

    # Delegate all other methods to the wrapped backend
    def load_metadata(self) -> Dict[str, Any]:
        return self.backend.load_metadata()

    def save_metadata(self, metadata: Dict[str, Any]):
        return self.backend.save_metadata(metadata)

    def list_entries(self) -> List[Dict[str, Any]]:
        return self.backend.list_entries()

    def iter_entry_summaries(self) -> List[Dict[str, Any]]:
        return self.backend.iter_entry_summaries()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics with optional entry cache stats."""
        stats = self.backend.get_stats()

        # Add entry cache stats if available
        cache_stats = self.get_cache_stats()
        if cache_stats:
            stats.update(cache_stats)

        return stats

    def update_access_time(self, cache_key: str):
        # Update backend access time
        self.backend.update_access_time(cache_key)

        # Invalidate memory cache entry to force refresh with new access time
        if self._memory_cache is not None:
            with self._lock:
                entry_cache_key = self._cache_key_for_entry(cache_key)
                self._memory_cache.pop(entry_cache_key, None)

    def increment_hits(self):
        return self.backend.increment_hits()

    def increment_misses(self):
        return self.backend.increment_misses()

    def cleanup_expired(self, ttl_seconds: float) -> int:
        count = self.backend.cleanup_expired(ttl_seconds)

        # Clear entire memory cache after cleanup (entries might be stale)
        if self._memory_cache is not None and count > 0:
            with self._lock:
                self._memory_cache.clear()
                logger.debug("Memory cache cleared after expired cleanup")

        return count

    def cleanup_by_size(self, target_size_mb: float) -> Dict[str, Any]:
        """Delegate cleanup_by_size to wrapped backend and clear memory cache."""
        result = self.backend.cleanup_by_size(target_size_mb)

        # Clear entire memory cache after cleanup (entries might be stale)
        removed_count = result.get("count", 0)
        if self._memory_cache is not None and removed_count > 0:
            with self._lock:
                self._memory_cache.clear()
                logger.debug("Memory cache cleared after size-based cleanup")

        return result

    def close(self):
        """Close and clean up resources including the wrapped backend."""
        # Clear memory cache if it exists
        if self._memory_cache is not None:
            with self._lock:
                self._memory_cache.clear()
                logger.debug("Memory cache cleared during close")

        # Close the wrapped backend
        if hasattr(self.backend, "close"):
            self.backend.close()


class JsonBackend(MetadataBackend):
    """JSON file-based metadata backend with batching support."""

    def __init__(self, metadata_file: Path):
        """
        Initialize JSON metadata backend.

        Args:
            metadata_file: Path to JSON metadata file
        """
        self.metadata_file = metadata_file
        self._lock = (
            threading.RLock()
        )  # Use RLock to allow reentrant calls (cleanup_by_size -> get_stats)
        self._metadata = self._load_from_disk()

        # --- Namespace registry ---
        # Registry lives in the same directory as the metadata file.
        self._registry_file = self.metadata_file.parent / "cacheness_namespaces.json"
        self._ensure_namespace_registry()

    def _load_from_disk(self) -> Dict[str, Any]:
        """Load metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json_loads(f.read())
                # Validate schema: must be a dict with an "entries" key
                if not isinstance(data, dict) or "entries" not in data:
                    logger.warning("JSON metadata has invalid schema, starting fresh")
                else:
                    return data
            except Exception:
                logger.warning("JSON metadata corrupted, starting fresh")

        return {
            "entries": {},  # cache_key -> complete entry dict (with structured fields)
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _save_to_disk(self):
        """Save metadata to JSON file using atomic write pattern to prevent corruption."""
        import tempfile

        try:
            # Check if parent directory exists (may have been deleted during cleanup)
            if not self.metadata_file.parent.exists():
                logger.debug(
                    f"Metadata directory no longer exists: {self.metadata_file.parent}"
                )
                return

            # Write to temp file first, then rename for atomicity
            fd, temp_path = tempfile.mkstemp(
                suffix=".json.tmp",
                dir=self.metadata_file.parent,
                prefix="cache_metadata_",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(json_dumps(self._metadata, default=str))
                # Atomic rename (works on same filesystem)
                import shutil

                shutil.move(temp_path, self.metadata_file)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.error(f"Failed to save JSON metadata: {e}")

    def get_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get specific cache entry metadata (simple entry lookup)."""
        with self._lock:
            # Simple lookup - entry contains all structured fields
            entry = self._metadata.get("entries", {}).get(cache_key)
            if entry is None:
                return None

            # Entry already contains the structured fields matching SQLite schema
            return entry

    def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
        """Store cache entry metadata as complete entry (matching SQLite schema structure)."""
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()

            # Extract and restructure metadata to match SQLite schema
            metadata = entry_data.get("metadata", {}).copy()

            # Build complete entry with structured fields (matching SQLite columns)
            entry = {
                "description": entry_data.get("description", ""),
                "data_type": entry_data.get("data_type", "unknown"),
                "prefix": entry_data.get("prefix", ""),
                "created_at": entry_data.get("created_at", now),
                "accessed_at": entry_data.get("accessed_at", now),
                "file_size": entry_data.get("file_size", 0),
                "metadata": metadata,  # Include all metadata as nested structure
            }

            # Store complete entry - simple and efficient
            self._metadata["entries"][cache_key] = entry
            self._save_to_disk()

    def remove_entry(self, cache_key: str) -> bool:
        """Remove cache entry metadata."""
        with self._lock:
            entries = self._metadata.get("entries", {})
            if cache_key in entries:
                del entries[cache_key]
                self._save_to_disk()
                return True
            return False

    def update_entry_metadata(self, cache_key: str, updates: Dict[str, Any]) -> bool:
        """
        Update metadata fields for an existing cache entry.

        Only updates metadata — blob I/O is handled by UnifiedCache.update_data().

        Args:
            cache_key: The unique identifier for the cache entry to update
            updates: Dict of metadata fields to update (file_size, file_hash,
                    actual_path, data_type, storage_format, serializer, etc.)

        Returns:
            bool: True if entry was updated, False if entry doesn't exist
        """
        with self._lock:
            entries = self._metadata.get("entries", {})
            entry = entries.get(cache_key)
            if not entry:
                return False

            # Update derived metadata (file_size, content_hash, created_at)
            now = datetime.now(timezone.utc)
            entry["created_at"] = now.isoformat()  # Reset timestamp

            if "file_size" in updates:
                entry["file_size"] = updates["file_size"]
            if "data_type" in updates:
                entry["data_type"] = updates["data_type"]
            if "storage_format" in updates:
                entry["storage_format"] = updates.get(
                    "storage_format", entry.get("storage_format")
                )
            if "serializer" in updates:
                entry["serializer"] = updates["serializer"]

            # Update metadata dict with new values
            metadata = entry.get("metadata", {})
            for key in (
                "file_size",
                "content_hash",
                "file_hash",
                "actual_path",
                "storage_format",
                "serializer",
            ):
                if key in updates:
                    metadata[key] = updates[key]
            entry["metadata"] = metadata

            # Save to disk immediately for atomicity
            self._save_to_disk()
            return True

    def iter_entry_summaries(self) -> List[Dict[str, Any]]:
        """Return lightweight flat entry dicts for internal filtering."""
        with self._lock:
            result = []
            for cache_key, entry in self._metadata.get("entries", {}).items():
                flat = {
                    "cache_key": cache_key,
                    "data_type": entry.get("data_type", "unknown"),
                    "description": entry.get("description", ""),
                    "prefix": entry.get("prefix", ""),
                    "created_at": entry.get("created_at"),
                    "accessed_at": entry.get("accessed_at"),
                    "file_size": entry.get("file_size", 0),
                }
                # Merge technical metadata fields flat
                for k, v in entry.get("metadata", {}).items():
                    if k not in flat:
                        flat[k] = v
                result.append(flat)
            return result

    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries with metadata (simple entries iteration)."""
        with self._lock:
            entries = []

            # Simple iteration over entries dict
            for cache_key, entry in self._metadata.get("entries", {}).items():
                # Ensure timestamps are timezone-aware when returned
                creation_time = entry.get("created_at")
                access_time = entry.get("accessed_at")

                if creation_time and isinstance(creation_time, str):
                    try:
                        # Parse ISO format string back to timezone-aware datetime
                        creation_time = datetime.fromisoformat(creation_time)
                        if creation_time.tzinfo is None:
                            # If somehow it's naive, make it UTC
                            creation_time = creation_time.replace(tzinfo=timezone.utc)
                        creation_time = creation_time.isoformat()
                    except (ValueError, TypeError):
                        pass

                if access_time and isinstance(access_time, str):
                    try:
                        # Parse ISO format string back to timezone-aware datetime
                        access_time = datetime.fromisoformat(access_time)
                        if access_time.tzinfo is None:
                            # If somehow it's naive, make it UTC
                            access_time = access_time.replace(tzinfo=timezone.utc)
                        access_time = access_time.isoformat()
                    except (ValueError, TypeError):
                        pass

                # Build entry for list output
                list_entry = {
                    "cache_key": cache_key,
                    "data_type": entry.get("data_type", "unknown"),
                    "description": entry.get("description", ""),
                    "metadata": entry.get("metadata", {}),
                    "created": creation_time,
                    "last_accessed": access_time,
                    "size_mb": round(entry.get("file_size", 0) / (1024 * 1024), 3),
                }

                entries.append(list_entry)

            # Sort by creation time (newest first)
            entries.sort(key=lambda x: x["created"] or "", reverse=True)
            return entries

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics (simple entries-based counting)."""
        with self._lock:
            entries = self._metadata.get("entries", {})

            # Count total entries
            total_entries = len(entries)

            # Calculate total size
            total_size_mb = sum(
                entry.get("file_size", 0) for entry in entries.values()
            ) / (1024 * 1024)

            # Count by data type
            dataframe_count = sum(
                1 for entry in entries.values() if entry.get("data_type") == "dataframe"
            )
            array_count = sum(
                1 for entry in entries.values() if entry.get("data_type") == "array"
            )

            # Cache hit rate
            hits = self._metadata.get("cache_hits", 0)
            misses = self._metadata.get("cache_misses", 0)
            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0

            return {
                "total_entries": total_entries,
                "dataframe_entries": dataframe_count,
                "array_entries": array_count,
                "total_size_mb": total_size_mb,  # Don't round - precise size needed for cleanup calculations
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate": round(hit_rate, 3),
            }

    def update_access_time(self, cache_key: str):
        """Update last access time for cache entry (simple entries structure)."""
        with self._lock:
            entries = self._metadata.get("entries", {})
            if cache_key in entries:
                entries[cache_key]["accessed_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                self._save_to_disk()

    def increment_hits(self):
        """Increment cache hits counter."""
        with self._lock:
            self._metadata["cache_hits"] = self._metadata.get("cache_hits", 0) + 1
            self._save_to_disk()

    def increment_misses(self):
        """Increment cache misses counter."""
        with self._lock:
            self._metadata["cache_misses"] = self._metadata.get("cache_misses", 0) + 1
            self._save_to_disk()

    def cleanup_expired(self, ttl_seconds: float) -> int:
        """Remove expired entries and return count removed (simple entries structure)."""

        with self._lock:
            expired_keys = []
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
            entries = self._metadata.get("entries", {})

            for cache_key, entry in entries.items():
                try:
                    creation_time_str = entry.get("created_at")
                    if creation_time_str:
                        creation_time = datetime.fromisoformat(creation_time_str)
                        if creation_time < cutoff_time:
                            expired_keys.append(cache_key)
                except (ValueError, TypeError):
                    # Invalid timestamp, consider expired
                    expired_keys.append(cache_key)

            # Remove expired entries
            for cache_key in expired_keys:
                entries.pop(cache_key, None)

            if expired_keys:
                self._save_to_disk()

            return len(expired_keys)

    def cleanup_by_size(self, target_size_mb: float) -> Dict[str, Any]:
        """Remove least-recently-accessed entries until cache size drops to or below target."""
        with self._lock:
            # Get current total size
            stats = self.get_stats()
            total_size_mb = stats.get("total_size_mb", 0)

            logger.debug(
                f"cleanup_by_size: current size {total_size_mb:.6f}MB, target {target_size_mb:.6f}MB"
            )

            if total_size_mb <= target_size_mb:
                return {"count": 0, "removed_entries": []}  # Already at or below target

            entries = self._metadata.get("entries", {})
            logger.debug(f"cleanup_by_size: {len(entries)} total entries")

            # Sort entries by last_accessed (oldest first) for LRU eviction
            sorted_entries = sorted(
                entries.items(),
                key=lambda item: item[1].get(
                    "accessed_at", item[1].get("created_at", "")
                ),
            )

            # Calculate how many bytes we need to remove
            target_size_bytes = target_size_mb * 1024 * 1024
            current_size_bytes = total_size_mb * 1024 * 1024
            bytes_to_remove = current_size_bytes - target_size_bytes

            logger.debug(f"cleanup_by_size: need to remove {bytes_to_remove:.0f} bytes")

            removed_entries = []
            accumulated_bytes = 0

            # Remove entries until we've freed up enough space
            for cache_key, entry in sorted_entries:
                if accumulated_bytes >= bytes_to_remove:
                    break

                # file_size is stored at top level of entry dict
                entry_size_bytes = entry.get("file_size", 0)

                # actual_path is stored in nested metadata dict
                actual_path = entry.get("metadata", {}).get("actual_path") or entry.get(
                    "actual_path"
                )

                # Remove entry
                entries.pop(cache_key, None)
                removed_entries.append(
                    {"cache_key": cache_key, "actual_path": actual_path}
                )
                accumulated_bytes += entry_size_bytes

                logger.debug(
                    f"cleanup_by_size: removed {cache_key}, freed {entry_size_bytes} bytes (total freed: {accumulated_bytes})"
                )

            logger.debug(
                f"cleanup_by_size: done, removed {len(removed_entries)} entries"
            )

            if removed_entries:
                self._save_to_disk()

            return {"count": len(removed_entries), "removed_entries": removed_entries}

    def clear_all(self) -> int:
        """Remove all cache entries and return count removed (simple entries structure)."""
        with self._lock:
            # Count entries from the entries dict
            entry_count = len(self._metadata.get("entries", {}))

            # Reset to simple structure
            self._metadata = {
                "entries": {},
                "cache_hits": 0,
                "cache_misses": 0,
            }

            self._save_to_disk()
            return entry_count

    def load_metadata(self) -> Dict[str, Any]:
        """Load complete metadata structure."""
        with self._lock:
            return self._metadata.copy()

    def save_metadata(self, metadata: Dict[str, Any]):
        """Save complete metadata structure."""
        with self._lock:
            self._metadata = metadata
            self._save_to_disk()

    # --- Namespace registry helpers ---

    def _load_registry(self) -> Dict[str, Any]:
        """Load the namespace registry from disk."""
        if self._registry_file.exists():
            try:
                with open(self._registry_file, "r") as f:
                    data = json_loads(f.read())
                if isinstance(data, dict) and "namespaces" in data:
                    return data
            except Exception:
                logger.warning("Namespace registry corrupted, starting fresh")
        return {"namespaces": {}}

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """Save the namespace registry to disk atomically."""
        import tempfile
        import shutil

        try:
            if not self._registry_file.parent.exists():
                return
            fd, temp_path = tempfile.mkstemp(
                suffix=".json.tmp",
                dir=self._registry_file.parent,
                prefix="cacheness_ns_",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(json_dumps(registry, default=str))
                shutil.move(temp_path, self._registry_file)
            except Exception:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.error(f"Failed to save namespace registry: {e}")

    def _ensure_namespace_registry(self) -> None:
        """Ensure the namespace registry exists with a 'default' entry.

        Seeds the 'default' namespace pointing to the existing metadata file
        on first use.  Subsequent calls are no-ops.
        """
        registry = self._load_registry()
        if DEFAULT_NAMESPACE not in registry["namespaces"]:
            now = datetime.now(timezone.utc).isoformat()
            registry["namespaces"][DEFAULT_NAMESPACE] = {
                "namespace_id": DEFAULT_NAMESPACE,
                "display_name": "Default",
                "schema_version": 1,
                "created_at": now,
                "signature": None,
            }
            self._save_registry(registry)
            logger.info("Registered 'default' namespace in JSON registry")

    def _metadata_file_for_namespace(self, namespace_id: str) -> Path:
        """Return the metadata file path for a given namespace.

        The 'default' namespace maps to the existing ``metadata_file`` (no
        suffix) for backward compatibility.  Other namespaces get a
        ``{namespace_id}_metadata.json`` file in the same directory.
        """
        if namespace_id == DEFAULT_NAMESPACE:
            return self.metadata_file
        return self.metadata_file.parent / f"{namespace_id}_metadata.json"

    # --- Schema versioning overrides ---

    def get_schema_version(self, namespace_id: str = DEFAULT_NAMESPACE) -> int:
        """Read schema version from the JSON namespace registry."""
        with self._lock:
            registry = self._load_registry()
            ns = registry["namespaces"].get(namespace_id)
            return ns["schema_version"] if ns else 0

    def set_schema_version(self, namespace_id: str, version: int) -> None:
        """Write schema version to the JSON namespace registry."""
        with self._lock:
            registry = self._load_registry()
            ns = registry["namespaces"].get(namespace_id)
            if ns is not None:
                ns["schema_version"] = version
                self._save_registry(registry)

    def get_migrations(self) -> list:
        """Return JSON-specific schema migrations.

        Currently there are no JSON-specific migrations (the format has
        always been the same), so this returns an empty list.  When the
        JSON schema changes in the future, migrations will be added here.
        """
        return []

    # --- Namespace registry overrides ---

    def create_namespace(
        self,
        namespace_id: str,
        display_name: str = "",
    ) -> NamespaceInfo:
        """Register a new namespace and create its metadata file."""
        validate_namespace_id(namespace_id)

        if namespace_id == DEFAULT_NAMESPACE:
            raise ValueError(
                "The 'default' namespace is pre-registered and cannot be created"
            )

        with self._lock:
            registry = self._load_registry()

            if namespace_id in registry["namespaces"]:
                raise ValueError(f"Namespace {namespace_id!r} already exists")

            now = datetime.now(timezone.utc)

            # Create the per-namespace metadata file with empty structure
            ns_file = self._metadata_file_for_namespace(namespace_id)
            if not ns_file.exists():
                ns_data = {
                    "entries": {},
                    "cache_hits": 0,
                    "cache_misses": 0,
                }
                try:
                    with open(ns_file, "w") as f:
                        f.write(json_dumps(ns_data, default=str))
                except Exception as e:
                    raise OSError(
                        f"Failed to create metadata file for namespace "
                        f"{namespace_id!r}: {e}"
                    ) from e

            # Register in the namespace registry
            registry["namespaces"][namespace_id] = {
                "namespace_id": namespace_id,
                "display_name": display_name,
                "schema_version": 1,
                "created_at": now.isoformat(),
                "signature": None,
            }
            self._save_registry(registry)

            logger.info(f"Created namespace {namespace_id!r} with file {ns_file}")

            return NamespaceInfo(
                namespace_id=namespace_id,
                display_name=display_name,
                schema_version=1,
                created_at=now,
            )

    def drop_namespace(self, namespace_id: str) -> bool:
        """Remove a namespace, its metadata file, and registry entry."""
        if namespace_id == DEFAULT_NAMESPACE:
            raise ValueError("Cannot drop the 'default' namespace")

        validate_namespace_id(namespace_id)

        with self._lock:
            registry = self._load_registry()

            if namespace_id not in registry["namespaces"]:
                return False

            # Delete the per-namespace metadata file
            ns_file = self._metadata_file_for_namespace(namespace_id)
            if ns_file.exists():
                try:
                    os.remove(ns_file)
                except OSError as e:
                    logger.warning(
                        f"Failed to remove metadata file for namespace "
                        f"{namespace_id!r}: {e}"
                    )

            # Remove from registry
            del registry["namespaces"][namespace_id]
            self._save_registry(registry)

            logger.info(f"Dropped namespace {namespace_id!r}")
            return True

    def list_namespaces(self) -> List[NamespaceInfo]:
        """List all registered namespaces from the JSON registry."""
        with self._lock:
            registry = self._load_registry()
            result = []
            for ns_data in registry["namespaces"].values():
                created_at = ns_data.get("created_at")
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at)
                    except (ValueError, TypeError):
                        created_at = datetime.now(timezone.utc)
                result.append(
                    NamespaceInfo(
                        namespace_id=ns_data["namespace_id"],
                        display_name=ns_data.get("display_name", ""),
                        schema_version=ns_data.get("schema_version", 0),
                        created_at=created_at,
                        signature=ns_data.get("signature"),
                    )
                )
            # Sort by created_at for consistent ordering
            result.sort(key=lambda ns: ns.created_at)
            return result

    def get_namespace(self, namespace_id: str) -> Optional[NamespaceInfo]:
        """Get info for a specific namespace from the JSON registry."""
        with self._lock:
            registry = self._load_registry()
            ns_data = registry["namespaces"].get(namespace_id)
            if ns_data is None:
                return None
            created_at = ns_data.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except (ValueError, TypeError):
                    created_at = datetime.now(timezone.utc)
            return NamespaceInfo(
                namespace_id=ns_data["namespace_id"],
                display_name=ns_data.get("display_name", ""),
                schema_version=ns_data.get("schema_version", 0),
                created_at=created_at,
                signature=ns_data.get("signature"),
            )

    def close(self):
        """Close and clean up resources (JSON backend saves any pending changes)."""
        with self._lock:
            # Ensure any pending changes are saved to disk
            self._save_to_disk()


class SqliteBackend(MetadataBackend):
    """SQLite database-based metadata backend using SQLAlchemy ORM."""

    def __init__(self, db_file: str = "cache_metadata.db", echo: bool = False):
        """
        Initialize SQLite metadata backend.

        Args:
            db_file: Path to SQLite database file
            echo: Whether to echo SQL queries (for debugging)
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is required for SQLite backend. Install with: pip install sqlalchemy"
            )

        self.db_file = db_file

        # Configure SQLite engine with appropriate optimizations
        # Note: SQLite uses SingletonThreadPool which doesn't support pool_size/max_overflow
        self.engine = create_engine(
            f"sqlite:///{db_file}",
            echo=echo,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            connect_args={
                "check_same_thread": False,  # Allow multi-threading
                "timeout": 30,  # Longer timeout for database locks
            },
        )

        # Enable SQLite optimizations via events
        from sqlalchemy import event

        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for maximum performance."""
            cursor = dbapi_connection.cursor()

            # WAL mode for better concurrency (most important)
            cursor.execute("PRAGMA journal_mode=WAL")

            # Aggressive performance optimizations
            cursor.execute("PRAGMA synchronous=NORMAL")  # Good balance of safety/speed
            cursor.execute(
                "PRAGMA cache_size=20000"
            )  # 20MB cache (increased from 10MB)
            cursor.execute("PRAGMA temp_store=MEMORY")  # Temp tables in memory
            cursor.execute(
                "PRAGMA mmap_size=536870912"
            )  # 512MB memory mapped I/O (doubled)
            cursor.execute("PRAGMA page_size=32768")  # Larger page size for better I/O

            # Query optimization pragmas
            cursor.execute("PRAGMA optimize")  # Enable query planner optimizations
            cursor.execute("PRAGMA analysis_limit=1000")  # Better statistics

            # Concurrent access optimizations
            cursor.execute("PRAGMA busy_timeout=30000")  # 30s busy timeout
            cursor.execute(
                "PRAGMA wal_autocheckpoint=1000"
            )  # WAL checkpoint every 1000 pages

            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys=ON")

            cursor.close()

        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        self._lock = threading.Lock()

        # Create tables (including cacheness_namespaces registry)
        Base.metadata.create_all(self.engine)

        # Run formal schema versioning migrations
        self._ensure_namespace_registry()
        self.run_migrations(DEFAULT_NAMESPACE)

        # Initialize stats if not exists
        self._init_stats()

        logger.info(f"✅ SQLAlchemy metadata backend initialized: {db_file}")

    # --- Schema versioning overrides ---

    def _ensure_namespace_registry(self):
        """Ensure the namespace registry exists and has a 'default' entry.

        This handles the v0→v1 transition: if cacheness_namespaces was just
        created (by ``create_all``), seed it with the ``'default'`` namespace
        pointing to the existing unsuffixed tables.
        """
        with self.SessionLocal() as session:
            existing = session.execute(
                select(CacheNamespace).where(
                    CacheNamespace.namespace_id == DEFAULT_NAMESPACE
                )
            ).scalar_one_or_none()

            if existing is None:
                # Seed the default namespace
                ns = CacheNamespace(
                    namespace_id=DEFAULT_NAMESPACE,
                    display_name="Default",
                    schema_version=1,
                )
                session.add(ns)
                session.commit()
                logger.info("Registered 'default' namespace in cacheness_namespaces")

    def get_schema_version(self, namespace_id: str = DEFAULT_NAMESPACE) -> int:
        """Read schema version from the namespace registry."""
        with self.SessionLocal() as session:
            ns = session.execute(
                select(CacheNamespace).where(
                    CacheNamespace.namespace_id == namespace_id
                )
            ).scalar_one_or_none()
            return ns.schema_version if ns else 0

    def set_schema_version(self, namespace_id: str, version: int) -> None:
        """Write schema version to the namespace registry."""
        with self.SessionLocal() as session:
            session.execute(
                update(CacheNamespace)
                .where(CacheNamespace.namespace_id == namespace_id)
                .values(schema_version=version)
            )
            session.commit()

    def get_migrations(self) -> list:
        """Return SQLite-specific schema migrations.

        v0 → v1: Legacy column additions (s3_etag, cache_key_params, metadata_dict).
                  These used to be ad-hoc checks; now formalized.
        """

        def _migrate_v0_to_v1(backend: "SqliteBackend", namespace_id: str):
            """Add columns that may be missing from pre-versioning databases."""
            with backend.SessionLocal() as session:
                inspector = inspect(backend.engine)
                table_name = "cache_entries"  # default namespace only for v0→v1
                if table_name not in inspector.get_table_names():
                    return
                existing = {col["name"] for col in inspector.get_columns(table_name)}
                if "s3_etag" not in existing:
                    logger.info("Migrating: Adding s3_etag column to cache_entries")
                    session.execute(
                        text(
                            "ALTER TABLE cache_entries ADD COLUMN s3_etag VARCHAR(100)"
                        )
                    )
                if "cache_key_params" not in existing:
                    logger.info(
                        "Migrating: Adding cache_key_params column to cache_entries"
                    )
                    session.execute(
                        text(
                            "ALTER TABLE cache_entries ADD COLUMN cache_key_params TEXT"
                        )
                    )
                if "metadata_dict" not in existing:
                    logger.info(
                        "Migrating: Adding metadata_dict column to cache_entries"
                    )
                    session.execute(
                        text("ALTER TABLE cache_entries ADD COLUMN metadata_dict TEXT")
                    )
                session.commit()

        return [
            (0, 1, _migrate_v0_to_v1),
        ]

    # --- Namespace registry overrides ---

    def create_namespace(
        self,
        namespace_id: str,
        display_name: str = "",
    ) -> NamespaceInfo:
        """Register a new namespace and create its backing tables."""
        validate_namespace_id(namespace_id)

        if namespace_id == DEFAULT_NAMESPACE:
            raise ValueError(
                "The 'default' namespace is pre-registered and cannot be created"
            )

        with self._lock, self.SessionLocal() as session:
            # Check for duplicates
            existing = session.execute(
                select(CacheNamespace).where(
                    CacheNamespace.namespace_id == namespace_id
                )
            ).scalar_one_or_none()
            if existing is not None:
                raise ValueError(f"Namespace {namespace_id!r} already exists")

            # Create the per-namespace tables
            entries_table = f"cache_entries_{namespace_id}"
            stats_table = f"cache_stats_{namespace_id}"

            session.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS "{entries_table}" (
                    cache_key       VARCHAR(16) PRIMARY KEY,
                    description     VARCHAR(500) NOT NULL DEFAULT '',
                    data_type       VARCHAR(20) NOT NULL,
                    prefix          VARCHAR(100) NOT NULL DEFAULT '',
                    created_at      DATETIME NOT NULL,
                    accessed_at     DATETIME NOT NULL,
                    file_size       INTEGER NOT NULL DEFAULT 0,
                    file_hash       VARCHAR(16),
                    entry_signature VARCHAR(64),
                    s3_etag         VARCHAR(100),
                    object_type     VARCHAR(100),
                    storage_format  VARCHAR(20),
                    serializer      VARCHAR(20),
                    compression_codec VARCHAR(20),
                    actual_path     VARCHAR(500),
                    cache_key_params TEXT,
                    metadata_dict   TEXT
                )
            """)
            )

            session.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS "{stats_table}" (
                    id              INTEGER PRIMARY KEY DEFAULT 1,
                    cache_hits      INTEGER NOT NULL DEFAULT 0,
                    cache_misses    INTEGER NOT NULL DEFAULT 0,
                    last_updated    DATETIME NOT NULL
                )
            """)
            )

            # Create indexes matching the default table layout
            session.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS "idx_{namespace_id}_list_entries" '
                    f'ON "{entries_table}" (created_at DESC)'
                )
            )
            session.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS "idx_{namespace_id}_cleanup" '
                    f'ON "{entries_table}" (created_at)'
                )
            )
            session.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS "idx_{namespace_id}_size_mgmt" '
                    f'ON "{entries_table}" (file_size, created_at)'
                )
            )

            # Register in the namespace registry
            now = datetime.now(timezone.utc)
            ns = CacheNamespace(
                namespace_id=namespace_id,
                display_name=display_name,
                schema_version=1,
                created_at=now,
            )
            session.add(ns)

            # Initialize stats row for the new namespace
            session.execute(
                text(
                    f'INSERT OR IGNORE INTO "{stats_table}" '
                    f"(id, cache_hits, cache_misses, last_updated) "
                    f"VALUES (1, 0, 0, :now)"
                ),
                {"now": now},
            )

            session.commit()

            return NamespaceInfo(
                namespace_id=namespace_id,
                display_name=display_name,
                schema_version=1,
                created_at=now,
            )

    def drop_namespace(self, namespace_id: str) -> bool:
        """Remove a namespace and drop its tables."""
        if namespace_id == DEFAULT_NAMESPACE:
            raise ValueError("Cannot drop the 'default' namespace")

        validate_namespace_id(namespace_id)

        with self._lock, self.SessionLocal() as session:
            existing = session.execute(
                select(CacheNamespace).where(
                    CacheNamespace.namespace_id == namespace_id
                )
            ).scalar_one_or_none()

            if existing is None:
                return False

            # Drop per-namespace tables
            entries_table = f"cache_entries_{namespace_id}"
            stats_table = f"cache_stats_{namespace_id}"
            session.execute(text(f'DROP TABLE IF EXISTS "{entries_table}"'))
            session.execute(text(f'DROP TABLE IF EXISTS "{stats_table}"'))

            # Remove from registry
            session.execute(
                delete(CacheNamespace).where(
                    CacheNamespace.namespace_id == namespace_id
                )
            )
            session.commit()

            logger.info(f"Dropped namespace {namespace_id!r} and its tables")
            return True

    def list_namespaces(self) -> list:
        """List all registered namespaces from the registry."""
        with self.SessionLocal() as session:
            rows = (
                session.execute(
                    select(CacheNamespace).order_by(CacheNamespace.created_at)
                )
                .scalars()
                .all()
            )
            return [
                NamespaceInfo(
                    namespace_id=row.namespace_id,
                    display_name=row.display_name,
                    schema_version=row.schema_version,
                    created_at=row.created_at,
                    signature=row.signature,
                )
                for row in rows
            ]

    def get_namespace(self, namespace_id: str):
        """Get info for a specific namespace."""
        with self.SessionLocal() as session:
            row = session.execute(
                select(CacheNamespace).where(
                    CacheNamespace.namespace_id == namespace_id
                )
            ).scalar_one_or_none()

            if row is None:
                return None

            return NamespaceInfo(
                namespace_id=row.namespace_id,
                display_name=row.display_name,
                schema_version=row.schema_version,
                created_at=row.created_at,
                signature=row.signature,
            )

    def _run_migrations(self):
        """Legacy migration method — delegates to formal schema versioning.

        Kept for backward compatibility.  New code should call
        ``run_migrations()`` directly.
        """
        # Already handled in __init__ via run_migrations()
        pass

    def _init_stats(self):
        """Initialize cache stats if not exists."""
        with self.SessionLocal() as session:
            stats = session.execute(
                select(CacheStats).where(CacheStats.id == 1)
            ).scalar_one_or_none()

            if not stats:
                stats = CacheStats(id=1)
                session.add(stats)
                session.commit()

    def _get_stats_row(self, session) -> "CacheStats":
        """Get the single stats row."""
        stats = session.execute(
            select(CacheStats).where(CacheStats.id == 1)
        ).scalar_one_or_none()

        if not stats:
            stats = CacheStats(id=1)
            session.add(stats)
            session.commit()
        return stats

    def get_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get specific cache entry metadata using columns directly - zero JSON parsing."""
        with self.SessionLocal() as session:
            # Single optimized query - get entry using columns only
            entry = session.execute(
                select(CacheEntry).where(CacheEntry.cache_key == cache_key)
            ).scalar_one_or_none()

            if not entry:
                return None

            # Build metadata from columns directly - zero JSON parsing for backend data
            metadata = {}

            # Add backend technical metadata from dedicated columns (not JSON)
            if entry.object_type is not None:
                metadata["object_type"] = entry.object_type
            if entry.storage_format is not None:
                metadata["storage_format"] = entry.storage_format
            if entry.serializer is not None:
                metadata["serializer"] = entry.serializer
            if entry.compression_codec is not None:
                metadata["compression_codec"] = entry.compression_codec
            if entry.actual_path is not None:
                metadata["actual_path"] = entry.actual_path

            # Add optional security fields from columns (not JSON)
            if entry.file_hash is not None:
                metadata["file_hash"] = entry.file_hash
            if entry.entry_signature is not None:
                metadata["entry_signature"] = entry.entry_signature
            if entry.s3_etag is not None:
                metadata["s3_etag"] = entry.s3_etag

            # Only parse cache_key_params JSON if it exists (should be disabled by default)
            if entry.cache_key_params is not None:
                try:
                    metadata["cache_key_params"] = json_loads(entry.cache_key_params)
                except (ValueError, TypeError):
                    pass  # Skip malformed cache_key_params

            # Ensure timestamps are always in UTC for consistency
            created_at_utc = (
                entry.created_at.astimezone(timezone.utc)
                if entry.created_at.tzinfo
                else entry.created_at.replace(tzinfo=timezone.utc)
            )
            accessed_at_utc = (
                entry.accessed_at.astimezone(timezone.utc)
                if entry.accessed_at.tzinfo
                else entry.accessed_at.replace(tzinfo=timezone.utc)
            )

            return {
                "description": entry.description,
                "data_type": entry.data_type,
                "prefix": entry.prefix,
                "created_at": created_at_utc.isoformat(),
                "accessed_at": accessed_at_utc.isoformat(),
                "file_size": entry.file_size,
                "metadata": metadata,
            }

    def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
        """Store cache entry metadata using dedicated columns - zero JSON overhead for backend data."""
        with self._lock, self.SessionLocal() as session:
            # Extract and process metadata fields efficiently
            metadata = entry_data.get("metadata", {}).copy()

            # Extract full metadata copy FIRST (before any pop() operations) if present
            full_metadata_dict = metadata.pop("_full_metadata", None)
            full_metadata_json = None
            if full_metadata_dict is not None:
                try:
                    full_metadata_json = json_dumps(full_metadata_dict)
                    logger.debug(
                        f"Serialized full_metadata: {len(full_metadata_json)} chars"
                    )
                except Exception as e:
                    # If serialization fails, skip full_metadata
                    logger.warning(f"Failed to serialize full_metadata JSON: {e}")
                    full_metadata_json = None

            # Extract backend technical metadata to dedicated columns (not JSON)
            object_type = metadata.pop("object_type", None)
            storage_format = metadata.pop("storage_format", None)
            serializer = metadata.pop("serializer", None)
            compression_codec = metadata.pop("compression_codec", None)
            actual_path = metadata.pop("actual_path", None)

            # Extract security fields from metadata (remove from JSON to avoid duplication)
            file_hash = metadata.pop("file_hash", None)
            entry_signature = metadata.pop("entry_signature", None)
            s3_etag = metadata.pop("s3_etag", None)  # S3 ETag if using S3 backend
            # These fields are pre-serialized JSON strings from the caching layer
            cache_key_params = metadata.pop("cache_key_params", None)
            metadata_dict_value = metadata.pop(
                "metadata_dict", None
            )  # User metadata for querying

            # Remove redundant fields that are already stored as columns
            metadata.pop("prefix", None)  # Already stored in prefix column
            metadata.pop("data_type", None)  # Already stored in data_type column

            # Handle timestamps with proper defaults
            created_at = entry_data.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            elif created_at is None:
                created_at = datetime.now(timezone.utc)

            accessed_at = entry_data.get("accessed_at")
            if isinstance(accessed_at, str):
                accessed_at = datetime.fromisoformat(accessed_at)
            elif accessed_at is None:
                accessed_at = datetime.now(timezone.utc)

            # Use efficient INSERT OR REPLACE with dedicated columns - zero JSON overhead
            from sqlalchemy import text

            session.execute(
                text("""
                    INSERT OR REPLACE INTO cache_entries 
                    (cache_key, description, data_type, prefix, file_size, 
                     file_hash, entry_signature, s3_etag, cache_key_params, metadata_dict,
                     object_type, storage_format, serializer, compression_codec, actual_path,
                     created_at, accessed_at)
                    VALUES (:cache_key, :description, :data_type, :prefix, :file_size, 
                           :file_hash, :entry_signature, :s3_etag, :cache_key_params, :metadata_dict,
                           :object_type, :storage_format, :serializer, :compression_codec, :actual_path,
                           :created_at, :accessed_at)
                """),
                {
                    "cache_key": cache_key,
                    "description": entry_data.get("description", ""),
                    "data_type": entry_data.get("data_type", "unknown"),
                    "prefix": entry_data.get("prefix", ""),
                    "file_size": entry_data.get("file_size", 0),
                    "file_hash": file_hash,
                    "entry_signature": entry_signature,
                    "s3_etag": s3_etag,
                    "cache_key_params": cache_key_params,
                    "metadata_dict": metadata_dict_value,
                    "object_type": object_type,
                    "storage_format": storage_format,
                    "serializer": serializer,
                    "compression_codec": compression_codec,
                    "actual_path": actual_path,
                    "created_at": created_at,
                    "accessed_at": accessed_at,
                },
            )
            session.commit()

    def remove_entry(self, cache_key: str) -> bool:
        """Remove cache entry metadata (custom metadata will cascade delete via FK)."""
        with self._lock, self.SessionLocal() as session:
            # Delete cache entry - custom metadata records will cascade delete automatically
            # due to ondelete="CASCADE" on the cache_key foreign key
            result = session.execute(
                delete(CacheEntry).where(CacheEntry.cache_key == cache_key)
            )
            session.commit()
            return result.rowcount > 0

    def update_entry_metadata(self, cache_key: str, updates: Dict[str, Any]) -> bool:
        """
        Update metadata fields for an existing cache entry.

        Only updates metadata — blob I/O is handled by UnifiedCache.update_data().

        Args:
            cache_key: The unique identifier for the cache entry to update
            updates: Dict of metadata fields to update (file_size, file_hash,
                    actual_path, data_type, storage_format, serializer, etc.)

        Returns:
            bool: True if entry was updated, False if entry doesn't exist
        """
        with self._lock, self.SessionLocal() as session:
            # Check if entry exists
            entry = session.execute(
                select(CacheEntry).where(CacheEntry.cache_key == cache_key)
            ).scalar_one_or_none()

            if not entry:
                return False

            # Update derived metadata fields
            now = datetime.now(timezone.utc)
            entry.created_at = now  # Reset timestamp

            if "file_size" in updates:
                entry.file_size = updates["file_size"]
            if "file_hash" in updates:
                entry.file_hash = updates["file_hash"]
            elif "content_hash" in updates:
                entry.file_hash = updates["content_hash"]
            if "actual_path" in updates:
                entry.actual_path = str(updates["actual_path"])
            if "data_type" in updates:
                entry.data_type = updates["data_type"]
            if "storage_format" in updates:
                entry.storage_format = updates["storage_format"]
            if "serializer" in updates:
                entry.serializer = updates["serializer"]
            if "compression_codec" in updates:
                entry.compression_codec = updates["compression_codec"]
            if "object_type" in updates:
                entry.object_type = updates["object_type"]

            session.commit()
            return True

    def iter_entry_summaries(self) -> List[Dict[str, Any]]:
        """Return lightweight flat entry dicts — raw SQL, no ORM hydration."""
        from sqlalchemy import text

        with self.SessionLocal() as session:
            rows = session.execute(
                text(
                    "SELECT cache_key, data_type, description, prefix, "
                    "       file_size, created_at, accessed_at, "
                    "       object_type, storage_format, serializer, "
                    "       compression_codec, actual_path, "
                    "       file_hash, entry_signature "
                    "FROM cache_entries"
                )
            ).fetchall()
            result = []
            for row in rows:
                flat = {
                    "cache_key": row[0],
                    "data_type": row[1],
                    "description": row[2] or "",
                    "prefix": row[3] or "",
                    "file_size": row[4] or 0,
                    "created_at": row[5],
                    "accessed_at": row[6],
                }
                # Only include non-None technical metadata
                if row[7] is not None:
                    flat["object_type"] = row[7]
                if row[8] is not None:
                    flat["storage_format"] = row[8]
                if row[9] is not None:
                    flat["serializer"] = row[9]
                if row[10] is not None:
                    flat["compression_codec"] = row[10]
                if row[11] is not None:
                    flat["actual_path"] = row[11]
                if row[12] is not None:
                    flat["file_hash"] = row[12]
                if row[13] is not None:
                    flat["entry_signature"] = row[13]
                result.append(flat)
            return result

    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries using columns directly - zero JSON parsing overhead for backend data."""
        with self.SessionLocal() as session:
            # Use a single optimized query to get all data at once
            entries = (
                session.execute(
                    select(CacheEntry).order_by(desc(CacheEntry.created_at))
                )
                .scalars()
                .all()
            )

            result = []
            for entry in entries:
                # Build metadata from columns directly - zero JSON parsing for backend data
                entry_metadata = {}

                # Add backend technical metadata from dedicated columns (not JSON)
                if entry.object_type is not None:
                    entry_metadata["object_type"] = entry.object_type
                if entry.storage_format is not None:
                    entry_metadata["storage_format"] = entry.storage_format
                if entry.serializer is not None:
                    entry_metadata["serializer"] = entry.serializer
                if entry.compression_codec is not None:
                    entry_metadata["compression_codec"] = entry.compression_codec
                if entry.actual_path is not None:
                    entry_metadata["actual_path"] = entry.actual_path

                # Add optional security fields from columns (not JSON)
                if entry.file_hash is not None:
                    entry_metadata["file_hash"] = entry.file_hash
                if entry.entry_signature is not None:
                    entry_metadata["entry_signature"] = entry.entry_signature
                if entry.s3_etag is not None:
                    entry_metadata["s3_etag"] = entry.s3_etag

                # Only parse cache_key_params JSON if it exists (should be disabled by default)
                if entry.cache_key_params is not None:
                    try:
                        entry_metadata["cache_key_params"] = json_loads(
                            entry.cache_key_params
                        )
                    except (ValueError, TypeError):
                        pass  # Skip malformed cache_key_params

                result.append(
                    {
                        "cache_key": entry.cache_key,
                        "data_type": entry.data_type,
                        "description": entry.description,
                        "metadata": entry_metadata,
                        "created": entry.created_at.isoformat(),
                        "last_accessed": entry.accessed_at.isoformat(),
                        "size_mb": round(entry.file_size / (1024 * 1024), 3),
                    }
                )

            return result

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics using SQL aggregates (no full table scan)."""
        with self.SessionLocal() as session:
            # Single aggregation query — no Python-side iteration
            row = session.execute(
                select(
                    func.count(CacheEntry.cache_key).label("total"),
                    func.coalesce(func.sum(CacheEntry.file_size), 0).label(
                        "total_size"
                    ),
                    func.count(
                        case(
                            (CacheEntry.data_type == "dataframe", 1),
                        )
                    ).label("dataframe_count"),
                    func.count(
                        case(
                            (CacheEntry.data_type == "array", 1),
                        )
                    ).label("array_count"),
                )
            ).one()

            total_size_mb = row.total_size / (1024 * 1024)

            # Get hit/miss stats
            stats = self._get_stats_row(session)
            hit_rate = (
                stats.cache_hits / (stats.cache_hits + stats.cache_misses)
                if (stats.cache_hits + stats.cache_misses) > 0
                else 0.0
            )

            return {
                "total_entries": row.total,
                "dataframe_entries": row.dataframe_count,
                "array_entries": row.array_count,
                "total_size_mb": round(total_size_mb, 2),
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "hit_rate": round(hit_rate, 3),
            }

    def update_access_time(self, cache_key: str):
        """Update last access time for cache entry."""
        with self._lock, self.SessionLocal() as session:
            session.execute(
                update(CacheEntry)
                .where(CacheEntry.cache_key == cache_key)
                .values(accessed_at=datetime.now(timezone.utc))
            )
            session.commit()

    def increment_hits(self):
        """Increment cache hits counter."""
        with self._lock, self.SessionLocal() as session:
            session.execute(
                update(CacheStats)
                .where(CacheStats.id == 1)
                .values(
                    cache_hits=CacheStats.cache_hits + 1,
                    last_updated=datetime.now(timezone.utc),
                )
            )
            session.commit()

    def increment_misses(self):
        """Increment cache misses counter."""
        with self._lock, self.SessionLocal() as session:
            session.execute(
                update(CacheStats)
                .where(CacheStats.id == 1)
                .values(
                    cache_misses=CacheStats.cache_misses + 1,
                    last_updated=datetime.now(timezone.utc),
                )
            )
            session.commit()

    def cleanup_expired(self, ttl_seconds: float) -> int:
        """Remove expired entries and return count removed."""

        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)

        with self._lock, self.SessionLocal() as session:
            # Delete expired entries
            result = session.execute(
                delete(CacheEntry).where(CacheEntry.created_at < cutoff_time)
            )
            deleted_count = result.rowcount
            session.commit()
            return deleted_count

    def cleanup_by_size(self, target_size_mb: float) -> Dict[str, Any]:
        """Remove least-recently-accessed entries until cache size drops to or below target."""
        with self._lock, self.SessionLocal() as session:
            # Get current total size
            result = session.execute(
                select(func.sum(CacheEntry.file_size)).select_from(CacheEntry)
            )
            total_size_bytes = result.scalar() or 0
            total_size_mb = total_size_bytes / (1024 * 1024)

            if total_size_mb <= target_size_mb:
                return {"count": 0, "removed_entries": []}  # Already at or below target

            target_size_bytes = target_size_mb * 1024 * 1024
            bytes_to_remove = total_size_bytes - target_size_bytes

            # Get entries sorted by accessed_at (oldest first) with actual_path
            entries_to_delete = session.execute(
                select(
                    CacheEntry.cache_key, CacheEntry.file_size, CacheEntry.actual_path
                ).order_by(CacheEntry.accessed_at.asc())
            ).all()

            # Calculate which entries to delete to reach target
            removed_entries = []
            accumulated_size = 0

            for cache_key, file_size, actual_path in entries_to_delete:
                if accumulated_size >= bytes_to_remove:
                    break
                removed_entries.append(
                    {"cache_key": cache_key, "actual_path": actual_path}
                )
                accumulated_size += file_size

            if removed_entries:
                # Delete the selected entries
                removed_keys = [e["cache_key"] for e in removed_entries]
                session.execute(
                    delete(CacheEntry).where(CacheEntry.cache_key.in_(removed_keys))
                )
                session.commit()

            return {"count": len(removed_entries), "removed_entries": removed_entries}

    def clear_all(self) -> int:
        """Remove all cache entries and return count removed."""
        with self._lock, self.SessionLocal() as session:
            # Count existing entries
            result = session.execute(select(func.count(CacheEntry.cache_key)))
            entry_count = result.scalar() or 0

            # Delete all entries
            session.execute(delete(CacheEntry))

            # Reset stats
            stats = session.execute(
                select(CacheStats).where(CacheStats.id == 1)
            ).scalar_one_or_none()

            if stats:
                stats.cache_hits = 0
                stats.cache_misses = 0

            session.commit()
            return entry_count

    def load_metadata(self) -> Dict[str, Any]:
        """Load complete metadata structure (SQLite backend operates on individual entries)."""
        # SQLite backend doesn't use bulk metadata operations - returns empty dict
        return {}

    def save_metadata(self, metadata: Dict[str, Any]):
        """Save complete metadata structure (SQLite backend operates on individual entries)."""
        # SQLite backend doesn't use bulk metadata operations - no-op
        pass

    def close(self):
        """Close all database connections and clean up resources."""
        if hasattr(self, "engine") and self.engine:
            # Close all connections in the pool
            self.engine.dispose()
            # On Windows, we need to be more aggressive
            import gc

            gc.collect()  # Force garbage collection to release file handles
            logger.debug("SQLite engine disposed and connections closed")

    def __del__(self):
        """Ensure connections are closed when the backend is garbage collected."""
        try:
            self.close()
        except Exception:
            # Suppress errors during interpreter shutdown
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure connections are closed."""
        self.close()
        return False


def create_metadata_backend(backend_type: str = "auto", **kwargs) -> MetadataBackend:
    """
    Factory function to create metadata backends with optional entry caching.

    Args:
        backend_type: "auto", "sqlite", "json", "sqlite_memory", or "postgresql"
                     "auto" prefers SQLite (file-based) if available, falls back to in-memory SQLite, then JSON
        **kwargs: Backend-specific configuration including optional 'config' for caching

    Returns:
        MetadataBackend instance (potentially wrapped with caching)
    """
    # Extract cache config if provided
    cache_config = kwargs.pop("config", None)

    # Create the base backend
    if backend_type == "json":
        metadata_file = kwargs.get("metadata_file", Path("cache_metadata.json"))
        backend = JsonBackend(metadata_file)
    elif backend_type == "sqlite":
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is required for SQLite backend but is not available. Install with: uv add sqlalchemy"
            )
        db_file = kwargs.get("db_file", "cache_metadata.db")
        echo = kwargs.get("echo", False)
        backend = SqliteBackend(db_file, echo)
    elif backend_type == "sqlite_memory":
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is required for in-memory SQLite backend but is not available. Install with: uv add sqlalchemy"
            )
        echo = kwargs.get("echo", False)
        backend = SqliteBackend(":memory:", echo)  # In-memory SQLite database
    elif backend_type == "postgresql":
        # Import PostgresBackend from the storage backends
        try:
            from .storage.backends.postgresql_backend import PostgresBackend
        except ImportError as e:
            raise ImportError(
                f"PostgreSQL backend is not available. Install with: pip install psycopg2-binary sqlalchemy. Error: {e}"
            )

        connection_url = kwargs.get("connection_url")
        if not connection_url:
            raise ValueError("PostgreSQL backend requires 'connection_url' parameter")

        backend = PostgresBackend(
            connection_url=connection_url,
            pool_size=kwargs.get("pool_size", 10),
            max_overflow=kwargs.get("max_overflow", 20),
            pool_pre_ping=kwargs.get("pool_pre_ping", True),
            pool_recycle=kwargs.get("pool_recycle", 3600),
            echo=kwargs.get("echo", False),
            table_prefix=kwargs.get("table_prefix", ""),
        )
    elif backend_type == "auto":
        # Auto mode: prefer file-based SQLite > in-memory SQLite > JSON
        if SQLALCHEMY_AVAILABLE:
            try:
                # Try file-based SQLite first
                db_file = kwargs.get("db_file", "cache_metadata.db")
                echo = kwargs.get("echo", False)
                backend = SqliteBackend(db_file, echo)
            except Exception:
                try:
                    # Fall back to in-memory SQLite if file-based fails
                    logger.warning("File-based SQLite failed, using in-memory SQLite")
                    echo = kwargs.get("echo", False)
                    backend = SqliteBackend(":memory:", echo)
                except Exception:
                    # Final fallback to JSON
                    logger.warning("SQLite backends failed, falling back to JSON")
                    metadata_file = kwargs.get(
                        "metadata_file", Path("cache_metadata.json")
                    )
                    backend = JsonBackend(metadata_file)
        else:
            # SQLAlchemy not available, use JSON
            metadata_file = kwargs.get("metadata_file", Path("cache_metadata.json"))
            backend = JsonBackend(metadata_file)
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. Supported: 'auto', 'json', 'sqlite', 'sqlite_memory', 'postgresql'"
        )

    # Apply memory cache layer wrapper for disk-persistent backends
    if (
        cache_config is not None
        and hasattr(cache_config, "enable_memory_cache")
        and cache_config.enable_memory_cache
    ):
        logger.debug(f"Wrapping {backend_type} backend with memory cache layer")
        backend = CachedMetadataBackend(backend, cache_config)

    return backend
