"""Abstract base class for metadata backends and caching wrapper."""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from ..interfaces import EntrySummary
from ._compat import (
    DEFAULT_NAMESPACE,
    NamespaceInfo,
    Migration,
    CACHETOOLS_AVAILABLE,
)

# Import TTLCache only if cachetools is available
if CACHETOOLS_AVAILABLE:
    from cachetools import TTLCache

logger = logging.getLogger(__name__)


class MetadataBackend(ABC):
    """Abstract base class for cache metadata backends.

    This is the canonical MetadataBackend interface.  Every metadata backend
    (JSON, SQLite, PostgreSQL, custom) must extend this ABC.

    The class is re-exported by ``cacheness.storage.backends.base`` so that
    ``from cacheness.storage.backends import MetadataBackend`` and
    ``from cacheness.metadata import MetadataBackend`` both resolve to the
    same type.
    """

    @property
    def active_namespace(self) -> str:
        """Return the active namespace for this backend instance.

        Defaults to DEFAULT_NAMESPACE if not set by subclass __init__.
        """
        return getattr(self, "_active_namespace", DEFAULT_NAMESPACE)

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

        The returned dict conforms to the :class:`~cacheness.interfaces.EntryData`
        TypedDict contract::

            description, data_type, created_at, accessed_at,
            file_size (bytes), metadata (nested dict)

        See also: list_entries() returns user-facing format with different keys.
        """
        pass

    @abstractmethod
    def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
        """Store cache entry metadata.

        *entry_data* should conform to the
        :class:`~cacheness.interfaces.EntryData` contract.
        """
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
    def cleanup_by_size(self, target_size_bytes: int) -> Dict[str, Any]:
        """Remove least-recently-accessed entries until cache size drops to or below target.

        Args:
            target_size_bytes: Target cache size in bytes

        Returns:
            Dict with 'count' (int) and 'removed_entries' (list of dicts with 'cache_key' and 'actual_path')
        """
        pass

    @abstractmethod
    def clear_all(self) -> int:
        """Remove all cache entries and return count removed."""
        pass

    def iter_entry_summaries(self) -> List[EntrySummary]:
        """Return lightweight entry summaries for internal filtering.

        Each dict contains flat, unprocessed column values:
            cache_key, data_type, description, created_at (raw),
            accessed_at (raw), file_size (bytes int), plus any
            backend-specific technical fields (object_type,
            storage_format, serializer, compression_codec,
            actual_path, file_hash, entry_signature).

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
        result: List[EntrySummary] = []
        for entry in self.list_entries():
            flat: Dict[str, Any] = {k: v for k, v in entry.items() if k != "metadata"}
            flat.update(entry.get("metadata", {}))
            result.append(flat)  # type: ignore[arg-type]
        return result

    def keys_by_prefix(self, prefix: str) -> List[str]:
        """Return cache keys that start with *prefix*.

        The default implementation filters ``iter_entry_summaries()`` in
        Python.  SQL-backed backends should override with a ``LIKE``
        query for better performance on large caches.
        """
        return [
            e["cache_key"]
            for e in self.iter_entry_summaries()
            if e.get("cache_key", "").startswith(prefix)
        ]

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

    def run_all_migrations(self) -> None:
        """Run pending migrations for every registered namespace.

        Iterates the namespace registry and calls :meth:`run_migrations`
        on each.  Safe to call during init — the default namespace is
        always present, and any future namespaces are migrated lazily
        when created or at the next startup.
        """
        for ns in self.list_namespaces():
            self.run_migrations(ns.namespace_id)

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

    def set_namespace_signature(self, namespace_id: str, signature: str) -> None:
        """Store a cryptographic signature for a namespace registry row.

        Subclasses that support namespaces should override this.  The
        default implementation is a no-op (signature column stays NULL).

        Args:
            namespace_id: The namespace to update.
            signature: The HMAC signature string to store.
        """
        pass  # default no-op for backends without namespace tables

    def namespace_exists(self, namespace_id: str) -> bool:
        """Check whether a namespace is registered.

        Default implementation delegates to :meth:`get_namespace`.
        """
        return self.get_namespace(namespace_id) is not None

    def clear_all_namespaces(self) -> Dict[str, int]:
        """Nuclear option: drop every non-default namespace and clear the default.

        Iterates the namespace registry.  Non-default namespaces are fully
        dropped (tables/files removed via :meth:`drop_namespace`).  The
        default namespace is cleared (rows deleted, stats reset via
        :meth:`clear_all`).

        Returns:
            Mapping of ``namespace_id`` → entries removed (``-1`` means the
            namespace was dropped entirely rather than row-cleared).
        """
        results: Dict[str, int] = {}
        for ns in self.list_namespaces():
            if ns.namespace_id == DEFAULT_NAMESPACE:
                results[DEFAULT_NAMESPACE] = self.clear_all()
            else:
                self.drop_namespace(ns.namespace_id)
                results[ns.namespace_id] = -1
        return results

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

        # Proxy the active namespace from the wrapped backend
        if hasattr(wrapped_backend, "_active_namespace"):
            self._active_namespace = wrapped_backend._active_namespace

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

    def clear_all_namespaces(self) -> Dict[str, int]:
        """Delegate to wrapped backend and clear memory cache."""
        results = self.backend.clear_all_namespaces()

        if self._memory_cache is not None:
            with self._lock:
                self._memory_cache.clear()
                logger.debug("Memory cache cleared (all namespaces)")

        return results

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

    def iter_entry_summaries(self) -> List[EntrySummary]:
        return self.backend.iter_entry_summaries()

    def keys_by_prefix(self, prefix: str) -> List[str]:
        return self.backend.keys_by_prefix(prefix)

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

    def cleanup_by_size(self, target_size_bytes: int) -> Dict[str, Any]:
        """Delegate cleanup_by_size to wrapped backend and clear memory cache."""
        result = self.backend.cleanup_by_size(target_size_bytes)

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

    # --- Namespace registry delegation to wrapped backend ---

    def create_namespace(
        self, namespace_id: str, display_name: str | None = None
    ) -> NamespaceInfo:
        return self.backend.create_namespace(namespace_id, display_name)

    def drop_namespace(self, namespace_id: str) -> bool:
        return self.backend.drop_namespace(namespace_id)

    def list_namespaces(self) -> list:
        return self.backend.list_namespaces()

    def get_namespace(self, namespace_id: str):
        return self.backend.get_namespace(namespace_id)

    def set_namespace_signature(self, namespace_id: str, signature: str) -> None:
        return self.backend.set_namespace_signature(namespace_id, signature)

    def get_schema_version(self, namespace_id: str = DEFAULT_NAMESPACE) -> int:
        return self.backend.get_schema_version(namespace_id)

    def set_schema_version(self, namespace_id: str, version: int) -> None:
        return self.backend.set_schema_version(namespace_id, version)

    def get_migrations(self) -> list:
        return self.backend.get_migrations()
