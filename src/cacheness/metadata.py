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

import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import logging

from .json_utils import dumps as json_dumps, loads as json_loads

logger = logging.getLogger(__name__)

# Try to import cachetools for entry caching
try:
    from cachetools import LRUCache, LFUCache, FIFOCache, RRCache, TTLCache
    from functools import wraps
    import time
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
        entry_signature = Column(String(64), nullable=True)  # HMAC-SHA256 hex (64 chars)

        # Original cache key parameters - only store, don't index (rarely queried)
        cache_key_params = Column(Text, nullable=True)  # Removed index=True

        # Relationship to custom metadata links (loaded lazily to avoid circular imports)
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            # This will be set up after custom_metadata module is imported

        def get_custom_metadata_links(self):
            """Get custom metadata links for this cache entry."""
            # Import here to avoid circular dependency
            try:
                from .custom_metadata import CacheMetadataLink
                from sqlalchemy.orm import object_session

                session = object_session(self)
                if session:
                    return (
                        session.query(CacheMetadataLink)
                        .filter(CacheMetadataLink.cache_key == self.cache_key)
                        .all()
                    )
            except ImportError:
                pass
            return []

        # Data-specific metadata (stored as JSON)
        metadata_json = Column(Text, default="{}", nullable=False)

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

except ImportError:
    # SQLAlchemy not available
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not available, SQLite backend will not work")

    # Define dummy classes to avoid runtime errors
    class CacheEntry:
        pass

    class CacheStats:
        pass

    Base = None


class MetadataBackend(ABC):
    """Abstract base class for cache metadata backends."""

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
        """Get specific cache entry metadata."""
        pass

    @abstractmethod
    def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
        """Store cache entry metadata."""
        pass

    @abstractmethod
    def remove_entry(self, cache_key: str):
        """Remove cache entry metadata."""
        pass

    @abstractmethod
    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries with metadata."""
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
    def cleanup_expired(self, ttl_hours: int) -> int:
        """Remove expired entries and return count removed."""
        pass

    @abstractmethod
    def clear_all(self) -> int:
        """Remove all cache entries and return count removed."""
        pass


def create_entry_cache(cache_type: str, maxsize: int, ttl_seconds: float):
    """Create a cache instance based on configuration."""
    if not CACHETOOLS_AVAILABLE:
        return None
        
    if cache_type == "lru":
        # Use TTLCache with LRU eviction for both size and TTL limits
        return TTLCache(maxsize=maxsize, ttl=ttl_seconds)
    elif cache_type == "lfu":
        # LFU with TTL wrapper (manual TTL implementation)
        return TTLCache(maxsize=maxsize, ttl=ttl_seconds)  # cachetools TTLCache uses LRU internally
    elif cache_type == "fifo":
        return TTLCache(maxsize=maxsize, ttl=ttl_seconds)  # TTLCache is good enough
    elif cache_type == "rr":
        return TTLCache(maxsize=maxsize, ttl=ttl_seconds)  # Random replacement with TTL
    else:
        logger.warning(f"Unknown cache type {cache_type}, using LRU")
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
                config.memory_cache_ttl_seconds
            )
            
            # Optional cache statistics
            if config.memory_cache_stats:
                self._cache_hits = 0
                self._cache_misses = 0
            else:
                self._cache_hits = None
                self._cache_misses = None
                
            logger.info(f"🚀 Memory cache layer enabled: {config.memory_cache_type} "
                       f"(maxsize={config.memory_cache_maxsize}, ttl={config.memory_cache_ttl_seconds}s)")
        else:
            self._memory_cache = None
            self._cache_hits = None
            self._cache_misses = None
            
            if not CACHETOOLS_AVAILABLE:
                logger.warning("cachetools not available, memory cache layer disabled. Install with: pip install cachetools")
    
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
    
    def remove_entry(self, cache_key: str):
        """Remove cache entry metadata and invalidate memory cache."""
        # Remove from disk backend first
        self.backend.remove_entry(cache_key)
        
        # Remove from memory cache if enabled
        if self._memory_cache is not None:
            with self._lock:
                entry_cache_key = self._cache_key_for_entry(cache_key)
                self._memory_cache.pop(entry_cache_key, None)
                logger.debug(f"Memory cache invalidated: {cache_key}")
    
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
        if self._memory_cache is None or self._cache_hits is None or self._cache_misses is None:
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
    
    def cleanup_expired(self, ttl_hours: int) -> int:
        count = self.backend.cleanup_expired(ttl_hours)
        
        # Clear entire memory cache after cleanup (entries might be stale)
        if self._memory_cache is not None and count > 0:
            with self._lock:
                self._memory_cache.clear()
                logger.debug("Memory cache cleared after expired cleanup")
        
        return count


class InMemoryBackend(MetadataBackend):
    """Ultra-fast in-memory metadata backend optimized for maximum PUT/GET performance.
    
    Uses a single unified dictionary structure to minimize lookups:
    - Single dict for O(1) key-based lookups
    - All metadata stored in one place to minimize object overhead
    - Minimal object creation and copying
    - No I/O operations for maximum speed
    - Thread-safe with minimal locking overhead
    
    Perfect for:
    - High-frequency temporary caching
    - Development and testing 
    - Scenarios where persistence is not needed
    - Maximum performance applications
    """
    
    def __init__(self):
        """Initialize in-memory backend with optimized data structures."""
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Single unified dict for maximum PUT/GET performance
        # Structure: cache_key -> {
        #   'metadata': {...},
        #   'description': str,
        #   'data_type': str,
        #   'prefix': str,
        #   'file_size': int,
        #   'created_at': datetime,
        #   'accessed_at': datetime
        # }
        self._entries = {}
        
        # Statistics - single values for speed
        self._stats = {
            "hits": 0,
            "misses": 0,
            "total_entries": 0,
            "total_size": 0
        }
        
        # Cache for expensive operations (only used for list operations)
        self._list_cache = None
        self._list_cache_dirty = True
    
    def get_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get specific cache entry metadata with O(1) lookup."""
        entry = self._entries.get(cache_key)
        if entry is None:
            return None
            
        # Fast single-dict access - no multiple .get() calls
        created_at = entry.get('created_at')
        accessed_at = entry.get('accessed_at')
        
        return {
            "description": entry.get('description', ''),
            "data_type": entry.get('data_type', 'unknown'),
            "prefix": entry.get('prefix', ''),
            "created_at": created_at.isoformat() if created_at else None,
            "accessed_at": accessed_at.isoformat() if accessed_at else None,
            "file_size": entry.get('file_size', 0),
            "metadata": entry.get('metadata', {}),
        }
    
    def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
        """Store cache entry metadata with optimized O(1) operations."""
        with self._lock:
            now = datetime.now(timezone.utc)
            
            # Handle timestamps efficiently - reuse datetime objects when possible
            created_at = entry_data.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except ValueError:
                    created_at = now
            elif not isinstance(created_at, datetime):
                created_at = now
            
            accessed_at = entry_data.get("accessed_at")
            if isinstance(accessed_at, str):
                try:
                    accessed_at = datetime.fromisoformat(accessed_at)
                except ValueError:
                    accessed_at = now
            elif not isinstance(accessed_at, datetime):
                accessed_at = now
            
            # Update statistics efficiently - check if this is a new entry
            is_new_entry = cache_key not in self._entries
            old_size = 0 if is_new_entry else self._entries[cache_key].get('file_size', 0)
            new_size = entry_data.get("file_size", 0)
            
            if is_new_entry:
                self._stats["total_entries"] += 1
            self._stats["total_size"] += (new_size - old_size)
            
            # Store everything in unified structure with single dict assignment
            self._entries[cache_key] = {
                'metadata': entry_data.get("metadata", {}),
                'description': entry_data.get("description", ""),
                'data_type': entry_data.get("data_type", "unknown"),
                'prefix': entry_data.get("prefix", ""),
                'file_size': new_size,
                'created_at': created_at,
                'accessed_at': accessed_at,
            }
            
            # Invalidate list cache
            self._list_cache_dirty = True
    
    def remove_entry(self, cache_key: str):
        """Remove cache entry metadata with O(1) operations."""
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is None:
                return
                
            # Update statistics before removal
            self._stats["total_entries"] -= 1
            self._stats["total_size"] -= entry.get('file_size', 0)
            
            # Single dict removal - much faster than multiple pops
            del self._entries[cache_key]
            
            # Invalidate list cache
            self._list_cache_dirty = True
    
    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries with smart caching for repeated calls."""
        with self._lock:
            # Use cached result if available and not dirty
            if not self._list_cache_dirty and self._list_cache is not None:
                return self._list_cache  # Return reference for speed (caller shouldn't modify)
            
            # Build optimized entry list with minimal overhead
            entries = []
            for cache_key, entry in self._entries.items():
                # Single dict access for all fields - much faster
                file_size = entry.get('file_size', 0)
                created_at = entry.get('created_at')
                accessed_at = entry.get('accessed_at')
                
                entry_dict = {
                    "cache_key": cache_key,
                    "description": entry.get('description', ''),
                    "data_type": entry.get('data_type', 'unknown'),
                    "prefix": entry.get('prefix', ''),
                    "file_size": file_size,
                    "size_mb": file_size / (1024 * 1024),
                }
                
                # Only convert timestamps if they exist (avoid unnecessary work)
                if created_at:
                    entry_dict["created"] = created_at.isoformat()
                if accessed_at:
                    entry_dict["last_accessed"] = accessed_at.isoformat()
                
                entries.append(entry_dict)
            
            # Cache the result for future calls
            self._list_cache = entries
            self._list_cache_dirty = False
            
            return entries
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics with O(1) performance."""
        with self._lock:
            total_size_mb = self._stats["total_size"] / (1024 * 1024)
            
            return {
                "backend_type": "memory",
                "total_entries": self._stats["total_entries"],
                "total_size_mb": round(total_size_mb, 2),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": (
                    self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
                    if (self._stats["hits"] + self._stats["misses"]) > 0
                    else 0.0
                ),
                "avg_entry_size_mb": (
                    total_size_mb / self._stats["total_entries"]
                    if self._stats["total_entries"] > 0
                    else 0.0
                ),
            }
    
    def update_access_time(self, cache_key: str):
        """Update last access time with minimal overhead."""
        entry = self._entries.get(cache_key)
        if entry is not None:
            entry['accessed_at'] = datetime.now(timezone.utc)
            # Only invalidate list cache occasionally for better performance
            if len(self._entries) % 100 == 0:  # Every 100 access updates
                self._list_cache_dirty = True
    
    def increment_hits(self):
        """Increment cache hits counter with O(1) performance."""
        self._stats["hits"] += 1
    
    def increment_misses(self):
        """Increment cache misses counter with O(1) performance."""
        self._stats["misses"] += 1
    
    def cleanup_expired(self, ttl_hours: int) -> int:
        """Remove expired entries efficiently with batch operations."""
        if ttl_hours <= 0:
            return 0
            
        with self._lock:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
            expired_keys = []
            
            # Find expired keys in single pass
            for cache_key, entry in self._entries.items():
                created_at = entry.get('created_at')
                if created_at and created_at < cutoff_time:
                    expired_keys.append(cache_key)
            
            # Remove expired entries in batch
            for cache_key in expired_keys:
                self.remove_entry(cache_key)  # Reuse optimized removal
            
            return len(expired_keys)
    
    def clear_all(self) -> int:
        """Clear all entries with O(1) performance."""
        with self._lock:
            count = self._stats["total_entries"]
            
            # Clear unified data structure efficiently - single operation
            self._entries.clear()
            
            # Reset statistics
            self._stats = {
                "hits": 0,
                "misses": 0,
                "total_entries": 0,
                "total_size": 0
            }
            
            # Clear cache
            self._list_cache = None
            self._list_cache_dirty = True
            
            return count
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load complete metadata structure (in-memory backend doesn't persist data)."""
        # In-memory backend doesn't persist - return empty dict
        return {}
    
    def save_metadata(self, metadata: Dict[str, Any]):
        """Save complete metadata structure (in-memory backend doesn't persist data)."""
        # In-memory backend doesn't persist - no-op
        pass


class JsonBackend(MetadataBackend):
    """JSON file-based metadata backend with batching support."""

    def __init__(self, metadata_file: Path):
        """
        Initialize JSON metadata backend.

        Args:
            metadata_file: Path to JSON metadata file
        """
        self.metadata_file = metadata_file
        self._lock = threading.Lock()
        self._metadata = self._load_from_disk()
        self._pending_writes = {}
        self._batch_size = 10  # Write after 10 operations
        self._write_count = 0

    def _load_from_disk(self) -> Dict[str, Any]:
        """Load metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json_loads(f.read())
            except Exception:
                logger.warning("JSON metadata corrupted, starting fresh")

        return {
            "entries": {},
            "access_times": {},
            "creation_times": {},
            "file_sizes": {},
            "data_types": {},
            "cache_key_params": {},  # Store original key-value params separately
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _save_to_disk(self):
        """Save metadata to JSON file using high-performance orjson."""
        try:
            with open(self.metadata_file, "w") as f:
                f.write(json_dumps(self._metadata, default=str))
        except Exception as e:
            logger.error(f"Failed to save JSON metadata: {e}")

    def _flush_pending_writes(self):
        """Flush pending writes to disk."""
        if self._pending_writes:
            self._metadata.update(self._pending_writes)
            self._pending_writes.clear()
            self._save_to_disk()
            self._write_count = 0

    def load_metadata(self) -> Dict[str, Any]:
        """Load complete metadata structure."""
        with self._lock:
            return self._metadata.copy()

    def save_metadata(self, metadata: Dict[str, Any]):
        """Save complete metadata structure."""
        with self._lock:
            self._metadata = metadata
            self._save_to_disk()

    def get_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get specific cache entry metadata."""
        with self._lock:
            if cache_key not in self._metadata["entries"]:
                return None

            # Extract cache_key_params and add to dedicated storage if it exists
            entry_metadata = self._metadata["entries"][cache_key].copy()
            cache_key_params = self._metadata.get("cache_key_params", {}).get(cache_key)
            if cache_key_params is not None:
                entry_metadata["cache_key_params"] = cache_key_params

            return {
                "description": self._metadata["entries"][cache_key].get(
                    "description", ""
                ),
                "data_type": self._metadata["data_types"].get(cache_key, "unknown"),
                "prefix": self._metadata["entries"][cache_key].get("prefix", ""),
                "created_at": self._metadata["creation_times"].get(cache_key),
                "accessed_at": self._metadata["access_times"].get(cache_key),
                "file_size": self._metadata["file_sizes"].get(cache_key, 0),
                "metadata": entry_metadata,
            }

    def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
        """Store cache entry metadata with batching."""
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()

            # Extract cache_key_params from metadata and store separately
            metadata = entry_data.get("metadata", {}).copy()
            cache_key_params = metadata.pop("cache_key_params", None)

            # Store the complete entry metadata, including description (but without cache_key_params)
            self._metadata["entries"][cache_key] = {
                **metadata,
                "description": entry_data.get("description", ""),
            }
            self._metadata["data_types"][cache_key] = entry_data.get(
                "data_type", "unknown"
            )
            self._metadata["creation_times"][cache_key] = entry_data.get(
                "created_at", now
            )
            self._metadata["access_times"][cache_key] = entry_data.get(
                "accessed_at", now
            )
            self._metadata["file_sizes"][cache_key] = entry_data.get("file_size", 0)

            # Store cache_key_params separately for efficient querying
            if cache_key_params is not None:
                if "cache_key_params" not in self._metadata:
                    self._metadata["cache_key_params"] = {}
                try:
                    from .serialization import serialize_for_cache_key
                    serializable_params = {
                        key: serialize_for_cache_key(value) 
                        for key, value in cache_key_params.items()
                    }
                    self._metadata["cache_key_params"][cache_key] = serializable_params
                except Exception:
                    try:
                        self._metadata["cache_key_params"][cache_key] = cache_key_params
                    except Exception:
                        pass

            # Batch writes for better performance
            self._write_count += 1
            if self._write_count >= self._batch_size:
                self._save_to_disk()
                self._write_count = 0
            else:
                # For immediate consistency, still save to disk
                self._save_to_disk()

    def remove_entry(self, cache_key: str):
        """Remove cache entry metadata."""
        with self._lock:
            for metadata_dict in self._metadata.values():
                if isinstance(metadata_dict, dict) and cache_key in metadata_dict:
                    del metadata_dict[cache_key]
            # Also specifically ensure cache_key_params is cleaned up
            if (
                "cache_key_params" in self._metadata
                and cache_key in self._metadata["cache_key_params"]
            ):
                del self._metadata["cache_key_params"][cache_key]
            self._save_to_disk()

    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries with metadata."""
        with self._lock:
            entries = []

            for cache_key, entry_data in self._metadata["entries"].items():
                creation_time = self._metadata["creation_times"].get(cache_key)
                access_time = self._metadata["access_times"].get(cache_key)
                file_size = self._metadata["file_sizes"].get(cache_key, 0)
                data_type = self._metadata["data_types"].get(cache_key, "unknown")

                # Ensure timestamps are timezone-aware when returned
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

                # Include cache_key_params in metadata if available
                complete_metadata = entry_data.copy()
                cache_key_params = self._metadata.get("cache_key_params", {}).get(
                    cache_key
                )
                if cache_key_params is not None:
                    complete_metadata["cache_key_params"] = cache_key_params

                entries.append(
                    {
                        "cache_key": cache_key,
                        "data_type": data_type,
                        "description": entry_data.get("description", ""),
                        "metadata": complete_metadata,
                        "created": creation_time,
                        "last_accessed": access_time,
                        "size_mb": round(file_size / (1024 * 1024), 3),
                    }
                )

            # Sort by creation time (newest first)
            entries.sort(key=lambda x: x["created"] or "", reverse=True)
            return entries

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_entries = len(self._metadata["entries"])
            total_size_mb = sum(self._metadata["file_sizes"].values()) / (1024 * 1024)

            # Count by data type
            dataframe_count = sum(
                1
                for dt in self._metadata.get("data_types", {}).values()
                if dt == "dataframe"
            )
            array_count = sum(
                1
                for dt in self._metadata.get("data_types", {}).values()
                if dt == "array"
            )

            # Cache hit rate
            hits = self._metadata.get("cache_hits", 0)
            misses = self._metadata.get("cache_misses", 0)
            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0

            return {
                "total_entries": total_entries,
                "dataframe_entries": dataframe_count,
                "array_entries": array_count,
                "total_size_mb": round(total_size_mb, 2),
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate": round(hit_rate, 3),
            }

    def update_access_time(self, cache_key: str):
        """Update last access time for cache entry."""
        with self._lock:
            self._metadata["access_times"][cache_key] = datetime.now(
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

    def cleanup_expired(self, ttl_hours: int) -> int:
        """Remove expired entries and return count removed."""
        from datetime import timedelta

        with self._lock:
            expired_keys = []
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)

            for cache_key, creation_time_str in self._metadata[
                "creation_times"
            ].items():
                try:
                    creation_time = datetime.fromisoformat(creation_time_str)
                    if creation_time < cutoff_time:
                        expired_keys.append(cache_key)
                except (ValueError, TypeError):
                    # Invalid timestamp, consider expired
                    expired_keys.append(cache_key)

            # Remove expired entries
            for cache_key in expired_keys:
                for metadata_dict in self._metadata.values():
                    if isinstance(metadata_dict, dict) and cache_key in metadata_dict:
                        del metadata_dict[cache_key]

            if expired_keys:
                self._save_to_disk()

            return len(expired_keys)

    def clear_all(self) -> int:
        """Remove all cache entries and return count removed."""
        with self._lock:
            entry_count = len(self._metadata.get("entries", {}))
            
            # Clear all metadata dictionaries
            self._metadata = {
                "entries": {},
                "creation_times": {},
                "access_times": {},
                "cache_hits": 0,
                "cache_misses": 0,
            }
            
            self._save_to_disk()
            return entry_count


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
        
        # Configure SQLite engine with aggressive performance optimizations
        self.engine = create_engine(
            f"sqlite:///{db_file}", 
            echo=echo,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            pool_size=20,       # Larger connection pool
            max_overflow=30,    # Allow more overflow connections
            connect_args={
                "check_same_thread": False,  # Allow multi-threading
                "timeout": 30,  # Longer timeout for database locks
            }
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
            cursor.execute("PRAGMA synchronous=NORMAL")    # Good balance of safety/speed
            cursor.execute("PRAGMA cache_size=20000")      # 20MB cache (increased from 10MB)
            cursor.execute("PRAGMA temp_store=MEMORY")     # Temp tables in memory
            cursor.execute("PRAGMA mmap_size=536870912")   # 512MB memory mapped I/O (doubled)
            cursor.execute("PRAGMA page_size=32768")       # Larger page size for better I/O
            
            # Query optimization pragmas
            cursor.execute("PRAGMA optimize")              # Enable query planner optimizations
            cursor.execute("PRAGMA analysis_limit=1000")  # Better statistics
            
            # Concurrent access optimizations
            cursor.execute("PRAGMA busy_timeout=30000")    # 30s busy timeout
            cursor.execute("PRAGMA wal_autocheckpoint=1000") # WAL checkpoint every 1000 pages
            
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys=ON")
            
            cursor.close()
        
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        self._lock = threading.Lock()

        # Create tables
        Base.metadata.create_all(self.engine)

        # Initialize stats if not exists
        self._init_stats()

        logger.info(f"✅ SQLAlchemy metadata backend initialized: {db_file}")

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
        """Get specific cache entry metadata."""
        with self.SessionLocal() as session:
            # Single optimized query with all needed data
            entry = session.execute(
                select(CacheEntry).where(CacheEntry.cache_key == cache_key)
            ).scalar_one_or_none()

            if not entry:
                return None

            # Parse metadata once
            metadata = json_loads(entry.metadata_json)
            
            # Add extracted fields back to metadata
            if entry.file_hash is not None:
                metadata["file_hash"] = entry.file_hash
            if entry.entry_signature is not None:
                metadata["entry_signature"] = entry.entry_signature
            if entry.cache_key_params is not None:
                metadata["cache_key_params"] = json_loads(entry.cache_key_params)

            return {
                "description": entry.description,
                "data_type": entry.data_type,
                "prefix": entry.prefix,
                "created_at": entry.created_at.isoformat(),
                "accessed_at": entry.accessed_at.isoformat(),
                "file_size": entry.file_size,
                "metadata": metadata,
            }

    def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
        """Store cache entry metadata using efficient upsert with optimized transaction."""
        with self._lock, self.SessionLocal() as session:
            # Extract and process metadata fields efficiently
            metadata = entry_data.get("metadata", {})
            file_hash = metadata.pop("file_hash", None)
            entry_signature = metadata.pop("entry_signature", None)
            cache_key_params = metadata.pop("cache_key_params", None)
            
            # Serialize cache_key_params efficiently (only if needed)
            serialized_params = None
            if cache_key_params is not None:
                try:
                    from .serialization import serialize_for_cache_key
                    serializable_params = {
                        key: serialize_for_cache_key(value) 
                        for key, value in cache_key_params.items()
                    }
                    serialized_params = json_dumps(serializable_params)
                except Exception:
                    try:
                        serialized_params = json_dumps(cache_key_params)
                    except Exception:
                        serialized_params = None
            
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
            
            # Use more efficient INSERT OR REPLACE instead of merge
            from sqlalchemy import text
            session.execute(
                text("""
                    INSERT OR REPLACE INTO cache_entries 
                    (cache_key, description, data_type, prefix, file_size, 
                     file_hash, entry_signature, cache_key_params, metadata_json, 
                     created_at, accessed_at)
                    VALUES (:cache_key, :description, :data_type, :prefix, :file_size, 
                           :file_hash, :entry_signature, :cache_key_params, :metadata_json, 
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
                    "cache_key_params": serialized_params,
                    "metadata_json": json_dumps(metadata),
                    "created_at": created_at,
                    "accessed_at": accessed_at
                }
            )
            session.commit()

    def remove_entry(self, cache_key: str):
        """Remove cache entry metadata and associated custom metadata links."""
        with self._lock, self.SessionLocal() as session:
            # Delete cache entry (this will cascade to metadata links due to foreign key constraint)
            session.execute(delete(CacheEntry).where(CacheEntry.cache_key == cache_key))

            # Explicit cleanup of custom metadata links (redundant but ensures consistency)
            try:
                from .custom_metadata import CacheMetadataLink

                cleaned_count = CacheMetadataLink.cleanup_for_cache_key(
                    session, cache_key
                )
                if cleaned_count > 0:
                    logger.debug(
                        f"Cleaned up {cleaned_count} custom metadata links for {cache_key}"
                    )
            except ImportError:
                # Custom metadata not available, skip cleanup
                pass
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup custom metadata links for {cache_key}: {e}"
                )

            session.commit()

    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries with metadata."""
        with self.SessionLocal() as session:
            entries = (
                session.execute(
                    select(CacheEntry).order_by(desc(CacheEntry.created_at))
                )
                .scalars()
                .all()
            )

            result = []
            for entry in entries:
                entry_metadata = json_loads(entry.metadata_json)
                # Add file_hash back into metadata if it exists
                if entry.file_hash is not None:
                    entry_metadata["file_hash"] = entry.file_hash
                # Add entry_signature back into metadata if it exists
                if entry.entry_signature is not None:
                    entry_metadata["entry_signature"] = entry.entry_signature
                # Add cache_key_params back into metadata if it exists
                if entry.cache_key_params is not None:
                    entry_metadata["cache_key_params"] = json_loads(
                        entry.cache_key_params
                    )
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
        """Get cache statistics."""
        with self.SessionLocal() as session:
            # Get counts by data type
            total_entries = session.execute(select(CacheEntry)).scalars().all()

            dataframe_count = len(
                [e for e in total_entries if e.data_type == "dataframe"]
            )
            array_count = len([e for e in total_entries if e.data_type == "array"])
            total_size_mb = sum(e.file_size for e in total_entries) / (1024 * 1024)

            # Get hit/miss stats
            stats = self._get_stats_row(session)
            hit_rate = (
                stats.cache_hits / (stats.cache_hits + stats.cache_misses)
                if (stats.cache_hits + stats.cache_misses) > 0
                else 0.0
            )

            return {
                "total_entries": len(total_entries),
                "dataframe_entries": dataframe_count,
                "array_entries": array_count,
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

    def cleanup_expired(self, ttl_hours: int) -> int:
        """Remove expired entries and return count removed."""
        from datetime import timedelta

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)

        with self._lock, self.SessionLocal() as session:
            # Delete expired entries
            result = session.execute(
                delete(CacheEntry).where(CacheEntry.created_at < cutoff_time)
            )
            deleted_count = result.rowcount
            session.commit()
            return deleted_count

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


def create_metadata_backend(backend_type: str = "auto", **kwargs) -> MetadataBackend:
    """
    Factory function to create metadata backends with optional entry caching.

    Args:
        backend_type: "auto", "memory", "sqlite", "json", or "sqlite_memory"
                     "auto" prefers SQLite (file-based) if available, falls back to in-memory SQLite, then JSON
        **kwargs: Backend-specific configuration including optional 'config' for caching

    Returns:
        MetadataBackend instance (potentially wrapped with caching)
    """
    # Extract cache config if provided
    cache_config = kwargs.pop('config', None)
    
    # Create the base backend
    if backend_type == "memory":
        backend = InMemoryBackend()
    elif backend_type == "json":
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
                    metadata_file = kwargs.get("metadata_file", Path("cache_metadata.json"))
                    backend = JsonBackend(metadata_file)
        else:
            # SQLAlchemy not available, use JSON
            metadata_file = kwargs.get("metadata_file", Path("cache_metadata.json"))
            backend = JsonBackend(metadata_file)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Supported: 'auto', 'memory', 'json', 'sqlite', 'sqlite_memory'")
    
    # Apply memory cache layer wrapper for disk-persistent backends (not memory)
    if (cache_config is not None and 
        backend_type != "memory" and 
        hasattr(cache_config, 'enable_memory_cache') and 
        cache_config.enable_memory_cache):
        
        logger.debug(f"Wrapping {backend_type} backend with memory cache layer")
        backend = CachedMetadataBackend(backend, cache_config)
    
    return backend
