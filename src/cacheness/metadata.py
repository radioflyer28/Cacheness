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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

import logging

from .json_utils import dumps as json_dumps, loads as json_loads

logger = logging.getLogger(__name__)

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
        data_type = Column(
            String(20), index=True, nullable=False
        )  # "dataframe", "array", "object"
        prefix = Column(String(100), default="", index=True, nullable=False)

        # Timestamps - indexed for time-based queries and cleanup operations
        created_at = Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            index=True,
            nullable=False,
        )
        accessed_at = Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            index=True,
            nullable=False,
        )

        # File info - indexed for size-based queries and cache management
        file_size = Column(Integer, default=0, index=True, nullable=False)

        # Cache integrity verification
        file_hash = Column(String(16), nullable=True)  # XXH3_64 hash (16 hex chars)

        # Original cache key parameters (extracted from metadata for efficient querying)
        cache_key_params = Column(
            Text, nullable=True, index=True
        )  # JSON string of original key-value pairs

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

        # Composite indexes for common query patterns
        __table_args__ = (
            # Index for listing entries (most common query pattern)
            Index("idx_created_at_desc", "created_at"),
            # Index for cleanup operations (critical for performance)
            Index("idx_cleanup_created_at", "created_at"),
            # Index for statistics generation by data type
            Index("idx_stats_data_type", "data_type"),
            # Index for cache size management
            Index("idx_size_management", "file_size", "created_at"),
            # Index for access pattern analysis
            Index("idx_lru_access", "accessed_at"),
        )

    class CacheStats(Base):
        """SQLAlchemy model for cache statistics."""

        __tablename__ = "cache_stats"

        id = Column(Integer, primary_key=True, default=1)
        cache_hits = Column(Integer, default=0, index=True, nullable=False)
        cache_misses = Column(Integer, default=0, index=True, nullable=False)
        last_updated = Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            index=True,
            nullable=False,
        )

        # Composite index for hit rate calculations
        __table_args__ = (
            Index(
                "idx_hits_misses_updated", "cache_hits", "cache_misses", "last_updated"
            ),
        )

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


class JsonBackend(MetadataBackend):
    """JSON file-based metadata backend (original implementation)."""

    def __init__(self, metadata_file: Path):
        """
        Initialize JSON metadata backend.

        Args:
            metadata_file: Path to JSON metadata file
        """
        self.metadata_file = metadata_file
        self._lock = threading.Lock()
        self._metadata = self._load_from_disk()

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
        """Save metadata to JSON file."""
        try:
            with open(self.metadata_file, "w") as f:
                f.write(json_dumps(self._metadata, default=str))
        except Exception as e:
            logger.error(f"Failed to save JSON metadata: {e}")

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
        """Store cache entry metadata."""
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

            # Store cache_key_params separately for efficient querying with unified serialization (only if params exist)
            if cache_key_params is not None:
                if "cache_key_params" not in self._metadata:
                    self._metadata["cache_key_params"] = {}
                try:
                    from .serialization import serialize_for_cache_key

                    # Convert each value to a serializable format
                    serializable_params = {}
                    for key, value in cache_key_params.items():
                        serializable_params[key] = serialize_for_cache_key(value)
                    self._metadata["cache_key_params"][cache_key] = serializable_params
                except Exception:
                    # Fallback to original parameters, which might work for simple types
                    try:
                        self._metadata["cache_key_params"][cache_key] = cache_key_params
                    except Exception:
                        # If all else fails, don't store cache_key_params
                        pass

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
        self.engine = create_engine(f"sqlite:///{db_file}", echo=echo)
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
            entry = session.execute(
                select(CacheEntry).where(CacheEntry.cache_key == cache_key)
            ).scalar_one_or_none()

            if not entry:
                return None

            metadata = json_loads(entry.metadata_json)
            # Add file_hash back into metadata if it exists
            if entry.file_hash is not None:
                metadata["file_hash"] = entry.file_hash
            # Add cache_key_params back into metadata if it exists
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
        """Store cache entry metadata."""
        with self._lock, self.SessionLocal() as session:
            # Get existing entry or create new one
            entry = session.execute(
                select(CacheEntry).where(CacheEntry.cache_key == cache_key)
            ).scalar_one_or_none()

            if not entry:
                entry = CacheEntry(cache_key=cache_key)

            # Update fields
            entry.description = entry_data.get("description", "")
            entry.data_type = entry_data.get("data_type", "unknown")
            entry.prefix = entry_data.get("prefix", "")
            entry.file_size = entry_data.get("file_size", 0)

            # Extract file_hash and cache_key_params from metadata and store in dedicated columns
            metadata = entry_data.get("metadata", {})
            entry.file_hash = metadata.pop("file_hash", None)  # Remove from metadata
            cache_key_params = metadata.pop(
                "cache_key_params", None
            )  # Remove from metadata

            # Use unified serialization for cache_key_params to handle complex objects (only if params exist)
            if cache_key_params is not None:
                try:
                    from .serialization import serialize_for_cache_key

                    # Convert each value to a serializable format
                    serializable_params = {}
                    for key, value in cache_key_params.items():
                        serializable_params[key] = serialize_for_cache_key(value)
                    entry.cache_key_params = json_dumps(serializable_params)
                except Exception:
                    # Fallback to simple JSON serialization, or None if it fails
                    try:
                        entry.cache_key_params = json_dumps(cache_key_params)
                    except Exception:
                        entry.cache_key_params = None
            else:
                entry.cache_key_params = None
            entry.metadata_json = json_dumps(metadata)

            # Update timestamps
            if "created_at" in entry_data:
                if isinstance(entry_data["created_at"], str):
                    entry.created_at = datetime.fromisoformat(entry_data["created_at"])
                else:
                    entry.created_at = entry_data["created_at"]

            if "accessed_at" in entry_data:
                if isinstance(entry_data["accessed_at"], str):
                    entry.accessed_at = datetime.fromisoformat(
                        entry_data["accessed_at"]
                    )
                else:
                    entry.accessed_at = entry_data["accessed_at"]

            session.merge(entry)
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
    Factory function to create metadata backends.

    Args:
        backend_type: "auto", "sqlite", or "json"
                     "auto" prefers SQLite if available, falls back to JSON
        **kwargs: Backend-specific configuration

    Returns:
        MetadataBackend instance
    """
    if backend_type == "json":
        metadata_file = kwargs.get("metadata_file", Path("cache_metadata.json"))
        return JsonBackend(metadata_file)
    elif backend_type == "sqlite":
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is required for SQLite backend but is not available. Install with: uv add sqlalchemy"
            )
        db_file = kwargs.get("db_file", "cache_metadata.db")
        echo = kwargs.get("echo", False)
        return SqliteBackend(db_file, echo)
    elif backend_type == "auto":
        # Auto mode: prefer SQLite, fallback to JSON
        if SQLALCHEMY_AVAILABLE:
            try:
                db_file = kwargs.get("db_file", "cache_metadata.db")
                echo = kwargs.get("echo", False)
                return SqliteBackend(db_file, echo)
            except Exception:
                # Fall back to JSON if SQLite fails
                metadata_file = kwargs.get("metadata_file", Path("cache_metadata.json"))
                return JsonBackend(metadata_file)
        else:
            metadata_file = kwargs.get("metadata_file", Path("cache_metadata.json"))
            return JsonBackend(metadata_file)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
