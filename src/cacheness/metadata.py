"""
Cache Metadata Backend System
=============================

This module provides pluggable metadata backends for the unified cache system,
supporting both JSON file storage and SQLite database storage with SQLModel ORM.

Features:
- Abstract base class for metadata backends
- JSON file backend (original implementation)
- SQLite backend with SQLModel ORM
- Dependency injection for backend selection
- Thread-safe operations
- Migration utilities

Usage:
    # Use auto backend (defaults to SQLite if available, falls back to JSON)
    cache = UnifiedCache()

    # Force SQLite backend
    from shared.cache.metadata import SQLiteMetadataBackend
    backend = SQLiteMetadataBackend("cache.db")
    cache = UnifiedCache(metadata_backend=backend)

    # Force JSON backend
    from shared.cache.metadata import JsonMetadataBackend
    backend = JsonMetadataBackend(Path("cache_metadata.json"))
    cache = UnifiedCache(metadata_backend=backend)
"""

import json
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

import logging

logger = logging.getLogger(__name__)

try:
    from sqlmodel import SQLModel, Field, create_engine, Session, select
    from sqlalchemy import Column, Text, desc, Index

    SQLMODEL_AVAILABLE = True

    # Define SQLModel classes only when SQLModel is available
    class CacheEntry(SQLModel, table=True):
        """SQLModel for cache entry metadata."""

        __tablename__ = "cache_entries"

        cache_key: str = Field(primary_key=True, max_length=16)
        description: str = Field(default="", max_length=500)
        data_type: str = Field(
            max_length=20, index=True
        )  # "dataframe", "array", "object" - indexed for filtering
        prefix: str = Field(
            default="", max_length=100, index=True
        )  # Indexed for prefix-based queries

        # Timestamps - indexed for time-based queries and cleanup operations
        created_at: datetime = Field(
            default_factory=lambda: datetime.now(timezone.utc), index=True
        )
        accessed_at: datetime = Field(
            default_factory=lambda: datetime.now(timezone.utc), index=True
        )

        # File info - indexed for size-based queries and cache management
        file_size: int = Field(default=0, index=True)  # Size in bytes

        # Cache integrity verification
        file_hash: Optional[str] = Field(
            default=None, max_length=16
        )  # XXH3_64 hash (16 hex chars)

        # Data-specific metadata (stored as JSON)
        metadata_json: str = Field(default="{}", sa_column=Column(Text))

        # Composite indexes for common query patterns
        __table_args__ = (
            # Index for listing entries (most common query pattern)
            # ORDER BY created_at DESC - used in list_entries()
            Index("idx_created_at_desc", "created_at"),
            # Index for cleanup operations (critical for performance)
            # WHERE created_at < cutoff_time - used in cleanup_expired()
            Index("idx_cleanup_created_at", "created_at"),
            # Index for statistics generation by data type
            # COUNT(*) WHERE data_type = 'dataframe' - used in get_stats()
            Index("idx_stats_data_type", "data_type"),
            # Index for cache size management (most to least disk usage)
            # ORDER BY file_size DESC, created_at ASC - for LRU eviction of large files
            Index("idx_size_management", "file_size", "created_at"),
            # Index for access pattern analysis
            # ORDER BY accessed_at DESC - for LRU-based cleanup
            Index("idx_lru_access", "accessed_at"),
        )

    class CacheStats(SQLModel, table=True):
        """SQLModel for cache statistics."""

        __tablename__ = "cache_stats"

        id: int = Field(primary_key=True, default=1)  # Only one row
        cache_hits: int = Field(default=0, index=True)  # Indexed for analytics queries
        cache_misses: int = Field(
            default=0, index=True
        )  # Indexed for analytics queries
        last_updated: datetime = Field(
            default_factory=lambda: datetime.now(timezone.utc), index=True
        )  # Indexed for monitoring

        # Composite index for hit rate calculations
        __table_args__ = (
            Index(
                "idx_hits_misses_updated", "cache_hits", "cache_misses", "last_updated"
            ),
        )

except ImportError:
    # SQLModel not available
    SQLMODEL_AVAILABLE = False
    logger.warning("SQLModel not available, SQLite backend will not work")

    # Define dummy classes to avoid runtime errors
    class CacheEntry:
        pass

    class CacheStats:
        pass


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


class JsonMetadataBackend(MetadataBackend):
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
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning("JSON metadata corrupted, starting fresh")

        return {
            "entries": {},
            "access_times": {},
            "creation_times": {},
            "file_sizes": {},
            "data_types": {},
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _save_to_disk(self):
        """Save metadata to JSON file."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2, default=str)
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

            return {
                "description": self._metadata["entries"][cache_key].get(
                    "description", ""
                ),
                "data_type": self._metadata["data_types"].get(cache_key, "unknown"),
                "prefix": self._metadata["entries"][cache_key].get("prefix", ""),
                "created_at": self._metadata["creation_times"].get(cache_key),
                "accessed_at": self._metadata["access_times"].get(cache_key),
                "file_size": self._metadata["file_sizes"].get(cache_key, 0),
                "metadata": self._metadata["entries"][cache_key],
            }

    def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
        """Store cache entry metadata."""
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()

            # Store the complete entry metadata, including description
            self._metadata["entries"][cache_key] = {
                **entry_data.get("metadata", {}),
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

            self._save_to_disk()

    def remove_entry(self, cache_key: str):
        """Remove cache entry metadata."""
        with self._lock:
            for metadata_dict in self._metadata.values():
                if isinstance(metadata_dict, dict) and cache_key in metadata_dict:
                    del metadata_dict[cache_key]
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

                entries.append(
                    {
                        "cache_key": cache_key,
                        "data_type": data_type,
                        "description": entry_data.get("description", ""),
                        "metadata": entry_data,
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


class SQLiteMetadataBackend(MetadataBackend):
    """SQLite database-based metadata backend using SQLModel ORM."""

    def __init__(self, db_file: str = "cache_metadata.db", echo: bool = False):
        """
        Initialize SQLite metadata backend.

        Args:
            db_file: Path to SQLite database file
            echo: Whether to echo SQL queries (for debugging)
        """
        if not SQLMODEL_AVAILABLE:
            raise ImportError(
                "SQLModel is required for SQLite backend. Install with: pip install sqlmodel"
            )

        self.db_file = db_file
        self.engine = create_engine(f"sqlite:///{db_file}", echo=echo)
        self._lock = threading.Lock()

        # Create tables
        SQLModel.metadata.create_all(self.engine)

        # Initialize stats if not exists
        self._init_stats()

        logger.info(f"✅ SQLite metadata backend initialized: {db_file}")

    def _init_stats(self):
        """Initialize cache stats if not exists."""
        with Session(self.engine) as session:
            stats = session.get(CacheStats, 1)
            if not stats:
                stats = CacheStats(id=1)
                session.add(stats)
                session.commit()

    def _get_stats_row(self, session) -> "CacheStats":
        """Get the single stats row."""
        stats = session.get(CacheStats, 1)
        if not stats:
            stats = CacheStats(id=1)
            session.add(stats)
            session.commit()
        return stats

    def load_metadata(self) -> Dict[str, Any]:
        """Load complete metadata structure (compatibility method)."""
        # This method is for compatibility with the original JSON structure
        # In practice, the SQLite backend doesn't need to load everything at once
        with Session(self.engine) as session:
            entries = session.exec(select(CacheEntry)).all()
            stats = self._get_stats_row(session)

            metadata = {
                "entries": {},
                "access_times": {},
                "creation_times": {},
                "file_sizes": {},
                "data_types": {},
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
            }

            for entry in entries:
                entry_metadata = json.loads(entry.metadata_json)
                # Add file_hash back into metadata if it exists
                if entry.file_hash is not None:
                    entry_metadata["file_hash"] = entry.file_hash
                metadata["entries"][entry.cache_key] = entry_metadata
                metadata["access_times"][entry.cache_key] = (
                    entry.accessed_at.isoformat()
                )
                metadata["creation_times"][entry.cache_key] = (
                    entry.created_at.isoformat()
                )
                metadata["file_sizes"][entry.cache_key] = entry.file_size
                metadata["data_types"][entry.cache_key] = entry.data_type

            return metadata

    def save_metadata(self, metadata: Dict[str, Any]):
        """Save complete metadata structure (compatibility method)."""
        # This method is mainly for compatibility
        # The SQLite backend saves individual entries as they're updated
        with self._lock, Session(self.engine) as session:
            # Update stats
            stats = self._get_stats_row(session)
            stats.cache_hits = metadata.get("cache_hits", 0)
            stats.cache_misses = metadata.get("cache_misses", 0)
            stats.last_updated = datetime.now(timezone.utc)
            session.add(stats)
            session.commit()

    def get_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get specific cache entry metadata."""
        with Session(self.engine) as session:
            entry = session.get(CacheEntry, cache_key)
            if not entry:
                return None

            metadata = json.loads(entry.metadata_json)
            # Add file_hash back into metadata if it exists
            if entry.file_hash is not None:
                metadata["file_hash"] = entry.file_hash

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
        with self._lock, Session(self.engine) as session:
            # Get existing entry or create new one
            entry = session.get(CacheEntry, cache_key)
            if not entry:
                entry = CacheEntry(cache_key=cache_key)

            # Update fields
            entry.description = entry_data.get("description", "")
            entry.data_type = entry_data.get("data_type", "unknown")
            entry.prefix = entry_data.get("prefix", "")
            entry.file_size = entry_data.get("file_size", 0)

            # Extract file_hash from metadata and store in dedicated column
            metadata = entry_data.get("metadata", {})
            entry.file_hash = metadata.pop("file_hash", None)  # Remove from metadata
            entry.metadata_json = json.dumps(metadata)

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

            session.add(entry)
            session.commit()

    def remove_entry(self, cache_key: str):
        """Remove cache entry metadata."""
        with self._lock, Session(self.engine) as session:
            entry = session.get(CacheEntry, cache_key)
            if entry:
                session.delete(entry)
                session.commit()

    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries with metadata."""
        with Session(self.engine) as session:
            entries = session.exec(
                select(CacheEntry).order_by(desc(CacheEntry.created_at))
            ).all()

            result = []
            for entry in entries:
                entry_metadata = json.loads(entry.metadata_json)
                # Add file_hash back into metadata if it exists
                if entry.file_hash is not None:
                    entry_metadata["file_hash"] = entry.file_hash
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
        with Session(self.engine) as session:
            # Get counts by data type
            total_entries = session.exec(select(CacheEntry)).all()

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
        with self._lock, Session(self.engine) as session:
            entry = session.get(CacheEntry, cache_key)
            if entry:
                entry.accessed_at = datetime.now(timezone.utc)
                session.add(entry)
                session.commit()

    def increment_hits(self):
        """Increment cache hits counter."""
        with self._lock, Session(self.engine) as session:
            stats = self._get_stats_row(session)
            stats.cache_hits += 1
            stats.last_updated = datetime.now(timezone.utc)
            session.add(stats)
            session.commit()

    def increment_misses(self):
        """Increment cache misses counter."""
        with self._lock, Session(self.engine) as session:
            stats = self._get_stats_row(session)
            stats.cache_misses += 1
            stats.last_updated = datetime.now(timezone.utc)
            session.add(stats)
            session.commit()

    def cleanup_expired(self, ttl_hours: int) -> int:
        """Remove expired entries and return count removed."""
        from datetime import timedelta

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)

        with self._lock, Session(self.engine) as session:
            # Find expired entries
            expired_entries = session.exec(
                select(CacheEntry).where(CacheEntry.created_at < cutoff_time)
            ).all()

            # Delete expired entries
            for entry in expired_entries:
                session.delete(entry)

            session.commit()
            return len(expired_entries)


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
        return JsonMetadataBackend(metadata_file)
    elif backend_type == "sqlite":
        if not SQLMODEL_AVAILABLE:
            raise ImportError(
                "SQLModel is required for SQLite backend but is not available. Install with: uv add sqlmodel"
            )
        db_file = kwargs.get("db_file", "cache_metadata.db")
        echo = kwargs.get("echo", False)
        return SQLiteMetadataBackend(db_file, echo)
    elif backend_type == "auto":
        # Auto mode: prefer SQLite, fallback to JSON
        if SQLMODEL_AVAILABLE:
            try:
                db_file = kwargs.get("db_file", "cache_metadata.db")
                echo = kwargs.get("echo", False)
                return SQLiteMetadataBackend(db_file, echo)
            except Exception:
                # Fall back to JSON if SQLite fails
                metadata_file = kwargs.get("metadata_file", Path("cache_metadata.json"))
                return JsonMetadataBackend(metadata_file)
        else:
            # SQLModel not available, use JSON
            metadata_file = kwargs.get("metadata_file", Path("cache_metadata.json"))
            return JsonMetadataBackend(metadata_file)
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. Use 'auto', 'sqlite', or 'json'"
        )
