"""
Abstract Base Class for Metadata Backends
========================================

Defines the interface that all metadata backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class MetadataBackend(ABC):
    """
    Abstract base class for cache metadata backends.
    
    This interface defines the contract for storing and retrieving cache entry
    metadata. Implementations can use various storage mechanisms:
    - JSON files (simple, portable)
    - SQLite databases (queryable, concurrent access)
    - In-memory dictionaries (fast, ephemeral)
    - Redis/external databases (distributed)
    
    All implementations must be thread-safe.
    """

    @abstractmethod
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load complete metadata structure.
        
        Returns:
            Dictionary containing all cache metadata including entries and stats.
        """
        pass

    @abstractmethod
    def save_metadata(self, metadata: Dict[str, Any]):
        """
        Save complete metadata structure.
        
        Args:
            metadata: Complete metadata dictionary to persist.
        """
        pass

    @abstractmethod
    def get_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get specific cache entry metadata.
        
        Args:
            cache_key: The unique identifier for the cache entry.
            
        Returns:
            Entry metadata dictionary, or None if not found.
        """
        pass

    @abstractmethod
    def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
        """
        Store cache entry metadata.
        
        Args:
            cache_key: The unique identifier for the cache entry.
            entry_data: Metadata dictionary to store.
        """
        pass

    @abstractmethod
    def remove_entry(self, cache_key: str):
        """
        Remove cache entry metadata.
        
        Args:
            cache_key: The unique identifier for the cache entry to remove.
        """
        pass

    @abstractmethod
    def list_entries(self) -> List[Dict[str, Any]]:
        """
        List all cache entries with metadata.
        
        Returns:
            List of all entry metadata dictionaries.
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics (hits, misses, entry count, etc.)
        """
        pass

    @abstractmethod
    def update_access_time(self, cache_key: str):
        """
        Update last access time for cache entry.
        
        Args:
            cache_key: The unique identifier for the cache entry.
        """
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
        """
        Remove expired entries and return count removed.
        
        Args:
            ttl_seconds: Time-to-live in seconds. Entries older than this are removed.
            
        Returns:
            Number of entries removed.
        """
        pass

    @abstractmethod
    def clear_all(self) -> int:
        """
        Remove all cache entries and return count removed.
        
        Returns:
            Number of entries removed.
        """
        pass

    def close(self):
        """
        Close and clean up any resources.
        
        Default implementation does nothing. Override in backends that hold
        resources like database connections or file handles.
        """
        pass
    
    def __enter__(self):
        """Support context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are cleaned up."""
        self.close()
        return False
