"""
Cache Handler Interfaces
=======================

This module defines focused interfaces for cache handlers, following the Interface Segregation Principle.
Each interface is responsible for a specific aspect of cache handling.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CacheabilityChecker(ABC):
    """Interface for checking if data can be cached with a specific handler."""

    @abstractmethod
    def can_handle(self, data: Any) -> bool:
        """
        Check if this handler can process the given data type.

        Args:
            data: The data to check

        Returns:
            True if this handler can cache the data, False otherwise
        """
        pass


class CacheWriter(ABC):
    """Interface for writing data to cache."""

    @abstractmethod
    def put(self, data: Any, file_path: Path, config: Any) -> Dict[str, Any]:
        """
        Store data to cache and return metadata.

        Args:
            data: The data to cache
            file_path: Base file path (without extension)
            config: Cache configuration

        Returns:
            Dictionary containing:
                - storage_format: Format used for storage
                - file_size: Size of cached file in bytes
                - actual_path: Actual file path used (with extension)
                - metadata: Handler-specific metadata

        Raises:
            CacheWriteError: If data cannot be written
        """
        pass


class CacheReader(ABC):
    """Interface for reading data from cache."""

    @abstractmethod
    def get(self, file_path: Path, metadata: Dict[str, Any]) -> Any:
        """
        Retrieve data from cache file.

        Args:
            file_path: Path to the cached file
            metadata: Metadata from when data was cached

        Returns:
            The cached data

        Raises:
            CacheReadError: If data cannot be read
        """
        pass


class FormatProvider(ABC):
    """Interface for providing format information."""

    @abstractmethod
    def get_file_extension(self, config: Any) -> str:
        """
        Get the file extension used by this handler.

        Args:
            config: Cache configuration

        Returns:
            File extension including the dot (e.g., '.parquet')
        """
        pass

    @property
    @abstractmethod
    def data_type(self) -> str:
        """
        Return the data type identifier for this handler.

        Returns:
            String identifier for the data type (e.g., 'pandas_dataframe')
        """
        pass


class CacheHandler(CacheabilityChecker, CacheWriter, CacheReader, FormatProvider):
    """
    Complete cache handler interface combining all capabilities.

    This is a convenience interface for handlers that implement all functionality.
    Handlers can also implement individual interfaces for more focused responsibilities.
    """

    pass


# Specific handler interfaces for different data categories
class DataFrameHandler(CacheHandler):
    """Specialized interface for DataFrame handlers."""

    @abstractmethod
    def validate_dataframe(self, data: Any) -> bool:
        """
        Validate that the DataFrame can be cached in the target format.

        Args:
            data: DataFrame to validate

        Returns:
            True if DataFrame is compatible with handler's storage format
        """
        pass


class SeriesHandler(CacheHandler):
    """Specialized interface for Series handlers."""

    @abstractmethod
    def preserve_series_metadata(
        self, data: Any, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Preserve Series-specific metadata during caching.

        Args:
            data: Series data
            metadata: Base metadata dictionary

        Returns:
            Enhanced metadata with Series-specific information
        """
        pass


class ArrayHandler(CacheHandler):
    """Specialized interface for array handlers."""

    @abstractmethod
    def optimize_array_storage(self, data: Any, config: Any) -> str:
        """
        Determine optimal storage format for the given array.

        Args:
            data: Array data
            config: Cache configuration

        Returns:
            Recommended storage format ('blosc2', 'npz', etc.)
        """
        pass


class ObjectHandler(CacheHandler):
    """Specialized interface for general object handlers."""

    @abstractmethod
    def validate_pickleable(self, data: Any) -> bool:
        """
        Validate that the object can be pickled.

        Args:
            data: Object to validate

        Returns:
            True if object can be safely pickled
        """
        pass


# Exception classes for handler errors
class CacheHandlerError(Exception):
    """Base exception for cache handler errors."""

    def __init__(
        self,
        message: str,
        handler_type: Optional[str] = None,
        data_type: Optional[str] = None,
    ):
        self.handler_type = handler_type
        self.data_type = data_type
        super().__init__(message)

        # Log the error for debugging
        logger.error(
            f"Cache handler error: {message} (handler={handler_type}, data_type={data_type})"
        )


class CacheWriteError(CacheHandlerError):
    """Exception raised when data cannot be written to cache."""

    pass


class CacheReadError(CacheHandlerError):
    """Exception raised when data cannot be read from cache."""

    pass


class CacheFormatError(CacheHandlerError):
    """Exception raised when data format is incompatible with handler."""

    pass


class CacheValidationError(CacheHandlerError):
    """Exception raised when data validation fails."""

    pass


# Factory interface for creating handlers
class HandlerFactory(ABC):
    """Interface for creating cache handlers."""

    @abstractmethod
    def create_handler(self, data_type: str, config: Any = None) -> CacheHandler:
        """
        Create a handler for the specified data type.

        Args:
            data_type: Type of data to handle
            config: Optional configuration

        Returns:
            Appropriate cache handler instance

        Raises:
            ValueError: If no handler available for data type
        """
        pass

    @abstractmethod
    def get_available_handlers(self) -> Dict[str, type]:
        """
        Get all available handler types.

        Returns:
            Dictionary mapping data type names to handler classes
        """
        pass


# Registry interface for managing handlers
class HandlerRegistry(ABC):
    """Interface for registering and retrieving cache handlers."""

    @abstractmethod
    def register_handler(self, handler: CacheHandler, priority: int = 0) -> None:
        """
        Register a new cache handler.

        Args:
            handler: Handler instance to register
            priority: Priority for handler selection (higher = higher priority)
        """
        pass

    @abstractmethod
    def get_handler(self, data: Any) -> CacheHandler:
        """
        Get the most appropriate handler for the given data.

        Args:
            data: Data to find handler for

        Returns:
            Best matching handler

        Raises:
            ValueError: If no suitable handler found
        """
        pass

    @abstractmethod
    def get_handler_by_type(self, data_type: str) -> CacheHandler:
        """
        Get handler by data type identifier.

        Args:
            data_type: Data type identifier

        Returns:
            Handler for the specified data type

        Raises:
            ValueError: If no handler found for data type
        """
        pass

    @abstractmethod
    def list_handlers(self) -> Dict[str, CacheHandler]:
        """
        List all registered handlers.

        Returns:
            Dictionary mapping data types to handlers
        """
        pass
