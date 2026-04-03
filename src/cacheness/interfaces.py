"""
Cache Handler Interfaces
=======================

This module defines focused interfaces for cache handlers, following the Interface Segregation Principle.
Each interface is responsible for a specific aspect of cache handling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed contracts for metadata dicts
# ---------------------------------------------------------------------------


class SignableFields(TypedDict, total=False):
    """Superset of fields that may be included in cache-entry HMAC signatures.

    Built by ``UnifiedCache._extract_signable_fields()`` and consumed by
    ``CacheEntrySigner.sign_entry()`` / ``verify_entry()``.

    ``total=False`` because individual fields may legitimately be ``None``
    (the signer coerces missing values to empty strings).
    """

    cache_key: str
    data_type: str
    file_size: int
    created_at: str  # ISO-format, timezone stripped for consistency
    actual_path: str
    file_hash: Optional[str]
    object_type: Optional[str]
    storage_format: Optional[str]
    serializer: Optional[str]
    compression_codec: Optional[str]


class EntrySummary(TypedDict, total=False):
    """Lightweight flat dict returned by ``iter_entry_summaries()``.

    Required keys are always present; optional keys appear only when the
    backend column is non-NULL.  All backends (JSON, SQLite, PostgreSQL)
    MUST return at least the required keys.

    Unlike ``list_entries()``, timestamps are **raw** (no isoformat conversion),
    there is no nested ``metadata`` dict, and no ``size_mb`` calculation.
    """

    # --- always present ---
    cache_key: str
    data_type: str
    description: str
    created_at: Any  # raw timestamp — str or datetime depending on backend
    accessed_at: Any
    file_size: int

    # --- present when non-NULL ---
    object_type: str
    storage_format: str
    serializer: str
    compression_codec: str
    actual_path: str
    file_hash: str
    entry_signature: str
    metadata_dict: str
    s3_etag: str
    access_count: int
    ttl_seconds: int
    expires_at: Any  # raw timestamp — str or datetime depending on backend
    is_inline: int  # 1 if blob data is stored inline in metadata, 0 otherwise
    blob_data: bytes  # raw inline blob bytes (only present when is_inline=1)


@dataclass
class RotationResult:
    """Result of a key rotation operation.

    Returned by ``UnifiedCache.rotate_key()`` and ``BlobStore.rotate_key()``.
    """

    total: int = 0
    re_signed: int = 0
    re_encrypted: int = 0
    failed: int = 0
    skipped: int = 0
    failures: list = field(default_factory=list)


class BlobReadContext(TypedDict, total=False):
    """Metadata dict passed to ``handler.get()`` during deserialization.

    Contains handler-specific fields written during ``put()`` plus
    metadata columns like ``actual_path`` and ``file_hash``.

    All keys are optional (``total=False``) because each handler reads
    only the subset it needs.  The dict is built from the entry's
    ``metadata`` sub-dict by ``UnifiedCache`` before calling ``_read_blob``.

    This is a **documentation-only** contract — existing handlers and
    plugins that accept ``Dict[str, Any]`` remain compatible.
    """

    # Storage / serialization info (ObjectHandler, ArrayHandler)
    storage_format: str
    serializer: str
    compression_codec: str
    object_type: str
    actual_path: str
    file_hash: Optional[str]

    # Series metadata (PandasSeriesHandler, PolarsSeriesHandler)
    is_series: bool
    series_name: str

    # Signature (present when entry signing is enabled)
    entry_signature: str


class EntryData(TypedDict, total=False):
    """Canonical shape returned by ``get_entry()`` / accepted by ``put_entry()``.

    All metadata backends (JSON, SQLite, PostgreSQL) produce and consume
    dicts conforming to this contract.  ``total=False`` because optional
    fields may be absent depending on the backend or entry state.

    **Structure:** Top-level keys are the "envelope" (description, timing,
    size).  The nested ``metadata`` dict holds handler-written fields
    (storage format, hashes, handler extras).

    **Backend divergences:**

    * PostgreSQL includes ``cache_key`` at the top level (informational).
    * ``metadata`` sub-keys vary by handler — only ``actual_path`` and
      ``storage_format`` are reliably present for on-disk entries.
    """

    # --- always present from all backends ---
    description: str
    data_type: str
    created_at: Any  # ISO str or float timestamp depending on backend
    accessed_at: Any
    file_size: int
    metadata: Dict[str, Any]  # nested handler / storage metadata

    # --- present in some backends ---
    cache_key: str  # PostgreSQL includes this; JSON/SQLite do not
    access_count: int  # per-entry access counter (0 if never read)
    ttl_seconds: int  # per-entry TTL in seconds (None = use config default)
    expires_at: Any  # expiry timestamp (None = no TTL)
    is_inline: int  # 1 if blob data is stored inline in metadata, 0 otherwise
    blob_data: bytes  # raw inline blob bytes (only present when is_inline=1)


@dataclass
class HandlerResult:
    """Typed return contract for handler put() methods.

    Replaces the untyped Dict[str, Any] previously returned by handlers.
    Eliminates top-level vs nested key ambiguity (root cause of CACHE-198).

    Top-level fields map to dedicated metadata columns in the backend.
    The ``extra`` dict carries handler-specific metadata (shape, dtypes,
    backend, is_series, etc.) that goes into the nested metadata blob.

    Provides dict-compatible accessors (__getitem__, get, setdefault)
    so existing code that treats the result as a dict continues to work
    during the migration period.
    """

    storage_format: str
    file_size: int
    actual_path: str
    compression_codec: Optional[str] = None
    serializer: Optional[str] = None
    object_type: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- dict-compatible accessors for transitional use --

    def _as_legacy_dict(self) -> Dict[str, Any]:
        """Return the legacy dict representation for backward compatibility."""
        metadata = dict(self.extra)
        if self.compression_codec is not None:
            metadata["compression_codec"] = self.compression_codec
        if self.serializer is not None:
            metadata["serializer"] = self.serializer
        if self.object_type is not None:
            metadata["object_type"] = self.object_type
        # Some handlers duplicated storage_format inside metadata
        metadata.setdefault("storage_format", self.storage_format)
        return {
            "storage_format": self.storage_format,
            "file_size": self.file_size,
            "actual_path": self.actual_path,
            "metadata": metadata,
        }

    def __getitem__(self, key: str) -> Any:
        d = self._as_legacy_dict()
        return d[key]

    def __contains__(self, key: object) -> bool:
        """Support ``'actual_path' in result`` checks."""
        d = self._as_legacy_dict()
        return key in d

    def get(self, key: str, default: Any = None) -> Any:
        d = self._as_legacy_dict()
        return d.get(key, default)

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Support ``result.setdefault('metadata', {})`` pattern in blob_store."""
        if key == "metadata":
            return self.extra
        d = self._as_legacy_dict()
        return d.setdefault(key, default)


@dataclass
class WriteBlobResult:
    """Typed return contract for ``BlobStore._write_blob()``.

    Replaces the unnamed ``tuple[CacheHandler, HandlerResult, Optional[str]]``
    previously returned.  Named fields make call-site intent explicit and
    prevent positional-index mistakes.
    """

    handler: Any  # CacheHandler (avoiding circular import)
    result: HandlerResult
    file_hash: Optional[str] = None


@dataclass
class IntegrityReport:
    """Typed return contract for ``verify_integrity()``.

    Replaces the untyped ``Dict[str, Any]`` previously returned.
    Provides attribute access while keeping dict-compatible accessors
    so existing ``report["orphaned_blobs"]`` patterns continue to work.
    """

    orphaned_blobs: List[str] = field(default_factory=list)
    dangling_entries: List[Dict[str, Any]] = field(default_factory=list)
    size_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    hash_mismatches: Optional[List[Dict[str, Any]]] = None
    signature_failures: Optional[List[Dict[str, Any]]] = None
    repaired: Optional[Dict[str, Any]] = None

    # -- dict-compatible accessors (76+ test accesses use report["key"]) --

    _FIELDS = frozenset(
        {
            "orphaned_blobs",
            "dangling_entries",
            "size_mismatches",
            "hash_mismatches",
            "signature_failures",
            "repaired",
        }
    )

    def _as_dict(self) -> Dict[str, Any]:
        """Return a dict mirroring the legacy report structure.

        Only includes ``hash_mismatches`` / ``repaired`` when they are set
        (matching the old conditional-key behaviour).
        """
        d: Dict[str, Any] = {
            "orphaned_blobs": self.orphaned_blobs,
            "dangling_entries": self.dangling_entries,
            "size_mismatches": self.size_mismatches,
        }
        if self.hash_mismatches is not None:
            d["hash_mismatches"] = self.hash_mismatches
        if self.signature_failures is not None:
            d["signature_failures"] = self.signature_failures
        if self.repaired is not None:
            d["repaired"] = self.repaired
        return d

    def __getitem__(self, key: str) -> Any:
        d = self._as_dict()
        return d[key]

    def __contains__(self, key: object) -> bool:
        return key in self._as_dict()

    def get(self, key: str, default: Any = None) -> Any:
        return self._as_dict().get(key, default)

    def __len__(self) -> int:
        """Number of keys in the report (matches dict len)."""
        return len(self._as_dict())

    def __iter__(self):
        """Iterate over keys (matches dict iteration)."""
        return iter(self._as_dict())

    def keys(self):
        return self._as_dict().keys()

    def values(self):
        return self._as_dict().values()

    def items(self):
        return self._as_dict().items()


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
    def put(self, data: Any, file_path: Path, config: Any) -> HandlerResult:
        """
        Store data to cache and return metadata.

        Args:
            data: The data to cache
            file_path: Base file path (without extension)
            config: Cache configuration

        Returns:
            HandlerResult with storage metadata.

        Raises:
            CacheWriteError: If data cannot be written
        """
        pass

    def put_bytes(self, data: Any, config: Any) -> tuple[bytes, HandlerResult]:
        """Serialize *data* to bytes in-memory (zero disk I/O).

        This is an optional fast-path used by the inline-blob machinery in
        :pyclass:`UnifiedCache` to avoid a write→read round-trip when the
        blob is small enough to embed in the metadata row.

        The returned :class:`HandlerResult` carries the same metadata fields
        as :meth:`put` (``storage_format``, ``compression_codec``, etc.)
        but ``actual_path`` is set to ``""`` because no file was written.

        Args:
            data: The data to serialize.
            config: Cache configuration.

        Returns:
            ``(blob_bytes, result)`` — the raw serialized bytes and a
            :class:`HandlerResult` describing the serialization.

        Raises:
            NotImplementedError: Handler does not support in-memory
                serialization (caller should fall back to :meth:`put`
                followed by a file read-back).
        """
        raise NotImplementedError


class CacheReader(ABC):
    """Interface for reading data from cache."""

    @abstractmethod
    def get(self, file_path: Path, metadata: "BlobReadContext") -> Any:
        """
        Retrieve data from cache file.

        Args:
            file_path: Path to the cached file
            metadata: Handler metadata from when data was cached.
                See :class:`BlobReadContext` for available keys.

        Returns:
            The cached data

        Raises:
            CacheReadError: If data cannot be read
        """
        pass

    def get_bytes(self, blob: bytes, metadata: "BlobReadContext") -> Any:
        """Deserialize *blob* bytes in-memory (zero disk I/O).

        This is the read-side counterpart of :meth:`CacheWriter.put_bytes`.
        Used by the inline-blob machinery to reconstruct the original object
        directly from the bytes stored in the metadata row, without writing
        a temporary file on disk.

        Args:
            blob: Raw serialized bytes (as produced by :meth:`put_bytes`).
            metadata: Handler metadata from when data was cached.
                See :class:`BlobReadContext` for available keys.

        Returns:
            The deserialized data.

        Raises:
            NotImplementedError: Handler does not support in-memory
                deserialization (caller should fall back to the temp-file
                path via :meth:`get`).
        """
        raise NotImplementedError


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
