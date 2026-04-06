"""Shared imports, ORM models, namespace utilities, and availability flags for metadata backends."""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Callable

import logging


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
Migration = Tuple[int, int, Callable[["object", str], None]]

# Check cachetools availability
try:
    import cachetools  # noqa: F401

    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    logger.debug("cachetools not available, entry caching disabled")

try:
    from sqlalchemy import (
        Column,
        String,
        Integer,
        DateTime,
        Text,
        LargeBinary,
        Index,
        desc,
        text,
    )
    from sqlalchemy.orm import declarative_base

    SQLALCHEMY_AVAILABLE = True

    # Create declarative base
    Base = declarative_base()

    # ------------------------------------------------------------------
    # Abstract mixin bases for the EntityName pattern (dynamic table
    # names per namespace).  Concrete subclasses provide __tablename__.
    # ------------------------------------------------------------------

    class CacheEntryMixin:
        """Column definitions shared by all cache_entries tables."""

        cache_key = Column(String(16), primary_key=True)
        description = Column(String(500), default="", nullable=False)
        data_type = Column(String(20), nullable=False)

        created_at = Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        )
        accessed_at = Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        )

        file_size = Column(Integer, default=0, nullable=False)
        file_hash = Column(String(16), nullable=True)
        entry_signature = Column(String(100), nullable=True)
        s3_etag = Column(String(100), nullable=True)

        object_type = Column(String(100), nullable=True)
        storage_format = Column(String(20), nullable=True)
        serializer = Column(String(20), nullable=True)
        compression_codec = Column(String(20), nullable=True)
        actual_path = Column(String(500), nullable=True)

        cache_key_params = Column(Text, nullable=True)
        metadata_dict = Column(Text, nullable=True)

        access_count = Column(Integer, default=0, nullable=False, server_default="0")
        ttl_seconds = Column(Integer, nullable=True)
        expires_at = Column(DateTime(timezone=True), nullable=True)
        blob_data = Column(LargeBinary, nullable=True)
        is_inline = Column(Integer, default=0, nullable=False, server_default="0")
        inline_ext = Column(String(20), nullable=True)
        encryption_algorithm = Column(Text, nullable=True)
        encryption_iv = Column(Text, nullable=True)
        cacheness_version = Column(Text, nullable=True)

    class CacheStatsMixin:
        """Column definitions shared by all cache_stats tables."""

        id = Column(Integer, primary_key=True, default=1)
        cache_hits = Column(Integer, default=0, nullable=False)
        cache_misses = Column(Integer, default=0, nullable=False)
        last_updated = Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        )

    # ------------------------------------------------------------------
    # Default-namespace concrete models (backward-compatible table names)
    # ------------------------------------------------------------------

    class CacheEntry(CacheEntryMixin, Base):
        """SQLAlchemy model for cache entry metadata (default namespace)."""

        __tablename__ = "cache_entries"

        __table_args__ = (
            Index("idx_list_entries", desc("created_at")),
            Index("idx_cleanup", "created_at"),
            Index("idx_size_mgmt", "file_size", "created_at"),
            Index("idx_data_type", "data_type"),
            Index(
                "idx_metadata_notnull",
                desc("created_at"),
                sqlite_where=text("metadata_dict IS NOT NULL"),
            ),
            Index(
                "idx_expires_at",
                "expires_at",
                sqlite_where=text("expires_at IS NOT NULL"),
            ),
            Index("idx_access_count", "access_count", "accessed_at"),
        )

    class CacheStats(CacheStatsMixin, Base):
        """SQLAlchemy model for cache statistics (default namespace)."""

        __tablename__ = "cache_stats"

    # ------------------------------------------------------------------
    # Model factory — EntityName pattern (Mike Bayer's recommendation)
    # ------------------------------------------------------------------
    # For each non-default namespace we dynamically create ORM classes
    # with per-namespace __tablename__ using ``type()``.  Results are
    # cached so that each namespace only produces one class pair.
    # ------------------------------------------------------------------

    _ns_model_cache: Dict[str, tuple] = {}

    def _get_namespace_models(
        namespace_id: str, base: type = Base
    ) -> "tuple[type, type]":
        """Return ``(CacheEntryModel, CacheStatsModel)`` for *namespace_id*.

        For the ``'default'`` namespace the canonical ``CacheEntry`` /
        ``CacheStats`` classes are returned (unchanged table names).

        For any other namespace, dynamic subclasses with table names
        ``cache_entries_{namespace_id}`` / ``cache_stats_{namespace_id}``
        are created once and cached.
        """
        if namespace_id in _ns_model_cache:
            return _ns_model_cache[namespace_id]

        if namespace_id == DEFAULT_NAMESPACE:
            pair = (CacheEntry, CacheStats)
            _ns_model_cache[namespace_id] = pair
            return pair

        entries_table = f"cache_entries_{namespace_id}"
        stats_table = f"cache_stats_{namespace_id}"

        NsEntry = type(
            f"CacheEntry_{namespace_id}",
            (CacheEntryMixin, base),
            {
                "__tablename__": entries_table,
                "__table_args__": (
                    Index(f"idx_{namespace_id}_list_entries", desc("created_at")),
                    Index(f"idx_{namespace_id}_cleanup", "created_at"),
                    Index(f"idx_{namespace_id}_size_mgmt", "file_size", "created_at"),
                    Index(f"idx_{namespace_id}_data_type", "data_type"),
                    Index(
                        f"idx_{namespace_id}_metadata_notnull",
                        desc("created_at"),
                        sqlite_where=text("metadata_dict IS NOT NULL"),
                    ),
                    Index(
                        f"idx_{namespace_id}_expires_at",
                        "expires_at",
                        sqlite_where=text("expires_at IS NOT NULL"),
                    ),
                    Index(
                        f"idx_{namespace_id}_access_count",
                        "access_count",
                        "accessed_at",
                    ),
                ),
            },
        )

        NsStats = type(
            f"CacheStats_{namespace_id}",
            (CacheStatsMixin, base),
            {
                "__tablename__": stats_table,
            },
        )

        pair = (NsEntry, NsStats)
        _ns_model_cache[namespace_id] = pair
        return pair

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
        signature = Column(String(100), nullable=True)

    # ------------------------------------------------------------------
    # Core table registry — only these are created by SqliteBackend.__init__.
    # Custom metadata tables (from @custom_metadata_model) are created
    # separately via migrate_custom_metadata_tables() so they don't
    # pollute unrelated databases when Base.metadata is shared globally.
    # ------------------------------------------------------------------
    _CORE_TABLES = frozenset(
        {
            CacheEntry.__table__,
            CacheStats.__table__,
            CacheNamespace.__table__,
        }
    )

except ImportError:
    # SQLAlchemy not available
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not available, SQLite backend will not work")

    # Define dummy classes to avoid runtime errors
    class CacheEntryMixin:  # type: ignore[no-redef]
        pass

    class CacheStatsMixin:  # type: ignore[no-redef]
        pass

    class CacheEntry:  # type: ignore[no-redef]
        pass

    class CacheStats:  # type: ignore[no-redef]
        pass

    class CacheNamespace:  # type: ignore[no-redef]
        pass

    def _get_namespace_models(namespace_id: str, base: type = None) -> tuple:  # type: ignore[no-redef]
        return (CacheEntry, CacheStats)

    _ns_model_cache: Dict[str, tuple] = {}
    Base = None
