"""
PostgreSQL Metadata Backend
===========================

A high-performance metadata backend using PostgreSQL for distributed caching scenarios.

Features:
- Connection pooling for high concurrency
- Automatic table creation with optimized indexes
- Full compatibility with the MetadataBackend interface
- SSL/TLS support
- Transaction-safe operations

Usage:
    from cacheness.storage.backends.postgresql_backend import PostgresBackend
    from cacheness.storage.backends import register_metadata_backend

    # Register the backend
    register_metadata_backend("postgresql", PostgresBackend)

    # Create instance directly
    backend = PostgresBackend(
        connection_url="postgresql://user:pass@localhost:5432/cacheness"
    )

    # Or use via registry
    backend = get_metadata_backend(
        "postgresql",
        connection_url="postgresql://localhost/cache",
        pool_size=20
    )

Requirements:
    - psycopg2-binary or psycopg (PostgreSQL adapter)
    - SQLAlchemy >= 2.0
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check SQLAlchemy availability
try:
    from sqlalchemy import (
        create_engine,
        Column,
        Integer,
        String,
        DateTime,
        Text,
        Index,
        select,
        update,
        delete,
        desc,
        func,
        text,
    )
    from sqlalchemy.orm import sessionmaker, declarative_base
    from sqlalchemy.pool import QueuePool

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Check psycopg2/psycopg availability
try:
    import psycopg2  # noqa: F401 — availability check

    PSYCOPG_AVAILABLE = True
except ImportError:
    try:
        import psycopg  # noqa: F401 — availability check

        PSYCOPG_AVAILABLE = True
    except ImportError:
        PSYCOPG_AVAILABLE = False

from .base import (  # noqa: E402
    MetadataBackend,
    NamespaceInfo,
    validate_namespace_id,
    DEFAULT_NAMESPACE,
)

# JSON serialization utilities
try:
    import orjson

    def json_dumps(obj):
        return orjson.dumps(obj).decode("utf-8")

    def json_loads(s):
        return orjson.loads(s)
except ImportError:
    import json

    def json_dumps(obj):
        return json.dumps(obj, default=str)

    def json_loads(s):
        return json.loads(s)


if SQLALCHEMY_AVAILABLE:
    # Create a separate base for PostgreSQL to avoid conflicts with SQLite models
    PostgresBase = declarative_base()

    # ------------------------------------------------------------------
    # Abstract mixin bases for the EntityName pattern (dynamic table
    # names per namespace).  Concrete subclasses provide __tablename__.
    # ------------------------------------------------------------------

    class PgCacheEntryMixin:
        """Column definitions shared by all PG cache_entries tables."""

        cache_key = Column(String(16), primary_key=True)
        description = Column(String(500), default="", nullable=False)
        data_type = Column(String(20), nullable=False)
        prefix = Column(String(100), default="", nullable=False)

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
        entry_signature = Column(String(64), nullable=True)
        s3_etag = Column(String(100), nullable=True)

        object_type = Column(String(100), nullable=True)
        storage_format = Column(String(20), nullable=True)
        serializer = Column(String(20), nullable=True)
        compression_codec = Column(String(20), nullable=True)
        actual_path = Column(String(500), nullable=True)

        cache_key_params = Column(Text, nullable=True)

    class PgCacheStatsMixin:
        """Column definitions shared by all PG cache_stats tables."""

        id = Column(Integer, primary_key=True, default=1)
        cache_hits = Column(Integer, default=0, nullable=False)
        cache_misses = Column(Integer, default=0, nullable=False)
        total_entries = Column(Integer, default=0, nullable=False)
        total_size_bytes = Column(Integer, default=0, nullable=False)
        last_cleanup_at = Column(DateTime(timezone=True), nullable=True)

    # ------------------------------------------------------------------
    # Default-namespace concrete models (backward-compatible table names)
    # ------------------------------------------------------------------

    class PgCacheEntry(PgCacheEntryMixin, PostgresBase):
        """PostgreSQL cache entry model (default namespace)."""

        __tablename__ = "cache_entries"

        __table_args__ = (
            Index("idx_pg_list_entries", desc("created_at")),
            Index("idx_pg_cleanup", "created_at"),
            Index("idx_pg_size_mgmt", "file_size", "created_at"),
            Index("idx_pg_data_type", "data_type"),
            Index("idx_pg_prefix", "prefix"),
        )

    class PgCacheStats(PgCacheStatsMixin, PostgresBase):
        """PostgreSQL cache statistics model (default namespace)."""

        __tablename__ = "cache_stats"

    # ------------------------------------------------------------------
    # Model factory — EntityName pattern (Mike Bayer's recommendation)
    # ------------------------------------------------------------------

    _pg_ns_model_cache: Dict[str, tuple] = {}

    def _get_pg_namespace_models(
        namespace_id: str, base: type = PostgresBase
    ) -> "tuple[type, type]":
        """Return ``(PgCacheEntryModel, PgCacheStatsModel)`` for *namespace_id*.

        For ``'default'`` the canonical ``PgCacheEntry`` / ``PgCacheStats``
        classes are returned.  For other namespaces, dynamic subclasses
        with ``cache_entries_{namespace_id}`` / ``cache_stats_{namespace_id}``
        table names are created once and cached.
        """
        if namespace_id in _pg_ns_model_cache:
            return _pg_ns_model_cache[namespace_id]

        if namespace_id == DEFAULT_NAMESPACE:
            pair = (PgCacheEntry, PgCacheStats)
            _pg_ns_model_cache[namespace_id] = pair
            return pair

        entries_table = f"cache_entries_{namespace_id}"
        stats_table = f"cache_stats_{namespace_id}"

        NsEntry = type(
            f"PgCacheEntry_{namespace_id}",
            (PgCacheEntryMixin, base),
            {
                "__tablename__": entries_table,
                "__table_args__": (
                    Index(f"idx_pg_{namespace_id}_list_entries", desc("created_at")),
                    Index(f"idx_pg_{namespace_id}_cleanup", "created_at"),
                    Index(
                        f"idx_pg_{namespace_id}_size_mgmt", "file_size", "created_at"
                    ),
                    Index(f"idx_pg_{namespace_id}_data_type", "data_type"),
                    Index(f"idx_pg_{namespace_id}_prefix", "prefix"),
                ),
            },
        )

        NsStats = type(
            f"PgCacheStats_{namespace_id}",
            (PgCacheStatsMixin, base),
            {
                "__tablename__": stats_table,
            },
        )

        pair = (NsEntry, NsStats)
        _pg_ns_model_cache[namespace_id] = pair
        return pair

    class PgCacheNamespace(PostgresBase):
        """PostgreSQL namespace registry model."""

        __tablename__ = "cacheness_namespaces"

        namespace_id = Column(String(48), primary_key=True)
        display_name = Column(String(200), default="", nullable=False)
        schema_version = Column(Integer, default=1, nullable=False)
        created_at = Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        )
        signature = Column(String(128), nullable=True)


class PostgresBackend(MetadataBackend):
    """
    PostgreSQL metadata backend for distributed caching.

    This backend is designed for production environments requiring:
    - High concurrency (multiple workers, distributed systems)
    - Advanced querying capabilities
    - ACID transactions for metadata consistency
    - Centralized cache index across multiple machines

    Args:
        connection_url: PostgreSQL connection URL
            Format: postgresql://user:password@host:port/database

        pool_size: Connection pool size (default: 10)
        max_overflow: Max connections beyond pool_size (default: 20)
        pool_pre_ping: Test connections before use (default: True)
        pool_recycle: Recycle connections after N seconds (default: 3600)
        echo: Echo SQL statements for debugging (default: False)
        table_prefix: Optional prefix for table names (default: "")

    Example:
        >>> backend = PostgresBackend(
        ...     connection_url="postgresql://cacheuser:secret@db.example.com:5432/cacheness",
        ...     pool_size=20,
        ...     max_overflow=40
        ... )
        >>> backend.put_entry("abc123", {"data_type": "pickle", "file_size": 1024})
    """

    def __init__(
        self,
        connection_url: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_pre_ping: bool = True,
        pool_recycle: int = 3600,
        echo: bool = False,
        table_prefix: str = "",
        namespace: str = DEFAULT_NAMESPACE,
    ):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is required for PostgreSQL backend. "
                "Install with: pip install sqlalchemy"
            )

        if not PSYCOPG_AVAILABLE:
            raise ImportError(
                "PostgreSQL driver is required. "
                "Install with: pip install psycopg2-binary  OR  pip install psycopg[binary]"
            )

        self._active_namespace = validate_namespace_id(namespace)
        self.connection_url = connection_url
        self.table_prefix = table_prefix

        # Create engine with connection pooling
        self.engine = create_engine(
            connection_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=pool_pre_ping,
            pool_recycle=pool_recycle,
            echo=echo,
        )

        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        self._lock = threading.Lock()

        # Create tables (including cacheness_namespaces)
        PostgresBase.metadata.create_all(self.engine)

        # Resolve namespace-specific ORM models (EntityName pattern)
        self._PgCacheEntry, self._PgCacheStats = _get_pg_namespace_models(
            self._active_namespace
        )
        self._entries_table = self._PgCacheEntry.__tablename__
        self._stats_table = self._PgCacheStats.__tablename__

        # For non-default namespaces, ensure their tables exist too
        if self._active_namespace != DEFAULT_NAMESPACE:
            self._PgCacheEntry.__table__.create(self.engine, checkfirst=True)
            self._PgCacheStats.__table__.create(self.engine, checkfirst=True)

        # Ensure the namespace registry has a 'default' entry
        self._ensure_namespace_registry()

        # Run formal migrations for schema evolution
        self.run_migrations(DEFAULT_NAMESPACE)

        # Initialize stats
        self._init_stats()

        logger.info(f"✅ PostgreSQL metadata backend initialized: {self._safe_url()}")

    def _safe_url(self) -> str:
        """Return connection URL with password masked."""
        # Simple masking - just show host/db
        if "@" in self.connection_url:
            parts = self.connection_url.split("@")
            return f"postgresql://***@{parts[-1]}"
        return self.connection_url

    def _run_migrations(self):
        """Legacy migration entry point — kept for backward compatibility.

        New code uses ``run_migrations()`` via ``get_migrations()``.
        """
        pass

    # --- Schema versioning overrides ---

    def _ensure_namespace_registry(self):
        """Ensure the namespace registry has a 'default' entry.

        Handles the v0→v1 transition: if cacheness_namespaces was just
        created by ``create_all()`` and has no rows, seed it with the
        ``'default'`` namespace.
        """
        with self.SessionLocal() as session:
            existing = session.execute(
                select(PgCacheNamespace).where(
                    PgCacheNamespace.namespace_id == DEFAULT_NAMESPACE
                )
            ).scalar_one_or_none()

            if existing is None:
                ns = PgCacheNamespace(
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
                select(PgCacheNamespace).where(
                    PgCacheNamespace.namespace_id == namespace_id
                )
            ).scalar_one_or_none()
            return ns.schema_version if ns else 0

    def set_schema_version(self, namespace_id: str, version: int) -> None:
        """Write schema version to the namespace registry."""
        with self.SessionLocal() as session:
            session.execute(
                update(PgCacheNamespace)
                .where(PgCacheNamespace.namespace_id == namespace_id)
                .values(schema_version=version)
            )
            session.commit()

    def get_migrations(self) -> list:
        """Return PostgreSQL-specific schema migrations.

        v0 → v1: Legacy column additions (s3_etag, cache_key_params).
                  These used to be ad-hoc checks; now formalized.
        """

        def _migrate_v0_to_v1(backend: "PostgresBackend", namespace_id: str):
            """Add columns that may be missing from pre-versioning databases."""
            from sqlalchemy import inspect as sa_inspect

            with backend.SessionLocal() as session:
                inspector = sa_inspect(backend.engine)
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

        with self._lock:
            with self.SessionLocal() as session:
                # Check for duplicates
                existing = session.execute(
                    select(PgCacheNamespace).where(
                        PgCacheNamespace.namespace_id == namespace_id
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    raise ValueError(f"Namespace {namespace_id!r} already exists")

                # Create per-namespace tables
                entries_table = f"cache_entries_{namespace_id}"
                stats_table = f"cache_stats_{namespace_id}"

                session.execute(
                    text(f"""
                    CREATE TABLE IF NOT EXISTS "{entries_table}" (
                        cache_key       VARCHAR(16) PRIMARY KEY,
                        description     VARCHAR(500) NOT NULL DEFAULT '',
                        data_type       VARCHAR(20) NOT NULL,
                        prefix          VARCHAR(100) NOT NULL DEFAULT '',
                        created_at      TIMESTAMP WITH TIME ZONE NOT NULL,
                        accessed_at     TIMESTAMP WITH TIME ZONE NOT NULL,
                        file_size       INTEGER NOT NULL DEFAULT 0,
                        file_hash       VARCHAR(16),
                        entry_signature VARCHAR(64),
                        s3_etag         VARCHAR(100),
                        object_type     VARCHAR(100),
                        storage_format  VARCHAR(20),
                        serializer      VARCHAR(20),
                        compression_codec VARCHAR(20),
                        actual_path     VARCHAR(500),
                        cache_key_params TEXT
                    )
                """)
                )

                session.execute(
                    text(f"""
                    CREATE TABLE IF NOT EXISTS "{stats_table}" (
                        id              INTEGER PRIMARY KEY DEFAULT 1,
                        cache_hits      INTEGER NOT NULL DEFAULT 0,
                        cache_misses    INTEGER NOT NULL DEFAULT 0,
                        total_entries   INTEGER NOT NULL DEFAULT 0,
                        total_size_bytes INTEGER NOT NULL DEFAULT 0,
                        last_cleanup_at TIMESTAMP WITH TIME ZONE
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
                ns = PgCacheNamespace(
                    namespace_id=namespace_id,
                    display_name=display_name,
                    schema_version=1,
                    created_at=now,
                )
                session.add(ns)

                # Initialize stats row for the new namespace
                session.execute(
                    text(
                        f'INSERT INTO "{stats_table}" '
                        f"(id, cache_hits, cache_misses, total_entries, "
                        f"total_size_bytes) "
                        f"VALUES (1, 0, 0, 0, 0) "
                        f"ON CONFLICT (id) DO NOTHING"
                    )
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

        with self._lock:
            with self.SessionLocal() as session:
                existing = session.execute(
                    select(PgCacheNamespace).where(
                        PgCacheNamespace.namespace_id == namespace_id
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
                    delete(PgCacheNamespace).where(
                        PgCacheNamespace.namespace_id == namespace_id
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
                    select(PgCacheNamespace).order_by(PgCacheNamespace.created_at)
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

    def get_namespace(self, namespace_id: str) -> Optional[NamespaceInfo]:
        """Get info for a specific namespace."""
        with self.SessionLocal() as session:
            row = session.execute(
                select(PgCacheNamespace).where(
                    PgCacheNamespace.namespace_id == namespace_id
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

    def _init_stats(self):
        """Initialize cache statistics if not exists."""
        with self.SessionLocal() as session:
            try:
                stats = session.execute(
                    select(self._PgCacheStats).where(self._PgCacheStats.id == 1)
                ).scalar_one_or_none()

                if not stats:
                    stats = self._PgCacheStats(id=1)
                    session.add(stats)
                    session.commit()
            except Exception as e:
                logger.warning(f"Failed to initialize stats: {e}")
                session.rollback()

    def load_metadata(self) -> Dict[str, Any]:
        """Load complete metadata (mainly for compatibility)."""
        with self.SessionLocal() as session:
            entries = session.execute(select(self._PgCacheEntry)).scalars().all()

            result = {"entries": {}, "stats": self.get_stats()}
            for entry in entries:
                result["entries"][entry.cache_key] = self._entry_to_dict(entry)

            return result

    def save_metadata(self, metadata: Dict[str, Any]):
        """Save complete metadata (mainly for compatibility)."""
        # PostgreSQL backend stores incrementally, this is mainly for bulk import
        with self._lock:
            with self.SessionLocal() as session:
                try:
                    for cache_key, entry_data in metadata.get("entries", {}).items():
                        self._upsert_entry(session, cache_key, entry_data)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Failed to save metadata: {e}")
                    raise

    def get_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cache entry metadata."""
        with self.SessionLocal() as session:
            entry = session.execute(
                select(self._PgCacheEntry).where(
                    self._PgCacheEntry.cache_key == cache_key
                )
            ).scalar_one_or_none()

            if entry is None:
                return None

            return self._entry_to_dict(entry)

    def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
        """Store cache entry metadata."""
        with self._lock:
            with self.SessionLocal() as session:
                try:
                    self._upsert_entry(session, cache_key, entry_data)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Failed to put entry {cache_key}: {e}")
                    raise

    def _upsert_entry(self, session, cache_key: str, entry_data: Dict[str, Any]):
        """Insert or update an entry."""
        # Extract nested metadata if present
        metadata = entry_data.get("metadata", {}).copy()

        # Extract optional cache_key_params for storage if needed (disabled by default)
        # No separate extraction needed - it's already in metadata

        # Extract fields from nested metadata
        object_type = metadata.pop("object_type", None)
        storage_format = metadata.pop("storage_format", None)
        serializer = metadata.pop("serializer", None)
        compression_codec = metadata.pop("compression_codec", None)
        actual_path = metadata.pop("actual_path", None)
        file_hash = metadata.pop("file_hash", None)
        entry_signature = metadata.pop("entry_signature", None)
        s3_etag = metadata.pop("s3_etag", None)  # S3 ETag if using S3 backend
        cache_key_params = metadata.pop("cache_key_params", None)

        # Handle timestamps - always use UTC
        created_at = entry_data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
            # If naive datetime, assume UTC
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            else:
                # Convert to UTC if different timezone
                created_at = created_at.astimezone(timezone.utc)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)
        elif isinstance(created_at, datetime):
            # Ensure it's UTC
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            else:
                created_at = created_at.astimezone(timezone.utc)

        accessed_at = entry_data.get("accessed_at")
        if isinstance(accessed_at, str):
            accessed_at = datetime.fromisoformat(accessed_at)
            # If naive datetime, assume UTC
            if accessed_at.tzinfo is None:
                accessed_at = accessed_at.replace(tzinfo=timezone.utc)
            else:
                accessed_at = accessed_at.astimezone(timezone.utc)
        elif accessed_at is None:
            accessed_at = datetime.now(timezone.utc)
        elif isinstance(accessed_at, datetime):
            # Ensure it's UTC
            if accessed_at.tzinfo is None:
                accessed_at = accessed_at.replace(tzinfo=timezone.utc)
            else:
                accessed_at = accessed_at.astimezone(timezone.utc)

        # Serialize cache_key_params if present
        serialized_params = None
        if cache_key_params is not None:
            try:
                serialized_params = json_dumps(cache_key_params)
            except Exception:
                pass

        # Check if entry exists
        existing = session.execute(
            select(self._PgCacheEntry).where(self._PgCacheEntry.cache_key == cache_key)
        ).scalar_one_or_none()

        if existing:
            # Update existing entry
            session.execute(
                update(self._PgCacheEntry)
                .where(self._PgCacheEntry.cache_key == cache_key)
                .values(
                    description=entry_data.get("description", ""),
                    data_type=entry_data.get("data_type", "unknown"),
                    prefix=entry_data.get("prefix", ""),
                    created_at=created_at,
                    accessed_at=accessed_at,
                    file_size=entry_data.get("file_size", 0),
                    file_hash=file_hash,
                    entry_signature=entry_signature,
                    s3_etag=s3_etag,
                    object_type=object_type,
                    storage_format=storage_format,
                    serializer=serializer,
                    compression_codec=compression_codec,
                    actual_path=actual_path,
                    cache_key_params=serialized_params,
                )
            )
        else:
            # Insert new entry
            entry = self._PgCacheEntry(
                cache_key=cache_key,
                description=entry_data.get("description", ""),
                data_type=entry_data.get("data_type", "unknown"),
                prefix=entry_data.get("prefix", ""),
                created_at=created_at,
                accessed_at=accessed_at,
                file_size=entry_data.get("file_size", 0),
                file_hash=file_hash,
                entry_signature=entry_signature,
                s3_etag=s3_etag,
                object_type=object_type,
                storage_format=storage_format,
                serializer=serializer,
                compression_codec=compression_codec,
                actual_path=actual_path,
                cache_key_params=serialized_params,
            )
            session.add(entry)

    def _entry_to_dict(self, entry: "PgCacheEntry") -> Dict[str, Any]:
        """Convert entry model to dictionary."""
        result = {
            "cache_key": entry.cache_key,
            "description": entry.description or "",
            "data_type": entry.data_type,
            "prefix": entry.prefix or "",
            "created_at": entry.created_at.astimezone(timezone.utc).isoformat()
            if entry.created_at
            else None,
            "accessed_at": entry.accessed_at.astimezone(timezone.utc).isoformat()
            if entry.accessed_at
            else None,
            "file_size": entry.file_size or 0,
        }

        # Build nested metadata
        metadata = {}
        if entry.object_type:
            metadata["object_type"] = entry.object_type
        if entry.storage_format:
            metadata["storage_format"] = entry.storage_format
        if entry.serializer:
            metadata["serializer"] = entry.serializer
        if entry.compression_codec:
            metadata["compression_codec"] = entry.compression_codec
        if entry.actual_path:
            metadata["actual_path"] = entry.actual_path
        if entry.file_hash:
            metadata["file_hash"] = entry.file_hash
        if entry.entry_signature:
            metadata["entry_signature"] = entry.entry_signature
        if entry.s3_etag:
            metadata["s3_etag"] = entry.s3_etag
        if entry.actual_path:
            metadata["actual_path"] = entry.actual_path
        if entry.file_hash:
            metadata["file_hash"] = entry.file_hash
        if entry.entry_signature:
            metadata["entry_signature"] = entry.entry_signature

        if metadata:
            result["metadata"] = metadata

        # Parse cache_key_params if present
        if entry.cache_key_params:
            try:
                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"]["cache_key_params"] = json_loads(
                    entry.cache_key_params
                )
            except Exception:
                pass

        return result

    def remove_entry(self, cache_key: str) -> bool:
        """Remove cache entry metadata.

        Returns:
            bool: True if the entry existed and was removed, False if not found.
        """
        with self._lock:
            with self.SessionLocal() as session:
                try:
                    result = session.execute(
                        delete(self._PgCacheEntry).where(
                            self._PgCacheEntry.cache_key == cache_key
                        )
                    )
                    session.commit()
                    return result.rowcount > 0
                except Exception as e:
                    session.rollback()
                    logger.error(f"Failed to remove entry {cache_key}: {e}")
                    raise

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
            with self.SessionLocal() as session:
                try:
                    # Check if entry exists
                    entry = session.execute(
                        select(self._PgCacheEntry).where(
                            self._PgCacheEntry.cache_key == cache_key
                        )
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
                    if "s3_etag" in updates:
                        entry.s3_etag = updates["s3_etag"]

                    session.commit()
                    return True
                except Exception as e:
                    session.rollback()
                    logger.error(f"Failed to update entry {cache_key}: {e}")
                    raise

    def iter_entry_summaries(self) -> List[Dict[str, Any]]:
        """Return lightweight flat entry dicts — raw SQL, no ORM hydration."""
        with self.SessionLocal() as session:
            tbl = self._entries_table
            rows = session.execute(
                text(
                    f"SELECT cache_key, data_type, description, prefix, "
                    f"       file_size, created_at, accessed_at, "
                    f"       object_type, storage_format, serializer, "
                    f"       compression_codec, actual_path, "
                    f"       file_hash, entry_signature "
                    f'FROM "{tbl}"'
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
        """List all cache entries."""
        with self.SessionLocal() as session:
            entries = (
                session.execute(
                    select(self._PgCacheEntry).order_by(
                        desc(self._PgCacheEntry.created_at)
                    )
                )
                .scalars()
                .all()
            )

            return [self._entry_to_dict(entry) for entry in entries]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.SessionLocal() as session:
            stats = session.execute(
                select(self._PgCacheStats).where(self._PgCacheStats.id == 1)
            ).scalar_one_or_none()

            if stats is None:
                return {
                    "hits": 0,
                    "misses": 0,
                    "total_entries": 0,
                    "total_size_bytes": 0,
                }

            return {
                "hits": stats.cache_hits,
                "misses": stats.cache_misses,
                "total_entries": stats.total_entries,
                "total_size_bytes": stats.total_size_bytes,
                "last_cleanup_at": (
                    stats.last_cleanup_at.isoformat() if stats.last_cleanup_at else None
                ),
            }

    def update_access_time(self, cache_key: str):
        """Update last access time for cache entry."""
        with self.SessionLocal() as session:
            try:
                session.execute(
                    update(self._PgCacheEntry)
                    .where(self._PgCacheEntry.cache_key == cache_key)
                    .values(accessed_at=datetime.now(timezone.utc))
                )
                session.commit()
            except Exception as e:
                session.rollback()
                logger.warning(f"Failed to update access time for {cache_key}: {e}")

    def increment_hits(self):
        """Increment cache hits counter."""
        with self.SessionLocal() as session:
            try:
                session.execute(
                    update(self._PgCacheStats)
                    .where(self._PgCacheStats.id == 1)
                    .values(cache_hits=self._PgCacheStats.cache_hits + 1)
                )
                session.commit()
            except Exception as e:
                session.rollback()
                logger.warning(f"Failed to increment hits: {e}")

    def increment_misses(self):
        """Increment cache misses counter."""
        with self.SessionLocal() as session:
            try:
                session.execute(
                    update(self._PgCacheStats)
                    .where(self._PgCacheStats.id == 1)
                    .values(cache_misses=self._PgCacheStats.cache_misses + 1)
                )
                session.commit()
            except Exception as e:
                session.rollback()
                logger.warning(f"Failed to increment misses: {e}")

    def cleanup_expired(self, ttl_seconds: float) -> int:
        """Remove expired entries and return count removed."""
        if ttl_seconds <= 0:
            return 0

        cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(
            seconds=ttl_seconds
        )

        with self._lock:
            with self.SessionLocal() as session:
                try:
                    # Count entries to be removed
                    count = (
                        session.execute(
                            select(func.count())
                            .select_from(self._PgCacheEntry)
                            .where(self._PgCacheEntry.created_at < cutoff)
                        ).scalar()
                        or 0
                    )

                    if count > 0:
                        session.execute(
                            delete(self._PgCacheEntry).where(
                                self._PgCacheEntry.created_at < cutoff
                            )
                        )

                        # Update last cleanup timestamp
                        session.execute(
                            update(self._PgCacheStats)
                            .where(self._PgCacheStats.id == 1)
                            .values(last_cleanup_at=datetime.now(timezone.utc))
                        )

                        session.commit()
                        logger.info(
                            f"Cleaned up {count} expired entries (TTL: {ttl_seconds}s)"
                        )

                    return count

                except Exception as e:
                    session.rollback()
                    logger.error(f"Cleanup failed: {e}")
                    return 0

    def cleanup_by_size(self, target_size_mb: float) -> Dict[str, Any]:
        """Remove least-recently-accessed entries until cache size drops to or below target."""
        with self._lock:
            with self.SessionLocal() as session:
                try:
                    # Get current total size
                    CE = self._PgCacheEntry
                    result = session.execute(
                        select(func.sum(CE.file_size)).select_from(CE)
                    )
                    total_size_bytes = result.scalar() or 0
                    total_size_mb = total_size_bytes / (1024 * 1024)

                    if total_size_mb <= target_size_mb:
                        return {
                            "count": 0,
                            "removed_entries": [],
                        }  # Already at or below target

                    target_size_bytes = target_size_mb * 1024 * 1024
                    bytes_to_remove = total_size_bytes - target_size_bytes

                    # Get entries sorted by accessed_at (oldest first) with actual_path
                    entries_to_delete = session.execute(
                        select(
                            CE.cache_key,
                            CE.file_size,
                            CE.actual_path,
                        ).order_by(CE.accessed_at.asc())
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
                            delete(self._PgCacheEntry).where(
                                self._PgCacheEntry.cache_key.in_(removed_keys)
                            )
                        )

                        # Update last cleanup timestamp
                        session.execute(
                            update(self._PgCacheStats)
                            .where(self._PgCacheStats.id == 1)
                            .values(last_cleanup_at=datetime.now(timezone.utc))
                        )

                        session.commit()
                        logger.info(
                            f"LRU cleanup: removed {len(removed_entries)} entries to reach {target_size_mb:.2f}MB"
                        )

                    return {
                        "count": len(removed_entries),
                        "removed_entries": removed_entries,
                    }

                except Exception as e:
                    session.rollback()
                    logger.error(f"LRU cleanup failed: {e}")
                    return {"count": 0, "removed_entries": []}

    def clear_all(self) -> int:
        """Remove all cache entries and return count removed."""
        with self._lock:
            with self.SessionLocal() as session:
                try:
                    count = (
                        session.execute(
                            select(func.count()).select_from(self._PgCacheEntry)
                        ).scalar()
                        or 0
                    )

                    session.execute(delete(self._PgCacheEntry))

                    # Reset stats
                    session.execute(
                        update(self._PgCacheStats)
                        .where(self._PgCacheStats.id == 1)
                        .values(
                            cache_hits=0,
                            cache_misses=0,
                            total_entries=0,
                            total_size_bytes=0,
                        )
                    )

                    session.commit()
                    logger.info(f"Cleared all {count} cache entries")
                    return count

                except Exception as e:
                    session.rollback()
                    logger.error(f"Clear all failed: {e}")
                    return 0

    def close(self):
        """Close database connections."""
        try:
            self.engine.dispose()
            logger.debug("PostgreSQL engine disposed")
        except Exception as e:
            logger.warning(f"Error closing PostgreSQL connection: {e}")

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# Auto-register if this module is imported and dependencies are available
def _auto_register():
    """Automatically register PostgreSQL backend if available."""
    if SQLALCHEMY_AVAILABLE and PSYCOPG_AVAILABLE:
        try:
            from . import register_metadata_backend, _metadata_backend_registry

            if "postgresql" not in _metadata_backend_registry:
                register_metadata_backend("postgresql", PostgresBackend)
                logger.debug("Auto-registered PostgreSQL metadata backend")
        except Exception as e:
            logger.debug(f"Could not auto-register PostgreSQL backend: {e}")


# Attempt auto-registration
_auto_register()
