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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ...interfaces import EntrySummary
from ...size_utils import format_size

logger = logging.getLogger(__name__)

# Check SQLAlchemy availability
try:
    from sqlalchemy import (
        create_engine,
        Column,
        Integer,
        String,
        DateTime,
        LargeBinary,
        Text,
        Index,
        select,
        update,
        delete,
        desc,
        func,
        text,
        and_,
        or_,
    )
    from sqlalchemy.dialects.postgresql import JSONB
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


def _ensure_jsonb_value(value):
    """Convert a JSON-encoded string to a Python dict for JSONB storage.

    JSONB columns require a Python dict (or None), not a JSON string.
    Handles: None → None, dict → dict (pass-through), str → parsed dict.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json_loads(value)
            return parsed if isinstance(parsed, dict) else None
        except Exception:  # intentionally broad — JSON parsing fallback
            return None
    return None


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

        cache_key_params = Column(JSONB, nullable=True)
        metadata_dict = Column(JSONB, nullable=True)

        access_count = Column(Integer, default=0, nullable=False, server_default="0")
        ttl_seconds = Column(Integer, nullable=True)
        expires_at = Column(DateTime(timezone=True), nullable=True)
        blob_data = Column(LargeBinary, nullable=True)
        is_inline = Column(Integer, default=0, nullable=False, server_default="0")
        inline_ext = Column(String(20), nullable=True)
        encryption_algorithm = Column(Text, nullable=True)
        encryption_iv = Column(Text, nullable=True)
        cacheness_version = Column(Text, nullable=True)

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
            Index(
                "idx_pg_metadata_gin",
                "metadata_dict",
                postgresql_using="gin",
                postgresql_ops={"metadata_dict": "jsonb_path_ops"},
            ),
            Index(
                "idx_pg_expires_at",
                "expires_at",
                postgresql_where=text("expires_at IS NOT NULL"),
            ),
            Index("idx_pg_access_count", "access_count", "accessed_at"),
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
                    Index(
                        f"idx_pg_{namespace_id}_metadata_gin",
                        "metadata_dict",
                        postgresql_using="gin",
                        postgresql_ops={"metadata_dict": "jsonb_path_ops"},
                    ),
                    Index(
                        f"idx_pg_{namespace_id}_expires_at",
                        "expires_at",
                        postgresql_where=text("expires_at IS NOT NULL"),
                    ),
                    Index(
                        f"idx_pg_{namespace_id}_access_count",
                        "access_count",
                        "accessed_at",
                    ),
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


# ------------------------------------------------------------------
# Schema migrations
# ------------------------------------------------------------------


def _pg_migrate_v1_to_v2(backend: "PostgresBackend", namespace_id: str) -> None:
    """Migrate v1 → v2: metadata_dict and cache_key_params Text → JSONB.

    Converts the columns from Text to JSONB using ``::jsonb`` cast and
    creates a GIN index on ``metadata_dict`` for fast ``@>`` containment
    queries.
    """
    table = (
        "cache_entries"
        if namespace_id == DEFAULT_NAMESPACE
        else f"cache_entries_{namespace_id}"
    )
    with backend.SessionLocal() as session:
        # Convert metadata_dict Text → JSONB
        session.execute(
            text(
                f'ALTER TABLE "{table}" '
                f"ALTER COLUMN metadata_dict TYPE JSONB "
                f"USING metadata_dict::jsonb"
            )
        )
        # Convert cache_key_params Text → JSONB
        session.execute(
            text(
                f'ALTER TABLE "{table}" '
                f"ALTER COLUMN cache_key_params TYPE JSONB "
                f"USING cache_key_params::jsonb"
            )
        )
        # Create GIN index for fast JSONB containment queries
        idx_name = (
            "idx_pg_metadata_gin"
            if namespace_id == DEFAULT_NAMESPACE
            else f"idx_pg_{namespace_id}_metadata_gin"
        )
        session.execute(
            text(
                f'CREATE INDEX IF NOT EXISTS "{idx_name}" '
                f'ON "{table}" USING GIN (metadata_dict jsonb_path_ops)'
            )
        )
        session.commit()


def _pg_migrate_v2_to_v3(backend: "PostgresBackend", namespace_id: str) -> None:
    """Migrate v2 → v3: add access_count, ttl_seconds, expires_at, blob_data, is_inline columns.

    Adds five new columns plus two indexes:

    * ``access_count`` — per-entry access counter
    * ``ttl_seconds`` — per-entry TTL storage
    * ``expires_at`` — pre-computed expiry timestamp
    * ``blob_data`` — inline blob content (BYTEA)
    * ``is_inline`` — flag: 1 if blob stored inline, 0 otherwise
    * ``idx_pg_expires_at`` — partial index on ``expires_at`` WHERE NOT NULL.
    * ``idx_pg_access_count`` — composite (access_count, accessed_at).

    All ALTER TABLE ADD COLUMN uses ``IF NOT EXISTS`` (PG 9.6+) for idempotency.
    """
    table = (
        "cache_entries"
        if namespace_id == DEFAULT_NAMESPACE
        else f"cache_entries_{namespace_id}"
    )
    prefix = "idx_pg" if namespace_id == DEFAULT_NAMESPACE else f"idx_pg_{namespace_id}"

    with backend.SessionLocal() as session:
        # Add new columns (PG supports IF NOT EXISTS on ADD COLUMN since 9.6)
        session.execute(
            text(
                f'ALTER TABLE "{table}" '
                f"ADD COLUMN IF NOT EXISTS access_count INTEGER NOT NULL DEFAULT 0"
            )
        )
        session.execute(
            text(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS ttl_seconds INTEGER')
        )
        session.execute(
            text(
                f'ALTER TABLE "{table}" '
                f"ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP WITH TIME ZONE"
            )
        )
        session.execute(
            text(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS blob_data BYTEA')
        )
        session.execute(
            text(
                f'ALTER TABLE "{table}" '
                f"ADD COLUMN IF NOT EXISTS is_inline INTEGER NOT NULL DEFAULT 0"
            )
        )
        session.execute(
            text(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS inline_ext TEXT')
        )

        # Create indexes
        session.execute(
            text(
                f'CREATE INDEX IF NOT EXISTS "{prefix}_expires_at" '
                f'ON "{table}" (expires_at) '
                f"WHERE expires_at IS NOT NULL"
            )
        )
        session.execute(
            text(
                f'CREATE INDEX IF NOT EXISTS "{prefix}_access_count" '
                f'ON "{table}" (access_count, accessed_at)'
            )
        )
        session.commit()

    logger.info(
        "PG v2→v3: added access_count/ttl_seconds/expires_at/blob_data/is_inline "
        "columns and indexes on %r",
        table,
    )


def _pg_migrate_v3_to_v4(backend: "PostgresBackend", namespace_id: str) -> None:
    """Migrate v3 → v4: add encryption_algorithm, encryption_iv, cacheness_version.

    * ``encryption_algorithm`` — algorithm used for encryption (e.g. 'AES-256-GCM')
    * ``encryption_iv`` — initialization vector for encrypted blobs
    * ``cacheness_version`` — Cacheness version that wrote the entry

    All ALTER TABLE ADD COLUMN uses ``IF NOT EXISTS`` (PG 9.6+) for idempotency.
    """
    table = (
        "cache_entries"
        if namespace_id == DEFAULT_NAMESPACE
        else f"cache_entries_{namespace_id}"
    )

    with backend.SessionLocal() as session:
        session.execute(
            text(
                f'ALTER TABLE "{table}" '
                f"ADD COLUMN IF NOT EXISTS encryption_algorithm TEXT"
            )
        )
        session.execute(
            text(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS encryption_iv TEXT')
        )
        session.execute(
            text(
                f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS cacheness_version TEXT'
            )
        )
        session.commit()

    logger.info(
        "PG v3→v4: added encryption_algorithm/encryption_iv/cacheness_version "
        "columns on %r",
        table,
    )


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
        self.run_all_migrations()

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

        Schema baseline is v1 (current).
        v1 → v2: wired via ``_pg_migrate_v1_to_v2`` but v2 does not
                  exist yet.  Uncomment the entry below when a v2 schema
                  change is defined.
        """
        return [
            (1, 2, _pg_migrate_v1_to_v2),
            (2, 3, _pg_migrate_v2_to_v3),
            (3, 4, _pg_migrate_v3_to_v4),
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
                        created_at      TIMESTAMP WITH TIME ZONE NOT NULL,
                        accessed_at     TIMESTAMP WITH TIME ZONE NOT NULL,
                        file_size       INTEGER NOT NULL DEFAULT 0,
                        file_hash       VARCHAR(16),
                        entry_signature VARCHAR(100),
                        s3_etag         VARCHAR(100),
                        object_type     VARCHAR(100),
                        storage_format  VARCHAR(20),
                        serializer      VARCHAR(20),
                        compression_codec VARCHAR(20),
                        actual_path     VARCHAR(500),
                        cache_key_params JSONB,
                        metadata_dict JSONB,
                        access_count    INTEGER NOT NULL DEFAULT 0,
                        ttl_seconds     INTEGER,
                        expires_at      TIMESTAMP WITH TIME ZONE,
                        blob_data       BYTEA,
                        is_inline       INTEGER NOT NULL DEFAULT 0,
                        inline_ext      TEXT,
                        encryption_algorithm TEXT,
                        encryption_iv   TEXT,
                        cacheness_version TEXT
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
                session.execute(
                    text(
                        f'CREATE INDEX IF NOT EXISTS "idx_{namespace_id}_metadata_gin" '
                        f'ON "{entries_table}" USING GIN (metadata_dict jsonb_path_ops)'
                    )
                )
                session.execute(
                    text(
                        f'CREATE INDEX IF NOT EXISTS "idx_pg_{namespace_id}_expires_at" '
                        f'ON "{entries_table}" (expires_at) '
                        f"WHERE expires_at IS NOT NULL"
                    )
                )
                session.execute(
                    text(
                        f'CREATE INDEX IF NOT EXISTS "idx_pg_{namespace_id}_access_count" '
                        f'ON "{entries_table}" (access_count, accessed_at)'
                    )
                )

                # Register in the namespace registry
                now = datetime.now(timezone.utc)
                ns = PgCacheNamespace(
                    namespace_id=namespace_id,
                    display_name=display_name,
                    schema_version=4,
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

                # Run migrations for the new namespace
                self.run_migrations(namespace_id)

                return NamespaceInfo(
                    namespace_id=namespace_id,
                    display_name=display_name,
                    schema_version=self.get_schema_version(namespace_id),
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

    def set_namespace_signature(self, namespace_id: str, signature: str) -> None:
        """Store namespace signature in the PostgreSQL registry."""
        with self.SessionLocal() as session:
            session.execute(
                update(PgCacheNamespace)
                .where(PgCacheNamespace.namespace_id == namespace_id)
                .values(signature=signature)
            )
            session.commit()

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
            except Exception as e:  # intentionally broad — stats init is best-effort
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
                except Exception as e:  # intentionally broad — re-raises after rollback
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
                except Exception as e:  # intentionally broad — re-raises after rollback
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
        metadata_dict_value = metadata.pop("metadata_dict", None)
        inline_ext = metadata.pop("inline_ext", None)
        encryption_algorithm = metadata.pop("encryption_algorithm", None)
        encryption_iv = metadata.pop("encryption_iv", None)
        cacheness_version = metadata.pop("cacheness_version", None)

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

        # Convert to JSONB-compatible dicts (handles JSON strings from core.py)
        jsonb_params = _ensure_jsonb_value(cache_key_params)
        jsonb_metadata = _ensure_jsonb_value(metadata_dict_value)

        # Handle TTL fields
        ttl_seconds_val = entry_data.get("ttl_seconds")
        expires_at = entry_data.get("expires_at")
        if expires_at is None and ttl_seconds_val is not None:
            # Compute expires_at from created_at + ttl_seconds
            expires_at = created_at + timedelta(seconds=float(ttl_seconds_val))
        elif isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            else:
                expires_at = expires_at.astimezone(timezone.utc)

        access_count_val = entry_data.get("access_count", 0)

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
                    cache_key_params=jsonb_params,
                    metadata_dict=jsonb_metadata,
                    ttl_seconds=ttl_seconds_val,
                    expires_at=expires_at,
                    blob_data=entry_data.get("blob_data"),
                    is_inline=entry_data.get("is_inline", 0),
                    inline_ext=inline_ext,
                    encryption_algorithm=encryption_algorithm,
                    encryption_iv=encryption_iv,
                    cacheness_version=cacheness_version,
                )
            )
        else:
            # Insert new entry
            entry = self._PgCacheEntry(
                cache_key=cache_key,
                description=entry_data.get("description", ""),
                data_type=entry_data.get("data_type", "unknown"),
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
                cache_key_params=jsonb_params,
                metadata_dict=jsonb_metadata,
                access_count=access_count_val,
                ttl_seconds=ttl_seconds_val,
                expires_at=expires_at,
                blob_data=entry_data.get("blob_data"),
                is_inline=entry_data.get("is_inline", 0),
                inline_ext=inline_ext,
                encryption_algorithm=encryption_algorithm,
                encryption_iv=encryption_iv,
                cacheness_version=cacheness_version,
            )
            session.add(entry)

    def _entry_to_dict(self, entry: "PgCacheEntry") -> Dict[str, Any]:
        """Convert entry model to dictionary."""
        result = {
            "cache_key": entry.cache_key,
            "description": entry.description or "",
            "data_type": entry.data_type,
            "created_at": entry.created_at.astimezone(timezone.utc).isoformat()
            if entry.created_at
            else None,
            "accessed_at": entry.accessed_at.astimezone(timezone.utc).isoformat()
            if entry.accessed_at
            else None,
            "file_size": entry.file_size or 0,
            "access_count": getattr(entry, "access_count", 0) or 0,
            "ttl_seconds": getattr(entry, "ttl_seconds", None),
            "expires_at": (
                entry.expires_at.astimezone(timezone.utc).isoformat()
                if getattr(entry, "expires_at", None)
                else None
            ),
            "is_inline": getattr(entry, "is_inline", 0) or 0,
            "blob_data": getattr(entry, "blob_data", None),
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
        if getattr(entry, "inline_ext", None):
            metadata["inline_ext"] = entry.inline_ext
        if getattr(entry, "encryption_algorithm", None):
            metadata["encryption_algorithm"] = entry.encryption_algorithm
        if getattr(entry, "encryption_iv", None):
            metadata["encryption_iv"] = entry.encryption_iv
        if getattr(entry, "cacheness_version", None):
            metadata["cacheness_version"] = entry.cacheness_version

        if metadata:
            result["metadata"] = metadata

        # Parse cache_key_params if present (JSONB returns dict natively)
        if entry.cache_key_params:
            try:
                if "metadata" not in result:
                    result["metadata"] = {}
                ckp = entry.cache_key_params
                if isinstance(ckp, str):
                    ckp = json_loads(ckp)
                result["metadata"]["cache_key_params"] = ckp
            except Exception:  # intentionally broad — JSON parsing for cache_key_params
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
                except Exception as e:  # intentionally broad — re-raises after rollback
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

                    if "file_size" in updates:
                        entry.file_size = updates["file_size"]
                    if "file_hash" in updates:
                        entry.file_hash = updates["file_hash"]
                    elif "content_hash" in updates:
                        entry.file_hash = updates["content_hash"]
                    if "actual_path" in updates:
                        ap = updates["actual_path"]
                        entry.actual_path = str(ap) if ap is not None else None
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
                    if "metadata_dict" in updates:
                        entry.metadata_dict = _ensure_jsonb_value(
                            updates["metadata_dict"]
                        )
                    if "blob_data" in updates:
                        entry.blob_data = updates["blob_data"]
                    if "is_inline" in updates:
                        entry.is_inline = updates["is_inline"]
                    if "inline_ext" in updates:
                        entry.inline_ext = updates["inline_ext"]
                    if "encryption_algorithm" in updates:
                        entry.encryption_algorithm = updates["encryption_algorithm"]
                    if "encryption_iv" in updates:
                        entry.encryption_iv = updates["encryption_iv"]
                    if "cacheness_version" in updates:
                        entry.cacheness_version = updates["cacheness_version"]
                    if "created_at" in updates:
                        created_at = updates["created_at"]
                        if isinstance(created_at, str):
                            created_at = datetime.fromisoformat(created_at)
                            if created_at.tzinfo is None:
                                created_at = created_at.replace(tzinfo=timezone.utc)
                            else:
                                created_at = created_at.astimezone(timezone.utc)
                        entry.created_at = created_at
                    if "accessed_at" in updates:
                        accessed_at = updates["accessed_at"]
                        if isinstance(accessed_at, str):
                            accessed_at = datetime.fromisoformat(accessed_at)
                            if accessed_at.tzinfo is None:
                                accessed_at = accessed_at.replace(tzinfo=timezone.utc)
                            else:
                                accessed_at = accessed_at.astimezone(timezone.utc)
                        entry.accessed_at = accessed_at
                    if "ttl_seconds" in updates:
                        entry.ttl_seconds = updates["ttl_seconds"]
                    if "expires_at" in updates:
                        expires_at = updates["expires_at"]
                        if isinstance(expires_at, str):
                            expires_at = datetime.fromisoformat(expires_at)
                            if expires_at.tzinfo is None:
                                expires_at = expires_at.replace(tzinfo=timezone.utc)
                            else:
                                expires_at = expires_at.astimezone(timezone.utc)
                        entry.expires_at = expires_at

                    session.commit()
                    return True
                except Exception as e:  # intentionally broad — re-raises after rollback
                    session.rollback()
                    logger.error(f"Failed to update entry {cache_key}: {e}")
                    raise

    def iter_entry_summaries(self) -> List[EntrySummary]:
        """Return lightweight flat entry dicts — raw SQL, no ORM hydration."""
        with self.SessionLocal() as session:
            tbl = self._entries_table
            rows = session.execute(
                text(
                    f"SELECT cache_key, data_type, description, "
                    f"       file_size, created_at, accessed_at, "
                    f"       object_type, storage_format, serializer, "
                    f"       compression_codec, actual_path, "
                    f"       file_hash, entry_signature, metadata_dict, "
                    f"       s3_etag, access_count, ttl_seconds, expires_at, "
                    f"       is_inline, "
                    f"       encryption_algorithm, encryption_iv, cacheness_version "
                    f'FROM "{tbl}"'
                )
            ).fetchall()
            result: List[EntrySummary] = []
            for row in rows:
                flat: EntrySummary = {
                    "cache_key": row[0],
                    "data_type": row[1],
                    "description": row[2] or "",
                    "file_size": row[3] or 0,
                    "created_at": row[4],
                    "accessed_at": row[5],
                }
                if row[6] is not None:
                    flat["object_type"] = row[6]
                if row[7] is not None:
                    flat["storage_format"] = row[7]
                if row[8] is not None:
                    flat["serializer"] = row[8]
                if row[9] is not None:
                    flat["compression_codec"] = row[9]
                if row[10] is not None:
                    flat["actual_path"] = row[10]
                if row[11] is not None:
                    flat["file_hash"] = row[11]
                if row[12] is not None:
                    flat["entry_signature"] = row[12]
                if row[13] is not None:
                    flat["metadata_dict"] = row[13]
                if row[14] is not None:
                    flat["s3_etag"] = row[14]
                # Phase 1 columns
                flat["access_count"] = row[15] or 0
                if row[16] is not None:
                    flat["ttl_seconds"] = row[16]
                if row[17] is not None:
                    flat["expires_at"] = row[17]
                # Phase 2 inline blob flag
                flat["is_inline"] = row[18] or 0
                if row[19] is not None:
                    flat["encryption_algorithm"] = row[19]
                if row[20] is not None:
                    flat["encryption_iv"] = row[20]
                if row[21] is not None:
                    flat["cacheness_version"] = row[21]
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
                "total_size_mb": stats.total_size_bytes
                / (1024 * 1024),  # Backward compat
                "last_cleanup_at": (
                    stats.last_cleanup_at.isoformat() if stats.last_cleanup_at else None
                ),
            }

    def update_access_time(self, cache_key: str):
        """Update last access time and increment access count for cache entry."""
        with self.SessionLocal() as session:
            try:
                session.execute(
                    update(self._PgCacheEntry)
                    .where(self._PgCacheEntry.cache_key == cache_key)
                    .values(
                        accessed_at=datetime.now(timezone.utc),
                        access_count=self._PgCacheEntry.access_count + 1,
                    )
                )
                session.commit()
            except (
                Exception
            ) as e:  # intentionally broad — access time update is best-effort
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
            except (
                Exception
            ) as e:  # intentionally broad — hits increment is best-effort
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
            except (
                Exception
            ) as e:  # intentionally broad — misses increment is best-effort
                session.rollback()
                logger.warning(f"Failed to increment misses: {e}")

    def cleanup_expired(self, ttl_seconds: float) -> int:
        """Remove expired entries and return count removed."""
        now = datetime.now(timezone.utc)
        expired_stored = and_(
            self._PgCacheEntry.expires_at.is_not(None),
            self._PgCacheEntry.expires_at < now,
        )
        if ttl_seconds and ttl_seconds > 0:
            cutoff = now - timedelta(seconds=ttl_seconds)
            expired_filter = or_(
                expired_stored,
                and_(
                    self._PgCacheEntry.expires_at.is_(None),
                    self._PgCacheEntry.created_at < cutoff,
                ),
            )
        else:
            expired_filter = expired_stored

        with self._lock:
            with self.SessionLocal() as session:
                try:
                    # Count entries to be removed
                    count = (
                        session.execute(
                            select(func.count())
                            .select_from(self._PgCacheEntry)
                            .where(expired_filter)
                        ).scalar()
                        or 0
                    )

                    if count > 0:
                        session.execute(
                            delete(self._PgCacheEntry).where(expired_filter)
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

                except (
                    Exception
                ) as e:  # intentionally broad — re-raises cleanup failure
                    session.rollback()
                    logger.error(f"Cleanup failed: {e}")
                    return 0

    def cleanup_by_size(self, target_size_bytes: int) -> Dict[str, Any]:
        """Remove least-recently-accessed entries until cache size drops to or below target."""
        with self._lock:
            with self.SessionLocal() as session:
                try:
                    # Get current total size in bytes
                    CE = self._PgCacheEntry
                    result = session.execute(
                        select(func.sum(CE.file_size)).select_from(CE)
                    )
                    current_size_bytes = result.scalar() or 0

                    if current_size_bytes <= target_size_bytes:
                        return {
                            "count": 0,
                            "removed_entries": [],
                        }  # Already at or below target

                    bytes_to_remove = current_size_bytes - target_size_bytes

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
                            f"LRU cleanup: removed {len(removed_entries)} entries to reach {format_size(target_size_bytes)}"
                        )

                    return {
                        "count": len(removed_entries),
                        "removed_entries": removed_entries,
                    }

                except Exception as e:  # intentionally broad — LRU cleanup failure
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

                except Exception as e:  # intentionally broad — clear all failure
                    session.rollback()
                    logger.error(f"Clear all failed: {e}")
                    return 0

    def close(self):
        """Close database connections."""
        try:
            self.engine.dispose()
            logger.debug("PostgreSQL engine disposed")
        except Exception as e:  # intentionally broad — connection close is best-effort
            logger.warning(f"Error closing PostgreSQL connection: {e}")

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:  # intentionally broad — cleanup must not raise
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
        except Exception as e:  # intentionally broad — auto-registration is best-effort
            logger.debug(f"Could not auto-register PostgreSQL backend: {e}")


# Attempt auto-registration
_auto_register()
