"""SQLite database-based metadata backend using SQLAlchemy ORM."""

import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

from sqlalchemy import (
    create_engine,
    select,
    update,
    delete,
    desc,
    func,
    text,
    case,
)
from sqlalchemy.orm import sessionmaker

from ..interfaces import EntrySummary
from ..json_utils import dumps as json_dumps, loads as json_loads
from ..size_utils import bytes_to_mb_display
from ._compat import (
    SQLALCHEMY_AVAILABLE,
    Base,
    CacheStats,
    CacheNamespace,
    _get_namespace_models,
    _CORE_TABLES,
    DEFAULT_NAMESPACE,
    validate_namespace_id,
    NamespaceInfo,
)
from .base import MetadataBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema migration functions (used by SqliteBackend._get_migrations)
# ---------------------------------------------------------------------------


def _sqlite_migrate_v1_to_v2(backend: "SqliteBackend", namespace_id: str) -> None:
    """v1 → v2: add partial index on ``metadata_dict IS NOT NULL``.

    The ``query_meta()`` fast path always filters
    ``WHERE metadata_dict IS NOT NULL``.  A partial B-tree index on
    ``created_at DESC`` (filtered to non-NULL rows) lets SQLite skip
    entries that have no custom metadata and return results pre-sorted.

    Uses ``CREATE INDEX IF NOT EXISTS`` to be fully idempotent.
    """
    if namespace_id == DEFAULT_NAMESPACE:
        table = "cache_entries"
        idx_name = "idx_metadata_notnull"
    else:
        table = f"cache_entries_{namespace_id}"
        idx_name = f"idx_{namespace_id}_metadata_notnull"

    with backend.SessionLocal() as session:
        session.execute(
            text(
                f'CREATE INDEX IF NOT EXISTS "{idx_name}" '
                f'ON "{table}" (created_at DESC) '
                f"WHERE metadata_dict IS NOT NULL"
            )
        )
        session.commit()

    logger.info("SQLite v1→v2: created partial index %r on %r", idx_name, table)


def _sqlite_migrate_v2_to_v3(backend: "SqliteBackend", namespace_id: str) -> None:
    """v2 → v3: add access_count, ttl_seconds, expires_at, blob_data, is_inline columns.

    Adds five new columns:

    * ``access_count`` — per-entry access counter (INTEGER NOT NULL DEFAULT 0)
    * ``ttl_seconds`` — per-entry TTL storage (INTEGER nullable)
    * ``expires_at`` — pre-computed expiry timestamp (DATETIME nullable)
    * ``blob_data`` — inline blob content (BLOB nullable)
    * ``is_inline`` — flag: 1 if blob stored inline, 0 otherwise (INTEGER NOT NULL DEFAULT 0)

    Also creates two indexes:

    * ``idx_expires_at`` — partial index on ``expires_at`` WHERE NOT NULL.
    * ``idx_access_count`` — composite (access_count, accessed_at).

    All columns are nullable (or DEFAULT 0) so existing rows are unaffected.
    Uses ``IF NOT EXISTS`` for full idempotency.
    """
    if namespace_id == DEFAULT_NAMESPACE:
        table = "cache_entries"
        idx_expires = "idx_expires_at"
        idx_access = "idx_access_count"
    else:
        table = f"cache_entries_{namespace_id}"
        idx_expires = f"idx_{namespace_id}_expires_at"
        idx_access = f"idx_{namespace_id}_access_count"

    with backend.SessionLocal() as session:
        # Add new columns (SQLite ignores ADD COLUMN if column already exists
        # when wrapped in try/except — but we use a pragma check to be safe)
        existing_cols = {
            row[1]
            for row in session.execute(text(f'PRAGMA table_info("{table}")')).fetchall()
        }

        if "access_count" not in existing_cols:
            session.execute(
                text(
                    f'ALTER TABLE "{table}" ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0'
                )
            )
        if "ttl_seconds" not in existing_cols:
            session.execute(
                text(f'ALTER TABLE "{table}" ADD COLUMN ttl_seconds INTEGER')
            )
        if "expires_at" not in existing_cols:
            session.execute(
                text(f'ALTER TABLE "{table}" ADD COLUMN expires_at DATETIME')
            )
        if "blob_data" not in existing_cols:
            session.execute(text(f'ALTER TABLE "{table}" ADD COLUMN blob_data BLOB'))
        if "is_inline" not in existing_cols:
            session.execute(
                text(
                    f'ALTER TABLE "{table}" ADD COLUMN is_inline INTEGER NOT NULL DEFAULT 0'
                )
            )
        if "inline_ext" not in existing_cols:
            session.execute(text(f'ALTER TABLE "{table}" ADD COLUMN inline_ext TEXT'))

        # Create indexes
        session.execute(
            text(
                f'CREATE INDEX IF NOT EXISTS "{idx_expires}" '
                f'ON "{table}" (expires_at) '
                f"WHERE expires_at IS NOT NULL"
            )
        )
        session.execute(
            text(
                f'CREATE INDEX IF NOT EXISTS "{idx_access}" '
                f'ON "{table}" (access_count, accessed_at)'
            )
        )
        session.commit()

    logger.info(
        "SQLite v2→v3: added access_count/ttl_seconds/expires_at/blob_data/is_inline "
        "columns and indexes on %r",
        table,
    )


def _sqlite_migrate_v3_to_v4(backend: "SqliteBackend", namespace_id: str) -> None:
    """v3 → v4: add encryption_algorithm, encryption_iv, cacheness_version columns."""
    if namespace_id == DEFAULT_NAMESPACE:
        table = "cache_entries"
    else:
        table = f"cache_entries_{namespace_id}"

    with backend.SessionLocal() as session:
        existing_cols = {
            row[1]
            for row in session.execute(text(f'PRAGMA table_info("{table}")')).fetchall()
        }

        if "encryption_algorithm" not in existing_cols:
            session.execute(
                text(f'ALTER TABLE "{table}" ADD COLUMN encryption_algorithm TEXT')
            )
        if "encryption_iv" not in existing_cols:
            session.execute(
                text(f'ALTER TABLE "{table}" ADD COLUMN encryption_iv TEXT')
            )
        if "cacheness_version" not in existing_cols:
            session.execute(
                text(f'ALTER TABLE "{table}" ADD COLUMN cacheness_version TEXT')
            )
        session.commit()

    logger.info(
        "SQLite v3→v4: added encryption_algorithm/encryption_iv/cacheness_version "
        "columns on %r",
        table,
    )


class SqliteBackend(MetadataBackend):
    """SQLite database-based metadata backend using SQLAlchemy ORM."""

    def __init__(
        self,
        db_file: str = "cache_metadata.db",
        echo: bool = False,
        namespace: str = DEFAULT_NAMESPACE,
    ):
        """
        Initialize SQLite metadata backend.

        Args:
            db_file: Path to SQLite database file
            echo: Whether to echo SQL queries (for debugging)
            namespace: Active namespace for this backend instance
        """
        self._active_namespace = validate_namespace_id(namespace)
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is required for SQLite backend. Install with: pip install sqlalchemy"
            )

        self.db_file = db_file

        # Configure SQLite engine with appropriate optimizations
        # Note: SQLite uses SingletonThreadPool which doesn't support pool_size/max_overflow
        self.engine = create_engine(
            f"sqlite:///{db_file}",
            echo=echo,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            connect_args={
                "check_same_thread": False,  # Allow multi-threading
                "timeout": 30,  # Longer timeout for database locks
            },
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
            cursor.execute("PRAGMA synchronous=NORMAL")  # Good balance of safety/speed
            cursor.execute(
                "PRAGMA cache_size=20000"
            )  # 20MB cache (increased from 10MB)
            cursor.execute("PRAGMA temp_store=MEMORY")  # Temp tables in memory
            cursor.execute(
                "PRAGMA mmap_size=536870912"
            )  # 512MB memory mapped I/O (doubled)
            cursor.execute("PRAGMA page_size=32768")  # Larger page size for better I/O

            # Query optimization pragmas
            cursor.execute("PRAGMA optimize")  # Enable query planner optimizations
            cursor.execute("PRAGMA analysis_limit=1000")  # Better statistics

            # Concurrent access optimizations
            cursor.execute("PRAGMA busy_timeout=30000")  # 30s busy timeout
            cursor.execute(
                "PRAGMA wal_autocheckpoint=1000"
            )  # WAL checkpoint every 1000 pages

            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys=ON")

            cursor.close()

        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        self._lock = (
            threading.RLock()
        )  # RLock: allows re-entrant acquisition from composed operations

        # Create core tables only (cache_entries, cache_stats,
        # cacheness_namespaces).  Custom metadata tables are created
        # separately via migrate_custom_metadata_tables() to prevent
        # stale models from polluting unrelated databases (CACHE-qg3).
        Base.metadata.create_all(self.engine, tables=list(_CORE_TABLES))

        # Resolve namespace-specific ORM models (EntityName pattern)
        self._CacheEntry, self._CacheStats = _get_namespace_models(
            self._active_namespace
        )
        self._entries_table = self._CacheEntry.__tablename__
        self._stats_table = self._CacheStats.__tablename__

        # For non-default namespaces, ensure their tables exist too
        if self._active_namespace != DEFAULT_NAMESPACE:
            self._CacheEntry.__table__.create(self.engine, checkfirst=True)
            self._CacheStats.__table__.create(self.engine, checkfirst=True)

        # Run formal schema versioning migrations
        self._ensure_namespace_registry()
        self.run_all_migrations()

        # Initialize stats if not exists
        self._init_stats()

        logger.info(f"✅ SQLAlchemy metadata backend initialized: {db_file}")

    # --- Schema versioning overrides ---

    def _ensure_namespace_registry(self):
        """Ensure the namespace registry exists and has a 'default' entry.

        This handles the v0→v1 transition: if cacheness_namespaces was just
        created (by ``create_all``), seed it with the ``'default'`` namespace
        pointing to the existing unsuffixed tables.
        """
        with self.SessionLocal() as session:
            existing = session.execute(
                select(CacheNamespace).where(
                    CacheNamespace.namespace_id == DEFAULT_NAMESPACE
                )
            ).scalar_one_or_none()

            if existing is None:
                # Seed the default namespace
                ns = CacheNamespace(
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
                select(CacheNamespace).where(
                    CacheNamespace.namespace_id == namespace_id
                )
            ).scalar_one_or_none()
            return ns.schema_version if ns else 0

    def set_schema_version(self, namespace_id: str, version: int) -> None:
        """Write schema version to the namespace registry."""
        with self.SessionLocal() as session:
            session.execute(
                update(CacheNamespace)
                .where(CacheNamespace.namespace_id == namespace_id)
                .values(schema_version=version)
            )
            session.commit()

    def get_migrations(self) -> list:
        """Return SQLite-specific schema migrations.

        Schema v1: baseline (namespace registry).
        Schema v2: partial index on ``metadata_dict IS NOT NULL``
                   for ``query_meta()`` performance.
        """
        return [
            (1, 2, _sqlite_migrate_v1_to_v2),
            (2, 3, _sqlite_migrate_v2_to_v3),
            (3, 4, _sqlite_migrate_v3_to_v4),
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

        with self._lock, self.SessionLocal() as session:
            # Check for duplicates
            existing = session.execute(
                select(CacheNamespace).where(
                    CacheNamespace.namespace_id == namespace_id
                )
            ).scalar_one_or_none()
            if existing is not None:
                raise ValueError(f"Namespace {namespace_id!r} already exists")

            # Create the per-namespace tables
            entries_table = f"cache_entries_{namespace_id}"
            stats_table = f"cache_stats_{namespace_id}"

            session.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS "{entries_table}" (
                    cache_key       VARCHAR(16) PRIMARY KEY,
                    description     VARCHAR(500) NOT NULL DEFAULT '',
                    data_type       VARCHAR(20) NOT NULL,
                    created_at      DATETIME NOT NULL,
                    accessed_at     DATETIME NOT NULL,
                    file_size       INTEGER NOT NULL DEFAULT 0,
                    file_hash       VARCHAR(16),
                    entry_signature VARCHAR(100),
                    s3_etag         VARCHAR(100),
                    object_type     VARCHAR(100),
                    storage_format  VARCHAR(20),
                    serializer      VARCHAR(20),
                    compression_codec VARCHAR(20),
                    actual_path     VARCHAR(500),
                    cache_key_params TEXT,
                    metadata_dict   TEXT,
                    access_count    INTEGER NOT NULL DEFAULT 0,
                    ttl_seconds     INTEGER,
                    expires_at      DATETIME,
                    blob_data       BLOB,
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
                    last_updated    DATETIME NOT NULL
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
                    f'CREATE INDEX IF NOT EXISTS "idx_{namespace_id}_metadata_notnull" '
                    f'ON "{entries_table}" (created_at DESC) '
                    f"WHERE metadata_dict IS NOT NULL"
                )
            )
            session.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS "idx_{namespace_id}_expires_at" '
                    f'ON "{entries_table}" (expires_at) '
                    f"WHERE expires_at IS NOT NULL"
                )
            )
            session.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS "idx_{namespace_id}_access_count" '
                    f'ON "{entries_table}" (access_count, accessed_at)'
                )
            )

            # Register in the namespace registry
            now = datetime.now(timezone.utc)
            ns = CacheNamespace(
                namespace_id=namespace_id,
                display_name=display_name,
                schema_version=4,
                created_at=now,
            )
            session.add(ns)

            # Initialize stats row for the new namespace
            session.execute(
                text(
                    f'INSERT OR IGNORE INTO "{stats_table}" '
                    f"(id, cache_hits, cache_misses, last_updated) "
                    f"VALUES (1, 0, 0, :now)"
                ),
                {"now": now},
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

        with self._lock, self.SessionLocal() as session:
            existing = session.execute(
                select(CacheNamespace).where(
                    CacheNamespace.namespace_id == namespace_id
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
                delete(CacheNamespace).where(
                    CacheNamespace.namespace_id == namespace_id
                )
            )
            session.commit()

            logger.info(f"Dropped namespace {namespace_id!r} and its tables")
            return True

    def list_namespaces(self) -> list:
        """List all registered namespaces from the registry."""
        with self._lock, self.SessionLocal() as session:
            rows = (
                session.execute(
                    select(CacheNamespace).order_by(CacheNamespace.created_at)
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

    def get_namespace(self, namespace_id: str):
        """Get info for a specific namespace."""
        with self.SessionLocal() as session:
            row = session.execute(
                select(CacheNamespace).where(
                    CacheNamespace.namespace_id == namespace_id
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
        """Store namespace signature in the SQLite registry."""
        with self.SessionLocal() as session:
            session.execute(
                update(CacheNamespace)
                .where(CacheNamespace.namespace_id == namespace_id)
                .values(signature=signature)
            )
            session.commit()

    def _run_migrations(self):
        """Legacy migration method — delegates to formal schema versioning.

        Kept for backward compatibility.  New code should call
        ``run_migrations()`` directly.
        """
        # Already handled in __init__ via run_migrations()
        pass

    def _init_stats(self):
        """Initialize cache stats if not exists."""
        with self.SessionLocal() as session:
            stats = session.execute(
                select(self._CacheStats).where(self._CacheStats.id == 1)
            ).scalar_one_or_none()

            if not stats:
                stats = self._CacheStats(id=1)
                session.add(stats)
                session.commit()

    def _get_stats_row(self, session) -> "CacheStats":
        """Get the single stats row."""
        stats = session.execute(
            select(self._CacheStats).where(self._CacheStats.id == 1)
        ).scalar_one_or_none()

        if not stats:
            stats = self._CacheStats(id=1)
            session.add(stats)
            session.commit()
        return stats

    def get_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get specific cache entry metadata — Core column select, no ORM hydration."""
        with self.SessionLocal() as session:
            CE = self._CacheEntry
            # Core column select — avoids ORM identity-map overhead
            row = session.execute(
                select(
                    CE.description,
                    CE.data_type,
                    CE.created_at,
                    CE.accessed_at,
                    CE.file_size,
                    CE.object_type,
                    CE.storage_format,
                    CE.serializer,
                    CE.compression_codec,
                    CE.actual_path,
                    CE.file_hash,
                    CE.entry_signature,
                    CE.s3_etag,
                    CE.cache_key_params,
                    CE.access_count,
                    CE.ttl_seconds,
                    CE.expires_at,
                    CE.blob_data,
                    CE.is_inline,
                    CE.inline_ext,
                    CE.encryption_algorithm,
                    CE.encryption_iv,
                    CE.cacheness_version,
                    CE.metadata_dict,
                ).where(CE.cache_key == cache_key)
            ).one_or_none()

            if row is None:
                return None

            # Build metadata dict from Row tuple — no ORM attribute overhead
            metadata: Dict[str, Any] = {}

            # Backend technical metadata from dedicated columns (not JSON)
            if row.object_type is not None:
                metadata["object_type"] = row.object_type
            if row.storage_format is not None:
                metadata["storage_format"] = row.storage_format
            if row.serializer is not None:
                metadata["serializer"] = row.serializer
            if row.compression_codec is not None:
                metadata["compression_codec"] = row.compression_codec
            if row.actual_path is not None:
                metadata["actual_path"] = row.actual_path

            # Optional security fields
            if row.file_hash is not None:
                metadata["file_hash"] = row.file_hash
            if row.entry_signature is not None:
                metadata["entry_signature"] = row.entry_signature
            if row.s3_etag is not None:
                metadata["s3_etag"] = row.s3_etag
            if row.inline_ext is not None:
                metadata["inline_ext"] = row.inline_ext
            if row.encryption_algorithm is not None:
                metadata["encryption_algorithm"] = row.encryption_algorithm
            if row.encryption_iv is not None:
                metadata["encryption_iv"] = row.encryption_iv
            if row.cacheness_version is not None:
                metadata["cacheness_version"] = row.cacheness_version

            # Include metadata_dict (user-facing kwargs) if stored
            if row.metadata_dict is not None:
                metadata["metadata_dict"] = row.metadata_dict

            # Only parse cache_key_params JSON if it exists (disabled by default)
            if row.cache_key_params is not None:
                try:
                    metadata["cache_key_params"] = json_loads(row.cache_key_params)
                except (ValueError, TypeError):
                    pass  # Skip malformed cache_key_params

            # Ensure timestamps are always in UTC for consistency
            created_at_utc = (
                row.created_at.astimezone(timezone.utc)
                if row.created_at.tzinfo
                else row.created_at.replace(tzinfo=timezone.utc)
            )
            accessed_at_utc = (
                row.accessed_at.astimezone(timezone.utc)
                if row.accessed_at.tzinfo
                else row.accessed_at.replace(tzinfo=timezone.utc)
            )

            return {
                "description": row.description,
                "data_type": row.data_type,
                "created_at": created_at_utc.isoformat(),
                "accessed_at": accessed_at_utc.isoformat(),
                "file_size": row.file_size,
                "access_count": row.access_count or 0,
                "ttl_seconds": row.ttl_seconds,
                "expires_at": (
                    row.expires_at.astimezone(timezone.utc).isoformat()
                    if row.expires_at and hasattr(row.expires_at, "astimezone")
                    else row.expires_at
                ),
                "is_inline": row.is_inline or 0,
                "blob_data": row.blob_data,
                "metadata": metadata,
            }

    def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
        """Store cache entry metadata using dedicated columns - zero JSON overhead for backend data."""
        with self._lock, self.SessionLocal() as session:
            # Extract and process metadata fields efficiently
            metadata = entry_data.get("metadata", {}).copy()

            # Extract full metadata copy FIRST (before any pop() operations) if present
            full_metadata_dict = metadata.pop("_full_metadata", None)
            full_metadata_json = None
            if full_metadata_dict is not None:
                try:
                    full_metadata_json = json_dumps(full_metadata_dict)
                    logger.debug(
                        f"Serialized full_metadata: {len(full_metadata_json)} chars"
                    )
                except (
                    Exception
                ) as e:  # intentionally broad — JSON serialization fallback
                    # If serialization fails, skip full_metadata
                    logger.warning(f"Failed to serialize full_metadata JSON: {e}")
                    full_metadata_json = None

            # Extract backend technical metadata to dedicated columns (not JSON)
            object_type = metadata.pop("object_type", None)
            storage_format = metadata.pop("storage_format", None)
            serializer = metadata.pop("serializer", None)
            compression_codec = metadata.pop("compression_codec", None)
            actual_path = metadata.pop("actual_path", None)

            # Extract security fields from metadata (remove from JSON to avoid duplication)
            file_hash = metadata.pop("file_hash", None)
            entry_signature = metadata.pop("entry_signature", None)
            s3_etag = metadata.pop("s3_etag", None)  # S3 ETag if using S3 backend
            # These fields are pre-serialized JSON strings from the caching layer
            cache_key_params = metadata.pop("cache_key_params", None)
            metadata_dict_value = metadata.pop(
                "metadata_dict", None
            )  # User metadata for querying
            inline_ext = metadata.pop("inline_ext", None)
            encryption_algorithm = metadata.pop("encryption_algorithm", None)
            encryption_iv = metadata.pop("encryption_iv", None)
            cacheness_version = metadata.pop("cacheness_version", None)

            # Remove redundant fields that are already stored as columns
            metadata.pop("data_type", None)  # Already stored in data_type column

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

            # Handle TTL fields
            ttl_seconds_val = entry_data.get("ttl_seconds")
            expires_at = entry_data.get("expires_at")
            if expires_at is None and ttl_seconds_val is not None:
                # Compute expires_at from created_at + ttl_seconds
                created_dt = created_at
                if isinstance(created_dt, str):
                    created_dt = datetime.fromisoformat(created_dt)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                expires_at = created_dt + timedelta(seconds=float(ttl_seconds_val))
            elif isinstance(expires_at, str):
                expires_at = datetime.fromisoformat(expires_at)

            # Use efficient INSERT OR REPLACE with dedicated columns - zero JSON overhead
            from sqlalchemy import text

            tbl = self._entries_table
            session.execute(
                text(f"""
                    INSERT OR REPLACE INTO "{tbl}"
                    (cache_key, description, data_type, file_size, 
                     file_hash, entry_signature, s3_etag, cache_key_params, metadata_dict,
                     object_type, storage_format, serializer, compression_codec, actual_path,
                     created_at, accessed_at, access_count, ttl_seconds, expires_at,
                     blob_data, is_inline, inline_ext,
                     encryption_algorithm, encryption_iv, cacheness_version)
                    VALUES (:cache_key, :description, :data_type, :file_size, 
                           :file_hash, :entry_signature, :s3_etag, :cache_key_params, :metadata_dict,
                           :object_type, :storage_format, :serializer, :compression_codec, :actual_path,
                           :created_at, :accessed_at, :access_count, :ttl_seconds, :expires_at,
                           :blob_data, :is_inline, :inline_ext,
                           :encryption_algorithm, :encryption_iv, :cacheness_version)
                """),
                {
                    "cache_key": cache_key,
                    "description": entry_data.get("description", ""),
                    "data_type": entry_data.get("data_type", "unknown"),
                    "file_size": entry_data.get("file_size", 0),
                    "file_hash": file_hash,
                    "entry_signature": entry_signature,
                    "s3_etag": s3_etag,
                    "cache_key_params": cache_key_params,
                    "metadata_dict": metadata_dict_value,
                    "object_type": object_type,
                    "storage_format": storage_format,
                    "serializer": serializer,
                    "compression_codec": compression_codec,
                    "actual_path": actual_path,
                    "created_at": created_at,
                    "accessed_at": accessed_at,
                    "access_count": entry_data.get("access_count", 0),
                    "ttl_seconds": ttl_seconds_val,
                    "expires_at": expires_at,
                    "blob_data": entry_data.get("blob_data"),
                    "is_inline": entry_data.get("is_inline", 0),
                    "inline_ext": inline_ext,
                    "encryption_algorithm": encryption_algorithm,
                    "encryption_iv": encryption_iv,
                    "cacheness_version": cacheness_version,
                },
            )
            session.commit()

    def remove_entry(self, cache_key: str) -> bool:
        """Remove cache entry metadata (custom metadata will cascade delete via FK)."""
        with self._lock, self.SessionLocal() as session:
            # Delete cache entry - custom metadata records will cascade delete automatically
            # due to ondelete="CASCADE" on the cache_key foreign key
            result = session.execute(
                delete(self._CacheEntry).where(self._CacheEntry.cache_key == cache_key)
            )
            session.commit()
            return result.rowcount > 0

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
        with self._lock, self.SessionLocal() as session:
            # Check if entry exists
            entry = session.execute(
                select(self._CacheEntry).where(self._CacheEntry.cache_key == cache_key)
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

            session.commit()
            return True

    def iter_entry_summaries(self) -> List[EntrySummary]:
        """Return lightweight flat entry dicts — raw SQL, no ORM hydration."""
        from sqlalchemy import text

        with self._lock, self.SessionLocal() as session:
            tbl = self._entries_table
            rows = session.execute(
                text(
                    f"SELECT cache_key, data_type, description, "
                    f"       file_size, created_at, accessed_at, "
                    f"       object_type, storage_format, serializer, "
                    f"       compression_codec, actual_path, "
                    f"       file_hash, entry_signature, metadata_dict, "
                    f"       s3_etag, access_count, ttl_seconds, expires_at, "
                    f"       is_inline "
                    f"       ,encryption_algorithm, encryption_iv, cacheness_version "
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
                # Only include non-None technical metadata
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

    def keys_by_prefix(self, prefix: str) -> list[str]:
        """Return cache keys starting with *prefix* using SQL LIKE."""
        with self._lock, self.SessionLocal() as session:
            from sqlalchemy import text

            tbl = self._entries_table
            rows = session.execute(
                text(f'SELECT cache_key FROM "{tbl}" WHERE cache_key LIKE :pattern'),
                {"pattern": prefix + "%"},
            ).fetchall()
            return [row[0] for row in rows]

    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries — Core column select, no ORM hydration."""
        with self._lock, self.SessionLocal() as session:
            CE = self._CacheEntry
            # Core column select — avoids SQLAlchemy ORM identity-map overhead
            rows = session.execute(
                select(
                    CE.cache_key,
                    CE.data_type,
                    CE.description,
                    CE.created_at,
                    CE.accessed_at,
                    CE.file_size,
                    CE.object_type,
                    CE.storage_format,
                    CE.serializer,
                    CE.compression_codec,
                    CE.actual_path,
                    CE.file_hash,
                    CE.entry_signature,
                    CE.s3_etag,
                    CE.cache_key_params,
                ).order_by(desc(CE.created_at))
            ).fetchall()

            result = []
            for row in rows:
                # Build metadata dict from Row tuple — no ORM attribute overhead
                entry_metadata: Dict[str, Any] = {}

                # Backend technical metadata from dedicated columns (not JSON)
                if row.object_type is not None:
                    entry_metadata["object_type"] = row.object_type
                if row.storage_format is not None:
                    entry_metadata["storage_format"] = row.storage_format
                if row.serializer is not None:
                    entry_metadata["serializer"] = row.serializer
                if row.compression_codec is not None:
                    entry_metadata["compression_codec"] = row.compression_codec
                if row.actual_path is not None:
                    entry_metadata["actual_path"] = row.actual_path

                # Optional security fields
                if row.file_hash is not None:
                    entry_metadata["file_hash"] = row.file_hash
                if row.entry_signature is not None:
                    entry_metadata["entry_signature"] = row.entry_signature
                if row.s3_etag is not None:
                    entry_metadata["s3_etag"] = row.s3_etag

                # Only parse cache_key_params JSON if it exists (disabled by default)
                if row.cache_key_params is not None:
                    try:
                        entry_metadata["cache_key_params"] = json_loads(
                            row.cache_key_params
                        )
                    except (ValueError, TypeError):
                        pass  # Skip malformed cache_key_params

                result.append(
                    {
                        "cache_key": row.cache_key,
                        "data_type": row.data_type,
                        "description": row.description,
                        "metadata": entry_metadata,
                        "created": row.created_at.isoformat(),
                        "last_accessed": row.accessed_at.isoformat(),
                        "size_mb": bytes_to_mb_display(row.file_size),
                    }
                )

            return result

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics using SQL aggregates (no full table scan)."""
        with self._lock, self.SessionLocal() as session:
            # Single aggregation query — no Python-side iteration
            CE = self._CacheEntry
            row = session.execute(
                select(
                    func.count(CE.cache_key).label("total"),
                    func.coalesce(func.sum(CE.file_size), 0).label("total_size"),
                    func.count(
                        case(
                            (CE.data_type == "dataframe", 1),
                        )
                    ).label("dataframe_count"),
                    func.count(
                        case(
                            (CE.data_type == "array", 1),
                        )
                    ).label("array_count"),
                )
            ).one()

            total_size_bytes = row.total_size
            total_size_mb = total_size_bytes / (1024 * 1024)

            # Get hit/miss stats
            stats = self._get_stats_row(session)
            hit_rate = (
                stats.cache_hits / (stats.cache_hits + stats.cache_misses)
                if (stats.cache_hits + stats.cache_misses) > 0
                else 0.0
            )

            return {
                "total_entries": row.total,
                "dataframe_entries": row.dataframe_count,
                "array_entries": row.array_count,
                "total_size_bytes": total_size_bytes,
                "total_size_mb": total_size_mb,  # Backward compat — prefer total_size_bytes
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "hit_rate": round(hit_rate, 3),
            }

    def update_access_time(self, cache_key: str):
        """Update last access time and increment access count for cache entry."""
        with self._lock, self.SessionLocal() as session:
            session.execute(
                update(self._CacheEntry)
                .where(self._CacheEntry.cache_key == cache_key)
                .values(
                    accessed_at=datetime.now(timezone.utc),
                    access_count=self._CacheEntry.access_count + 1,
                )
            )
            session.commit()

    def increment_hits(self):
        """Increment cache hits counter."""
        with self._lock, self.SessionLocal() as session:
            session.execute(
                update(self._CacheStats)
                .where(self._CacheStats.id == 1)
                .values(
                    cache_hits=self._CacheStats.cache_hits + 1,
                    last_updated=datetime.now(timezone.utc),
                )
            )
            session.commit()

    def increment_misses(self):
        """Increment cache misses counter."""
        with self._lock, self.SessionLocal() as session:
            session.execute(
                update(self._CacheStats)
                .where(self._CacheStats.id == 1)
                .values(
                    cache_misses=self._CacheStats.cache_misses + 1,
                    last_updated=datetime.now(timezone.utc),
                )
            )
            session.commit()

    def cleanup_expired(self, ttl_seconds: float) -> int:
        """Remove expired entries and return count removed."""

        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)

        with self._lock, self.SessionLocal() as session:
            # Delete expired entries
            result = session.execute(
                delete(self._CacheEntry).where(
                    self._CacheEntry.created_at < cutoff_time
                )
            )
            deleted_count = result.rowcount
            session.commit()
            return deleted_count

    def cleanup_by_size(self, target_size_bytes: int) -> Dict[str, Any]:
        """Remove least-recently-accessed entries until cache size drops to or below target."""
        with self._lock, self.SessionLocal() as session:
            # Get current total size in bytes
            CE = self._CacheEntry
            result = session.execute(select(func.sum(CE.file_size)).select_from(CE))
            current_size_bytes = result.scalar() or 0

            if current_size_bytes <= target_size_bytes:
                return {"count": 0, "removed_entries": []}  # Already at or below target

            bytes_to_remove = current_size_bytes - target_size_bytes

            # Get entries sorted by accessed_at (oldest first) with actual_path
            entries_to_delete = session.execute(
                select(CE.cache_key, CE.file_size, CE.actual_path).order_by(
                    CE.accessed_at.asc()
                )
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
                    delete(self._CacheEntry).where(
                        self._CacheEntry.cache_key.in_(removed_keys)
                    )
                )
                session.commit()

            return {"count": len(removed_entries), "removed_entries": removed_entries}

    def clear_all(self) -> int:
        """Remove all cache entries and return count removed."""
        with self._lock, self.SessionLocal() as session:
            # Count existing entries
            result = session.execute(select(func.count(self._CacheEntry.cache_key)))
            entry_count = result.scalar() or 0

            # Delete all entries
            session.execute(delete(self._CacheEntry))

            # Reset stats
            stats = session.execute(
                select(self._CacheStats).where(self._CacheStats.id == 1)
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

    def close(self):
        """Close all database connections and clean up resources."""
        if hasattr(self, "engine") and self.engine:
            # Close all connections in the pool
            self.engine.dispose()
            # On Windows, we need to be more aggressive
            import gc

            gc.collect()  # Force garbage collection to release file handles
            logger.debug("SQLite engine disposed and connections closed")

    def __del__(self):
        """Ensure connections are closed when the backend is garbage collected."""
        try:
            self.close()
        except Exception:  # intentionally broad — cleanup must not raise
            # Suppress errors during interpreter shutdown
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure connections are closed."""
        self.close()
        return False
