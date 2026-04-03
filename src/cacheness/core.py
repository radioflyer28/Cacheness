"""
Simplified Unified Cache with Handler-based Architecture
=======================================================

This module provides a cleaner, more maintainable cache system using the Strategy pattern.
The main UnifiedCache class is now focused on coordination and delegates format-specific
operations to specialized handlers.
"""

import inspect
import threading
import logging
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Callable, Tuple

from .config import CacheConfig, _DEFAULT_TTL, create_cache_config
from .entry_list import EntryList
from .handlers import HandlerRegistry
from .interfaces import HandlerResult
from .metadata import DEFAULT_NAMESPACE
from .serialization import create_unified_cache_key
from .size_utils import format_size, resolve_ttl
from .storage.paths import resolve_actual_path
from ._verification_mixin import VerificationMixin
from ._stats_mixin import StatsMixin
from ._custom_metadata_mixin import CustomMetadataMixin
from ._storage_mode_mixin import StorageModeMixin

if TYPE_CHECKING:
    from .interfaces import RotationResult

logger = logging.getLogger(__name__)


def _normalize_function_args(
    func: Callable, args: Tuple, kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Normalize function call arguments to consistent parameter mapping.

    This ensures that func(1, 2, 10), func(a=1, b=2, c=10), and func(1, b=2, c=10)
    all produce the same cache key when they represent the same logical call.

    Args:
        func: The function being called
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Normalized parameter dictionary
    """
    try:
        # Use inspect.signature to normalize calling conventions
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.arguments
    except (ValueError, TypeError):  # intentionally broad
        # Fallback: convert to consistent dict format if signature inspection fails
        param_dict = {}

        # Add positional args with generic names
        for i, arg in enumerate(args):
            param_dict[f"__arg_{i}"] = arg

        # Add keyword args
        param_dict.update(kwargs)

        return param_dict


class _PutCleanup:
    """Tracks resources written during ``put()`` so they can be rolled back.

    Usage::

        cleanup = _PutCleanup()
        try:
            # ... write blob ...
            cleanup.blob_path = Path(actual_path)
            # ... upload to S3 ...
            cleanup.set_remote(blob_backend, blob_uri)
            # ... write metadata ...
            cleanup.commit()   # disarm — nothing will be rolled back
        except Exception:  # intentionally broad — docstring example
            cleanup.rollback()
            raise

    On ``rollback()``, every tracked resource is deleted (local file **and**
    remote object).  ``commit()`` disarms the cleanup so ``rollback()`` is a
    no-op.
    """

    __slots__ = ("_committed", "blob_path", "_blob_backend", "_blob_uri")

    def __init__(self) -> None:
        self._committed = False
        self.blob_path: Optional[Path] = None
        self._blob_backend: Any = None
        self._blob_uri: Optional[str] = None

    def set_remote(self, blob_backend: Any, blob_uri: str) -> None:
        """Register a remote blob (e.g. S3 object) for rollback."""
        self._blob_backend = blob_backend
        self._blob_uri = blob_uri

    def commit(self) -> None:
        """Disarm — a subsequent ``rollback()`` will be a no-op."""
        self._committed = True

    def rollback(self) -> None:
        """Delete every tracked resource.  Safe to call multiple times."""
        if self._committed:
            return

        # Clean up local blob file
        if self.blob_path is not None:
            try:
                if self.blob_path.exists():
                    self.blob_path.unlink()
                    logger.debug(f"Cleaned up orphaned local blob: {self.blob_path}")
            except OSError:
                pass

        # Clean up remote blob (e.g. S3 object)
        if self._blob_backend is not None and self._blob_uri is not None:
            try:
                self._blob_backend.delete_blob(self._blob_uri)
                logger.debug(f"Cleaned up orphaned remote blob: {self._blob_uri}")
            except Exception:  # intentionally broad — cleanup must not raise
                logger.warning(
                    f"Failed to clean up orphaned remote blob: {self._blob_uri}"
                )

        self._committed = True  # prevent double-rollback


class UnifiedCache(
    VerificationMixin,
    StatsMixin,
    CustomMetadataMixin,
    StorageModeMixin,
):
    """
    Simplified unified caching system using the Strategy pattern.

    This class focuses on coordination and delegates format-specific operations
    to specialized handlers for better maintainability and extensibility.
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        metadata_backend=None,
    ):
        """
        Initialize the unified cache system.

        Args:
            config: CacheConfig object (uses defaults if None)
            metadata_backend: Optional metadata backend instance (if None, creates based on config)
        """
        # Use provided config or create default
        self.config = config or CacheConfig()

        # Namespace is immutable after init
        self.namespace = self.config.namespace

        self.cache_dir = Path(self.config.storage.cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Thread safety — RLock allows re-entrant calls
        # (e.g. delete_where → invalidate, touch_batch → touch)
        self._lock = threading.RLock()

        # Initialize handler registry with config
        self.handlers = HandlerRegistry(self.config)

        # Initialize metadata backend
        self._init_metadata_backend(metadata_backend)

        # Initialize custom metadata support
        self._init_custom_metadata_support()

        # Initialize entry signer for metadata integrity
        self._init_entry_signer()

        # Sign / verify the current namespace registry row
        self._sign_current_namespace()

        # Initialize internal BlobStore for storage delegation
        # Shares metadata_backend, handlers, lock, signer, and config
        self._init_blob_store()

        # Initialize write intent journal for crash recovery
        from .write_intent import WriteIntentJournal

        self._write_journal = WriteIntentJournal(
            self.cache_dir,
            self.config.storage.stale_intent_threshold_seconds,
        )

        # Clean up expired entries and stale write intents on initialization
        if self.config.storage.cleanup_on_init:
            self._cleanup_expired()
            self._cleanup_stale_intents()

        logger.info(
            f"✅ Unified cache initialized: {self.cache_dir} (backend: {self.actual_backend})"
        )

    def _init_metadata_backend(self, metadata_backend):
        """Initialize the metadata backend based on config or provided instance."""
        from .metadata import create_metadata_backend, SQLALCHEMY_AVAILABLE

        # User-provided backend takes priority
        if metadata_backend is not None:
            self.metadata_backend = metadata_backend
            self.actual_backend = "custom"
            return

        requested = self.config.metadata.metadata_backend

        # Backends that require SQLAlchemy
        _SQLALCHEMY_BACKENDS = {"sqlite", "sqlite_memory", "postgresql"}

        if requested in _SQLALCHEMY_BACKENDS and not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                f"SQLAlchemy is required for {requested} backend but is not available. "
                f"Install with: uv add sqlalchemy"
            )

        # Build kwargs for the factory based on backend type
        kwargs = self._build_backend_kwargs(requested)

        if requested == "auto":
            self._init_auto_backend(create_metadata_backend, SQLALCHEMY_AVAILABLE)
        else:
            self.metadata_backend = create_metadata_backend(requested, **kwargs)
            self.actual_backend = requested
            if requested == "sqlite_memory":
                logger.info("⚡ Using in-memory SQLite backend (no persistence)")

    def _build_backend_kwargs(self, requested: str) -> Dict[str, Any]:
        """Build keyword arguments for ``create_metadata_backend``.

        Centralises the per-backend kwargs construction that was previously
        duplicated across five ``if/elif`` branches.
        """
        base: Dict[str, Any] = {
            "config": self.config.metadata,
            "namespace": self.namespace,
        }

        if requested == "json":
            base["metadata_file"] = self.cache_dir / "cache_metadata.json"

        elif requested in ("sqlite", "auto"):
            base["db_file"] = str(self.cache_dir / self.config.metadata.sqlite_db_file)

        elif requested == "sqlite_memory":
            pass  # no extra kwargs needed

        elif requested == "postgresql":
            opts = self.config.metadata.metadata_backend_options or {}
            connection_url = opts.get("connection_url")
            if not connection_url:
                raise ValueError(
                    "PostgreSQL backend requires 'connection_url' in "
                    "metadata_backend_options"
                )
            base.update(
                {
                    "connection_url": connection_url,
                    "pool_size": opts.get("pool_size", 10),
                    "max_overflow": opts.get("max_overflow", 20),
                    "pool_pre_ping": opts.get("pool_pre_ping", True),
                    "pool_recycle": opts.get("pool_recycle", 3600),
                    "echo": opts.get("echo", False),
                    "table_prefix": opts.get("table_prefix", ""),
                }
            )

        return base

    def _init_auto_backend(self, create_metadata_backend, sqlalchemy_available: bool):
        """Auto-select the best available metadata backend."""
        if sqlalchemy_available:
            try:
                self.metadata_backend = create_metadata_backend(
                    "sqlite",
                    db_file=str(self.cache_dir / self.config.metadata.sqlite_db_file),
                    config=self.config.metadata,
                    namespace=self.namespace,
                )
                self.actual_backend = "sqlite"
                logger.info(
                    "🗄️  Using SQLite backend (auto-selected for better performance)"
                )
                return
            except (ImportError, OSError, ValueError) as e:
                logger.warning(f"SQLite backend failed, falling back to JSON: {e}")
        else:
            logger.info("📝 SQLModel not available, using JSON backend")

        self.metadata_backend = create_metadata_backend(
            "json",
            metadata_file=self.cache_dir / "cache_metadata.json",
            config=self.config.metadata,
            namespace=self.namespace,
        )
        self.actual_backend = "json"

    def _init_custom_metadata_support(self):
        """Initialize custom metadata support if SQLite or PostgreSQL backend is available."""
        try:
            from .custom_metadata import is_custom_metadata_available

            if is_custom_metadata_available() and self.actual_backend in (
                "sqlite",
                "postgresql",
            ):
                self._custom_metadata_enabled = True
                logger.info("🏷️  Custom metadata support enabled")
            else:
                self._custom_metadata_enabled = False
        except ImportError:
            self._custom_metadata_enabled = False

    def _init_entry_signer(self):
        """Initialize cache entry signer for metadata integrity protection."""
        try:
            if self.config.security.enable_entry_signing:
                from .security import create_cache_signer

                self.signer = create_cache_signer(
                    cache_dir=self.cache_dir,
                    key_file=self.config.security.signing_key_file,
                    use_in_memory_key=self.config.security.use_in_memory_key,
                    key_fallback_policy=self.config.security.key_fallback_policy,
                    namespace_id=self.namespace,
                    use_hkdf_derivation=self.config.security.use_hkdf_derivation,
                )

                info = self.signer.get_field_info()
                logger.info(
                    f"🔒 Entry signing enabled (v{info['signature_version']}, "
                    f"{len(info['signed_fields'])} fields)"
                )
            else:
                self.signer = None
                logger.debug("Entry signing disabled")
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to initialize entry signer: {e}")
            self.signer = None

    def _sign_current_namespace(self):
        """Sign or verify the current namespace registry row.

        Called once during ``__init__`` after the signer is available.
        - If the namespace has no signature yet, compute one and store it.
        - If a signature already exists, verify it and warn on mismatch
          (non-fatal — a key rotation or schema change may invalidate it).
        """
        if not self.signer:
            return

        try:
            ns_info = self.metadata_backend.get_namespace(self.namespace)
            if ns_info is None:
                return

            ns_data = {
                "namespace_id": ns_info.namespace_id,
                "display_name": ns_info.display_name,
                "created_at": ns_info.created_at,
            }

            if ns_info.signature is None:
                # First time — sign and persist
                sig = self.signer.sign_namespace(ns_data)
                self.metadata_backend.set_namespace_signature(ns_info.namespace_id, sig)
                logger.info(f"🔏 Signed namespace {ns_info.namespace_id!r}")
            else:
                # Verify existing signature
                if not self.signer.verify_namespace(ns_data, ns_info.signature):
                    logger.warning(
                        f"⚠️  Namespace {ns_info.namespace_id!r} signature "
                        f"verification failed (key rotation or tampering?)"
                    )
                else:
                    logger.debug(
                        f"Namespace {ns_info.namespace_id!r} signature verified"
                    )
        except (ValueError, TypeError, OSError) as e:
            logger.warning(f"Namespace signing/verification failed: {e}")

    def rotate_key(self, new_key_file: str | Path) -> "RotationResult":
        """Rotate the signing key and re-sign all entries and namespace.

        Loads a new 32-byte signing key from *new_key_file*, replaces the
        current key file, creates a fresh :class:`CacheEntrySigner` with
        HKDF derivation for the current namespace, then iterates every
        entry to re-sign it as v3.  The namespace registry row is also
        re-signed.

        The operation is **best-effort**: if an individual entry fails to
        re-sign (e.g. corrupt metadata), it is skipped and recorded in
        :attr:`RotationResult.failures`.  The lock is held for the entire
        duration so concurrent ``put()``/``get()`` calls block until
        rotation finishes.

        Args:
            new_key_file: Path to a file containing exactly 32 random bytes
                          (the new signing key).

        Returns:
            :class:`RotationResult` with counts of re-signed / failed /
            skipped entries.

        Raises:
            CacheSecurityError: If signing is not enabled, *new_key_file*
                does not exist, or the key is not exactly 32 bytes.
        """
        from .error_handling import CacheSecurityError
        from .interfaces import RotationResult
        from .security import create_cache_signer

        if self.signer is None:
            raise CacheSecurityError("Entry signing is not enabled — cannot rotate key")

        new_key_path = Path(new_key_file)
        if not new_key_path.is_file():
            raise CacheSecurityError(f"Key file does not exist: {new_key_path}")
        new_key_bytes = new_key_path.read_bytes()
        if len(new_key_bytes) != 32:
            raise CacheSecurityError(
                f"Invalid key length ({len(new_key_bytes)} bytes) — expected 32"
            )

        with self._lock:
            # Overwrite the current key file with the new key
            dest = Path(self.signer.key_file_path)
            dest.write_bytes(new_key_bytes)

            # Create a new signer using the replaced key file
            new_signer = create_cache_signer(
                cache_dir=self.cache_dir,
                key_file=dest.name,
                use_in_memory_key=False,
                key_fallback_policy=self.config.security.key_fallback_policy,
                namespace_id=self.namespace,
                use_hkdf_derivation=self.config.security.use_hkdf_derivation,
            )

            result = RotationResult()
            entries = self.metadata_backend.iter_entry_summaries()
            result.total = len(entries)

            for entry in entries:
                cache_key = entry["cache_key"]
                try:
                    full_entry = self.metadata_backend.get_entry(cache_key)
                    if full_entry is None:
                        result.skipped += 1
                        continue

                    metadata = full_entry.get("metadata", {})
                    signable = self._extract_signable_fields(
                        cache_key, full_entry, metadata
                    )
                    new_sig = new_signer.sign_entry(signable)
                    metadata["entry_signature"] = new_sig
                    full_entry["metadata"] = metadata
                    self.metadata_backend.put_entry(cache_key, full_entry)
                    result.re_signed += 1
                except Exception as e:  # intentionally broad — best-effort re-sign
                    result.failed += 1
                    result.failures.append({"cache_key": cache_key, "error": str(e)})
                    logger.warning(f"Failed to re-sign entry {cache_key}: {e}")

            # Re-sign namespace (D-12)
            try:
                ns_info = self.metadata_backend.get_namespace(self.namespace)
                if ns_info is not None:
                    ns_data = {
                        "namespace_id": ns_info.namespace_id,
                        "display_name": ns_info.display_name,
                        "created_at": ns_info.created_at,
                    }
                    ns_sig = new_signer.sign_namespace(ns_data)
                    self.metadata_backend.set_namespace_signature(
                        ns_info.namespace_id, ns_sig
                    )
            except (
                Exception
            ) as e:  # intentionally broad — namespace re-sign failure is non-fatal
                logger.warning(f"Failed to re-sign namespace: {e}")

            # Replace the signer instance (all subsequent ops use new key)
            self.signer = new_signer

            # Also update the shared BlobStore signer if it exists
            if hasattr(self, "_blob_store") and self._blob_store is not None:
                self._blob_store.signer = new_signer

            logger.info(
                f"Key rotation complete: {result.re_signed}/{result.total} "
                f"entries re-signed, {result.failed} failed, "
                f"{result.skipped} skipped"
            )

        return result

    def _init_blob_store(self):
        """Initialize internal BlobStore for storage delegation.

        The BlobStore shares the same metadata_backend, handler registry,
        lock, signer, and config as the UnifiedCache.  This avoids resource
        duplication and ensures consistent behaviour.

        Storage operations (file I/O, handler dispatch, integrity verification)
        are delegated to BlobStore while UnifiedCache retains cache-specific
        concerns (TTL, eviction, stats, decorators).
        """
        from .storage import BlobStore

        self._blob_store = BlobStore(
            cache_dir=self.cache_dir,
            backend=self.metadata_backend,  # shared metadata backend
            config=self.config,
            namespace=self.namespace,
        )
        # Share resources — avoids duplication and double-initialization
        self._blob_store._lock = self._lock  # same reentrant lock
        self._blob_store.handlers = self.handlers  # same handler registry
        self._blob_store.signer = self.signer  # same signer (may be None)

    def query_meta(self, **filters):
        """
        Query cache entries by their stored metadata key-value pairs.

        Works with **all** backends.  When the SQLite backend is active and
        ``store_full_metadata=True``, a fast SQL path using ``JSON_EXTRACT``
        is used.  When the PostgreSQL backend is active, a fast JSONB path
        using the ``@>`` containment operator (with GIN index) is used.
        For every other backend (JSON, custom) a Python-side fallback
        iterates stored entries and matches against the ``metadata_dict``
        field.

        Args:
            **filters: Key-value pairs to filter cache entries.
                       An entry matches when **all** pairs are present in
                       its ``metadata_dict`` with equal values (numeric
                       comparison for int/float, string equality otherwise).

        Returns:
            EntryList of dicts (one per matching entry) with keys
            ``cache_key``, ``description``, ``data_type``, ``created_at``,
            ``accessed_at``, ``file_size``, ``metadata_dict``.
            Returns an empty EntryList when no entries match.
            Returns ``None`` only on unexpected errors.

        Example:
            config = CacheConfig(store_full_metadata=True)
            cache = Cacheness(config=config)

            cache.put(model, experiment="exp_001", model_type="xgboost", accuracy=0.95)
            cache.put(data, experiment="exp_002", model_type="cnn", accuracy=0.88)

            xgb_experiments = cache.query_meta(model_type="xgboost")
            specific_exp = cache.query_meta(experiment="exp_001")
        """
        if not self.config.metadata.store_full_metadata:
            logger.warning(
                "query_meta() requires store_full_metadata=True in cache configuration"
            )
            return None

        # ── SQLite fast path: JSON_EXTRACT ──────────────────────────
        if self.actual_backend == "sqlite" and hasattr(
            self.metadata_backend, "SessionLocal"
        ):
            return self._query_meta_sqlite(**filters)

        # ── PostgreSQL fast path: JSONB @> containment ───────────
        if self.actual_backend == "postgresql" and hasattr(
            self.metadata_backend, "SessionLocal"
        ):
            return self._query_meta_postgres(**filters)

        # ── Generic fallback: Python-side filtering ─────────────────
        return self._query_meta_generic(**filters)

    # ── Private helpers ─────────────────────────────────────────────

    def _query_meta_generic(self, **filters) -> EntryList | None:
        """Python-side ``query_meta`` that works with any backend."""
        try:
            from .json_utils import loads as json_loads
        except ImportError:
            import json

            json_loads = json.loads

        try:
            summaries = self.metadata_backend.iter_entry_summaries()
            entries: EntryList = EntryList()

            for summary in summaries:
                # metadata_dict may be a JSON string or already a dict
                raw = summary.get("metadata_dict")
                if raw is None:
                    continue

                if isinstance(raw, str):
                    try:
                        meta = json_loads(raw)
                    except (
                        ValueError,
                        KeyError,
                    ):  # intentionally broad — malformed JSON
                        continue
                elif isinstance(raw, dict):
                    meta = raw
                else:
                    continue

                # Check all filters match
                if filters and not all(
                    self._meta_value_matches(meta.get(k), v) for k, v in filters.items()
                ):
                    continue

                entries.append(
                    {
                        "cache_key": summary.get("cache_key"),
                        "description": summary.get("description", ""),
                        "data_type": summary.get("data_type", "unknown"),
                        "created_at": self._fmt_timestamp(summary.get("created_at")),
                        "accessed_at": self._fmt_timestamp(summary.get("accessed_at")),
                        "file_size": summary.get("file_size", 0),
                        "metadata_dict": meta,
                    }
                )

            return entries

        except Exception as e:  # intentionally broad — backend query may fail any way
            logger.error(f"Failed to query metadata (generic): {e}")
            return None

    def _query_meta_postgres(self, **filters) -> EntryList | None:
        """PostgreSQL fast path using JSONB ``@>`` containment for ``query_meta``.

        Leverages the GIN ``jsonb_path_ops`` index on ``metadata_dict``
        for sub-millisecond filtered lookups instead of pulling all rows
        into Python.
        """
        try:
            from sqlalchemy import text

            table = self.metadata_backend._entries_table

            with self.metadata_backend.SessionLocal() as session:
                if filters:
                    # JSONB @> operator — leverages GIN index
                    from .json_utils import dumps as json_dumps

                    filter_json = json_dumps(filters)
                    query = (
                        f"SELECT cache_key, description, data_type, "
                        f"       created_at, accessed_at, file_size, "
                        f"       metadata_dict "
                        f'FROM "{table}" '
                        f"WHERE metadata_dict @> CAST(:filter_json AS jsonb) "
                        f"ORDER BY created_at DESC"
                    )
                    result = session.execute(text(query), {"filter_json": filter_json})
                else:
                    query = (
                        f"SELECT cache_key, description, data_type, "
                        f"       created_at, accessed_at, file_size, "
                        f"       metadata_dict "
                        f'FROM "{table}" '
                        f"WHERE metadata_dict IS NOT NULL "
                        f"ORDER BY created_at DESC"
                    )
                    result = session.execute(text(query))

                entries = EntryList()
                for row in result:
                    # JSONB may return dict (psycopg3) or str (psycopg2)
                    meta_raw = row.metadata_dict
                    if isinstance(meta_raw, str):
                        try:
                            from .json_utils import loads as json_loads

                            meta_raw = json_loads(meta_raw)
                        except (
                            ValueError,
                            KeyError,
                        ):  # intentionally broad — malformed JSONB
                            meta_raw = {}
                    elif not isinstance(meta_raw, dict):
                        meta_raw = {}

                    entries.append(
                        {
                            "cache_key": row.cache_key,
                            "description": row.description or "",
                            "data_type": row.data_type or "unknown",
                            "created_at": self._fmt_timestamp(row.created_at),
                            "accessed_at": self._fmt_timestamp(row.accessed_at),
                            "file_size": row.file_size or 0,
                            "metadata_dict": meta_raw,
                        }
                    )

                return entries

        except Exception as e:  # intentionally broad — backend query may fail any way
            logger.error(f"Failed to query metadata (postgres): {e}")
            return None

    def _query_meta_sqlite(self, **filters) -> EntryList | None:
        """SQLite fast path using JSON_EXTRACT for ``query_meta``."""
        try:
            from sqlalchemy import text

            # Use the namespace-specific table name (EntityName pattern)
            table = self.metadata_backend._entries_table

            with self.metadata_backend.SessionLocal() as session:
                where_conditions: list[str] = []
                params: dict = {}

                for key, value in filters.items():
                    param_name = f"param_{len(params)}"
                    if isinstance(value, (int, float)):
                        where_conditions.append(
                            f"CAST(JSON_EXTRACT(metadata_dict, '$.{key}') AS REAL) = :{param_name}"
                        )
                    else:
                        where_conditions.append(
                            f"JSON_EXTRACT(metadata_dict, '$.{key}') = :{param_name}"
                        )
                    params[param_name] = value

                if where_conditions:
                    where_clause = " AND ".join(where_conditions)
                    query = f"""
                        SELECT cache_key, description, data_type, created_at, accessed_at,
                               file_size, metadata_dict
                        FROM {table}
                        WHERE metadata_dict IS NOT NULL AND ({where_clause})
                        ORDER BY created_at DESC
                    """
                else:
                    query = f"""
                        SELECT cache_key, description, data_type, created_at, accessed_at,
                               file_size, metadata_dict
                        FROM {table}
                        WHERE metadata_dict IS NOT NULL
                        ORDER BY created_at DESC
                    """

                result = session.execute(text(query), params)

                entries = EntryList()
                for row in result:
                    entry = {
                        "cache_key": row.cache_key,
                        "description": row.description,
                        "data_type": row.data_type,
                        "created_at": self._fmt_timestamp(row.created_at),
                        "accessed_at": self._fmt_timestamp(row.accessed_at),
                        "file_size": row.file_size,
                    }

                    if row.metadata_dict:
                        try:
                            from .json_utils import loads as json_loads

                            entry["metadata_dict"] = json_loads(row.metadata_dict)
                        except (
                            ValueError,
                            KeyError,
                        ):  # intentionally broad — malformed JSON
                            entry["metadata_dict"] = {}

                    entries.append(entry)

                return entries

        except Exception as e:  # intentionally broad — backend query may fail any way
            logger.error(f"Failed to query metadata (sqlite): {e}")
            return None

    @staticmethod
    def _meta_value_matches(stored, expected) -> bool:
        """Compare a stored metadata value against an expected filter value.

        Handles type coercion: stored JSON numbers may round-trip as int or
        float, so numeric comparisons cast both sides.
        """
        if stored is None:
            return False
        if isinstance(expected, (int, float)):
            try:
                return float(stored) == float(expected)
            except (TypeError, ValueError):
                return False
        return str(stored) == str(expected)

    @staticmethod
    def _fmt_timestamp(ts) -> str:
        """Format a timestamp value to ISO string."""
        if ts is None:
            return ""
        if hasattr(ts, "isoformat"):
            return ts.isoformat()
        return str(ts)

    def _create_cache_key(self, params: Dict) -> str:
        """
        Create cache key using unified serialization approach.

        Uses the unified serialization system that:
        - Handles Path objects with content hashing based on config
        - Leverages __hash__() when available for hashable objects
        - Provides consistent behavior with decorators
        - Falls back gracefully for complex objects

        Args:
            params: Dictionary of parameters to hash

        Returns:
            16-character hex string cache key
        """
        # Use unified cache key generation with config
        # Path objects will be handled by the serialization system
        return create_unified_cache_key(params, self.config)

    @staticmethod
    def _resolve_hash_key_alias(
        cache_key: Optional[str], hash_key: Optional[str]
    ) -> Optional[str]:
        """Normalize hash_key alias to cache_key.

        ``hash_key`` is a storage-oriented alias for ``cache_key``.
        Both refer to the same underlying key. If only ``hash_key`` is
        provided it is returned as ``cache_key``; if both are supplied
        a ``ValueError`` is raised.
        """
        if hash_key is not None and cache_key is not None:
            raise ValueError(
                "Cannot specify both 'cache_key' and 'hash_key'. "
                "They are aliases — use one or the other."
            )
        return cache_key if hash_key is None else hash_key

    @staticmethod
    def content_key(data: Any) -> str:
        """Compute a content-addressable key from data using SHA-256.

        This produces a deterministic 16-character hex key based on the
        serialised content of *data*.  Storing the same data twice will
        always yield the same key, enabling deduplication.

        Args:
            data: Any pickleable object.

        Returns:
            16-character hex string suitable for ``cache_key`` / ``hash_key``.

        Example:
            key = UnifiedCache.content_key(my_dataframe)
            cache.put(my_dataframe, hash_key=key)
        """
        import hashlib
        import pickle

        try:
            serialized = pickle.dumps(data)
        except (TypeError, pickle.PicklingError):
            serialized = repr(data).encode()
        return hashlib.sha256(serialized).hexdigest()[:16]

    def _resolve_cache_key(
        self,
        cache_key: Optional[str],
        on: Optional[Dict],
        kwargs: Dict,
    ) -> str:
        """
        Resolve cache key from three sources with priority: cache_key > on > kwargs.

        This centralizes the key resolution logic used by all public methods.
        The ``on`` parameter prevents namespace collisions between user key
        parameters and cache control parameters (description, etc.).

        Args:
            cache_key: Explicit pre-computed cache key (highest priority)
            on: Dictionary of key parameters (medium priority, no namespace collision)
            kwargs: Legacy keyword arguments for key derivation (lowest priority)

        Returns:
            16-character hex string cache key

        Raises:
            ValueError: If ``on`` and ``**kwargs`` are both provided (ambiguous)
        """
        if cache_key is not None:
            return cache_key
        if on is not None:
            if kwargs:
                raise ValueError(
                    "Cannot use both 'on' and **kwargs for key derivation. "
                    "Use 'on' for explicit key params or **kwargs for legacy "
                    "compatibility, not both."
                )
            return self._create_cache_key(on)
        return self._create_cache_key(kwargs)

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get base cache file path (without extension).

        All namespaces (including default) store blob files under
        ``cache_dir/{namespace}/`` so that blob files mirror the
        namespace isolation provided by the metadata backend.
        """
        filename_base = cache_key

        base = self.cache_dir / self.namespace
        base.mkdir(parents=True, exist_ok=True)
        return base / filename_base

    def _resolve_actual_path(self, actual_path_str: str) -> Path | str:
        """Resolve a stored ``actual_path`` to a usable path.

        Returns ``Path`` for filesystem paths, ``str`` for URIs.
        Delegates to :func:`cacheness.storage.paths.resolve_actual_path`.
        """
        return resolve_actual_path(actual_path_str, self.cache_dir)

    def _is_expired(self, cache_key: str, ttl_seconds=_DEFAULT_TTL) -> bool:
        """Check if cache entry is expired.

        Args:
            cache_key: The cache key to check
            ttl_seconds: TTL in seconds (numeric). None means never expire.
                Use _DEFAULT_TTL sentinel to fall back to config default.
        """
        entry = self.metadata_backend.get_entry(cache_key)
        if not entry:
            return True

        # Handle infinite TTL: if ttl_seconds is explicitly None, never expire
        if ttl_seconds is None:
            return False  # Never expires
        elif ttl_seconds is _DEFAULT_TTL:
            ttl_seconds = self.config.metadata.default_ttl_seconds
            if ttl_seconds is None:
                return False  # Config says never expire

        # Type guard to ensure ttl is numeric
        if isinstance(ttl_seconds, str):
            raise TypeError(
                f'ttl_seconds must be numeric, got string "{ttl_seconds}". '
                f'Use ttl="{ttl_seconds}" for duration strings.'
            )
        assert isinstance(ttl_seconds, (int, float)), (
            f"TTL must be numeric, got {type(ttl_seconds)}"
        )

        creation_time_str = entry["created_at"]

        # Handle timezone-aware datetime strings
        if isinstance(creation_time_str, str):
            creation_time = datetime.fromisoformat(creation_time_str)
        else:
            creation_time = creation_time_str

        # Ensure both datetimes are timezone-aware
        if creation_time.tzinfo is None:
            creation_time = creation_time.replace(tzinfo=timezone.utc)

        expiry_time = creation_time + timedelta(seconds=ttl_seconds)
        current_time = datetime.now(timezone.utc)

        return current_time > expiry_time

    # ── Lifecycle hook helpers ────────────────────────────────────────

    def _invoke_hook(self, hook_name: str, *args: object) -> None:
        """Safely invoke a HooksConfig callback (never raises)."""
        hook = getattr(self.config.hooks, hook_name, None)
        if hook is not None:
            try:
                hook(*args)
            except Exception as exc:  # intentionally broad — hooks must not crash the caller  # noqa: BLE001
                logger.warning(
                    f"Hook {hook_name} raised an exception (swallowed): {exc}"
                )

    def _cleanup_expired(self):
        """Remove expired cache entries."""
        ttl_seconds = self.config.metadata.default_ttl_seconds
        if ttl_seconds is None:
            return  # No TTL configured — nothing to expire
        removed_count = self.metadata_backend.cleanup_expired(ttl_seconds)

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache entries")

    def _cleanup_stale_intents(self):
        """Remove orphaned blobs from stale write intents (crash recovery)."""
        cleaned = self._write_journal.cleanup_stale_intents()
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} stale write intents")

    # ── Shared put helpers ────────────────────────────────────────────
    # Extracted from _storage_mode_put() and put() to eliminate duplicated
    # metadata construction, signing, and stale-blob cleanup logic.

    def _try_inline_blob(
        self,
        result: HandlerResult,
        file_hash: Optional[str],
        cleanup: "_PutCleanup",
    ) -> Optional[Dict[str, Any]]:
        """Try to inline a small blob into the metadata row.

        If the blob is small enough (≤ ``max_inline_size`` from config),
        reads the file bytes, computes an xxhash from the bytes, and
        returns a dict with ``blob_data``, ``is_inline=1``, and
        ``file_hash``.  The blob file is deleted and the cleanup guard
        is disarmed so rollback won't try to delete it again.

        Returns ``None`` when inlining is disabled or the blob is too large.
        """
        max_inline = self.config.blob.max_inline_size
        if max_inline <= 0:
            return None
        if result.file_size > max_inline:
            return None

        # Resolve the blob path written by _write_blob
        actual_path_str = result.actual_path
        if "://" in actual_path_str:
            # Remote blobs (S3, etc.) are not inlined
            return None

        blob_path = self._resolve_actual_path(actual_path_str)
        if not isinstance(blob_path, Path) or not blob_path.exists():
            return None

        blob_bytes = blob_path.read_bytes()

        # Compute hash from the raw bytes (matches file-based hashing)
        computed_hash = file_hash
        if computed_hash is None and self.config.metadata.verify_cache_integrity:
            import xxhash

            computed_hash = xxhash.xxh3_64(blob_bytes).hexdigest()

        # Preserve the original file suffix so _read_inline_blob can
        # recreate a temp file that the handler recognises (handlers
        # derive the expected path from the suffix, e.g. .pkl.zstd).
        # Extract all suffixes after the hash portion of the filename.
        inline_ext = "".join(blob_path.suffixes) or ".bin"

        # Delete the blob file — data now lives in metadata
        try:
            blob_path.unlink()
        except OSError as exc:
            logger.warning(f"Failed to remove inlined blob file {blob_path}: {exc}")

        # Disarm the cleanup guard so rollback doesn't try to delete
        cleanup.blob_path = None

        return {
            "blob_data": blob_bytes,
            "is_inline": 1,
            "file_hash": computed_hash,
            "inline_ext": inline_ext,
        }

    def _try_direct_inline(
        self,
        data: Any,
        handler: Any,
    ) -> Optional[Dict[str, Any]]:
        """Try zero-disk in-memory serialization for inline blob storage.

        Invokes ``handler.put_bytes()`` to serialize *data* entirely in
        memory.  If the handler supports it and the serialized blob fits
        within ``max_inline_size``, returns a dict ready for embedding in
        the metadata row — **no file is ever written to disk**.

        Returns ``None`` when:
        * Inlining is disabled (``max_inline_size ≤ 0``).
        * The handler raises :class:`NotImplementedError`.
        * The serialized blob exceeds ``max_inline_size``.

        The returned dict contains:

        * ``blob_data`` – the raw bytes to store in the metadata row.
        * ``file_hash`` – xxhash digest (or ``None`` when integrity
          checking is disabled).
        * ``inline_ext`` – file extension hint for ``get_bytes``/fallback.
        * ``result`` – :class:`HandlerResult` with serialization metadata
          (``storage_format``, ``compression_codec``, etc.).
        * ``handler`` – the handler instance (for ``data_type``).
        """
        max_inline = self.config.blob.max_inline_size
        if max_inline <= 0:
            return None

        try:
            blob_bytes, result = handler.put_bytes(data, self.config)
        except (NotImplementedError, Exception) as exc:
            # NotImplementedError → handler doesn't support in-memory path.
            # Any other exception → safer to fall back to disk path.
            if not isinstance(exc, NotImplementedError):
                logger.debug(
                    "put_bytes failed for %s, falling back to disk: %s",
                    handler.data_type,
                    exc,
                )
            return None

        if len(blob_bytes) > max_inline:
            return None

        # Compute hash from raw bytes
        computed_hash: Optional[str] = None
        if self.config.metadata.verify_cache_integrity:
            import xxhash

            computed_hash = xxhash.xxh3_64(blob_bytes).hexdigest()

        inline_ext = handler.get_file_extension(self.config)

        return {
            "blob_data": blob_bytes,
            "file_hash": computed_hash,
            "inline_ext": inline_ext,
            "result": result,
            "handler": handler,
        }

    def _read_inline_blob(
        self,
        entry: Dict[str, Any],
        data_type: str,
        metadata: Dict[str, Any],
    ) -> Any:
        """Deserialize an inline blob without touching the blob backend.

        Tries the handler's ``get_bytes()`` first for zero-disk
        deserialization.  Falls back to writing the raw bytes to a
        temporary file and delegating to the handler's ``get()`` method.
        """
        blob_bytes: bytes = entry["blob_data"]

        # Fast path — zero-disk deserialization via get_bytes()
        try:
            handler = self._blob_store.handlers.get_handler_by_type(data_type)
            return handler.get_bytes(blob_bytes, metadata)
        except NotImplementedError:
            pass  # Fall through to temp-file path
        except Exception as exc:  # intentionally broad — handler may raise anything
            logger.debug(
                "get_bytes failed for %s, falling back to temp file: %s",
                data_type,
                exc,
            )

        # Slow path — write to temp file, delegate to handler.get()
        import tempfile

        suffix = metadata.get("inline_ext", ".bin")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(blob_bytes)
            tmp_path = Path(tmp.name)

        try:
            return self._blob_store._read_blob(tmp_path, data_type, metadata)
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    def _build_metadata_dict(
        self, result: HandlerResult, file_hash: Optional[str]
    ) -> Dict[str, Any]:
        """Build the metadata dict from a :class:`HandlerResult`.

        Merges handler ``extra`` fields with top-level columns
        (``actual_path``, ``file_hash``, ``storage_format``, etc.)
        so the backend can extract them to dedicated columns.
        """
        metadata_dict: Dict[str, Any] = {
            **result.extra,
            "actual_path": result.actual_path,
            "file_hash": file_hash,
            "storage_format": result.storage_format,
        }
        if result.serializer:
            metadata_dict["serializer"] = result.serializer
        if result.compression_codec:
            metadata_dict["compression_codec"] = result.compression_codec
        if result.object_type:
            metadata_dict["object_type"] = result.object_type
        return metadata_dict

    def _cleanup_stale_blob(
        self,
        cache_key: str,
        old_blob_path: Optional[str],
        new_actual_path: str,
    ) -> None:
        """Remove old blob file when the actual path changed.

        This happens when a data type change causes a different file
        extension (e.g. ``.parquet`` → ``.pkl.lz4``).  Best-effort:
        failure just logs a warning and leaves an orphan for
        ``verify_integrity`` to clean up later.
        """
        if not old_blob_path or old_blob_path == new_actual_path:
            return
        try:
            if "://" in old_blob_path:
                self._blob_store.blob_backend.delete_blob(old_blob_path)
            else:
                old_resolved = self._resolve_actual_path(old_blob_path)
                if isinstance(old_resolved, Path) and old_resolved.exists():
                    old_resolved.unlink()
            logger.debug(f"Cleaned up old blob for {cache_key}: {old_blob_path}")
        except (OSError, IOError) as exc:
            logger.warning(
                f"Failed to clean up old blob for {cache_key} at {old_blob_path}: {exc}"
            )

    def put(
        self,
        data: Any,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        description: str = "",
        custom_metadata=None,
        hash_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Store any supported data type in cache.

        Args:
            data: Data to cache (DataFrame, array, or general object)
            cache_key: Explicit cache key (if provided, on and **kwargs are ignored)
            hash_key: Alias for cache_key (storage-oriented name). Cannot be
                used together with cache_key.
            on: Dictionary of key parameters for cache key derivation.
                Use this to avoid namespace collisions with cache control
                parameters like description, etc.
            description: Human-readable description
            custom_metadata: Custom metadata for the cache entry. Supports:
                           - Single metadata object: experiment_metadata
                           - List of objects: [experiment_metadata, performance_metadata]
                           - Tuple of objects: (experiment_metadata, performance_metadata)
                           - Dictionary (legacy): {"experiments": experiment_metadata}
            **kwargs: Parameters identifying this data (legacy, use 'on' instead)

        Examples:
            # Explicit cache key
            cache.put(data, cache_key="my-key-123")

            # Dict-based key params (recommended, no namespace collisions)
            cache.put(data, on={'date': '2026-02-08', 'description': 'user val'})

            # Legacy kwargs (still works)
            cache.put(data, date='2026-02-08', region='CA')
        """
        with self._lock:
            cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
            cache_key = self._resolve_cache_key(cache_key, on, kwargs)

            if self.config.storage_mode:
                return self._storage_mode_put(data, cache_key, description)

            base_file_path = self._get_cache_file_path(cache_key)
            cleanup = _PutCleanup()

            # Save old blob path before overwriting — if the data type changes,
            # the new blob may use a different file extension, orphaning the old one
            old_blob_path: Optional[str] = None
            existing = self.metadata_backend.get_entry(cache_key)
            if existing:
                old_meta = existing.get("metadata", {})
                old_blob_path = old_meta.get("actual_path")

            try:
                # Try zero-disk inline serialization (no file I/O at all)
                handler = self._blob_store.handlers.get_handler(data)
                direct = self._try_direct_inline(data, handler)

                if direct is not None:
                    result = direct["result"]
                    file_hash = direct["file_hash"]
                    metadata_dict = self._build_metadata_dict(result, file_hash)
                    metadata_dict["actual_path"] = None
                    metadata_dict["inline_ext"] = direct["inline_ext"]
                    if file_hash is not None:
                        metadata_dict["file_hash"] = file_hash
                else:
                    # Delegate file I/O + handler dispatch to BlobStore
                    wb = self._blob_store._write_blob(
                        data,
                        base_file_path,
                        compute_hash=self.config.metadata.verify_cache_integrity,
                    )
                    handler, result, file_hash = wb.handler, wb.result, wb.file_hash

                    # Track the blob path so we can clean up on failure
                    actual_path_str = result.actual_path
                    if "://" not in actual_path_str:
                        cleanup.blob_path = self._resolve_actual_path(actual_path_str)

                    # If the blob was uploaded to a remote backend (e.g. S3),
                    # track it for rollback in case metadata write fails.
                    if "://" in actual_path_str:
                        cleanup.set_remote(
                            self._blob_store.blob_backend, actual_path_str
                        )

                    # Update metadata
                    metadata_dict = self._build_metadata_dict(result, file_hash)

                    # Record write intent for crash recovery (non-inline only)
                    self._write_journal.record_intent(cache_key, result.actual_path)

                # Store complete cache key parameters as JSON for debugging/querying (if enabled)
                # This captures the original kwargs used to derive the cache key
                # Pre-serialize to JSON strings so backend can be standalone storage
                if self.config.metadata.store_full_metadata:
                    from .serialization import serialize_for_cache_key
                    from .json_utils import dumps as json_dumps

                    try:
                        # Serialize kwargs to a consistent, queryable format
                        serializable_kwargs = {
                            key: serialize_for_cache_key(value, self.config)
                            for key, value in kwargs.items()
                        }
                        # Pre-serialize to JSON string - backend just stores strings
                        metadata_dict["cache_key_params"] = json_dumps(
                            serializable_kwargs
                        )

                        # Also store kwargs as metadata_dict (raw values for easy querying)
                        # Pre-serialize to JSON string - backend just stores strings
                        metadata_dict["metadata_dict"] = json_dumps(kwargs.copy())
                    except (TypeError, ValueError) as e:
                        # If serialization fails, skip cache_key_params
                        logger.warning(f"Failed to serialize cache_key_params: {e}")

                entry_data = {
                    "data_type": handler.data_type,
                    "description": description,
                    "file_size": result.file_size,
                    "metadata": metadata_dict,
                }

                if direct is not None:
                    # Direct inline — data already in memory, no disk file
                    entry_data["blob_data"] = direct["blob_data"]
                    entry_data["is_inline"] = 1
                else:
                    # Try disk-based inline (read back from file)
                    inline = self._try_inline_blob(result, file_hash, cleanup)
                    if inline is not None:
                        entry_data["blob_data"] = inline["blob_data"]
                        entry_data["is_inline"] = 1
                        metadata_dict["actual_path"] = None
                        metadata_dict["inline_ext"] = inline["inline_ext"]
                        if inline["file_hash"] is not None:
                            metadata_dict["file_hash"] = inline["file_hash"]

                # Populate per-entry TTL from config default (if set)
                default_ttl = self.config.metadata.default_ttl_seconds
                if default_ttl is not None:
                    entry_data["ttl_seconds"] = int(default_ttl)
                    # expires_at is computed by the backend from created_at + ttl_seconds

                # Sign the entry if signing is enabled
                self._sign_entry_if_enabled(cache_key, entry_data, metadata_dict)

                self.metadata_backend.put_entry(cache_key, entry_data)

                # Clear write intent — metadata committed successfully
                self._write_journal.clear_intent(cache_key)
                self._cleanup_stale_blob(cache_key, old_blob_path, result.actual_path)

                # Handle custom metadata if provided
                if custom_metadata and self._supports_custom_metadata():
                    self._store_custom_metadata(cache_key, custom_metadata)

                self._enforce_size_limit()

                file_size_display = format_size(result.file_size)
                format_info = f"({result.storage_format} format)"
                logger.info(
                    f"Cached {handler.data_type} {cache_key} ({file_size_display}) {format_info}: {description}"
                )

                cleanup.commit()
                return cache_key

            except (OSError, IOError) as e:
                cleanup.rollback()
                self._write_journal.clear_intent(cache_key)
                data_type = handler.data_type if "handler" in locals() else "unknown"
                logger.error(f"Failed to cache {data_type} (I/O error): {e}")
                raise
            except Exception as e:  # intentionally broad — re-raises after cleanup
                cleanup.rollback()
                self._write_journal.clear_intent(cache_key)
                data_type = handler.data_type if "handler" in locals() else "unknown"
                logger.error(f"Failed to cache {data_type}: {type(e).__name__}: {e}")
                raise

    def get(
        self,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        ttl: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        hash_key: Optional[str] = None,
        **kwargs,
    ) -> Optional[Any]:
        """
        Retrieve any supported data type from cache.

        Args:
            cache_key: Direct cache key (if provided, on and **kwargs are ignored)
            hash_key: Alias for cache_key (storage-oriented name).
            on: Dictionary of key parameters for cache key derivation.
                Use this to avoid namespace collisions with cache control
                parameters like ttl, ttl_seconds, etc.
            ttl: TTL as a human-readable duration string (e.g. "6h", "2d").
                Overrides default. Mutually exclusive with ``ttl_seconds``.
            ttl_seconds: TTL in seconds (numeric only). Overrides default.
                Mutually exclusive with ``ttl``. None = never expire.
            **kwargs: Parameters identifying the cached data (legacy, use 'on' instead)

        Returns:
            Cached data or None if not found/expired
        """
        with self._lock:
            resolved_ttl = resolve_ttl(
                ttl, ttl_seconds, _param_owner="UnifiedCache.get"
            )
            cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
            cache_key = self._resolve_cache_key(cache_key, on, kwargs)

            if self.config.storage_mode:
                return self._storage_mode_get(cache_key)

            # Check if entry exists and is not expired
            entry = self.metadata_backend.get_entry(cache_key)
            if not entry or self._is_expired(cache_key, resolved_ttl):
                self._record_miss()
                return None

            data_type = entry.get("data_type")
            if not data_type:
                self._record_miss()
                return None

            try:
                base_file_path = self._get_cache_file_path(cache_key)

                # Use actual path from metadata if available, otherwise use base path
                metadata = entry.get("metadata", {})
                actual_path = metadata.get("actual_path")
                if actual_path:
                    file_path = self._resolve_actual_path(actual_path)
                else:
                    file_path = base_file_path

                # Integrity + signature verification
                if not self._verify_entry(cache_key, entry, metadata, file_path):
                    self._record_miss()
                    return None

                # Delegate blob read to BlobStore
                if entry.get("is_inline") and entry.get("blob_data") is not None:
                    data = self._read_inline_blob(entry, data_type, metadata)
                else:
                    data = self._blob_store._read_blob(file_path, data_type, metadata)

                # Update access time
                self.metadata_backend.update_access_time(cache_key)
                self._record_hit()

                logger.debug(f"Cache hit ({data_type}): {cache_key}")
                return data

            except FileNotFoundError as e:
                # Cache file was deleted externally — blob already gone,
                # just clean up metadata (no blob to delete)
                logger.warning(f"Cache file missing for {cache_key}: {e}")
                self.metadata_backend.remove_entry(cache_key)
                self._record_miss()
                return None
            except (OSError, IOError) as e:
                # I/O errors may be transient (disk temporarily unavailable, etc.)
                # Do NOT delete metadata — the entry may be readable on retry
                logger.warning(f"I/O error loading cached {data_type} {cache_key}: {e}")
                self._record_miss()
                return None
            except (
                Exception
            ) as e:  # intentionally broad — deserialization may fail any way
                # Unexpected errors (deserialization failures, corruption, etc.)
                if self.config.metadata.delete_on_error:
                    logger.warning(
                        f"Failed to load cached {data_type} {cache_key}: {type(e).__name__}: {e}. "
                        f"Removing corrupted cache entry."
                    )
                    self._blob_store.delete(cache_key)
                else:
                    logger.warning(
                        f"Failed to load cached {data_type} {cache_key}: {type(e).__name__}: {e}. "
                        f"Entry retained due to delete_on_error=False."
                    )
                self._record_miss()
                return None

    def get_with_metadata(
        self,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        ttl_seconds=_DEFAULT_TTL,
        hash_key: Optional[str] = None,
        **kwargs,
    ) -> Optional[tuple[Any, Dict[str, Any]]]:
        """
        Retrieve cached data along with its metadata in a single atomic operation.

        This method combines get() and get_metadata() into one call, avoiding
        separate metadata lookups. Useful when you need both the data and its
        metadata (e.g., created_at, file_size, custom metadata).

        Args:
            cache_key: Direct cache key (if provided, on and **kwargs are ignored)
            hash_key: Alias for cache_key (storage-oriented name).
            on: Dictionary of key parameters for cache key derivation.
                Use this to avoid namespace collisions with cache control parameters.
            ttl_seconds: Custom TTL in seconds (overrides default). None = never expire.
                         Use _DEFAULT_TTL sentinel (default) to use config's default_ttl_seconds.
            **kwargs: Parameters identifying the cached data (legacy, use 'on' instead)

        Returns:
            Tuple of (data, metadata_dict) if found and not expired, None otherwise

        Example:
            result = cache.get_with_metadata(on={'experiment': 'exp_001'})
            if result:
                data, metadata = result
                print(f"Created: {metadata['created_at']}")
                print(f"Size: {metadata.get('file_size_bytes', 0)} bytes")
                process(data)
        """
        with self._lock:
            cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
            cache_key = self._resolve_cache_key(cache_key, on, kwargs)

            if self.config.storage_mode:
                return self._storage_mode_get_with_metadata(cache_key)

            # Single metadata lookup
            entry = self.metadata_backend.get_entry(cache_key)
            if not entry:
                self._record_miss()
                return None

            # Check TTL expiration directly using the already-retrieved entry
            # to avoid a second metadata lookup
            if ttl_seconds is _DEFAULT_TTL:  # Use config default
                actual_ttl = self.config.metadata.default_ttl_seconds
            else:  # Use provided TTL (could be None for infinite or a specific value)
                actual_ttl = ttl_seconds

            # If TTL is not None (infinite), check expiration
            if actual_ttl is not None:
                creation_time_str = entry.get("created_at")
                if creation_time_str:
                    if isinstance(creation_time_str, str):
                        creation_time = datetime.fromisoformat(creation_time_str)
                    else:
                        creation_time = creation_time_str

                    if creation_time.tzinfo is None:
                        creation_time = creation_time.replace(tzinfo=timezone.utc)

                    expiry_time = creation_time + timedelta(seconds=actual_ttl)
                    current_time = datetime.now(timezone.utc)

                    if current_time > expiry_time:
                        self._record_miss()
                        return None

            # Get appropriate handler
            data_type = entry.get("data_type")
            if not data_type:
                self._record_miss()
                return None

            try:
                base_file_path = self._get_cache_file_path(cache_key)

                # Use actual path from metadata if available, otherwise use base path
                metadata = entry.get("metadata", {})
                actual_path = metadata.get("actual_path")
                if actual_path:
                    file_path = self._resolve_actual_path(actual_path)
                else:
                    file_path = base_file_path

                # Integrity + signature verification
                if not self._verify_entry(cache_key, entry, metadata, file_path):
                    self._record_miss()
                    return None

                # Delegate blob read to BlobStore
                if entry.get("is_inline") and entry.get("blob_data") is not None:
                    data = self._read_inline_blob(entry, data_type, metadata)
                else:
                    data = self._blob_store._read_blob(file_path, data_type, metadata)

                # Update access time
                self.metadata_backend.update_access_time(cache_key)
                self._record_hit()

                # Include cache_key in returned metadata
                entry["cache_key"] = cache_key

                logger.debug(f"Cache hit with metadata ({data_type}): {cache_key}")
                return (data, entry)

            except FileNotFoundError as e:
                # Cache file was deleted externally — blob already gone,
                # just clean up metadata (no blob to delete)
                logger.warning(f"Cache file missing for {cache_key}: {e}")
                self.metadata_backend.remove_entry(cache_key)
                self._record_miss()
                return None
            except (OSError, IOError) as e:
                # I/O errors may be transient (disk temporarily unavailable, etc.)
                # Do NOT delete metadata — the entry may be readable on retry
                logger.warning(f"I/O error loading cached {data_type} {cache_key}: {e}")
                self._record_miss()
                return None
            except (
                Exception
            ) as e:  # intentionally broad — deserialization may fail any way
                # Unexpected errors (deserialization failures, corruption, etc.)
                if self.config.metadata.delete_on_error:
                    logger.warning(
                        f"Failed to load cached {data_type} {cache_key}: {type(e).__name__}: {e}. "
                        f"Removing corrupted cache entry."
                    )
                    self._blob_store.delete(cache_key)
                else:
                    logger.warning(
                        f"Failed to load cached {data_type} {cache_key}: {type(e).__name__}: {e}. "
                        f"Entry retained due to delete_on_error=False."
                    )
                self._record_miss()
                return None

    def get_metadata(
        self,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        check_expiration: bool = True,
        hash_key: Optional[str] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Get entry metadata without loading blob data.

        This is useful for inspecting cache entries (TTL, file size, data type)
        before deciding whether to load the actual data.

        Args:
            cache_key: Direct cache key (if provided, on and **kwargs are ignored)
            on: Dictionary of key parameters for cache key derivation.
                Use this to avoid namespace collisions with cache control parameters.
            check_expiration: If True, returns None for expired entries (default: True)
            **kwargs: Parameters identifying the cached data (legacy, use 'on' instead)

        Returns:
            Metadata dictionary or None if not found or expired

        Example:
            # Check metadata before loading large file
            meta = cache.get_metadata(on={'experiment': 'exp_001'})
            if meta and meta.get("file_size_bytes", 0) > 1e9:
                print("Large file - loading may take time")
                data = cache.get(on={'experiment': 'exp_001'})
        """
        with self._lock:
            cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
            cache_key = self._resolve_cache_key(cache_key, on, kwargs)

            entry = self.metadata_backend.get_entry(cache_key)
            if not entry:
                return None

            # Check expiration if requested (respects cache TTL policy)
            if check_expiration and self._is_expired(cache_key):
                return None

            # Ensure cache_key is included in returned metadata
            entry["cache_key"] = cache_key

            return entry

    # ── Convenience helpers: auto-populate metadata from cache key params ──

    def put_with_meta(
        self,
        data: Any,
        *,
        on: Optional[Dict] = None,
        description: str = "",
        **kwargs,
    ) -> str:
        """Store data using kwargs as both the cache key and metadata_dict.

        Every keyword argument is used to derive a deterministic cache key
        **and** stored as a queryable ``metadata_dict`` entry.  This avoids
        the common pattern of passing the same values twice (once for key
        derivation and once for metadata).

        Requires ``store_full_metadata=True`` in :class:`CacheConfig`.

        Args:
            data: Data to cache.
            on: Optional extra key-only parameters that participate in cache
                key derivation but are **not** stored in ``metadata_dict``.
                Use this to create distinct cache entries that share the
                same metadata (e.g. ``on={"epoch": 5}``).
            description: Human-readable description (not part of the cache key).
            **kwargs: Key-value pairs that become *both* cache key params and
                ``metadata_dict`` entries.

        Returns:
            The 16-character hex cache key.

        Raises:
            ValueError: If no kwargs are provided, ``store_full_metadata``
                is disabled, or *on* keys overlap with kwargs.

        Example:
            cache.put_with_meta(df, experiment="exp_001", model="xgboost",
                                accuracy=0.95)
            # With key discriminator:
            cache.put_with_meta(df, on={"epoch": 5},
                                model="xgboost", lr=0.01)
        """
        if not kwargs:
            raise ValueError(
                "put_with_meta() requires at least one keyword argument "
                "to derive the cache key and populate metadata."
            )
        if not self.config.metadata.store_full_metadata:
            raise ValueError(
                "put_with_meta() requires store_full_metadata=True in "
                "CacheConfig so that kwargs are persisted as metadata_dict."
            )
        key_params = self._merge_on_and_kwargs(on, kwargs)
        cache_key = self._create_cache_key(key_params)
        # Pass cache_key (pre-computed) so _resolve_cache_key won't
        # conflict with **kwargs.  kwargs still flow to put() for
        # metadata_dict storage via store_full_metadata.
        return self.put(data, cache_key=cache_key, description=description, **kwargs)

    def get_with_meta(
        self,
        *,
        on: Optional[Dict] = None,
        ttl: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        **kwargs,
    ) -> Optional[tuple[Any, Dict[str, Any]]]:
        """Retrieve data and its metadata_dict by exact key derived from kwargs.

        This is the read counterpart of :meth:`put_with_meta`.  All kwargs
        (and any *on* discriminators) are used to derive the same
        deterministic cache key; on a hit the stored ``metadata_dict``
        is returned alongside the data.

        Args:
            on: Optional extra key-only parameters that were used as
                discriminators during :meth:`put_with_meta`.  Must match
                the same *on* dict used at store time.
            ttl: TTL as a human-readable duration string (e.g. ``"6h"``).
            ttl_seconds: TTL in seconds.  Mutually exclusive with *ttl*.
            **kwargs: The same key-value pairs used when the entry was stored
                via :meth:`put_with_meta`.

        Returns:
            ``(data, metadata_dict)`` on a cache hit, ``None`` on a miss.
            ``metadata_dict`` is a plain ``dict`` of the originally stored
            kwargs (not the raw entry envelope).

        Example:
            result = cache.get_with_meta(experiment="exp_001", model="xgboost")
            if result:
                data, meta = result
                print(meta["accuracy"])
        """
        if not kwargs:
            raise ValueError("get_with_meta() requires at least one keyword argument.")
        key_params = self._merge_on_and_kwargs(on, kwargs)
        cache_key = self._create_cache_key(key_params)
        data = self.get(cache_key=cache_key, ttl=ttl, ttl_seconds=ttl_seconds)
        if data is None:
            return None
        entry = self.metadata_backend.get_entry(cache_key)
        if entry is None:
            return None
        return (data, self._extract_metadata_dict(entry))

    def put_with_model(
        self,
        data: Any,
        model_class: type,
        *,
        on: Optional[Dict] = None,
        description: str = "",
        **kwargs,
    ) -> str:
        """Store data with kwargs as both cache key and ORM custom metadata.

        A convenience wrapper that constructs an ORM instance from *kwargs*,
        derives the cache key from the same values, and stores everything in
        a single call.  Requires a SQLite or PostgreSQL metadata backend.

        Args:
            data: Data to cache.
            model_class: A custom metadata model class decorated with
                ``@register_custom_metadata``.  Must accept all *kwargs*
                as column keyword arguments.
            on: Optional extra key-only parameters that participate in cache
                key derivation but are **not** stored in ORM columns.
                Use this to create distinct cache entries that share the
                same ORM metadata values.
            description: Human-readable description (not part of the cache key).
            **kwargs: Values passed to ``model_class(...)`` *and* used
                for cache key derivation.

        Returns:
            The 16-character hex cache key.

        Raises:
            ValueError: If no kwargs are provided, custom metadata is
                not supported, or *on* keys overlap with kwargs.
            TypeError: If *model_class* cannot be instantiated with the
                given kwargs.

        Example:
            cache.put_with_model(df, ExperimentMetadata,
                                 experiment_id="exp_001",
                                 model_type="xgboost", accuracy=0.95)
            # With key discriminator:
            cache.put_with_model(df, ExperimentMetadata,
                                 on={"run_id": "run_42"},
                                 experiment_id="exp_001",
                                 model_type="xgboost", accuracy=0.95)
        """
        if not kwargs:
            raise ValueError(
                "put_with_model() requires at least one keyword argument "
                "to derive the cache key and populate ORM columns."
            )
        if not self._supports_custom_metadata():
            raise ValueError(
                "put_with_model() requires a SQLite or PostgreSQL metadata "
                "backend for custom metadata support."
            )
        key_params = self._merge_on_and_kwargs(on, kwargs)
        instance = model_class(**kwargs)
        return self.put(
            data, on=key_params, description=description, custom_metadata=instance
        )

    def get_with_model(
        self,
        model_class: type,
        *,
        on: Optional[Dict] = None,
        ttl: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        **kwargs,
    ) -> Optional[tuple[Any, Any]]:
        """Retrieve data and its ORM metadata instance by exact key.

        The read counterpart of :meth:`put_with_model`.  All kwargs (and
        any *on* discriminators) derive the cache key; on a hit the
        matching ORM instance is fetched from the custom metadata table.

        Args:
            model_class: The same model class used when the entry was stored.
            on: Optional extra key-only parameters that were used as
                discriminators during :meth:`put_with_model`.  Must match
                the same *on* dict used at store time.
            ttl: TTL as a human-readable duration string.
            ttl_seconds: TTL in seconds.  Mutually exclusive with *ttl*.
            **kwargs: The same key-value pairs used in :meth:`put_with_model`.

        Returns:
            ``(data, orm_instance)`` on a cache hit, ``None`` on a miss or
            if no ORM row exists for the entry.

        Example:
            result = cache.get_with_model(ExperimentMetadata,
                                          experiment_id="exp_001",
                                          model_type="xgboost")
            if result:
                data, exp = result
                print(exp.accuracy)
        """
        if not kwargs:
            raise ValueError("get_with_model() requires at least one keyword argument.")
        key_params = self._merge_on_and_kwargs(on, kwargs)
        cache_key = self._create_cache_key(key_params)
        data = self.get(cache_key=cache_key, ttl=ttl, ttl_seconds=ttl_seconds)
        if data is None:
            return None

        from .custom_metadata import get_schema_name_for_model

        schema_name = get_schema_name_for_model(model_class)
        if schema_name is None:
            logger.warning(
                f"Model class {model_class.__name__} is not registered "
                f"with @register_custom_metadata"
            )
            return None
        custom = self._get_custom_metadata(cache_key)
        instance = custom.get(schema_name)
        if instance is None:
            return None
        return (data, instance)

    def query_with_meta(self, **kwargs):
        """Generate ``(data, metadata_dict)`` tuples for entries matching filters.

        Uses :meth:`query_meta` internally to find entries whose
        ``metadata_dict`` contains all the given key-value pairs, then
        lazily loads the blob data for each match.

        Args:
            **kwargs: Metadata filters (all must match).

        Yields:
            ``(data, metadata_dict)`` for each matching entry whose data
            can be loaded successfully.

        Example:
            for data, meta in cache.query_with_meta(model="xgboost"):
                print(meta["accuracy"])
        """
        matches = self.query_meta(**kwargs)
        if not matches:
            return
        for entry in matches:
            cache_key = entry.get("cache_key")
            if cache_key is None:
                continue
            data = self.get(cache_key=cache_key)
            if data is not None:
                yield (data, entry.get("metadata_dict", {}))

    def query_with_model(self, model_class: type, **kwargs):
        """Generate ``(data, orm_instance)`` tuples for entries matching ORM filters.

        Queries the custom metadata table for *model_class* using *kwargs*
        as column equality filters, then lazily loads the blob data for
        each match.

        Args:
            model_class: A registered custom metadata model class.
            **kwargs: Column equality filters (all must match).

        Yields:
            ``(data, orm_instance)`` for each matching row whose cache
            entry can be loaded successfully.

        Raises:
            ValueError: If custom metadata is not supported or the model
                class is not registered.

        Example:
            for data, exp in cache.query_with_model(ExperimentMetadata,
                                                     model_type="xgboost"):
                print(exp.accuracy)
        """
        if not self._supports_custom_metadata():
            raise ValueError(
                "query_with_model() requires a SQLite or PostgreSQL metadata backend."
            )

        from .custom_metadata import (
            get_schema_name_for_model,
            get_namespace_custom_model,
        )

        schema_name = get_schema_name_for_model(model_class)
        if schema_name is None:
            raise ValueError(
                f"Model class {model_class.__name__} is not registered "
                f"with @register_custom_metadata."
            )

        ns_model = get_namespace_custom_model(schema_name, self.namespace)
        if ns_model is None:
            raise ValueError(f"No namespace model found for schema '{schema_name}'.")

        if not hasattr(self.metadata_backend, "SessionLocal"):
            raise ValueError("SQLAlchemy session not available.")

        # Eagerly load all matching ORM instances and detach them from the
        # session so the caller can use them freely after the session closes.
        with self.metadata_backend.SessionLocal() as session:
            query = session.query(ns_model)
            for field, value in kwargs.items():
                if not hasattr(ns_model, field):
                    raise ValueError(
                        f"Unknown column '{field}' on model '{model_class.__name__}'."
                    )
                query = query.filter(getattr(ns_model, field) == value)

            instances = query.all()
            for inst in instances:
                session.expunge(inst)

        # Yield (data, orm_instance) lazily — data loading may be expensive.
        for instance in instances:
            cache_key = instance.cache_key
            data = self.get(cache_key=cache_key)
            if data is not None:
                yield (data, instance)

    # ── File convenience helpers ────────────────────────────────────

    def put_file(
        self,
        file_path: str | Path,
        *,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        description: str = "",
        custom_metadata=None,
        move: bool = False,
        **kwargs,
    ) -> str:
        """Store an arbitrary file in the cache.

        Reads the file into memory as raw bytes and delegates to
        :meth:`put`.  File metadata (original filename, MIME type,
        file size) is automatically recorded in ``metadata_dict``
        when ``store_full_metadata=True``.

        Args:
            file_path: Path to the source file (``str`` or ``pathlib.Path``).
            cache_key: Explicit cache key.  When provided, *on* and
                ``**kwargs`` are ignored for key derivation.
            on: Dictionary of key parameters for cache key derivation.
            description: Human-readable description.
            custom_metadata: Custom metadata for the cache entry.  Supports
                single ORM objects, lists/tuples of ORM objects, or dicts.
                Passed through to :meth:`put` unchanged.
            move: If ``True``, delete the source file after a successful
                store (move-in semantics).  Defaults to ``False`` (copy-in).
            **kwargs: Extra key-value pairs for key derivation and/or
                ``metadata_dict`` (when ``store_full_metadata=True``).

        Returns:
            The 16-character hex cache key.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            IsADirectoryError: If *file_path* is a directory.

        Example:
            key = cache.put_file("data/model.onnx",
                                  description="ONNX model v2")
            key = cache.put_file("output.csv", on={"run": "exp_01"})
        """
        import mimetypes

        src = Path(file_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file does not exist: {src}")
        if src.is_dir():
            raise IsADirectoryError(f"Expected a file, got a directory: {src}")

        data = src.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(src))

        # File metadata must NOT participate in cache key derivation.
        # Resolve the key from user-supplied params first, then pass
        # everything (file meta + user kwargs) with an explicit cache_key
        # so that _resolve_cache_key returns immediately.
        file_meta = {
            "original_filename": src.name,
            "mime_type": mime_type or "application/octet-stream",
            "original_size": len(data),
        }
        merged_kwargs = {**file_meta, **kwargs}  # user kwargs win on conflict

        if cache_key is None:
            cache_key = self._resolve_cache_key(None, on, kwargs)

        result_key = self.put(
            data,
            cache_key=cache_key,
            description=description,
            custom_metadata=custom_metadata,
            **merged_kwargs,
        )

        if move:
            src.unlink()

        return result_key

    def get_file(
        self,
        cache_key: Optional[str] = None,
        *,
        dest: str | Path | None = None,
        on: Optional[Dict] = None,
        ttl: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        move: bool = False,
        overwrite: bool = True,
        **kwargs,
    ) -> Optional[bytes | Path]:
        """Retrieve cached file data, optionally writing it to disk.

        This is the read counterpart of :meth:`put_file`.  When *dest*
        is provided the raw bytes are written to that path and a
        ``pathlib.Path`` is returned.  Otherwise the raw ``bytes`` are
        returned directly.

        Args:
            cache_key: Explicit cache key.  When provided, *on* and
                ``**kwargs`` are ignored.
            dest: Optional destination path.  Parent directories are
                created automatically.  If *dest* is a directory, the
                original filename from metadata is used (falls back to
                ``<cache_key>.bin`` when unavailable).
            on: Dictionary of key parameters for key lookup.
            ttl: TTL as a human-readable duration string (e.g. ``"6h"``).
            ttl_seconds: TTL in seconds.  Mutually exclusive with *ttl*.
            move: If ``True``, delete the cache entry after a successful
                write to *dest* (move-out semantics).  Requires *dest*
                to be set — raises ``ValueError`` otherwise.  Defaults
                to ``False`` (copy-out).
            overwrite: If ``False``, raise ``FileExistsError`` when
                *dest* already exists on disk.  Defaults to ``True``
                (silently overwrite).
            **kwargs: Key-value pairs for key derivation (must match what
                was passed to :meth:`put_file`).

        Returns:
            * ``bytes`` — when *dest* is ``None`` and entry exists.
            * ``pathlib.Path`` — when *dest* is given and entry exists.
            * ``None`` — on a cache miss.

        Raises:
            ValueError: If *move* is ``True`` but *dest* is ``None``.
            FileExistsError: If *overwrite* is ``False`` and *dest*
                already exists.

        Example:
            raw = cache.get_file("abc123def4567890")
            path = cache.get_file("abc123def4567890",
                                   dest="output/model.onnx")
        """
        if move and dest is None:
            raise ValueError(
                "move=True requires dest to be set. "
                "Without a destination path, use get() + invalidate() instead."
            )

        data = self.get(
            cache_key=cache_key,
            on=on,
            ttl=ttl,
            ttl_seconds=ttl_seconds,
            **kwargs,
        )
        if data is None:
            return None

        # Ensure we have bytes (in case the entry was stored without put_file)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"Expected bytes from cache, got {type(data).__name__}. "
                "get_file() should only be used with entries stored via put_file()."
            )
        raw = bytes(data) if not isinstance(data, bytes) else data

        if dest is None:
            return raw

        dest_path = Path(dest)
        if dest_path.is_dir():
            # Resolve filename from metadata
            resolved_key = (
                cache_key
                if cache_key is not None
                else self._resolve_cache_key(None, on, kwargs)
            )
            filename = self._resolve_original_filename(resolved_key)
            dest_path = dest_path / filename

        if not overwrite and dest_path.exists():
            raise FileExistsError(
                f"Destination already exists: {dest_path}. "
                "Pass overwrite=True to overwrite."
            )

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(raw)

        if move:
            resolved_key = (
                cache_key
                if cache_key is not None
                else self._resolve_cache_key(None, on, kwargs)
            )
            self.invalidate(cache_key=resolved_key)

        return dest_path

    def _resolve_original_filename(self, cache_key: str) -> str:
        """Look up the original filename from metadata, with fallback."""
        entry = self.metadata_backend.get_entry(cache_key)
        if entry:
            meta = self._extract_metadata_dict(entry)
            name = meta.get("original_filename")
            if name:
                return name
        return f"{cache_key}.bin"

    # ── Private helpers for convenience methods ─────────────────────

    @staticmethod
    def _merge_on_and_kwargs(
        on: Optional[Dict], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge *on* discriminators with *kwargs* for cache key derivation.

        Raises :class:`ValueError` if *on* contains keys that also appear
        in *kwargs* (ambiguous key sources are a bug).

        Returns a merged dict suitable for :meth:`_create_cache_key`.
        """
        if not on:
            return dict(kwargs)
        overlap = set(on) & set(kwargs)
        if overlap:
            raise ValueError(
                f"'on' keys overlap with kwargs: {sorted(overlap)}. "
                "Each parameter must appear in either 'on' or kwargs, not both."
            )
        return {**on, **kwargs}

    @staticmethod
    def _extract_metadata_dict(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the user-facing ``metadata_dict`` from a raw entry.

        The stored value may be a JSON string (SQLite/PG) or already a
        ``dict`` (in-memory / JSON backend).  Returns an empty dict when
        the field is absent or unparseable.
        """
        meta = entry.get("metadata", {})
        raw = meta.get("metadata_dict")
        if raw is None:
            raw = entry.get("metadata_dict")
        if isinstance(raw, str):
            try:
                from .json_utils import loads as json_loads

                return json_loads(raw)
            except (ValueError, KeyError):  # intentionally broad — malformed JSON
                return {}
        if isinstance(raw, dict):
            return raw
        return {}

    def exists(
        self,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        check_expiration: bool = True,
        hash_key: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Check if a cache entry exists without loading the blob file.

        This is a lightweight metadata-only check that avoids loading large cached
        objects into memory. Useful for existence checks before calling get().

        Args:
            cache_key: Direct cache key (if provided, on and **kwargs are ignored)
            on: Dictionary of key parameters for cache key derivation.
                Use this to avoid namespace collisions with cache control parameters.
            check_expiration: If True, returns False for expired entries (default: True)
            **kwargs: Parameters identifying the cached data (legacy, use 'on' instead)

        Returns:
            bool: True if entry exists and is not expired, False otherwise

        Example:
            # Check before loading large DataFrame
            params = {'experiment': 'exp_001', 'run_id': 42}
            if cache.exists(on=params):
                df = cache.get(on=params)
            else:
                df = expensive_computation()
                cache.put(df, on=params)
        """
        with self._lock:
            cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
            cache_key = self._resolve_cache_key(cache_key, on, kwargs)

            entry = self.metadata_backend.get_entry(cache_key)
            if not entry:
                return False

            # Check expiration if requested
            if check_expiration and self._is_expired(cache_key):
                return False

            return True

    def update_data(
        self,
        data: Any,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        hash_key: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Update blob data at an existing cache entry without changing the cache_key.

        This replaces the stored data at a fixed cache_key while updating derived
        metadata (file_size, content_hash, created_at timestamp). The cache_key
        itself remains unchanged to maintain referential integrity.

        Args:
            data: New data to store (must be serializable by handler)
            cache_key: Direct cache key (if provided, on and **kwargs are ignored)
            on: Dictionary of key parameters for cache key derivation.
                Use this to avoid namespace collisions with cache control parameters.
            **kwargs: Parameters identifying the cached data (legacy, use 'on' instead)

        Returns:
            bool: True if entry was updated, False if entry doesn't exist

        Example:
            # Update cached DataFrame with new data
            success = cache.update_data(
                new_df,
                on={'experiment': 'exp_001', 'run_id': 42}
            )

            if not success:
                print("Entry not found - use put() to create new entry")

        Note:
            - Cache_key is immutable and derived from input params (not content)
            - Use update_data() to refresh data at same logical location
            - Use put() to create new entries
            - created_at timestamp is reset to now (acts like touch)
        """
        with self._lock:
            cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
            cache_key = self._resolve_cache_key(cache_key, on, kwargs)

            # Check if entry exists before doing any I/O
            existing_entry = self.metadata_backend.get_entry(cache_key)
            if not existing_entry:
                logger.warning(
                    f"⚠️ Cache entry not found for update: {cache_key[:16]}..."
                )
                return False

            # Capture old blob path so we can delete it AFTER metadata
            # succeeds.  This is the write-then-swap strategy: write the
            # new blob to a staging location, update metadata, then
            # remove the old blob.  If metadata update fails, rollback
            # deletes only the staging blob; the old data stays intact.
            old_metadata = existing_entry.get("metadata", {})
            old_actual_path: Optional[str] = old_metadata.get("actual_path")

            base_file_path = self._get_cache_file_path(cache_key)
            staging_suffix = f"_stg{uuid.uuid4().hex[:8]}"
            staging_base = base_file_path.parent / (
                base_file_path.name + staging_suffix
            )
            cleanup = _PutCleanup()

            try:
                # Try zero-disk inline serialization first
                handler = self._blob_store.handlers.get_handler(data)
                direct = self._try_direct_inline(data, handler)

                if direct is not None:
                    result = direct["result"]
                    updates = {
                        "file_size": result.file_size,
                        "content_hash": result.extra.get("content_hash"),
                        "file_hash": direct["file_hash"],
                        "actual_path": None,
                        "storage_format": result.storage_format,
                        "blob_data": direct["blob_data"],
                        "is_inline": 1,
                        "inline_ext": direct["inline_ext"],
                    }
                    if hasattr(handler, "data_type"):
                        updates["data_type"] = handler.data_type
                    if result.serializer:
                        updates["serializer"] = result.serializer
                    if result.compression_codec:
                        updates["compression_codec"] = result.compression_codec
                    if result.object_type:
                        updates["object_type"] = result.object_type
                else:
                    # Write new blob to staging path (different blob_id)
                    wb = self._blob_store._write_blob(
                        data, staging_base, compute_hash=False
                    )
                    handler, result = wb.handler, wb.result

                    actual_path_str = result.actual_path
                    if "://" not in actual_path_str:
                        cleanup.blob_path = self._resolve_actual_path(actual_path_str)

                    # Track remote blob for rollback on S3
                    if "://" in actual_path_str:
                        cleanup.set_remote(
                            self._blob_store.blob_backend, actual_path_str
                        )

                    # Build metadata updates dict from handler result
                    updates = {
                        "file_size": result.file_size,
                        "content_hash": result.extra.get("content_hash"),
                        "file_hash": result.extra.get("file_hash"),
                        "actual_path": actual_path_str,
                        "storage_format": result.storage_format,
                    }
                    if hasattr(handler, "data_type"):
                        updates["data_type"] = handler.data_type
                    if result.serializer:
                        updates["serializer"] = result.serializer
                    if result.compression_codec:
                        updates["compression_codec"] = result.compression_codec
                    if result.object_type:
                        updates["object_type"] = result.object_type
                    if result.extra.get("s3_etag"):
                        updates["s3_etag"] = result.extra["s3_etag"]

                    # Try disk-based inline (read back from file)
                    inline = self._try_inline_blob(result, None, cleanup)
                    if inline is not None:
                        updates["blob_data"] = inline["blob_data"]
                        updates["is_inline"] = 1
                        updates["actual_path"] = None
                        updates["inline_ext"] = inline["inline_ext"]
                        if inline["file_hash"] is not None:
                            updates["file_hash"] = inline["file_hash"]
                    else:
                        # Ensure previous inline data is cleared if blob is now external
                        updates["blob_data"] = None
                        updates["is_inline"] = 0
                        updates["inline_ext"] = None

                # Delegate metadata-only update to backend (no I/O in metadata layer)
                self.metadata_backend.update_entry_metadata(
                    cache_key=cache_key, updates=updates
                )

                # Re-sign the entry if signing is enabled (security: signature must match updated data)
                if self.signer:
                    try:
                        # Get the updated entry from backend
                        updated_entry = self.metadata_backend.get_entry(cache_key)
                        if updated_entry:
                            # Recalculate file hash for integrity verification
                            metadata = updated_entry.get("metadata", {})
                            actual_path = metadata.get("actual_path")
                            if (
                                actual_path
                                and self.config.metadata.verify_cache_integrity
                            ):
                                new_file_hash = self._blob_store._calculate_file_hash(
                                    self._resolve_actual_path(actual_path)
                                )
                                metadata["file_hash"] = new_file_hash

                            # Extract signable fields and create new signature
                            complete_entry_data = self._extract_signable_fields(
                                cache_key=cache_key,
                                entry_data=updated_entry,
                                metadata=metadata,
                            )

                            # Generate new signature for updated entry
                            new_signature = self.signer.sign_entry(complete_entry_data)
                            metadata["entry_signature"] = new_signature

                            # Update entry with new signature and file hash
                            updated_entry["metadata"] = metadata
                            self.metadata_backend.put_entry(cache_key, updated_entry)

                            logger.debug(f"Re-signed updated entry {cache_key}")
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Failed to re-sign updated entry {cache_key}: {e}"
                        )
                        # Continue - update succeeded, just missing signature

                # Delete the OLD blob now that metadata points to the
                # new staging location.  Failure here is non-fatal: the
                # update already succeeded; the old blob is just orphaned.
                if old_actual_path and old_actual_path != actual_path_str:
                    try:
                        old_resolved = self._resolve_actual_path(old_actual_path)
                        self._blob_store.blob_backend.delete_blob(str(old_resolved))
                        logger.debug(
                            f"Deleted old blob after update: {old_actual_path}"
                        )
                    except (OSError, IOError):  # intentionally broad — orphan cleanup
                        logger.warning(
                            f"Failed to delete old blob after update: {old_actual_path}"
                        )

                logger.info(f"Updated cache entry: {cache_key[:16]}...")
                cleanup.commit()
                return True

            except Exception as e:  # intentionally broad — re-raises after cleanup
                cleanup.rollback()
                logger.error(
                    f"Failed to update cache entry {cache_key[:16]}...: "
                    f"{type(e).__name__}: {e}"
                )
                raise

    def touch(
        self,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        hash_key: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Update entry timestamp to extend TTL without reloading data.

        This "touches" the cache entry to reset its creation timestamp to now,
        effectively extending the entry's lifetime by the full configured TTL.
        Useful for keeping frequently accessed data alive or preventing
        expiration of long-running computations.

        Args:
            cache_key: Direct cache key (if provided, on and **kwargs are ignored)
            on: Dictionary of key parameters for cache key derivation.
                Use this to avoid namespace collisions with cache control parameters.
            **kwargs: Parameters identifying the cached data (legacy, use 'on' instead)

        Returns:
            bool: True if entry exists and was touched, False if entry doesn't exist

        Example:
            # Reset TTL to full default duration from now
            cache.touch(on={'experiment': 'exp_001'})

            # Keep long-running computation alive
            for i in range(100):
                process_chunk(i)
                if i % 10 == 0:
                    cache.touch(on={'job_id': 'long_job'})  # Prevent expiration

        Note:
            - This is a cache-layer operation (TTL-aware)
            - Resets ``created_at`` to now, giving a full config-TTL extension
            - Does not reload or re-serialize data — much faster than get() + put()
            - TTL duration is always determined by the global config
              (``CacheMetadataConfig.default_ttl_seconds``)
        """
        with self._lock:
            cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
            cache_key = self._resolve_cache_key(cache_key, on, kwargs)

            # Get existing entry
            entry = self.metadata_backend.get_entry(cache_key)
            if not entry:
                logger.warning(
                    f"⚠️ Cache entry not found for touch: {cache_key[:16]}..."
                )
                return False

            # Update timestamp to now (resets TTL)
            now = datetime.now(timezone.utc)
            entry["created_at"] = now.isoformat()
            entry["accessed_at"] = now.isoformat()

            # Re-sign if signing is enabled (timestamp is part of signature)
            if self.signer:
                try:
                    metadata = entry.get("metadata", {})
                    complete_entry_data = self._extract_signable_fields(
                        cache_key=cache_key,
                        entry_data=entry,
                        metadata=metadata,
                    )

                    # Generate new signature with updated timestamp
                    new_signature = self.signer.sign_entry(complete_entry_data)
                    metadata["entry_signature"] = new_signature
                    entry["metadata"] = metadata

                    logger.debug(f"Re-signed touched entry {cache_key}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to re-sign touched entry {cache_key}: {e}")
                    # Continue - touch succeeded, just missing signature

            # Store updated entry
            self.metadata_backend.put_entry(cache_key, entry)

            logger.info(f"👆 Touched cache entry: {cache_key[:16]}... (TTL extended)")
            return True

    # ── Bulk & Batch Operations ───────────────────────────────────────────

    def delete_by_prefix(self, prefix: str) -> int:
        """Delete all cache entries whose cache key starts with *prefix*.

        Uses backend-optimized prefix lookup when available (SQL ``LIKE``
        on SQLite/PostgreSQL) and falls back to Python-side filtering for
        the JSON backend.

        Both the metadata entry **and** the corresponding blob file are
        removed for each matching key.

        Args:
            prefix: The cache key prefix to match.  An empty string
                matches everything (equivalent to :meth:`clear_all`).

        Returns:
            int: Number of entries deleted.

        Example::

            deleted = cache.delete_by_prefix("myapp/models/")
        """
        with self._lock:
            keys = self.metadata_backend.keys_by_prefix(prefix)
            deleted = 0
            for key in keys:
                if self._blob_store.delete(key):
                    deleted += 1
            logger.info(
                f"🗑️ Prefix delete: removed {deleted} entries matching '{prefix}*'"
            )
            return deleted

    def delete_where(self, filter_fn: Callable[[Dict[str, Any]], bool]) -> int:
        """
        Delete all cache entries matching a filter function.

        Iterates over every entry and deletes those for which ``filter_fn``
        returns ``True``.  This works with **all** backends.

        Args:
            filter_fn: A callable that receives an entry dict and returns True
                       if the entry should be deleted.  Each dict contains at
                       least ``cache_key``, ``data_type``, ``description``,
                       ``metadata``, ``created``, ``last_accessed``, and
                       ``size_mb``.

        Returns:
            int: Number of entries deleted

        Example:
            # Delete all entries older than 7 days
            from datetime import datetime, timezone, timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            deleted = cache.delete_where(
                lambda e: (e.get("created") or "") < cutoff
            )

            # Delete all DataFrames
            deleted = cache.delete_where(
                lambda e: e.get("data_type") == "dataframe"
            )
        """
        with self._lock:
            summaries = self.metadata_backend.iter_entry_summaries()
            deleted = 0
            for entry in summaries:
                # Add user-facing aliases so filter functions written for
                # list_entries() dicts continue to work
                if "created" not in entry and "created_at" in entry:
                    raw = entry["created_at"]
                    entry["created"] = (
                        raw.isoformat() if hasattr(raw, "isoformat") else raw
                    )
                if "last_accessed" not in entry and "accessed_at" in entry:
                    raw = entry["accessed_at"]
                    entry["last_accessed"] = (
                        raw.isoformat() if hasattr(raw, "isoformat") else raw
                    )
                if "size_mb" not in entry and "file_size" in entry:
                    entry["size_mb"] = round(entry["file_size"] / (1024 * 1024), 3)
                try:
                    if filter_fn(entry):
                        cache_key = entry.get("cache_key")
                        if cache_key:
                            self.invalidate(cache_key=cache_key)
                            deleted += 1
                except Exception as exc:  # intentionally broad — user callback may fail
                    logger.warning(
                        f"filter_fn raised for entry {entry.get('cache_key', '?')}: {exc}"
                    )
            logger.info(f"🗑️ Bulk delete: removed {deleted} entries")
            return deleted

    def delete_matching(self, **kwargs) -> int:
        """
        Delete all cache entries whose metadata contains the given key/value
        pairs.

        This is a convenience wrapper around :meth:`delete_where` that checks
        each entry's metadata dict for matching values.  Works with all
        backends; for SQLite with ``store_full_metadata=True`` it also checks
        the ``metadata_dict`` column via ``query_meta()``.

        Args:
            **kwargs: Key-value pairs to match against entry metadata.
                      An entry is deleted when **all** pairs match.

        Returns:
            int: Number of entries deleted

        Example:
            # Delete all entries for a specific project
            deleted = cache.delete_matching(project="ml_models")

            # Delete all entries for a specific experiment + model type
            deleted = cache.delete_matching(
                experiment="exp_001",
                model_type="xgboost"
            )
        """
        with self._lock:
            if not kwargs:
                return 0

            # Fast path: use query_meta when store_full_metadata is enabled
            # Works across all backends (SQLite uses JSON_EXTRACT, others use Python)
            if self.config.metadata.store_full_metadata:
                results = self.query_meta(**kwargs)
                if results is not None:
                    deleted = 0
                    for entry in results:
                        cache_key = entry.get("cache_key")
                        if cache_key:
                            self.invalidate(cache_key=cache_key)
                            deleted += 1
                    logger.info(
                        f"🗑️ Bulk delete (query_meta): removed {deleted} entries"
                    )
                    return deleted

            # Generic path: scan summaries and match flat fields directly
            summaries = self.metadata_backend.iter_entry_summaries()
            deleted = 0
            for entry in summaries:
                if all(entry.get(k) == v for k, v in kwargs.items()):
                    cache_key = entry.get("cache_key")
                    if cache_key:
                        self.invalidate(cache_key=cache_key)
                        deleted += 1
            logger.info(f"🗑️ Bulk delete (matching): removed {deleted} entries")
            return deleted

    def put_batch(
        self,
        items: List[Tuple[Any, Dict[str, Any]]],
    ) -> int:
        """
        Put multiple cache entries in one call.

        Args:
            items: List of ``(data, kwargs)`` tuples. Each *kwargs* dict
                   accepts the same parameters as :meth:`put` (e.g.
                   ``cache_key``, ``ttl_seconds``, ``description``,
                   ``custom_metadata``, plus any domain kwargs for key
                   generation).

        Returns:
            int: Number of entries that were successfully stored.

        Example:
            stored = cache.put_batch([
                (df_train, {"experiment": "exp_001", "split": "train"}),
                (df_test,  {"experiment": "exp_001", "split": "test"}),
            ])
            print(f"Cached {stored} entries")
        """
        with self._lock:
            stored = 0
            for data, kw in items:
                try:
                    self.put(data, **kw)
                    stored += 1
                except Exception:  # intentionally broad — partial success allowed
                    logger.warning(
                        "📝 Batch put: failed to store entry with kwargs %s",
                        {k: v for k, v in kw.items() if k != "custom_metadata"},
                    )
            logger.info(f"📝 Batch put: cached {stored}/{len(items)} entries")
            return stored

    def get_batch(
        self,
        kwargs_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Get multiple cache entries in one call.

        Args:
            kwargs_list: List of kwarg dicts, each identifying one entry
                         (same parameters you would pass to :meth:`get`).

        Returns:
            dict mapping each generated cache_key to its data (or ``None``
            if not found / expired).

        Example:
            results = cache.get_batch([
                {"experiment": "exp_001"},
                {"experiment": "exp_002"},
                {"experiment": "exp_003"},
            ])
            for key, data in results.items():
                if data is not None:
                    print(f"{key}: loaded")
        """
        with self._lock:
            results: Dict[str, Any] = {}
            for kw in kwargs_list:
                # Strip named params that get() consumes so the cache key
                # matches the one computed during put()
                hash_kwargs = {
                    k: v for k, v in kw.items() if k not in ("cache_key", "ttl_seconds")
                }
                cache_key = kw.get("cache_key") or self._create_cache_key(hash_kwargs)
                results[cache_key] = self.get(**kw)
            return results

    def delete_batch(
        self,
        kwargs_list: List[Dict[str, Any]],
    ) -> int:
        """
        Delete multiple cache entries in one call.

        Args:
            kwargs_list: List of kwarg dicts, each identifying one entry
                         (same parameters you would pass to :meth:`invalidate`).

        Returns:
            int: Number of entries that were actually deleted (existed).

        Example:
            deleted = cache.delete_batch([
                {"experiment": "exp_001"},
                {"experiment": "exp_002"},
            ])
            print(f"Removed {deleted} entries")
        """
        with self._lock:
            deleted = 0
            for kw in kwargs_list:
                # Strip named params that invalidate()/put() consume so the
                # cache key matches the one computed during put()
                hash_kwargs = {
                    k: v
                    for k, v in kw.items()
                    if k not in ("cache_key", "description", "custom_metadata")
                }
                cache_key = kw.get("cache_key") or self._create_cache_key(hash_kwargs)
                entry = self.metadata_backend.get_entry(cache_key)
                if entry is not None:
                    self.invalidate(cache_key=cache_key)
                    deleted += 1
            logger.info(f"🗑️ Batch delete: removed {deleted}/{len(kwargs_list)} entries")
            return deleted

    def touch_batch(self, **filter_kwargs) -> int:
        """
        Touch (refresh TTL of) all cache entries whose metadata matches
        the given key/value pairs.

        Args:
            **filter_kwargs: Key-value pairs to match against entry metadata.

        Returns:
            int: Number of entries touched.

        Example:
            # Extend TTL for all entries in a project
            touched = cache.touch_batch(project="ml_models")
        """
        with self._lock:
            if not filter_kwargs:
                return 0

            summaries = self.metadata_backend.iter_entry_summaries()
            touched = 0
            for entry in summaries:
                # Summaries are already flat — no need to merge with metadata
                if all(entry.get(k) == v for k, v in filter_kwargs.items()):
                    cache_key = entry.get("cache_key")
                    if cache_key and self.touch(cache_key=cache_key):
                        touched += 1
            logger.info(f"👆 Batch touch: refreshed {touched} entries")
            return touched

    def _enforce_size_limit(self):
        """Enforce cache size limits using LRU eviction."""
        max_size_bytes = self.config.storage.max_cache_size_bytes
        if max_size_bytes is None:
            return  # No size limit configured

        # Get current total size from metadata backend
        stats = self.metadata_backend.get_stats()
        total_size_bytes = stats.get("total_size_bytes", 0)

        if total_size_bytes <= max_size_bytes:
            return

        # Use metadata backend's cleanup functionality — clean to 80% of limit
        target_size_bytes = int(max_size_bytes * 0.8)

        result = self.metadata_backend.cleanup_by_size(target_size_bytes)
        removed_count = result.get("count", 0)
        removed_entries = result.get("removed_entries", [])

        # Delete blob files for removed entries
        blobs_deleted = 0
        for entry in removed_entries:
            entry_key = entry.get("cache_key", "unknown")
            self._invoke_hook("on_evict", entry_key, "size_limit")
            actual_path = entry.get("actual_path")
            if actual_path and "://" not in actual_path:
                blob_file = self._resolve_actual_path(actual_path)
                if isinstance(blob_file, Path) and blob_file.exists():
                    try:
                        blob_file.unlink()
                        blobs_deleted += 1
                    except OSError as exc:
                        logger.warning(
                            f"Failed to delete blob file {actual_path} during size enforcement: {exc}"
                        )

        if removed_count > 0:
            logger.info(
                f"Cache size enforcement: removed {removed_count} entries "
                f"(deleted {blobs_deleted} blob files)"
            )

    def invalidate(
        self,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        hash_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Invalidate (remove) specific cache entries.

        Delegates blob file deletion and metadata removal to BlobStore.delete().

        Args:
            cache_key: Direct cache key (if provided, on and **kwargs are ignored)
            hash_key: Alias for cache_key (storage-oriented name).
            on: Dictionary of key parameters for cache key derivation.
                Use this to avoid namespace collisions with cache control parameters.
            **kwargs: Parameters identifying the cached data (legacy, use 'on' instead)
        """
        with self._lock:
            cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
            cache_key = self._resolve_cache_key(cache_key, on, kwargs)

            if self._blob_store.delete(cache_key):
                logger.info(f"Invalidated cache entry {cache_key}")
            else:
                logger.debug(f"Cache entry {cache_key} not found for invalidation")

    def clear_all(self):
        """Clear all cache entries and remove cache files.

        Delegates blob file cleanup and metadata clearing to BlobStore.clear().
        """
        with self._lock:
            removed_count = self._blob_store.clear()
            logger.info(f"Cleared {removed_count} cache entries and cache files")
            return removed_count

    def clear(self):
        """Alias for clear_all() - clear all cache entries and remove cache files.

        This method provides a shorter, more intuitive name for clearing the cache.

        Returns:
            int: The number of cache entries removed.
        """
        return self.clear_all()

    def clear_all_namespaces(self) -> Dict[str, int]:
        """Nuclear option: drop all non-default namespaces and clear the default.

        Removes metadata *and* blob files across every registered namespace.
        Non-default namespaces are fully dropped (tables + blob directory
        removed).  The default namespace has its rows deleted and stats reset
        while its tables/directory are preserved.

        Returns:
            Mapping of ``namespace_id`` → entries removed (``-1`` means the
            namespace was dropped entirely rather than row-cleared).
        """
        import shutil

        with self._lock:
            cache_root = Path(self.config.storage.cache_dir)

            # Discover namespace blob directories *before* metadata drop
            ns_dirs: list[Path] = []
            for ns in self.metadata_backend.list_namespaces():
                if ns.namespace_id != DEFAULT_NAMESPACE:
                    ns_dir = cache_root / ns.namespace_id
                    if ns_dir.is_dir():
                        ns_dirs.append(ns_dir)

            # Clear default namespace blob files
            self._blob_store._clear_blob_files()

            # Drop metadata for all namespaces
            results = self.metadata_backend.clear_all_namespaces()

            # Remove non-default namespace blob directories
            for ns_dir in ns_dirs:
                try:
                    shutil.rmtree(ns_dir)
                except OSError as exc:
                    logger.warning(
                        f"Failed to remove namespace blob dir {ns_dir}: {exc}"
                    )

            logger.info(f"Cleared all namespaces: {results}")
            return results

    def cleanup_expired(self, ttl_seconds: Optional[float] = None) -> int:
        """Remove all expired cache entries and their blob files.

        This method allows on-demand TTL cleanup for long-running applications
        without restarting the cache. Removes both metadata entries and associated
        blob files for entries that have exceeded their TTL.

        Args:
            ttl_seconds: Time-to-live in seconds. If None, uses the configured
                        default_ttl_seconds from config.metadata.default_ttl_seconds.
                        If config also has no default, no cleanup is performed.

        Returns:
            int: The number of expired entries removed.

        Example:
            # Clean up entries older than 1 hour
            removed = cache.cleanup_expired(ttl_seconds=3600)
            print(f"Removed {removed} expired entries")

            # Use configured default TTL
            removed = cache.cleanup_expired()
        """
        import time
        from datetime import datetime

        with self._lock:
            # Determine TTL to use
            if ttl_seconds is None:
                ttl_seconds = self.config.metadata.default_ttl_seconds

            if not ttl_seconds:
                logger.debug("cleanup_expired: no TTL configured, nothing to do")
                return 0

            # Find expired entries by scanning metadata
            cutoff_time = time.time() - ttl_seconds
            expired_entries = []

            for entry in self.metadata_backend.iter_entry_summaries():
                created_at = entry.get("created_at")
                if created_at:
                    # Handle both raw timestamp and ISO format
                    if isinstance(created_at, str):
                        try:
                            created_dt = datetime.fromisoformat(created_at)
                            created_timestamp = created_dt.timestamp()
                        except (ValueError, TypeError):
                            continue
                    else:
                        created_timestamp = created_at

                    if created_timestamp < cutoff_time:
                        expired_entries.append(entry)

            # Delete blob files for expired entries
            blobs_deleted = 0
            for entry in expired_entries:
                entry_key = entry.get("cache_key", "unknown")
                self._invoke_hook("on_evict", entry_key, "expired")
                actual_path = entry.get("actual_path")
                if actual_path and "://" not in actual_path:
                    blob_file = self._resolve_actual_path(actual_path)
                    if isinstance(blob_file, Path) and blob_file.exists():
                        try:
                            blob_file.unlink()
                            blobs_deleted += 1
                        except OSError as exc:
                            logger.warning(
                                f"Failed to delete expired blob file {actual_path}: {exc}"
                            )

            # Remove metadata entries
            removed_count = self.metadata_backend.cleanup_expired(ttl_seconds)

            if removed_count > 0:
                logger.info(
                    f"Cleaned up {removed_count} expired entries "
                    f"(deleted {blobs_deleted} blob files)"
                )

            return removed_count

    def list_entries(self) -> EntryList:
        """List all cache entries with metadata.

        Returns:
            EntryList of entry dicts.  Extends ``list`` so existing
            iteration/indexing code is unaffected.  Adds convenience
            methods: ``.to_dataframe()``, ``.to_json()``, ``.keys()``,
            ``.sort_by()``, ``.filter()``, ``.first()``/``.last()``.
        """
        entries = self.metadata_backend.list_entries()

        # Add expiration status for each entry
        for entry in entries:
            entry["expired"] = self._is_expired(entry["cache_key"])

        return EntryList(entries)

    def close(self):
        """Close all resources (database connections, etc.)."""
        if hasattr(self, "metadata_backend") and self.metadata_backend:
            if hasattr(self.metadata_backend, "close"):
                self.metadata_backend.close()

    def __len__(self) -> int:
        """Return the number of cache entries.

        Enables ``len(cache)`` to check how many entries exist.
        This is a lightweight metadata-only operation.

        Returns:
            int: Total number of cache entries.

        Example:
            cache = UnifiedCache(cache_dir="/tmp/cache")
            cache.put("value", on={"key": "a"})
            assert len(cache) == 1
        """
        with self._lock:
            stats = self.metadata_backend.get_stats()
            return stats.get("total_entries", 0)

    def __contains__(self, cache_key: str) -> bool:
        """Check if a cache key exists (metadata-only, respects TTL).

        Enables ``"my_key" in cache`` syntax.  Delegates to :meth:`exists`
        so expired entries are treated as absent.

        Args:
            cache_key: The cache key to look up.

        Returns:
            bool: True if the entry exists and has not expired.

        Example:
            if "my_key" in cache:
                data = cache.get(cache_key="my_key")
        """
        return self.exists(cache_key=cache_key)

    def __iter__(self):
        """Iterate over entry summaries.

        Enables ``for entry in cache`` to loop over all cached entries.
        Each yielded item is a lightweight summary dict produced by
        :meth:`metadata_backend.iter_entry_summaries`.

        Yields:
            dict: Entry summary dictionaries.

        Example:
            for entry in cache:
                print(entry["cache_key"])
        """
        with self._lock:
            yield from self.metadata_backend.iter_entry_summaries()

    def __del__(self):
        """Ensure resources are cleaned up when the cache is garbage collected."""
        try:
            self.close()
        except Exception:  # intentionally broad — cleanup must not raise
            pass  # Ignore errors during cleanup

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure resources are cleaned up."""
        self.close()
        return False

    # Factory methods for common use cases
    @classmethod
    def for_api(
        cls,
        cache_dir: Optional[str] = None,
        ttl: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        ignore_errors: bool = True,
        **kwargs,
    ) -> "UnifiedCache":
        """
        Create a cache optimized for API requests.

        Defaults:
        - TTL: 6 hours (21600 seconds, good for most API data)
        - ignore_errors: True (don't fail if cache has issues)
        - Compression: LZ4 (fast for JSON/text data)

        Args:
            cache_dir: Cache directory (default: ./cache)
            ttl: TTL as a human-readable duration string (e.g. "6h").
                Mutually exclusive with ``ttl_seconds``.
            ttl_seconds: TTL in seconds (numeric only). Default: 21600 = 6 hours.
                Mutually exclusive with ``ttl``.
            ignore_errors: Continue on cache errors
            **kwargs: Additional config options
        """
        resolved = resolve_ttl(ttl, ttl_seconds, _param_owner="UnifiedCache.for_api")
        ttl_value = resolved if resolved is not None else 21600
        config = create_cache_config(
            cache_dir=cache_dir or "./cache",
            default_ttl_seconds=ttl_value,
            pickle_compression_codec="zstd",  # Fast for JSON/text
            pickle_compression_level=3,
            **kwargs,
        )
        return cls(config)


# Global cache instance for convenience
_global_cache: Optional[UnifiedCache] = None


def get_cache(
    config: Optional[CacheConfig] = None, metadata_backend=None
) -> UnifiedCache:
    """Get the global cache instance, creating it if necessary."""
    global _global_cache
    if _global_cache is None:
        _global_cache = UnifiedCache(config, metadata_backend)
    return _global_cache


def reset_cache(config: Optional[CacheConfig] = None, metadata_backend=None):
    """Reset the global cache instance, properly closing the previous one."""
    global _global_cache
    # Close the existing cache to prevent connection leaks
    if _global_cache is not None:
        try:
            _global_cache.close()
        except Exception:  # intentionally broad — cleanup must not raise
            pass  # Ignore errors during cleanup
    _global_cache = UnifiedCache(config, metadata_backend)
