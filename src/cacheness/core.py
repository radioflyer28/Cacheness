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
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional, Dict, Any, Callable, Tuple

from .config import CacheConfig, _DEFAULT_TTL, create_cache_config
from .entry_list import EntryList
from .handlers import HandlerRegistry
from .metadata import DEFAULT_NAMESPACE
from .serialization import create_unified_cache_key
from .size_utils import format_size, resolve_ttl
from .storage.paths import resolve_actual_path
from ._verification_mixin import VerificationMixin
from ._stats_mixin import StatsMixin
from ._custom_metadata_mixin import CustomMetadataMixin
from ._storage_mode_mixin import StorageModeMixin
from ._query_mixin import QueryMixin
from ._convenience_mixin import ConvenienceMixin
from ._batch_mixin import BatchMixin
from ._file_ops_mixin import FileOpsMixin
from ._get_variants_mixin import GetVariantsMixin
from ._update_mixin import UpdateMixin
from ._put_cleanup import _PutCleanup
from ._inline_blob_mixin import InlineBlobMixin

if TYPE_CHECKING:
    from .interfaces import RotationResult

logger = logging.getLogger(__name__)
_DATETIME_CLASS = datetime


def _parse_datetime_utc(value: Any) -> Optional[datetime]:
    """Parse stored timestamp values and normalize naive datetimes to UTC."""
    if value is None:
        return None
    try:
        if isinstance(value, _DATETIME_CLASS):
            parsed = value
        elif isinstance(value, (int, float)):
            parsed = _DATETIME_CLASS.fromtimestamp(value, tz=timezone.utc)
        else:
            parsed = _DATETIME_CLASS.fromisoformat(str(value))
    except (TypeError, ValueError, OSError):
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


class UnifiedCache(
    VerificationMixin,
    StatsMixin,
    CustomMetadataMixin,
    StorageModeMixin,
    QueryMixin,
    ConvenienceMixin,
    BatchMixin,
    FileOpsMixin,
    GetVariantsMixin,
    UpdateMixin,
    InlineBlobMixin,
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

        # Clean up expired entries on initialization when enabled. Stale write
        # intents are always cleaned conservatively so storage mode can recover
        # uncommitted orphan blobs without deleting committed entries.
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

            # Re-encrypt entries if encryption is enabled
            if (
                hasattr(self, "_blob_store")
                and self._blob_store is not None
                and self._blob_store._encryption_key is not None
            ):
                from .encryption import (
                    decrypt_blob,
                    encrypt_blob,
                    derive_encryption_key,
                )

                old_enc_key = self._blob_store._encryption_key
                new_enc_key = derive_encryption_key(new_key_bytes, self.namespace)

                for entry_summary in entries:
                    cache_key = entry_summary["cache_key"]
                    try:
                        full_entry = self.metadata_backend.get_entry(cache_key)
                        if full_entry is None:
                            continue
                        meta = full_entry.get("metadata", {})
                        if meta.get("encryption_algorithm") is None:
                            continue

                        actual_path_str = meta.get("actual_path")
                        old_iv = bytes.fromhex(meta["encryption_iv"])

                        if actual_path_str:
                            # File-backed blob: read from disk, re-encrypt, write back
                            blob_path = Path(self._resolve_actual_path(actual_path_str))
                            if not blob_path.is_file():
                                continue
                            ciphertext = blob_path.read_bytes()
                            plaintext = decrypt_blob(ciphertext, old_enc_key, old_iv)
                            new_ciphertext, new_iv, _ = encrypt_blob(
                                plaintext, new_enc_key
                            )
                            blob_path.write_bytes(new_ciphertext)

                            meta["encryption_iv"] = new_iv.hex()
                            file_hash = self._calculate_file_hash(blob_path)
                            if file_hash:
                                meta["file_hash"] = file_hash
                                full_entry["file_hash"] = file_hash
                            full_entry["file_size"] = len(new_ciphertext)
                        elif (
                            full_entry.get("is_inline")
                            and full_entry.get("blob_data") is not None
                        ):
                            # Inline blob: decrypt/re-encrypt blob_data in metadata
                            ciphertext = full_entry["blob_data"]
                            plaintext = decrypt_blob(ciphertext, old_enc_key, old_iv)
                            new_ciphertext, new_iv, _ = encrypt_blob(
                                plaintext, new_enc_key
                            )
                            full_entry["blob_data"] = new_ciphertext

                            meta["encryption_iv"] = new_iv.hex()
                            if self.config.metadata.verify_cache_integrity:
                                import xxhash

                                computed_hash = xxhash.xxh3_64(
                                    new_ciphertext
                                ).hexdigest()
                                meta["file_hash"] = computed_hash
                                full_entry["file_hash"] = computed_hash
                            full_entry["file_size"] = len(new_ciphertext)
                        else:
                            # No actual_path and not inline — skip
                            continue
                        full_entry["metadata"] = meta

                        # Re-sign after re-encryption
                        signable = self._extract_signable_fields(
                            cache_key, full_entry, meta
                        )
                        new_sig = new_signer.sign_entry(signable)
                        meta["entry_signature"] = new_sig
                        full_entry["metadata"] = meta

                        self.metadata_backend.put_entry(cache_key, full_entry)
                        result.re_encrypted += 1
                    except (
                        Exception
                    ) as e:  # intentionally broad — best-effort re-encrypt
                        result.failures.append(
                            {"cache_key": cache_key, "error": f"re-encrypt: {e}"}
                        )
                        logger.warning(f"Failed to re-encrypt {cache_key}: {e}")

                self._blob_store._encryption_key = new_enc_key

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

        expires_at = entry.get("expires_at")
        if expires_at:
            expiry_time = _parse_datetime_utc(expires_at)
            if expiry_time is None:
                return True
            return datetime.now(timezone.utc) > expiry_time

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

        creation_time = _parse_datetime_utc(entry["created_at"])
        if creation_time is None:
            return True

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
        removed_count = self.cleanup_expired(ttl_seconds)

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache entries")

    def _cleanup_stale_intents(self):
        """Remove orphaned blobs from stale write intents (crash recovery)."""
        cleaned = self._write_journal.cleanup_stale_intents(
            entry_exists=lambda key: self.metadata_backend.get_entry(key) is not None
        )
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} stale write intents")

    # ── Shared put helpers ────────────────────────────────────────────
    # Extracted from _storage_mode_put() and put() to eliminate duplicated
    # metadata construction, signing, and stale-blob cleanup logic.

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
                    # Propagate encryption metadata from inline path (D-01)
                    if "encryption_algorithm" in direct:
                        metadata_dict["encryption_algorithm"] = direct[
                            "encryption_algorithm"
                        ]
                        metadata_dict["encryption_iv"] = direct["encryption_iv"]
                else:
                    # Delegate file I/O + handler dispatch to BlobStore
                    planned_blob_path = base_file_path.with_suffix(
                        handler.get_file_extension(self.config)
                    )
                    if old_blob_path and "://" not in old_blob_path:
                        old_resolved = self._resolve_actual_path(old_blob_path)
                        if isinstance(old_resolved, Path) and old_resolved.resolve(
                            strict=False
                        ) == planned_blob_path.resolve(strict=False):
                            cleanup.snapshot_previous_blob(old_resolved)
                    self._write_journal.record_intent(
                        cache_key, str(planned_blob_path.relative_to(self.cache_dir))
                    )
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
            if actual_path and "://" in actual_path:
                try:
                    if self._blob_store.blob_backend.delete_blob(actual_path):
                        blobs_deleted += 1
                except Exception as exc:  # backend cleanup must not crash eviction
                    logger.warning(
                        f"Failed to delete remote blob {actual_path} during size enforcement: {exc}"
                    )
            elif actual_path:
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
        with self._lock:
            # Determine TTL to use
            if ttl_seconds is None:
                ttl_seconds = self.config.metadata.default_ttl_seconds

            # Find expired entries by scanning metadata
            now = datetime.now(timezone.utc)
            cutoff_time = (
                now - timedelta(seconds=ttl_seconds)
                if ttl_seconds and ttl_seconds > 0
                else None
            )
            expired_entries = []

            for entry in self.metadata_backend.iter_entry_summaries():
                expires_at = entry.get("expires_at")
                if expires_at:
                    expiry_time = _parse_datetime_utc(expires_at)
                    if expiry_time is None or expiry_time < now:
                        expired_entries.append(entry)
                    continue

                if cutoff_time is None:
                    continue

                created_at = entry.get("created_at")
                if created_at:
                    creation_time = _parse_datetime_utc(created_at)
                    if creation_time is not None and creation_time < cutoff_time:
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
            removed_count = self.metadata_backend.cleanup_expired(ttl_seconds or 0)

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
