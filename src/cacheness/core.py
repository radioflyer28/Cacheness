"""
Simplified Unified Cache with Handler-based Architecture
=======================================================

This module provides a cleaner, more maintainable cache system using the Strategy pattern.
The main UnifiedCache class is now focused on coordination and delegates format-specific
operations to specialized handlers.
"""

import xxhash
import inspect
import threading
import logging
import sys
import uuid
import warnings
from contextlib import contextmanager, nullcontext
from collections.abc import Mapping
from dataclasses import replace
from functools import wraps
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Callable, Tuple

from .config import CacheConfig, _DEFAULT_TTL, create_cache_config
from .error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheIntegrityError,
    CacheLegacyFormatError,
    CacheQueryValidationError,
    CacheReason,
    CacheStorageError,
    CacheUnsafePathError,
)
from .handlers import HandlerRegistry
from .metadata import InMemoryBackend, ProjectionMutationResult
from .serialization import create_unified_cache_key, serialize_for_cache_key
from .storage.blob_store import BlobStore
from .storage.guarded_handler_io import GuardedHandlerIO
from .storage.path_security import encode_physical_name, resolve_managed_locator

logger = logging.getLogger(__name__)


def _facade_operation_lock(instance: Any):
    """Return instance admission when a compatibility test double omits it."""
    lock = getattr(instance, "_lock", None)
    return lock if lock is not None else nullcontext()


def _clear_coordinated(method: Callable) -> Callable:
    """Retain the compatibility decorator without serializing operations."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        # BlobStore owns key and store admission. Holding the historical facade
        # lock here would serialize payload I/O and distinct-key authority work.
        return method(self, *args, **kwargs)

    return wrapped


def _clear_read_coordinated(method: Callable) -> Callable:
    """Retain the compatibility decorator without serializing reads."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        return method(self, *args, **kwargs)

    return wrapped


def _normalize_function_args(func: Callable, args: Tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
    except Exception:
        # Fallback: convert to consistent dict format if signature inspection fails
        param_dict = {}
        
        # Add positional args with generic names
        for i, arg in enumerate(args):
            param_dict[f"__arg_{i}"] = arg
        
        # Add keyword args
        param_dict.update(kwargs)
        
        return param_dict


class UnifiedCache:
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

        self.cache_dir = Path(self.config.storage.cache_dir)
        # Do not materialize the public cache root before the lifecycle
        # authority accepts its topology. On Windows this root is an
        # offline-provisioned trust boundary.
        self.guarded_handler_io = None

        # Thread safety
        self._lock = threading.RLock()

        # Initialize handler registry with config
        self.handlers = HandlerRegistry(self.config)

        self._init_lifecycle_state()
        # Use the same non-mutating authority preflight as a write. It must
        # precede both compatibility metadata initialization and cache-root
        # creation because either can materialize caller-visible state.
        self._cache_blob_store.lifecycle_authority.preflight_mutation()

        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.guarded_handler_io = GuardedHandlerIO(self.cache_dir)

        # SQLite and JSON metadata backends can create files, so defer them
        # until the authority topology has been accepted.
        self._init_metadata_backend(metadata_backend)

        # Initialize custom metadata support
        self._init_custom_metadata_support()

        # Initialize entry signer for metadata integrity
        self._init_entry_signer()
        self._cache_blob_store._before_authority_promotion = (
            self._prepare_authority_projection
        )

        # Clean up expired entries on initialization
        if self.config.storage.cleanup_on_init:
            self._cleanup_expired()

        logger.info(
            f"✅ Unified cache initialized: {self.cache_dir} (backend: {self.actual_backend})"
        )

    def _init_metadata_backend(self, metadata_backend):
        """Initialize the metadata backend."""
        # Import here to avoid circular imports
        from .metadata import create_metadata_backend, SQLALCHEMY_AVAILABLE
        
        actual_backend = "unknown"  # Default value
        if metadata_backend is not None:
            self.metadata_backend = metadata_backend
            actual_backend = "custom"
        else:
            pass  # Will be determined below

        # Determine backend based on config and availability
        if self.config.metadata.metadata_backend == "json":
            # Explicitly requested JSON
            self.metadata_backend = create_metadata_backend(
                "json", 
                metadata_file=self.cache_dir / "cache_metadata.json",
                config=self.config.metadata
            )
            actual_backend = "json"
        elif self.config.metadata.metadata_backend == "memory":
            # Explicitly requested in-memory backend
            self.metadata_backend = create_metadata_backend("memory", config=self.config.metadata)
            actual_backend = "memory"
            logger.info("⚡ Using ultra-fast in-memory backend (no persistence)")
        elif self.config.metadata.metadata_backend == "sqlite":
            # Explicitly requested SQLite
            if not SQLALCHEMY_AVAILABLE:
                raise ImportError(
                    "SQLAlchemy is required for SQLite backend but is not available. Install with: uv add sqlalchemy"
                )
            self.metadata_backend = create_metadata_backend(
                "sqlite",
                db_file=str(self.cache_dir / self.config.metadata.sqlite_db_file),
                config=self.config.metadata
            )
            actual_backend = "sqlite"
        elif self.config.metadata.metadata_backend == "sqlite_memory":
            # Explicitly requested in-memory SQLite
            if not SQLALCHEMY_AVAILABLE:
                raise ImportError(
                    "SQLAlchemy is required for in-memory SQLite backend but is not available. Install with: uv add sqlalchemy"
                )
            self.metadata_backend = create_metadata_backend("sqlite_memory", config=self.config.metadata)
            actual_backend = "sqlite_memory"
            logger.info("⚡ Using in-memory SQLite backend (no persistence)")
        elif self.config.metadata.metadata_backend == "postgresql":
            # Explicitly requested PostgreSQL
            if not SQLALCHEMY_AVAILABLE:
                raise ImportError(
                    "SQLAlchemy is required for PostgreSQL backend but is not available. Install with: uv add sqlalchemy"
                )
            # Get connection URL from options or raise error
            backend_options = self.config.metadata.metadata_backend_options or {}
            connection_url = backend_options.get("connection_url")
            if not connection_url:
                raise ValueError(
                    "PostgreSQL backend requires 'connection_url' in metadata_backend_options"
                )
            self.metadata_backend = create_metadata_backend(
                "postgresql",
                connection_url=connection_url,
                pool_size=backend_options.get("pool_size", 10),
                max_overflow=backend_options.get("max_overflow", 20),
                pool_pre_ping=backend_options.get("pool_pre_ping", True),
                pool_recycle=backend_options.get("pool_recycle", 3600),
                echo=backend_options.get("echo", False),
                table_prefix=backend_options.get("table_prefix", ""),
                config=self.config.metadata
            )
            actual_backend = "postgresql"
        else:
            # Auto mode: prefer SQLite, fallback to JSON
            if SQLALCHEMY_AVAILABLE:
                try:
                    self.metadata_backend = create_metadata_backend(
                        "sqlite",
                        db_file=str(
                            self.cache_dir / self.config.metadata.sqlite_db_file
                        ),
                        config=self.config.metadata
                    )
                    actual_backend = "sqlite"
                    logger.info(
                        "🗄️  Using SQLite backend (auto-selected for better performance)"
                    )
                except Exception as e:
                    logger.warning(f"SQLite backend failed, falling back to JSON: {e}")
                    self.metadata_backend = create_metadata_backend(
                        "json", 
                        metadata_file=self.cache_dir / "cache_metadata.json",
                        config=self.config.metadata
                    )
                    actual_backend = "json"
            else:
                logger.info("📝 SQLModel not available, using JSON backend")
                self.metadata_backend = create_metadata_backend(
                    "json", 
                    metadata_file=self.cache_dir / "cache_metadata.json",
                    config=self.config.metadata
                )
                actual_backend = "json"  # Store the actual backend used for reporting
        self.actual_backend = actual_backend

    def _init_lifecycle_state(self) -> None:
        """Initialize BlobStore as the single payload lifecycle authority."""
        self._lifecycle_state = "ready"
        self._cache_blob_store = BlobStore(
            self.cache_dir / ".cacheness" / "unified-cache-v1",
            # An object backend selects durable SQLite authority without trying
            # to render handler diagnostics as an independent JSON authority.
            backend=InMemoryBackend(),
            config=self.config,
        )
        self._cache_blob_store.handlers = self.handlers
        if self._cache_blob_store.guarded_handler_io is not None:
            self.guarded_handler_io = self._cache_blob_store.guarded_handler_io

    def _supports_custom_metadata(self) -> bool:
        """Check if custom metadata is supported (requires SQLite or PostgreSQL backend with SQLAlchemy)."""
        if not (
            self.actual_backend in ("sqlite", "postgresql")
            and getattr(self, "_custom_metadata_enabled", False)
        ):
            return False
        supports = getattr(self.metadata_backend, "supports_custom_metadata", None)
        # Existing third-party adapters retain their characterized exact-current
        # method surface. Built-in SQL adapters and their cached wrapper expose
        # the explicit capability method above.
        return not callable(supports) or supports()

    def _normalize_custom_metadata(self, custom_metadata):
        """
        Normalize custom_metadata input to a list of metadata objects.
        
        Supports:
        - Single metadata object: custom_metadata=experiment_metadata
        - List of objects: custom_metadata=[experiment_metadata, performance_metadata]
        - Tuple of objects: custom_metadata=(experiment_metadata, performance_metadata)
        - Dictionary (legacy): custom_metadata={"experiments": experiment_metadata}
        """
        if custom_metadata is None:
            return []
        
        # Check if it's a single metadata object (has _schema_name attribute)
        if hasattr(custom_metadata, '_schema_name') or hasattr(type(custom_metadata), '_schema_name'):
            return [custom_metadata]
        
        # Check if it's a list or tuple of metadata objects
        if isinstance(custom_metadata, (list, tuple)):
            return list(custom_metadata)
        
        # Check if it's a dictionary (legacy format)
        if isinstance(custom_metadata, dict):
            return list(custom_metadata.values())
        
        # Invalid format
        raise ValueError(
            f"Invalid custom_metadata format. Expected metadata object, list/tuple of objects, "
            f"or dictionary, got {type(custom_metadata)}"
        )

    def _init_custom_metadata_support(self):
        """Initialize custom metadata support if SQLite or PostgreSQL backend is available."""
        try:
            from .custom_metadata import is_custom_metadata_available

            if is_custom_metadata_available() and self.actual_backend in ("sqlite", "postgresql"):
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
                    custom_fields=self.config.security.custom_signed_fields,
                    use_in_memory_key=self.config.security.use_in_memory_key
                )
                
                logger.info(f"🔒 Entry signing enabled with fields: {self.signer.signed_fields}")
            else:
                self.signer = None
                logger.debug("Entry signing disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize entry signer: {e}")
            self.signer = None

    def _store_custom_metadata(
        self,
        cache_key: str,
        custom_metadata,
        *,
        expected_locator: Optional[str] = None,
    ):
        """Store custom metadata using link table architecture."""
        if not self._supports_custom_metadata():
            logger.warning("Custom metadata not supported - requires SQLite or PostgreSQL backend")
            return

        try:
            metadata_objects = self._normalize_custom_metadata(custom_metadata)
            if not metadata_objects:
                return
            conditional_store = getattr(
                self.metadata_backend, "store_custom_metadata_if_current", None
            )
            if expected_locator is None or not callable(conditional_store):
                logger.warning(
                    "Custom metadata storage requires an exact-current backend protocol"
                )
                return
            conditional_store(cache_key, expected_locator, metadata_objects)
            logger.debug("Stored custom metadata for cache key %s", cache_key)
            return
        except CacheBlobLifecycleConflictError:
            raise
        except Exception as e:
            logger.error(f"Failed to store custom metadata: {e}")

    def _get_custom_metadata(self, cache_key: str) -> Dict[str, Any]:
        """Retrieve custom metadata for a cache key."""
        if not self._supports_custom_metadata():
            return {}

        try:
            from .custom_metadata import CacheMetadataLink
            from sqlalchemy import select

            session_context = getattr(
                self.metadata_backend, "custom_metadata_session", None
            )
            if not callable(session_context):
                logger.warning("Custom metadata query protocol is not available")
                return {}
            with session_context() as session:
                links = (
                    session.execute(
                        select(CacheMetadataLink).where(
                            CacheMetadataLink.cache_key == cache_key
                        )
                    )
                    .scalars()
                    .all()
                )

                result = {}
                for link in links:
                    for schema_name, model_class in self._get_registered_schemas().items():
                        if model_class.__tablename__ == link.metadata_table:
                            metadata_instance = session.execute(
                                select(model_class).where(model_class.id == link.metadata_id)
                            ).scalar_one_or_none()
                            if metadata_instance:
                                result[schema_name] = metadata_instance
                            break
                return result
        except Exception as e:
            logger.error(f"Failed to retrieve custom metadata: {e}")
            return {}

    def _get_registered_schemas(self) -> Dict[str, Any]:
        """Get all registered custom metadata schemas."""
        try:
            from .custom_metadata import get_all_custom_metadata_models

            return get_all_custom_metadata_models()
        except ImportError:
            return {}

    @_clear_read_coordinated
    def query_custom(self, schema_name: str, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Query custom metadata for a specific schema with automatic session cleanup.

        This method provides safe querying with proper session lifecycle management.
        For advanced queries requiring the SQLAlchemy query object directly, use
        the query_custom_session() context manager instead.

        Args:
            schema_name: Name of the custom metadata schema to query
            filters: Optional dict of field_name -> value for equality filtering

        Returns:
            List of results (empty list if not supported or on error)

        Example:
            # Get all entries
            results = cache.query_custom("ml_experiments")
            
            # Filter by field values
            results = cache.query_custom("ml_experiments", {"model_type": "xgboost"})
            
            # For advanced filtering, use the context manager:
            with cache.query_custom_session("ml_experiments") as query:
                high_accuracy = query.filter(MLExperimentMetadata.accuracy >= 0.9).all()
        """
        if not self._supports_custom_metadata():
            logger.warning(
                "Custom metadata querying not supported - requires SQLite or PostgreSQL backend"
            )
            return []

        try:
            from .custom_metadata import CacheMetadataLink, get_custom_metadata_model

            model_class = get_custom_metadata_model(schema_name)
            if not model_class:
                logger.warning(f"Unknown custom metadata schema: {schema_name}")
                return []

            session_context = getattr(
                self.metadata_backend, "custom_metadata_session", None
            )
            if not callable(session_context):
                logger.warning("Custom metadata query protocol is not available")
                return []
            live_keys = self._live_authority_projection_keys()
            if not live_keys:
                return []
            with session_context() as session:
                query = (
                    session.query(model_class)
                    .join(
                        CacheMetadataLink,
                        (CacheMetadataLink.metadata_table == model_class.__tablename__)
                        & (CacheMetadataLink.metadata_id == model_class.id),
                    )
                    .filter(CacheMetadataLink.cache_key.in_(live_keys))
                    .distinct()
                )
                if filters:
                    for field_name, value in filters.items():
                        if hasattr(model_class, field_name):
                            query = query.filter(getattr(model_class, field_name) == value)
                        else:
                            logger.warning(
                                "Unknown filter field '%s' for schema '%s'",
                                field_name,
                                schema_name,
                            )
                return query.all()
        except Exception as e:
            logger.error(f"Failed to query schema {schema_name}: {e}")
            return []

    def query_custom_session(self, schema_name: str):
        """
        Context manager for custom metadata queries with proper session cleanup.
        
        Use this for advanced queries that need direct access to the SQLAlchemy
        query object for complex filtering, ordering, or joining.

        Args:
            schema_name: Name of the custom metadata schema to query

        Yields:
            SQLAlchemy query object for advanced querying

        Raises:
            ValueError: If schema not found or custom metadata not supported

        Example:
            with cache.query_custom_session("ml_experiments") as query:
                # Complex filtering
                high_accuracy = query.filter(
                    MLExperimentMetadata.accuracy >= 0.9,
                    MLExperimentMetadata.model_type == "xgboost"
                ).order_by(MLExperimentMetadata.accuracy.desc()).limit(10).all()
        """
        @contextmanager
        def _session_context():
            if not self._supports_custom_metadata():
                raise ValueError(
                    "Custom metadata querying not supported - requires SQLite or PostgreSQL backend"
                )

            from .custom_metadata import CacheMetadataLink, get_custom_metadata_model

            model_class = get_custom_metadata_model(schema_name)
            if not model_class:
                raise ValueError(f"Unknown custom metadata schema: {schema_name}")

            session_context = getattr(
                self.metadata_backend, "custom_metadata_session", None
            )
            if not callable(session_context):
                raise ValueError("Custom metadata query protocol is not available")

            with session_context() as session:
                live_keys = self._live_authority_projection_keys()
                query = session.query(model_class)
                if not live_keys:
                    yield query.filter(False)
                    return
                yield (
                    query.join(
                        CacheMetadataLink,
                        (CacheMetadataLink.metadata_table == model_class.__tablename__)
                        & (CacheMetadataLink.metadata_id == model_class.id),
                    )
                    .filter(CacheMetadataLink.cache_key.in_(live_keys))
                    .distinct()
                )
        
        return _session_context()

    def query_custom_metadata(self, schema_name: str, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Query custom metadata for a specific schema.
        
        **Deprecated:** Use query_custom() instead for shorter syntax.

        Args:
            schema_name: Name of the custom metadata schema to query
            filters: Optional dict of field_name -> value for equality filtering

        Returns:
            List of results (empty list if not supported or on error)
        """
        logger.warning("query_custom_metadata() is deprecated, use query_custom() instead")
        return self.query_custom(schema_name, filters)

    @_clear_read_coordinated
    def query_meta(self, **filters):
        """
        Query built-in cache metadata using SQLite JSON1 extension.
        
        This method allows querying cache entries based on their stored cache_key_params
        when store_cache_key_params=True is configured.

        Args:
            **filters: Key-value pairs to filter cache entries by their parameters
                      Supports nested dictionary access with dot notation

        Returns:
            List of cache entries matching the filters, or None if not supported

        Example:
            # Configure cache to store parameters
            config = CacheConfig(store_cache_key_params=True)
            cache = cacheness(config)
            
            # Store some data
            cache.put(model, experiment="exp_001", model_type="xgboost", accuracy=0.95)
            cache.put(data, experiment="exp_002", model_type="cnn", accuracy=0.88)
            
            # Query by parameters  
            xgb_experiments = cache.query_meta(model_type="xgboost")
            high_accuracy = cache.query_meta(accuracy=0.9)  # >= comparison
            specific_exp = cache.query_meta(experiment="exp_001")
        """
        try:
            from .query_validation import (
                to_sqlite_json_path,
                validate_query_fields,
                validate_query_numeric_filters,
            )

            validated_fields = validate_query_fields(filters)
            validate_query_numeric_filters(filters)
            sqlite_paths = tuple(
                to_sqlite_json_path(field) for field in validated_fields
            )

            if self.actual_backend != "sqlite":
                logger.warning("query_meta() requires SQLite backend")
                return None

            if not self.config.metadata.store_cache_key_params:
                logger.warning(
                    "query_meta() requires store_cache_key_params=True in cache configuration"
                )
                return None

            if not hasattr(self.metadata_backend, "SessionLocal"):
                logger.warning("SQLAlchemy session not available for query_meta()")
                return None

            from sqlalchemy import Float, and_, bindparam, case, cast, func, or_, select

            from .metadata import CacheEntry
            from .serialization import serialize_for_cache_key

            live_keys = self._live_authority_projection_keys()
            if not live_keys:
                return []
            with self.metadata_backend.SessionLocal() as session:
                query = (
                    select(
                        CacheEntry.cache_key,
                        CacheEntry.description,
                        CacheEntry.data_type,
                        CacheEntry.created_at,
                        CacheEntry.accessed_at,
                        CacheEntry.file_size,
                        CacheEntry.cache_key_params,
                    )
                    .where(CacheEntry.cache_key_params.is_not(None))
                    .where(CacheEntry.cache_key.in_(live_keys))
                    .order_by(CacheEntry.created_at.desc())
                )

                for index, ((_, value), sqlite_path) in enumerate(
                    zip(filters.items(), sqlite_paths)
                ):
                    path_parameter = bindparam(
                        f"query_meta_path_{index}", value=sqlite_path
                    )
                    json_value = func.json_extract(
                        CacheEntry.cache_key_params, path_parameter
                    )
                    value_parameter = f"query_meta_value_{index}"

                    if isinstance(value, bool):
                        query = query.where(
                            json_value
                            == bindparam(
                                value_parameter,
                                value=serialize_for_cache_key(value),
                            )
                        )
                    elif isinstance(value, (int, float)):
                        numeric_type = or_(
                            func.substr(json_value, 1, 4)
                            == bindparam(
                                f"query_meta_int_prefix_{index}", value="int:"
                            ),
                            func.substr(json_value, 1, 6)
                            == bindparam(
                                f"query_meta_float_prefix_{index}", value="float:"
                            ),
                        )
                        numeric_value = func.substr(
                            json_value,
                            func.instr(json_value, ":") + 1,
                        )
                        is_valid_json_number = func.json_type(
                            case(
                                (
                                    func.json_valid(numeric_value)
                                    == bindparam(
                                        f"query_meta_json_valid_{index}", value=1
                                    ),
                                    numeric_value,
                                ),
                                else_=bindparam(
                                    f"query_meta_invalid_json_{index}", value="null"
                                ),
                            )
                        ).in_(("integer", "real"))
                        finite_numeric_value = and_(
                            is_valid_json_number,
                            func.abs(cast(numeric_value, Float))
                            <= bindparam(
                                f"query_meta_max_finite_{index}",
                                value=sys.float_info.max,
                            ),
                        )
                        query = query.where(
                            numeric_type,
                            finite_numeric_value,
                            cast(numeric_value, Float)
                            >= bindparam(value_parameter, value=value),
                        )
                    else:
                        serialized_value = None if value is None else value
                        if value is not None and not (
                            isinstance(value, str)
                            and value.startswith(("str:", "int:", "float:", "bool:"))
                        ):
                            serialized_value = serialize_for_cache_key(value)
                        query = query.where(
                            json_value
                            == bindparam(value_parameter, value=serialized_value)
                        )

                result = session.execute(query)

                # Convert results to dictionaries
                entries = []
                for row in result:
                    entry = {
                        'cache_key': row.cache_key,
                        'description': row.description,
                        'data_type': row.data_type,
                        'created_at': row.created_at.isoformat() if hasattr(row.created_at, 'isoformat') else str(row.created_at),
                        'accessed_at': row.accessed_at.isoformat() if hasattr(row.accessed_at, 'isoformat') else str(row.accessed_at),
                        'file_size': row.file_size,
                    }
                    
                    # Parse cache_key_params JSON
                    if row.cache_key_params:
                        try:
                            from .json_utils import loads as json_loads
                            entry['cache_key_params'] = json_loads(row.cache_key_params)
                        except Exception:
                            entry['cache_key_params'] = {}
                    
                    entries.append(entry)
                
                return entries

        except CacheQueryValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to query metadata: {e}")
            return None

    @_clear_read_coordinated
    def get_custom_metadata_for_entry(
        self, cache_key: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Get custom metadata for a specific cache entry.

        Args:
            cache_key: Direct cache key (if provided, **kwargs are ignored)
            **kwargs: Parameters identifying the cached data (used if cache_key is None)

        Returns:
            Dictionary mapping schema names to metadata instances
        """
        if cache_key is None:
            cache_key = self._create_cache_key(kwargs)

        snapshot, projection = self._authority_snapshot_entry(cache_key)
        if snapshot is None or projection is None:
            return {}
        return self._get_custom_metadata(cache_key)

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

    def _get_cache_file_path(self, cache_key: str, prefix: str = "") -> Path:
        """Get an opaque managed base path without exposing a logical prefix."""
        return self.cache_dir / self._storage_id_for_cache_key(cache_key, prefix)

    @staticmethod
    def _storage_id_for_cache_key(cache_key: str, prefix: str = "") -> str:
        """Translate public cache identity into one backend-safe physical ID."""
        return encode_physical_name(
            cache_key,
            prefix,
            namespace="unified-cache",
        )

    @classmethod
    def _candidate_storage_id_for_cache_key(cls, cache_key: str, prefix: str = "") -> str:
        """Return a private replacement ID that cannot overwrite a live payload."""
        return f"{cls._storage_id_for_cache_key(cache_key, prefix)}-candidate-{uuid.uuid4().hex}"

    def _delete_or_prove_absent(self, locator: Path) -> None:
        """Remove one contained payload or prove it is already absent."""
        deletion_error: Exception | None = None
        try:
            if self.guarded_handler_io.file_ops.delete(locator):
                return
        except Exception as exc:
            deletion_error = exc

        try:
            if not self.guarded_handler_io.file_ops.exists(locator):
                return
        except Exception as exc:
            deletion_error = deletion_error or exc

        raise CacheStorageError(
            "Unable to prove removal of a managed cache payload",
            context={"operation": "put", "locator": str(locator)},
        ) from deletion_error

    def _cleanup_uncommitted_candidate(
        self,
        candidate_locator: Path,
        triggering_error: BaseException,
    ) -> None:
        """Erase an uncommitted candidate or retain both failure contexts."""
        try:
            self._delete_or_prove_absent(candidate_locator)
        except Exception as cleanup_error:
            raise CacheStorageError(
                "Failed to remove an uncommitted cache candidate",
                context={
                    "operation": "put",
                    "locator": str(candidate_locator),
                    "cleanup_error": type(cleanup_error).__name__,
                },
            ) from triggering_error

    def _cleanup_prior_payload(self, previous_locator: Path) -> None:
        """Report post-commit prior cleanup without reviving stale metadata."""
        try:
            self._delete_or_prove_absent(previous_locator)
        except Exception as cleanup_error:
            raise CacheStorageError(
                "Cache metadata committed but prior payload cleanup failed",
                context={"operation": "put", "locator": str(previous_locator)},
            ) from cleanup_error

    def _entry_locator(
        self,
        entry: Dict[str, Any],
        cache_key: str,
        *,
        operation: str,
        prefix: str = "",
    ) -> Path:
        """Validate every supported persisted locator shape before use.

        Metadata backends may surface ``actual_path`` at the entry top level or
        within nested metadata. Both forms are validated before returning the
        historical top-level-first value, preventing a dormant hostile sibling
        field from becoming an unsafe later operation.
        """
        metadata = entry.get("metadata", {})
        nested_path = metadata.get("actual_path") if isinstance(metadata, dict) else None
        locator_values = [entry.get("actual_path"), nested_path]
        validated = []
        for locator in locator_values:
            if locator is not None:
                validated.append(
                    resolve_managed_locator(
                        self.guarded_handler_io.root,
                        locator,
                        operation=operation,
                    )
                )
        if validated:
            return validated[0]

        stored_prefix = entry.get("prefix", prefix)
        return self._get_cache_file_path(cache_key, stored_prefix)

    def _preflight_entries(
        self, entries: List[Dict[str, Any]], *, operation: str
    ) -> None:
        """Validate an entire affected set before exposing or mutating it."""
        for entry in entries:
            self._entry_locator(entry, entry.get("cache_key"), operation=operation)

    def _is_expired(
        self,
        cache_key: str,
        ttl_hours=_DEFAULT_TTL,
        entry: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if cache entry is expired."""
        if entry is None:
            entry = self.metadata_backend.get_entry(cache_key)
        if not entry:
            return True

        # Handle infinite TTL: if ttl_hours is explicitly None, never expire
        if ttl_hours is None:
            return False  # Never expires
        elif ttl_hours is _DEFAULT_TTL:
            ttl = self.config.metadata.default_ttl_hours
        else:
            ttl = ttl_hours

        # Type guard to ensure ttl is numeric
        assert isinstance(ttl, (int, float)), f"TTL must be numeric, got {type(ttl)}"

        creation_time_str = entry["created_at"]

        # Handle timezone-aware datetime strings
        if isinstance(creation_time_str, str):
            creation_time = datetime.fromisoformat(creation_time_str)
        else:
            creation_time = creation_time_str

        # Ensure both datetimes are timezone-aware
        if creation_time.tzinfo is None:
            creation_time = creation_time.replace(tzinfo=timezone.utc)

        expiry_time = creation_time + timedelta(hours=ttl)
        current_time = datetime.now(timezone.utc)

        return current_time > expiry_time

    def _extract_signable_fields(
        self, 
        cache_key: str,
        entry_data: Dict[str, Any], 
        metadata: Dict[str, Any],
        cache_key_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract fields for signing/verification in a consistent manner.
        
        This ensures that the same fields are used during both put() and get()
        to prevent signature mismatches.
        
        Args:
            cache_key: The cache key
            entry_data: The entry data dictionary (data_type, prefix, description, etc.)
            metadata: The metadata dictionary from handler result
            cache_key_params: Optional cache key parameters
            
        Returns:
            Dictionary containing all fields that should be signed
        """
        # Build complete entry data with all signable fields
        signable_data = {
            "cache_key": cache_key,
            "data_type": entry_data.get("data_type"),
            "prefix": entry_data.get("prefix", ""),
            "description": entry_data.get("description", ""),
            "file_size": entry_data.get("file_size", 0),
            "created_at": entry_data.get("created_at"),
            "actual_path": metadata.get("actual_path", ""),
            "file_hash": metadata.get("file_hash"),
            # Include handler-specific metadata fields
            "object_type": metadata.get("object_type"),
            "storage_format": metadata.get("storage_format"),
            "serializer": metadata.get("serializer"),
            "compression_codec": metadata.get("compression_codec"),
        }
        
        # Include cache_key_params if provided
        if cache_key_params is not None:
            signable_data["cache_key_params"] = cache_key_params
            
        return signable_data

    def _calculate_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate XXH3_64 for a caller-owned private snapshot.

        Managed payloads must first be copied through ``GuardedHandlerIO``;
        callers of this helper therefore hash a stable private path rather than
        reopening a metadata-controlled managed locator.
        """
        try:
            hasher = xxhash.xxh3_64()
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate hash for {file_path}: {e}")
            return None

    @staticmethod
    def _is_valid_file_hash(file_hash: Any) -> bool:
        """Return whether ``file_hash`` is one complete XXH3_64 digest."""
        return (
            isinstance(file_hash, str)
            and len(file_hash) == 16
            and all(character in "0123456789abcdef" for character in file_hash)
        )

    def _cleanup_expired(self):
        """Remove expired entries through their observed authority generation."""
        authority_keys = self._cache_blob_store.list()
        if self._recognized_legacy_backend() is None:
            self._preflight_entries(
                self.metadata_backend.list_entries(), operation="cleanup_expired"
            )
            removed_count = 0
            for cache_key in authority_keys:
                snapshot, entry = self._authority_snapshot_entry(cache_key)
                if snapshot is None or entry is None:
                    continue
                if self._is_expired(cache_key, entry=entry):
                    if self._retire_exact_authority_snapshot(cache_key, snapshot):
                        removed_count += 1
            if removed_count > 0:
                logger.info("Cleaned up %s expired cache entries", removed_count)
            return

        try:
            self._preflight_entries(
                self.metadata_backend.list_entries(),
                operation="cleanup_expired",
            )
            ttl = self.config.metadata.default_ttl_hours
            removed_count = self.metadata_backend.cleanup_expired(ttl)
        except CacheLegacyFormatError as exc:
            if not self._is_legacy_read_only_error(exc):
                raise
            warnings.warn(
                "Legacy metadata cleanup is deprecated and remains read-only.",
                DeprecationWarning,
                stacklevel=2,
            )
            return

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache entries")

    def _pending_authority_projection_locators(self) -> dict[str, frozenset[str]]:
        """Return exact published candidate locators that are not yet committed.

        Prepared mutations are lifecycle evidence rather than normal-read
        authority. Candidates no longer publish projections; the locator set
        remains a defensive classification seam for historical residue during
        authority-derived repair.
        """
        locators: dict[str, set[str]] = {}
        for prepared in self._cache_blob_store.lifecycle_authority.pending_mutations():
            locators.setdefault(prepared.spec.key, set()).add(
                str(self._authority_payload_locator(prepared.spec.candidate_locator))
            )
        return {
            cache_key: frozenset(candidate_locators)
            for cache_key, candidate_locators in locators.items()
        }

    def _is_pending_authority_projection(
        self,
        cache_key: str,
        entry: Any,
        pending_locators: dict[str, frozenset[str]] | None = None,
    ) -> bool:
        """Return whether a projection row names a currently prepared candidate."""
        locator = self._projection_locator_from_entry(entry)
        if locator is None:
            return False
        if pending_locators is None:
            pending_locators = self._pending_authority_projection_locators()
        return locator in pending_locators.get(
            cache_key, frozenset()
        )

    def _authority_snapshot_entry(self, cache_key: str):
        """Return a matching authority snapshot and compatibility projection.

        The metadata backend is deliberately a projection for authority-backed
        cache entries.  Refresh it before applying cache policy so a stale
        projection can neither select nor retire a newer payload generation.
        """
        if self._recognized_legacy_backend() is not None:
            # The documented immutable legacy adapters predate LifecycleAuthority
            # and are intentionally read-only compatibility evidence.  They
            # cannot be conditionally repaired without mutating their source.
            return None, self.metadata_backend.get_entry(cache_key)
        for attempt in range(2):
            entry = self.metadata_backend.get_entry(cache_key)
            snapshot, manifest = self._authority_snapshot_manifest(cache_key)
            projection_differs_from_committed = (
                entry is not None
                and (
                    snapshot is None
                    or manifest is None
                    or getattr(manifest, "state", None) != "committed"
                    or not self._projection_matches_authority_snapshot(entry, snapshot)
                )
            )
            if (
                projection_differs_from_committed
                and self._is_pending_authority_projection(cache_key, entry)
            ):
                # A candidate may be public in the compatibility backend while
                # its lifecycle mutation is still prepared. It is never a read
                # authority. Preserve a previous committed generation for a
                # direct read without rewriting the candidate underneath its
                # promoter; without an old generation this remains a miss.
                if (
                    snapshot is not None
                    and manifest is not None
                    and getattr(manifest, "state", None) == "committed"
                ):
                    return snapshot, self._projection_entry_from_manifest(
                        manifest, snapshot
                    )
                return None, None
            if snapshot is None and entry is not None and not self._is_authority_projection(
                entry
            ):
                # Pre-authority compatibility records stay readable through the
                # established migration path.  Only a projection inside the
                # BlobStore-managed root can be an in-flight candidate that
                # must be hidden while authority has not committed it.
                return None, entry
            if (
                snapshot is None
                or manifest is None
                or getattr(manifest, "state", None) != "committed"
            ):
                self._sync_authority_projection(
                    cache_key,
                    observed_entry=entry,
                    snapshot=snapshot,
                    manifest=manifest,
                )
                latest = self._cache_blob_store.lifecycle_authority.read_entry(cache_key)
                if latest == snapshot:
                    return None, None
            else:
                if entry is not None:
                    self._entry_locator(
                        entry,
                        cache_key,
                        operation="authority_projection",
                    )
                if not self._projection_matches_authority_snapshot(entry, snapshot):
                    self._sync_authority_projection(
                        cache_key,
                        observed_entry=entry,
                        snapshot=snapshot,
                        manifest=manifest,
                    )
                projection = self.metadata_backend.get_entry(cache_key)
                latest = self._cache_blob_store.lifecycle_authority.read_entry(cache_key)
                if (
                    latest == snapshot
                    and self._projection_matches_authority_snapshot(projection, snapshot)
                ):
                    return snapshot, projection
            if attempt == 1:
                raise CacheBlobLifecycleConflictError(
                    "Authority and compatibility projection did not stabilize",
                    context={
                        "operation": "authority_projection",
                        "key": cache_key,
                        "attempt": attempt + 1,
                    },
                )
        raise AssertionError("bounded authority projection loop exhausted")

    def _is_authority_projection(self, entry: Any) -> bool:
        """Return whether a row names a payload managed by BlobStore authority."""
        locator = self._projection_locator_from_entry(entry)
        guarded_io = getattr(self._cache_blob_store, "guarded_handler_io", None)
        if locator is None or guarded_io is None:
            return False
        try:
            Path(locator).resolve().relative_to(guarded_io.root.resolve())
        except ValueError:
            return False
        return True

    def _live_authority_projection_keys(self) -> list[str]:
        """Return only keys whose compatibility rows match live authority state.

        Custom metadata has its own table and link rows, so querying it must
        not bypass the lifecycle authority that makes tombstones and abandoned
        candidates invisible everywhere else.  Visiting each key also repairs
        the stale projection with its observed exact locator; that conditional
        operation cannot remove a peer's replacement generation.
        """
        live_keys: list[str] = []
        projections = {
            entry["cache_key"]: entry
            for entry in self.metadata_backend.list_entries()
        }
        pending_locators = self._pending_authority_projection_locators()
        for authority_entry in self._cache_blob_store.lifecycle_authority.list_entries():
            observed_entry = projections.get(authority_entry.key)
            if self._is_pending_authority_projection(
                authority_entry.key, observed_entry, pending_locators
            ):
                # SQL metadata filtering operates on the stored row rather
                # than the authority-derived direct-read fallback above. Omit
                # the key until the candidate promotes so query filters cannot
                # expose an uncommitted generation.
                continue
            if self._projection_matches_authority_snapshot(
                observed_entry, authority_entry
            ):
                manifest = self._cache_blob_store.lifecycle._entry_manifest(
                    authority_entry, allow_tombstone=True
                )
                if manifest.state == "committed":
                    live_keys.append(authority_entry.key)
                continue
            try:
                snapshot, projection = self._authority_snapshot_entry(authority_entry.key)
            except CacheBlobLifecycleConflictError:
                # Aggregate query surfaces cannot safely include a key whose
                # pair changed twice during their bounded observation. Omit it
                # for this snapshot rather than leaking a mixed generation.
                logger.debug(
                    "Omitted unstable authority projection from aggregate query: %s",
                    authority_entry.key,
                )
                continue
            if snapshot is not None and projection is not None:
                live_keys.append(authority_entry.key)
        return live_keys

    @staticmethod
    def _list_projection_timestamp(value: Any) -> Any:
        """Preserve the established timezone-aware timestamp list contract."""
        if not isinstance(value, str):
            return value
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.isoformat()

    def _recognized_legacy_backend(self):
        """Return only one of the exact metadata compatibility adapters."""
        backend = self.metadata_backend
        while backend is not None:
            layout = getattr(backend, "_legacy_layout", None)
            if layout in {
                "json_split_v037",
                "json_split_v038_signed",
                "sqlite_metadata_json_v039",
            }:
                return backend
            wrapped = getattr(backend, "backend", None)
            if wrapped is backend:
                break
            backend = wrapped
        return None

    def _is_legacy_read_only_error(self, exc: CacheLegacyFormatError) -> bool:
        """Identify the only compatibility error that permits no-op bookkeeping."""
        return (
            self._recognized_legacy_backend() is not None
            and exc.context.get("reason") == CacheReason.READ_ONLY_LEGACY_STORE.value
        )

    def _record_successful_read(self, cache_key: str, entry: Dict[str, Any]) -> None:
        """Persist normal reads or record exact legacy reads in process memory only."""
        try:
            self.metadata_backend.update_access_time(cache_key)
            self.metadata_backend.increment_hits()
        except CacheLegacyFormatError as exc:
            if not self._is_legacy_read_only_error(exc):
                raise
            recorder = getattr(self.metadata_backend, "record_legacy_read", None)
            if callable(recorder):
                recorder(cache_key)
            warnings.warn(
                "Legacy metadata access bookkeeping is deprecated and read-only.",
                DeprecationWarning,
                stacklevel=2,
            )

    @staticmethod
    def _plain_projection_value(value: Any) -> Any:
        """Copy handler diagnostics out of immutable mapping wrappers."""
        if isinstance(value, Mapping):
            return {
                str(name): UnifiedCache._plain_projection_value(item)
                for name, item in value.items()
            }
        if isinstance(value, tuple):
            return [UnifiedCache._plain_projection_value(item) for item in value]
        if isinstance(value, list):
            return [UnifiedCache._plain_projection_value(item) for item in value]
        return value

    @staticmethod
    def _canonical_cache_key_params(params: Dict[str, Any]) -> Dict[str, str]:
        """Render public key diagnostics into manifest-safe primitive values.

        Lifecycle manifests are authenticated canonical JSON, but this public
        API accepts values such as ``Path`` instances. The established cache
        key serialization is both JSON-safe and query-compatible.
        """
        return {
            str(name): serialize_for_cache_key(value)
            for name, value in params.items()
        }

    def _authority_payload_locator(self, locator: Path | str) -> Path:
        """Resolve an authority locator to the absolute guarded payload path.

        The manifest stores locators relative to BlobStore's private root.  A
        compatibility projection, on the other hand, is also consumed by the
        legacy facade and must retain the absolute path that its guarded I/O
        boundary expects.  Keeping this conversion in one place avoids
        prepending a relative public ``cache_dir`` twice when callers choose a
        relative cache root.
        """
        guarded_io = self._cache_blob_store.guarded_handler_io
        if guarded_io is None:
            raise CacheStorageError(
                "Authority payload root is unavailable for compatibility projection",
                context={"operation": "authority_projection"},
            )
        return resolve_managed_locator(
            guarded_io.root,
            locator,
            operation="authority_projection",
        )

    @staticmethod
    def _projection_created_at(created_at: str) -> str:
        """Match the stable naive SQLite timestamp representation used in HMACs."""
        parsed = datetime.fromisoformat(created_at)
        return parsed.replace(tzinfo=None).isoformat()

    def _projection_matches_authority_snapshot(self, entry: Any, snapshot: Any) -> bool:
        """Return whether a projection names the exact observed generation.

        Older metadata adapters do not retain arbitrary projection fields such
        as ``authority_generation``. The managed payload locator is therefore
        also a durable generation identity. Requiring it prevents a projection
        refresh from overwriting an intentional integrity-test edit merely
        because that adapter omitted the auxiliary generation field.
        """
        if not isinstance(entry, dict):
            return False
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            return False
        actual_path = metadata.get("actual_path")
        if not isinstance(actual_path, str):
            return False
        expected_path = self._authority_payload_locator(snapshot.locator)
        if Path(actual_path) != expected_path:
            return False
        recorded_generation = metadata.get("authority_generation")
        return recorded_generation is None or recorded_generation == snapshot.generation

    def _authority_snapshot_manifest(self, cache_key: str):
        """Read and authenticate one coherent authority observation.

        Projection contents must come from the same ``EntrySnapshot``. Mixing
        a public metadata read with a later authority read can stamp an old
        locator with a newer generation during an overwrite race.
        """
        snapshot = self._cache_blob_store.lifecycle_authority.read_entry(cache_key)
        if snapshot is None:
            return None, None
        manifest = self._cache_blob_store.lifecycle._entry_manifest(
            snapshot, allow_tombstone=True
        )
        return snapshot, manifest

    def _prepare_authority_projection(self, manifest, _put_result):
        """Build signing metadata without publishing compatibility state.

        The lifecycle engine invokes this hook before it has made a candidate
        visible through authority.  It may calculate integrity/signing fields
        for the immutable manifest, but it must not mutate a projection or
        retire custom-metadata links that still belong to M1.
        """
        payload_path = self._authority_payload_locator(manifest.locator)
        metadata = self._plain_projection_value(
            {**dict(manifest.handler_metadata), **dict(manifest.user_metadata)}
        )
        metadata["actual_path"] = str(payload_path)
        file_hash = None
        if self.config.metadata.verify_cache_integrity:
            file_hash = self._calculate_file_hash(payload_path)
            if not self._is_valid_file_hash(file_hash):
                raise CacheIntegrityError(
                    "Unable to calculate a complete payload integrity digest"
                )
        metadata["file_hash"] = file_hash
        metadata["authority_generation"] = manifest.generation
        entry_data = {
            "data_type": manifest.handler_type,
            "prefix": metadata.get("prefix", ""),
            "description": metadata.get("description", ""),
            "file_size": manifest.byte_size,
            "created_at": self._projection_created_at(manifest.created_at),
            "metadata": metadata,
            # Manifest cache-key diagnostics have already gone through the
            # historical serializer. SQLite must not serialize those strings
            # a second time (``str:str:value``).
            "_cache_key_params_serialized": True,
        }
        signing_required = (
            self.config.security.enable_entry_signing
            and not self.config.security.allow_unsigned_entries
        )
        if self.config.security.enable_entry_signing:
            try:
                if self.signer is None:
                    raise RuntimeError("Entry signer is unavailable")
                signature = self.signer.sign_entry(
                    self._extract_signable_fields(
                        cache_key=manifest.key,
                        entry_data=entry_data,
                        metadata=metadata,
                        cache_key_params=metadata.get("cache_key_params"),
                    )
                )
                if not isinstance(signature, str) or not signature:
                    raise ValueError("Entry signer returned an empty signature")
                metadata["entry_signature"] = signature
            except Exception as exc:
                if signing_required:
                    raise CacheIntegrityError(
                        "Unable to sign cache entry while unsigned entries are disabled"
                    ) from exc
        return replace(
            manifest,
            user_metadata={
                **dict(manifest.user_metadata),
                "file_hash": file_hash,
                "authority_generation": manifest.generation,
                **(
                    {"entry_signature": metadata["entry_signature"]}
                    if "entry_signature" in metadata
                    else {}
                ),
            },
        )

    def _publish_promoted_authority_projection(self, put_result) -> str:
        """Replace M1's projection only when this exact M2 remains authority.

        The pre-operation locator belongs to the caller and is never refreshed
        from an observed projection.  In particular, a racing candidate cannot
        lend its token to a losing operation.  SQL backends retire M1 links in
        the same successful exact-token transaction as this replacement.
        """
        promoted = put_result.promoted
        if promoted is None:
            raise CacheBlobLifecycleConflictError(
                "Authority put did not return a promoted generation",
                context={"operation": "projection_publish", "key": put_result.key},
            )
        snapshot, manifest = self._authority_snapshot_manifest(put_result.key)
        if (
            snapshot != promoted
            or manifest is None
            or manifest.state != "committed"
            or manifest.locator != promoted.locator
        ):
            raise CacheBlobLifecycleConflictError(
                "Authority advanced before compatibility projection publication",
                context={"operation": "projection_publish", "key": put_result.key},
            )
        outcome = self._conditional_projection_mutation(
            put_result.key,
            expected_locator=put_result.expected_projection_locator,
            replacement=self._projection_entry_from_manifest(manifest, snapshot),
        )
        if outcome.status == "mismatch":
            # A concurrent reader can repair the exact promoted generation
            # before its writer publishes the compatibility row. Revalidate
            # both sides before treating that as safe convergence: an M3
            # promotion must still reject this M2 publisher rather than lend
            # it a newer candidate token.
            current_snapshot, _current_manifest = self._authority_snapshot_manifest(
                put_result.key
            )
            current_projection = self.metadata_backend.get_entry(put_result.key)
            if (
                current_snapshot == promoted
                and self._projection_matches_authority_snapshot(
                    current_projection, current_snapshot
                )
            ):
                return str(self._authority_payload_locator(promoted.locator))
            raise CacheBlobLifecycleConflictError(
                "Compatibility projection changed before promoted publication",
                context={"operation": "projection_publish", "key": put_result.key},
            )
        return str(self._authority_payload_locator(promoted.locator))

    @staticmethod
    def _projection_locator_from_entry(entry: Any) -> Optional[str]:
        """Return the immutable projection identity from a public metadata row."""
        if not isinstance(entry, dict):
            return None
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            return None
        actual_path = metadata.get("actual_path")
        return actual_path if isinstance(actual_path, str) else None

    def _conditional_projection_mutation(
        self,
        cache_key: str,
        *,
        expected_locator: Optional[str],
        replacement: Optional[Dict[str, Any]],
    ) -> ProjectionMutationResult:
        """Delegate an exact projection transition without key-only teardown."""
        mutation = getattr(self.metadata_backend, "conditional_projection_mutation", None)
        if not callable(mutation):
            raise CacheBlobBackendError(
                "Metadata backend cannot conditionally mutate compatibility projections",
                context={
                    "operation": "projection_mutation",
                    "key": cache_key,
                    "capability": "conditional_projection_mutation",
                },
                reason=CacheReason.BLOB_BACKEND_CAPABILITY_UNSUPPORTED,
            )
        return mutation(
            cache_key,
            expected_locator=expected_locator,
            replacement=replacement,
        )

    def _projection_entry_from_manifest(self, manifest, snapshot) -> Dict[str, Any]:
        """Render one committed authority observation as compatibility metadata."""
        metadata = self._plain_projection_value(
            {**dict(manifest.handler_metadata), **dict(manifest.user_metadata)}
        )
        metadata["actual_path"] = str(self._authority_payload_locator(manifest.locator))
        metadata["authority_generation"] = snapshot.generation
        return {
            "data_type": manifest.handler_type,
            "prefix": metadata.get("prefix", ""),
            "description": metadata.get("description", ""),
            "file_size": manifest.byte_size,
            "created_at": self._projection_created_at(manifest.created_at),
            "metadata": metadata,
            "_cache_key_params_serialized": True,
        }

    def _sync_authority_projection(
        self,
        cache_key: str,
        *,
        observed_entry: Any = None,
        snapshot: Any = None,
        manifest: Any = None,
    ) -> ProjectionMutationResult:
        """Repair a compatibility projection from a committed BlobStore entry."""
        if observed_entry is None:
            observed_entry = self.metadata_backend.get_entry(cache_key)
        if snapshot is None and manifest is None:
            snapshot, manifest = self._authority_snapshot_manifest(cache_key)
        expected_locator = self._projection_locator_from_entry(observed_entry)
        if (
            snapshot is None
            or manifest is None
            or getattr(manifest, "state", None) != "committed"
        ):
            return self._conditional_projection_mutation(
                cache_key, expected_locator=expected_locator, replacement=None
            )
        return self._conditional_projection_mutation(
            cache_key,
            expected_locator=expected_locator,
            replacement=self._projection_entry_from_manifest(manifest, snapshot),
        )

    def _repair_projection_after_failed_put(self, cache_key: str) -> None:
        """Restore only a projection that no longer names committed authority.

        A failure before the projection hook leaves the old projection intact.
        Rewriting it would change public access timestamps despite no committed
        mutation. A failure after the hook instead leaves a candidate locator
        visible, and must be repaired from the authority immediately.
        """
        entry = self.metadata_backend.get_entry(cache_key)
        snapshot, manifest = self._authority_snapshot_manifest(cache_key)
        if snapshot is None or manifest is None or manifest.state != "committed":
            self._sync_authority_projection(
                cache_key, observed_entry=entry, snapshot=snapshot, manifest=manifest
            )
            return
        if not self._projection_matches_authority_snapshot(entry, snapshot):
            self._sync_authority_projection(
                cache_key, observed_entry=entry, snapshot=snapshot, manifest=manifest
            )

    @_clear_coordinated
    def put(
        self,
        data: Any,
        prefix: str = "",
        description: str = "",
        custom_metadata=None,
        **kwargs,
    ):
        """Store data through BlobStore and retain metadata as a projection."""
        cache_key = self._create_cache_key(kwargs)
        # Public callers may replace the compatibility handler registry after
        # construction.  Keep the authority path on that same registry rather
        # than selecting a second serializer policy.
        self._cache_blob_store.handlers = self.handlers
        pre_operation_projection = self.metadata_backend.get_entry(cache_key)
        if pre_operation_projection is not None:
            self._entry_locator(
                pre_operation_projection,
                cache_key,
                operation="put",
                prefix=prefix,
            )
        expected_projection_locator = self._projection_locator_from_entry(
            pre_operation_projection
        )
        # Hold one facade-level admission reference through every public
        # consequence of the authority mutation.  ``_put_with_result`` sees
        # this existing reference and deliberately does not double-count it.
        # This prevents close from releasing authority resources between the
        # immutable promotion and the compatibility policy work below.
        with self._cache_blob_store._instance_admission.operation():
            try:
                put_result = self._cache_blob_store._put_with_result(
                    data,
                    key=cache_key,
                    metadata={
                        "prefix": prefix,
                        "description": description,
                        **(
                            {
                                "cache_key_params": self._canonical_cache_key_params(
                                    kwargs
                                )
                            }
                            if self.config.metadata.store_cache_key_params
                            else {}
                        ),
                    },
                    projection_context=expected_projection_locator,
                )
                promoted_locator = self._publish_promoted_authority_projection(
                    put_result
                )
                self._cache_blob_store._settle_put_cleanup(put_result)
            except BaseException:
                # Pre-promotion failures leave M1 untouched. A failure after
                # promotion may leave a derived row stale, so reconciliation is
                # authority-derived and never reconstructs/destructs a candidate.
                try:
                    self._repair_projection_after_failed_put(cache_key)
                except Exception:
                    logger.exception(
                        "Unable to repair compatibility projection after failed put: %s",
                        cache_key,
                    )
                raise
            self.guarded_handler_io = self._cache_blob_store.guarded_handler_io
            if custom_metadata and self._supports_custom_metadata():
                self._store_custom_metadata(
                    cache_key,
                    custom_metadata,
                    expected_locator=promoted_locator,
                )
            self._enforce_size_limit()
            return cache_key

    def _put_legacy(
        self,
        data: Any,
        prefix: str = "",
        description: str = "",
        custom_metadata=None,
        **kwargs,
    ):
        """
        Store any supported data type in cache.

        Args:
            data: Data to cache (DataFrame, array, or general object)
            prefix: Descriptive prefix prepended to the cache filename
            description: Human-readable description
            custom_metadata: Custom metadata for the cache entry. Supports:
                           - Single metadata object: experiment_metadata
                           - List of objects: [experiment_metadata, performance_metadata]
                           - Tuple of objects: (experiment_metadata, performance_metadata)
                           - Dictionary (legacy): {"experiments": experiment_metadata}
            **kwargs: Parameters identifying this data
        """
        # Get appropriate handler
        handler = self.handlers.get_handler(data)
        cache_key = self._create_cache_key(kwargs)
        storage_id = self._candidate_storage_id_for_cache_key(cache_key, prefix)

        # A put can overwrite an entry or trigger the backend's conservative
        # size cleanup. Validate the complete visible set before publishing a
        # byte or writing replacement metadata.
        self._preflight_entries(
            self.metadata_backend.list_entries(),
            operation="put",
        )
        previous_entry = self.metadata_backend.get_entry(cache_key)
        previous_locator = (
            self._entry_locator(previous_entry, cache_key, operation="put")
            if previous_entry is not None
            else None
        )

        candidate_locator: Path | None = None
        metadata_committed = False
        try:
            # The candidate remains private throughout serialization, snapshot
            # validation, integrity hashing, signing, and metadata publication.
            result = self.guarded_handler_io.put(
                handler,
                data,
                storage_id,
                self.config,
            )
            candidate_locator = resolve_managed_locator(
                self.guarded_handler_io.root,
                result["actual_path"],
                operation="candidate_publish",
            )

            file_hash = None
            if self.config.metadata.verify_cache_integrity:
                with self.guarded_handler_io.open_snapshot(
                    candidate_locator,
                    {},
                ) as snapshot:
                    file_hash = self._calculate_file_hash(snapshot.path)
                if not self._is_valid_file_hash(file_hash):
                    raise CacheIntegrityError(
                        "Unable to calculate a complete payload integrity digest"
                    )

            metadata_dict = {
                **result["metadata"],
                "prefix": prefix,
                "actual_path": str(candidate_locator),
                "file_hash": file_hash,
            }
            if self.config.metadata.store_cache_key_params:
                metadata_dict["cache_key_params"] = kwargs

            entry_data = {
                "data_type": handler.data_type,
                "prefix": prefix,
                "description": description,
                "file_size": result["file_size"],
                "metadata": metadata_dict,
            }

            signing_required = (
                self.config.security.enable_entry_signing
                and not self.config.security.allow_unsigned_entries
            )
            if self.config.security.enable_entry_signing:
                try:
                    if self.signer is None:
                        raise RuntimeError("Entry signer is unavailable")

                    creation_timestamp = datetime.now(timezone.utc)
                    entry_data["created_at"] = creation_timestamp.replace(
                        tzinfo=None
                    ).isoformat()
                    cache_key_params = (
                        kwargs if self.config.metadata.store_cache_key_params else None
                    )
                    complete_entry_data = self._extract_signable_fields(
                        cache_key=cache_key,
                        entry_data=entry_data,
                        metadata=metadata_dict,
                        cache_key_params=cache_key_params,
                    )
                    signature = self.signer.sign_entry(complete_entry_data)
                    if not isinstance(signature, str) or not signature:
                        raise ValueError("Entry signer returned an empty signature")
                    metadata_dict["entry_signature"] = signature
                    logger.debug(f"Created signature for entry {cache_key}")
                except Exception as exc:
                    logger.warning(f"Failed to sign entry {cache_key}: {exc}")
                    if signing_required:
                        raise CacheIntegrityError(
                            "Unable to sign cache entry while unsigned entries are disabled"
                        ) from exc
                    logger.warning(
                        "Committing unsigned entry because allow_unsigned_entries=True"
                    )

            self.metadata_backend.put_entry(cache_key, entry_data)
            metadata_committed = True
        except BaseException as exc:
            if candidate_locator is not None and not metadata_committed:
                self._cleanup_uncommitted_candidate(candidate_locator, exc)
            raise

        # Publication changes authority exactly once. The superseded payload is
        # now cleanup-only and cannot be used to roll metadata back.
        if previous_locator is not None and previous_locator != candidate_locator:
            self._cleanup_prior_payload(previous_locator)

        if custom_metadata and self._supports_custom_metadata():
            self._store_custom_metadata(cache_key, custom_metadata)

        self._enforce_size_limit()

        file_size_mb = result["file_size"] / (1024 * 1024)
        format_info = f"({result['storage_format']} format)"
        logger.info(
            f"Cached {handler.data_type} {cache_key} ({file_size_mb:.3f}MB) {format_info}: {description}"
        )
        return cache_key

    def _verify_legacy_entry_signature(
        self,
        cache_key: str,
        entry: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """Verify only the normalized 0.3.8 six-field signature payload."""
        if metadata.get("legacy_compat_layout") != "json_split_v038_signed":
            return False

        stored_signature = metadata.get("legacy_entry_signature")
        if not isinstance(stored_signature, str) or self.signer is None:
            return False

        from .security import verify_legacy_v038_entry

        verified = verify_legacy_v038_entry(
            self.signer.secret_key,
            {
                "cache_key": cache_key,
                "created_at": entry.get("created_at"),
                "data_type": entry.get("data_type"),
                "file_hash": metadata.get("file_hash"),
                "file_size": entry.get("file_size"),
                "prefix": entry.get("prefix"),
            },
            stored_signature,
        )
        if verified:
            warnings.warn(
                "Reading a legacy 0.3.8 signed cache entry is deprecated.",
                DeprecationWarning,
                stacklevel=3,
            )
        return verified

    @staticmethod
    def _is_exact_legacy_signature_entry(metadata: Dict[str, Any]) -> bool:
        """Identify the sole historical signature failure that must propagate."""
        return (
            metadata.get("legacy_compat_layout") == "json_split_v038_signed"
            and isinstance(metadata.get("legacy_entry_signature"), str)
        )

    def _legacy_decorator_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Rebase the one historical decorator payload without changing evidence."""
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            raise CacheLegacyFormatError(
                "Unsupported legacy decorator metadata",
                reason=CacheReason.UNSUPPORTED_LEGACY_LAYOUT,
            )
        actual_path = metadata.get("actual_path")
        if not isinstance(actual_path, str):
            raise CacheLegacyFormatError(
                "Unsupported legacy decorator payload locator",
                reason=CacheReason.UNSUPPORTED_LEGACY_LAYOUT,
            )

        payload_name = Path(actual_path).name
        if payload_name in {"", ".", ".."}:
            raise CacheLegacyFormatError(
                "Unsupported legacy decorator payload locator",
                reason=CacheReason.UNSUPPORTED_LEGACY_LAYOUT,
            )

        compatibility_entry = entry.copy()
        compatibility_metadata = metadata.copy()
        compatibility_metadata["actual_path"] = str(self.cache_dir / payload_name)
        compatibility_entry["metadata"] = compatibility_metadata
        return compatibility_entry

    def _is_signature_authorized(
        self,
        cache_key: str,
        entry: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """Verify current/legacy signatures before handler deserialization."""
        if not (self.signer and self.config.security.enable_entry_signing):
            return True

        if (
            metadata.get("legacy_entry_signature") is not None
            and not self._is_exact_legacy_signature_entry(metadata)
        ):
            return False

        stored_signature = metadata.get("entry_signature")
        if stored_signature is not None:
            cache_key_params = metadata.get("cache_key_params")
            verify_entry_data = self._extract_signable_fields(
                cache_key=cache_key,
                entry_data=entry,
                metadata=metadata,
                cache_key_params=cache_key_params,
            )
            if self.signer.verify_entry(verify_entry_data, stored_signature):
                return True
            if not self._is_exact_legacy_signature_entry(metadata):
                return False

        if self._is_exact_legacy_signature_entry(metadata):
            return self._verify_legacy_entry_signature(cache_key, entry, metadata)

        return self.config.security.allow_unsigned_entries

    def _retire_exact_authority_snapshot(self, cache_key: str, snapshot: Any) -> bool:
        """Delete only a cache generation still equal to this read observation."""
        if snapshot is None:
            if self._recognized_legacy_backend() is not None:
                entry = self.metadata_backend.get_entry(cache_key)
                if entry is None:
                    return False
                self._entry_locator(entry, cache_key, operation="retire_legacy")
                self.metadata_backend.remove_entry(cache_key)
                return True
            expected = self._cache_blob_store.lifecycle_authority.read_expectation(
                cache_key
            )
            try:
                deleted = self._cache_blob_store.delete(cache_key, expected=expected)
            except CacheBlobLifecycleConflictError:
                logger.info("Preserved replacement generation during stale cleanup: %s", cache_key)
                self._sync_authority_projection(cache_key)
                return False
            entry = self.metadata_backend.get_entry(cache_key)
            if entry is not None:
                self._entry_locator(entry, cache_key, operation="retire_authority")
            self._sync_authority_projection(cache_key, observed_entry=entry)
            return deleted
        try:
            deleted = self._cache_blob_store.delete(
                cache_key, expected=snapshot.expectation
            )
        except CacheBlobLifecycleConflictError:
            logger.info("Preserved replacement generation during stale cleanup: %s", cache_key)
            self._sync_authority_projection(cache_key)
            return False
        self._sync_authority_projection(cache_key)
        return deleted

    def _reject_untrusted_entry(
        self, cache_key: str, *, reason: str, snapshot: Any = None
    ) -> None:
        """Record a safe miss and optionally remove verified-bad evidence."""
        logger.warning("Cache entry %s was rejected before deserialization: %s", cache_key, reason)
        if self.config.security.delete_invalid_signatures:
            self._retire_exact_authority_snapshot(cache_key, snapshot)
        self.metadata_backend.increment_misses()

    @_clear_read_coordinated
    def get(
        self,
        cache_key: Optional[str] = None,
        ttl_hours: Optional[int] = None,
        prefix: str = "",
        _legacy_decorator_v0313: bool = False,
        **kwargs,
    ) -> Optional[Any]:
        """Retrieve a cached value only after guarded snapshot verification."""
        if cache_key is None:
            cache_key = self._create_cache_key(kwargs)
        self._cache_blob_store.handlers = self.handlers

        authority_snapshot, entry = self._authority_snapshot_entry(cache_key)
        if not entry:
            if not _legacy_decorator_v0313:
                self.metadata_backend.increment_misses()
            return None

        if _legacy_decorator_v0313:
            entry = self._legacy_decorator_entry(entry)

        # Validate before expiration checks, miss accounting, handler lookup, or
        # any cleanup policy can alter unsafe evidence.
        file_path = self._entry_locator(
            entry,
            cache_key,
            operation="get",
            prefix=prefix,
        )
        is_expired = (
            self._is_expired(cache_key, ttl_hours, entry)
            if _legacy_decorator_v0313
            else self._is_expired(cache_key, ttl_hours)
        )
        if is_expired:
            if authority_snapshot is not None:
                self._retire_exact_authority_snapshot(cache_key, authority_snapshot)
            if not _legacy_decorator_v0313:
                self.metadata_backend.increment_misses()
            return None

        data_type = entry.get("data_type")
        if not data_type:
            if not _legacy_decorator_v0313:
                self.metadata_backend.increment_misses()
            return None

        metadata = entry.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        try:
            handler = self.handlers.get_handler_by_type(data_type)
            with self.guarded_handler_io.open_snapshot(file_path, metadata) as snapshot:
                if self.config.metadata.verify_cache_integrity:
                    stored_hash = metadata.get("file_hash")
                    if not self._is_valid_file_hash(stored_hash):
                        self._reject_untrusted_entry(
                            cache_key,
                            reason="missing or malformed payload integrity digest",
                            snapshot=authority_snapshot,
                        )
                        return None
                    current_hash = self._calculate_file_hash(snapshot.path)
                    if current_hash != stored_hash:
                        self._reject_untrusted_entry(
                            cache_key,
                            reason="payload integrity hash mismatch",
                            snapshot=authority_snapshot,
                        )
                        return None

                if not self._is_signature_authorized(cache_key, entry, metadata):
                    if self._is_exact_legacy_signature_entry(metadata):
                        raise CacheLegacyFormatError(
                            "Invalid legacy entry signature",
                            reason=CacheReason.INVALID_LEGACY_SIGNATURE,
                        )
                    self._reject_untrusted_entry(
                        cache_key,
                        reason="missing or invalid entry signature",
                        snapshot=authority_snapshot,
                    )
                    return None

                data = (
                    self._cache_blob_store.get(cache_key)
                    if authority_snapshot is not None
                    else handler.get(snapshot.path, snapshot.metadata)
                )

        except CacheUnsafePathError:
            # No unsafe locator becomes a miss, cleanup, or evidence mutation.
            raise
        except FileNotFoundError as e:
            logger.warning(f"Cache file missing for {cache_key}: {e}")
            self._retire_exact_authority_snapshot(cache_key, authority_snapshot)
            self.metadata_backend.increment_misses()
            return None
        except (OSError, IOError) as e:
            logger.warning(f"I/O error loading cached {data_type} {cache_key}: {e}")
            self._retire_exact_authority_snapshot(cache_key, authority_snapshot)
            self.metadata_backend.increment_misses()
            return None
        except CacheLegacyFormatError:
            # Typed compatibility failures must never be converted into mutation.
            raise
        except Exception as e:
            logger.warning(
                f"Failed to load cached {data_type} {cache_key}: {type(e).__name__}: {e}"
            )
            self._retire_exact_authority_snapshot(cache_key, authority_snapshot)
            self.metadata_backend.increment_misses()
            return None

        if not _legacy_decorator_v0313:
            self._record_successful_read(cache_key, entry)
        logger.debug(f"Cache hit ({data_type}): {cache_key}")
        return data

    def _enforce_size_limit(self):
        """Enforce size limits without deleting a generation by key alone."""
        authority_keys = self._cache_blob_store.list()
        if self._recognized_legacy_backend() is None:
            self._preflight_entries(
                self.metadata_backend.list_entries(), operation="cleanup_by_size"
            )
            if not authority_keys:
                return
            max_size_mb = self.config.storage.max_cache_size_mb
            projected_size_mb = self.metadata_backend.get_stats().get(
                "total_size_mb", 0
            )
            # Authority promotion is preceded by an exact projection write, so
            # the raw compatibility size is a safe fast-path upper bound. Only
            # a possible limit breach needs the expensive coherent per-key
            # scan below; scanning after every distinct-key put serializes the
            # otherwise independent lifecycle through metadata I/O.
            if (
                isinstance(projected_size_mb, (int, float))
                and projected_size_mb <= max_size_mb
            ):
                return
            candidates = []
            total_size_bytes = 0
            for cache_key in authority_keys:
                try:
                    snapshot, entry = self._authority_snapshot_entry(cache_key)
                except CacheBlobLifecycleConflictError:
                    # Size enforcement observes an aggregate snapshot. A
                    # different key may still be publishing its projection,
                    # so omit that unstable candidate instead of failing an
                    # otherwise independent put or selecting a mixed row.
                    logger.debug(
                        "Omitted unstable authority projection from size enforcement: %s",
                        cache_key,
                    )
                    continue
                if snapshot is None or entry is None:
                    continue
                file_size = entry.get("file_size", 0)
                size = file_size if isinstance(file_size, int) and file_size > 0 else 0
                total_size_bytes += size
                candidates.append(
                    (
                        entry.get("created_at", ""),
                        cache_key,
                        snapshot,
                        size,
                    )
                )

            max_size_bytes = int(max_size_mb * 1024 * 1024)
            if total_size_bytes <= max_size_bytes:
                return

            target_size_bytes = int(max_size_bytes * 0.8)
            removed_count = 0
            for _, cache_key, snapshot, size in sorted(candidates):
                if total_size_bytes <= target_size_bytes:
                    break
                if self._retire_exact_authority_snapshot(cache_key, snapshot):
                    total_size_bytes -= size
                    removed_count += 1
            if removed_count > 0:
                logger.info("Cache size enforcement: removed %s entries", removed_count)
            return

        self._preflight_entries(
            self.metadata_backend.list_entries(),
            operation="cleanup_by_size",
        )
        # Get current total size from metadata backend
        stats = self.metadata_backend.get_stats()
        total_size_mb = stats.get("total_size_mb", 0)

        if total_size_mb <= self.config.storage.max_cache_size_mb:
            return

        # Use metadata backend's cleanup functionality
        target_size = (
            self.config.storage.max_cache_size_mb * 0.8
        )  # Clean to 80% of limit
        removed_count = self.metadata_backend.cleanup_by_size(target_size)

        if removed_count > 0:
            logger.info(f"Cache size enforcement: removed {removed_count} entries")

    @_clear_coordinated
    def invalidate(self, cache_key: Optional[str] = None, prefix: str = "", **kwargs):
        """
        Invalidate (remove) specific cache entries.

        Args:
            cache_key: Direct cache key (if provided, **kwargs are ignored)
            prefix: Descriptive prefix of the cache filename
            **kwargs: Parameters identifying the cached data (used if cache_key is None)
        """
        if cache_key is None:
            cache_key = self._create_cache_key(kwargs)

        authority_snapshot = self._cache_blob_store.lifecycle_authority.read_entry(
            cache_key
        )
        if authority_snapshot is not None:
            if not self._retire_exact_authority_snapshot(cache_key, authority_snapshot):
                raise CacheBlobLifecycleConflictError(
                    "Cache generation changed before invalidation could retire it",
                    context={"operation": "invalidate", "key": cache_key},
                )
            logger.info("Invalidated authority-backed cache entry %s", cache_key)
            return

        if self._recognized_legacy_backend() is None:
            # A canonical absence is still an authority decision. A concurrent
            # first promotion changes this expected-absence delete into a
            # conflict rather than authorizing key-only metadata cleanup.
            self._retire_exact_authority_snapshot(cache_key, None)
            logger.debug("Canonical cache entry %s was absent for invalidation", cache_key)
            return

        entry = self.metadata_backend.get_entry(cache_key)
        if entry is not None:
            self._entry_locator(entry, cache_key, operation="invalidate", prefix=prefix)
            self.metadata_backend.remove_entry(cache_key)
            logger.info(f"Invalidated cache entry {cache_key}")
        else:
            logger.debug(f"Cache entry {cache_key} not found for invalidation")

    @_clear_coordinated
    def clear_all(self):
        """Clear known cache entries without reviving a second authority."""
        if self._recognized_legacy_backend() is None:
            # Compatibility metadata is not lifecycle authority, but malformed
            # paths remain fail-closed evidence and must block a bulk mutation.
            entries = self.metadata_backend.list_entries()
            self._preflight_entries(entries, operation="clear_all")
            cleared = self._cache_blob_store.clear()
            for entry in entries:
                self._sync_authority_projection(
                    entry["cache_key"], observed_entry=entry
                )
            logger.info("Cleared %s authority-backed cache entries", cleared)
            return cleared

        entries = self.metadata_backend.list_entries()
        self._preflight_entries(entries, operation="clear_all")
        for entry in entries:
            self._delete_or_prove_absent(
                self._entry_locator(
                    entry, entry["cache_key"], operation="clear_all"
                )
            )
            self.metadata_backend.remove_entry(entry["cache_key"])
        logger.info(f"Cleared {len(entries)} cache entries")
        return len(entries)

    @_clear_read_coordinated
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.metadata_backend.get_stats()
        live_entries = self.list_entries()
        live_size_mb = sum(entry.get("size_mb", 0) for entry in live_entries)
        stats["total_entries"] = len(live_entries)
        stats["dataframe_entries"] = sum(
            entry.get("data_type") == "dataframe" for entry in live_entries
        )
        stats["array_entries"] = sum(
            entry.get("data_type") == "array" for entry in live_entries
        )
        stats["total_size_mb"] = round(live_size_mb, 2)

        # Add cache-specific information
        stats.update(
            {
                "cache_dir": str(self.cache_dir),
                "max_size_mb": self.config.storage.max_cache_size_mb,
                "default_ttl_hours": self.config.metadata.default_ttl_hours,
                "backend_type": self.actual_backend,  # Report actual backend used
            }
        )

        return stats

    @_clear_read_coordinated
    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries with metadata."""
        raw_entries = self.metadata_backend.list_entries()
        if self._recognized_legacy_backend() is not None:
            self._preflight_entries(raw_entries, operation="list_entries")
            for entry in raw_entries:
                entry["expired"] = self._is_expired(entry["cache_key"])
            return raw_entries
        entries = []
        authority_entries = {
            snapshot.key: snapshot
            for snapshot in self._cache_blob_store.lifecycle_authority.list_entries()
        }
        for entry in raw_entries:
            cache_key = entry["cache_key"]
            snapshot = authority_entries.get(cache_key)
            if snapshot is not None and self._projection_matches_authority_snapshot(
                entry, snapshot
            ):
                manifest = self._cache_blob_store.lifecycle._entry_manifest(
                    snapshot, allow_tombstone=True
                )
                if manifest.state != "committed":
                    continue
                projection = entry
            else:
                try:
                    snapshot, projection = self._authority_snapshot_entry(cache_key)
                except CacheBlobLifecycleConflictError:
                    logger.debug(
                        "Omitted unstable authority projection from entry list: %s",
                        cache_key,
                    )
                    continue
            if snapshot is not None and projection is not None:
                visible_entry = entry.copy()
                visible_entry["data_type"] = projection["data_type"]
                visible_entry["description"] = projection["description"]
                visible_entry["metadata"] = projection["metadata"]
                if "created_at" in projection:
                    visible_entry["created"] = self._list_projection_timestamp(
                        projection["created_at"]
                    )
                    visible_entry["last_accessed"] = self._list_projection_timestamp(
                        projection.get("accessed_at", projection["created_at"])
                    )
                    visible_entry["size_mb"] = round(
                        projection["file_size"] / (1024 * 1024), 3
                    )
                entries.append(visible_entry)
        self._preflight_entries(entries, operation="list_entries")

        # Add expiration status for each entry
        for entry in entries:
            entry["expired"] = self._is_expired(entry["cache_key"])

        return entries

    def close(self):
        """Close all resources (database connections, etc.)."""
        blob_store = getattr(self, "_cache_blob_store", None)
        blob_io = getattr(blob_store, "guarded_handler_io", None)
        if blob_store is not None:
            blob_store.close()
        if (
            hasattr(self, "guarded_handler_io")
            and self.guarded_handler_io
            and self.guarded_handler_io is not blob_io
        ):
            self.guarded_handler_io.close()
        if hasattr(self, 'metadata_backend') and self.metadata_backend:
            if hasattr(self.metadata_backend, 'close'):
                self.metadata_backend.close()

    def __del__(self):
        """Ensure resources are cleaned up when the cache is garbage collected."""
        try:
            self.close()
        except Exception:
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
        ttl_hours: int = 6,
        ignore_errors: bool = True,
        **kwargs
    ) -> "UnifiedCache":
        """
        Create a cache optimized for API requests.
        
        Defaults:
        - TTL: 6 hours (good for most API data)
        - ignore_errors: True (don't fail if cache has issues)
        - Compression: LZ4 (fast for JSON/text data)
        
        Args:
            cache_dir: Cache directory (default: ./cache)
            ttl_hours: Time-to-live in hours
            ignore_errors: Continue on cache errors
            **kwargs: Additional config options
        """
        config = create_cache_config(
            cache_dir=cache_dir or "./cache",
            default_ttl_hours=ttl_hours,
            pickle_compression_codec="zstd",  # Fast for JSON/text
            pickle_compression_level=3,
            **kwargs
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
        except Exception:
            pass  # Ignore errors during cleanup
    _global_cache = UnifiedCache(config, metadata_backend)
