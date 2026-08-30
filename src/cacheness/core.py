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
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Callable, Tuple

from .config import CacheConfig, _DEFAULT_TTL, create_cache_config
from .error_handling import (
    CacheIntegrityError,
    CacheLegacyFormatError,
    CacheQueryValidationError,
    CacheReason,
    CacheUnsafePathError,
)
from .handlers import HandlerRegistry
from .serialization import create_unified_cache_key
from .storage.guarded_handler_io import GuardedHandlerIO
from .storage.path_security import encode_physical_name, resolve_managed_locator

logger = logging.getLogger(__name__)


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
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.guarded_handler_io = GuardedHandlerIO(self.cache_dir)

        # Thread safety
        self._lock = threading.Lock()

        # Initialize handler registry with config
        self.handlers = HandlerRegistry(self.config)

        # Initialize metadata backend
        self._init_metadata_backend(metadata_backend)

        # Initialize custom metadata support
        self._init_custom_metadata_support()

        # Initialize entry signer for metadata integrity
        self._init_entry_signer()

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

    def _supports_custom_metadata(self) -> bool:
        """Check if custom metadata is supported (requires SQLite or PostgreSQL backend with SQLAlchemy)."""
        return (
            self.actual_backend in ("sqlite", "postgresql")
            and hasattr(self, "_custom_metadata_enabled")
            and self._custom_metadata_enabled
        )

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

    def _store_custom_metadata(self, cache_key: str, custom_metadata):
        """Store custom metadata using link table architecture."""
        if not self._supports_custom_metadata():
            logger.warning("Custom metadata not supported - requires SQLite or PostgreSQL backend")
            return

        try:
            from .custom_metadata import get_custom_metadata_model, CacheMetadataLink, get_schema_name_for_model, get_all_custom_metadata_models
            from .metadata import Base

            # Normalize custom_metadata to iterable of metadata objects
            metadata_objects = self._normalize_custom_metadata(custom_metadata)
            if not metadata_objects:
                return

            # Get SQLAlchemy session from the metadata backend
            if hasattr(self.metadata_backend, "SessionLocal"):
                with self.metadata_backend.SessionLocal() as session:
                    # Create only custom metadata tables and link table
                    # This avoids conflicts with cache_entries/cache_stats tables
                    # which are managed by the metadata backend
                    tables_to_create = [CacheMetadataLink.__table__]
                    for model_class in get_all_custom_metadata_models().values():
                        if hasattr(model_class, "__table__") and model_class.__table__ is not None:
                            tables_to_create.append(model_class.__table__)
                    Base.metadata.create_all(self.metadata_backend.engine, tables=tables_to_create)

                    for metadata_instance in metadata_objects:
                        # Get schema name from the metadata object's class
                        schema_name = getattr(type(metadata_instance), '_schema_name', None)
                        if not schema_name:
                            logger.warning(
                                f"Metadata object {type(metadata_instance).__name__} is not properly registered"
                            )
                            continue

                        model_class = get_custom_metadata_model(schema_name)
                        if not model_class:
                            logger.warning(
                                f"Unknown custom metadata schema: {schema_name}"
                            )
                            continue

                        # Ensure metadata instance is of the correct type
                        if not isinstance(metadata_instance, model_class):
                            logger.warning(
                                f"Invalid metadata type for schema {schema_name}"
                            )
                            continue

                        # Save the metadata instance
                        session.add(metadata_instance)
                        session.flush()  # Get the ID

                        # Create link table entry
                        link = CacheMetadataLink(
                            cache_key=cache_key,
                            metadata_table=model_class.__tablename__,
                            metadata_id=metadata_instance.id,
                        )
                        session.add(link)

                    session.commit()
                    logger.debug(f"Stored custom metadata for cache key {cache_key}")
        except Exception as e:
            logger.error(f"Failed to store custom metadata: {e}")

    def _get_custom_metadata(self, cache_key: str) -> Dict[str, Any]:
        """Retrieve custom metadata for a cache key."""
        if not self._supports_custom_metadata():
            return {}

        try:
            from .custom_metadata import get_custom_metadata_model, CacheMetadataLink

            if hasattr(self.metadata_backend, "SessionLocal"):
                with self.metadata_backend.SessionLocal() as session:
                    # Get all links for this cache key
                    from sqlalchemy import select

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
                        # Find the schema name for this table
                        for (
                            schema_name,
                            model_class,
                        ) in self._get_registered_schemas().items():
                            if model_class.__tablename__ == link.metadata_table:
                                # Retrieve the metadata instance
                                metadata_instance = session.execute(
                                    select(model_class).where(
                                        model_class.id == link.metadata_id
                                    )
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
            from .custom_metadata import get_custom_metadata_model

            model_class = get_custom_metadata_model(schema_name)
            if not model_class:
                logger.warning(f"Unknown custom metadata schema: {schema_name}")
                return []

            if hasattr(self.metadata_backend, "SessionLocal"):
                # Use context manager to ensure proper session cleanup
                with self.metadata_backend.SessionLocal() as session:
                    query = session.query(model_class)
                    
                    # Apply optional filters
                    if filters:
                        for field_name, value in filters.items():
                            if hasattr(model_class, field_name):
                                query = query.filter(getattr(model_class, field_name) == value)
                            else:
                                logger.warning(f"Unknown filter field '{field_name}' for schema '{schema_name}'")
                    
                    return query.all()
            else:
                logger.warning("SQLAlchemy session not available")
                return []
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
        from contextlib import contextmanager
        
        @contextmanager
        def _session_context():
            if not self._supports_custom_metadata():
                raise ValueError("Custom metadata querying not supported - requires SQLite or PostgreSQL backend")

            from .custom_metadata import get_custom_metadata_model

            model_class = get_custom_metadata_model(schema_name)
            if not model_class:
                raise ValueError(f"Unknown custom metadata schema: {schema_name}")

            if not hasattr(self.metadata_backend, "SessionLocal"):
                raise ValueError("SQLAlchemy session not available")

            session = self.metadata_backend.SessionLocal()
            try:
                yield session.query(model_class)
            finally:
                session.close()
        
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

    def _discard_uncommitted_payload(self, locator: Path | str) -> None:
        """Remove a candidate payload that has never been published in metadata."""
        self.guarded_handler_io.file_ops.delete(locator)

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
        """Remove expired cache entries."""
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

    def put(
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

        try:
            # Handler serialization happens only in a private stage. The
            # adapter publishes a guarded opaque physical name.
            result = self.guarded_handler_io.put(
                handler,
                data,
                storage_id,
                self.config,
            )

            # Calculate file hash for integrity verification (if enabled)
            file_hash = None
            if self.config.metadata.verify_cache_integrity:
                with self.guarded_handler_io.open_snapshot(
                    result["actual_path"],
                    {},
                ) as snapshot:
                    file_hash = self._calculate_file_hash(snapshot.path)
                if not self._is_valid_file_hash(file_hash):
                    # Integrity-enabled entries are not allowed to fall back to
                    # the same missing-digest representation used when the
                    # feature is explicitly disabled. The candidate has never
                    # been recorded in metadata, so discarding it cannot damage
                    # an older payload still committed for this cache key.
                    self._discard_uncommitted_payload(result["actual_path"])
                    raise CacheIntegrityError(
                        "Unable to calculate a complete payload integrity digest"
                    )

            # Update metadata
            metadata_dict = {
                **result["metadata"],
                "prefix": prefix,
                "actual_path": result["actual_path"],
                "file_hash": file_hash,  # Store file hash for verification
            }

            # Only store cache_key_params if enabled in configuration
            if self.config.metadata.store_cache_key_params:
                metadata_dict["cache_key_params"] = (
                    kwargs  # Store original key-value parameters for efficient querying
                )

            entry_data = {
                "data_type": handler.data_type,
                "prefix": prefix,
                "description": description,
                "file_size": result["file_size"],
                "metadata": metadata_dict,
            }

            # Sign the entry if signing is enabled
            if self.signer:
                try:
                    # Use consistent timestamp for both storage and signing
                    creation_timestamp = datetime.now(timezone.utc)
                    # Store without timezone info for consistency with database format
                    creation_timestamp_str = creation_timestamp.replace(tzinfo=None).isoformat()
                    
                    # Store the creation timestamp in entry_data for the database
                    entry_data["created_at"] = creation_timestamp_str
                    
                    # Use helper method for consistent field extraction
                    cache_key_params = kwargs if self.config.metadata.store_cache_key_params else None
                    complete_entry_data = self._extract_signable_fields(
                        cache_key=cache_key,
                        entry_data=entry_data,
                        metadata=metadata_dict,
                        cache_key_params=cache_key_params
                    )
                    
                    signature = self.signer.sign_entry(complete_entry_data)
                    metadata_dict["entry_signature"] = signature
                    
                    logger.debug(f"Created signature for entry {cache_key}")
                    
                except Exception as e:
                    logger.warning(f"Failed to sign entry {cache_key}: {e}")
                    # Continue without signature for backward compatibility
            
            self.metadata_backend.put_entry(cache_key, entry_data)

            if previous_locator is not None:
                try:
                    self.guarded_handler_io.file_ops.delete(previous_locator)
                except Exception:
                    logger.exception(
                        "Replacement metadata committed but prior payload cleanup failed"
                    )

            # Handle custom metadata if provided
            if custom_metadata and self._supports_custom_metadata():
                self._store_custom_metadata(cache_key, custom_metadata)

            self._enforce_size_limit()

            file_size_mb = result["file_size"] / (1024 * 1024)
            format_info = f"({result['storage_format']} format)"
            logger.info(
                f"Cached {handler.data_type} {cache_key} ({file_size_mb:.3f}MB) {format_info}: {description}"
            )
            
            return cache_key

        except CacheUnsafePathError:
            # Unsafe locators never become a generic storage failure.
            raise
        except (OSError, IOError) as e:
            # I/O errors (disk full, permissions, etc.)
            logger.error(f"Failed to cache {handler.data_type} (I/O error): {e}")
            raise
        except Exception as e:
            # Include exception type for easier debugging
            logger.error(f"Failed to cache {handler.data_type}: {type(e).__name__}: {e}")
            raise

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

    def _reject_untrusted_entry(self, cache_key: str, *, reason: str) -> None:
        """Record a safe miss and optionally remove verified-bad evidence."""
        logger.warning("Cache entry %s was rejected before deserialization: %s", cache_key, reason)
        if self.config.security.delete_invalid_signatures:
            self.metadata_backend.remove_entry(cache_key)
        self.metadata_backend.increment_misses()

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

        entry = self.metadata_backend.get_entry(cache_key)
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
                        )
                        return None
                    current_hash = self._calculate_file_hash(snapshot.path)
                    if current_hash != stored_hash:
                        self._reject_untrusted_entry(
                            cache_key,
                            reason="payload integrity hash mismatch",
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
                    )
                    return None

                data = handler.get(snapshot.path, snapshot.metadata)

        except CacheUnsafePathError:
            # No unsafe locator becomes a miss, cleanup, or evidence mutation.
            raise
        except FileNotFoundError as e:
            logger.warning(f"Cache file missing for {cache_key}: {e}")
            self.metadata_backend.remove_entry(cache_key)
            self.metadata_backend.increment_misses()
            return None
        except (OSError, IOError) as e:
            logger.warning(f"I/O error loading cached {data_type} {cache_key}: {e}")
            self.metadata_backend.remove_entry(cache_key)
            self.metadata_backend.increment_misses()
            return None
        except CacheLegacyFormatError:
            # Typed compatibility failures must never be converted into mutation.
            raise
        except Exception as e:
            logger.warning(
                f"Failed to load cached {data_type} {cache_key}: {type(e).__name__}: {e}"
            )
            self.metadata_backend.remove_entry(cache_key)
            self.metadata_backend.increment_misses()
            return None

        if not _legacy_decorator_v0313:
            self._record_successful_read(cache_key, entry)
        logger.debug(f"Cache hit ({data_type}): {cache_key}")
        return data

    def _enforce_size_limit(self):
        """Enforce cache size limits using LRU eviction."""
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

        entry = self.metadata_backend.get_entry(cache_key)
        if entry is not None:
            self._entry_locator(entry, cache_key, operation="invalidate", prefix=prefix)
            self.metadata_backend.remove_entry(cache_key)
            logger.info(f"Invalidated cache entry {cache_key}")
        else:
            logger.debug(f"Cache entry {cache_key} not found for invalidation")

    def clear_all(self):
        """Clear all cache entries and remove cache files."""
        entries = self.metadata_backend.list_entries()
        self._preflight_entries(entries, operation="clear_all")

        # Known payload deletion remains guarded. Orphan reconciliation is a
        # later lifecycle concern, so this does not glob arbitrary paths.
        files_removed = 0
        for entry in entries:
            locator = self._entry_locator(
                entry,
                entry.get("cache_key"),
                operation="clear_all",
            )
            files_removed += int(self.guarded_handler_io.file_ops.delete(locator))

        removed_count = self.metadata_backend.clear_all()
        logger.info(
            f"Cleared {removed_count} cache entries and removed {files_removed} cache files"
        )
        return removed_count

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.metadata_backend.get_stats()

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

    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries with metadata."""
        entries = self.metadata_backend.list_entries()
        self._preflight_entries(entries, operation="list_entries")

        # Add expiration status for each entry
        for entry in entries:
            entry["expired"] = self._is_expired(entry["cache_key"])

        return entries

    def close(self):
        """Close all resources (database connections, etc.)."""
        if hasattr(self, "guarded_handler_io") and self.guarded_handler_io:
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
