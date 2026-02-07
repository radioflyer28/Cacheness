"""
Simplified Unified Cache with Handler-based Architecture
=======================================================

This module provides a cleaner, more maintainable cache system using the Strategy pattern.
The main UnifiedCache class is now focused on coordination and delegates format-specific
operations to specialized handlers.
"""

import os
import xxhash
import inspect
import threading
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Callable, Tuple

from .config import CacheConfig, _DEFAULT_TTL, create_cache_config
from .handlers import HandlerRegistry
from .serialization import create_unified_cache_key

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

        # Clean up expired entries on initialization
        if self.config.storage.cleanup_on_init:
            self._cleanup_expired()

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
        
        if requested == "json":
            self.metadata_backend = create_metadata_backend(
                "json",
                metadata_file=self.cache_dir / "cache_metadata.json",
                config=self.config.metadata,
            )
            self.actual_backend = "json"
            
        elif requested == "sqlite":
            self.metadata_backend = create_metadata_backend(
                "sqlite",
                db_file=str(self.cache_dir / self.config.metadata.sqlite_db_file),
                config=self.config.metadata,
            )
            self.actual_backend = "sqlite"
            
        elif requested == "sqlite_memory":
            self.metadata_backend = create_metadata_backend("sqlite_memory", config=self.config.metadata)
            self.actual_backend = "sqlite_memory"
            logger.info("⚡ Using in-memory SQLite backend (no persistence)")
            
        elif requested == "postgresql":
            opts = self.config.metadata.metadata_backend_options or {}
            connection_url = opts.get("connection_url")
            if not connection_url:
                raise ValueError(
                    "PostgreSQL backend requires 'connection_url' in metadata_backend_options"
                )
            self.metadata_backend = create_metadata_backend(
                "postgresql",
                connection_url=connection_url,
                pool_size=opts.get("pool_size", 10),
                max_overflow=opts.get("max_overflow", 20),
                pool_pre_ping=opts.get("pool_pre_ping", True),
                pool_recycle=opts.get("pool_recycle", 3600),
                echo=opts.get("echo", False),
                table_prefix=opts.get("table_prefix", ""),
                config=self.config.metadata,
            )
            self.actual_backend = "postgresql"
            
        else:
            # Auto mode: prefer SQLite if available, fallback to JSON
            self._init_auto_backend(create_metadata_backend, SQLALCHEMY_AVAILABLE)
    
    def _init_auto_backend(self, create_metadata_backend, sqlalchemy_available: bool):
        """Auto-select the best available metadata backend."""
        if sqlalchemy_available:
            try:
                self.metadata_backend = create_metadata_backend(
                    "sqlite",
                    db_file=str(self.cache_dir / self.config.metadata.sqlite_db_file),
                    config=self.config.metadata,
                )
                self.actual_backend = "sqlite"
                logger.info("🗄️  Using SQLite backend (auto-selected for better performance)")
                return
            except Exception as e:
                logger.warning(f"SQLite backend failed, falling back to JSON: {e}")
        else:
            logger.info("📝 SQLModel not available, using JSON backend")
        
        self.metadata_backend = create_metadata_backend(
            "json",
            metadata_file=self.cache_dir / "cache_metadata.json",
            config=self.config.metadata,
        )
        self.actual_backend = "json"

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
            from .custom_metadata import get_custom_metadata_model, get_schema_name_for_model, get_all_custom_metadata_models
            from .metadata import Base

            # Normalize custom_metadata to iterable of metadata objects
            metadata_objects = self._normalize_custom_metadata(custom_metadata)
            if not metadata_objects:
                return

            # Get SQLAlchemy session from the metadata backend
            if hasattr(self.metadata_backend, "SessionLocal"):
                with self.metadata_backend.SessionLocal() as session:
                    # Create only custom metadata tables
                    # This avoids conflicts with cache_entries/cache_stats tables
                    # which are managed by the metadata backend
                    tables_to_create = []
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

                        # Set the cache_key on the metadata instance (direct FK)
                        metadata_instance.cache_key = cache_key

                        # Save the metadata instance
                        session.add(metadata_instance)

                    session.commit()
                    logger.debug(f"Stored custom metadata for cache key {cache_key}")
        except Exception as e:
            logger.error(f"Failed to store custom metadata: {e}")

    def _get_custom_metadata(self, cache_key: str) -> Dict[str, Any]:
        """Retrieve custom metadata for a cache key."""
        if not self._supports_custom_metadata():
            return {}

        try:
            from .custom_metadata import get_custom_metadata_model

            if hasattr(self.metadata_backend, "SessionLocal"):
                with self.metadata_backend.SessionLocal() as session:
                    from sqlalchemy import select

                    result = {}
                    # Query each registered schema for metadata with this cache_key
                    for schema_name, model_class in self._get_registered_schemas().items():
                        metadata_instance = session.execute(
                            select(model_class).where(
                                model_class.cache_key == cache_key
                            )
                        ).scalar_one_or_none()

                        if metadata_instance:
                            result[schema_name] = metadata_instance

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
        when store_full_metadata=True is configured.

        Args:
            **filters: Key-value pairs to filter cache entries by their parameters
                      Supports nested dictionary access with dot notation

        Returns:
            List of cache entries matching the filters, or None if not supported

        Example:
            # Configure cache to store parameters
            config = CacheConfig(store_full_metadata=True)
            cache = Cacheness(config=config)
            
            # Store some data
            cache.put(model, experiment="exp_001", model_type="xgboost", accuracy=0.95)
            cache.put(data, experiment="exp_002", model_type="cnn", accuracy=0.88)
            
            # Query by parameters  
            xgb_experiments = cache.query_meta(model_type="xgboost")
            high_accuracy = cache.query_meta(accuracy=0.9)  # >= comparison
            specific_exp = cache.query_meta(experiment="exp_001")
        """
        if self.actual_backend != "sqlite":
            logger.warning("query_meta() requires SQLite backend")
            return None

        if not self.config.metadata.store_full_metadata:
            logger.warning(
                "query_meta() requires store_full_metadata=True in cache configuration"
            )
            return None

        if not hasattr(self.metadata_backend, "SessionLocal"):
            logger.warning("SQLAlchemy session not available for query_meta()")
            return None

        try:
            from sqlalchemy import text
            
            with self.metadata_backend.SessionLocal() as session:
                # Build WHERE conditions using SQLite JSON1 extension
                where_conditions = []
                params = {}
                
                for key, value in filters.items():
                    # Use JSON_EXTRACT for querying JSON fields
                    param_name = f"param_{len(params)}"
                    
                    if isinstance(value, (int, float)):
                        # For numeric values, cast to REAL for proper comparison
                        where_conditions.append(
                            f"CAST(JSON_EXTRACT(metadata_dict, '$.{key}') AS REAL) = :{param_name}"
                        )
                    else:
                        # For string values, exact match
                        where_conditions.append(
                            f"JSON_EXTRACT(metadata_dict, '$.{key}') = :{param_name}"
                        )
                    
                    params[param_name] = value

                # Build the query
                if where_conditions:
                    where_clause = " AND ".join(where_conditions)
                    query = f"""
                        SELECT cache_key, description, data_type, created_at, accessed_at, 
                               file_size, metadata_dict
                        FROM cache_entries 
                        WHERE metadata_dict IS NOT NULL AND ({where_clause})
                        ORDER BY created_at DESC
                    """
                else:
                    # No filters - return all entries with metadata_dict
                    query = """
                        SELECT cache_key, description, data_type, created_at, accessed_at,
                               file_size, metadata_dict  
                        FROM cache_entries
                        WHERE metadata_dict IS NOT NULL
                        ORDER BY created_at DESC
                    """

                result = session.execute(text(query), params)
                
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
                    
                    # Parse metadata_dict JSON
                    if row.metadata_dict:
                        try:
                            from .json_utils import loads as json_loads
                            entry['metadata_dict'] = json_loads(row.metadata_dict)
                        except Exception:
                            entry['metadata_dict'] = {}
                    
                    entries.append(entry)
                
                return entries
                
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
        """Get base cache file path (without extension)."""
        if prefix:
            filename_base = f"{prefix}_{cache_key}"
        else:
            filename_base = cache_key

        return self.cache_dir / filename_base

    def _is_expired(self, cache_key: str, ttl_seconds=_DEFAULT_TTL) -> bool:
        """Check if cache entry is expired.
        
        Args:
            cache_key: The cache key to check
            ttl_seconds: TTL in seconds. None means never expire.
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

        # Type guard to ensure ttl is numeric
        assert isinstance(ttl_seconds, (int, float)), f"TTL must be numeric, got {type(ttl_seconds)}"

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
        # Normalize created_at to ISO format string without timezone
        # This ensures consistent signatures regardless of database format
        created_at = entry_data.get("created_at")
        if isinstance(created_at, datetime):
            # Convert datetime to ISO string, removing timezone info for consistency
            created_at = created_at.replace(tzinfo=None).isoformat()
        elif isinstance(created_at, str):
            # Parse and re-format to ensure consistency
            try:
                dt = datetime.fromisoformat(created_at)
                created_at = dt.replace(tzinfo=None).isoformat()
            except (ValueError, TypeError):
                # If parsing fails, use as-is
                pass
        
        # Build complete entry data with all signable fields
        signable_data = {
            "cache_key": cache_key,
            "data_type": entry_data.get("data_type"),
            "prefix": entry_data.get("prefix", ""),
            "description": entry_data.get("description", ""),
            "file_size": entry_data.get("file_size", 0),
            "created_at": created_at,
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
        """
        Calculate XXH3_64 hash of a cache file for integrity verification.

        Args:
            file_path: Path to the cache file

        Returns:
            Hex string of the file hash, or None if file doesn't exist or error
        """
        try:
            if not file_path.exists():
                return None

            hasher = xxhash.xxh3_64()
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate hash for {file_path}: {e}")
            return None

    def _cleanup_expired(self):
        """Remove expired cache entries."""
        ttl_seconds = self.config.metadata.default_ttl_seconds
        removed_count = self.metadata_backend.cleanup_expired(ttl_seconds)

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache entries")

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
        with self._lock:
            # Get appropriate handler
            handler = self.handlers.get_handler(data)
            cache_key = self._create_cache_key(kwargs)
            base_file_path = self._get_cache_file_path(cache_key, prefix)

            try:
                # Use handler to store the data
                result = handler.put(data, base_file_path, self.config)

                # Track the blob path so we can clean up on failure
                blob_path = Path(result.get("actual_path", str(base_file_path)))

                # Calculate file hash for integrity verification (if enabled)
                file_hash = None
                if self.config.metadata.verify_cache_integrity:
                    actual_path = result.get("actual_path", str(base_file_path))
                    file_hash = self._calculate_file_hash(Path(actual_path))

                # Update metadata
                metadata_dict = {
                    **result["metadata"],
                    "prefix": prefix,
                    "actual_path": result.get("actual_path", str(base_file_path)),
                    "file_hash": file_hash,  # Store file hash for verification
                }

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
                        metadata_dict["cache_key_params"] = json_dumps(serializable_kwargs)
                        
                        # Also store kwargs as metadata_dict (raw values for easy querying)
                        # Pre-serialize to JSON string - backend just stores strings
                        metadata_dict["metadata_dict"] = json_dumps(kwargs.copy())
                    except Exception as e:
                        # If serialization fails, skip cache_key_params
                        logger.warning(f"Failed to serialize cache_key_params: {e}")

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
                        # Use consistent timestamp for both storage and signing (always UTC)
                        creation_timestamp = datetime.now(timezone.utc)
                        # Store as UTC ISO format with timezone info
                        creation_timestamp_str = creation_timestamp.isoformat()
                        
                        # Store the creation timestamp in entry_data for the database
                        entry_data["created_at"] = creation_timestamp_str
                        
                        # Use helper method for consistent field extraction
                        cache_key_params = kwargs if self.config.metadata.store_full_metadata else None
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

            except (OSError, IOError) as e:
                # I/O errors (disk full, permissions, etc.)
                # Clean up orphaned blob file if it was written
                if 'blob_path' in locals() and blob_path.exists():
                    try:
                        blob_path.unlink()
                        logger.debug(f"Cleaned up orphaned blob: {blob_path}")
                    except OSError:
                        pass
                logger.error(f"Failed to cache {handler.data_type} (I/O error): {e}")
                raise
            except Exception as e:
                # Clean up orphaned blob file if it was written
                if 'blob_path' in locals() and blob_path.exists():
                    try:
                        blob_path.unlink()
                        logger.debug(f"Cleaned up orphaned blob: {blob_path}")
                    except OSError:
                        pass
                # Include exception type for easier debugging
                logger.error(f"Failed to cache {handler.data_type}: {type(e).__name__}: {e}")
                raise

    def get(
        self,
        cache_key: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        prefix: str = "",
        **kwargs,
    ) -> Optional[Any]:
        """
        Retrieve any supported data type from cache.

        Args:
            cache_key: Direct cache key (if provided, **kwargs are ignored)
            ttl_seconds: Custom TTL in seconds (overrides default). None = never expire.
            prefix: Descriptive prefix prepended to the cache filename
            **kwargs: Parameters identifying the cached data (used if cache_key is None)

        Returns:
            Cached data or None if not found/expired
        """
        with self._lock:
            if cache_key is None:
                cache_key = self._create_cache_key(kwargs)

            # Check if entry exists and is not expired
            entry = self.metadata_backend.get_entry(cache_key)
            if not entry or self._is_expired(cache_key, ttl_seconds):
                self.metadata_backend.increment_misses()
                return None

            # Get appropriate handler
            data_type = entry.get("data_type")
            if not data_type:
                self.metadata_backend.increment_misses()
                return None

            try:
                handler = self.handlers.get_handler_by_type(data_type)
                base_file_path = self._get_cache_file_path(cache_key, prefix)

                # Use actual path from metadata if available, otherwise use base path
                metadata = entry.get("metadata", {})
                actual_path = metadata.get("actual_path")
                if actual_path:
                    file_path = Path(actual_path)
                else:
                    file_path = base_file_path

                # Verify cache file integrity if enabled and hash is available
                if self.config.metadata.verify_cache_integrity:
                    stored_hash = metadata.get("file_hash")
                    if stored_hash is not None:
                        current_hash = self._calculate_file_hash(file_path)
                        if current_hash != stored_hash:
                            logger.warning(
                                f"Cache integrity verification failed for {cache_key}: "
                                f"stored hash {stored_hash} != current hash {current_hash}. "
                                f"Removing corrupted cache entry."
                            )
                            self.metadata_backend.remove_entry(cache_key)
                            self.metadata_backend.increment_misses()
                            return None

                # Verify entry signature if signing is enabled
                if self.signer and self.config.security.enable_entry_signing:
                    stored_signature = metadata.get("entry_signature")
                    
                    if stored_signature is not None:
                        # Use helper method for consistent field extraction
                        cache_key_params = metadata.get("cache_key_params")
                        verify_entry_data = self._extract_signable_fields(
                            cache_key=cache_key,
                            entry_data=entry,
                            metadata=metadata,
                            cache_key_params=cache_key_params
                        )
                        
                        if not self.signer.verify_entry(verify_entry_data, stored_signature):
                            if self.config.security.delete_invalid_signatures:
                                logger.warning(
                                    f"Entry signature verification failed for {cache_key}. "
                                    f"Removing potentially tampered cache entry."
                                )
                                self.metadata_backend.remove_entry(cache_key)
                                self.metadata_backend.increment_misses()
                                return None
                            else:
                                logger.warning(
                                    f"Entry signature verification failed for {cache_key}. "
                                    f"Entry retained due to delete_invalid_signatures=False."
                                )
                                # Continue with loading despite invalid signature
                            
                    elif not self.config.security.allow_unsigned_entries:
                        logger.warning(
                            f"Entry {cache_key} has no signature but unsigned entries are not allowed. "
                            f"Removing entry."
                        )
                        self.metadata_backend.remove_entry(cache_key)
                        self.metadata_backend.increment_misses()
                        return None

                # Use handler to load the data
                data = handler.get(file_path, metadata)

                # Update access time
                self.metadata_backend.update_access_time(cache_key)
                self.metadata_backend.increment_hits()

                logger.debug(f"Cache hit ({data_type}): {cache_key}")
                return data

            except FileNotFoundError as e:
                # Cache file was deleted externally — permanent, clean up metadata
                logger.warning(f"Cache file missing for {cache_key}: {e}")
                self.metadata_backend.remove_entry(cache_key)
                self.metadata_backend.increment_misses()
                return None
            except (OSError, IOError) as e:
                # I/O errors may be transient (disk temporarily unavailable, etc.)
                # Do NOT delete metadata — the entry may be readable on retry
                logger.warning(f"I/O error loading cached {data_type} {cache_key}: {e}")
                self.metadata_backend.increment_misses()
                return None
            except Exception as e:
                # Unexpected errors (deserialization failures, corruption, etc.)
                # These are likely permanent — clean up the metadata entry
                logger.warning(
                    f"Failed to load cached {data_type} {cache_key}: {type(e).__name__}: {e}"
                )
                self.metadata_backend.remove_entry(cache_key)
                self.metadata_backend.increment_misses()
                return None

    def get_metadata(
        self,
        cache_key: Optional[str] = None,
        check_expiration: bool = True,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Get entry metadata without loading blob data.
        
        This is useful for inspecting cache entries (TTL, file size, data type)
        before deciding whether to load the actual data.
        
        Args:
            cache_key: Direct cache key (if provided, **kwargs are ignored)
            check_expiration: If True, returns None for expired entries (default: True)
            **kwargs: Parameters identifying the cached data (used if cache_key is None)
        
        Returns:
            Metadata dictionary or None if not found or expired
            
        Example:
            # Check metadata before loading large file
            meta = cache.get_metadata(experiment="exp_001")
            if meta and meta.get("file_size_bytes", 0) > 1e9:
                print("Large file - loading may take time")
                data = cache.get(experiment="exp_001")
        """
        with self._lock:
            if cache_key is None:
                cache_key = self._create_cache_key(kwargs)
            
            entry = self.metadata_backend.get_entry(cache_key)
            if not entry:
                return None
            
            # Check expiration if requested (respects cache TTL policy)
            if check_expiration and self._is_expired(cache_key):
                return None
            
            # Ensure cache_key is included in returned metadata
            entry["cache_key"] = cache_key
            
            return entry
    
    def update_data(
        self,
        data: Any,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Update blob data at an existing cache entry without changing the cache_key.
        
        This replaces the stored data at a fixed cache_key while updating derived
        metadata (file_size, content_hash, created_at timestamp). The cache_key
        itself remains unchanged to maintain referential integrity.
        
        Args:
            data: New data to store (must be serializable by handler)
            cache_key: Direct cache key (if provided, **kwargs are ignored)
            **kwargs: Parameters identifying the cached data (used if cache_key is None)
        
        Returns:
            bool: True if entry was updated, False if entry doesn't exist
            
        Example:
            # Update cached DataFrame with new data
            success = cache.update_data(
                new_df,
                experiment="exp_001",
                run_id=42
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
            if cache_key is None:
                cache_key = self._create_cache_key(kwargs)
            
            # Check if entry exists before doing any I/O
            existing_entry = self.metadata_backend.get_entry(cache_key)
            if not existing_entry:
                logger.warning(f"⚠️ Cache entry not found for update: {cache_key[:16]}...")
                return False
            
            # Get appropriate handler for the data type
            handler = self.handlers.get_handler(data)
            
            # Reconstruct file path from existing metadata (blob I/O belongs here, not in metadata layer)
            prefix = existing_entry.get("prefix", "")
            base_file_path = self._get_cache_file_path(cache_key, prefix)
            
            # Perform blob I/O: write new data to disk
            result = handler.put(data, base_file_path, self.config)
            
            # Build metadata updates dict from handler result
            updates = {
                "file_size": result.get("file_size", 0),
                "content_hash": result.get("content_hash"),
                "file_hash": result.get("file_hash"),
                "actual_path": str(result.get("actual_path", base_file_path)),
                "storage_format": result.get("storage_format"),
            }
            if hasattr(handler, "data_type"):
                updates["data_type"] = handler.data_type
            if hasattr(handler, "serializer"):
                updates["serializer"] = handler.serializer
            if result.get("compression_codec"):
                updates["compression_codec"] = result["compression_codec"]
            if result.get("object_type"):
                updates["object_type"] = result["object_type"]
            if result.get("s3_etag"):
                updates["s3_etag"] = result["s3_etag"]
            
            # Delegate metadata-only update to backend (no I/O in metadata layer)
            success = self.metadata_backend.update_entry_metadata(
                cache_key=cache_key,
                updates=updates
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
                        if actual_path and self.config.metadata.verify_cache_integrity:
                            new_file_hash = self._calculate_file_hash(Path(actual_path))
                            metadata["file_hash"] = new_file_hash
                        
                        # Extract signable fields and create new signature
                        cache_key_params = kwargs if self.config.metadata.store_full_metadata else None
                        complete_entry_data = self._extract_signable_fields(
                            cache_key=cache_key,
                            entry_data=updated_entry,
                            metadata=metadata,
                            cache_key_params=cache_key_params
                        )
                        
                        # Generate new signature for updated entry
                        new_signature = self.signer.sign_entry(complete_entry_data)
                        metadata["entry_signature"] = new_signature
                        
                        # Update entry with new signature and file hash
                        updated_entry["metadata"] = metadata
                        self.metadata_backend.put_entry(cache_key, updated_entry)
                        
                        logger.debug(f"Re-signed updated entry {cache_key}")
                except Exception as e:
                    logger.warning(f"Failed to re-sign updated entry {cache_key}: {e}")
                    # Continue - update succeeded, just missing signature
            
            logger.info(f"✅ Updated cache entry: {cache_key[:16]}...")
            return True
    
    def touch(
        self,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Update entry timestamp to extend TTL without reloading data.
        
        This "touches" the cache entry to reset its creation timestamp to now,
        effectively extending the entry's lifetime by the full configured TTL.
        Useful for keeping frequently accessed data alive or preventing
        expiration of long-running computations.
        
        Args:
            cache_key: Direct cache key (if provided, **kwargs are ignored)
            **kwargs: Parameters identifying the cached data (used if cache_key is None)
        
        Returns:
            bool: True if entry exists and was touched, False if entry doesn't exist
            
        Example:
            # Reset TTL to full default duration from now
            cache.touch(experiment="exp_001")
            
            # Keep long-running computation alive
            for i in range(100):
                process_chunk(i)
                if i % 10 == 0:
                    cache.touch(job_id="long_job")  # Prevent expiration
        
        Note:
            - This is a cache-layer operation (TTL-aware)
            - Resets ``created_at`` to now, giving a full config-TTL extension
            - Does not reload or re-serialize data — much faster than get() + put()
            - TTL duration is always determined by the global config
              (``CacheMetadataConfig.default_ttl_seconds``)
        """
        with self._lock:
            if cache_key is None:
                cache_key = self._create_cache_key(kwargs)
            
            # Get existing entry
            entry = self.metadata_backend.get_entry(cache_key)
            if not entry:
                logger.warning(f"⚠️ Cache entry not found for touch: {cache_key[:16]}...")
                return False
            
            # Update timestamp to now (resets TTL)
            now = datetime.now(timezone.utc)
            entry["created_at"] = now.isoformat()
            entry["accessed_at"] = now.isoformat()
            
            # Re-sign if signing is enabled (timestamp is part of signature)
            if self.signer:
                try:
                    metadata = entry.get("metadata", {})
                    cache_key_params = kwargs if self.config.metadata.store_full_metadata else None
                    complete_entry_data = self._extract_signable_fields(
                        cache_key=cache_key,
                        entry_data=entry,
                        metadata=metadata,
                        cache_key_params=cache_key_params
                    )
                    
                    # Generate new signature with updated timestamp
                    new_signature = self.signer.sign_entry(complete_entry_data)
                    metadata["entry_signature"] = new_signature
                    entry["metadata"] = metadata
                    
                    logger.debug(f"Re-signed touched entry {cache_key}")
                except Exception as e:
                    logger.warning(f"Failed to re-sign touched entry {cache_key}: {e}")
                    # Continue - touch succeeded, just missing signature
            
            # Store updated entry
            self.metadata_backend.put_entry(cache_key, entry)
            
            logger.info(f"👆 Touched cache entry: {cache_key[:16]}... (TTL extended)")
            return True

    # ── Bulk & Batch Operations ───────────────────────────────────────────

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
                    entry["created"] = raw.isoformat() if hasattr(raw, "isoformat") else raw
                if "last_accessed" not in entry and "accessed_at" in entry:
                    raw = entry["accessed_at"]
                    entry["last_accessed"] = raw.isoformat() if hasattr(raw, "isoformat") else raw
                if "size_mb" not in entry and "file_size" in entry:
                    entry["size_mb"] = round(entry["file_size"] / (1024 * 1024), 3)
                try:
                    if filter_fn(entry):
                        cache_key = entry.get("cache_key")
                        if cache_key:
                            self.invalidate(cache_key=cache_key)
                            deleted += 1
                except Exception as exc:
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

            # Fast path: use query_meta on SQLite for indexed lookups
            if self.actual_backend == "sqlite" and self.config.metadata.store_full_metadata:
                results = self.query_meta(**kwargs)
                if results is not None:
                    deleted = 0
                    for entry in results:
                        cache_key = entry.get("cache_key")
                        if cache_key:
                            self.invalidate(cache_key=cache_key)
                            deleted += 1
                    logger.info(f"🗑️ Bulk delete (query_meta): removed {deleted} entries")
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
                hash_kwargs = {k: v for k, v in kw.items()
                               if k not in ("cache_key", "prefix", "ttl_seconds")}
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
                hash_kwargs = {k: v for k, v in kw.items()
                               if k not in ("cache_key", "prefix", "description",
                                            "custom_metadata")}
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
        with self._lock:
            if cache_key is None:
                cache_key = self._create_cache_key(kwargs)

            # Look up the entry to find the blob path before removing metadata
            entry = self.metadata_backend.get_entry(cache_key)
            blob_deleted = False
            if entry:
                actual_path = None
                # actual_path may be in the top-level entry or nested in metadata
                metadata = entry.get("metadata", {})
                if isinstance(metadata, dict):
                    actual_path = metadata.get("actual_path")
                if not actual_path:
                    actual_path = entry.get("actual_path")
                if actual_path:
                    blob_file = Path(actual_path)
                    if blob_file.exists():
                        try:
                            blob_file.unlink()
                            blob_deleted = True
                        except OSError as exc:
                            logger.warning(
                                f"Failed to delete blob file {actual_path} for "
                                f"cache entry {cache_key}: {exc}"
                            )

            # Remove from metadata backend
            if self.metadata_backend.remove_entry(cache_key):
                logger.info(
                    f"Invalidated cache entry {cache_key}"
                    + (" (blob deleted)" if blob_deleted else "")
                )
            else:
                logger.debug(f"Cache entry {cache_key} not found for invalidation")

    def verify_integrity(self, repair: bool = False, verify_hashes: bool = False) -> Dict[str, Any]:
        """
        Verify cache integrity by cross-checking blob files and metadata entries.

        Detects:
        - Orphaned blobs: files in cache_dir with no metadata entry
        - Dangling metadata: entries pointing to missing blob files
        - Size mismatches: metadata file_size != actual file size on disk
        - Hash mismatches: metadata file_hash != actual file hash (if verify_hashes=True)

        Args:
            repair: If True, delete orphaned blobs and remove dangling metadata entries.
            verify_hashes: If True, also verify file hashes (slower but catches corruption).

        Returns:
            Dict with keys: orphaned_blobs, dangling_entries, size_mismatches,
            hash_mismatches (if verify_hashes), repaired (if repair).
        """
        import glob

        with self._lock:
            # 1. Inventory all blob files in cache_dir
            blob_extensions = ["pkl", "npz", "b2nd", "b2tr", "parquet"]
            pickle_codecs = ["lz4", "zstd", "gzip", "zst", "gz", "bz2", "xz"]

            blob_files: set[str] = set()
            patterns = [str(self.cache_dir / f"*.{ext}") for ext in blob_extensions]
            for codec in pickle_codecs:
                patterns.append(str(self.cache_dir / f"*.pkl.{codec}"))
            patterns.append(str(self.cache_dir / "*.pkl.*"))

            for pattern in patterns:
                for file_path in glob.glob(pattern):
                    blob_files.add(os.path.normpath(file_path))

            # 2. Inventory all metadata entries and their actual_paths
            entry_paths: dict[str, dict] = {}  # actual_path -> {cache_key, file_size, file_hash}
            for entry in self.metadata_backend.iter_entry_summaries():
                actual_path = entry.get("actual_path")
                if actual_path:
                    norm_path = os.path.normpath(actual_path)
                    entry_paths[norm_path] = {
                        "cache_key": entry["cache_key"],
                        "file_size": entry.get("file_size"),
                        "file_hash": entry.get("file_hash"),
                    }

            # 3. Find orphaned blobs (files with no metadata entry)
            known_paths = set(entry_paths.keys())
            orphaned_blobs = sorted(blob_files - known_paths)

            # 4. Find dangling metadata (entries pointing to missing files)
            dangling_entries = []
            for path, info in entry_paths.items():
                if not os.path.exists(path):
                    dangling_entries.append({
                        "cache_key": info["cache_key"],
                        "expected_path": path,
                    })

            # 5. Check size mismatches
            size_mismatches = []
            for path, info in entry_paths.items():
                if os.path.exists(path) and info["file_size"] is not None:
                    actual_size = os.path.getsize(path)
                    if actual_size != info["file_size"]:
                        size_mismatches.append({
                            "cache_key": info["cache_key"],
                            "path": path,
                            "expected_size": info["file_size"],
                            "actual_size": actual_size,
                        })

            # 6. Check hash mismatches (optional, expensive)
            hash_mismatches = []
            if verify_hashes:
                for path, info in entry_paths.items():
                    if os.path.exists(path) and info.get("file_hash"):
                        current_hash = self._calculate_file_hash(Path(path))
                        if current_hash != info["file_hash"]:
                            hash_mismatches.append({
                                "cache_key": info["cache_key"],
                                "path": path,
                                "expected_hash": info["file_hash"],
                                "actual_hash": current_hash,
                            })

            # 7. Repair if requested
            repaired = {"orphans_deleted": 0, "dangling_removed": 0}
            if repair:
                for blob_path in orphaned_blobs:
                    try:
                        os.remove(blob_path)
                        repaired["orphans_deleted"] += 1
                    except OSError as e:
                        logger.warning(f"Failed to remove orphaned blob {blob_path}: {e}")

                for entry in dangling_entries:
                    try:
                        self.metadata_backend.remove_entry(entry["cache_key"])
                        repaired["dangling_removed"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove dangling entry {entry['cache_key']}: {e}")

            report = {
                "orphaned_blobs": orphaned_blobs,
                "dangling_entries": dangling_entries,
                "size_mismatches": size_mismatches,
            }
            if verify_hashes:
                report["hash_mismatches"] = hash_mismatches
            if repair:
                report["repaired"] = repaired

            total_issues = len(orphaned_blobs) + len(dangling_entries) + len(size_mismatches) + len(hash_mismatches)
            if total_issues == 0:
                logger.info("Cache integrity check passed — no issues found")
            else:
                logger.warning(
                    f"Cache integrity check found {total_issues} issue(s): "
                    f"{len(orphaned_blobs)} orphaned blobs, "
                    f"{len(dangling_entries)} dangling entries, "
                    f"{len(size_mismatches)} size mismatches"
                    + (f", {len(hash_mismatches)} hash mismatches" if verify_hashes else "")
                )

            return report

    def clear_all(self):
        """Clear all cache entries and remove cache files."""
        import glob
        
        with self._lock:
            # Clear metadata first and get count
            removed_count = self.metadata_backend.clear_all()
            
            # Remove all cache files - comprehensive pattern matching
            # Primary extensions
            cache_patterns = [
                str(self.cache_dir / "*.pkl"),      # Uncompressed pickle
                str(self.cache_dir / "*.npz"),      # NumPy arrays
                str(self.cache_dir / "*.b2nd"),     # Blosc2 arrays  
                str(self.cache_dir / "*.b2tr"),     # TensorFlow tensors with blosc2
                str(self.cache_dir / "*.parquet"),  # DataFrame files
            ]
            
            # Compressed pickle files with valid compression codecs
            pickle_codecs = ["lz4", "zstd", "gzip", "zst", "gz", "bz2", "xz"]
            for codec in pickle_codecs:
                cache_patterns.append(str(self.cache_dir / f"*.pkl.{codec}"))
            
            # Additional patterns for any other compression extensions user might use
            # This catches any .pkl.* pattern that might not be in our known list
            cache_patterns.append(str(self.cache_dir / "*.pkl.*"))
            
            files_removed = 0
            processed_files = set()  # Avoid double-counting due to overlapping patterns
            
            for pattern in cache_patterns:
                for file_path in glob.glob(pattern):
                    if file_path not in processed_files:
                        try:
                            os.remove(file_path)
                            files_removed += 1
                            processed_files.add(file_path)
                        except OSError as e:
                            logger.warning(f"Failed to remove cache file {file_path}: {e}")
            
            logger.info(f"Cleared {removed_count} cache entries and removed {files_removed} cache files")
            return removed_count

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.metadata_backend.get_stats()

        # Add cache-specific information
        stats.update(
            {
                "cache_dir": str(self.cache_dir),
                "max_size_mb": self.config.storage.max_cache_size_mb,
                "default_ttl_seconds": self.config.metadata.default_ttl_seconds,
                "backend_type": self.actual_backend,  # Report actual backend used
            }
        )

        return stats

    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries with metadata."""
        entries = self.metadata_backend.list_entries()

        # Add expiration status for each entry
        for entry in entries:
            entry["expired"] = self._is_expired(entry["cache_key"])

        return entries

    def close(self):
        """Close all resources (database connections, etc.)."""
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
        ttl_seconds: float = 21600,  # 6 hours
        ignore_errors: bool = True,
        **kwargs
    ) -> "UnifiedCache":
        """
        Create a cache optimized for API requests.
        
        Defaults:
        - TTL: 6 hours (21600 seconds, good for most API data)
        - ignore_errors: True (don't fail if cache has issues)
        - Compression: LZ4 (fast for JSON/text data)
        
        Args:
            cache_dir: Cache directory (default: ./cache)
            ttl_seconds: Time-to-live in seconds (default: 21600 = 6 hours)
            ignore_errors: Continue on cache errors
            **kwargs: Additional config options
        """
        config = create_cache_config(
            cache_dir=cache_dir or "./cache",
            default_ttl_seconds=ttl_seconds,
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
