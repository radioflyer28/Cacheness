"""Custom metadata operations mixin for UnifiedCache."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CustomMetadataMixin:
    """Custom metadata storage, retrieval, and querying."""

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
        if hasattr(custom_metadata, "_schema_name") or hasattr(
            type(custom_metadata), "_schema_name"
        ):
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

    def _store_custom_metadata(self, cache_key: str, custom_metadata):
        """Store custom metadata using link table architecture."""
        if not self._supports_custom_metadata():
            logger.warning(
                "Custom metadata not supported - requires SQLite or PostgreSQL backend"
            )
            return

        try:
            from .custom_metadata import (
                get_custom_metadata_model,
                get_namespace_custom_model,
                convert_to_namespace_instance,
            )
            from .metadata import Base

            # Normalize custom_metadata to iterable of metadata objects
            metadata_objects = self._normalize_custom_metadata(custom_metadata)
            if not metadata_objects:
                return

            # Get SQLAlchemy session from the metadata backend
            if hasattr(self.metadata_backend, "SessionLocal"):
                with self.metadata_backend.SessionLocal() as session:
                    # Resolve namespace-specific model classes and ensure
                    # their tables exist.  For non-default namespaces the
                    # model (and therefore the table) is created dynamically.
                    tables_to_create = set()
                    for metadata_instance in metadata_objects:
                        schema_name = getattr(
                            type(metadata_instance), "_schema_name", None
                        )
                        if schema_name:
                            ns_model = get_namespace_custom_model(
                                schema_name, self.namespace
                            )
                            if ns_model is not None:
                                tbl = getattr(ns_model, "__table__", None)
                                if tbl is not None:
                                    tables_to_create.add(tbl)
                    if tables_to_create:
                        # Create tables individually so that a stale index
                        # or duplicate definition on one table doesn't block
                        # the others (mirrors migrate_custom_metadata_tables).
                        for tbl in tables_to_create:
                            try:
                                Base.metadata.create_all(
                                    self.metadata_backend.engine,
                                    tables=[tbl],
                                )
                            except Exception:
                                pass  # table already exists — fine

                    for metadata_instance in metadata_objects:
                        # Get schema name from the metadata object's class
                        schema_name = getattr(
                            type(metadata_instance), "_schema_name", None
                        )
                        if not schema_name:
                            logger.warning(
                                f"Metadata object {type(metadata_instance).__name__} is not properly registered"
                            )
                            continue

                        template_class = get_custom_metadata_model(schema_name)
                        if not template_class:
                            logger.warning(
                                f"Unknown custom metadata schema: {schema_name}"
                            )
                            continue

                        # Ensure metadata instance is of the correct template type
                        if not isinstance(metadata_instance, template_class):
                            logger.warning(
                                f"Invalid metadata type for schema {schema_name}"
                            )
                            continue

                        # Resolve the namespace-specific model
                        ns_model = get_namespace_custom_model(
                            schema_name, self.namespace
                        )
                        if ns_model is None:
                            continue

                        # For non-default namespaces, convert the template
                        # instance to the namespace-specific model class.
                        if ns_model is not type(metadata_instance):
                            metadata_instance = convert_to_namespace_instance(
                                metadata_instance, ns_model
                            )

                        # Set the cache_key on the metadata instance (direct FK)
                        metadata_instance.cache_key = cache_key

                        # Delete any existing custom metadata for this
                        # cache_key + schema before inserting.  This ensures
                        # overwriting a cache entry replaces its custom
                        # metadata rather than accumulating duplicate rows.
                        session.query(ns_model).filter(
                            ns_model.cache_key == cache_key
                        ).delete()

                        # Save the metadata instance
                        session.add(metadata_instance)

                    session.commit()
                    logger.debug(f"Stored custom metadata for cache key {cache_key}")
        except Exception as e:
            logger.error(f"Failed to store custom metadata: {e}")

    def _get_custom_metadata(self, cache_key: str) -> Dict[str, Any]:
        """Retrieve custom metadata for a cache key in the current namespace."""
        if not self._supports_custom_metadata():
            return {}

        try:
            if hasattr(self.metadata_backend, "SessionLocal"):
                with self.metadata_backend.SessionLocal() as session:
                    from sqlalchemy import select
                    from sqlalchemy.exc import OperationalError as SAOperationalError
                    from .custom_metadata import get_namespace_custom_model

                    result = {}
                    # Query each registered schema's namespace-specific
                    # table for metadata with this cache_key.  Skip schemas
                    # whose tables don't exist in this database (the global
                    # registry may contain models from other sessions/tests
                    # — CACHE-qg3).
                    for schema_name in self._get_registered_schemas():
                        ns_model = get_namespace_custom_model(
                            schema_name, self.namespace
                        )
                        if ns_model is None:
                            continue
                        try:
                            metadata_instance = session.execute(
                                select(ns_model).where(ns_model.cache_key == cache_key)
                            ).scalar_one_or_none()

                            if metadata_instance:
                                result[schema_name] = metadata_instance
                        except SAOperationalError:
                            # Table doesn't exist in this database — skip
                            session.rollback()

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

    def query_custom(
        self, schema_name: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
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
            from .custom_metadata import get_namespace_custom_model

            ns_model = get_namespace_custom_model(schema_name, self.namespace)
            if not ns_model:
                logger.warning(f"Unknown custom metadata schema: {schema_name}")
                return []

            if hasattr(self.metadata_backend, "SessionLocal"):
                # Use context manager to ensure proper session cleanup
                with self.metadata_backend.SessionLocal() as session:
                    query = session.query(ns_model)

                    # Apply optional filters
                    if filters:
                        for field_name, value in filters.items():
                            if hasattr(ns_model, field_name):
                                query = query.filter(
                                    getattr(ns_model, field_name) == value
                                )
                            else:
                                logger.warning(
                                    f"Unknown filter field '{field_name}' for schema '{schema_name}'"
                                )

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
                raise ValueError(
                    "Custom metadata querying not supported - requires SQLite or PostgreSQL backend"
                )

            from .custom_metadata import get_namespace_custom_model

            ns_model = get_namespace_custom_model(schema_name, self.namespace)
            if not ns_model:
                raise ValueError(f"Unknown custom metadata schema: {schema_name}")

            if not hasattr(self.metadata_backend, "SessionLocal"):
                raise ValueError("SQLAlchemy session not available")

            session = self.metadata_backend.SessionLocal()
            try:
                yield session.query(ns_model)
            finally:
                session.close()

        return _session_context()

    def query_custom_metadata(
        self, schema_name: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Query custom metadata for a specific schema.

        **Deprecated:** Use query_custom() instead for shorter syntax.

        Args:
            schema_name: Name of the custom metadata schema to query
            filters: Optional dict of field_name -> value for equality filtering

        Returns:
            List of results (empty list if not supported or on error)
        """
        logger.warning(
            "query_custom_metadata() is deprecated, use query_custom() instead"
        )
        return self.query_custom(schema_name, filters)

    def get_custom_metadata_for_entry(
        self,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        hash_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get custom metadata for a specific cache entry.

        Args:
            cache_key: Direct cache key (if provided, on and **kwargs are ignored)
            on: Dictionary of key parameters for cache key derivation.
                Use this to avoid namespace collisions with cache control parameters.
            **kwargs: Parameters identifying the cached data (legacy, use 'on' instead)

        Returns:
            Dictionary mapping schema names to metadata instances
        """
        cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
        cache_key = self._resolve_cache_key(cache_key, on, kwargs)

        return self._get_custom_metadata(cache_key)
