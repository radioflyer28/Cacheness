"""
Custom Metadata Models for Advanced Cache Querying
=================================================

This module provides SQLAlchemy-based custom metadata support with direct foreign key
relationships to cache entries for simple, clear ownership.

Features:
- Direct foreign key to cache_entries for one-to-many relationships
- Registry decorator pattern for clean schema definition
- SQLAlchemy ORM models for type safety and advanced querying
- Support for custom indexes and constraints
- Automatic cascade delete on cache entry removal

Usage:
    from cacheness.custom_metadata import custom_metadata_model, CustomMetadataBase
    from cacheness.metadata import Base
    from sqlalchemy import Column, String, Float, Integer, Text

    @custom_metadata_model("ml_experiments")
    class MLExperimentMetadata(Base, CustomMetadataBase):
        __tablename__ = "custom_ml_experiments"

        experiment_id = Column(String(100), nullable=False, unique=True, index=True)
        model_type = Column(String(50), nullable=False, index=True)
        accuracy = Column(Float, nullable=False, index=True)
        hyperparams = Column(Text, nullable=True)  # JSON
        created_by = Column(String(100), nullable=False, index=True)

    # Usage with cache
    cache = cacheness()

    metadata = MLExperimentMetadata(
        experiment_id="exp_001",
        model_type="xgboost",
        accuracy=0.95,
        created_by="alice"
    )

    cache.put(
        model_data,
        experiment="exp_001",
        custom_metadata={"ml_experiments": metadata}
    )

    # Advanced querying
    query = cache.query_custom_metadata("ml_experiments")
    results = query.filter(MLExperimentMetadata.accuracy >= 0.9).all()
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Type, Optional, List

try:
    from sqlalchemy import (
        Column,
        String,
        Integer,
        DateTime,
        Text,  # noqa: F401 — used at L27, L182
        ForeignKey,
    )
    from sqlalchemy.orm import relationship, declared_attr
    from .metadata import Base, SQLALCHEMY_AVAILABLE

    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy not available")

except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("SQLAlchemy not available, custom metadata will not work")

    # Dummy classes for when SQLAlchemy is not available
    class Base:
        pass

    def Column(*args, **kwargs):
        return None

    def relationship(*args, **kwargs):
        return None

    def declared_attr(f):
        return f


logger = logging.getLogger(__name__)

# Global registry for custom metadata models (schema_name → template class)
_custom_metadata_registry: Dict[str, Type] = {}

# Cache for namespace-specific custom metadata ORM classes.
# Key: (schema_name, namespace_id), Value: ORM class.
# For the default namespace, the template class from _custom_metadata_registry
# is reused directly.  For other namespaces, a dynamic class with the correct
# __tablename__ and FK target is created lazily and cached here.
_ns_custom_model_cache: Dict[tuple, Type] = {}

# Column names defined by CustomMetadataBase (not user-defined).
_BASE_COLUMN_NAMES = frozenset({"id", "cache_key", "created_at", "updated_at"})


if SQLALCHEMY_AVAILABLE:

    class CustomMetadataBase:
        """Base class for all custom metadata models with direct FK to cache entries."""

        @declared_attr
        def id(cls):
            """Primary key for the metadata table."""
            return Column(Integer, primary_key=True, autoincrement=True)

        @declared_attr
        def cache_key(cls):
            """Foreign key to cache_entries - each metadata record belongs to ONE cache entry."""
            return Column(
                String(16),
                ForeignKey("cache_entries.cache_key", ondelete="CASCADE"),
                nullable=False,
                index=True,
            )

        @declared_attr
        def created_at(cls):
            """Timestamp when the metadata was created."""
            return Column(
                DateTime(timezone=True),
                default=lambda: datetime.now(timezone.utc),
                nullable=False,
            )

        @declared_attr
        def updated_at(cls):
            """Timestamp when the metadata was last updated."""
            return Column(
                DateTime(timezone=True),
                default=lambda: datetime.now(timezone.utc),
                onupdate=lambda: datetime.now(timezone.utc),
                nullable=False,
            )

        def __repr__(self):
            cache_key_display = getattr(self, "cache_key", "N/A")
            return f"<{self.__class__.__name__}(id={self.id}, cache_key={cache_key_display})>"

        @classmethod
        def cleanup_orphaned_metadata(cls, session, cache_entry_model=None) -> int:
            """Remove metadata records with no corresponding cache entry.

            Args:
                session: SQLAlchemy session.
                cache_entry_model: The CacheEntry ORM class for the target
                    namespace.  When *None*, defaults to the canonical
                    ``CacheEntry`` (the default-namespace model).
            """
            from sqlalchemy import delete, select

            if cache_entry_model is None:
                from .metadata import CacheEntry

                cache_entry_model = CacheEntry

            # Find metadata with cache keys that don't exist in the target entries table
            orphaned_subquery = (
                select(cls.cache_key)
                .select_from(cls)
                .outerjoin(
                    cache_entry_model,
                    cls.cache_key == cache_entry_model.cache_key,
                )
                .where(cache_entry_model.cache_key.is_(None))
            )

            result = session.execute(
                delete(cls).where(cls.cache_key.in_(orphaned_subquery))
            )
            return result.rowcount if result.rowcount else 0

else:
    # Dummy class when SQLAlchemy is not available
    class CustomMetadataBase:
        pass


def custom_metadata_model(schema_name: str):
    """
    Decorator to register a SQLAlchemy model as a custom metadata schema.

    The model will have a direct foreign key to cache_entries.cache_key,
    establishing a clear one-to-many relationship (one cache entry can have
    many custom metadata records of different types).

    Args:
        schema_name: Unique name for the metadata schema

    Usage:
        @custom_metadata_model("ml_experiments")
        class MLExperimentMetadata(Base, CustomMetadataBase):
            __tablename__ = "custom_ml_experiments"

            experiment_id = Column(String(100), nullable=False, unique=True, index=True)
            model_type = Column(String(50), nullable=False, index=True)
            accuracy = Column(Float, nullable=False, index=True)
            hyperparams = Column(Text, nullable=True)  # JSON serialized
            created_by = Column(String(100), nullable=False, index=True)

    Returns:
        The decorated class, registered in the global schema registry

    Raises:
        ValueError: If SQLAlchemy is not available or invalid model definition
    """

    def decorator(cls):
        if not SQLALCHEMY_AVAILABLE:
            logger.warning(
                f"SQLAlchemy not available, custom metadata schema '{schema_name}' will not work"
            )
            return cls

        # Validate the model class
        if not issubclass(cls, CustomMetadataBase):
            raise ValueError(
                f"Custom metadata model {cls.__name__} must inherit from CustomMetadataBase"
            )

        if not hasattr(cls, "__tablename__") or not cls.__tablename__:
            raise ValueError(
                f"Custom metadata model {cls.__name__} must define __tablename__"
            )

        if not cls.__tablename__.startswith("custom_"):
            logger.warning(
                f"Custom metadata table name should start with 'custom_': {cls.__tablename__}"
            )

        # Check for duplicate schema names
        if schema_name in _custom_metadata_registry:
            existing_class = _custom_metadata_registry[schema_name]
            logger.warning(
                f"Schema name '{schema_name}' already registered with {existing_class.__name__}"
            )

        # Register the model
        _custom_metadata_registry[schema_name] = cls

        # Store schema name on the class for reverse lookup
        cls._schema_name = schema_name

        logger.debug(
            f"Registered custom metadata schema: {schema_name} -> {cls.__name__}"
        )

        return cls

    return decorator


def get_custom_metadata_model(schema_name: str) -> Optional[Type]:
    """
    Get a registered custom metadata model by name.

    Args:
        schema_name: Name of the schema to retrieve

    Returns:
        The model class if found, None otherwise
    """
    return _custom_metadata_registry.get(schema_name)


def get_all_custom_metadata_models() -> Dict[str, Type]:
    """
    Get all registered custom metadata models.

    Returns:
        Dictionary mapping schema names to model classes
    """
    return _custom_metadata_registry.copy()


def get_schema_name_for_model(model_class: Type) -> Optional[str]:
    """
    Get the schema name for a given model class.

    Args:
        model_class: The model class to look up

    Returns:
        Schema name if found, None otherwise
    """
    for schema_name, cls in _custom_metadata_registry.items():
        if cls == model_class:
            return schema_name
    return None


def list_registered_schemas() -> List[str]:
    """
    List all registered custom metadata schema names.

    Returns:
        List of schema names that have been registered
    """
    return list(_custom_metadata_registry.keys())


def _reset_registry():
    """
    Reset the custom metadata registry. FOR TESTING ONLY.

    This function is intended for use in test environments to ensure
    clean state between test runs.

    NOTE: ``_ns_custom_model_cache`` is intentionally NOT cleared here.
    Those dynamic ORM classes are registered in SQLAlchemy's ``MetaData``
    and remain valid.  Clearing and recreating them would cause duplicate
    index definitions.
    """
    global _custom_metadata_registry
    _custom_metadata_registry.clear()


# ---------------------------------------------------------------------------
# Namespace-aware custom metadata model factory
# ---------------------------------------------------------------------------


def _recreate_column(col):
    """Create a new Column mirroring *col*, detached from any table.

    Handles the common column attributes (type, nullable, unique, index,
    default, onupdate) and preserves any user-defined foreign keys.
    """
    kwargs: Dict[str, Any] = {"nullable": col.nullable}
    if col.unique:
        kwargs["unique"] = True
    if col.index:
        kwargs["index"] = True
    if col.primary_key:
        kwargs["primary_key"] = True
    if col.default is not None:
        kwargs["default"] = getattr(col.default, "arg", None)
    if col.onupdate is not None:
        kwargs["onupdate"] = getattr(col.onupdate, "arg", None)
    # Preserve user-defined FKs (unlikely on user columns, but possible)
    fk_args = [
        ForeignKey(fk.target_fullname, ondelete=fk.ondelete) for fk in col.foreign_keys
    ]
    return Column(col.type, *fk_args, **kwargs)


def get_namespace_custom_model(schema_name: str, namespace_id: str) -> Optional[Type]:
    """Return a namespace-specific custom metadata ORM model.

    For the **default** namespace the template class (registered via
    ``@custom_metadata_model``) is returned unchanged — its FK already
    points to ``cache_entries``.

    For any other namespace a dynamic ORM class is created (once, then
    cached) with:

    * ``__tablename__`` = ``<template_tablename>_<namespace_id>``
    * ``cache_key`` FK pointing to ``cache_entries_<namespace_id>``
    * All user-defined columns recreated as fresh ``Column`` instances

    The ``id``, ``created_at`` and ``updated_at`` columns are supplied by
    ``CustomMetadataBase`` via ``@declared_attr``.
    """
    from .metadata import DEFAULT_NAMESPACE

    template = _custom_metadata_registry.get(schema_name)
    if template is None:
        return None

    if namespace_id == DEFAULT_NAMESPACE:
        return template

    key = (schema_name, namespace_id)
    if key in _ns_custom_model_cache:
        return _ns_custom_model_cache[key]

    from .metadata import Base

    ns_tablename = f"{template.__tablename__}_{namespace_id}"
    entries_table = f"cache_entries_{namespace_id}"

    # --- Build the class dict ---
    # Override cache_key with namespace-specific FK target.
    class_dict: Dict[str, Any] = {
        "__tablename__": ns_tablename,
        "__table_args__": {"extend_existing": True},
        "_schema_name": schema_name,
        "cache_key": Column(
            String(16),
            ForeignKey(f"{entries_table}.cache_key", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
    }

    # Recreate user-defined columns (those NOT provided by CustomMetadataBase).
    for col in template.__table__.columns:
        if col.name not in _BASE_COLUMN_NAMES:
            class_dict[col.name] = _recreate_column(col)

    # Inherit from (Base, CustomMetadataBase) — the declared_attrs on
    # CustomMetadataBase will supply ``id``, ``created_at``, ``updated_at``.
    # The explicit ``cache_key`` in *class_dict* overrides the declared_attr.
    NsModel = type(
        f"{template.__name__}_{namespace_id}",
        (Base, CustomMetadataBase),
        class_dict,
    )

    _ns_custom_model_cache[key] = NsModel
    return NsModel


def convert_to_namespace_instance(source, namespace_model):
    """Copy column values from a template instance to a *namespace_model* instance.

    Users instantiate the template class (the one decorated with
    ``@custom_metadata_model``).  For non-default namespaces the ORM
    needs an instance of the namespace-specific class instead.  This
    helper copies all column values across.
    """
    target = namespace_model()
    for col in namespace_model.__table__.columns:
        if col.name == "id":
            continue  # Auto-increment PK — skip
        value = getattr(source, col.name, None)
        if value is not None:
            setattr(target, col.name, value)
    return target


def is_custom_metadata_available() -> bool:
    """
    Check if custom metadata functionality is available.

    Returns:
        True if SQLAlchemy is available and custom metadata can be used
    """
    return SQLALCHEMY_AVAILABLE


def validate_custom_metadata_model(model_class: Type) -> List[str]:
    """
    Validate a custom metadata model for common issues.

    Args:
        model_class: The model class to validate

    Returns:
        List of validation warnings/errors
    """
    issues = []

    if not SQLALCHEMY_AVAILABLE:
        issues.append("SQLAlchemy not available")
        return issues

    if not issubclass(model_class, CustomMetadataBase):
        issues.append("Model must inherit from CustomMetadataBase")

    if not hasattr(model_class, "__tablename__"):
        issues.append("Model must define __tablename__")
    elif not model_class.__tablename__.startswith("custom_"):
        issues.append("Table name should start with 'custom_' prefix")

    # Check for required indexes on commonly queried fields
    if hasattr(model_class, "__table__"):
        table = model_class.__table__
        indexed_columns = set()

        for index in table.indexes:
            for column in index.columns:
                indexed_columns.add(column.name)

        # Check for columns that should probably be indexed
        for column in table.columns:
            if (
                column.name.endswith("_id")
                or column.name in ["created_by", "model_type", "status"]
                or (
                    hasattr(column.type, "python_type")
                    and column.type.python_type in [float, int]
                )
            ):
                if column.name not in indexed_columns and not column.primary_key:
                    issues.append(
                        f"Consider adding index to column '{column.name}' for better query performance"
                    )

    return issues


def migrate_custom_metadata_tables(engine=None):
    """
    Create/update custom metadata tables for all registered schemas.

    This function ensures all registered custom metadata models have their
    tables created in the database. It's safe to call multiple times.

    Args:
        engine: SQLAlchemy engine (if None, will try to get from metadata backend)
    """
    if not SQLALCHEMY_AVAILABLE:
        logger.warning(
            "SQLAlchemy not available, cannot migrate custom metadata tables"
        )
        return

    try:
        # Import here to avoid circular imports
        from .metadata import Base

        if engine is None:
            # Try to get engine from a global cache instance
            try:
                from .core import get_cache

                cache = get_cache()
                if hasattr(cache.metadata_backend, "engine"):
                    engine = cache.metadata_backend.engine
                else:
                    logger.warning("No SQLAlchemy engine available for migration")
                    return
            except Exception:
                logger.warning("Could not access cache engine for migration")
                return

        # Create only the custom metadata tables
        # This avoids conflicts with cache_entries/cache_stats tables
        # which may be managed by different backends (SQLite vs PostgreSQL)
        #
        # Create tables individually so that one stale / duplicate-index
        # model doesn't prevent other tables from being created (CACHE-qg3).
        for schema_name, model_class in _custom_metadata_registry.items():
            if hasattr(model_class, "__table__") and model_class.__table__ is not None:
                try:
                    Base.metadata.create_all(engine, tables=[model_class.__table__])
                except Exception as table_err:
                    logger.debug(
                        f"Skipping table for schema '{schema_name}': {table_err}"
                    )

        logger.info(
            f"✅ Custom metadata tables migrated for {len(_custom_metadata_registry)} schemas"
        )

        # Validate all registered models
        for schema_name, model_class in _custom_metadata_registry.items():
            issues = validate_custom_metadata_model(model_class)
            if issues:
                logger.warning(
                    f"Schema '{schema_name}' validation issues: {', '.join(issues)}"
                )

    except Exception as e:
        logger.error(f"Failed to migrate custom metadata tables: {e}")


def cleanup_orphaned_metadata(
    engine=None, model_class=None, namespace_id: Optional[str] = None
) -> int:
    """
    Clean up orphaned custom metadata records that no longer have corresponding cache entries.

    With direct FK architecture, orphaned metadata should not exist (cascade delete handles it),
    but this utility is provided for manual cleanup if needed (e.g., after FK constraint issues).

    Args:
        engine: SQLAlchemy engine (if None, will try to get from metadata backend)
        model_class: Specific model class to clean (if None, cleans all registered models)
        namespace_id: Namespace to clean up in.  When *None*, defaults to the
            ``'default'`` namespace.

    Returns:
        Number of orphaned metadata records removed
    """
    if not SQLALCHEMY_AVAILABLE:
        logger.warning("SQLAlchemy not available, cannot cleanup orphaned metadata")
        return 0

    try:
        from .metadata import DEFAULT_NAMESPACE, _get_namespace_models

        ns = namespace_id or DEFAULT_NAMESPACE
        cache_entry_model, _ = _get_namespace_models(ns)

        if engine is None:
            try:
                from .core import get_cache

                cache = get_cache()
                if hasattr(cache.metadata_backend, "engine"):
                    engine = cache.metadata_backend.engine
                else:
                    logger.warning("No SQLAlchemy engine available for cleanup")
                    return 0
            except Exception:
                logger.warning("Could not access cache engine for cleanup")
                return 0

        # Get session
        from sqlalchemy.orm import sessionmaker

        SessionLocal = sessionmaker(bind=engine)

        with SessionLocal() as session:
            total_cleaned = 0

            # Clean specific model or all registered models
            models_to_clean = (
                [model_class]
                if model_class
                else list(_custom_metadata_registry.values())
            )

            for model in models_to_clean:
                if hasattr(model, "cleanup_orphaned_metadata"):
                    count = model.cleanup_orphaned_metadata(
                        session, cache_entry_model=cache_entry_model
                    )
                    total_cleaned += count

            if total_cleaned > 0:
                logger.info(
                    f"🧹 Cleaned up {total_cleaned} orphaned custom metadata records"
                )

            session.commit()
            return total_cleaned

    except Exception as e:
        logger.error(f"Failed to cleanup orphaned metadata: {e}")
        return 0


def export_custom_metadata_schema(
    schema_name: str, output_file: Optional[str] = None
) -> Optional[str]:
    """
    Export the schema definition for a custom metadata model as SQL DDL.

    Args:
        schema_name: Name of the schema to export
        output_file: Optional file path to write DDL (if None, returns as string)

    Returns:
        SQL DDL string if output_file is None, otherwise None
    """
    if not SQLALCHEMY_AVAILABLE:
        logger.warning("SQLAlchemy not available, cannot export schema")
        return None

    model_class = get_custom_metadata_model(schema_name)
    if not model_class:
        logger.warning(f"Schema '{schema_name}' not found")
        return None

    try:
        from sqlalchemy import create_engine
        from sqlalchemy.schema import CreateTable

        # Create a dummy engine to generate DDL
        engine = create_engine("sqlite:///:memory:")

        # Generate DDL for the table
        table_ddl = str(CreateTable(model_class.__table__).compile(engine))

        full_ddl = f"""-- Custom Metadata Schema: {schema_name}
-- Model: {model_class.__name__}
-- Table: {model_class.__tablename__}

-- Metadata table (direct FK to cache_entries)
{table_ddl};

-- Indexes and constraints are included in the table definition above
"""

        if output_file:
            with open(output_file, "w") as f:
                f.write(full_ddl)
            logger.info(f"Schema DDL exported to {output_file}")
            return None
        else:
            return full_ddl

    except Exception as e:
        logger.error(f"Failed to export schema {schema_name}: {e}")
        return None
