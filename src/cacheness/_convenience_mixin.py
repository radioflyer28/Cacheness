"""Convenience metadata helpers mixin for UnifiedCache."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConvenienceMixin:
    """put/get_with_meta, put/get_with_model, query_with_meta/model."""

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
