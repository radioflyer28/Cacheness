"""Metadata query mixin for UnifiedCache."""

import logging

from .entry_list import EntryList

logger = logging.getLogger(__name__)


class QueryMixin:
    """Backend-specific metadata query operations."""

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
