"""Batch and bulk operations mixin for UnifiedCache."""

import logging
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)


class BatchMixin:
    """Bulk delete, batch put/get/delete/touch operations."""

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
