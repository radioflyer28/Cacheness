"""Storage-mode passthrough mixin for UnifiedCache."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StorageModeMixin:
    """Storage-mode put/get that bypass cache concerns (TTL, eviction, stats)."""

    def _storage_mode_put(
        self,
        data: Any,
        cache_key: str,
        description: str,
    ) -> str:
        """BlobStore passthrough for put() — no eviction or stats."""
        from .core import _PutCleanup

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

                entry_data = {
                    "data_type": handler.data_type,
                    "description": description,
                    "file_size": result.file_size,
                    "metadata": metadata_dict,
                    "blob_data": direct["blob_data"],
                    "is_inline": 1,
                }
            else:
                wb = self._blob_store._write_blob(
                    data, base_file_path, compute_hash=True
                )
                handler, result, file_hash = wb.handler, wb.result, wb.file_hash

                actual_path_str = result.actual_path
                if "://" not in actual_path_str:
                    cleanup.blob_path = self._resolve_actual_path(actual_path_str)
                if "://" in actual_path_str:
                    cleanup.set_remote(self._blob_store.blob_backend, actual_path_str)

                metadata_dict = self._build_metadata_dict(result, file_hash)

                # Record write intent for crash recovery (non-inline only)
                self._write_journal.record_intent(cache_key, result.actual_path)

                entry_data = {
                    "data_type": handler.data_type,
                    "description": description,
                    "file_size": result.file_size,
                    "metadata": metadata_dict,
                }

                # Try disk-based inline (read back from file)
                inline = self._try_inline_blob(result, file_hash, cleanup)
                if inline is not None:
                    entry_data["blob_data"] = inline["blob_data"]
                    entry_data["is_inline"] = 1
                    metadata_dict["actual_path"] = None
                    metadata_dict["inline_ext"] = inline["inline_ext"]
                    if inline["file_hash"] is not None:
                        metadata_dict["file_hash"] = inline["file_hash"]

            self._sign_entry_if_enabled(cache_key, entry_data, metadata_dict)
            self.metadata_backend.put_entry(cache_key, entry_data)

            # Clear write intent — metadata committed successfully
            self._write_journal.clear_intent(cache_key)
            self._cleanup_stale_blob(cache_key, old_blob_path, result.actual_path)

            logger.debug(f"Stored {handler.data_type} {cache_key} (storage mode)")
            cleanup.commit()
            return cache_key

        except Exception as e:  # intentionally broad — re-raises after cleanup
            cleanup.rollback()
            self._write_journal.clear_intent(cache_key)
            data_type = handler.data_type if "handler" in locals() else "unknown"
            logger.error(
                f"Failed to store {data_type} (storage mode): {type(e).__name__}: {e}"
            )
            raise

    def _storage_mode_get(
        self,
        cache_key: str,
    ) -> Optional[Any]:
        """BlobStore passthrough for get() — no TTL, stats, or auto-delete.

        Integrity and signature verification are still performed if enabled,
        but entries are never deleted on failure (storage-mode guarantee).
        """
        entry = self.metadata_backend.get_entry(cache_key)
        if entry is None:
            return None

        data_type = entry.get("data_type")
        if not data_type:
            return None

        metadata = entry.get("metadata", {})
        actual_path = metadata.get("actual_path")
        file_path = (
            self._resolve_actual_path(actual_path)
            if actual_path
            else self._get_cache_file_path(cache_key)
        )

        # Integrity + signature verification — never deletes in storage mode
        if not self._verify_entry(
            cache_key, entry, metadata, file_path, storage_mode=True
        ):
            return None

        try:
            if entry.get("is_inline") and entry.get("blob_data") is not None:
                data = self._read_inline_blob(entry, data_type, metadata)
            else:
                data = self._blob_store._read_blob(file_path, data_type, metadata)
            self.metadata_backend.update_access_time(cache_key)
            return data
        except Exception as e:  # intentionally broad — deserialization may fail any way
            # Never delete metadata in storage mode
            logger.warning(
                f"Failed to load {data_type} {cache_key}: "
                f"{type(e).__name__}: {e} (entry preserved, storage mode)"
            )
            return None

    def _storage_mode_get_with_metadata(
        self,
        cache_key: str,
    ) -> Optional[tuple[Any, Dict[str, Any]]]:
        """BlobStore passthrough for get_with_metadata() — no TTL, stats, or auto-delete."""
        entry = self.metadata_backend.get_entry(cache_key)
        if entry is None:
            return None

        data_type = entry.get("data_type")
        if not data_type:
            return None

        metadata = entry.get("metadata", {})
        actual_path = metadata.get("actual_path")
        file_path = (
            self._resolve_actual_path(actual_path)
            if actual_path
            else self._get_cache_file_path(cache_key)
        )

        # Integrity + signature verification — never deletes in storage mode
        if not self._verify_entry(
            cache_key, entry, metadata, file_path, storage_mode=True
        ):
            return None

        try:
            if entry.get("is_inline") and entry.get("blob_data") is not None:
                data = self._read_inline_blob(entry, data_type, metadata)
            else:
                data = self._blob_store._read_blob(file_path, data_type, metadata)
            self.metadata_backend.update_access_time(cache_key)
            entry["cache_key"] = cache_key
            return (data, entry)
        except Exception as e:  # intentionally broad — deserialization may fail any way
            logger.warning(
                f"Failed to load {data_type} {cache_key}: "
                f"{type(e).__name__}: {e} (entry preserved, storage mode)"
            )
            return None
