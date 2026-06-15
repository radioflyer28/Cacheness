"""Update / existence / touch mixin for UnifiedCache."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ._put_cleanup import _PutCleanup

logger = logging.getLogger(__name__)


class UpdateMixin:
    """exists, update_data, and touch — entry-level mutation helpers."""

    def exists(
        self,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        check_expiration: bool = True,
        hash_key: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Check if a cache entry exists without loading the blob file.

        This is a lightweight metadata-only check that avoids loading large cached
        objects into memory. Useful for existence checks before calling get().

        Args:
            cache_key: Direct cache key (if provided, on and **kwargs are ignored)
            on: Dictionary of key parameters for cache key derivation.
                Use this to avoid namespace collisions with cache control parameters.
            check_expiration: If True, returns False for expired entries (default: True)
            **kwargs: Parameters identifying the cached data (legacy, use 'on' instead)

        Returns:
            bool: True if entry exists and is not expired, False otherwise

        Example:
            # Check before loading large DataFrame
            params = {'experiment': 'exp_001', 'run_id': 42}
            if cache.exists(on=params):
                df = cache.get(on=params)
            else:
                df = expensive_computation()
                cache.put(df, on=params)
        """
        with self._lock:
            cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
            cache_key = self._resolve_cache_key(cache_key, on, kwargs)

            entry = self.metadata_backend.get_entry(cache_key)
            if not entry:
                return False

            # Check expiration if requested
            if check_expiration and self._is_expired(cache_key, entry=entry):
                return False

            return True

    def update_data(
        self,
        data: Any,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        hash_key: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Update blob data at an existing cache entry without changing the cache_key.

        This replaces the stored data at a fixed cache_key while updating derived
        metadata (file_size, content_hash, created_at timestamp). The cache_key
        itself remains unchanged to maintain referential integrity.

        Args:
            data: New data to store (must be serializable by handler)
            cache_key: Direct cache key (if provided, on and **kwargs are ignored)
            on: Dictionary of key parameters for cache key derivation.
                Use this to avoid namespace collisions with cache control parameters.
            **kwargs: Parameters identifying the cached data (legacy, use 'on' instead)

        Returns:
            bool: True if entry was updated, False if entry doesn't exist

        Example:
            # Update cached DataFrame with new data
            success = cache.update_data(
                new_df,
                on={'experiment': 'exp_001', 'run_id': 42}
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
            cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
            cache_key = self._resolve_cache_key(cache_key, on, kwargs)

            # Check if entry exists before doing any I/O
            existing_entry = self.metadata_backend.get_entry(cache_key)
            if not existing_entry:
                logger.warning(
                    f"⚠️ Cache entry not found for update: {cache_key[:16]}..."
                )
                return False

            # Capture old blob path so we can delete it AFTER metadata
            # succeeds.  This is the write-then-swap strategy: write the
            # new blob to a staging location, update metadata, then
            # remove the old blob.  If metadata update fails, rollback
            # deletes only the staging blob; the old data stays intact.
            old_metadata = existing_entry.get("metadata", {})
            old_actual_path: Optional[str] = old_metadata.get("actual_path")

            base_file_path = self._get_cache_file_path(cache_key)
            staging_suffix = f"_stg{uuid.uuid4().hex[:8]}"
            staging_base = base_file_path.parent / (
                base_file_path.name + staging_suffix
            )
            cleanup = _PutCleanup()

            try:
                # Try zero-disk inline serialization first
                handler = self._blob_store.handlers.get_handler(data)
                direct = self._try_direct_inline(data, handler)

                if direct is not None:
                    result = direct["result"]
                    updates = {
                        "file_size": result.file_size,
                        "content_hash": result.extra.get("content_hash"),
                        "file_hash": direct["file_hash"],
                        "actual_path": None,
                        "storage_format": result.storage_format,
                        "blob_data": direct["blob_data"],
                        "is_inline": 1,
                        "inline_ext": direct["inline_ext"],
                    }
                    # Propagate encryption metadata from inline path (D-01)
                    if "encryption_algorithm" in direct:
                        updates["encryption_algorithm"] = direct["encryption_algorithm"]
                        updates["encryption_iv"] = direct["encryption_iv"]
                    if hasattr(handler, "data_type"):
                        updates["data_type"] = handler.data_type
                    if result.serializer:
                        updates["serializer"] = result.serializer
                    if result.compression_codec:
                        updates["compression_codec"] = result.compression_codec
                    if result.object_type:
                        updates["object_type"] = result.object_type
                else:
                    # Write new blob to staging path (different blob_id)
                    wb = self._blob_store._write_blob(
                        data, staging_base, compute_hash=False
                    )
                    handler, result = wb.handler, wb.result

                    actual_path_str = result.actual_path
                    if "://" not in actual_path_str:
                        cleanup.blob_path = self._resolve_actual_path(actual_path_str)

                    # Track remote blob for rollback on S3
                    if "://" in actual_path_str:
                        cleanup.set_remote(
                            self._blob_store.blob_backend, actual_path_str
                        )

                    # Build metadata updates dict from handler result
                    updates = {
                        "file_size": result.file_size,
                        "content_hash": result.extra.get("content_hash"),
                        "file_hash": result.extra.get("file_hash"),
                        "actual_path": actual_path_str,
                        "storage_format": result.storage_format,
                    }
                    if hasattr(handler, "data_type"):
                        updates["data_type"] = handler.data_type
                    if result.serializer:
                        updates["serializer"] = result.serializer
                    if result.compression_codec:
                        updates["compression_codec"] = result.compression_codec
                    if result.object_type:
                        updates["object_type"] = result.object_type
                    if result.extra.get("s3_etag"):
                        updates["s3_etag"] = result.extra["s3_etag"]

                    # Try disk-based inline (read back from file)
                    inline = self._try_inline_blob(result, None, cleanup)
                    if inline is not None:
                        updates["blob_data"] = inline["blob_data"]
                        updates["is_inline"] = 1
                        updates["actual_path"] = None
                        updates["inline_ext"] = inline["inline_ext"]
                        if inline["file_hash"] is not None:
                            updates["file_hash"] = inline["file_hash"]
                    else:
                        # Ensure previous inline data is cleared if blob is now external
                        updates["blob_data"] = None
                        updates["is_inline"] = 0
                        updates["inline_ext"] = None

                # Delegate metadata-only update to backend (no I/O in metadata layer)
                now = datetime.now(timezone.utc).isoformat()
                updates["created_at"] = now
                updates["accessed_at"] = now
                self.metadata_backend.update_entry_metadata(
                    cache_key=cache_key, updates=updates
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
                            if (
                                actual_path
                                and self.config.metadata.verify_cache_integrity
                            ):
                                new_file_hash = self._blob_store._calculate_file_hash(
                                    self._resolve_actual_path(actual_path)
                                )
                                metadata["file_hash"] = new_file_hash

                            # Extract signable fields and create new signature
                            complete_entry_data = self._extract_signable_fields(
                                cache_key=cache_key,
                                entry_data=updated_entry,
                                metadata=metadata,
                            )

                            # Generate new signature for updated entry
                            new_signature = self.signer.sign_entry(complete_entry_data)
                            metadata["entry_signature"] = new_signature

                            # Update entry with new signature and file hash
                            updated_entry["metadata"] = metadata
                            self.metadata_backend.put_entry(cache_key, updated_entry)

                            logger.debug(f"Re-signed updated entry {cache_key}")
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Failed to re-sign updated entry {cache_key}: {e}"
                        )
                        # Continue - update succeeded, just missing signature

                # Delete the OLD blob now that metadata points to the
                # new staging location.  Failure here is non-fatal: the
                # update already succeeded; the old blob is just orphaned.
                if old_actual_path and old_actual_path != actual_path_str:
                    try:
                        old_resolved = self._resolve_actual_path(old_actual_path)
                        self._blob_store.blob_backend.delete_blob(str(old_resolved))
                        logger.debug(
                            f"Deleted old blob after update: {old_actual_path}"
                        )
                    except (OSError, IOError):  # intentionally broad — orphan cleanup
                        logger.warning(
                            f"Failed to delete old blob after update: {old_actual_path}"
                        )

                logger.info(f"Updated cache entry: {cache_key[:16]}...")
                cleanup.commit()
                return True

            except Exception as e:  # intentionally broad — re-raises after cleanup
                cleanup.rollback()
                logger.error(
                    f"Failed to update cache entry {cache_key[:16]}...: "
                    f"{type(e).__name__}: {e}"
                )
                raise

    def touch(
        self,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        hash_key: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Update entry timestamp to extend TTL without reloading data.

        This "touches" the cache entry to reset its creation timestamp to now,
        effectively extending the entry's lifetime by the full configured TTL.
        Useful for keeping frequently accessed data alive or preventing
        expiration of long-running computations.

        Args:
            cache_key: Direct cache key (if provided, on and **kwargs are ignored)
            on: Dictionary of key parameters for cache key derivation.
                Use this to avoid namespace collisions with cache control parameters.
            **kwargs: Parameters identifying the cached data (legacy, use 'on' instead)

        Returns:
            bool: True if entry exists and was touched, False if entry doesn't exist

        Example:
            # Reset TTL to full default duration from now
            cache.touch(on={'experiment': 'exp_001'})

            # Keep long-running computation alive
            for i in range(100):
                process_chunk(i)
                if i % 10 == 0:
                    cache.touch(on={'job_id': 'long_job'})  # Prevent expiration

        Note:
            - This is a cache-layer operation (TTL-aware)
            - Resets ``created_at`` to now, giving a full config-TTL extension
            - Does not reload or re-serialize data — much faster than get() + put()
            - TTL duration is always determined by the global config
              (``CacheMetadataConfig.default_ttl_seconds``)
        """
        with self._lock:
            cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
            cache_key = self._resolve_cache_key(cache_key, on, kwargs)

            # Get existing entry
            entry = self.metadata_backend.get_entry(cache_key)
            if not entry:
                logger.warning(
                    f"⚠️ Cache entry not found for touch: {cache_key[:16]}..."
                )
                return False

            # Update timestamp to now (resets TTL)
            now = datetime.now(timezone.utc)
            entry["created_at"] = now.isoformat()
            entry["accessed_at"] = now.isoformat()

            # Re-sign if signing is enabled (timestamp is part of signature)
            if self.signer:
                try:
                    metadata = entry.get("metadata", {})
                    complete_entry_data = self._extract_signable_fields(
                        cache_key=cache_key,
                        entry_data=entry,
                        metadata=metadata,
                    )

                    # Generate new signature with updated timestamp
                    new_signature = self.signer.sign_entry(complete_entry_data)
                    metadata["entry_signature"] = new_signature
                    entry["metadata"] = metadata

                    logger.debug(f"Re-signed touched entry {cache_key}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to re-sign touched entry {cache_key}: {e}")
                    # Continue - touch succeeded, just missing signature

            # Store updated entry
            self.metadata_backend.put_entry(cache_key, entry)

            logger.info(f"👆 Touched cache entry: {cache_key[:16]}... (TTL extended)")
            return True
