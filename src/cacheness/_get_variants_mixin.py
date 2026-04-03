"""Get-variant operations mixin for UnifiedCache."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from .config import _DEFAULT_TTL

logger = logging.getLogger(__name__)


class GetVariantsMixin:
    """get_with_metadata and get_metadata — metadata-aware retrieval."""

    def get_with_metadata(
        self,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        ttl_seconds=_DEFAULT_TTL,
        hash_key: Optional[str] = None,
        **kwargs,
    ) -> Optional[tuple[Any, Dict[str, Any]]]:
        """
        Retrieve cached data along with its metadata in a single atomic operation.

        This method combines get() and get_metadata() into one call, avoiding
        separate metadata lookups. Useful when you need both the data and its
        metadata (e.g., created_at, file_size, custom metadata).

        Args:
            cache_key: Direct cache key (if provided, on and **kwargs are ignored)
            hash_key: Alias for cache_key (storage-oriented name).
            on: Dictionary of key parameters for cache key derivation.
                Use this to avoid namespace collisions with cache control parameters.
            ttl_seconds: Custom TTL in seconds (overrides default). None = never expire.
                         Use _DEFAULT_TTL sentinel (default) to use config's default_ttl_seconds.
            **kwargs: Parameters identifying the cached data (legacy, use 'on' instead)

        Returns:
            Tuple of (data, metadata_dict) if found and not expired, None otherwise

        Example:
            result = cache.get_with_metadata(on={'experiment': 'exp_001'})
            if result:
                data, metadata = result
                print(f"Created: {metadata['created_at']}")
                print(f"Size: {metadata.get('file_size_bytes', 0)} bytes")
                process(data)
        """
        with self._lock:
            cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
            cache_key = self._resolve_cache_key(cache_key, on, kwargs)

            if self.config.storage_mode:
                return self._storage_mode_get_with_metadata(cache_key)

            # Single metadata lookup
            entry = self.metadata_backend.get_entry(cache_key)
            if not entry:
                self._record_miss()
                return None

            # Check TTL expiration directly using the already-retrieved entry
            # to avoid a second metadata lookup
            if ttl_seconds is _DEFAULT_TTL:  # Use config default
                actual_ttl = self.config.metadata.default_ttl_seconds
            else:  # Use provided TTL (could be None for infinite or a specific value)
                actual_ttl = ttl_seconds

            # If TTL is not None (infinite), check expiration
            if actual_ttl is not None:
                creation_time_str = entry.get("created_at")
                if creation_time_str:
                    if isinstance(creation_time_str, str):
                        creation_time = datetime.fromisoformat(creation_time_str)
                    else:
                        creation_time = creation_time_str

                    if creation_time.tzinfo is None:
                        creation_time = creation_time.replace(tzinfo=timezone.utc)

                    expiry_time = creation_time + timedelta(seconds=actual_ttl)
                    current_time = datetime.now(timezone.utc)

                    if current_time > expiry_time:
                        self._record_miss()
                        return None

            # Get appropriate handler
            data_type = entry.get("data_type")
            if not data_type:
                self._record_miss()
                return None

            try:
                base_file_path = self._get_cache_file_path(cache_key)

                # Use actual path from metadata if available, otherwise use base path
                metadata = entry.get("metadata", {})
                actual_path = metadata.get("actual_path")
                if actual_path:
                    file_path = self._resolve_actual_path(actual_path)
                else:
                    file_path = base_file_path

                # Integrity + signature verification
                if not self._verify_entry(cache_key, entry, metadata, file_path):
                    self._record_miss()
                    return None

                # Delegate blob read to BlobStore
                if entry.get("is_inline") and entry.get("blob_data") is not None:
                    data = self._read_inline_blob(entry, data_type, metadata)
                else:
                    data = self._blob_store._read_blob(file_path, data_type, metadata)

                # Update access time
                self.metadata_backend.update_access_time(cache_key)
                self._record_hit()

                # Include cache_key in returned metadata
                entry["cache_key"] = cache_key

                logger.debug(f"Cache hit with metadata ({data_type}): {cache_key}")
                return (data, entry)

            except FileNotFoundError as e:
                # Cache file was deleted externally — blob already gone,
                # just clean up metadata (no blob to delete)
                logger.warning(f"Cache file missing for {cache_key}: {e}")
                self.metadata_backend.remove_entry(cache_key)
                self._record_miss()
                return None
            except (OSError, IOError) as e:
                # I/O errors may be transient (disk temporarily unavailable, etc.)
                # Do NOT delete metadata — the entry may be readable on retry
                logger.warning(f"I/O error loading cached {data_type} {cache_key}: {e}")
                self._record_miss()
                return None
            except (
                Exception
            ) as e:  # intentionally broad — deserialization may fail any way
                # Unexpected errors (deserialization failures, corruption, etc.)
                if self.config.metadata.delete_on_error:
                    logger.warning(
                        f"Failed to load cached {data_type} {cache_key}: {type(e).__name__}: {e}. "
                        f"Removing corrupted cache entry."
                    )
                    self._blob_store.delete(cache_key)
                else:
                    logger.warning(
                        f"Failed to load cached {data_type} {cache_key}: {type(e).__name__}: {e}. "
                        f"Entry retained due to delete_on_error=False."
                    )
                self._record_miss()
                return None

    def get_metadata(
        self,
        cache_key: Optional[str] = None,
        on: Optional[Dict] = None,
        check_expiration: bool = True,
        hash_key: Optional[str] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Get entry metadata without loading blob data.

        This is useful for inspecting cache entries (TTL, file size, data type)
        before deciding whether to load the actual data.

        Args:
            cache_key: Direct cache key (if provided, on and **kwargs are ignored)
            on: Dictionary of key parameters for cache key derivation.
                Use this to avoid namespace collisions with cache control parameters.
            check_expiration: If True, returns None for expired entries (default: True)
            **kwargs: Parameters identifying the cached data (legacy, use 'on' instead)

        Returns:
            Metadata dictionary or None if not found or expired

        Example:
            # Check metadata before loading large file
            meta = cache.get_metadata(on={'experiment': 'exp_001'})
            if meta and meta.get("file_size_bytes", 0) > 1e9:
                print("Large file - loading may take time")
                data = cache.get(on={'experiment': 'exp_001'})
        """
        with self._lock:
            cache_key = self._resolve_hash_key_alias(cache_key, hash_key)
            cache_key = self._resolve_cache_key(cache_key, on, kwargs)

            entry = self.metadata_backend.get_entry(cache_key)
            if not entry:
                return None

            # Check expiration if requested (respects cache TTL policy)
            if check_expiration and self._is_expired(cache_key):
                return None

            # Ensure cache_key is included in returned metadata
            entry["cache_key"] = cache_key

            return entry
