"""JSON file-based metadata backend."""

import logging
import os
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..interfaces import EntrySummary
from ..json_utils import dumps as json_dumps, loads as json_loads
from ..size_utils import bytes_to_mb_display
from ._compat import (
    DEFAULT_NAMESPACE,
    validate_namespace_id,
    NamespaceInfo,
)
from .base import MetadataBackend

logger = logging.getLogger(__name__)


class JsonBackend(MetadataBackend):
    """JSON file-based metadata backend with batching support."""

    def __init__(self, metadata_file: Path, namespace: str = DEFAULT_NAMESPACE):
        """
        Initialize JSON metadata backend.

        Args:
            metadata_file: Path to the *default* namespace's JSON metadata file.
                Non-default namespaces automatically resolve to a
                ``{namespace_id}_metadata.json`` file in the same directory.
            namespace: Active namespace for this backend instance
        """
        self._active_namespace = validate_namespace_id(namespace)
        # Store the root metadata file for _metadata_file_for_namespace()
        self._root_metadata_file = Path(metadata_file)
        # Resolve to the namespace-specific file (default → metadata_file as-is)
        self.metadata_file = self._metadata_file_for_namespace(namespace)
        self._lock = (
            threading.RLock()
        )  # Use RLock to allow reentrant calls (cleanup_by_size -> get_stats)
        self._metadata = self._load_from_disk()

        # --- Namespace registry ---
        # Registry lives in the same directory as the root metadata file.
        self._registry_file = (
            self._root_metadata_file.parent / "cacheness_namespaces.json"
        )
        self._ensure_namespace_registry()
        self.run_all_migrations()

    def _load_from_disk(self) -> Dict[str, Any]:
        """Load metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    data = json_loads(f.read())
                # Validate schema: must be a dict with an "entries" key
                if not isinstance(data, dict) or "entries" not in data:
                    logger.warning("JSON metadata has invalid schema, starting fresh")
                else:
                    return data
            except Exception:
                logger.warning("JSON metadata corrupted, starting fresh")

        return {
            "entries": {},  # cache_key -> complete entry dict (with structured fields)
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def _save_to_disk(self):
        """Save metadata to JSON file using atomic write pattern to prevent corruption."""
        import tempfile

        try:
            # Check if parent directory exists (may have been deleted during cleanup)
            if not self.metadata_file.parent.exists():
                logger.debug(
                    f"Metadata directory no longer exists: {self.metadata_file.parent}"
                )
                return

            # Write to temp file first, then rename for atomicity
            fd, temp_path = tempfile.mkstemp(
                suffix=".json.tmp",
                dir=self.metadata_file.parent,
                prefix="cache_metadata_",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(json_dumps(self._metadata, default=str))
                # Atomic rename (works on same filesystem)
                import shutil

                shutil.move(temp_path, self.metadata_file)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.error(f"Failed to save JSON metadata: {e}")

    def get_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get specific cache entry metadata (simple entry lookup)."""
        with self._lock:
            # Simple lookup - entry contains all structured fields
            entry = self._metadata.get("entries", {}).get(cache_key)
            if entry is None:
                return None

            # Entry already contains the structured fields matching SQLite schema
            return entry

    def put_entry(self, cache_key: str, entry_data: Dict[str, Any]):
        """Store cache entry metadata as complete entry (matching SQLite schema structure)."""
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()

            # Extract and restructure metadata to match SQLite schema
            metadata = entry_data.get("metadata", {}).copy()

            # Build complete entry with structured fields (matching SQLite columns)
            entry = {
                "description": entry_data.get("description", ""),
                "data_type": entry_data.get("data_type", "unknown"),
                "created_at": entry_data.get("created_at", now),
                "accessed_at": entry_data.get("accessed_at", now),
                "file_size": entry_data.get("file_size", 0),
                "access_count": entry_data.get("access_count", 0),
                "is_inline": entry_data.get("is_inline", 0),
                "metadata": metadata,  # Include all metadata as nested structure
            }
            # Note: blob_data (bytes) is intentionally NOT stored in JSON backend —
            # inline blobs are only supported by SQLite/PG backends with binary columns.

            # Store per-entry TTL and pre-computed expires_at if provided
            ttl_val = entry_data.get("ttl_seconds")
            if ttl_val is not None:
                entry["ttl_seconds"] = ttl_val
                # Compute expires_at from created_at + ttl_seconds
                created = entry["created_at"]
                if isinstance(created, str):
                    created_dt = datetime.fromisoformat(created)
                else:
                    created_dt = created
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                entry["expires_at"] = (
                    created_dt + timedelta(seconds=float(ttl_val))
                ).isoformat()

            # Store complete entry - simple and efficient
            self._metadata["entries"][cache_key] = entry
            self._save_to_disk()

    def remove_entry(self, cache_key: str) -> bool:
        """Remove cache entry metadata."""
        with self._lock:
            entries = self._metadata.get("entries", {})
            if cache_key in entries:
                del entries[cache_key]
                self._save_to_disk()
                return True
            return False

    def update_entry_metadata(self, cache_key: str, updates: Dict[str, Any]) -> bool:
        """
        Update metadata fields for an existing cache entry.

        Only updates metadata — blob I/O is handled by UnifiedCache.update_data().

        Args:
            cache_key: The unique identifier for the cache entry to update
            updates: Dict of metadata fields to update (file_size, file_hash,
                    actual_path, data_type, storage_format, serializer, etc.)

        Returns:
            bool: True if entry was updated, False if entry doesn't exist
        """
        with self._lock:
            entries = self._metadata.get("entries", {})
            entry = entries.get(cache_key)
            if not entry:
                return False

            # Update derived metadata (file_size, content_hash, created_at)
            now = datetime.now(timezone.utc)
            entry["created_at"] = now.isoformat()  # Reset timestamp

            if "file_size" in updates:
                entry["file_size"] = updates["file_size"]
            if "data_type" in updates:
                entry["data_type"] = updates["data_type"]
            if "storage_format" in updates:
                entry["storage_format"] = updates.get(
                    "storage_format", entry.get("storage_format")
                )
            if "serializer" in updates:
                entry["serializer"] = updates["serializer"]

            # Update metadata dict with new values
            metadata = entry.get("metadata", {})
            for key in (
                "file_size",
                "content_hash",
                "file_hash",
                "actual_path",
                "storage_format",
                "serializer",
            ):
                if key in updates:
                    metadata[key] = updates[key]
            entry["metadata"] = metadata

            # Save to disk immediately for atomicity
            self._save_to_disk()
            return True

    def iter_entry_summaries(self) -> List[EntrySummary]:
        """Return lightweight flat entry dicts for internal filtering."""
        with self._lock:
            result: List[EntrySummary] = []
            for cache_key, entry in self._metadata.get("entries", {}).items():
                flat: Dict[str, Any] = {
                    "cache_key": cache_key,
                    "data_type": entry.get("data_type", "unknown"),
                    "description": entry.get("description", ""),
                    "created_at": entry.get("created_at"),
                    "accessed_at": entry.get("accessed_at"),
                    "file_size": entry.get("file_size", 0),
                    "access_count": entry.get("access_count", 0),
                }
                # Include TTL/expiry fields if present
                if "ttl_seconds" in entry:
                    flat["ttl_seconds"] = entry["ttl_seconds"]
                if "expires_at" in entry:
                    flat["expires_at"] = entry["expires_at"]
                # Merge technical metadata fields flat
                for k, v in entry.get("metadata", {}).items():
                    if k not in flat:
                        flat[k] = v
                result.append(flat)  # type: ignore[arg-type]
            return result

    def list_entries(self) -> List[Dict[str, Any]]:
        """List all cache entries with metadata (simple entries iteration)."""
        with self._lock:
            entries = []

            # Simple iteration over entries dict
            for cache_key, entry in self._metadata.get("entries", {}).items():
                # Ensure timestamps are timezone-aware when returned
                creation_time = entry.get("created_at")
                access_time = entry.get("accessed_at")

                if creation_time and isinstance(creation_time, str):
                    try:
                        # Parse ISO format string back to timezone-aware datetime
                        creation_time = datetime.fromisoformat(creation_time)
                        if creation_time.tzinfo is None:
                            # If somehow it's naive, make it UTC
                            creation_time = creation_time.replace(tzinfo=timezone.utc)
                        creation_time = creation_time.isoformat()
                    except (ValueError, TypeError):
                        pass

                if access_time and isinstance(access_time, str):
                    try:
                        # Parse ISO format string back to timezone-aware datetime
                        access_time = datetime.fromisoformat(access_time)
                        if access_time.tzinfo is None:
                            # If somehow it's naive, make it UTC
                            access_time = access_time.replace(tzinfo=timezone.utc)
                        access_time = access_time.isoformat()
                    except (ValueError, TypeError):
                        pass

                # Build entry for list output
                list_entry = {
                    "cache_key": cache_key,
                    "data_type": entry.get("data_type", "unknown"),
                    "description": entry.get("description", ""),
                    "metadata": entry.get("metadata", {}),
                    "created": creation_time,
                    "last_accessed": access_time,
                    "size_mb": bytes_to_mb_display(entry.get("file_size", 0)),
                }

                entries.append(list_entry)

            # Sort by creation time (newest first)
            entries.sort(key=lambda x: x["created"] or "", reverse=True)
            return entries

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics (simple entries-based counting)."""
        with self._lock:
            entries = self._metadata.get("entries", {})

            # Count total entries
            total_entries = len(entries)

            # Calculate total size in bytes (canonical)
            total_size_bytes = sum(
                entry.get("file_size", 0) for entry in entries.values()
            )
            total_size_mb = total_size_bytes / (1024 * 1024)

            # Count by data type
            dataframe_count = sum(
                1 for entry in entries.values() if entry.get("data_type") == "dataframe"
            )
            array_count = sum(
                1 for entry in entries.values() if entry.get("data_type") == "array"
            )

            # Cache hit rate
            hits = self._metadata.get("cache_hits", 0)
            misses = self._metadata.get("cache_misses", 0)
            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0

            return {
                "total_entries": total_entries,
                "dataframe_entries": dataframe_count,
                "array_entries": array_count,
                "total_size_bytes": total_size_bytes,
                "total_size_mb": total_size_mb,  # Backward compat — prefer total_size_bytes
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate": round(hit_rate, 3),
            }

    def update_access_time(self, cache_key: str):
        """Update last access time and increment access count for cache entry."""
        with self._lock:
            entries = self._metadata.get("entries", {})
            if cache_key in entries:
                entries[cache_key]["accessed_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                entries[cache_key]["access_count"] = (
                    entries[cache_key].get("access_count", 0) + 1
                )
                self._save_to_disk()

    def increment_hits(self):
        """Increment cache hits counter."""
        with self._lock:
            self._metadata["cache_hits"] = self._metadata.get("cache_hits", 0) + 1
            self._save_to_disk()

    def increment_misses(self):
        """Increment cache misses counter."""
        with self._lock:
            self._metadata["cache_misses"] = self._metadata.get("cache_misses", 0) + 1
            self._save_to_disk()

    def cleanup_expired(self, ttl_seconds: float) -> int:
        """Remove expired entries and return count removed (simple entries structure)."""

        with self._lock:
            expired_keys = []
            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)
            entries = self._metadata.get("entries", {})

            for cache_key, entry in entries.items():
                try:
                    creation_time_str = entry.get("created_at")
                    if creation_time_str:
                        creation_time = datetime.fromisoformat(creation_time_str)
                        if creation_time < cutoff_time:
                            expired_keys.append(cache_key)
                except (ValueError, TypeError):
                    # Invalid timestamp, consider expired
                    expired_keys.append(cache_key)

            # Remove expired entries
            for cache_key in expired_keys:
                entries.pop(cache_key, None)

            if expired_keys:
                self._save_to_disk()

            return len(expired_keys)

    def cleanup_by_size(self, target_size_bytes: int) -> Dict[str, Any]:
        """Remove least-recently-accessed entries until cache size drops to or below target."""
        with self._lock:
            # Get current total size in bytes
            stats = self.get_stats()
            current_size_bytes = stats.get("total_size_bytes", 0)

            logger.debug(
                f"cleanup_by_size: current size {current_size_bytes} bytes, target {target_size_bytes} bytes"
            )

            if current_size_bytes <= target_size_bytes:
                return {"count": 0, "removed_entries": []}  # Already at or below target

            entries = self._metadata.get("entries", {})
            logger.debug(f"cleanup_by_size: {len(entries)} total entries")

            # Sort entries by last_accessed (oldest first) for LRU eviction
            sorted_entries = sorted(
                entries.items(),
                key=lambda item: item[1].get(
                    "accessed_at", item[1].get("created_at", "")
                ),
            )

            # Calculate how many bytes we need to remove
            bytes_to_remove = current_size_bytes - target_size_bytes

            logger.debug(f"cleanup_by_size: need to remove {bytes_to_remove:.0f} bytes")

            removed_entries = []
            accumulated_bytes = 0

            # Remove entries until we've freed up enough space
            for cache_key, entry in sorted_entries:
                if accumulated_bytes >= bytes_to_remove:
                    break

                # file_size is stored at top level of entry dict
                entry_size_bytes = entry.get("file_size", 0)

                # actual_path is stored in nested metadata dict
                actual_path = entry.get("metadata", {}).get("actual_path") or entry.get(
                    "actual_path"
                )

                # Remove entry
                entries.pop(cache_key, None)
                removed_entries.append(
                    {"cache_key": cache_key, "actual_path": actual_path}
                )
                accumulated_bytes += entry_size_bytes

                logger.debug(
                    f"cleanup_by_size: removed {cache_key}, freed {entry_size_bytes} bytes (total freed: {accumulated_bytes})"
                )

            logger.debug(
                f"cleanup_by_size: done, removed {len(removed_entries)} entries"
            )

            if removed_entries:
                self._save_to_disk()

            return {"count": len(removed_entries), "removed_entries": removed_entries}

    def clear_all(self) -> int:
        """Remove all cache entries and return count removed (simple entries structure)."""
        with self._lock:
            # Count entries from the entries dict
            entry_count = len(self._metadata.get("entries", {}))

            # Reset to simple structure
            self._metadata = {
                "entries": {},
                "cache_hits": 0,
                "cache_misses": 0,
            }

            self._save_to_disk()
            return entry_count

    def load_metadata(self) -> Dict[str, Any]:
        """Load complete metadata structure."""
        with self._lock:
            return self._metadata.copy()

    def save_metadata(self, metadata: Dict[str, Any]):
        """Save complete metadata structure."""
        with self._lock:
            self._metadata = metadata
            self._save_to_disk()

    # --- Namespace registry helpers ---

    def _load_registry(self) -> Dict[str, Any]:
        """Load the namespace registry from disk."""
        if self._registry_file.exists():
            try:
                with open(self._registry_file, "r") as f:
                    data = json_loads(f.read())
                if isinstance(data, dict) and "namespaces" in data:
                    return data
            except Exception:
                logger.warning("Namespace registry corrupted, starting fresh")
        return {"namespaces": {}}

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """Save the namespace registry to disk atomically."""
        import tempfile
        import shutil

        try:
            if not self._registry_file.parent.exists():
                return
            fd, temp_path = tempfile.mkstemp(
                suffix=".json.tmp",
                dir=self._registry_file.parent,
                prefix="cacheness_ns_",
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(json_dumps(registry, default=str))
                shutil.move(temp_path, self._registry_file)
            except Exception:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.error(f"Failed to save namespace registry: {e}")

    def _ensure_namespace_registry(self) -> None:
        """Ensure the namespace registry exists with a 'default' entry.

        Seeds the 'default' namespace pointing to the existing metadata file
        on first use.  Subsequent calls are no-ops.
        """
        registry = self._load_registry()
        if DEFAULT_NAMESPACE not in registry["namespaces"]:
            now = datetime.now(timezone.utc).isoformat()
            registry["namespaces"][DEFAULT_NAMESPACE] = {
                "namespace_id": DEFAULT_NAMESPACE,
                "display_name": "Default",
                "schema_version": 1,
                "created_at": now,
                "signature": None,
            }
            self._save_registry(registry)
            logger.info("Registered 'default' namespace in JSON registry")

    def _metadata_file_for_namespace(self, namespace_id: str) -> Path:
        """Return the metadata file path for a given namespace.

        The 'default' namespace maps to the root ``metadata_file`` (no
        suffix) for backward compatibility.  Other namespaces get a
        ``{namespace_id}_metadata.json`` file in the same directory.
        """
        root = self._root_metadata_file
        if namespace_id == DEFAULT_NAMESPACE:
            return root
        return root.parent / f"{namespace_id}_metadata.json"

    # --- Schema versioning overrides ---

    def get_schema_version(self, namespace_id: str = DEFAULT_NAMESPACE) -> int:
        """Read schema version from the JSON namespace registry."""
        with self._lock:
            registry = self._load_registry()
            ns = registry["namespaces"].get(namespace_id)
            return ns["schema_version"] if ns else 0

    def set_schema_version(self, namespace_id: str, version: int) -> None:
        """Write schema version to the JSON namespace registry."""
        with self._lock:
            registry = self._load_registry()
            ns = registry["namespaces"].get(namespace_id)
            if ns is not None:
                ns["schema_version"] = version
                self._save_registry(registry)

    def get_migrations(self) -> list:
        """Return JSON-specific schema migrations.

        Currently there are no JSON-specific migrations (the format has
        always been the same), so this returns an empty list.  When the
        JSON schema changes in the future, migrations will be added here.
        """
        return []

    # --- Namespace registry overrides ---

    def create_namespace(
        self,
        namespace_id: str,
        display_name: str = "",
    ) -> NamespaceInfo:
        """Register a new namespace and create its metadata file."""
        validate_namespace_id(namespace_id)

        if namespace_id == DEFAULT_NAMESPACE:
            raise ValueError(
                "The 'default' namespace is pre-registered and cannot be created"
            )

        with self._lock:
            registry = self._load_registry()

            if namespace_id in registry["namespaces"]:
                raise ValueError(f"Namespace {namespace_id!r} already exists")

            now = datetime.now(timezone.utc)

            # Create the per-namespace metadata file with empty structure
            ns_file = self._metadata_file_for_namespace(namespace_id)
            if not ns_file.exists():
                ns_data = {
                    "entries": {},
                    "cache_hits": 0,
                    "cache_misses": 0,
                }
                try:
                    with open(ns_file, "w") as f:
                        f.write(json_dumps(ns_data, default=str))
                except Exception as e:
                    raise OSError(
                        f"Failed to create metadata file for namespace "
                        f"{namespace_id!r}: {e}"
                    ) from e

            # Register in the namespace registry
            registry["namespaces"][namespace_id] = {
                "namespace_id": namespace_id,
                "display_name": display_name,
                "schema_version": 1,
                "created_at": now.isoformat(),
                "signature": None,
            }
            self._save_registry(registry)

            # Run migrations for the new namespace
            self.run_migrations(namespace_id)

            logger.info(f"Created namespace {namespace_id!r} with file {ns_file}")

            return NamespaceInfo(
                namespace_id=namespace_id,
                display_name=display_name,
                schema_version=self.get_schema_version(namespace_id),
                created_at=now,
            )

    def drop_namespace(self, namespace_id: str) -> bool:
        """Remove a namespace, its metadata file, and registry entry."""
        if namespace_id == DEFAULT_NAMESPACE:
            raise ValueError("Cannot drop the 'default' namespace")

        validate_namespace_id(namespace_id)

        with self._lock:
            registry = self._load_registry()

            if namespace_id not in registry["namespaces"]:
                return False

            # Delete the per-namespace metadata file
            ns_file = self._metadata_file_for_namespace(namespace_id)
            if ns_file.exists():
                try:
                    os.remove(ns_file)
                except OSError as e:
                    logger.warning(
                        f"Failed to remove metadata file for namespace "
                        f"{namespace_id!r}: {e}"
                    )

            # Remove from registry
            del registry["namespaces"][namespace_id]
            self._save_registry(registry)

            logger.info(f"Dropped namespace {namespace_id!r}")
            return True

    def list_namespaces(self) -> List[NamespaceInfo]:
        """List all registered namespaces from the JSON registry."""
        with self._lock:
            registry = self._load_registry()
            result = []
            for ns_data in registry["namespaces"].values():
                created_at = ns_data.get("created_at")
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at)
                    except (ValueError, TypeError):
                        created_at = datetime.now(timezone.utc)
                result.append(
                    NamespaceInfo(
                        namespace_id=ns_data["namespace_id"],
                        display_name=ns_data.get("display_name", ""),
                        schema_version=ns_data.get("schema_version", 0),
                        created_at=created_at,
                        signature=ns_data.get("signature"),
                    )
                )
            # Sort by created_at for consistent ordering
            result.sort(key=lambda ns: ns.created_at)
            return result

    def get_namespace(self, namespace_id: str) -> Optional[NamespaceInfo]:
        """Get info for a specific namespace from the JSON registry."""
        with self._lock:
            registry = self._load_registry()
            ns_data = registry["namespaces"].get(namespace_id)
            if ns_data is None:
                return None
            created_at = ns_data.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except (ValueError, TypeError):
                    created_at = datetime.now(timezone.utc)
            return NamespaceInfo(
                namespace_id=ns_data["namespace_id"],
                display_name=ns_data.get("display_name", ""),
                schema_version=ns_data.get("schema_version", 0),
                created_at=created_at,
                signature=ns_data.get("signature"),
            )

    def set_namespace_signature(self, namespace_id: str, signature: str) -> None:
        """Store namespace signature in the JSON registry."""
        with self._lock:
            registry = self._load_registry()
            ns_data = registry["namespaces"].get(namespace_id)
            if ns_data is not None:
                ns_data["signature"] = signature
                self._save_registry(registry)

    def close(self):
        """Close and clean up resources (JSON backend saves any pending changes)."""
        with self._lock:
            # Ensure any pending changes are saved to disk
            self._save_to_disk()
