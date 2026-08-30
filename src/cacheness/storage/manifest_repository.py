"""Raw canonical manifest persistence behind the existing metadata backends."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol


_RAW_MANIFEST_FIELD = "canonical_manifest_v1"


class ManifestRepository(Protocol):
    """Backend-neutral persistence contract for exact canonical record bytes."""

    def get_raw(self, key: str) -> Optional[bytes]:
        """Return exact manifest bytes, or ``None`` only when the key is absent."""

    def put_raw(
        self,
        key: str,
        record: bytes,
        *,
        entry_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Persist one canonical record without reconstructing its bytes."""

    def remove(self, key: str) -> None:
        """Remove a canonical record by logical key."""

    def list_keys(self) -> list[str]:
        """List logical keys currently known by the underlying metadata backend."""


class MetadataManifestRepository:
    """Store exact UTF-8 canonical bytes inside a compatible metadata entry."""

    def __init__(self, backend: Any):
        self.backend = backend

    def get_raw(self, key: str) -> Optional[bytes]:
        entry = self.backend.get_entry(key)
        if entry is None:
            return None
        metadata = entry.get("metadata") if isinstance(entry, Mapping) else None
        record = metadata.get(_RAW_MANIFEST_FIELD) if isinstance(metadata, Mapping) else None
        if isinstance(record, str):
            return record.encode("utf-8")
        return b""

    def put_raw(
        self,
        key: str,
        record: bytes,
        *,
        entry_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        try:
            encoded = record.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Canonical manifest record must be UTF-8") from exc
        projection = dict(entry_data or {})
        metadata = dict(projection.pop("metadata", {}) or {})
        metadata[_RAW_MANIFEST_FIELD] = encoded
        projection["cache_key"] = key
        projection["metadata"] = metadata
        self.backend.put_entry(key, projection)

    def remove(self, key: str) -> None:
        self.backend.remove_entry(key)

    def list_keys(self) -> list[str]:
        return [
            entry.get("cache_key", "")
            for entry in self.backend.list_entries()
            if isinstance(entry, Mapping) and isinstance(entry.get("cache_key"), str)
        ]


__all__ = ["ManifestRepository", "MetadataManifestRepository"]
