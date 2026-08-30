"""Lossless local persistence for canonical BlobStore manifest bytes.

This module intentionally has a narrow Phase 2 scope: it supports only the
exact local metadata backend identities that can truthfully preserve a
canonical record. General metadata-backend composition belongs to Phase 4.
"""

from __future__ import annotations

import base64
from typing import Any, Mapping, Optional, Protocol

from cacheness.error_handling import CacheBlobBackendError, CacheError
from cacheness.metadata import InMemoryBackend, JsonBackend, SqliteBackend

try:
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:  # pragma: no cover - SQLite support is optional at runtime.
    SQLAlchemyError = OSError


_RAW_MANIFEST_FIELD = "canonical_manifest_v1"
_SQLITE_MANIFEST_TABLE = "cacheness_manifest_records_v1"
_BACKEND_OPERATION_ERRORS = (
    CacheError,
    OSError,
    TypeError,
    ValueError,
    SQLAlchemyError,
)


class ManifestRepository(Protocol):
    """Persistence contract for exact canonical manifest bytes."""

    def get_raw(self, key: str) -> Optional[bytes]:
        """Return exact manifest bytes, or ``None`` only when the key is absent."""

    def put_raw(
        self,
        key: str,
        record: bytes,
        *,
        entry_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Persist one raw canonical record without decoding it."""

    def remove(self, key: str) -> None:
        """Remove a canonical record by logical key."""

    def list_keys(self) -> list[str]:
        """List logical keys that have canonical records."""


def _backend_failure(
    operation: str, backend: object, exc: BaseException
) -> CacheBlobBackendError:
    """Translate a narrow repository failure without dropping its cause."""
    return CacheBlobBackendError(
        f"Canonical manifest repository {operation} failed",
        context={
            "operation": operation,
            "backend": type(backend).__name__,
        },
    )


class _MetadataManifestRepository:
    """Store exact bytes in a reversible JSON-safe metadata projection."""

    def __init__(self, backend: InMemoryBackend | JsonBackend):
        self.backend = backend

    def get_raw(self, key: str) -> Optional[bytes]:
        """Load one reversible byte projection without interpreting the manifest."""
        try:
            entry = self.backend.get_entry(key)
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("get_raw", self.backend, exc) from exc
        if entry is None:
            return None

        metadata = entry.get("metadata") if isinstance(entry, Mapping) else None
        if not isinstance(metadata, Mapping) or _RAW_MANIFEST_FIELD not in metadata:
            return None
        encoded = metadata[_RAW_MANIFEST_FIELD]
        if not isinstance(encoded, str):
            exc = ValueError("Canonical manifest record is not a base64 string")
            raise _backend_failure("get_raw", self.backend, exc) from exc
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise _backend_failure("get_raw", self.backend, exc) from exc

    def put_raw(
        self,
        key: str,
        record: bytes,
        *,
        entry_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Write one exact byte sequence through existing durable backend storage."""
        if not isinstance(record, bytes):
            raise TypeError("Canonical manifest records must be bytes")
        projection = dict(entry_data or {})
        source_metadata = projection.get("metadata", {})
        metadata = dict(source_metadata) if isinstance(source_metadata, Mapping) else {}
        metadata[_RAW_MANIFEST_FIELD] = base64.b64encode(record).decode("ascii")
        projection["cache_key"] = key
        projection["metadata"] = metadata
        try:
            self.backend.put_entry(key, projection)
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("put_raw", self.backend, exc) from exc

    def remove(self, key: str) -> None:
        """Remove the record and its compatibility metadata projection together."""
        try:
            self.backend.remove_entry(key)
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("remove", self.backend, exc) from exc

    def list_keys(self) -> list[str]:
        """Return only keys that carry a raw canonical record."""
        try:
            entries = self.backend.list_entries()
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("list_keys", self.backend, exc) from exc
        return [
            entry["cache_key"]
            for entry in entries
            if isinstance(entry, Mapping)
            and isinstance(entry.get("cache_key"), str)
            and isinstance(entry.get("metadata"), Mapping)
            and _RAW_MANIFEST_FIELD in entry["metadata"]
        ]


class InMemoryManifestRepository(_MetadataManifestRepository):
    """Exact raw records backed by one process-local metadata identity."""

    def __init__(self, backend: InMemoryBackend):
        super().__init__(backend)


class JsonManifestRepository(_MetadataManifestRepository):
    """Exact raw records persisted by JsonBackend's durable document protocol."""

    def __init__(self, backend: JsonBackend):
        super().__init__(backend)


class SqliteManifestRepository:
    """Store canonical bytes in a dedicated SQLite BLOB table.

    The table is deliberately separate from ``cache_entries`` so canonical
    fields never pass through its fixed-column metadata projection.
    """

    def __init__(self, backend: SqliteBackend):
        self.backend = backend
        if backend._legacy_layout is not None:
            raise CacheBlobBackendError(
                "Canonical manifest storage requires a current SQLite backend",
                context={"backend": type(backend).__name__},
            )
        self._create_table()

    def _create_table(self) -> None:
        """Create the isolated raw-byte table without touching legacy rows."""
        try:
            with self.backend._lock, self.backend.engine.begin() as connection:
                connection.exec_driver_sql(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_SQLITE_MANIFEST_TABLE} (
                        logical_key TEXT PRIMARY KEY NOT NULL,
                        canonical_bytes BLOB NOT NULL
                    )
                    """
                )
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("initialize", self.backend, exc) from exc

    def get_raw(self, key: str) -> Optional[bytes]:
        """Load one BLOB without decoding or authenticating it."""
        try:
            with self.backend._lock, self.backend.engine.begin() as connection:
                row = connection.exec_driver_sql(
                    f"SELECT canonical_bytes FROM {_SQLITE_MANIFEST_TABLE} "
                    "WHERE logical_key = ?",
                    (key,),
                ).first()
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("get_raw", self.backend, exc) from exc
        if row is None:
            return None
        return bytes(row[0])

    def put_raw(
        self,
        key: str,
        record: bytes,
        *,
        entry_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Upsert exact bytes and preserve existing BlobStore metadata when supplied."""
        if not isinstance(record, bytes):
            raise TypeError("Canonical manifest records must be bytes")
        try:
            if entry_data is not None:
                self.backend.put_entry(key, dict(entry_data))
            with self.backend._lock, self.backend.engine.begin() as connection:
                connection.exec_driver_sql(
                    f"""
                    INSERT INTO {_SQLITE_MANIFEST_TABLE}
                    (logical_key, canonical_bytes)
                    VALUES (?, ?)
                    ON CONFLICT(logical_key) DO UPDATE SET
                        canonical_bytes = excluded.canonical_bytes
                    """,
                    (key, record),
                )
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("put_raw", self.backend, exc) from exc

    def remove(self, key: str) -> None:
        """Remove the raw manifest and compatible BlobStore metadata if present."""
        try:
            with self.backend._lock, self.backend.engine.begin() as connection:
                connection.exec_driver_sql(
                    f"DELETE FROM {_SQLITE_MANIFEST_TABLE} WHERE logical_key = ?",
                    (key,),
                )
            self.backend.remove_entry(key)
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("remove", self.backend, exc) from exc

    def list_keys(self) -> list[str]:
        """List exact logical keys persisted in the dedicated table."""
        try:
            with self.backend._lock, self.backend.engine.begin() as connection:
                rows = connection.exec_driver_sql(
                    f"SELECT logical_key FROM {_SQLITE_MANIFEST_TABLE} "
                    "ORDER BY logical_key"
                ).all()
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("list_keys", self.backend, exc) from exc
        return [row[0] for row in rows]


def create_manifest_repository(backend: object) -> ManifestRepository:
    """Select the exact local adapter that can preserve canonical record bytes."""
    if type(backend) is InMemoryBackend:
        return InMemoryManifestRepository(backend)
    if type(backend) is JsonBackend:
        return JsonManifestRepository(backend)
    if type(backend) is SqliteBackend:
        return SqliteManifestRepository(backend)
    raise CacheBlobBackendError(
        "Canonical manifest storage supports only exact local backend identities",
        context={
            "operation": "create_manifest_repository",
            "backend": type(backend).__name__,
            "supported_backends": ["InMemoryBackend", "JsonBackend", "SqliteBackend"],
        },
    )


class MetadataManifestRepository:
    """Compatibility facade for callers of the original Phase 2 repository seam."""

    def __init__(self, backend: object):
        self._repository = create_manifest_repository(backend)

    def get_raw(self, key: str) -> Optional[bytes]:
        return self._repository.get_raw(key)

    def put_raw(
        self,
        key: str,
        record: bytes,
        *,
        entry_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._repository.put_raw(key, record, entry_data=entry_data)

    def remove(self, key: str) -> None:
        self._repository.remove(key)

    def list_keys(self) -> list[str]:
        return self._repository.list_keys()


__all__ = [
    "ManifestRepository",
    "InMemoryManifestRepository",
    "JsonManifestRepository",
    "SqliteManifestRepository",
    "MetadataManifestRepository",
    "create_manifest_repository",
]
