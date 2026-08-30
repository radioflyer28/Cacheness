"""Lossless local persistence for canonical BlobStore manifest bytes.

This module intentionally has a narrow Phase 2 scope: it supports only the
exact local metadata backend identities that can truthfully preserve a
canonical record. General metadata-backend composition belongs to Phase 4.
"""

from __future__ import annotations

import base64
import hashlib
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Protocol

from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobMigrationRequiredError,
    CacheError,
)
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


@dataclass(frozen=True)
class ManifestExpectation:
    """An authenticated opaque record expectation for conditional publication."""

    generation: str | None
    record_digest: str | None

    @classmethod
    def absent(cls) -> "ManifestExpectation":
        """Build the expectation used to create a previously absent key."""
        return cls(generation=None, record_digest=None)

    @classmethod
    def from_authenticated_record(
        cls, generation: str, raw_record: bytes
    ) -> "ManifestExpectation":
        """Bind a committed generation to the exact bytes observed by the engine."""
        if not isinstance(generation, str) or not generation:
            raise ValueError("Manifest expectation generation must be non-empty")
        if not isinstance(raw_record, bytes) or not raw_record:
            raise ValueError("Manifest expectation record must be non-empty bytes")
        return cls(generation, hashlib.sha256(raw_record).hexdigest())

    def matches(self, raw_record: bytes | None) -> bool:
        """Compare only opaque canonical bytes; repository code never authenticates."""
        if self.record_digest is None:
            return raw_record is None
        return raw_record is not None and hashlib.sha256(raw_record).hexdigest() == self.record_digest


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

    def publish_if_expected(
        self,
        key: str,
        expected: ManifestExpectation,
        record: bytes,
        *,
        entry_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Atomically publish only when one exact authenticated record remains."""

    def remove(self, key: str) -> None:
        """Remove a canonical record by logical key."""

    def list_keys(self) -> list[str]:
        """List logical keys that have canonical records."""

    def list_backend_entries(self) -> list[dict[str, Any]]:
        """Return compatibility projections with typed backend translation."""


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
            raise CacheBlobMigrationRequiredError(
                "Compatibility metadata exists without canonical manifest bytes",
                context={"key": key, "backend": type(self.backend).__name__},
            )
        encoded = metadata[_RAW_MANIFEST_FIELD]
        if not isinstance(encoded, str):
            exc = ValueError("Canonical manifest record is not a base64 string")
            raise _backend_failure("get_raw", self.backend, exc) from exc
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise _backend_failure("get_raw", self.backend, exc) from exc

    @staticmethod
    def _raw_from_entry(entry: object) -> bytes | None:
        """Decode one stored projection while preserving malformed-record failure."""
        if entry is None:
            return None
        metadata = entry.get("metadata") if isinstance(entry, Mapping) else None
        if not isinstance(metadata, Mapping) or _RAW_MANIFEST_FIELD not in metadata:
            raise CacheBlobMigrationRequiredError(
                "Compatibility metadata exists without canonical manifest bytes"
            )
        encoded = metadata[_RAW_MANIFEST_FIELD]
        if not isinstance(encoded, str):
            raise ValueError("Canonical manifest record is not a base64 string")
        return base64.b64decode(encoded, validate=True)

    @staticmethod
    def _projection(key: str, record: bytes, entry_data: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        """Build the reversible compatibility representation used by local backends."""
        projection = dict(entry_data or {})
        source_metadata = projection.get("metadata", {})
        metadata = dict(source_metadata) if isinstance(source_metadata, Mapping) else {}
        metadata[_RAW_MANIFEST_FIELD] = base64.b64encode(record).decode("ascii")
        projection["cache_key"] = key
        projection["metadata"] = metadata
        return projection

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
        projection = self._projection(key, record, entry_data)
        try:
            self.backend.put_entry(key, projection)
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("put_raw", self.backend, exc) from exc

    def publish_if_expected(
        self,
        key: str,
        expected: ManifestExpectation,
        record: bytes,
        *,
        entry_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Compare opaque bytes and update one local metadata record atomically."""
        if not isinstance(record, bytes):
            raise TypeError("Canonical manifest records must be bytes")
        try:
            with self.backend._lock:
                if type(self.backend) is JsonBackend:
                    current_entry = self.backend._metadata.get("entries", {}).get(key)
                else:
                    current_entry = self.backend._entries.get(key)
                current_raw = self._raw_from_entry(current_entry)
                if not expected.matches(current_raw):
                    raise CacheBlobLifecycleConflictError(
                        "Canonical manifest expectation no longer matches",
                        context={"key": key, "operation": "publish_if_expected"},
                    )
                projection = self._projection(key, record, entry_data)
                if type(self.backend) is JsonBackend:
                    self.backend._ensure_writable()
                    candidate = deepcopy(self.backend._metadata)
                    now = datetime.now(timezone.utc).isoformat()
                    candidate.setdefault("entries", {})[key] = {
                        "description": projection.get("description", ""),
                        "data_type": projection.get("data_type", "unknown"),
                        "prefix": projection.get("prefix", ""),
                        "created_at": projection.get("created_at", now),
                        "accessed_at": projection.get("accessed_at", now),
                        "file_size": projection.get("file_size", 0),
                        "metadata": projection["metadata"].copy(),
                    }
                    self.backend._save_to_disk(candidate)
                    self.backend._metadata = candidate
                else:
                    self.backend.put_entry(key, projection)
        except CacheBlobLifecycleConflictError:
            raise
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("publish_if_expected", self.backend, exc) from exc

    def remove(self, key: str) -> None:
        """Remove the record and its compatibility metadata projection together."""
        try:
            self.backend.remove_entry(key)
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("remove", self.backend, exc) from exc

    def list_keys(self) -> list[str]:
        """Return canonical keys and reject a partial compatibility projection."""
        entries = self.list_backend_entries()
        keys = []
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(
                entry.get("cache_key"), str
            ):
                raise CacheBlobMigrationRequiredError(
                    "Compatibility metadata entry has no canonical key",
                    context={"backend": type(self.backend).__name__},
                )
            metadata = entry.get("metadata")
            if not isinstance(metadata, Mapping) or _RAW_MANIFEST_FIELD not in metadata:
                raise CacheBlobMigrationRequiredError(
                    "Compatibility metadata exists without canonical manifest bytes",
                    context={
                        "key": entry["cache_key"],
                        "backend": type(self.backend).__name__,
                    },
                )
            keys.append(entry["cache_key"])
        return keys

    def list_backend_entries(self) -> list[dict[str, Any]]:
        """Read backend projections without allowing operational failures to leak."""
        try:
            return self.backend.list_entries()
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("list_entries", self.backend, exc) from exc


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
                metadata_row = None
                if row is None:
                    metadata_row = connection.exec_driver_sql(
                        "SELECT 1 FROM cache_entries WHERE cache_key = ?", (key,)
                    ).first()
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("get_raw", self.backend, exc) from exc
        if row is None:
            if metadata_row is not None:
                raise CacheBlobMigrationRequiredError(
                    "SQLite compatibility metadata exists without canonical manifest bytes",
                    context={"key": key, "backend": type(self.backend).__name__},
                )
            return None
        return bytes(row[0])

    def put_raw(
        self,
        key: str,
        record: bytes,
        *,
        entry_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Atomically publish the compatibility projection and canonical bytes."""
        if not isinstance(record, bytes):
            raise TypeError("Canonical manifest records must be bytes")
        try:
            with self.backend._lock, self.backend.engine.begin() as connection:
                if entry_data is not None:
                    self._write_compatibility_projection(connection, key, entry_data)
                self._write_raw_row(connection, key, record)
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("put_raw", self.backend, exc) from exc

    def publish_if_expected(
        self,
        key: str,
        expected: ManifestExpectation,
        record: bytes,
        *,
        entry_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Conditionally update local SQLite authority in one transaction."""
        if not isinstance(record, bytes):
            raise TypeError("Canonical manifest records must be bytes")
        try:
            with self.backend._lock, self.backend.engine.begin() as connection:
                row = connection.exec_driver_sql(
                    f"SELECT canonical_bytes FROM {_SQLITE_MANIFEST_TABLE} "
                    "WHERE logical_key = ?",
                    (key,),
                ).first()
                current_raw = None if row is None else bytes(row[0])
                if not expected.matches(current_raw):
                    raise CacheBlobLifecycleConflictError(
                        "Canonical manifest expectation no longer matches",
                        context={"key": key, "operation": "publish_if_expected"},
                    )
                if entry_data is not None:
                    self._write_compatibility_projection(connection, key, entry_data)
                self._write_raw_row(connection, key, record)
        except CacheBlobLifecycleConflictError:
            raise
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("publish_if_expected", self.backend, exc) from exc

    @staticmethod
    def _write_compatibility_projection(
        connection: Any, key: str, entry_data: Mapping[str, Any]
    ) -> None:
        """Write the established SQLite projection in the caller transaction."""
        metadata_value = entry_data.get("metadata", {})
        metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
        created_at = entry_data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        connection.exec_driver_sql(
            """
            INSERT OR REPLACE INTO cache_entries
            (cache_key, description, data_type, prefix, file_size,
             file_hash, entry_signature, cache_key_params,
             object_type, storage_format, serializer, compression_codec, actual_path,
             created_at, accessed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                entry_data.get("description", ""),
                entry_data.get("data_type", "unknown"),
                entry_data.get("prefix", ""),
                entry_data.get("file_size", 0),
                metadata.get("file_hash"),
                metadata.get("entry_signature"),
                None,
                metadata.get("object_type"),
                metadata.get("storage_format"),
                metadata.get("serializer"),
                metadata.get("compression_codec"),
                metadata.get("actual_path"),
                created_at,
                datetime.now(timezone.utc),
            ),
        )

    @staticmethod
    def _write_raw_row(connection: Any, key: str, record: bytes) -> None:
        """Upsert canonical bytes inside the surrounding SQLite transaction."""
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
        """List keys while treating compatibility-only rows as non-absence."""
        try:
            with self.backend._lock, self.backend.engine.begin() as connection:
                rows = connection.exec_driver_sql(
                    f"SELECT logical_key FROM {_SQLITE_MANIFEST_TABLE} "
                    "ORDER BY logical_key"
                ).all()
                metadata_rows = connection.exec_driver_sql(
                    "SELECT cache_key FROM cache_entries"
                ).all()
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("list_keys", self.backend, exc) from exc
        keys = [row[0] for row in rows]
        compatibility_only = {row[0] for row in metadata_rows}.difference(keys)
        if compatibility_only:
            raise CacheBlobMigrationRequiredError(
                "SQLite compatibility metadata exists without canonical manifest bytes",
                context={"keys": sorted(compatibility_only)},
            )
        return keys

    def list_backend_entries(self) -> list[dict[str, Any]]:
        """Read the SQLite compatibility projection with typed translation."""
        try:
            return self.backend.list_entries()
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("list_entries", self.backend, exc) from exc


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

    def publish_if_expected(
        self,
        key: str,
        expected: ManifestExpectation,
        record: bytes,
        *,
        entry_data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._repository.publish_if_expected(
            key, expected, record, entry_data=entry_data
        )

    def remove(self, key: str) -> None:
        self._repository.remove(key)

    def list_keys(self) -> list[str]:
        return self._repository.list_keys()

    def list_backend_entries(self) -> list[dict[str, Any]]:
        return self._repository.list_backend_entries()


__all__ = [
    "ManifestRepository",
    "ManifestExpectation",
    "InMemoryManifestRepository",
    "JsonManifestRepository",
    "SqliteManifestRepository",
    "MetadataManifestRepository",
    "create_manifest_repository",
]
