"""Lossless local persistence for canonical BlobStore manifest bytes.

This module intentionally has a narrow Phase 2 scope: it supports only the
exact local metadata backend identities that can truthfully preserve a
canonical record. General metadata-backend composition belongs to Phase 4.
"""

from __future__ import annotations

import base64
import hashlib
import os
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, BinaryIO, Callable, Mapping, Optional, Protocol

from cacheness.config import LifecycleLimits
from cacheness.error_handling import (
    CacheBlobBackendError,
    CacheBlobLifecycleConflictError,
    CacheBlobMigrationRequiredError,
    CacheError,
)
from cacheness.metadata import InMemoryBackend, JsonBackend, SqliteBackend
from cacheness.json_utils import dumps as json_dumps
from cacheness.json_utils import loads as json_loads

from .coordination import interprocess_open_file_lock
from .path_security import ManagedFileOps, resolve_managed_locator

try:
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:  # pragma: no cover - SQLite support is optional at runtime.
    SQLAlchemyError = OSError


_RAW_MANIFEST_FIELD = "canonical_manifest_v1"
_MANIFEST_INVENTORY_FIELD = "_cacheness_manifest_inventory_v1"
_SQLITE_MANIFEST_TABLE = "cacheness_manifest_records_v1"
_SQLITE_MANIFEST_INVENTORY_TABLE = "cacheness_manifest_inventory_v1"
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

    def __post_init__(self) -> None:
        """Reject ambiguous absence and partial exact-record expectations."""
        if (self.generation is None) != (self.record_digest is None):
            raise ValueError(
                "Manifest expectations require both generation and record digest"
            )
        if self.generation is not None and (
            not isinstance(self.generation, str) or not self.generation
        ):
            raise ValueError("Manifest expectation generation must be non-empty")
        if self.record_digest is not None and (
            not isinstance(self.record_digest, str)
            or len(self.record_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.record_digest)
        ):
            raise ValueError("Manifest expectation digest must be a SHA-256 hex value")

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


@dataclass(frozen=True)
class ManifestCursor:
    """Opaque generation-bound position after a bounded manifest page.

    ``key`` remains the v1 compatibility projection.  New cursors additionally
    bind a monotonic inventory high-water mark and the next sequence position,
    so a writer that inserts or republishes a key after page one cannot be
    pulled backwards through a lexical cursor.
    """

    key: str
    snapshot_high_water: int | None = None
    next_sequence: int | None = None

    def __post_init__(self) -> None:
        """Keep cursors bounded without imposing a path grammar on logical keys."""
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("Manifest cursor key must be a non-empty string")
        if len(self.key.encode("utf-8")) > 8_192:
            raise ValueError("Manifest cursor key exceeds the byte limit")
        if (self.snapshot_high_water is None) != (self.next_sequence is None):
            raise ValueError("Manifest cursor snapshot fields must be paired")
        for field_name in ("snapshot_high_water", "next_sequence"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"Manifest cursor {field_name} is invalid")


@dataclass(frozen=True)
class ManifestPage:
    """One bounded page of opaque exact manifest records."""

    entries: tuple[tuple[str, bytes], ...]
    next_cursor: ManifestCursor | None


class ManifestRepository(Protocol):
    """Persistence contract for exact canonical manifest bytes."""

    def refresh_authoritative_view(self) -> None:
        """Synchronize a repository view through its canonical authority path."""

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

    def remove_if_expected(self, key: str, expected: ManifestExpectation) -> None:
        """Atomically remove only the exact authenticated record observed."""

    def list_keys(self) -> list[str]:
        """List logical keys that have canonical records."""

    def list_page(
        self,
        cursor: ManifestCursor | None = None,
        *,
        page_size: int | None = None,
    ) -> ManifestPage:
        """Return one stable bounded page without authenticating its records."""

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

    def __init__(
        self,
        backend: InMemoryBackend | JsonBackend,
        *,
        lifecycle_limits: LifecycleLimits | None = None,
        file_ops: ManagedFileOps | None = None,
    ):
        self.backend = backend
        self.lifecycle_limits = (
            LifecycleLimits() if lifecycle_limits is None else lifecycle_limits
        )
        self._json_lock_file_ops: ManagedFileOps | None = None
        self._owns_json_lock_file_ops = False
        self._json_lock_locator: Path | None = None
        self._json_lock_handle: BinaryIO | None = None
        self._json_lock_identity: tuple[int, int] | None = None
        self._json_metadata_locator: Path | None = None
        self._json_lock_guard = RLock()
        # Deterministic race seams used only by containment regressions.  They
        # deliberately run while the retained descriptor remains authoritative.
        self.after_json_lock_validation: Callable[[], None] | None = None
        self.after_json_lock_acquisition: Callable[[], None] | None = None
        if type(backend) is JsonBackend:
            try:
                lock_root = Path(backend.metadata_file).parent
                if file_ops is None:
                    self._json_lock_file_ops = ManagedFileOps(lock_root)
                    self._owns_json_lock_file_ops = True
                else:
                    if file_ops.root != lock_root.resolve():
                        raise ValueError(
                            "JSON manifest authority lock must share the metadata root"
                        )
                    self._json_lock_file_ops = file_ops
                self._json_metadata_locator = resolve_managed_locator(
                    self._json_lock_file_ops.root,
                    Path(backend.metadata_file).name,
                    operation="manifest_repository_metadata",
                    allow_missing_leaf=True,
                )
                self._json_lock_locator = resolve_managed_locator(
                    self._json_lock_file_ops.root,
                    f".{Path(backend.metadata_file).name}.manifest-cas.lock",
                    operation="manifest_repository_authority_lock",
                    allow_missing_leaf=True,
                )
                expected_lock_identity = self._json_lock_file_ops.ensure_lifecycle_lock(
                    self._json_lock_locator
                )
                self._json_lock_handle = self._json_lock_file_ops.open_verified_regular_file(
                    self._json_lock_locator
                )
                lock_stat = os.fstat(self._json_lock_handle.fileno())
                self._json_lock_identity = (lock_stat.st_dev, lock_stat.st_ino)
                if self._json_lock_identity != expected_lock_identity:
                    raise CacheBlobBackendError(
                        "JSON canonical manifest authority lock changed during construction",
                        context={"backend": type(self.backend).__name__},
                    )
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        """Release only a lock root that this direct repository constructed."""
        if self._json_lock_handle is not None:
            self._json_lock_handle.close()
            self._json_lock_handle = None
            self._json_lock_identity = None
        if self._owns_json_lock_file_ops and self._json_lock_file_ops is not None:
            self._json_lock_file_ops.close()
            self._json_lock_file_ops = None
        self._json_metadata_locator = None

    def _page_size(self, page_size: int | None) -> int:
        """Resolve one caller page without bypassing the configured bound."""
        resolved = (
            self.lifecycle_limits.manifest_page_size
            if page_size is None
            else page_size
        )
        if type(resolved) is not int or resolved <= 0:
            raise ValueError("manifest page size must be a positive integer")
        if resolved > self.lifecycle_limits.manifest_page_size:
            raise ValueError("manifest page size exceeds configured lifecycle limit")
        return resolved

    def get_raw(self, key: str) -> Optional[bytes]:
        """Load one reversible byte projection without interpreting the manifest."""
        try:
            if type(self.backend) is JsonBackend:
                # Separate JSON backend instances cache their document. Refresh
                # under the same short OS-backed lock as conditional mutation so
                # direct repository reads cannot report a superseded authority.
                with self.backend._lock, self._json_compare_publish_lock():
                    self._refresh_json_for_conditional_operation()
                    entry = self.backend.get_entry(key)
            else:
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

    def refresh_authoritative_view(self) -> None:
        """Refresh JSON through the retained descriptor authority boundary.

        Ordinary BlobStore admission needs a current JSON projection before it
        builds lifecycle evidence.  Delegating that refresh here prevents a
        path-based ``JsonBackend`` read from switching roots between admission
        and the following canonical compare/publish transition.
        """
        if type(self.backend) is not JsonBackend:
            return
        try:
            with self.backend._lock, self._json_compare_publish_lock():
                self._refresh_json_for_conditional_operation()
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("refresh_authority", self.backend, exc) from exc

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
        try:
            with self.backend._lock, self._json_compare_publish_lock():
                if type(self.backend) is JsonBackend:
                    self._refresh_json_for_conditional_operation()
                self._publish_projection(key, record, entry_data)
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("put_raw", self.backend, exc) from exc

    @contextmanager
    def _json_compare_publish_lock(self):
        """Serialize one JSON refresh/compare/publish sequence across processes."""
        if type(self.backend) is not JsonBackend:
            yield
            return
        if (
            self._json_lock_file_ops is None
            or self._json_lock_locator is None
            or self._json_lock_handle is None
            or self._json_lock_identity is None
        ):
            raise CacheBlobBackendError(
                "JSON canonical manifest authority lock is unavailable",
                context={"backend": type(self.backend).__name__},
            )
        # Do not translate exceptions around ``yield`` here: a backend failure
        # raised by the caller's compare/publish body must retain its original
        # operation and cause rather than being mislabeled as lock acquisition.
        with self._json_lock_guard:
            self._json_lock_file_ops.assert_retained_lock_identity(
                self._json_lock_locator, self._json_lock_identity
            )
            if self.after_json_lock_validation is not None:
                self.after_json_lock_validation()
            with interprocess_open_file_lock(
                self._json_lock_handle,
                exclusive=True,
                operation="json_manifest_authority",
            ):
                # The lock descriptor and the managed root are authority data.
                # Checking again after OS acquisition prevents a contender from
                # proceeding under an inode that a later repository no longer
                # uses.
                self._json_lock_file_ops.assert_retained_lock_identity(
                    self._json_lock_locator, self._json_lock_identity
                )
                if self.after_json_lock_acquisition is not None:
                    self.after_json_lock_acquisition()
                self._json_lock_file_ops.assert_root_identity()
                yield

    def _refresh_json_for_conditional_operation(self) -> None:
        """Refresh JSON through the retained root descriptor while CAS is held."""
        if type(self.backend) is not JsonBackend:
            return
        if self._json_lock_file_ops is None or self._json_metadata_locator is None:
            raise CacheBlobBackendError(
                "JSON canonical manifest root descriptor is unavailable",
                context={"backend": type(self.backend).__name__},
            )
        self.backend._ensure_writable()
        self._json_lock_file_ops.assert_root_identity()
        try:
            raw_document = self._json_lock_file_ops.read_bytes(self._json_metadata_locator)
        except FileNotFoundError:
            document: object = {
                "entries": {},
                "cache_hits": 0,
                "cache_misses": 0,
            }
        else:
            document = json_loads(raw_document)
        if not isinstance(document, dict):
            raise ValueError("JSON metadata document must be an object")
        # Legacy normalization is pure document validation; no metadata path is
        # consulted after the retained root was verified.
        self.backend._metadata = self.backend._normalize_legacy_split_map(document)

    def _publish_json_document(self, candidate: dict[str, Any]) -> None:
        """Durably publish JSON authority through the retained root descriptor."""
        if self._json_lock_file_ops is None or self._json_metadata_locator is None:
            raise CacheBlobBackendError(
                "JSON canonical manifest root descriptor is unavailable",
                context={"backend": type(self.backend).__name__},
            )
        self._json_lock_file_ops.assert_root_identity()
        encoded = json_dumps(candidate, default=str).encode("utf-8")
        self._json_lock_file_ops.write_bytes_durable(self._json_metadata_locator, encoded)
        self._json_lock_file_ops.assert_root_identity()
        self.backend._metadata = candidate

    def _current_entry(self, key: str) -> object:
        """Return the current projection while the caller holds the backend lock."""
        if type(self.backend) is JsonBackend:
            return self.backend._metadata.get("entries", {}).get(key)
        return self.backend._entries.get(key)

    def _inventory_state(self, *, candidate: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return the versioned append-only manifest inventory state.

        The index is deliberately separate from canonical manifest authority.
        It records only sequence, logical key, and a digest of bytes already
        being published through the authority path.  A later page therefore
        proves that an event was part of its high-water membership before it
        reads current bytes; changed/deleted events are consumed as stale
        positions rather than accidentally becoming new snapshot members.
        """
        if type(self.backend) is JsonBackend:
            container = self.backend._metadata if candidate is None else candidate
            state = container.get(_MANIFEST_INVENTORY_FIELD)
            if state is None:
                state = {"version": 1, "next_sequence": 1, "events": []}
                container[_MANIFEST_INVENTORY_FIELD] = state
            return self._validate_inventory_state(state)
        state = getattr(self.backend, _MANIFEST_INVENTORY_FIELD, None)
        if state is None:
            state = {"version": 1, "next_sequence": 1, "events": []}
            setattr(self.backend, _MANIFEST_INVENTORY_FIELD, state)
        return self._validate_inventory_state(state)

    @staticmethod
    def _validate_inventory_state(state: object) -> dict[str, Any]:
        """Reject malformed local scheduling state before it drives paging."""
        if not isinstance(state, dict) or set(state) != {"version", "next_sequence", "events"}:
            raise CacheBlobBackendError(
                "Canonical manifest inventory is invalid",
                context={"operation": "manifest_inventory"},
            )
        if state["version"] != 1 or type(state["next_sequence"]) is not int:
            raise CacheBlobBackendError(
                "Canonical manifest inventory version is invalid",
                context={"operation": "manifest_inventory"},
            )
        events = state["events"]
        if not isinstance(events, list) or state["next_sequence"] != len(events) + 1:
            raise CacheBlobBackendError(
                "Canonical manifest inventory is not contiguous",
                context={"operation": "manifest_inventory"},
            )
        for sequence, event in enumerate(events, start=1):
            if (
                not isinstance(event, list)
                or len(event) != 2
                or not isinstance(event[0], str)
                or not event[0]
                or not isinstance(event[1], str)
                or len(event[1]) != 64
                or any(character not in "0123456789abcdef" for character in event[1])
            ):
                raise CacheBlobBackendError(
                    "Canonical manifest inventory event is invalid",
                    context={"operation": "manifest_inventory", "sequence": sequence},
                )
        return state

    @staticmethod
    def _append_inventory_event(state: dict[str, Any], key: str, record: bytes) -> None:
        """Append the exact publication event in the same local transaction."""
        state["events"].append([key, hashlib.sha256(record).hexdigest()])
        state["next_sequence"] += 1

    def _publish_projection(
        self, key: str, record: bytes, entry_data: Optional[Mapping[str, Any]]
    ) -> None:
        """Publish one reversible projection inside the established lock boundary."""
        projection = self._projection(key, record, entry_data)
        if type(self.backend) is JsonBackend:
            candidate = deepcopy(self.backend._metadata)
            inventory = self._inventory_state(candidate=candidate)
            self._append_inventory_event(inventory, key, record)
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
            self._publish_json_document(candidate)
            return
        self.backend.put_entry(key, projection)
        # The in-memory backend shares this repository's re-entrant local
        # boundary.  Its sequence index is process-local by definition, but
        # follows the exact same event contract as durable JSON.
        self._append_inventory_event(self._inventory_state(), key, record)

    def _remove_projection(self, key: str) -> None:
        """Retire the raw and compatibility projections in one local boundary."""
        if type(self.backend) is JsonBackend:
            candidate = deepcopy(self.backend._metadata)
            candidate.get("entries", {}).pop(key, None)
            self._publish_json_document(candidate)
            return
        self.backend._entries.pop(key, None)

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
            with self.backend._lock, self._json_compare_publish_lock():
                self._refresh_json_for_conditional_operation()
                current_raw = self._raw_from_entry(self._current_entry(key))
                if not expected.matches(current_raw):
                    raise CacheBlobLifecycleConflictError(
                        "Canonical manifest expectation no longer matches",
                        context={"key": key, "operation": "publish_if_expected"},
                    )
                self._publish_projection(key, record, entry_data)
        except CacheBlobLifecycleConflictError:
            raise
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("publish_if_expected", self.backend, exc) from exc

    def remove(self, key: str) -> None:
        """Remove the record and its compatibility metadata projection together."""
        try:
            if type(self.backend) is JsonBackend:
                with self.backend._lock, self._json_compare_publish_lock():
                    self._refresh_json_for_conditional_operation()
                    self._remove_projection(key)
                return
            self.backend.remove_entry(key)
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("remove", self.backend, exc) from exc

    def remove_if_expected(self, key: str, expected: ManifestExpectation) -> None:
        """Conditionally retire a JSON or memory projection without stale deletion."""
        try:
            with self.backend._lock, self._json_compare_publish_lock():
                self._refresh_json_for_conditional_operation()
                current_raw = self._raw_from_entry(self._current_entry(key))
                if not expected.matches(current_raw):
                    raise CacheBlobLifecycleConflictError(
                        "Canonical manifest expectation no longer matches",
                        context={"key": key, "operation": "remove_if_expected"},
                    )
                self._remove_projection(key)
        except CacheBlobLifecycleConflictError:
            raise
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("remove_if_expected", self.backend, exc) from exc

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

    def list_page(
        self,
        cursor: ManifestCursor | None = None,
        *,
        page_size: int | None = None,
    ) -> ManifestPage:
        """Read one generation-bound page without a namespace-size ceiling."""
        if cursor is not None and not isinstance(cursor, ManifestCursor):
            raise TypeError("manifest cursor must be a ManifestCursor or None")
        limit = self._page_size(page_size)
        try:
            with self.backend._lock, self._json_compare_publish_lock():
                self._refresh_json_for_conditional_operation()
                if type(self.backend) is JsonBackend:
                    entries = self.backend._metadata.get("entries", {})
                else:
                    entries = self.backend._entries
                inventory = self._inventory_state()
                # An old backend may contain canonical rows written before
                # this index existed.  Refusing to fabricate a lexical cursor
                # makes the required migration/rebuild explicit rather than
                # silently omitting legacy members from a claimed snapshot.
                if not inventory["events"] and entries:
                    raise CacheBlobMigrationRequiredError(
                        "Canonical manifest inventory rebuild is required",
                        context={"backend": type(self.backend).__name__},
                    )
                high_water = (
                    len(inventory["events"])
                    if cursor is None or cursor.snapshot_high_water is None
                    else cursor.snapshot_high_water
                )
                position = (
                    1
                    if cursor is None or cursor.next_sequence is None
                    else cursor.next_sequence
                )
                inspected = 0
                page_entries: list[tuple[str, bytes]] = []
                last_key = cursor.key if cursor is not None else "~"
                # ``max_inventory_items`` is now a per-call inspected-name
                # budget.  It is never a total-store eligibility ceiling.
                while (
                    position <= high_water
                    and inspected < self.lifecycle_limits.max_inventory_items
                    and len(page_entries) < limit
                ):
                    event_key, event_digest = inventory["events"][position - 1]
                    inspected += 1
                    position += 1
                    last_key = event_key
                    entry = entries.get(event_key)
                    try:
                        current_raw = self._raw_from_entry(entry)
                    except (TypeError, ValueError, CacheBlobMigrationRequiredError):
                        current_raw = None
                    if (
                        current_raw is not None
                        and hashlib.sha256(current_raw).hexdigest() == event_digest
                    ):
                        page_entries.append((event_key, current_raw))
                next_cursor = (
                    ManifestCursor(
                        last_key,
                        snapshot_high_water=high_water,
                        next_sequence=position,
                    )
                    if position <= high_water
                    else None
                )
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("list_page", self.backend, exc) from exc
        return ManifestPage(
            entries=tuple(page_entries),
            next_cursor=next_cursor,
        )

    def list_backend_entries(self) -> list[dict[str, Any]]:
        """Read backend projections without allowing operational failures to leak."""
        try:
            if type(self.backend) is JsonBackend:
                with self.backend._lock, self._json_compare_publish_lock():
                    self._refresh_json_for_conditional_operation()
                    entries = self.backend._metadata.get("entries", {})
                    return [
                        {"cache_key": key, **deepcopy(entry)}
                        for key, entry in entries.items()
                        if isinstance(key, str) and isinstance(entry, Mapping)
                    ]
            return self.backend.list_entries()
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("list_entries", self.backend, exc) from exc


class InMemoryManifestRepository(_MetadataManifestRepository):
    """Exact raw records backed by one process-local metadata identity."""

    def __init__(
        self,
        backend: InMemoryBackend,
        *,
        lifecycle_limits: LifecycleLimits | None = None,
        file_ops: ManagedFileOps | None = None,
    ):
        super().__init__(backend, lifecycle_limits=lifecycle_limits, file_ops=file_ops)


class JsonManifestRepository(_MetadataManifestRepository):
    """Exact raw records persisted by JsonBackend's durable document protocol."""

    def __init__(
        self,
        backend: JsonBackend,
        *,
        lifecycle_limits: LifecycleLimits | None = None,
        file_ops: ManagedFileOps | None = None,
    ):
        super().__init__(backend, lifecycle_limits=lifecycle_limits, file_ops=file_ops)


class SqliteManifestRepository:
    """Store canonical bytes in a dedicated SQLite BLOB table.

    The table is deliberately separate from ``cache_entries`` so canonical
    fields never pass through its fixed-column metadata projection.
    """

    def __init__(
        self, backend: SqliteBackend, *, lifecycle_limits: LifecycleLimits | None = None
    ):
        self.backend = backend
        self.lifecycle_limits = (
            LifecycleLimits() if lifecycle_limits is None else lifecycle_limits
        )
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
                connection.exec_driver_sql(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_SQLITE_MANIFEST_INVENTORY_TABLE} (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        logical_key TEXT NOT NULL,
                        record_digest TEXT NOT NULL
                    )
                    """
                )
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("initialize", self.backend, exc) from exc

    def refresh_authoritative_view(self) -> None:
        """SQLite reads are transactional and require no cached view refresh."""

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
                self._append_inventory_event(connection, key, record)
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("put_raw", self.backend, exc) from exc

    @contextmanager
    def _conditional_transaction(self):
        """Take SQLite's writer lock before comparing opaque authority bytes."""
        try:
            with self.backend._lock, self.backend.engine.connect() as connection:
                connection.exec_driver_sql("BEGIN IMMEDIATE")
                try:
                    yield connection
                except BaseException:
                    connection.rollback()
                    raise
                else:
                    connection.commit()
        except CacheBlobLifecycleConflictError:
            raise
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("conditional_transaction", self.backend, exc) from exc

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
            with self._conditional_transaction() as connection:
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
                self._append_inventory_event(connection, key, record)
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

    @staticmethod
    def _append_inventory_event(connection: Any, key: str, record: bytes) -> None:
        """Append publication membership in the same SQLite transaction."""
        connection.exec_driver_sql(
            f"INSERT INTO {_SQLITE_MANIFEST_INVENTORY_TABLE} "
            "(logical_key, record_digest) VALUES (?, ?)",
            (key, hashlib.sha256(record).hexdigest()),
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

    def remove_if_expected(self, key: str, expected: ManifestExpectation) -> None:
        """Atomically retire only the exact canonical record and its projection."""
        try:
            with self._conditional_transaction() as connection:
                row = connection.exec_driver_sql(
                    f"SELECT canonical_bytes FROM {_SQLITE_MANIFEST_TABLE} "
                    "WHERE logical_key = ?",
                    (key,),
                ).first()
                current_raw = None if row is None else bytes(row[0])
                if not expected.matches(current_raw):
                    raise CacheBlobLifecycleConflictError(
                        "Canonical manifest expectation no longer matches",
                        context={"key": key, "operation": "remove_if_expected"},
                    )
                connection.exec_driver_sql(
                    f"DELETE FROM {_SQLITE_MANIFEST_TABLE} WHERE logical_key = ?",
                    (key,),
                )
                connection.exec_driver_sql(
                    "DELETE FROM cache_entries WHERE cache_key = ?", (key,)
                )
        except CacheBlobLifecycleConflictError:
            raise
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("remove_if_expected", self.backend, exc) from exc

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

    def list_page(
        self,
        cursor: ManifestCursor | None = None,
        *,
        page_size: int | None = None,
    ) -> ManifestPage:
        """Fetch one high-water page using SQLite's indexed sequence table."""
        if cursor is not None and not isinstance(cursor, ManifestCursor):
            raise TypeError("manifest cursor must be a ManifestCursor or None")
        resolved = (
            self.lifecycle_limits.manifest_page_size
            if page_size is None
            else page_size
        )
        if type(resolved) is not int or resolved <= 0:
            raise ValueError("manifest page size must be a positive integer")
        if resolved > self.lifecycle_limits.manifest_page_size:
            raise ValueError("manifest page size exceeds configured lifecycle limit")
        try:
            with self.backend._lock, self.backend.engine.begin() as connection:
                high_water = (
                    cursor.snapshot_high_water
                    if cursor is not None and cursor.snapshot_high_water is not None
                    else connection.exec_driver_sql(
                        f"SELECT COALESCE(MAX(sequence), 0) "
                        f"FROM {_SQLITE_MANIFEST_INVENTORY_TABLE}"
                    ).scalar_one()
                )
                if high_water == 0:
                    raw_exists = connection.exec_driver_sql(
                        f"SELECT 1 FROM {_SQLITE_MANIFEST_TABLE} LIMIT 1"
                    ).first()
                    if raw_exists is not None:
                        raise CacheBlobMigrationRequiredError(
                            "SQLite canonical manifest inventory rebuild is required",
                            context={"backend": type(self.backend).__name__},
                        )
                position = (
                    cursor.next_sequence
                    if cursor is not None and cursor.next_sequence is not None
                    else 1
                )
                rows = connection.exec_driver_sql(
                    f"SELECT inventory.sequence, inventory.logical_key, "
                    "inventory.record_digest, manifests.canonical_bytes "
                    f"FROM {_SQLITE_MANIFEST_INVENTORY_TABLE} AS inventory "
                    f"LEFT JOIN {_SQLITE_MANIFEST_TABLE} AS manifests "
                    "ON manifests.logical_key = inventory.logical_key "
                    "WHERE inventory.sequence >= ? AND inventory.sequence <= ? "
                    "ORDER BY inventory.sequence LIMIT ?",
                    (
                        position,
                        high_water,
                        self.lifecycle_limits.max_inventory_items,
                    ),
                ).all()
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("list_page", self.backend, exc) from exc
        entries_list: list[tuple[str, bytes]] = []
        next_position = position
        last_key = cursor.key if cursor is not None else "~"
        for row in rows:
            sequence, key, digest, raw = int(row[0]), str(row[1]), str(row[2]), row[3]
            next_position = sequence + 1
            last_key = key
            if raw is not None:
                canonical = bytes(raw)
                if hashlib.sha256(canonical).hexdigest() == digest:
                    entries_list.append((key, canonical))
                    if len(entries_list) == resolved:
                        break
        entries = tuple(entries_list)
        return ManifestPage(
            entries=entries,
            next_cursor=(
                ManifestCursor(
                    last_key,
                    snapshot_high_water=int(high_water),
                    next_sequence=next_position,
                )
                if next_position <= int(high_water)
                else None
            ),
        )

    def list_backend_entries(self) -> list[dict[str, Any]]:
        """Read the SQLite compatibility projection with typed translation."""
        try:
            return self.backend.list_entries()
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("list_entries", self.backend, exc) from exc


def create_manifest_repository(
    backend: object,
    *,
    lifecycle_limits: LifecycleLimits | None = None,
    file_ops: ManagedFileOps | None = None,
) -> ManifestRepository:
    """Select the exact local adapter that can preserve canonical record bytes."""
    if type(backend) is InMemoryBackend:
        return InMemoryManifestRepository(
            backend, lifecycle_limits=lifecycle_limits, file_ops=file_ops
        )
    if type(backend) is JsonBackend:
        return JsonManifestRepository(
            backend, lifecycle_limits=lifecycle_limits, file_ops=file_ops
        )
    if type(backend) is SqliteBackend:
        return SqliteManifestRepository(backend, lifecycle_limits=lifecycle_limits)
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

    def __init__(
        self,
        backend: object,
        *,
        lifecycle_limits: LifecycleLimits | None = None,
        file_ops: ManagedFileOps | None = None,
    ):
        self._repository = create_manifest_repository(
            backend, lifecycle_limits=lifecycle_limits, file_ops=file_ops
        )

    def get_raw(self, key: str) -> Optional[bytes]:
        return self._repository.get_raw(key)

    def refresh_authoritative_view(self) -> None:
        self._repository.refresh_authoritative_view()

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

    def remove_if_expected(self, key: str, expected: ManifestExpectation) -> None:
        self._repository.remove_if_expected(key, expected)

    def list_keys(self) -> list[str]:
        return self._repository.list_keys()

    def list_page(
        self,
        cursor: ManifestCursor | None = None,
        *,
        page_size: int | None = None,
    ) -> ManifestPage:
        return self._repository.list_page(cursor, page_size=page_size)

    def list_backend_entries(self) -> list[dict[str, Any]]:
        return self._repository.list_backend_entries()


__all__ = [
    "ManifestRepository",
    "ManifestExpectation",
    "ManifestCursor",
    "ManifestPage",
    "InMemoryManifestRepository",
    "JsonManifestRepository",
    "SqliteManifestRepository",
    "MetadataManifestRepository",
    "create_manifest_repository",
]
