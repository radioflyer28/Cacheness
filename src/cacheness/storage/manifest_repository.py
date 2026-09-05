"""Lossless local persistence for canonical BlobStore manifest bytes.

This module intentionally has a narrow Phase 2 scope: it supports only the
exact local metadata backend identities that can truthfully preserve a
canonical record. General metadata-backend composition belongs to Phase 4.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import sqlite3
import tempfile
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
    CacheBlobManifestUnauthenticatedError,
    CacheBlobMigrationRequiredError,
    CacheManifestIntegrityError,
    CacheError,
)
from cacheness.metadata import InMemoryBackend, JsonBackend, SqliteBackend
from cacheness.json_utils import dumps as json_dumps
from cacheness.json_utils import loads as json_loads

from .coordination import interprocess_open_file_lock
from .integrity import ManifestKeyProvider, sign_hmac_sha256, verify_hmac_sha256
from .lifecycle_authority import LifecycleAuthority, ProjectionRevision
from .manifest import BlobManifestV1
from .path_security import ManagedFileOps, resolve_managed_locator

try:
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:  # pragma: no cover - SQLite support is optional at runtime.
    SQLAlchemyError = OSError


_RAW_MANIFEST_FIELD = "canonical_manifest_v1"
_MANIFEST_INVENTORY_FIELD = "_cacheness_manifest_inventory_v1"
_MANIFEST_INVENTORY_SCHEMA_VERSION = 3
_MANIFEST_INVENTORY_HEAD_MAX_BYTES = 4_096
_MANIFEST_INVENTORY_EVENT_MIN_BYTES = 32 * 1024
_MANIFEST_INVENTORY_COMPACTION_WINDOW = 64
_MANIFEST_INVENTORY_SEQUENCE_FIELD = "_cacheness_manifest_inventory_sequence_v2"
_MANIFEST_INVENTORY_TAIL_SCHEMA_VERSION = 2
_MANIFEST_INVENTORY_TAIL_WITNESS_SCHEMA_VERSION = 1
_SQLITE_MANIFEST_TABLE = "cacheness_manifest_records_v1"
_SQLITE_MANIFEST_INVENTORY_TABLE = "cacheness_manifest_inventory_v1"
_SQLITE_MANIFEST_INVENTORY_STATE_TABLE = "cacheness_manifest_inventory_state_v2"
_BACKEND_OPERATION_ERRORS = (
    CacheError,
    OSError,
    TypeError,
    ValueError,
    SQLAlchemyError,
)


class JsonProjectionExporter:
    """Rebuild compatible JSON metadata from a revision-bound authority snapshot.

    JSON is intentionally derived output: a failed or stale export is debt on
    the authority state, never a reason to roll back a committed lifecycle
    transition or to reconstruct authority rows.  The exporter writes only a
    bounded page at a time from an already-copied SQLite snapshot.
    """

    def __init__(
        self,
        authority: LifecycleAuthority,
        projection_path: Path | str,
        *,
        page_size: int = 128,
    ) -> None:
        if type(page_size) is not int or page_size <= 0:
            raise ValueError("projection page_size must be a positive integer")
        self.authority = authority
        self.projection_path = Path(projection_path)
        self.page_size = page_size

    @staticmethod
    def _entry_data(key: str, raw_manifest: bytes) -> dict[str, Any]:
        """Render the established public entry shape from one committed row."""
        manifest = BlobManifestV1.from_canonical_bytes(raw_manifest)
        if manifest.key != key or manifest.state != "committed":
            raise CacheBlobMigrationRequiredError(
                "Lifecycle authority projection entry is not committed"
            )
        return {
            "cache_key": manifest.key,
            "data_type": manifest.handler_type,
            "file_size": manifest.byte_size,
            "created_at": manifest.created_at,
            "metadata": {
                **dict(manifest.user_metadata),
                **dict(manifest.handler_metadata),
                "actual_path": manifest.locator,
            },
        }

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        """Acknowledge the compatible JSON directory entry durably."""
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _write_snapshot(self, snapshot_path: Path, revision: ProjectionRevision) -> Path:
        """Keyset-stream one private authority backup into a JSON candidate."""
        if not self.projection_path.parent.is_dir():
            raise CacheBlobBackendError(
                "JSON projection directory is unavailable",
                context={"operation": "export_json_projection"},
            )
        descriptor, candidate_name = tempfile.mkstemp(
            prefix=f".{self.projection_path.name}.",
            suffix=".tmp",
            dir=self.projection_path.parent,
        )
        candidate = Path(candidate_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                destination.write('{"entries":{')
                first = True
                last_key = ""
                connection = sqlite3.connect(
                    f"{snapshot_path.as_uri()}?mode=ro&immutable=1",
                    uri=True,
                    isolation_level=None,
                )
                try:
                    while True:
                        rows = connection.execute(
                            "SELECT key, manifest FROM entries WHERE key > ? "
                            "ORDER BY key LIMIT ?",
                            (last_key, self.page_size),
                        ).fetchall()
                        if not rows:
                            break
                        for key, raw_manifest in rows:
                            if not isinstance(key, str) or not isinstance(raw_manifest, bytes):
                                raise CacheBlobMigrationRequiredError(
                                    "Lifecycle authority projection row is incompatible"
                                )
                            if not first:
                                destination.write(",")
                            destination.write(json_dumps(key))
                            destination.write(":")
                            destination.write(json_dumps(self._entry_data(key, raw_manifest)))
                            first = False
                            last_key = key
                finally:
                    connection.close()
                destination.write("}")
                destination.write(
                    f',"cache_hits":0,"cache_misses":0,'
                    f'"_cacheness_authority_revision":{revision.value}}}'
                )
                destination.flush()
                os.fsync(destination.fileno())
            return candidate
        except BaseException:
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
            raise

    @contextmanager
    def _publish_lock(self):
        """Serialize derived publication without holding an authority transaction."""
        lock_path = self.projection_path.with_name(
            f".{self.projection_path.name}.projection.lock"
        )
        with open(lock_path, "a+b") as handle:
            with interprocess_open_file_lock(
                handle,
                exclusive=True,
                operation="json_projection_publish",
            ):
                yield

    def export(self) -> ProjectionRevision:
        """Atomically publish a snapshot and clean only its exact revision."""
        backup = getattr(self.authority, "projection_backup", None)
        if not callable(backup):
            raise CacheBlobBackendError(
                "Lifecycle authority cannot create a compatible JSON projection",
                context={"operation": "export_json_projection"},
            )
        candidate: Path | None = None
        try:
            with backup() as snapshot:
                candidate = self._write_snapshot(snapshot.path, snapshot.revision)
            with self._publish_lock():
                current_revision = self.authority.snapshot_state().revision
                if current_revision != snapshot.revision.value:
                    raise CacheBlobLifecycleConflictError("Projection revision changed")
                os.replace(candidate, self.projection_path)
                self._fsync_directory(self.projection_path)
                candidate = None
                return self.authority.compare_and_mark_projection(snapshot.revision)
        except (CacheBlobBackendError, CacheBlobLifecycleConflictError):
            raise
        except (CacheError, OSError, sqlite3.Error, ValueError) as error:
            raise CacheBlobBackendError(
                "Lifecycle authority JSON projection export failed",
                context={"operation": "export_json_projection"},
            ) from error
        finally:
            if candidate is not None:
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass


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
    # For a page that must be split into smaller durable clear sidecars, this
    # records the exact generation-bound cursor immediately after each yielded
    # record.  It avoids reconstructing a mutable lexical position from a key.
    entry_next_cursors: tuple[ManifestCursor, ...] = ()

    def __post_init__(self) -> None:
        """Require each yielded record to have one exact continuation cursor."""
        if self.entry_next_cursors and len(self.entry_next_cursors) != len(self.entries):
            raise ValueError("manifest entry cursors must align with page entries")


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

    def compact_inventory_for_recovery(self) -> bool:
        """Perform one bounded mutating pass and report snapshot readiness."""

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
        inventory_key_provider: Callable[[], bytes] | None = None,
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
        self._json_inventory_head_locator: Path | None = None
        self._json_inventory_tail_locator: Path | None = None
        self._json_inventory_tail_witness_locator: Path | None = None
        self._json_lock_guard = RLock()
        self._inventory_key_provider = inventory_key_provider
        self._fallback_inventory_key_provider: ManifestKeyProvider | None = None
        self._active_inventory_epoch: str | None = None
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
                self._json_inventory_head_locator = resolve_managed_locator(
                    self._json_lock_file_ops.root,
                    f".{Path(backend.metadata_file).name}.manifest-inventory-v2-head.json",
                    operation="manifest_inventory",
                    allow_missing_leaf=True,
                )
                self._json_inventory_tail_locator = resolve_managed_locator(
                    self._json_lock_file_ops.root,
                    f".{Path(backend.metadata_file).name}.manifest-inventory-v2-tail.json",
                    operation="manifest_inventory",
                    allow_missing_leaf=True,
                )
                self._json_inventory_tail_witness_locator = resolve_managed_locator(
                    self._json_lock_file_ops.root,
                    f".{Path(backend.metadata_file).name}.manifest-inventory-v2-tail-witness.json",
                    operation="manifest_inventory",
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

        if self._inventory_key_provider is None and type(backend) is InMemoryBackend:
            # In-memory backends never persist control records beyond this
            # backend object. Retain a random key on that object so separately
            # constructed repositories share one trust root without inventing
            # a deterministic/public fallback key.
            key = getattr(backend, "_cacheness_manifest_inventory_key_v3", None)
            if type(key) is not bytes or len(key) != 32:
                key = os.urandom(32)
                setattr(backend, "_cacheness_manifest_inventory_key_v3", key)
            self._inventory_key_provider = lambda: key

    def _inventory_store_id(self) -> str:
        """Return a stable managed-root identity for scheduler signatures."""
        if self._json_lock_file_ops is not None:
            root = self._json_lock_file_ops.root
            device, inode = self._json_lock_file_ops.root_identity
            material = f"{root}:{device}:{inode}"
        else:
            material = f"memory:{id(self.backend)}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _inventory_key(self, *, initialize_new_store: bool = False) -> bytes:
        """Read the scheduler trust root without accepting a public fallback."""
        provider = self._inventory_key_provider
        if provider is not None:
            key = provider()
        else:
            if self._json_lock_file_ops is None:
                raise CacheBlobMigrationRequiredError(
                    "Manifest inventory requires a signing-key provider",
                    context={"operation": "manifest_inventory"},
                )
            if self._fallback_inventory_key_provider is None:
                self._fallback_inventory_key_provider = ManifestKeyProvider(
                    self._json_lock_file_ops.root
                    / f".{Path(self.backend.metadata_file).name}.manifest-inventory-key.bin",
                    lifecycle_limits=self.lifecycle_limits,
                )
            try:
                key = (
                    self._fallback_inventory_key_provider.get_or_initialize_new_store()
                    if initialize_new_store
                    else self._fallback_inventory_key_provider.get_key()
                )
            except CacheBlobManifestUnauthenticatedError as exc:
                raise CacheBlobMigrationRequiredError(
                    "Manifest inventory requires an explicit migration before key initialization",
                    context={"operation": "manifest_inventory"},
                ) from exc
        if type(key) is not bytes or len(key) != 32:
            raise CacheBlobManifestUnauthenticatedError(
                "Manifest inventory signing key is invalid"
            )
        return key

    @staticmethod
    def _inventory_signing_bytes(value: Mapping[str, Any]) -> bytes:
        """Return the canonical domain-separated scheduler HMAC preimage."""
        unsigned = dict(value)
        unsigned.pop("signature", None)
        return b"cacheness.manifest-inventory.v4\x00" + json_dumps(
            unsigned, default=str
        ).encode("utf-8")

    def _sign_inventory_value(
        self, value: Mapping[str, Any], *, initialize_new_store: bool = False
    ) -> dict[str, Any]:
        signed = dict(value)
        signed["signature"] = sign_hmac_sha256(
            self._inventory_signing_bytes(signed),
            self._inventory_key(initialize_new_store=initialize_new_store),
        )
        return signed

    def _verify_inventory_value(self, value: Mapping[str, Any]) -> bool:
        signature = value.get("signature")
        return isinstance(signature, str) and verify_hmac_sha256(
            self._inventory_signing_bytes(value), signature, self._inventory_key()
        )

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
        self._json_inventory_head_locator = None
        self._json_inventory_tail_locator = None
        self._json_inventory_tail_witness_locator = None

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

    def _json_inventory_event_locator(self, sequence: int) -> Path:
        """Return one immutable JSON scheduling event without reading history."""
        if type(sequence) is not int or sequence <= 0:
            raise ValueError("manifest inventory sequence is invalid")
        if self._json_lock_file_ops is None:
            raise CacheBlobBackendError("JSON manifest inventory root is unavailable")
        return resolve_managed_locator(
            self._json_lock_file_ops.root,
            f".{Path(self.backend.metadata_file).name}.manifest-inventory-v2-"
            f"event-{sequence:020d}.json",
            operation="manifest_inventory",
            allow_missing_leaf=True,
        )

    def _json_inventory_skip_locator(self, sequence: int) -> Path:
        """Return one bounded sparse-successor marker for a retired run.

        Markers are scheduling-only accelerators.  They never establish
        membership: the target event is still authenticated by its immutable
        digest and current authoritative projection before a page yields it.
        """
        if type(sequence) is not int or sequence <= 0:
            raise ValueError("manifest inventory sparse-run sequence is invalid")
        if self._json_lock_file_ops is None:
            raise CacheBlobBackendError("JSON manifest inventory root is unavailable")
        return resolve_managed_locator(
            self._json_lock_file_ops.root,
            f".{Path(self.backend.metadata_file).name}.manifest-inventory-v2-"
            f"skip-{sequence:020d}.json",
            operation="manifest_inventory",
            allow_missing_leaf=True,
        )

    def _json_inventory_tail_locator_for_state(self) -> Path:
        """Return the one signed append-tail anchor for this inventory."""
        if self._json_inventory_tail_locator is None:
            raise CacheBlobBackendError("JSON manifest inventory root is unavailable")
        return self._json_inventory_tail_locator

    def _json_inventory_tail_witness_locator_for_state(self) -> Path:
        """Return the bounded proof that a compacted tail member was exact."""
        if self._json_inventory_tail_witness_locator is None:
            raise CacheBlobBackendError("JSON manifest inventory root is unavailable")
        return self._json_inventory_tail_witness_locator

    def _inventory_event_max_bytes(self) -> int:
        """Keep one event bounded independently from manifest record lifetime."""
        return max(
            _MANIFEST_INVENTORY_EVENT_MIN_BYTES,
            self.lifecycle_limits.max_operation_field_bytes * 2 + 512,
        )

    def _empty_inventory_state(self) -> dict[str, Any]:
        """Build the fixed-size v2 sequence head used by every local backend."""
        return {
            "version": _MANIFEST_INVENTORY_SCHEMA_VERSION,
            "store_id": self._inventory_store_id(),
            # A fresh random epoch distinguishes control objects produced by
            # separate initialization eras under the same persistent key.
            "epoch": secrets.token_hex(16),
            "next_sequence": 1,
            "compact_next_sequence": 1,
            # Sequence positions strictly below this floor have been
            # exact-revalidated as stale.  Keep it in the bounded head rather
            # than making every memory/JSON caller replay historical gaps.
            "first_live_sequence": 1,
            # ``0`` means no post-authority stale-event debt is known.  A
            # non-zero value is an exact high-water target; compaction resumes
            # from ``compact_next_sequence`` in bounded calls until it reaches
            # that target.  It is intentionally separate from normal reads.
            "maintenance_target_sequence": 0,
            # The first sequence in an in-progress sparse run.  Its durable
            # marker is extended only after every intervening event has been
            # exact-revalidated stale.  Zero means there is no run currently
            # being extended by maintenance.
            "open_sparse_run_start": 0,
            # The current tail event normally remains immutable.  If bounded
            # maintenance retires that exact stale event, this separate signed
            # witness proves the marker and terminal tuple before a reader may
            # accept the now-absent tail member.
            "tail_witness": None,
        }

    @staticmethod
    def _inventory_head_field_names() -> tuple[str, ...]:
        """Return the fixed signed scheduler-head schema in one place."""
        return (
            "version",
            "store_id",
            "epoch",
            "next_sequence",
            "compact_next_sequence",
            "first_live_sequence",
            "maintenance_target_sequence",
            "open_sparse_run_start",
            "signature",
        )

    def _validate_inventory_head(self, state: object) -> dict[str, Any]:
        """Validate the durable head without loading any event history."""
        if not isinstance(state, dict) or set(state) != {
            "version",
            "store_id",
            "epoch",
            "next_sequence",
            "compact_next_sequence",
            "first_live_sequence",
            "maintenance_target_sequence",
            "open_sparse_run_start",
            "signature",
        }:
            raise CacheBlobBackendError(
                "Canonical manifest inventory is invalid",
                context={"operation": "manifest_inventory"},
            )
        if (
            state["version"] != _MANIFEST_INVENTORY_SCHEMA_VERSION
            or state["store_id"] != self._inventory_store_id()
            or not isinstance(state["epoch"], str)
            or len(state["epoch"]) != 32
            or any(character not in "0123456789abcdef" for character in state["epoch"])
            or not isinstance(state["signature"], str)
            or type(state["next_sequence"]) is not int
            or type(state["compact_next_sequence"]) is not int
            or state["next_sequence"] <= 0
            or not 1 <= state["compact_next_sequence"] <= state["next_sequence"]
        ):
            raise CacheBlobBackendError(
                "Canonical manifest inventory version is invalid",
                context={"operation": "manifest_inventory"},
            )
        if (
            type(state["first_live_sequence"]) is not int
            or not 1 <= state["first_live_sequence"] <= state["next_sequence"]
        ):
            raise CacheBlobBackendError(
                "Canonical manifest inventory live-sequence floor is invalid",
                context={"operation": "manifest_inventory"},
            )
        if (
            type(state["maintenance_target_sequence"]) is not int
            or not 0 <= state["maintenance_target_sequence"] < state["next_sequence"]
        ):
            raise CacheBlobBackendError(
                "Canonical manifest inventory maintenance target is invalid",
                context={"operation": "manifest_inventory"},
            )
        if (
            type(state["open_sparse_run_start"]) is not int
            or not 0 <= state["open_sparse_run_start"] < state["next_sequence"]
        ):
            raise CacheBlobBackendError(
                "Canonical manifest inventory sparse-run continuation is invalid",
                context={"operation": "manifest_inventory"},
            )
        if not self._verify_inventory_value(state):
            raise CacheManifestIntegrityError(
                "Canonical manifest inventory head is unauthenticated"
            )
        return state

    def _inventory_state(
        self, *, allow_unacknowledged_tail: bool = False
    ) -> dict[str, Any]:
        """Load only a bounded v2 head and fail closed for pre-index evidence."""
        if type(self.backend) is JsonBackend:
            if self._json_lock_file_ops is None or self._json_inventory_head_locator is None:
                raise CacheBlobBackendError("JSON manifest inventory root is unavailable")
            try:
                raw = self._json_lock_file_ops.read_bytes_bounded(
                    self._json_inventory_head_locator,
                    max_bytes=_MANIFEST_INVENTORY_HEAD_MAX_BYTES,
                )
            except FileNotFoundError:
                if (
                    _MANIFEST_INVENTORY_FIELD in self.backend._metadata
                    or self.backend._metadata.get("entries")
                ):
                    raise CacheBlobMigrationRequiredError(
                        "Canonical manifest inventory migration is required",
                        context={"backend": type(self.backend).__name__},
                    )
                state = self._empty_inventory_state()
                self._active_inventory_epoch = state["epoch"]
                return state
            try:
                state = self._validate_inventory_head(json_loads(raw))
                self._active_inventory_epoch = state["epoch"]
                self._verify_inventory_tail(
                    state, allow_unacknowledged=allow_unacknowledged_tail
                )
                return state
            except (TypeError, ValueError) as exc:
                raise CacheBlobBackendError(
                    "Canonical manifest inventory is invalid",
                    context={"operation": "manifest_inventory"},
                ) from exc
        state = getattr(self.backend, "_cacheness_manifest_inventory_v2", None)
        if state is None:
            if self.backend._entries or getattr(self.backend, _MANIFEST_INVENTORY_FIELD, None):
                raise CacheBlobMigrationRequiredError(
                    "Canonical manifest inventory migration is required",
                    context={"backend": type(self.backend).__name__},
                )
            # Constructor recovery is read-only for a never-used backend.
            # The first scheduler append publishes this fresh head together
            # with an event, after BlobStore has performed its explicit
            # manifest-key initialization boundary.
            state = {
                **self._empty_inventory_state(),
                "events": {},
                "skips": {},
                "tail": None,
            }
            self._active_inventory_epoch = state["epoch"]
            return state
        if not isinstance(state, dict):
            raise CacheBlobBackendError(
                "Canonical manifest inventory is invalid",
                context={"operation": "manifest_inventory"},
            )
        expected_fields = {
            *self._inventory_head_field_names(), "events", "skips", "tail"
        }
        if set(state) not in (expected_fields, {*expected_fields, "tail_witness"}):
            raise CacheBlobBackendError(
                "Canonical manifest inventory is invalid",
                context={"operation": "manifest_inventory"},
            )
        head = self._validate_inventory_head(
            {name: state[name] for name in self._inventory_head_field_names()}
        )
        state.update(head)
        self._active_inventory_epoch = head["epoch"]
        if not isinstance(state["events"], dict):
            raise CacheBlobBackendError(
                "Canonical manifest inventory events are invalid",
                context={"operation": "manifest_inventory"},
            )
        if "skips" not in state:
            state["skips"] = {}
        if "tail_witness" not in state:
            state["tail_witness"] = None
        if not isinstance(state["skips"], dict):
            raise CacheBlobBackendError(
                "Canonical manifest inventory sparse runs are invalid",
                context={"operation": "manifest_inventory"},
            )
        self._verify_inventory_tail(
            state, allow_unacknowledged=allow_unacknowledged_tail
        )
        return state

    def _write_inventory_head(self, state: dict[str, Any]) -> None:
        """Persist only the fixed-size JSON head after an immutable event."""
        state["store_id"] = self._inventory_store_id()
        unsigned_head = {
            name: state[name]
            for name in self._inventory_head_field_names()
            if name != "signature"
        }
        signed = self._sign_inventory_value(unsigned_head, initialize_new_store=True)
        state.update(signed)
        self._active_inventory_epoch = signed["epoch"]
        head = self._validate_inventory_head(signed)
        if type(self.backend) is not JsonBackend:
            setattr(self.backend, "_cacheness_manifest_inventory_v2", state)
            return
        if self._json_lock_file_ops is None or self._json_inventory_head_locator is None:
            raise CacheBlobBackendError("JSON manifest inventory root is unavailable")
        encoded = json_dumps(head, default=str).encode("utf-8")
        if len(encoded) > _MANIFEST_INVENTORY_HEAD_MAX_BYTES:
            raise AssertionError("manifest inventory head unexpectedly exceeds bound")
        self._json_lock_file_ops.write_bytes_durable(self._json_inventory_head_locator, encoded)

    def _validate_inventory_tail(
        self, tail: object, state: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Validate the monotonic append anchor independently from a head.

        The head carries paging and maintenance cursors and can therefore be
        rewritten after an append.  The tail is a separate signed control
        object whose only mutable claim is the next allocated sequence.  A
        replayed earlier head cannot make a later immutable event invisible
        without also replacing this store-local authority object.
        """
        if isinstance(tail, dict) and tail.get("version") == 1:
            raise CacheBlobMigrationRequiredError(
                "Canonical manifest inventory append-tail migration is required",
                context={"operation": "manifest_inventory"},
            )
        if not isinstance(tail, dict) or set(tail) != {
            "version",
            "store_id",
            "epoch",
            "next_sequence",
            "terminal_sequence",
            "terminal_key",
            "terminal_digest",
            "signature",
        }:
            raise CacheBlobBackendError(
                "Canonical manifest inventory tail is invalid",
                context={"operation": "manifest_inventory"},
            )
        if (
            tail["version"] != _MANIFEST_INVENTORY_TAIL_SCHEMA_VERSION
            or tail["store_id"] != self._inventory_store_id()
            or tail["epoch"] != state["epoch"]
            or type(tail["next_sequence"]) is not int
            or tail["next_sequence"] <= 1
            or tail["terminal_sequence"] != tail["next_sequence"] - 1
            or not isinstance(tail["terminal_key"], str)
            or not tail["terminal_key"]
            or len(tail["terminal_key"].encode("utf-8")) > 8_192
            or not isinstance(tail["terminal_digest"], str)
            or len(tail["terminal_digest"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in tail["terminal_digest"]
            )
            or not isinstance(tail["signature"], str)
        ):
            raise CacheBlobBackendError(
                "Canonical manifest inventory tail is invalid",
                context={"operation": "manifest_inventory"},
            )
        if not self._verify_inventory_value(tail):
            raise CacheManifestIntegrityError(
                "Canonical manifest inventory tail is unauthenticated"
            )
        return tail

    @staticmethod
    def _tail_matches_inventory_event(
        tail: Mapping[str, Any], event: tuple[str, str], sequence: int
    ) -> bool:
        """Return whether one append-tail binds this exact terminal event."""
        key, digest = event
        return (
            tail["terminal_sequence"] == sequence
            and tail["terminal_key"] == key
            and tail["terminal_digest"] == digest
        )

    def _read_inventory_tail(self, state: Mapping[str, Any]) -> dict[str, Any] | None:
        """Read the bounded signed append tail without probing event names."""
        if type(self.backend) is not JsonBackend:
            tail = state.get("tail")
            return None if tail is None else self._validate_inventory_tail(tail, state)
        if self._json_lock_file_ops is None:
            raise CacheBlobBackendError("JSON manifest inventory root is unavailable")
        try:
            raw = self._json_lock_file_ops.read_bytes_bounded(
                self._json_inventory_tail_locator_for_state(),
                max_bytes=_MANIFEST_INVENTORY_HEAD_MAX_BYTES,
            )
        except FileNotFoundError:
            return None
        try:
            return self._validate_inventory_tail(json_loads(raw), state)
        except (TypeError, ValueError) as exc:
            raise CacheBlobBackendError(
                "Canonical manifest inventory tail is invalid",
                context={"operation": "manifest_inventory"},
            ) from exc

    def _validate_inventory_tail_witness(
        self,
        witness: object,
        tail: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate one exact proof for a deliberately retired tail member."""
        if not isinstance(witness, dict) or set(witness) != {
            "version",
            "store_id",
            "epoch",
            "terminal_sequence",
            "terminal_key",
            "terminal_digest",
            "marker_sequence",
            "marker_next_sequence",
            "marker_run_digest",
            "signature",
        }:
            raise CacheBlobBackendError(
                "Canonical manifest inventory tail witness is invalid",
                context={"operation": "manifest_inventory"},
            )
        if (
            witness["version"] != _MANIFEST_INVENTORY_TAIL_WITNESS_SCHEMA_VERSION
            or witness["store_id"] != self._inventory_store_id()
            or witness["epoch"] != state["epoch"]
            or witness["terminal_sequence"] != tail["terminal_sequence"]
            or witness["terminal_key"] != tail["terminal_key"]
            or witness["terminal_digest"] != tail["terminal_digest"]
            or type(witness["marker_sequence"]) is not int
            or type(witness["marker_next_sequence"]) is not int
            or not 1 <= witness["marker_sequence"] <= witness["terminal_sequence"]
            or not witness["terminal_sequence"] < witness["marker_next_sequence"]
            or witness["marker_next_sequence"] > state["next_sequence"]
            or not isinstance(witness["marker_run_digest"], str)
            or len(witness["marker_run_digest"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in witness["marker_run_digest"]
            )
            or not isinstance(witness["signature"], str)
        ):
            raise CacheBlobBackendError(
                "Canonical manifest inventory tail witness is invalid",
                context={"operation": "manifest_inventory"},
            )
        if not self._verify_inventory_value(witness):
            raise CacheManifestIntegrityError(
                "Canonical manifest inventory tail witness is unauthenticated"
            )
        return witness

    def _read_inventory_tail_witness(
        self, tail: Mapping[str, Any], state: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Read the fixed-size proof required after a tail event is compacted."""
        if type(self.backend) is not JsonBackend:
            witness = state.get("tail_witness")
            return (
                None
                if witness is None
                else self._validate_inventory_tail_witness(witness, tail, state)
            )
        if self._json_lock_file_ops is None:
            raise CacheBlobBackendError("JSON manifest inventory root is unavailable")
        try:
            raw = self._json_lock_file_ops.read_bytes_bounded(
                self._json_inventory_tail_witness_locator_for_state(),
                max_bytes=_MANIFEST_INVENTORY_HEAD_MAX_BYTES,
            )
        except FileNotFoundError:
            return None
        try:
            return self._validate_inventory_tail_witness(json_loads(raw), tail, state)
        except (TypeError, ValueError) as exc:
            raise CacheBlobBackendError(
                "Canonical manifest inventory tail witness is invalid",
                context={"operation": "manifest_inventory"},
            ) from exc

    def _write_inventory_tail_witness(
        self,
        state: dict[str, Any],
        tail: Mapping[str, Any],
        marker: Mapping[str, Any],
    ) -> None:
        """Publish exact terminal-retirement proof before unlinking its event."""
        witness = self._sign_inventory_value(
            {
                "version": _MANIFEST_INVENTORY_TAIL_WITNESS_SCHEMA_VERSION,
                "store_id": self._inventory_store_id(),
                "epoch": state["epoch"],
                "terminal_sequence": tail["terminal_sequence"],
                "terminal_key": tail["terminal_key"],
                "terminal_digest": tail["terminal_digest"],
                "marker_sequence": marker["start_sequence"],
                "marker_next_sequence": marker["next_sequence"],
                "marker_run_digest": marker["run_digest"],
            },
            initialize_new_store=True,
        )
        self._validate_inventory_tail_witness(witness, tail, state)
        if type(self.backend) is not JsonBackend:
            state["tail_witness"] = witness
            return
        if self._json_lock_file_ops is None:
            raise CacheBlobBackendError("JSON manifest inventory root is unavailable")
        encoded = json_dumps(witness, default=str).encode("utf-8")
        if len(encoded) > _MANIFEST_INVENTORY_HEAD_MAX_BYTES:
            raise AssertionError("manifest inventory tail witness unexpectedly exceeds bound")
        self._json_lock_file_ops.write_bytes_durable(
            self._json_inventory_tail_witness_locator_for_state(), encoded
        )

    def _verify_acknowledged_inventory_terminal(
        self, tail: Mapping[str, Any], state: dict[str, Any]
    ) -> None:
        """Require every accepted high water to retain its exact terminal proof."""
        sequence = tail["terminal_sequence"]
        event = self._read_inventory_event(sequence, state=state)
        if event is not None:
            if self._tail_matches_inventory_event(tail, event, sequence):
                return
            raise CacheManifestIntegrityError(
                "Canonical manifest inventory tail terminal event is inconsistent",
                context={"operation": "manifest_inventory", "sequence": sequence},
            )
        witness = self._read_inventory_tail_witness(tail, state)
        if witness is None:
            raise CacheManifestIntegrityError(
                "Canonical manifest inventory tail terminal event is missing",
                context={"operation": "manifest_inventory", "sequence": sequence},
            )
        marker = self._read_inventory_skip_record(witness["marker_sequence"], state)
        if (
            marker is None
            or marker["next_sequence"] != witness["marker_next_sequence"]
            or marker["run_digest"] != witness["marker_run_digest"]
        ):
            raise CacheManifestIntegrityError(
                "Canonical manifest inventory tail witness marker is inconsistent",
                context={"operation": "manifest_inventory", "sequence": sequence},
            )

    def _verify_inventory_tail(
        self, state: Mapping[str, Any], *, allow_unacknowledged: bool = False
    ) -> dict[str, Any] | None:
        """Reject a head that is not exactly anchored to the append tail."""
        tail = self._read_inventory_tail(state)
        if state["next_sequence"] == 1:
            if tail is not None:
                raise CacheBlobBackendError(
                    "Canonical manifest inventory tail is ahead of an empty head",
                    context={"operation": "manifest_inventory"},
                )
            return None
        if tail is None:
            raise CacheBlobMigrationRequiredError(
                "Canonical manifest inventory requires append-tail migration",
                context={"operation": "manifest_inventory"},
            )
        if tail["next_sequence"] == state["next_sequence"]:
            self._verify_acknowledged_inventory_terminal(tail, state)
            return tail
        # This is the narrow durable crash window after the immutable event
        # and tail commit but before head acknowledgement.  Only an append
        # transition may resume it; normal pages and recovery readers fail
        # closed rather than silently choosing one control object.
        if (
            allow_unacknowledged
            and tail["next_sequence"] == state["next_sequence"] + 1
        ):
            return tail
        raise CacheManifestIntegrityError(
            "Canonical manifest inventory head does not match its append tail"
        )

    def _acknowledge_unacknowledged_inventory_tail(
        self, state: dict[str, Any]
    ) -> None:
        """Acknowledge only the immutable member already bound by a tail.

        This is the sole durable crash window between an event/tail commit and
        the bounded head acknowledgement.  It runs before append attempts so a
        replayed head plus a missing terminal event cannot be replaced by a
        different writer at the same sequence.
        """
        tail = self._read_inventory_tail(state)
        if tail is None or tail["next_sequence"] == state["next_sequence"]:
            return
        if tail["next_sequence"] != state["next_sequence"] + 1:
            raise CacheManifestIntegrityError(
                "Canonical manifest inventory head does not match its append tail"
            )
        sequence = state["next_sequence"]
        event = self._read_inventory_event(sequence, state=state)
        if event is None:
            raise CacheManifestIntegrityError(
                "Canonical manifest inventory tail terminal event is missing",
                context={"operation": "manifest_inventory", "sequence": sequence},
            )
        if not self._tail_matches_inventory_event(tail, event, sequence):
            raise CacheManifestIntegrityError(
                "Canonical manifest inventory tail terminal event is inconsistent",
                context={"operation": "manifest_inventory", "sequence": sequence},
            )
        state["next_sequence"] = sequence + 1
        self._write_inventory_head(state)

    def _write_inventory_tail(
        self,
        state: dict[str, Any],
        next_sequence: int,
        *,
        terminal_event: tuple[str, str],
    ) -> None:
        """Durably advance the signed append tail after an immutable event."""
        terminal_sequence = next_sequence - 1
        if terminal_sequence <= 0:
            raise ValueError("manifest inventory tail requires a terminal event")
        existing = self._read_inventory_tail(state)
        if existing is not None:
            if existing["next_sequence"] == next_sequence:
                if not self._tail_matches_inventory_event(
                    existing, terminal_event, terminal_sequence
                ):
                    raise CacheManifestIntegrityError(
                        "Canonical manifest inventory tail terminal event changed"
                    )
                return
            if existing["next_sequence"] > next_sequence:
                raise CacheManifestIntegrityError(
                    "Canonical manifest inventory tail cannot move backwards"
                )
        tail = self._sign_inventory_value(
            {
                "version": _MANIFEST_INVENTORY_TAIL_SCHEMA_VERSION,
                "store_id": self._inventory_store_id(),
                "epoch": state["epoch"],
                "next_sequence": next_sequence,
                "terminal_sequence": terminal_sequence,
                "terminal_key": terminal_event[0],
                "terminal_digest": terminal_event[1],
            },
            initialize_new_store=True,
        )
        self._validate_inventory_tail(tail, state)
        if type(self.backend) is not JsonBackend:
            state["tail"] = tail
            return
        if self._json_lock_file_ops is None:
            raise CacheBlobBackendError("JSON manifest inventory root is unavailable")
        encoded = json_dumps(tail, default=str).encode("utf-8")
        if len(encoded) > _MANIFEST_INVENTORY_HEAD_MAX_BYTES:
            raise AssertionError("manifest inventory tail unexpectedly exceeds bound")
        self._json_lock_file_ops.write_bytes_durable(
            self._json_inventory_tail_locator_for_state(), encoded
        )

    def _validate_inventory_event(self, event: object, sequence: int) -> tuple[str, str]:
        """Validate a non-authoritative event before it drives a page position."""
        if (
            not isinstance(event, dict)
            or set(event)
            != {
                "version",
                "store_id",
                "epoch",
                "sequence",
                "key",
                "digest",
                "signature",
            }
            or event["version"] != _MANIFEST_INVENTORY_SCHEMA_VERSION
            or event["store_id"] != self._inventory_store_id()
            or event["epoch"] != self._active_inventory_epoch
            or event["sequence"] != sequence
            or not isinstance(event["key"], str)
            or not event["key"]
            or len(event["key"].encode("utf-8")) > 8_192
            or not isinstance(event["digest"], str)
            or len(event["digest"]) != 64
            or any(character not in "0123456789abcdef" for character in event["digest"])
        ):
            raise CacheBlobBackendError(
                "Canonical manifest inventory event is invalid",
                context={"operation": "manifest_inventory", "sequence": sequence},
            )
        if not self._verify_inventory_value(event):
            raise CacheManifestIntegrityError(
                "Canonical manifest inventory event is unauthenticated"
            )
        return event["key"], event["digest"]

    def _read_inventory_event(
        self, sequence: int, *, state: Mapping[str, Any] | None = None
    ) -> tuple[str, str] | None:
        """Read one event directly; a compacted slot is a safe sparse gap."""
        if type(self.backend) is JsonBackend:
            if self._json_lock_file_ops is None:
                raise CacheBlobBackendError("JSON manifest inventory root is unavailable")
            try:
                raw = self._json_lock_file_ops.read_bytes_bounded(
                    self._json_inventory_event_locator(sequence),
                    max_bytes=self._inventory_event_max_bytes(),
                )
            except FileNotFoundError:
                return None
            try:
                return self._validate_inventory_event(json_loads(raw), sequence)
            except (TypeError, ValueError) as exc:
                raise CacheBlobBackendError(
                    "Canonical manifest inventory event is invalid",
                    context={"operation": "manifest_inventory", "sequence": sequence},
                ) from exc
        inventory_state = self._inventory_state() if state is None else state
        event = inventory_state["events"].get(sequence)
        return None if event is None else self._validate_inventory_event(event, sequence)

    def _validate_inventory_skip(
        self, marker: object, sequence: int, *, next_sequence: int
    ) -> dict[str, Any]:
        """Validate a sparse-successor marker before it can skip positions."""
        if (
            not isinstance(marker, dict)
            or set(marker)
            != {
                "version",
                "store_id",
                "epoch",
                "start_sequence",
                "next_sequence",
                "run_digest",
                "signature",
            }
            or marker["version"] != _MANIFEST_INVENTORY_SCHEMA_VERSION
            or marker["store_id"] != self._inventory_store_id()
            or marker["epoch"] != self._active_inventory_epoch
            or marker["start_sequence"] != sequence
            or type(marker["next_sequence"]) is not int
            or not sequence < marker["next_sequence"] <= next_sequence
            or not isinstance(marker["run_digest"], str)
            or len(marker["run_digest"]) != 64
            or any(character not in "0123456789abcdef" for character in marker["run_digest"])
        ):
            raise CacheBlobBackendError(
                "Canonical manifest inventory sparse-successor marker is invalid",
                context={"operation": "manifest_inventory", "sequence": sequence},
            )
        if not self._verify_inventory_value(marker):
            raise CacheManifestIntegrityError(
                "Canonical manifest inventory sparse-successor marker is unauthenticated"
            )
        return marker

    def _read_inventory_skip_record(
        self, sequence: int, state: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Read one authenticated sparse-run proof without interpreting it."""
        if type(self.backend) is JsonBackend:
            if self._json_lock_file_ops is None:
                raise CacheBlobBackendError("JSON manifest inventory root is unavailable")
            try:
                raw = self._json_lock_file_ops.read_bytes_bounded(
                    self._json_inventory_skip_locator(sequence),
                    max_bytes=_MANIFEST_INVENTORY_HEAD_MAX_BYTES,
                )
            except FileNotFoundError:
                return None
            try:
                return self._validate_inventory_skip(
                    json_loads(raw), sequence, next_sequence=state["next_sequence"]
                )
            except (TypeError, ValueError) as exc:
                raise CacheBlobBackendError(
                    "Canonical manifest inventory sparse-successor marker is invalid",
                    context={"operation": "manifest_inventory", "sequence": sequence},
                ) from exc
        marker = state["skips"].get(sequence)
        return (
            None
            if marker is None
            else self._validate_inventory_skip(
                marker, sequence, next_sequence=state["next_sequence"]
            )
        )

    def _read_inventory_skip(
        self, sequence: int, state: dict[str, Any]
    ) -> int | None:
        """Read a non-authoritative sparse successor or fail closed if malformed."""
        marker = self._read_inventory_skip_record(sequence, state)
        return None if marker is None else marker["next_sequence"]

    @staticmethod
    def _inventory_skip_run_digest(
        *,
        start_sequence: int,
        successor: int,
        previous_digest: str | None,
        event: tuple[str, str] | None,
    ) -> str:
        """Extend a bounded authenticated proof of exact skipped evidence.

        Each marker commits to the previous authenticated run digest and the
        exact event (or proved-absent slot) just revalidated by compaction.
        The control record therefore remains fixed-size even if a stale run
        spans many bounded maintenance calls, while changing its range or any
        constituent event requires a new store-key HMAC.
        """
        proof = {
            "event": None if event is None else {"key": event[0], "digest": event[1]},
            "previous_digest": previous_digest,
            "start_sequence": start_sequence,
            "successor": successor,
        }
        return hashlib.sha256(
            b"cacheness.manifest-inventory.skip-run.v3\x00"
            + json_dumps(proof, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _write_inventory_skip(
        self,
        sequence: int,
        successor: int,
        state: dict[str, Any],
        *,
        event: tuple[str, str] | None,
    ) -> dict[str, Any]:
        """Durably extend one exact stale run before retiring its final event."""
        existing = self._read_inventory_skip_record(sequence, state)
        if existing is not None and existing["next_sequence"] >= successor:
            return existing
        if existing is not None and existing["next_sequence"] != successor - 1:
            raise CacheBlobBackendError(
                "Canonical manifest inventory sparse-successor continuation is invalid",
                context={"operation": "manifest_inventory", "sequence": sequence},
            )
        marker = self._sign_inventory_value({
            "version": _MANIFEST_INVENTORY_SCHEMA_VERSION,
            "store_id": self._inventory_store_id(),
            "epoch": state["epoch"],
            "start_sequence": sequence,
            "next_sequence": successor,
            "run_digest": self._inventory_skip_run_digest(
                start_sequence=sequence,
                successor=successor,
                previous_digest=(
                    None if existing is None else existing["run_digest"]
                ),
                event=event,
            ),
        }, initialize_new_store=True)
        self._validate_inventory_skip(
            marker, sequence, next_sequence=state["next_sequence"]
        )
        if type(self.backend) is not JsonBackend:
            state["skips"][sequence] = marker
            return marker
        if self._json_lock_file_ops is None:
            raise CacheBlobBackendError("JSON manifest inventory root is unavailable")
        encoded = json_dumps(marker, default=str).encode("utf-8")
        if len(encoded) > _MANIFEST_INVENTORY_HEAD_MAX_BYTES:
            raise AssertionError("manifest inventory sparse-successor marker exceeds bound")
        self._json_lock_file_ops.write_bytes_durable(
            self._json_inventory_skip_locator(sequence), encoded
        )
        return marker

    def _append_inventory_event(self, key: str, record: bytes) -> int:
        """Index publication before authority, without append/read/rewrite history."""
        state = self._inventory_state(allow_unacknowledged_tail=True)
        self._acknowledge_unacknowledged_inventory_tail(state)
        record_digest = hashlib.sha256(record).hexdigest()
        if type(self.backend) is not JsonBackend:
            sequence = state["next_sequence"]
            event = self._sign_inventory_value({
                "version": _MANIFEST_INVENTORY_SCHEMA_VERSION,
                "store_id": self._inventory_store_id(),
                "epoch": state["epoch"],
                "sequence": sequence,
                "key": key,
                "digest": record_digest,
            }, initialize_new_store=True)
            self._validate_inventory_event(event, sequence)
            state["events"][sequence] = event
            state["next_sequence"] = sequence + 1
            self._write_inventory_tail(
                state,
                state["next_sequence"],
                terminal_event=(event["key"], event["digest"]),
            )
            self._write_inventory_head(state)
            return sequence
        while True:
            sequence = state["next_sequence"]
            event = self._sign_inventory_value({
                "version": _MANIFEST_INVENTORY_SCHEMA_VERSION,
                "store_id": self._inventory_store_id(),
                "epoch": state["epoch"],
                "sequence": sequence,
                "key": key,
                "digest": record_digest,
            }, initialize_new_store=True)
            encoded = json_dumps(event, default=str).encode("utf-8")
            if len(encoded) > self._inventory_event_max_bytes():
                raise CacheBlobBackendError(
                    "Canonical manifest inventory event exceeds its field policy",
                    context={"operation": "manifest_inventory"},
                )
            self._validate_inventory_event(event, sequence)
            try:
                self._json_lock_file_ops.create_bytes_durable_exclusive(  # type: ignore[union-attr]
                    self._json_inventory_event_locator(sequence), encoded
                )
            except FileExistsError:
                # A crash after event durability but before the bounded head
                # acknowledgement leaves safe stale scheduling evidence.
                existing = self._read_inventory_event(sequence)
                if existing is None:
                    raise CacheBlobBackendError(
                        "Canonical manifest inventory event disappeared",
                        context={"operation": "manifest_inventory", "sequence": sequence},
                    )
                state["next_sequence"] = sequence + 1
                self._write_inventory_tail(
                    state, state["next_sequence"], terminal_event=existing
                )
                self._write_inventory_head(state)
                if existing == (key, record_digest):
                    return sequence
                continue
            state["next_sequence"] = sequence + 1
            self._write_inventory_tail(
                state,
                state["next_sequence"],
                terminal_event=(event["key"], event["digest"]),
            )
            self._write_inventory_head(state)
            return sequence

    @staticmethod
    def _inventory_sequence_from_entry(entry: object) -> int | None:
        """Read a private exact scheduler sequence from one current projection."""
        if not isinstance(entry, Mapping):
            return None
        metadata = entry.get("metadata")
        if not isinstance(metadata, Mapping):
            return None
        sequence = metadata.get(_MANIFEST_INVENTORY_SEQUENCE_FIELD)
        return sequence if type(sequence) is int and sequence > 0 else None

    def _inventory_event_is_current(
        self, entry: object, *, sequence: int, digest: str
    ) -> bytes | None:
        """Return current bytes only when this exact inventory slot still owns them.

        Old projections did not contain the private sequence field.  They are
        deliberately retained as a digest-only compatibility case until the
        bounded maintenance sweep rewrites them.  Once a projection has a
        sequence, digest equality alone is insufficient: replacing a value by
        identical bytes must not make both scheduling positions live.
        """
        current = self._raw_from_entry(entry)
        if current is None or hashlib.sha256(current).hexdigest() != digest:
            return None
        current_sequence = self._inventory_sequence_from_entry(entry)
        if current_sequence is not None and current_sequence != sequence:
            return None
        return current

    @staticmethod
    def _mark_inventory_maintenance_debt(
        state: dict[str, Any], *, sequence: int
    ) -> None:
        """Durably schedule exact stale-event revalidation after authority.

        The charged position is either a replacement candidate or a removed
        predecessor.  Revalidation determines whether it is stale.  Starting
        at the live floor lets the continuation relocate any earlier live
        positions and retire only the exact stale members.  The immutable
        position of a live member is never changed: sparse-successor markers
        let readers jump across retired runs without rewriting cursor order.
        """
        target = state["maintenance_target_sequence"]
        if target == 0:
            state["compact_next_sequence"] = state["first_live_sequence"]
            state["maintenance_target_sequence"] = sequence
            return
        state["compact_next_sequence"] = min(
            state["compact_next_sequence"], sequence
        )
        state["maintenance_target_sequence"] = max(target, sequence)

    def _compact_inventory_window(self) -> bool:
        """Advance one durable bounded post-authority maintenance continuation.

        Live events retain their original publication sequence.  Retiring a
        stale run publishes a bounded sparse-successor marker *before* the
        final event deletion, so a crash can at worst leave a stale event that
        is hidden by a marker; it can never hide a current authority record.
        """
        state = self._inventory_state()
        target = state["maintenance_target_sequence"]
        if target == 0:
            return True
        # ``first_live_sequence == next_sequence`` is a durable proof that
        # every event position is sparse.  Preserve that terminal floor rather
        # than wrapping maintenance back across a lifetime of gaps on reopen.
        if state["first_live_sequence"] == state["next_sequence"]:
            state["compact_next_sequence"] = state["next_sequence"]
            state["maintenance_target_sequence"] = 0
            self._write_inventory_head(state)
            return True
        start = max(state["compact_next_sequence"], state["first_live_sequence"])
        stop = min(
            target + 1,
            start + min(
                _MANIFEST_INVENTORY_COMPACTION_WINDOW,
                self.lifecycle_limits.max_inventory_items,
            ),
        )
        floor_unresolved = False
        first_remaining: int | None = None
        open_sparse_run_start = state["open_sparse_run_start"]
        for sequence in range(start, stop):
            successor = self._read_inventory_skip(sequence, state)
            if successor is not None:
                # A sparse successor is an authenticated proof for this exact
                # retired position.  It is written before the event unlink, so
                # recovery may advance across it without mistaking ordinary
                # control-data loss for compacted history.  Readers make the
                # same direct, bounded lookup before asking for an event.
                if (
                    open_sparse_run_start
                    and sequence >= open_sparse_run_start
                ):
                    open_sparse_run_start = 0
                continue
            event = self._read_inventory_event(sequence)
            if event is None:
                # A deleted event is not an authenticated sparse gap.  Normal
                # compaction writes a signed successor marker before retiring
                # an immutable event, and page readers consume that marker
                # before asking for the event.  Manufacturing a new marker
                # here would turn control-data loss into false-clean recovery.
                raise CacheManifestIntegrityError(
                    "Canonical manifest inventory event is missing",
                    context={"operation": "manifest_inventory", "sequence": sequence},
                )
            key, digest = event
            try:
                current = self._inventory_event_is_current(
                    self._current_entry(key), sequence=sequence, digest=digest
                )
            except (TypeError, ValueError, CacheBlobMigrationRequiredError):
                # A malformed or pre-migration projection is still live
                # authority.  Its immutable scheduling event is the only
                # bounded route by which a later inventory page can expose the
                # typed failure; treating this as absence would let unrelated
                # maintenance manufacture a false terminal reconciliation.
                # Compaction is post-publication best-effort maintenance, so
                # preserve the event and keep progressing through this bounded
                # window instead of failing the unrelated write/remove.
                if sequence >= state["first_live_sequence"]:
                    floor_unresolved = True
                # A prior marker must continue to land on the malformed event
                # rather than jumping past authority whose typed failure still
                # has to be observable by a page reader.
                if (
                    open_sparse_run_start
                    and sequence >= open_sparse_run_start
                ):
                    open_sparse_run_start = 0
                continue
            if current is not None:
                if first_remaining is None:
                    first_remaining = sequence
                # Do not close an open run merely because this maintenance
                # pass wrapped to an older live member.  Only its successor
                # ends the run, which preserves a marker that begins later.
                if (
                    open_sparse_run_start
                    and sequence >= open_sparse_run_start
                ):
                    open_sparse_run_start = 0
                continue
            if open_sparse_run_start == 0 or sequence < open_sparse_run_start:
                open_sparse_run_start = sequence
            # The marker becomes durable before the matching event is
            # removed.  Since this lock serializes authority and the exact
            # event comparison above proved it stale, a crash at either side
            # can only retain or hide stale scheduling evidence.
            marker = self._write_inventory_skip(
                open_sparse_run_start, sequence + 1, state, event=event
            )
            tail = self._read_inventory_tail(state)
            if (
                tail is not None
                and marker["start_sequence"]
                <= tail["terminal_sequence"]
                < marker["next_sequence"]
            ):
                self._write_inventory_tail_witness(state, tail, marker)
            if type(self.backend) is JsonBackend:
                try:
                    self._json_lock_file_ops.delete_durable(  # type: ignore[union-attr]
                        self._json_inventory_event_locator(sequence)
                    )
                except FileNotFoundError:
                    pass
            else:
                state["events"].pop(sequence, None)
        # Compaction can wrap and encounter the existing floor in a later
        # window.  Advancing only when the window begins at that floor strands
        # lifetime history after the floor's live record is retired.  This
        # branch uses only exact revalidation results from the inspected
        # window, and never lowers the durable monotonic floor.
        updated_state = self._inventory_state()
        if (
            start <= updated_state["first_live_sequence"] < stop
            and not floor_unresolved
        ):
            updated_state["first_live_sequence"] = (
                first_remaining if first_remaining is not None else stop
            )
        updated_state["compact_next_sequence"] = stop
        updated_state["open_sparse_run_start"] = open_sparse_run_start
        # ``stop`` is exclusive.  Debt is complete only after this call has
        # exact-revalidated the target position itself; a floor that merely
        # reaches the target still leaves that target for the next bounded
        # call when a prior window ended immediately before it.
        complete = stop > target
        if complete:
            updated_state["maintenance_target_sequence"] = 0
        self._write_inventory_head(updated_state)
        return complete

    def compact_inventory_for_recovery(self) -> bool:
        """Consume bounded post-authority compaction debt under authority locks.

        Manifest pages remain read-only.  Lifecycle startup and clear-snapshot
        admission are the explicit mutating opportunities that converge a
        process loss after authority publication but before compaction.
        """
        try:
            with self.backend._lock, self._json_compare_publish_lock():
                if type(self.backend) is JsonBackend:
                    try:
                        self._refresh_json_for_conditional_operation()
                    except (CacheError, OSError, TypeError, ValueError):
                        # A malformed, migration-required, or unreadable
                        # authority document pins compaction debt.  Recovery
                        # must not rewrite/index that unknown authority, and
                        # normal public operations retain their established
                        # typed refresh failure at their own admission point.
                        return False
                return self._compact_inventory_window()
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("compact_inventory", self.backend, exc) from exc

    def _publish_projection(
        self, key: str, record: bytes, entry_data: Optional[Mapping[str, Any]]
    ) -> None:
        """Publish one reversible projection inside the established lock boundary."""
        previous_sequence = self._inventory_sequence_from_entry(self._current_entry(key))
        projection = self._projection(key, record, entry_data)
        # The scheduler event is durable before authority publication. A
        # process loss therefore creates only a stale revalidated position;
        # it can never publish authority that a high-water snapshot omitted.
        sequence = self._append_inventory_event(key, record)
        projection["metadata"][_MANIFEST_INVENTORY_SEQUENCE_FIELD] = sequence
        if previous_sequence is not None:
            # Record the exact predecessor *before* authority moves.  A crash
            # at either side of publication now has the same bounded
            # continuation: revalidation retains the predecessor if the
            # candidate did not become current, or retires it if it did.
            state = self._inventory_state()
            # Charge the candidate as well as its predecessor.  If a process
            # dies before authority publication, that candidate is now an
            # exact stale position the continuation will retire; if authority
            # succeeds, the same sweep relocates the current candidate beyond
            # the compacted prefix.
            self._mark_inventory_maintenance_debt(state, sequence=sequence)
            self._write_inventory_head(state)
        if type(self.backend) is JsonBackend:
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
            self._publish_json_document(candidate)
            self._compact_inventory_window()
            return
        self.backend.put_entry(key, projection)
        self._compact_inventory_window()

    def _remove_projection(self, key: str) -> None:
        """Retire the raw and compatibility projections in one local boundary."""
        previous_sequence = self._inventory_sequence_from_entry(self._current_entry(key))
        if previous_sequence is not None:
            # As above, schedule before the authority boundary so a crash
            # cannot leave a removed projection's event permanently outside
            # bounded maintenance.
            state = self._inventory_state()
            self._mark_inventory_maintenance_debt(state, sequence=previous_sequence)
            self._write_inventory_head(state)
        if type(self.backend) is JsonBackend:
            candidate = deepcopy(self.backend._metadata)
            candidate.get("entries", {}).pop(key, None)
            self._publish_json_document(candidate)
            self._compact_inventory_window()
            return
        self.backend._entries.pop(key, None)
        self._compact_inventory_window()

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
                high_water = (
                    inventory["next_sequence"] - 1
                    if cursor is None or cursor.snapshot_high_water is None
                    else cursor.snapshot_high_water
                )
                requested_position = (
                    1
                    if cursor is None or cursor.next_sequence is None
                    else cursor.next_sequence
                )
                position = max(requested_position, inventory["first_live_sequence"])
                inspected = 0
                page_entries: list[tuple[str, bytes]] = []
                entry_next_cursors: list[ManifestCursor] = []
                last_key = cursor.key if cursor is not None else "~"
                # ``max_inventory_items`` is now a per-call inspected-name
                # budget.  It is never a total-store eligibility ceiling.
                while (
                    position <= high_water
                    and inspected < self.lifecycle_limits.max_inventory_items
                    and len(page_entries) < limit
                ):
                    successor = self._read_inventory_skip(position, inventory)
                    if successor is not None:
                        # Markers are scheduling-only and can never move an
                        # old finite snapshot forward.  A successor beyond an
                        # old high-water simply proves this cursor terminal;
                        # the successor itself is still exact-revalidated on
                        # a later current snapshot before yielding anything.
                        position = min(successor, high_water + 1)
                        continue
                    inspected += 1
                    event = self._read_inventory_event(position)
                    position += 1
                    if event is None:
                        raise CacheManifestIntegrityError(
                            "Canonical manifest inventory event is missing",
                            context={
                                "operation": "manifest_inventory",
                                "sequence": position - 1,
                            },
                        )
                    event_key, event_digest = event
                    last_key = event_key
                    entry = entries.get(event_key)
                    # ``None`` is a proven absent/stale scheduling member.
                    # A malformed current authority projection is different:
                    # treating it as a miss would let reconciliation report a
                    # false-clean terminal inventory.  Let the typed failure
                    # cross the repository boundary and fail closed instead.
                    current_raw = self._inventory_event_is_current(
                        entry, sequence=position - 1, digest=event_digest
                    )
                    if current_raw is not None:
                        page_entries.append((event_key, current_raw))
                        entry_next_cursors.append(
                            ManifestCursor(
                                event_key,
                                snapshot_high_water=high_water,
                                next_sequence=position,
                            )
                        )
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
            entry_next_cursors=tuple(entry_next_cursors),
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
        inventory_key_provider: Callable[[], bytes] | None = None,
    ):
        super().__init__(
            backend,
            lifecycle_limits=lifecycle_limits,
            file_ops=file_ops,
            inventory_key_provider=inventory_key_provider,
        )


class JsonManifestRepository(_MetadataManifestRepository):
    """Exact raw records persisted by JsonBackend's durable document protocol."""

    def __init__(
        self,
        backend: JsonBackend,
        *,
        lifecycle_limits: LifecycleLimits | None = None,
        file_ops: ManagedFileOps | None = None,
        inventory_key_provider: Callable[[], bytes] | None = None,
    ):
        super().__init__(
            backend,
            lifecycle_limits=lifecycle_limits,
            file_ops=file_ops,
            inventory_key_provider=inventory_key_provider,
        )


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
                    CREATE TABLE IF NOT EXISTS {_SQLITE_MANIFEST_INVENTORY_STATE_TABLE} (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        compact_next_sequence INTEGER NOT NULL
                    )
                    """
                )
                connection.exec_driver_sql(
                    f"INSERT OR IGNORE INTO {_SQLITE_MANIFEST_INVENTORY_STATE_TABLE} "
                    "(singleton, compact_next_sequence) VALUES (1, 1)"
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

    def compact_inventory_for_recovery(self) -> bool:
        """Consume one bounded stale-inventory window in a SQLite transaction."""
        try:
            with self.backend._lock, self.backend.engine.begin() as connection:
                self._compact_inventory_window(connection)
                return True
        except _BACKEND_OPERATION_ERRORS as exc:
            raise _backend_failure("compact_inventory", self.backend, exc) from exc

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
                self._compact_inventory_window(connection)
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
                self._compact_inventory_window(connection)
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

    @staticmethod
    def _compact_inventory_window(connection: Any) -> None:
        """Drop only stale exact events while preserving sparse sequence cursors."""
        state = connection.exec_driver_sql(
            f"SELECT compact_next_sequence FROM {_SQLITE_MANIFEST_INVENTORY_STATE_TABLE} "
            "WHERE singleton = 1"
        ).first()
        if state is None:
            raise CacheBlobBackendError(
                "SQLite manifest inventory compaction state is unavailable",
                context={"operation": "manifest_inventory"},
            )
        high_water = int(
            connection.exec_driver_sql(
                f"SELECT COALESCE(MAX(sequence), 0) FROM {_SQLITE_MANIFEST_INVENTORY_TABLE}"
            ).scalar_one()
        )
        if high_water == 0:
            return
        start = int(state[0])
        if start > high_water:
            start = 1
        stop = min(high_water + 1, start + _MANIFEST_INVENTORY_COMPACTION_WINDOW)
        rows = connection.exec_driver_sql(
            f"SELECT inventory.sequence, inventory.logical_key, inventory.record_digest, "
            "manifests.canonical_bytes "
            f"FROM {_SQLITE_MANIFEST_INVENTORY_TABLE} AS inventory "
            f"LEFT JOIN {_SQLITE_MANIFEST_TABLE} AS manifests "
            "ON manifests.logical_key = inventory.logical_key "
            "WHERE inventory.sequence >= ? AND inventory.sequence < ? "
            "ORDER BY inventory.sequence",
            (start, stop),
        ).all()
        for sequence, _key, digest, raw in rows:
            if raw is None or hashlib.sha256(bytes(raw)).hexdigest() != str(digest):
                connection.exec_driver_sql(
                    f"DELETE FROM {_SQLITE_MANIFEST_INVENTORY_TABLE} WHERE sequence = ?",
                    (int(sequence),),
                )
        next_sequence = stop if stop <= high_water else 1
        connection.exec_driver_sql(
            f"UPDATE {_SQLITE_MANIFEST_INVENTORY_STATE_TABLE} "
            "SET compact_next_sequence = ? WHERE singleton = 1",
            (next_sequence,),
        )

    def remove(self, key: str) -> None:
        """Remove the raw manifest and compatible BlobStore metadata if present."""
        try:
            with self.backend._lock, self.backend.engine.begin() as connection:
                connection.exec_driver_sql(
                    f"DELETE FROM {_SQLITE_MANIFEST_TABLE} WHERE logical_key = ?",
                    (key,),
                )
                self._compact_inventory_window(connection)
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
                self._compact_inventory_window(connection)
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
        entry_next_cursors: list[ManifestCursor] = []
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
                    entry_next_cursors.append(
                        ManifestCursor(
                            key,
                            snapshot_high_water=int(high_water),
                            next_sequence=next_position,
                        )
                    )
                    if len(entries_list) == resolved:
                        break
        entries = tuple(entries_list)
        # Compaction leaves deliberate sparse gaps. If the bounded query found
        # no later event rows, the high-water snapshot is complete even when
        # its numeric sequence still extends beyond the caller's old cursor.
        snapshot_exhausted = not rows or (
            len(rows) < self.lifecycle_limits.max_inventory_items
            and len(entries_list) < resolved
        )
        return ManifestPage(
            entries=entries,
            next_cursor=(
                ManifestCursor(
                    last_key,
                    snapshot_high_water=int(high_water),
                    next_sequence=next_position,
                )
                if next_position <= int(high_water) and not snapshot_exhausted
                else None
            ),
            entry_next_cursors=tuple(entry_next_cursors),
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
    inventory_key_provider: Callable[[], bytes] | None = None,
) -> ManifestRepository:
    """Select the exact local adapter that can preserve canonical record bytes."""
    if type(backend) is InMemoryBackend:
        return InMemoryManifestRepository(
            backend,
            lifecycle_limits=lifecycle_limits,
            file_ops=file_ops,
            inventory_key_provider=inventory_key_provider,
        )
    if type(backend) is JsonBackend:
        return JsonManifestRepository(
            backend,
            lifecycle_limits=lifecycle_limits,
            file_ops=file_ops,
            inventory_key_provider=inventory_key_provider,
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
        inventory_key_provider: Callable[[], bytes] | None = None,
    ):
        self._repository = create_manifest_repository(
            backend,
            lifecycle_limits=lifecycle_limits,
            file_ops=file_ops,
            inventory_key_provider=inventory_key_provider,
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

    def compact_inventory_for_recovery(self) -> bool:
        return self._repository.compact_inventory_for_recovery()

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
    "JsonProjectionExporter",
    "InMemoryManifestRepository",
    "JsonManifestRepository",
    "SqliteManifestRepository",
    "MetadataManifestRepository",
    "create_manifest_repository",
]
