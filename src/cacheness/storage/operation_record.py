"""Bounded authenticated evidence for one BlobStore lifecycle mutation.

Operation records are control metadata, never an alternate payload format or
normal-read authority. They are stricter than manifests because recovery may
use them to authorize cleanup after a process interruption.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import PurePath
from types import MappingProxyType
from typing import Any, Mapping

from cacheness.error_handling import (
    CacheManifestIntegrityError,
    CacheManifestUnsupportedVersionError,
    CacheReason,
)


OPERATION_RECORD_SCHEMA_VERSION = 2
OPERATION_RECORD_OWNER = "cacheness.blob-store.lifecycle"
OPERATION_SIGNATURE_ALGORITHM = "hmac-sha256"
OPERATION_SIGNING_DOMAIN = b"cacheness.operation-record.v2\x00"
CLEAR_TARGET_PAGE_SIGNING_DOMAIN = b"cacheness.clear-target-page.v1\x00"
CLEAR_TARGET_CHECKPOINT_SIGNING_DOMAIN = (
    b"cacheness.clear-target-checkpoint.v1\x00"
)
MAX_OPERATION_RECORD_BYTES = 1_048_576
MAX_OPERATION_FIELD_BYTES = 8_192
MAX_OPERATION_TOPOLOGY_FIELDS = 8
MAX_OPERATION_NESTING_DEPTH = 4
MAX_OPERATION_NODES = 64
MIN_SIGNED_64 = -(2**63)
MAX_SIGNED_64 = 2**63 - 1
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_HEX_UUID = re.compile(r"[0-9a-f]{32}")


class OperationKind(str, Enum):
    """Lifecycle mutations represented by durable operation evidence."""

    PUT = "put"
    DELETE = "delete"
    CLEAR = "clear"


class OperationTransition(str, Enum):
    """The intended authority transition encoded before managed side effects."""

    CREATE = "create"
    REPLACE = "replace"
    TOMBSTONE = "tombstone"
    CLEAR = "clear"


class OperationCheckpoint(str, Enum):
    """Monotonic checkpoints that classify authority and cleanup progress."""

    PREPARED = "prepared"
    CANDIDATE_PUBLISHED = "candidate_published"
    AUTHORITY_PUBLISHED = "authority_published"
    RECLAIMING = "reclaiming"
    TERMINAL = "terminal"
    CLEANUP_COMPLETED = "terminal"


_CHECKPOINT_TRANSITIONS = {
    OperationCheckpoint.PREPARED: frozenset({OperationCheckpoint.CANDIDATE_PUBLISHED}),
    OperationCheckpoint.CANDIDATE_PUBLISHED: frozenset(
        {OperationCheckpoint.AUTHORITY_PUBLISHED}
    ),
    OperationCheckpoint.AUTHORITY_PUBLISHED: frozenset({OperationCheckpoint.RECLAIMING}),
    OperationCheckpoint.RECLAIMING: frozenset({OperationCheckpoint.TERMINAL}),
    OperationCheckpoint.TERMINAL: frozenset(),
}
_RECORD_FIELDS = frozenset(
    {
        "candidate_locator",
        "checkpoint",
        "created_at",
        "expected_generation",
        "expected_record_digest",
        "generation",
        "key",
        "kind",
        "operation_id",
        "owner",
        "previous_locator",
        "schema_version",
        "signature",
        "signature_algorithm",
        "store_id",
        "topology",
        "transition",
        "updated_at",
    }
)
_CLEAR_TARGET_PAGE_FIELDS = frozenset(
    {
        "next_cursor",
        "operation_id",
        "page_id",
        "schema_version",
        "signature",
        "signature_algorithm",
        "source_cursor",
        "targets",
    }
)
_CLEAR_TARGET_CHECKPOINT_FIELDS = frozenset(
    {
        "completed_target_indices",
        "operation_id",
        "page_complete",
        "page_id",
        "page_record_digest",
        "schema_version",
        "signature",
        "signature_algorithm",
    }
)
CLEAR_TARGET_SCHEMA_VERSION = 1


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON mapping while rejecting ambiguous duplicate keys."""
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise CacheManifestIntegrityError("Duplicate operation record key")
        record[key] = value
    return record


def _reject_float(_value: str) -> None:
    """Reject floats, including non-finite values, from control evidence."""
    raise CacheManifestIntegrityError("Operation record values cannot be floating point")


def _reject_constant(_value: str) -> None:
    """Reject JSON's non-standard NaN and infinity constants."""
    raise CacheManifestIntegrityError("Operation record values cannot be non-finite")


def _parse_signed_64(value: str) -> int:
    """Parse an integer without accepting an unbounded Python integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise CacheManifestIntegrityError("Operation record integer is invalid") from exc
    if not MIN_SIGNED_64 <= parsed <= MAX_SIGNED_64:
        raise CacheManifestIntegrityError(
            "Operation record integer exceeds the signed-64 range",
            reason=CacheReason.MANIFEST_BOUNDS,
        )
    return parsed


def _bounded_string(
    value: object,
    field: str,
    *,
    allow_none: bool = False,
    allow_empty: bool = False,
) -> str | None:
    """Validate one bounded string field before semantic interpretation."""
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or (not value and not allow_empty):
        raise CacheManifestIntegrityError(f"Operation record {field} must be non-empty")
    if len(value.encode("utf-8")) > MAX_OPERATION_FIELD_BYTES:
        raise CacheManifestIntegrityError(
            f"Operation record {field} exceeds the byte limit",
            reason=CacheReason.MANIFEST_BOUNDS,
        )
    return value


def _validate_json_value(value: Any, *, depth: int, nodes: list[int]) -> None:
    """Reject unbounded or non-canonical nested control values before use."""
    if depth > MAX_OPERATION_NESTING_DEPTH:
        raise CacheManifestIntegrityError(
            "Operation record nesting limit exceeded", reason=CacheReason.MANIFEST_BOUNDS
        )
    nodes[0] += 1
    if nodes[0] > MAX_OPERATION_NODES:
        raise CacheManifestIntegrityError(
            "Operation record total nodes limit exceeded", reason=CacheReason.MANIFEST_BOUNDS
        )
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if not MIN_SIGNED_64 <= value <= MAX_SIGNED_64:
            raise CacheManifestIntegrityError(
                "Operation record integer exceeds the signed-64 range",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        return
    if isinstance(value, str):
        _bounded_string(value, "value")
        return
    if isinstance(value, Mapping):
        if len(value) > len(_RECORD_FIELDS):
            raise CacheManifestIntegrityError(
                "Operation record collection field limit exceeded",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        for key, item in value.items():
            if not isinstance(key, str):
                raise CacheManifestIntegrityError("Operation record topology keys must be strings")
            _validate_json_value(key, depth=depth + 1, nodes=nodes)
            _validate_json_value(item, depth=depth + 1, nodes=nodes)
        return
    raise CacheManifestIntegrityError("Operation record is not JSON-compatible")


def _canonical_bytes(record: Mapping[str, Any]) -> bytes:
    """Encode the exact bounded evidence projection deterministically."""
    _validate_json_value(record, depth=1, nodes=[0])
    try:
        encoded = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CacheManifestIntegrityError("Operation record is not JSON-compatible") from exc
    if len(encoded) > MAX_OPERATION_RECORD_BYTES:
        raise CacheManifestIntegrityError(
            "Operation record exceeds the byte limit",
            reason=CacheReason.MANIFEST_BOUNDS,
        )
    return encoded


def _validated_locator(value: object, field: str, *, allow_none: bool) -> str | None:
    """Require a relative, traversal-free locator before recovery can resolve it."""
    locator = _bounded_string(value, field, allow_none=allow_none)
    if locator is None:
        return None
    path = PurePath(locator)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise CacheManifestIntegrityError(f"Operation record {field} is not contained")
    return locator


def _validated_hex(
    value: object,
    field: str,
    *,
    allow_none: bool,
    pattern: re.Pattern[str],
) -> str | None:
    """Require an exact opaque identifier, never a permissive filename token."""
    identifier = _bounded_string(value, field, allow_none=allow_none)
    if identifier is None:
        return None
    if not pattern.fullmatch(identifier):
        raise CacheManifestIntegrityError(f"Operation record {field} is invalid")
    return identifier


def _validated_topology(value: object) -> Mapping[str, str]:
    """Freeze a small exact topology map that binds evidence to one store shape."""
    if (
        not isinstance(value, Mapping)
        or len(value) > MAX_OPERATION_TOPOLOGY_FIELDS
        or set(value) != {"backend", "root"}
    ):
        raise CacheManifestIntegrityError("Operation record topology is invalid")
    _validate_json_value(value, depth=2, nodes=[0])
    backend = _bounded_string(value["backend"], "topology.backend")
    root = _validated_hex(value["root"], "topology.root", allow_none=False, pattern=_HEX_SHA256)
    assert backend is not None and root is not None
    return MappingProxyType({"backend": backend, "root": root})


def _validate_timestamp(value: object, field: str) -> str:
    """Require one bounded timezone-aware ISO timestamp for durable evidence."""
    timestamp = _bounded_string(value, field)
    assert timestamp is not None
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise CacheManifestIntegrityError(f"Operation record {field} is invalid") from exc
    if parsed.tzinfo is None:
        raise CacheManifestIntegrityError(f"Operation record {field} requires a timezone")
    return timestamp


def _canonical_clear_bytes(record: Mapping[str, Any]) -> bytes:
    """Encode bounded clear control evidence without reusing payload codecs."""
    try:
        encoded = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CacheManifestIntegrityError("Clear control evidence is not JSON-compatible") from exc
    if len(encoded) > MAX_OPERATION_RECORD_BYTES:
        raise CacheManifestIntegrityError(
            "Clear control evidence exceeds the byte limit",
            reason=CacheReason.MANIFEST_BOUNDS,
        )
    return encoded


def _decode_clear_mapping(raw: bytes, fields: frozenset[str], label: str) -> dict[str, Any]:
    """Decode one strict clear evidence projection without assigning authority."""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_OPERATION_RECORD_BYTES:
        raise CacheManifestIntegrityError(f"{label} bytes are invalid")
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_int=_parse_signed_64,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, CacheManifestIntegrityError):
            raise
        raise CacheManifestIntegrityError(f"{label} is not valid JSON") from exc
    if not isinstance(decoded, dict) or set(decoded) != fields:
        raise CacheManifestIntegrityError(f"{label} has an unknown or missing field")
    return decoded


def _validated_clear_cursor(value: object, field: str) -> str | None:
    """Accept bounded opaque logical cursors without treating them as paths."""
    return _bounded_string(value, field, allow_none=True)


@dataclass(frozen=True)
class ClearTarget:
    """One authenticated-manifest observation preserved for a clear page."""

    key: str
    generation: str
    record_digest: str
    raw_record: bytes

    def __post_init__(self) -> None:
        """Bind the persisted target to its exact opaque manifest bytes."""
        _bounded_string(self.key, "clear_target.key")
        _validated_hex(self.generation, "clear_target.generation", allow_none=False, pattern=_HEX_UUID)
        _validated_hex(
            self.record_digest,
            "clear_target.record_digest",
            allow_none=False,
            pattern=_HEX_SHA256,
        )
        if not isinstance(self.raw_record, bytes) or not self.raw_record:
            raise CacheManifestIntegrityError("Clear target record must be non-empty bytes")
        if len(self.raw_record) > MAX_OPERATION_RECORD_BYTES:
            raise CacheManifestIntegrityError(
                "Clear target record exceeds the byte limit",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        if hashlib.sha256(self.raw_record).hexdigest() != self.record_digest:
            raise CacheManifestIntegrityError("Clear target digest does not match exact record")

    @classmethod
    def from_raw(cls, key: str, generation: str, raw_record: bytes) -> "ClearTarget":
        """Construct one exact target from the observed canonical manifest bytes."""
        return cls(
            key=key,
            generation=generation,
            record_digest=hashlib.sha256(raw_record).hexdigest(),
            raw_record=raw_record,
        )

    def to_mapping(self) -> dict[str, str]:
        """Return a reversible JSON-safe representation of one exact target."""
        return {
            "generation": self.generation,
            "key": self.key,
            "raw_record": base64.b64encode(self.raw_record).decode("ascii"),
            "record_digest": self.record_digest,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "ClearTarget":
        """Decode one strict target without authenticating it as authority."""
        if not isinstance(value, Mapping) or set(value) != {
            "generation",
            "key",
            "raw_record",
            "record_digest",
        }:
            raise CacheManifestIntegrityError("Clear target fields are invalid")
        encoded = value["raw_record"]
        if not isinstance(encoded, str):
            raise CacheManifestIntegrityError("Clear target raw record is invalid")
        try:
            raw_record = base64.b64decode(encoded, validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise CacheManifestIntegrityError("Clear target raw record is invalid") from exc
        return cls(
            key=value["key"],
            generation=value["generation"],
            record_digest=value["record_digest"],
            raw_record=raw_record,
        )


@dataclass(frozen=True)
class ClearTargetPage:
    """One bounded durable page of exact clear targets.

    Repository code stores this as opaque bytes. Lifecycle code authenticates
    the page before treating any target as destructive authority.
    """

    operation_id: str
    page_id: str
    source_cursor: str | None
    next_cursor: str | None
    targets: tuple[ClearTarget, ...]
    signature_algorithm: str = OPERATION_SIGNATURE_ALGORITHM
    signature: str = ""
    schema_version: int = CLEAR_TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject ambiguous, unbounded, or malformed page values."""
        if self.schema_version != CLEAR_TARGET_SCHEMA_VERSION:
            raise CacheManifestUnsupportedVersionError(
                f"Unsupported clear target page schema version: {self.schema_version}"
            )
        _validated_hex(self.operation_id, "clear_target_page.operation_id", allow_none=False, pattern=_HEX_UUID)
        _validated_hex(self.page_id, "clear_target_page.page_id", allow_none=False, pattern=_HEX_UUID)
        _validated_clear_cursor(self.source_cursor, "clear_target_page.source_cursor")
        _validated_clear_cursor(self.next_cursor, "clear_target_page.next_cursor")
        if not isinstance(self.targets, tuple) or not self.targets:
            raise CacheManifestIntegrityError("Clear target page must contain targets")
        if len(self.targets) > 4_096 or any(not isinstance(target, ClearTarget) for target in self.targets):
            raise CacheManifestIntegrityError("Clear target page targets are invalid")
        keys = [target.key for target in self.targets]
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise CacheManifestIntegrityError("Clear target page keys must be unique and ordered")
        if self.signature_algorithm != OPERATION_SIGNATURE_ALGORITHM:
            raise CacheManifestIntegrityError("Unsupported clear target page signature algorithm")
        _bounded_string(self.signature, "clear_target_page.signature", allow_empty=True)
        if self.signature and not _HEX_SHA256.fullmatch(self.signature):
            raise CacheManifestIntegrityError("Clear target page signature is invalid")

    def to_mapping(self, *, include_signature: bool = True) -> dict[str, Any]:
        """Return the deterministic bounded target-page projection."""
        result = {
            "next_cursor": self.next_cursor,
            "operation_id": self.operation_id,
            "page_id": self.page_id,
            "schema_version": self.schema_version,
            "signature_algorithm": self.signature_algorithm,
            "source_cursor": self.source_cursor,
            "targets": [target.to_mapping() for target in self.targets],
        }
        if include_signature:
            result["signature"] = self.signature
        return result

    def canonical_bytes(self, *, include_signature: bool = True) -> bytes:
        """Encode exact target evidence for conditional durable persistence."""
        return _canonical_clear_bytes(self.to_mapping(include_signature=include_signature))

    def signing_bytes(self) -> bytes:
        """Domain-separate page authentication from manifests and operations."""
        return CLEAR_TARGET_PAGE_SIGNING_DOMAIN + self.canonical_bytes(include_signature=False)

    def with_signature(self, signature: str) -> "ClearTargetPage":
        """Return a signed page without mutating its immutable target set."""
        return replace(self, signature=signature)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "ClearTargetPage":
        """Decode exact page bytes without assigning them lifecycle authority."""
        decoded = _decode_clear_mapping(raw, _CLEAR_TARGET_PAGE_FIELDS, "Clear target page")
        targets = decoded["targets"]
        if not isinstance(targets, list):
            raise CacheManifestIntegrityError("Clear target page targets are invalid")
        return cls(
            operation_id=decoded["operation_id"],
            page_id=decoded["page_id"],
            source_cursor=decoded["source_cursor"],
            next_cursor=decoded["next_cursor"],
            targets=tuple(ClearTarget.from_mapping(target) for target in targets),
            signature_algorithm=decoded["signature_algorithm"],
            signature=decoded["signature"],
            schema_version=decoded["schema_version"],
        )


@dataclass(frozen=True)
class ClearTargetCheckpoint:
    """Monotonic, exact-page-bound progress for one persisted target page."""

    operation_id: str
    page_id: str
    page_record_digest: str
    completed_target_indices: tuple[int, ...]
    page_complete: bool
    signature_algorithm: str = OPERATION_SIGNATURE_ALGORITHM
    signature: str = ""
    schema_version: int = CLEAR_TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Require canonical monotonic index sets before durable replacement."""
        if self.schema_version != CLEAR_TARGET_SCHEMA_VERSION:
            raise CacheManifestUnsupportedVersionError(
                f"Unsupported clear target checkpoint schema version: {self.schema_version}"
            )
        _validated_hex(self.operation_id, "clear_target_checkpoint.operation_id", allow_none=False, pattern=_HEX_UUID)
        _validated_hex(self.page_id, "clear_target_checkpoint.page_id", allow_none=False, pattern=_HEX_UUID)
        _validated_hex(
            self.page_record_digest,
            "clear_target_checkpoint.page_record_digest",
            allow_none=False,
            pattern=_HEX_SHA256,
        )
        if (
            not isinstance(self.completed_target_indices, tuple)
            or any(type(index) is not int or index < 0 for index in self.completed_target_indices)
            or tuple(sorted(set(self.completed_target_indices))) != self.completed_target_indices
        ):
            raise CacheManifestIntegrityError("Clear target checkpoint indices are invalid")
        if type(self.page_complete) is not bool:
            raise CacheManifestIntegrityError("Clear target checkpoint completion is invalid")
        if self.signature_algorithm != OPERATION_SIGNATURE_ALGORITHM:
            raise CacheManifestIntegrityError("Unsupported clear target checkpoint signature algorithm")
        _bounded_string(self.signature, "clear_target_checkpoint.signature", allow_empty=True)
        if self.signature and not _HEX_SHA256.fullmatch(self.signature):
            raise CacheManifestIntegrityError("Clear target checkpoint signature is invalid")

    @classmethod
    def initial_for(cls, page: ClearTargetPage) -> "ClearTargetCheckpoint":
        """Bind zero progress to one exact persisted target page."""
        return cls(
            operation_id=page.operation_id,
            page_id=page.page_id,
            page_record_digest=hashlib.sha256(page.canonical_bytes()).hexdigest(),
            completed_target_indices=(),
            page_complete=False,
        )

    def with_completed_target(self, index: int) -> "ClearTargetCheckpoint":
        """Advance one target without allowing a writer to drop prior progress."""
        if type(index) is not int or index < 0:
            raise ValueError("clear target index must be a non-negative integer")
        if self.page_complete:
            raise CacheManifestIntegrityError("Clear target page is already complete")
        return replace(
            self,
            completed_target_indices=tuple(
                sorted(set(self.completed_target_indices).union({index}))
            ),
            signature="",
        )

    def complete_page(self, target_count: int) -> "ClearTargetCheckpoint":
        """Mark a page complete only after every bounded target is checkpointed."""
        if type(target_count) is not int or target_count <= 0:
            raise ValueError("clear target count must be a positive integer")
        if self.completed_target_indices != tuple(range(target_count)):
            raise CacheManifestIntegrityError("Clear target page completion skipped targets")
        return replace(self, page_complete=True, signature="")

    def to_mapping(self, *, include_signature: bool = True) -> dict[str, Any]:
        """Return the exact checkpoint projection for conditional persistence."""
        result = {
            "completed_target_indices": list(self.completed_target_indices),
            "operation_id": self.operation_id,
            "page_complete": self.page_complete,
            "page_id": self.page_id,
            "page_record_digest": self.page_record_digest,
            "schema_version": self.schema_version,
            "signature_algorithm": self.signature_algorithm,
        }
        if include_signature:
            result["signature"] = self.signature
        return result

    def canonical_bytes(self, *, include_signature: bool = True) -> bytes:
        """Encode immutable checkpoint bytes for exact replacement."""
        return _canonical_clear_bytes(self.to_mapping(include_signature=include_signature))

    def signing_bytes(self) -> bytes:
        """Domain-separate checkpoint authentication from all other records."""
        return (
            CLEAR_TARGET_CHECKPOINT_SIGNING_DOMAIN
            + self.canonical_bytes(include_signature=False)
        )

    def with_signature(self, signature: str) -> "ClearTargetCheckpoint":
        """Return a signed checkpoint without mutating progress."""
        return replace(self, signature=signature)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "ClearTargetCheckpoint":
        """Decode a strict checkpoint without treating it as authority."""
        decoded = _decode_clear_mapping(
            raw, _CLEAR_TARGET_CHECKPOINT_FIELDS, "Clear target checkpoint"
        )
        indices = decoded["completed_target_indices"]
        if not isinstance(indices, list):
            raise CacheManifestIntegrityError("Clear target checkpoint indices are invalid")
        return cls(
            operation_id=decoded["operation_id"],
            page_id=decoded["page_id"],
            page_record_digest=decoded["page_record_digest"],
            completed_target_indices=tuple(indices),
            page_complete=decoded["page_complete"],
            signature_algorithm=decoded["signature_algorithm"],
            signature=decoded["signature"],
            schema_version=decoded["schema_version"],
        )


@dataclass(frozen=True)
class LifecycleOperationRecord:
    """The complete authenticated description of one lifecycle mutation.

    The record holds only authority and provenance metadata. It deliberately
    excludes payload bytes, handler data, and deserialization-derived fields.
    """

    schema_version: int
    operation_id: str
    kind: OperationKind
    key: str
    owner: str
    store_id: str
    topology: Mapping[str, str]
    expected_generation: str | None
    expected_record_digest: str | None
    generation: str
    candidate_locator: str
    previous_locator: str | None
    transition: OperationTransition
    checkpoint: OperationCheckpoint
    created_at: str
    updated_at: str
    signature_algorithm: str = OPERATION_SIGNATURE_ALGORITHM
    signature: str = ""

    def __post_init__(self) -> None:
        """Reject malformed, incomplete, or unsafe evidence before persistence."""
        if type(self.schema_version) is not int:
            raise CacheManifestIntegrityError("Operation record schema version must be an integer")
        if self.schema_version != OPERATION_RECORD_SCHEMA_VERSION:
            raise CacheManifestUnsupportedVersionError(
                f"Unsupported operation record schema version: {self.schema_version}"
            )
        if not isinstance(self.kind, OperationKind):
            raise CacheManifestIntegrityError("Operation record kind is invalid")
        if not isinstance(self.transition, OperationTransition):
            raise CacheManifestIntegrityError("Operation record transition is invalid")
        if not isinstance(self.checkpoint, OperationCheckpoint):
            raise CacheManifestIntegrityError("Operation record checkpoint is invalid")
        if self.owner != OPERATION_RECORD_OWNER:
            raise CacheManifestIntegrityError("Operation record owner is invalid")
        _validated_hex(self.operation_id, "operation_id", allow_none=False, pattern=_HEX_UUID)
        _bounded_string(self.key, "key")
        _validated_hex(self.store_id, "store_id", allow_none=False, pattern=_HEX_SHA256)
        topology = _validated_topology(self.topology)
        if topology["root"] != self.store_id:
            raise CacheManifestIntegrityError("Operation record topology does not match store")
        expected_generation = _validated_hex(
            self.expected_generation,
            "expected_generation",
            allow_none=True,
            pattern=_HEX_UUID,
        )
        expected_digest = _validated_hex(
            self.expected_record_digest,
            "expected_record_digest",
            allow_none=True,
            pattern=_HEX_SHA256,
        )
        if (expected_generation is None) != (expected_digest is None):
            raise CacheManifestIntegrityError(
                "Operation record expectation requires generation and exact digest together"
            )
        _validated_hex(self.generation, "generation", allow_none=False, pattern=_HEX_UUID)
        _validated_locator(self.candidate_locator, "candidate_locator", allow_none=False)
        _validated_locator(self.previous_locator, "previous_locator", allow_none=True)
        created = _validate_timestamp(self.created_at, "created_at")
        updated = _validate_timestamp(self.updated_at, "updated_at")
        if datetime.fromisoformat(updated) < datetime.fromisoformat(created):
            raise CacheManifestIntegrityError("Operation record updated_at regressed")
        _bounded_string(self.signature_algorithm, "signature_algorithm")
        _bounded_string(self.signature, "signature", allow_empty=True)
        if self.signature_algorithm != OPERATION_SIGNATURE_ALGORITHM:
            raise CacheManifestIntegrityError("Unsupported operation record signature algorithm")
        if self.signature and not _HEX_SHA256.fullmatch(self.signature):
            raise CacheManifestIntegrityError("Operation record signature is invalid")
        object.__setattr__(self, "topology", topology)

    def to_mapping(self, *, include_signature: bool = True) -> dict[str, Any]:
        """Return the exact canonical operation-record projection."""
        result = {
            "candidate_locator": self.candidate_locator,
            "checkpoint": self.checkpoint.value,
            "created_at": self.created_at,
            "expected_generation": self.expected_generation,
            "expected_record_digest": self.expected_record_digest,
            "generation": self.generation,
            "key": self.key,
            "kind": self.kind.value,
            "operation_id": self.operation_id,
            "owner": self.owner,
            "previous_locator": self.previous_locator,
            "schema_version": self.schema_version,
            "signature_algorithm": self.signature_algorithm,
            "store_id": self.store_id,
            "topology": dict(self.topology),
            "transition": self.transition.value,
            "updated_at": self.updated_at,
        }
        if include_signature:
            result["signature"] = self.signature
        return result

    def canonical_bytes(self, *, include_signature: bool = True) -> bytes:
        """Encode bounded persisted operation evidence deterministically."""
        return _canonical_bytes(self.to_mapping(include_signature=include_signature))

    def signing_bytes(self) -> bytes:
        """Domain-separate evidence signatures from all manifest signatures."""
        return OPERATION_SIGNING_DOMAIN + self.canonical_bytes(include_signature=False)

    def with_signature(self, signature: str) -> "LifecycleOperationRecord":
        """Return a signed copy without mutating durable evidence."""
        return replace(self, signature=signature)

    def at_checkpoint(
        self,
        checkpoint: OperationCheckpoint,
        *,
        updated_at: str | None = None,
    ) -> "LifecycleOperationRecord":
        """Advance one legal checkpoint without allowing rollback or skips."""
        if not isinstance(checkpoint, OperationCheckpoint):
            raise CacheManifestIntegrityError("Operation record checkpoint is invalid")
        if checkpoint == self.checkpoint:
            return self
        if checkpoint not in _CHECKPOINT_TRANSITIONS[self.checkpoint]:
            raise CacheManifestIntegrityError("Operation record checkpoint regressed or skipped")
        next_timestamp = self.updated_at if updated_at is None else updated_at
        return replace(self, checkpoint=checkpoint, updated_at=next_timestamp, signature="")

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "LifecycleOperationRecord":
        """Decode one bounded, exact-schema record without assigning authority."""
        if not isinstance(raw, bytes) or not raw:
            raise CacheManifestIntegrityError("Operation record bytes must be non-empty")
        if len(raw) > MAX_OPERATION_RECORD_BYTES:
            raise CacheManifestIntegrityError(
                "Operation record exceeds the byte limit", reason=CacheReason.MANIFEST_BOUNDS
            )
        try:
            decoded = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_int=_parse_signed_64,
                parse_float=_reject_float,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, CacheManifestIntegrityError):
                raise
            raise CacheManifestIntegrityError("Operation record is not valid JSON") from exc
        if not isinstance(decoded, dict) or set(decoded) != _RECORD_FIELDS:
            raise CacheManifestIntegrityError("Operation record has an unknown or missing field")
        _validate_json_value(decoded, depth=1, nodes=[0])
        try:
            return cls(
                schema_version=decoded["schema_version"],
                operation_id=decoded["operation_id"],
                kind=OperationKind(decoded["kind"]),
                key=decoded["key"],
                owner=decoded["owner"],
                store_id=decoded["store_id"],
                topology=decoded["topology"],
                expected_generation=decoded["expected_generation"],
                expected_record_digest=decoded["expected_record_digest"],
                generation=decoded["generation"],
                candidate_locator=decoded["candidate_locator"],
                previous_locator=decoded["previous_locator"],
                transition=OperationTransition(decoded["transition"]),
                checkpoint=OperationCheckpoint(decoded["checkpoint"]),
                created_at=decoded["created_at"],
                updated_at=decoded["updated_at"],
                signature_algorithm=decoded["signature_algorithm"],
                signature=decoded["signature"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheManifestIntegrityError("Operation record fields are invalid") from exc


def store_identity(root: str) -> str:
    """Return a stable opaque store binding without persisting the root path."""
    return hashlib.sha256(root.encode("utf-8")).hexdigest()


__all__ = [
    "CLEAR_TARGET_CHECKPOINT_SIGNING_DOMAIN",
    "CLEAR_TARGET_PAGE_SIGNING_DOMAIN",
    "CLEAR_TARGET_SCHEMA_VERSION",
    "ClearTarget",
    "ClearTargetCheckpoint",
    "ClearTargetPage",
    "LifecycleOperationRecord",
    "MAX_OPERATION_FIELD_BYTES",
    "MAX_OPERATION_RECORD_BYTES",
    "OPERATION_RECORD_OWNER",
    "OPERATION_RECORD_SCHEMA_VERSION",
    "OperationCheckpoint",
    "OperationKind",
    "OperationTransition",
    "store_identity",
]
