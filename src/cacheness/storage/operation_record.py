"""Bounded authenticated evidence for one BlobStore lifecycle mutation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping

from cacheness.error_handling import CacheManifestIntegrityError, CacheReason


OPERATION_RECORD_SCHEMA_VERSION = 1
OPERATION_SIGNATURE_ALGORITHM = "hmac-sha256"
OPERATION_SIGNING_DOMAIN = b"cacheness.operation-record.v1\x00"
MAX_OPERATION_RECORD_BYTES = 1_048_576
MAX_OPERATION_FIELD_BYTES = 8_192


class OperationKind(str, Enum):
    """Lifecycle mutations represented by durable operation evidence."""

    PUT = "put"


class OperationCheckpoint(str, Enum):
    """Monotonic checkpoints that classify authority and cleanup progress."""

    PREPARED = "prepared"
    CANDIDATE_PUBLISHED = "candidate_published"
    AUTHORITY_PUBLISHED = "authority_published"
    CLEANUP_COMPLETED = "cleanup_completed"


_CHECKPOINT_INDEX = {checkpoint: index for index, checkpoint in enumerate(OperationCheckpoint)}
_RECORD_FIELDS = frozenset(
    {
        "candidate_locator",
        "checkpoint",
        "expected_generation",
        "generation",
        "key",
        "kind",
        "operation_id",
        "previous_locator",
        "schema_version",
        "signature",
        "signature_algorithm",
        "store_id",
    }
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise CacheManifestIntegrityError("Duplicate operation record key")
        record[key] = value
    return record


def _bounded_string(
    value: object,
    field: str,
    *,
    allow_none: bool = False,
    allow_empty: bool = False,
) -> str | None:
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


def _canonical_bytes(record: Mapping[str, Any]) -> bytes:
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


@dataclass(frozen=True)
class LifecycleOperationRecord:
    """The authenticated, replay-safe description of one immutable generation."""

    schema_version: int
    operation_id: str
    kind: OperationKind
    key: str
    store_id: str
    expected_generation: str | None
    generation: str
    candidate_locator: str
    previous_locator: str | None
    checkpoint: OperationCheckpoint = OperationCheckpoint.PREPARED
    signature_algorithm: str = OPERATION_SIGNATURE_ALGORITHM
    signature: str = ""

    def __post_init__(self) -> None:
        """Reject malformed or oversized evidence before it reaches storage."""
        if self.schema_version != OPERATION_RECORD_SCHEMA_VERSION:
            raise CacheManifestIntegrityError("Unsupported operation record schema version")
        if not isinstance(self.kind, OperationKind):
            raise CacheManifestIntegrityError("Operation record kind is invalid")
        if not isinstance(self.checkpoint, OperationCheckpoint):
            raise CacheManifestIntegrityError("Operation record checkpoint is invalid")
        for field, value, allow_none in (
            ("operation_id", self.operation_id, False),
            ("key", self.key, False),
            ("store_id", self.store_id, False),
            ("expected_generation", self.expected_generation, True),
            ("generation", self.generation, False),
            ("candidate_locator", self.candidate_locator, False),
            ("previous_locator", self.previous_locator, True),
            ("signature_algorithm", self.signature_algorithm, False),
            ("signature", self.signature, False),
        ):
            _bounded_string(
                value,
                field,
                allow_none=allow_none,
                allow_empty=field == "signature",
            )
        if self.signature and not _is_sha256_hex(self.signature):
            raise CacheManifestIntegrityError("Operation record signature is invalid")
        if self.signature_algorithm != OPERATION_SIGNATURE_ALGORITHM:
            raise CacheManifestIntegrityError("Unsupported operation record signature algorithm")

    def to_mapping(self, *, include_signature: bool = True) -> dict[str, Any]:
        """Return the exact canonical operation-record projection."""
        result = {
            "candidate_locator": self.candidate_locator,
            "checkpoint": self.checkpoint.value,
            "expected_generation": self.expected_generation,
            "generation": self.generation,
            "key": self.key,
            "kind": self.kind.value,
            "operation_id": self.operation_id,
            "previous_locator": self.previous_locator,
            "schema_version": self.schema_version,
            "signature_algorithm": self.signature_algorithm,
            "store_id": self.store_id,
        }
        if include_signature:
            result["signature"] = self.signature
        return result

    def canonical_bytes(self) -> bytes:
        """Encode bounded persisted operation evidence."""
        return _canonical_bytes(self.to_mapping())

    def signing_bytes(self) -> bytes:
        """Domain-separate evidence signatures from canonical manifest signatures."""
        return OPERATION_SIGNING_DOMAIN + _canonical_bytes(
            self.to_mapping(include_signature=False)
        )

    def with_signature(self, signature: str) -> "LifecycleOperationRecord":
        """Return a signed copy without mutating persisted evidence."""
        return replace(self, signature=signature)

    def at_checkpoint(
        self, checkpoint: OperationCheckpoint
    ) -> "LifecycleOperationRecord":
        """Return a monotonic progress transition for durable checkpointing."""
        if _CHECKPOINT_INDEX[checkpoint] < _CHECKPOINT_INDEX[self.checkpoint]:
            raise CacheManifestIntegrityError("Operation record checkpoint regressed")
        return replace(self, checkpoint=checkpoint, signature="")

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "LifecycleOperationRecord":
        """Decode bounded evidence without treating it as authority."""
        if not isinstance(raw, bytes) or not raw:
            raise CacheManifestIntegrityError("Operation record bytes must be non-empty")
        if len(raw) > MAX_OPERATION_RECORD_BYTES:
            raise CacheManifestIntegrityError(
                "Operation record exceeds the byte limit",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        try:
            decoded = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, CacheManifestIntegrityError):
                raise
            raise CacheManifestIntegrityError("Operation record is not valid JSON") from exc
        if not isinstance(decoded, dict) or set(decoded) != _RECORD_FIELDS:
            raise CacheManifestIntegrityError("Operation record has an unknown or missing field")
        try:
            return cls(
                schema_version=decoded["schema_version"],
                operation_id=decoded["operation_id"],
                kind=OperationKind(decoded["kind"]),
                key=decoded["key"],
                store_id=decoded["store_id"],
                expected_generation=decoded["expected_generation"],
                generation=decoded["generation"],
                candidate_locator=decoded["candidate_locator"],
                previous_locator=decoded["previous_locator"],
                checkpoint=OperationCheckpoint(decoded["checkpoint"]),
                signature_algorithm=decoded["signature_algorithm"],
                signature=decoded["signature"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheManifestIntegrityError("Operation record fields are invalid") from exc


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def store_identity(root: str) -> str:
    """Return a stable opaque store binding without persisting the root path."""
    return hashlib.sha256(root.encode("utf-8")).hexdigest()


__all__ = [
    "LifecycleOperationRecord",
    "MAX_OPERATION_FIELD_BYTES",
    "MAX_OPERATION_RECORD_BYTES",
    "OPERATION_RECORD_SCHEMA_VERSION",
    "OperationCheckpoint",
    "OperationKind",
    "store_identity",
]
