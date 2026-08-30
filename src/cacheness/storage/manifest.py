"""Canonical, backend-neutral manifest records for BlobStore payloads.

The manifest describes a handler-owned payload.  It deliberately does not add
another container around native NumPy, parquet, pickle, or dill bytes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

from cacheness.error_handling import (
    CacheManifestIntegrityError,
    CacheManifestUnsupportedVersionError,
    CacheReason,
)


MANIFEST_SCHEMA_VERSION = 1
PAYLOAD_FORMAT_VERSION = 1
MANIFEST_SIGNATURE_ALGORITHM = "hmac-sha256"
PAYLOAD_DIGEST_ALGORITHM = "sha256"
MAX_MANIFEST_BYTES = 1_048_576
MAX_NESTING_DEPTH = 16
MAX_COLLECTION_ITEMS = 4_096
MAX_TOTAL_NODES = 16_384
MAX_STRING_UTF8_BYTES = 262_144
MIN_SIGNED_64 = -(2**63)
MAX_SIGNED_64 = 2**63 - 1
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_CANONICAL_FIELDS = frozenset(
    {
        "byte_size",
        "created_at",
        "digest",
        "digest_algorithm",
        "generation",
        "handler_metadata",
        "handler_type",
        "key",
        "locator",
        "payload_format",
        "payload_format_version",
        "schema_version",
        "signature",
        "signature_algorithm",
        "state",
        "user_metadata",
    }
)


ManifestDecodeError = CacheManifestIntegrityError
UnsupportedManifestVersionError = CacheManifestUnsupportedVersionError


def _freeze_json_value(value: Any) -> Any:
    """Freeze JSON-compatible metadata so a frozen manifest stays immutable."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CacheManifestIntegrityError("Manifest metadata keys must be strings")
        return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    """Return normal JSON-compatible containers for canonical serialization."""
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build an object while rejecting ambiguity from duplicate JSON keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CacheManifestIntegrityError(
                f"Duplicate canonical manifest key: {key}"
            )
        result[key] = value
    return result


def _parse_canonical_int(value: str) -> int:
    """Parse an integer token without accepting out-of-range Python integers."""
    try:
        number = int(value)
    except ValueError as exc:
        raise CacheManifestIntegrityError("Manifest integer is invalid") from exc
    if not MIN_SIGNED_64 <= number <= MAX_SIGNED_64:
        raise CacheManifestIntegrityError(
            "Manifest integer exceeds the signed-64 range",
            reason=CacheReason.MANIFEST_BOUNDS,
        )
    return number


def _reject_float(_value: str) -> None:
    """Reject floats and non-finite values from canonical JSON altogether."""
    raise CacheManifestIntegrityError("Manifest values cannot be floating point")


def _reject_constant(_value: str) -> None:
    """Reject JSON's NaN and infinity extensions from canonical JSON."""
    raise CacheManifestIntegrityError("Manifest values cannot be non-finite")


def _validate_canonical_value(value: Any, *, depth: int, nodes: list[int]) -> None:
    """Enforce bounded JSON-compatible metadata before model construction."""
    if depth > MAX_NESTING_DEPTH:
        raise CacheManifestIntegrityError(
            "Manifest nesting limit exceeded", reason=CacheReason.MANIFEST_BOUNDS
        )
    nodes[0] += 1
    if nodes[0] > MAX_TOTAL_NODES:
        raise CacheManifestIntegrityError(
            "Manifest total nodes limit exceeded", reason=CacheReason.MANIFEST_BOUNDS
        )
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if not MIN_SIGNED_64 <= value <= MAX_SIGNED_64:
            raise CacheManifestIntegrityError(
                "Manifest integer exceeds the signed-64 range",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_STRING_UTF8_BYTES:
            raise CacheManifestIntegrityError(
                "Manifest string byte limit exceeded", reason=CacheReason.MANIFEST_BOUNDS
            )
        return
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise CacheManifestIntegrityError(
                "Manifest collection item limit exceeded",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        for item in value:
            _validate_canonical_value(item, depth=depth + 1, nodes=nodes)
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise CacheManifestIntegrityError(
                "Manifest collection item limit exceeded",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        for key, item in value.items():
            if not isinstance(key, str):
                raise CacheManifestIntegrityError("Manifest metadata keys must be strings")
            _validate_canonical_value(key, depth=depth + 1, nodes=nodes)
            _validate_canonical_value(item, depth=depth + 1, nodes=nodes)
        return
    raise CacheManifestIntegrityError("Manifest metadata is not JSON-compatible")


def _canonical_encode(record: Mapping[str, Any]) -> bytes:
    """Encode one bounded raw record without applying semantic manifest rules."""
    _validate_canonical_value(record, depth=1, nodes=[0])
    try:
        text = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CacheManifestIntegrityError("Manifest metadata is not JSON-compatible") from exc
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise CacheManifestIntegrityError(
            "Canonical manifest byte limit exceeded",
            reason=CacheReason.MANIFEST_BOUNDS,
        )
    return encoded


def decode_canonical_manifest_record(raw: bytes) -> dict[str, Any]:
    """Boundedly decode only the framing needed before authentication.

    The schema version is needed to dispatch the wire format.  Every other
    structural claim remains raw until its complete signed projection has been
    authenticated by the BlobStore read boundary.
    """
    if not isinstance(raw, bytes) or not raw:
        raise CacheManifestIntegrityError("Canonical manifest bytes must be non-empty")
    if len(raw) > MAX_MANIFEST_BYTES:
        raise CacheManifestIntegrityError(
            "Canonical manifest byte limit exceeded",
            reason=CacheReason.MANIFEST_BOUNDS,
        )
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CacheManifestIntegrityError("Canonical manifest is not valid UTF-8") from exc
    try:
        record = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_int=_parse_canonical_int,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, CacheManifestIntegrityError):
            raise
        raise CacheManifestIntegrityError("Canonical manifest is not valid JSON") from exc
    if not isinstance(record, dict):
        raise CacheManifestIntegrityError("Canonical manifest must be a JSON object")
    _validate_canonical_value(record, depth=1, nodes=[0])
    schema_version = record.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise CacheManifestIntegrityError("Manifest schema version must be an integer")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise CacheManifestUnsupportedVersionError(
            f"Unsupported manifest schema version: {schema_version}"
        )
    return record


def canonical_signing_bytes_from_record(record: Mapping[str, Any]) -> bytes:
    """Return the fixed unsigned projection of a bounded raw manifest record."""
    unsigned = dict(record)
    unsigned.pop("signature", None)
    return _canonical_encode(unsigned)


@dataclass(frozen=True)
class BlobManifestV1:
    """The schema-1 signed description of one committed BlobStore payload."""

    schema_version: int
    key: str
    generation: str
    state: str
    locator: str
    handler_type: str
    payload_format: str
    payload_format_version: int
    digest_algorithm: str
    digest: str
    byte_size: int
    created_at: str
    handler_metadata: Mapping[str, Any]
    user_metadata: Mapping[str, Any]
    signature_algorithm: str = MANIFEST_SIGNATURE_ALGORITHM
    signature: str = ""

    def __post_init__(self) -> None:
        """Validate the immutable schema-1 shape before it reaches storage."""
        required_strings = {
            "key": self.key,
            "generation": self.generation,
            "state": self.state,
            "locator": self.locator,
            "handler_type": self.handler_type,
            "payload_format": self.payload_format,
            "digest_algorithm": self.digest_algorithm,
            "digest": self.digest,
            "created_at": self.created_at,
            "signature_algorithm": self.signature_algorithm,
        }
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise CacheManifestIntegrityError("Manifest schema version must be an integer")
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise CacheManifestUnsupportedVersionError(
                f"Unsupported manifest schema version: {self.schema_version}"
            )
        if not isinstance(self.payload_format_version, int) or isinstance(
            self.payload_format_version, bool
        ):
            raise CacheManifestIntegrityError(
                "Payload format version must be an integer"
            )
        if self.payload_format_version != PAYLOAD_FORMAT_VERSION:
            raise CacheManifestUnsupportedVersionError(
                "Unsupported payload format version: "
                f"{self.payload_format_version}"
            )
        if any(not isinstance(value, str) or not value for value in required_strings.values()):
            raise CacheManifestIntegrityError(
                "Canonical manifest requires non-empty string fields"
            )
        if any(
            len(value.encode("utf-8")) > MAX_STRING_UTF8_BYTES
            for value in required_strings.values()
        ):
            raise CacheManifestIntegrityError(
                "Manifest string byte limit exceeded",
                reason=CacheReason.MANIFEST_BOUNDS,
            )
        if self.signature_algorithm != MANIFEST_SIGNATURE_ALGORITHM:
            raise CacheManifestIntegrityError("Unsupported manifest signature algorithm")
        if self.state not in {
            "prepared",
            "committed",
            "replacing",
            "tombstoned",
            "conflicted",
        }:
            raise CacheManifestIntegrityError(
                f"Unsupported manifest lifecycle state: {self.state}"
            )
        if self.digest_algorithm != PAYLOAD_DIGEST_ALGORITHM:
            raise CacheManifestIntegrityError(
                f"Unsupported payload digest algorithm: {self.digest_algorithm}"
            )
        if not isinstance(self.byte_size, int) or isinstance(self.byte_size, bool):
            raise CacheManifestIntegrityError("Canonical manifest byte_size must be an integer")
        if not 0 <= self.byte_size <= MAX_SIGNED_64:
            raise CacheManifestIntegrityError(
                "Canonical manifest byte_size must be a non-negative signed-64 integer"
            )
        if not isinstance(self.handler_metadata, Mapping) or not isinstance(
            self.user_metadata, Mapping
        ):
            raise CacheManifestIntegrityError(
                "Canonical manifest metadata must be string-keyed maps"
            )
        if not _HEX_SHA256.fullmatch(self.digest):
            raise CacheManifestIntegrityError("Canonical manifest digest is invalid")
        if self.signature and not _HEX_SHA256.fullmatch(self.signature):
            raise CacheManifestIntegrityError("Canonical manifest signature is invalid")
        _validate_canonical_value(self.handler_metadata, depth=2, nodes=[0])
        _validate_canonical_value(self.user_metadata, depth=2, nodes=[0])
        object.__setattr__(
            self, "handler_metadata", _freeze_json_value(self.handler_metadata)
        )
        object.__setattr__(self, "user_metadata", _freeze_json_value(self.user_metadata))

    def to_mapping(self, *, include_signature: bool = True) -> dict[str, Any]:
        """Return a normal mapping suitable for canonical JSON serialization."""
        record = {
            "byte_size": self.byte_size,
            "created_at": self.created_at,
            "digest": self.digest,
            "digest_algorithm": self.digest_algorithm,
            "generation": self.generation,
            "handler_metadata": _thaw_json_value(self.handler_metadata),
            "handler_type": self.handler_type,
            "key": self.key,
            "locator": self.locator,
            "payload_format": self.payload_format,
            "payload_format_version": self.payload_format_version,
            "schema_version": self.schema_version,
            "signature_algorithm": self.signature_algorithm,
            "state": self.state,
            "user_metadata": _thaw_json_value(self.user_metadata),
        }
        if include_signature:
            record["signature"] = self.signature
        return record

    def canonical_bytes(self, *, include_signature: bool = True) -> bytes:
        """Encode this exact field set into deterministic UTF-8 JSON bytes."""
        record = self.to_mapping(include_signature=include_signature)
        # Outgoing records must obey the same aggregate structural limits as
        # untrusted records on the read path.  Validating the two metadata
        # maps independently is insufficient because their combined node
        # count, plus the enclosing schema fields, can exceed the wire limit.
        return _canonical_encode(record)

    def signing_bytes(self) -> bytes:
        """Return the complete signed projection, excluding only the signature."""
        return self.canonical_bytes(include_signature=False)

    def with_signature(self, signature: str) -> "BlobManifestV1":
        """Return a new manifest with an authenticated HMAC value."""
        if not isinstance(signature, str) or not signature:
            raise CacheManifestIntegrityError("Canonical manifest signature must be non-empty")
        return replace(self, signature=signature)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "BlobManifestV1":
        """Decode a schema-1 record without interpreting any payload bytes."""
        return cls.from_mapping(decode_canonical_manifest_record(raw))

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "BlobManifestV1":
        """Validate a previously bounded and, when required, authenticated map."""
        if set(record) != _CANONICAL_FIELDS:
            raise CacheManifestIntegrityError(
                "Canonical manifest has an unknown or missing field"
            )
        return cls(**record)


__all__ = [
    "BlobManifestV1",
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_SIGNATURE_ALGORITHM",
    "PAYLOAD_DIGEST_ALGORITHM",
    "PAYLOAD_FORMAT_VERSION",
    "MAX_MANIFEST_BYTES",
    "MAX_NESTING_DEPTH",
    "MAX_COLLECTION_ITEMS",
    "MAX_TOTAL_NODES",
    "MAX_STRING_UTF8_BYTES",
    "canonical_signing_bytes_from_record",
    "decode_canonical_manifest_record",
    "ManifestDecodeError",
    "UnsupportedManifestVersionError",
]
