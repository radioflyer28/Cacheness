"""Canonical, backend-neutral manifest records for BlobStore payloads.

The manifest describes a handler-owned payload.  It deliberately does not add
another container around native NumPy, parquet, pickle, or dill bytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping


MANIFEST_SCHEMA_VERSION = 1
PAYLOAD_FORMAT_VERSION = 1
MANIFEST_SIGNATURE_ALGORITHM = "hmac-sha256"
PAYLOAD_DIGEST_ALGORITHM = "sha256"


class ManifestDecodeError(ValueError):
    """Raised when a canonical manifest record cannot be decoded."""


class UnsupportedManifestVersionError(ManifestDecodeError):
    """Raised for a manifest or payload version this reader does not support."""


def _freeze_json_value(value: Any) -> Any:
    """Freeze JSON-compatible metadata so a frozen manifest stays immutable."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
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
            raise ManifestDecodeError(f"Duplicate canonical manifest key: {key}")
        result[key] = value
    return result


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
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise UnsupportedManifestVersionError(
                f"Unsupported manifest schema version: {self.schema_version}"
            )
        if self.payload_format_version != PAYLOAD_FORMAT_VERSION:
            raise UnsupportedManifestVersionError(
                "Unsupported payload format version: "
                f"{self.payload_format_version}"
            )
        if any(not isinstance(value, str) or not value for value in required_strings.values()):
            raise ManifestDecodeError("Canonical manifest requires non-empty string fields")
        if self.state not in {
            "prepared",
            "committed",
            "replacing",
            "tombstoned",
            "conflicted",
        }:
            raise ManifestDecodeError(f"Unsupported manifest lifecycle state: {self.state}")
        if self.digest_algorithm != PAYLOAD_DIGEST_ALGORITHM:
            raise ManifestDecodeError(
                f"Unsupported payload digest algorithm: {self.digest_algorithm}"
            )
        if not isinstance(self.byte_size, int) or isinstance(self.byte_size, bool):
            raise ManifestDecodeError("Canonical manifest byte_size must be an integer")
        if self.byte_size < 0:
            raise ManifestDecodeError("Canonical manifest byte_size cannot be negative")
        if not isinstance(self.handler_metadata, Mapping) or not isinstance(
            self.user_metadata, Mapping
        ):
            raise ManifestDecodeError("Canonical manifest metadata must be string-keyed maps")
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
        try:
            text = json.dumps(
                self.to_mapping(include_signature=include_signature),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ManifestDecodeError("Manifest metadata is not JSON-compatible") from exc
        return text.encode("utf-8")

    def signing_bytes(self) -> bytes:
        """Return the complete signed projection, excluding only the signature."""
        return self.canonical_bytes(include_signature=False)

    def with_signature(self, signature: str) -> "BlobManifestV1":
        """Return a new manifest with an authenticated HMAC value."""
        if not isinstance(signature, str) or not signature:
            raise ManifestDecodeError("Canonical manifest signature must be non-empty")
        return replace(self, signature=signature)

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "BlobManifestV1":
        """Decode a schema-1 record without interpreting any payload bytes."""
        if not isinstance(raw, bytes) or not raw:
            raise ManifestDecodeError("Canonical manifest bytes must be non-empty")
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ManifestDecodeError("Canonical manifest is not valid UTF-8") from exc
        try:
            record = json.loads(decoded, object_pairs_hook=_reject_duplicate_keys)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, ManifestDecodeError):
                raise
            raise ManifestDecodeError("Canonical manifest is not valid JSON") from exc
        if not isinstance(record, dict):
            raise ManifestDecodeError("Canonical manifest must be a JSON object")
        expected = {
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
        if set(record) != expected:
            raise ManifestDecodeError("Canonical manifest has an unknown or missing field")
        return cls(**record)


__all__ = [
    "BlobManifestV1",
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_SIGNATURE_ALGORITHM",
    "PAYLOAD_DIGEST_ALGORITHM",
    "PAYLOAD_FORMAT_VERSION",
    "ManifestDecodeError",
    "UnsupportedManifestVersionError",
]
