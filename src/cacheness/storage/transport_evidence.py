"""Authenticated, immutable-payload-bound transport observations.

Transport metadata such as an S3 ETag is useful corroboration of one exact
remote object, but it is not a content digest or a visibility authority.  This
module preserves that distinction by signing the observation only together with
the immutable payload identity selected by the lifecycle authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import re
from typing import Any, Mapping

from cacheness.error_handling import CacheManifestIntegrityError
from cacheness.storage.manifest import MAX_SIGNED_64


TRANSPORT_EVIDENCE_SCHEMA_VERSION = 1
TRANSPORT_EVIDENCE_SIGNATURE_ALGORITHM = "hmac-sha256"
MAX_TRANSPORT_EVIDENCE_BYTES = 16_384
MAX_TRANSPORT_TEXT_BYTES = 4_096

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OBSERVATION_FIELDS = frozenset({"byte_size", "e_tag", "version"})
_EVIDENCE_FIELDS = frozenset(
    {
        "byte_size",
        "e_tag",
        "generation",
        "key",
        "locator",
        "payload_byte_size",
        "payload_sha256",
        "schema_version",
        "signature",
        "signature_algorithm",
        "store_identity",
        "version",
    }
)
_SIGNING_DOMAIN = b"cacheness.payload-transport-evidence.v1\x00"


def _evidence_error(message: str) -> CacheManifestIntegrityError:
    """Create one fail-closed evidence boundary error."""
    return CacheManifestIntegrityError(message)


def _require_signing_key(signing_key: bytes) -> None:
    """Keep evidence authentication on the repository's fixed HMAC key contract."""
    if not isinstance(signing_key, bytes) or len(signing_key) != 32:
        raise _evidence_error("Transport evidence signing key must be exactly 32 bytes")


def _bounded_text(name: str, value: object, *, allow_none: bool = False) -> str | None:
    """Validate a small untrusted opaque text field before canonical encoding."""
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        raise _evidence_error(f"Transport evidence {name} must be non-empty text")
    if len(value.encode("utf-8")) > MAX_TRANSPORT_TEXT_BYTES:
        raise _evidence_error(f"Transport evidence {name} exceeds the byte limit")
    return value


def _bounded_size(name: str, value: object) -> int:
    """Validate an observed or canonical byte count before authentication."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise _evidence_error(f"Transport evidence {name} must be an integer")
    if not 0 <= value <= MAX_SIGNED_64:
        raise _evidence_error(f"Transport evidence {name} is outside signed-64 range")
    return value


def _canonical_encode(record: Mapping[str, Any]) -> bytes:
    """Encode a bounded record with one deterministic JSON representation."""
    try:
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise _evidence_error("Transport evidence is not canonical JSON") from exc
    if not encoded or len(encoded) > MAX_TRANSPORT_EVIDENCE_BYTES:
        raise _evidence_error("Transport evidence exceeds the byte limit")
    return encoded


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys rather than silently accepting the last one."""
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise _evidence_error(f"Duplicate transport evidence key: {key}")
        record[key] = value
    return record


def _parse_int(value: str) -> int:
    """Parse only signed-64 integer values from untrusted JSON."""
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - json guarantees digits here
        raise _evidence_error("Transport evidence integer is invalid") from exc
    return _bounded_size("integer", parsed)


def _reject_float(_value: str) -> float:
    """Reject JSON floating point extensions at the authentication boundary."""
    raise _evidence_error("Transport evidence floats are not supported")


def _reject_constant(_value: str) -> float:
    """Reject JSON non-finite constants at the authentication boundary."""
    raise _evidence_error("Transport evidence constants are not supported")


@dataclass(frozen=True)
class PayloadTransportObservation:
    """Bounded adapter output for one exact remote object.

    ``e_tag`` and ``version`` are opaque observations.  Neither is parsed as a
    hash or used to decide canonical payload integrity.
    """

    e_tag: str | None
    byte_size: int
    version: str | None

    def __post_init__(self) -> None:
        _bounded_text("e_tag", self.e_tag, allow_none=True)
        _bounded_size("byte_size", self.byte_size)
        _bounded_text("version", self.version, allow_none=True)

    def to_mapping(self) -> dict[str, object]:
        """Return the exact canonical observation projection."""
        return {
            "e_tag": self.e_tag,
            "byte_size": self.byte_size,
            "version": self.version,
        }

    @classmethod
    def from_mapping(cls, record: Mapping[str, object]) -> "PayloadTransportObservation":
        """Decode only the exact current observation field set."""
        if not isinstance(record, Mapping) or set(record) != _OBSERVATION_FIELDS:
            raise _evidence_error("Transport observation has an unsupported field set")
        return cls(
            e_tag=record["e_tag"],
            byte_size=record["byte_size"],
            version=record["version"],
        )


@dataclass(frozen=True)
class PayloadTransportEvidence:
    """Signed transport observations bound to one immutable payload generation."""

    store_identity: str
    key: str
    generation: str
    locator: str
    payload_sha256: str
    payload_byte_size: int
    observation: PayloadTransportObservation
    signature: str
    schema_version: int = TRANSPORT_EVIDENCE_SCHEMA_VERSION
    signature_algorithm: str = TRANSPORT_EVIDENCE_SIGNATURE_ALGORITHM

    def __post_init__(self) -> None:
        if self.schema_version != TRANSPORT_EVIDENCE_SCHEMA_VERSION:
            raise _evidence_error("Transport evidence schema version is unsupported")
        if self.signature_algorithm != TRANSPORT_EVIDENCE_SIGNATURE_ALGORITHM:
            raise _evidence_error("Transport evidence signature algorithm is invalid")
        _bounded_text("store_identity", self.store_identity)
        _bounded_text("key", self.key)
        _bounded_text("generation", self.generation)
        _bounded_text("locator", self.locator)
        if not isinstance(self.payload_sha256, str) or not _HEX_SHA256.fullmatch(
            self.payload_sha256
        ):
            raise _evidence_error("Transport evidence payload SHA-256 is invalid")
        _bounded_size("payload_byte_size", self.payload_byte_size)
        if not isinstance(self.observation, PayloadTransportObservation):
            raise _evidence_error("Transport evidence observation is invalid")
        if self.observation.byte_size != self.payload_byte_size:
            raise _evidence_error(
                "Transport evidence observation size disagrees with canonical payload size"
            )
        if not isinstance(self.signature, str) or not _HEX_SHA256.fullmatch(self.signature):
            raise _evidence_error("Transport evidence signature is invalid")

    @classmethod
    def issue(
        cls,
        observation: PayloadTransportObservation,
        *,
        store_identity: str,
        key: str,
        generation: str,
        locator: str,
        payload_sha256: str,
        payload_byte_size: int,
        signing_key: bytes,
    ) -> "PayloadTransportEvidence":
        """Sign one observation only after canonical payload verification succeeds."""
        _require_signing_key(signing_key)
        unsigned = cls(
            store_identity=store_identity,
            key=key,
            generation=generation,
            locator=locator,
            payload_sha256=payload_sha256,
            payload_byte_size=payload_byte_size,
            observation=observation,
            signature="0" * 64,
        )
        signature = hmac.new(
            signing_key, _SIGNING_DOMAIN + unsigned.signing_bytes(), hashlib.sha256
        ).hexdigest()
        return cls(
            store_identity=unsigned.store_identity,
            key=unsigned.key,
            generation=unsigned.generation,
            locator=unsigned.locator,
            payload_sha256=unsigned.payload_sha256,
            payload_byte_size=unsigned.payload_byte_size,
            observation=unsigned.observation,
            signature=signature,
        )

    def to_mapping(self, *, include_signature: bool = True) -> dict[str, object]:
        """Return the strict canonical signed representation."""
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "signature_algorithm": self.signature_algorithm,
            "store_identity": self.store_identity,
            "key": self.key,
            "generation": self.generation,
            "locator": self.locator,
            "payload_sha256": self.payload_sha256,
            "payload_byte_size": self.payload_byte_size,
            **self.observation.to_mapping(),
        }
        if include_signature:
            record["signature"] = self.signature
        return record

    def signing_bytes(self) -> bytes:
        """Return the domain input projection excluding the authentication tag."""
        return _canonical_encode(self.to_mapping(include_signature=False))

    def canonical_bytes(self) -> bytes:
        """Return the complete authenticated evidence representation."""
        return _canonical_encode(self.to_mapping())

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "PayloadTransportEvidence":
        """Boundedly decode exact-schema evidence before signature verification."""
        if not isinstance(raw, bytes) or not raw or len(raw) > MAX_TRANSPORT_EVIDENCE_BYTES:
            raise _evidence_error("Transport evidence bytes are invalid or too large")
        try:
            record = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_int=_parse_int,
                parse_float=_reject_float,
                parse_constant=_reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, CacheManifestIntegrityError) as exc:
            if isinstance(exc, CacheManifestIntegrityError):
                raise
            raise _evidence_error("Transport evidence is not valid canonical JSON") from exc
        if not isinstance(record, dict) or set(record) != _EVIDENCE_FIELDS:
            raise _evidence_error("Transport evidence has an unsupported field set")
        observation = PayloadTransportObservation.from_mapping(
            {
                "e_tag": record["e_tag"],
                "byte_size": record["byte_size"],
                "version": record["version"],
            }
        )
        evidence = cls(
            schema_version=record["schema_version"],
            signature_algorithm=record["signature_algorithm"],
            store_identity=record["store_identity"],
            key=record["key"],
            generation=record["generation"],
            locator=record["locator"],
            payload_sha256=record["payload_sha256"],
            payload_byte_size=record["payload_byte_size"],
            observation=observation,
            signature=record["signature"],
        )
        if not hmac.compare_digest(evidence.canonical_bytes(), raw):
            raise _evidence_error("Transport evidence is not canonical JSON")
        return evidence

    def verify(
        self,
        *,
        store_identity: str,
        key: str,
        generation: str,
        locator: str,
        payload_sha256: str,
        payload_byte_size: int,
        signing_key: bytes,
    ) -> PayloadTransportObservation:
        """Authenticate and bind evidence to one caller-supplied immutable identity."""
        _require_signing_key(signing_key)
        expected_signature = hmac.new(
            signing_key, _SIGNING_DOMAIN + self.signing_bytes(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(self.signature, expected_signature):
            raise _evidence_error("Transport evidence signature verification failed")

        expected_identity = (
            store_identity,
            key,
            generation,
            locator,
            payload_sha256,
            payload_byte_size,
        )
        signed_identity = (
            self.store_identity,
            self.key,
            self.generation,
            self.locator,
            self.payload_sha256,
            self.payload_byte_size,
        )
        if signed_identity != expected_identity:
            raise _evidence_error("Transport evidence identity does not match the payload")
        return self.observation


__all__ = [
    "MAX_TRANSPORT_EVIDENCE_BYTES",
    "MAX_TRANSPORT_TEXT_BYTES",
    "PayloadTransportEvidence",
    "PayloadTransportObservation",
    "TRANSPORT_EVIDENCE_SCHEMA_VERSION",
]
