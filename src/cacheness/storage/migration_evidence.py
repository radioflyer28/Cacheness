"""Authenticated bounded evidence for explicit offline maintenance runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hmac
import json
from typing import Any, Mapping

from .migration_authority import (
    ActivationReceipt,
    AuthorityIdentitySnapshot,
    VerifiedCandidateReceipt,
)


_EVIDENCE_VERSION = 1
_EVIDENCE_DOMAIN = b"cacheness-maintenance-evidence-v1\x00"
_MAX_EVIDENCE_BYTES = 65_536
_MAX_TEXT_BYTES = 512
_MAX_CANONICAL_DEPTH = 16
_MAX_CANONICAL_NODES = 16_384
_MAX_CANONICAL_ITEMS = 4_096


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON object keys before a plan can reinterpret them."""
    record: dict[str, object] = {}
    for key, value in pairs:
        if key in record:
            raise ValueError("canonical JSON cannot contain duplicate keys")
        record[key] = value
    return record


def _parse_canonical_int(raw: str) -> int:
    """Accept only the spelling emitted by the canonical JSON encoder."""
    value = int(raw)
    if str(value) != raw:
        raise ValueError("canonical JSON integer is non-canonical")
    return value


def _reject_float(raw: str) -> object:
    raise ValueError(f"canonical JSON does not permit floating-point values: {raw}")


def _reject_constant(raw: str) -> object:
    raise ValueError(f"canonical JSON does not permit constants: {raw}")


def _validate_canonical_value(
    value: object, *, depth: int, nodes: list[int], max_text_bytes: int
) -> None:
    """Apply allocation bounds before callers construct semantic plan values."""
    nodes[0] += 1
    if nodes[0] > _MAX_CANONICAL_NODES or depth > _MAX_CANONICAL_DEPTH:
        raise ValueError("canonical JSON exceeds structural bounds")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > max_text_bytes:
            raise ValueError("canonical JSON string exceeds the byte bound")
        return
    if isinstance(value, list):
        if len(value) > _MAX_CANONICAL_ITEMS:
            raise ValueError("canonical JSON collection exceeds the item bound")
        for item in value:
            _validate_canonical_value(
                item, depth=depth + 1, nodes=nodes, max_text_bytes=max_text_bytes
            )
        return
    if isinstance(value, dict):
        if len(value) > _MAX_CANONICAL_ITEMS:
            raise ValueError("canonical JSON collection exceeds the item bound")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical JSON object key is invalid")
            _validate_canonical_value(
                key, depth=depth + 1, nodes=nodes, max_text_bytes=max_text_bytes
            )
            _validate_canonical_value(
                item, depth=depth + 1, nodes=nodes, max_text_bytes=max_text_bytes
            )
        return
    raise ValueError("canonical JSON value is invalid")


def decode_bounded_canonical_json(
    raw: bytes, *, max_bytes: int, max_text_bytes: int = _MAX_TEXT_BYTES
) -> dict[str, object]:
    """Decode a bounded canonical JSON object without duplicate-key ambiguity."""
    if not isinstance(raw, bytes) or not raw or len(raw) > max_bytes:
        raise ValueError("canonical JSON bytes are invalid")
    try:
        decoded = raw.decode("utf-8")
        record = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_int=_parse_canonical_int,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("canonical JSON is invalid") from exc
    if not isinstance(record, dict):
        raise ValueError("canonical JSON must contain one object")
    _validate_canonical_value(
        record, depth=1, nodes=[0], max_text_bytes=max_text_bytes
    )
    return record


class MaintenanceEvidenceState(str, Enum):
    """Monotonic offline-service boundaries represented by one evidence model."""

    INSPECTED = "inspected"
    PLANNED = "planned"
    STAGED = "staged"
    VERIFIED = "verified"
    ACTIVATED = "activated"


def _bounded_text(value: object, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{field_name} must be a bounded string")
    if len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ValueError(f"{field_name} exceeds the byte bound")
    return value


def _identity_record(identity: AuthorityIdentitySnapshot) -> dict[str, object]:
    return {
        "authority_kind": identity.authority_kind,
        "capability": identity.capability,
        "revision": identity.revision,
        "schema_version": identity.schema_version,
        "store_id": identity.store_id,
    }


def _identity_from_record(record: object, field_name: str) -> AuthorityIdentitySnapshot:
    if not isinstance(record, Mapping) or set(record) != {
        "authority_kind",
        "capability",
        "revision",
        "schema_version",
        "store_id",
    }:
        raise ValueError(f"{field_name} is invalid")
    return AuthorityIdentitySnapshot(
        store_id=record["store_id"],
        revision=record["revision"],
        authority_kind=record["authority_kind"],
        capability=record["capability"],
        schema_version=record["schema_version"],
    )


def _receipt_record(receipt: VerifiedCandidateReceipt) -> dict[str, object]:
    return {
        "byte_count": receipt.byte_count,
        "candidate_digest": receipt.candidate_digest,
        "destination_identity": _identity_record(receipt.destination_identity),
        "destination_revision": receipt.destination_revision,
        "entry_count": receipt.entry_count,
        "plan_digest": receipt.plan_digest,
        "run_id": receipt.run_id,
        "source_identity": _identity_record(receipt.source_identity),
        "source_revision": receipt.source_revision,
    }


def _receipt_from_record(record: object) -> VerifiedCandidateReceipt:
    if not isinstance(record, Mapping) or set(record) != {
        "byte_count",
        "candidate_digest",
        "destination_identity",
        "destination_revision",
        "entry_count",
        "plan_digest",
        "run_id",
        "source_identity",
        "source_revision",
    }:
        raise ValueError("candidate_receipt is invalid")
    return VerifiedCandidateReceipt(
        run_id=record["run_id"],
        plan_digest=record["plan_digest"],
        source_identity=_identity_from_record(record["source_identity"], "source_identity"),
        source_revision=record["source_revision"],
        destination_identity=_identity_from_record(
            record["destination_identity"], "destination_identity"
        ),
        destination_revision=record["destination_revision"],
        candidate_digest=record["candidate_digest"],
        entry_count=record["entry_count"],
        byte_count=record["byte_count"],
    )


@dataclass(frozen=True)
class MaintenanceRunEvidence:
    """Versioned canonical envelope; signing material never appears in this value."""

    evidence_version: int
    run_id: str
    plan_digest: str
    source_identity: AuthorityIdentitySnapshot
    source_revision: int
    destination_identity: AuthorityIdentitySnapshot
    state: MaintenanceEvidenceState
    completed_steps: tuple[str, ...]
    candidate_receipt: VerifiedCandidateReceipt | None = None
    activation_receipt: ActivationReceipt | None = None

    def __post_init__(self) -> None:
        if self.evidence_version != _EVIDENCE_VERSION:
            raise ValueError("unsupported maintenance evidence version")
        _bounded_text(self.run_id, "run_id")
        _bounded_text(self.plan_digest, "plan_digest", allow_empty=True)
        if self.plan_digest and (
            len(self.plan_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.plan_digest)
        ):
            raise ValueError("plan_digest must be a SHA-256 hexadecimal value")
        if type(self.source_revision) is not int or self.source_revision < 0:
            raise ValueError("source_revision must be a non-negative integer")
        if not isinstance(self.state, MaintenanceEvidenceState):
            raise ValueError("state must be a maintenance evidence state")
        if not isinstance(self.completed_steps, tuple) or len(self.completed_steps) > 8:
            raise ValueError("completed_steps must be a bounded tuple")
        for step in self.completed_steps:
            _bounded_text(step, "completed_step")
        if self.candidate_receipt is not None and self.candidate_receipt.run_id != self.run_id:
            raise ValueError("candidate_receipt run_id disagrees with evidence")
        if self.activation_receipt is not None and self.candidate_receipt is None:
            raise ValueError("activation_receipt requires candidate_receipt")

    def to_record(self) -> dict[str, object]:
        """Return the public canonical record, omitting no authoritative fields."""
        record: dict[str, object] = {
            "activation_receipt": None,
            "candidate_receipt": None,
            "completed_steps": list(self.completed_steps),
            "destination_identity": _identity_record(self.destination_identity),
            "evidence_version": self.evidence_version,
            "plan_digest": self.plan_digest,
            "run_id": self.run_id,
            "source_identity": _identity_record(self.source_identity),
            "source_revision": self.source_revision,
            "state": self.state.value,
        }
        if self.candidate_receipt is not None:
            record["candidate_receipt"] = _receipt_record(self.candidate_receipt)
        if self.activation_receipt is not None:
            record["activation_receipt"] = {
                "activation_revision": self.activation_receipt.activation_revision
            }
        return record

    @classmethod
    def from_record(cls, record: object) -> "MaintenanceRunEvidence":
        """Decode an exact evidence record only after envelope authentication."""
        required = {
            "activation_receipt",
            "candidate_receipt",
            "completed_steps",
            "destination_identity",
            "evidence_version",
            "plan_digest",
            "run_id",
            "source_identity",
            "source_revision",
            "state",
        }
        if not isinstance(record, Mapping) or set(record) != required:
            raise ValueError("maintenance evidence fields are invalid")
        steps = record["completed_steps"]
        if not isinstance(steps, list):
            raise ValueError("completed_steps is invalid")
        candidate = record["candidate_receipt"]
        candidate_receipt = None if candidate is None else _receipt_from_record(candidate)
        activation = record["activation_receipt"]
        activation_receipt = None
        if activation is not None:
            if not isinstance(activation, Mapping) or set(activation) != {"activation_revision"}:
                raise ValueError("activation_receipt is invalid")
            if candidate_receipt is None:
                raise ValueError("activation_receipt lacks candidate_receipt")
            activation_receipt = ActivationReceipt(
                candidate_receipt=candidate_receipt,
                activation_revision=activation["activation_revision"],
            )
        try:
            state = MaintenanceEvidenceState(record["state"])
        except (TypeError, ValueError) as exc:
            raise ValueError("maintenance evidence state is invalid") from exc
        return cls(
            evidence_version=record["evidence_version"],
            run_id=record["run_id"],
            plan_digest=record["plan_digest"],
            source_identity=_identity_from_record(record["source_identity"], "source_identity"),
            source_revision=record["source_revision"],
            destination_identity=_identity_from_record(
                record["destination_identity"], "destination_identity"
            ),
            state=state,
            completed_steps=tuple(steps),
            candidate_receipt=candidate_receipt,
            activation_receipt=activation_receipt,
        )


def _canonical_bytes(record: Mapping[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encode_maintenance_evidence(evidence: MaintenanceRunEvidence, signing_key: bytes) -> bytes:
    """Authenticate a bounded evidence envelope using provider-supplied bytes."""
    if not isinstance(evidence, MaintenanceRunEvidence):
        raise TypeError("evidence must be a MaintenanceRunEvidence")
    if type(signing_key) is not bytes or len(signing_key) != 32:
        raise ValueError("maintenance evidence requires a 32-byte signing key")
    unsigned = evidence.to_record()
    signature = hmac.new(signing_key, _EVIDENCE_DOMAIN + _canonical_bytes(unsigned), "sha256").hexdigest()
    envelope = {"evidence": unsigned, "signature": signature}
    encoded = _canonical_bytes(envelope)
    if len(encoded) > _MAX_EVIDENCE_BYTES:
        raise ValueError("maintenance evidence exceeds the byte bound")
    return encoded


def decode_maintenance_evidence(raw: bytes, signing_key: bytes) -> MaintenanceRunEvidence:
    """Fail closed on noncanonical, oversized, or unauthenticated evidence."""
    if not isinstance(raw, bytes) or not raw or len(raw) > _MAX_EVIDENCE_BYTES:
        raise ValueError("maintenance evidence bytes are invalid")
    if type(signing_key) is not bytes or len(signing_key) != 32:
        raise ValueError("maintenance evidence requires a 32-byte signing key")
    try:
        envelope = decode_bounded_canonical_json(raw, max_bytes=_MAX_EVIDENCE_BYTES)
    except ValueError as exc:
        raise ValueError("maintenance evidence is not canonical JSON") from exc
    if set(envelope) != {"evidence", "signature"}:
        raise ValueError("maintenance evidence envelope is invalid")
    evidence = envelope["evidence"]
    signature = envelope["signature"]
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError("maintenance evidence signature is invalid")
    expected = hmac.new(signing_key, _EVIDENCE_DOMAIN + _canonical_bytes(evidence), "sha256").hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("maintenance evidence authentication failed")
    if raw != _canonical_bytes({"evidence": evidence, "signature": signature}):
        raise ValueError("maintenance evidence is not canonical")
    return MaintenanceRunEvidence.from_record(evidence)


__all__ = [
    "MaintenanceEvidenceState",
    "MaintenanceRunEvidence",
    "decode_bounded_canonical_json",
    "decode_maintenance_evidence",
    "encode_maintenance_evidence",
]
