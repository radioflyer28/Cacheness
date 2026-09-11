"""Authenticated bounded evidence for explicit offline maintenance runs.

Maintenance evidence records an operator-directed offline workflow. It is never
lifecycle authority: a selected ``MigrationAuthority`` alone publishes a
candidate. The codec accepts only a small canonical JSON language so corrupt or
ambiguous evidence fails before it can direct a resume.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from cacheness.error_handling import (
    CacheBlobMigrationEvidenceError,
    CacheBlobMigrationEvidenceMismatchError,
)

from .integrity import ManifestSigningKeyProvider
from .lifecycle_authority import EntryExpectation
from .migration_authority import (
    ActivationReceipt,
    AuthorityIdentitySnapshot,
    VerifiedCandidateReceipt,
)
from .read_contract import BlobReceipt


_EVIDENCE_VERSION = 2
_EVIDENCE_DOMAIN = b"cacheness-maintenance-evidence-v2\x00"
_MAX_EVIDENCE_BYTES = 65_536
_MAX_TEXT_BYTES = 512
_MAX_CANONICAL_DEPTH = 16
_MAX_CANONICAL_NODES = 16_384
_MAX_CANONICAL_ITEMS = 4_096
_MAX_COMPLETED_STEPS = 16
_MAX_CANDIDATE_BATCH_REFERENCES = 256
_MAX_CLEANUP_DEBT_REFERENCES = _MAX_CANDIDATE_BATCH_REFERENCES
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON object keys before a record can be reinterpreted."""
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
    """Apply structural bounds before callers construct semantic evidence values."""
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
    _validate_canonical_value(record, depth=1, nodes=[0], max_text_bytes=max_text_bytes)
    return record


class MaintenanceEvidenceState(str, Enum):
    """Explicit bounded states of one offline maintenance workflow."""

    INSPECTED = "inspected"
    PLANNED = "planned"
    STAGING = "staging"
    STAGED = "staged"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    REBUILDING = "rebuilding"
    REBUILD_STAGED = "rebuild_staged"
    REBUILD_VERIFYING = "rebuild_verifying"
    REBUILD_VERIFIED = "rebuild_verified"
    REBUILD_ACCEPTED = "rebuild_accepted"
    ACTIVATED = "activated"
    ROLLED_BACK = "rolled_back"
    FINALIZED = "finalized"
    PURGE_PENDING = "purge_pending"
    PURGED = "purged"
    ABORTED = "aborted"


_LEGAL_TRANSITIONS: Mapping[MaintenanceEvidenceState, frozenset[MaintenanceEvidenceState]] = {
    MaintenanceEvidenceState.INSPECTED: frozenset(
        {MaintenanceEvidenceState.PLANNED, MaintenanceEvidenceState.ABORTED}
    ),
    MaintenanceEvidenceState.PLANNED: frozenset(
        {
            MaintenanceEvidenceState.STAGING,
            MaintenanceEvidenceState.REBUILDING,
            MaintenanceEvidenceState.ABORTED,
        }
    ),
    MaintenanceEvidenceState.STAGING: frozenset(
        {
            MaintenanceEvidenceState.STAGING,
            MaintenanceEvidenceState.STAGED,
            MaintenanceEvidenceState.ABORTED,
        }
    ),
    MaintenanceEvidenceState.STAGED: frozenset(
        {
            MaintenanceEvidenceState.STAGING,
            MaintenanceEvidenceState.VERIFYING,
            MaintenanceEvidenceState.ABORTED,
        }
    ),
    MaintenanceEvidenceState.VERIFYING: frozenset(
        {
            MaintenanceEvidenceState.STAGING,
            MaintenanceEvidenceState.VERIFIED,
            MaintenanceEvidenceState.ABORTED,
        }
    ),
    MaintenanceEvidenceState.VERIFIED: frozenset(
        {
            MaintenanceEvidenceState.STAGING,
            MaintenanceEvidenceState.ACTIVATED,
            MaintenanceEvidenceState.ABORTED,
        }
    ),
    MaintenanceEvidenceState.REBUILDING: frozenset(
        {
            MaintenanceEvidenceState.REBUILDING,
            MaintenanceEvidenceState.REBUILD_STAGED,
            MaintenanceEvidenceState.ABORTED,
        }
    ),
    MaintenanceEvidenceState.REBUILD_STAGED: frozenset(
        {
            MaintenanceEvidenceState.REBUILD_STAGED,
            MaintenanceEvidenceState.REBUILD_VERIFYING,
            MaintenanceEvidenceState.ABORTED,
        }
    ),
    MaintenanceEvidenceState.REBUILD_VERIFYING: frozenset(
        {
            MaintenanceEvidenceState.REBUILD_VERIFYING,
            MaintenanceEvidenceState.REBUILD_VERIFIED,
            MaintenanceEvidenceState.ABORTED,
        }
    ),
    MaintenanceEvidenceState.REBUILD_VERIFIED: frozenset(
        {
            MaintenanceEvidenceState.REBUILD_VERIFIED,
            MaintenanceEvidenceState.REBUILD_ACCEPTED,
            MaintenanceEvidenceState.ABORTED,
        }
    ),
    MaintenanceEvidenceState.REBUILD_ACCEPTED: frozenset(),
    MaintenanceEvidenceState.ACTIVATED: frozenset(
        {MaintenanceEvidenceState.ROLLED_BACK, MaintenanceEvidenceState.FINALIZED}
    ),
    MaintenanceEvidenceState.FINALIZED: frozenset({MaintenanceEvidenceState.PURGE_PENDING}),
    MaintenanceEvidenceState.PURGE_PENDING: frozenset(
        {MaintenanceEvidenceState.PURGE_PENDING, MaintenanceEvidenceState.PURGED}
    ),
    MaintenanceEvidenceState.ROLLED_BACK: frozenset(),
    MaintenanceEvidenceState.PURGED: frozenset(),
    MaintenanceEvidenceState.ABORTED: frozenset(),
}


def _bounded_text(value: object, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{field_name} must be a bounded string")
    if len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ValueError(f"{field_name} exceeds the byte bound")
    return value


def _sha256(value: object, field_name: str, *, allow_empty: bool = False) -> str:
    value = _bounded_text(value, field_name, allow_empty=allow_empty)
    if not value and allow_empty:
        return value
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal value")
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


def _expectation_record(expectation: EntryExpectation) -> dict[str, object]:
    """Encode one exact authority comparison token without a manifest copy."""
    return {
        "generation": expectation.generation,
        "lineage": expectation.lineage,
        "manifest_digest": expectation.manifest_digest,
        "revision": expectation.revision,
    }


def _expectation_from_record(record: object) -> EntryExpectation:
    """Decode one exact authority comparison token."""
    if not isinstance(record, Mapping) or set(record) != {
        "generation",
        "lineage",
        "manifest_digest",
        "revision",
    }:
        raise ValueError("rebuild receipt expectation is invalid")
    return EntryExpectation(
        lineage=record["lineage"],
        revision=record["revision"],
        generation=record["generation"],
        manifest_digest=record["manifest_digest"],
    )


def _blob_receipt_record(receipt: BlobReceipt) -> dict[str, object]:
    """Encode only exact canonical ownership, never payload or key material."""
    if dict(receipt.projections):
        raise ValueError("rebuild receipts must remain projection-free")
    return {
        "catalog_revision": receipt.catalog_revision,
        "expectation": _expectation_record(receipt.expectation),
        "generation": receipt.generation,
        "key": receipt.key,
        "locator": receipt.locator,
        "operation_id": receipt.operation_id,
    }


def _blob_receipt_from_record(record: object) -> BlobReceipt:
    """Decode exact ownership with an intentionally empty derived-outcome map."""
    if not isinstance(record, Mapping) or set(record) != {
        "catalog_revision",
        "expectation",
        "generation",
        "key",
        "locator",
        "operation_id",
    }:
        raise ValueError("rebuild receipt is invalid")
    return BlobReceipt(
        operation_id=record["operation_id"],
        key=record["key"],
        generation=record["generation"],
        locator=record["locator"],
        expectation=_expectation_from_record(record["expectation"]),
        catalog_revision=record["catalog_revision"],
        projections={},
    )


@dataclass(frozen=True)
class StoppedWorkerAcknowledgement:
    """An operator assertion, not global-quiescence proof or an authority lease."""

    run_id: str
    plan_digest: str
    source_identity: AuthorityIdentitySnapshot
    source_revision: int
    acknowledged: bool = True

    def __post_init__(self) -> None:
        if _RUN_ID.fullmatch(_bounded_text(self.run_id, "run_id")) is None:
            raise ValueError("run_id must be an opaque maintenance identifier")
        _sha256(self.plan_digest, "plan_digest", allow_empty=True)
        if self.source_revision != self.source_identity.revision:
            raise ValueError("acknowledgement source revision must match its identity")
        if self.acknowledged is not True:
            raise ValueError("stopped-worker acknowledgement must be explicit")

    def to_record(self) -> dict[str, object]:
        return {
            "acknowledged": self.acknowledged,
            "plan_digest": self.plan_digest,
            "run_id": self.run_id,
            "source_identity": _identity_record(self.source_identity),
            "source_revision": self.source_revision,
        }

    @classmethod
    def from_record(cls, record: object) -> "StoppedWorkerAcknowledgement":
        if not isinstance(record, Mapping) or set(record) != {
            "acknowledged",
            "plan_digest",
            "run_id",
            "source_identity",
            "source_revision",
        }:
            raise ValueError("stopped-worker acknowledgement is invalid")
        return cls(
            run_id=record["run_id"],
            plan_digest=record["plan_digest"],
            source_identity=_identity_from_record(record["source_identity"], "source_identity"),
            source_revision=record["source_revision"],
            acknowledged=record["acknowledged"],
        )


@dataclass(frozen=True)
class RebuildReceiptBatch:
    """One bounded, authenticated prefix of exact canonical rebuild receipts.

    The batch is corroborative maintenance evidence only.  The destination
    lifecycle authority remains the sole source of visibility and operation
    replay; this record lets an interrupted offline run prove which exact
    authority results it may verify or retire.
    """

    batch_ordinal: int
    first_entry_ordinal: int
    past_last_entry_ordinal: int
    receipts: tuple[BlobReceipt, ...]
    batch_digest: str
    byte_count: int

    @staticmethod
    def digest_for(
        *,
        batch_ordinal: int,
        first_entry_ordinal: int,
        past_last_entry_ordinal: int,
        receipts: tuple[BlobReceipt, ...],
        byte_count: int,
    ) -> str:
        """Derive a stable public digest from ownership fields alone."""
        record = {
            "batch_ordinal": batch_ordinal,
            "byte_count": byte_count,
            "first_entry_ordinal": first_entry_ordinal,
            "past_last_entry_ordinal": past_last_entry_ordinal,
            "receipts": [_blob_receipt_record(receipt) for receipt in receipts],
        }
        return hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        batch_ordinal: int,
        first_entry_ordinal: int,
        receipts: tuple[BlobReceipt, ...],
        byte_count: int,
    ) -> "RebuildReceiptBatch":
        """Construct a single authenticated receipt batch before checkpointing."""
        past_last_entry_ordinal = first_entry_ordinal + len(receipts)
        return cls(
            batch_ordinal=batch_ordinal,
            first_entry_ordinal=first_entry_ordinal,
            past_last_entry_ordinal=past_last_entry_ordinal,
            receipts=receipts,
            batch_digest=cls.digest_for(
                batch_ordinal=batch_ordinal,
                first_entry_ordinal=first_entry_ordinal,
                past_last_entry_ordinal=past_last_entry_ordinal,
                receipts=receipts,
                byte_count=byte_count,
            ),
            byte_count=byte_count,
        )

    def __post_init__(self) -> None:
        if type(self.batch_ordinal) is not int or self.batch_ordinal < 0:
            raise ValueError("rebuild batch ordinal must be non-negative")
        if type(self.first_entry_ordinal) is not int or self.first_entry_ordinal < 0:
            raise ValueError("rebuild first entry ordinal must be non-negative")
        if (
            type(self.past_last_entry_ordinal) is not int
            or self.past_last_entry_ordinal <= self.first_entry_ordinal
        ):
            raise ValueError("rebuild batch entry range is invalid")
        if (
            not isinstance(self.receipts, tuple)
            or not self.receipts
            or len(self.receipts) > _MAX_CANDIDATE_BATCH_REFERENCES
            or not all(isinstance(receipt, BlobReceipt) for receipt in self.receipts)
        ):
            raise ValueError("rebuild receipt batch is invalid")
        if self.past_last_entry_ordinal - self.first_entry_ordinal != len(self.receipts):
            raise ValueError("rebuild receipt range disagrees with receipts")
        operation_ids = tuple(receipt.operation_id for receipt in self.receipts)
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("rebuild receipt batch operation identifiers must be unique")
        if any(dict(receipt.projections) for receipt in self.receipts):
            raise ValueError("rebuild receipt batches must remain projection-free")
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise ValueError("rebuild receipt byte_count must be non-negative")
        expected_digest = self.digest_for(
            batch_ordinal=self.batch_ordinal,
            first_entry_ordinal=self.first_entry_ordinal,
            past_last_entry_ordinal=self.past_last_entry_ordinal,
            receipts=self.receipts,
            byte_count=self.byte_count,
        )
        if not hmac.compare_digest(_sha256(self.batch_digest, "rebuild batch digest"), expected_digest):
            raise ValueError("rebuild receipt batch digest is invalid")

    def to_record(self) -> dict[str, object]:
        """Return canonical, payload-free evidence fields."""
        return {
            "batch_digest": self.batch_digest,
            "batch_ordinal": self.batch_ordinal,
            "byte_count": self.byte_count,
            "first_entry_ordinal": self.first_entry_ordinal,
            "past_last_entry_ordinal": self.past_last_entry_ordinal,
            "receipts": [_blob_receipt_record(receipt) for receipt in self.receipts],
        }

    @classmethod
    def from_record(cls, record: object) -> "RebuildReceiptBatch":
        """Decode one bounded receipt batch after outer authentication."""
        if not isinstance(record, Mapping) or set(record) != {
            "batch_digest",
            "batch_ordinal",
            "byte_count",
            "first_entry_ordinal",
            "past_last_entry_ordinal",
            "receipts",
        } or not isinstance(record["receipts"], list):
            raise ValueError("rebuild receipt batch record is invalid")
        return cls(
            batch_ordinal=record["batch_ordinal"],
            first_entry_ordinal=record["first_entry_ordinal"],
            past_last_entry_ordinal=record["past_last_entry_ordinal"],
            receipts=tuple(_blob_receipt_from_record(value) for value in record["receipts"]),
            batch_digest=record["batch_digest"],
            byte_count=record["byte_count"],
        )


@dataclass(frozen=True)
class MaintenanceRunEvidence:
    """Versioned signed maintenance state with no serialized key material."""

    evidence_version: int
    run_id: str
    plan_digest: str
    source_identity: AuthorityIdentitySnapshot
    source_revision: int
    destination_identity: AuthorityIdentitySnapshot
    destination_revision: int
    acknowledgement: StoppedWorkerAcknowledgement
    state: MaintenanceEvidenceState
    completed_steps: tuple[str, ...]
    candidate_receipt: VerifiedCandidateReceipt | None = None
    candidate_batch_references: tuple[str, ...] = ()
    candidate_checkpoint_revision: int | None = None
    candidate_entry_count: int = 0
    candidate_byte_count: int = 0
    activation_receipt: ActivationReceipt | None = None
    authority_receipts: tuple[str, ...] = ()
    cleanup_debt: tuple[str, ...] = ()
    rebuild_receipt_batches: tuple[RebuildReceiptBatch, ...] = ()
    retired_rebuild_operation_ids: tuple[str, ...] = ()
    completed_output_digests: Mapping[str, str] = field(default_factory=dict)
    signing_key_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.evidence_version != _EVIDENCE_VERSION:
            raise ValueError("unsupported maintenance evidence version")
        if _RUN_ID.fullmatch(_bounded_text(self.run_id, "run_id")) is None:
            raise ValueError("run_id must be an opaque maintenance identifier")
        _sha256(self.plan_digest, "plan_digest", allow_empty=True)
        if self.source_revision != self.source_identity.revision:
            raise ValueError("source_revision must match source_identity")
        if self.destination_revision != self.destination_identity.revision:
            raise ValueError("destination_revision must match destination_identity")
        if not isinstance(self.acknowledgement, StoppedWorkerAcknowledgement):
            raise ValueError("maintenance evidence requires a stopped-worker acknowledgement")
        if (
            self.acknowledgement.run_id != self.run_id
            or self.acknowledgement.plan_digest != self.plan_digest
            or self.acknowledgement.source_identity != self.source_identity
            or self.acknowledgement.source_revision != self.source_revision
        ):
            raise ValueError("stopped-worker acknowledgement does not bind this evidence")
        if not isinstance(self.state, MaintenanceEvidenceState):
            raise ValueError("state must be a maintenance evidence state")
        if (
            not isinstance(self.completed_steps, tuple)
            or not self.completed_steps
            or len(self.completed_steps) > _MAX_COMPLETED_STEPS
            or len(set(self.completed_steps)) != len(self.completed_steps)
        ):
            raise ValueError("completed_steps must be an ordered bounded tuple")
        for step in self.completed_steps:
            _bounded_text(step, "completed_step")
        if self.candidate_receipt is not None:
            if (
                self.candidate_receipt.run_id != self.run_id
                or self.candidate_receipt.plan_digest != self.plan_digest
                or self.candidate_receipt.source_identity != self.source_identity
                or self.candidate_receipt.source_revision != self.source_revision
                or self.candidate_receipt.destination_identity != self.destination_identity
                or self.candidate_receipt.destination_revision != self.destination_revision
            ):
                raise ValueError("candidate_receipt does not bind this evidence")
        if (
            not isinstance(self.candidate_batch_references, tuple)
            or len(self.candidate_batch_references) > _MAX_CANDIDATE_BATCH_REFERENCES
        ):
            raise ValueError("candidate_batch_references must be a bounded tuple")
        if len(set(self.candidate_batch_references)) != len(self.candidate_batch_references):
            raise ValueError("candidate_batch_references must be unique")
        for reference in self.candidate_batch_references:
            _sha256(reference, "candidate batch reference")
        if self.candidate_checkpoint_revision is not None and (
            type(self.candidate_checkpoint_revision) is not int
            or self.candidate_checkpoint_revision < 0
        ):
            raise ValueError("candidate_checkpoint_revision must be a non-negative integer")
        for field_name in ("candidate_entry_count", "candidate_byte_count"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.candidate_batch_references and (
            self.candidate_checkpoint_revision is None
            or self.candidate_entry_count == 0
        ):
            raise ValueError("candidate batch references require attributed progress")
        if self.candidate_receipt is not None and (
            self.candidate_receipt.entry_count != self.candidate_entry_count
            or self.candidate_receipt.byte_count != self.candidate_byte_count
        ):
            raise ValueError("candidate receipt disagrees with attributed progress")
        if self.activation_receipt is not None and self.candidate_receipt is None:
            raise ValueError("activation_receipt requires candidate_receipt")
        if (
            not isinstance(self.authority_receipts, tuple)
            or len(self.authority_receipts) > _MAX_COMPLETED_STEPS
        ):
            raise ValueError("authority_receipts must be a bounded tuple")
        if (
            not isinstance(self.cleanup_debt, tuple)
            or len(self.cleanup_debt) > _MAX_CLEANUP_DEBT_REFERENCES
        ):
            raise ValueError("cleanup_debt must be a bounded tuple")
        for value in self.authority_receipts:
            _sha256(value, "maintenance receipt")
        for value in self.cleanup_debt:
            if value.startswith("rebuild:"):
                parts = value.split(":", 4)
                if len(parts) != 5 or any(not item for item in parts):
                    raise ValueError("rebuild cleanup debt is invalid")
                _bounded_text(value, "rebuild cleanup debt")
            else:
                _sha256(value, "maintenance receipt")
        if (
            not isinstance(self.rebuild_receipt_batches, tuple)
            or len(self.rebuild_receipt_batches) > _MAX_CANDIDATE_BATCH_REFERENCES
            or not all(
                isinstance(batch, RebuildReceiptBatch)
                for batch in self.rebuild_receipt_batches
            )
        ):
            raise ValueError("rebuild_receipt_batches must be a bounded tuple")
        expected_entry_ordinal = 0
        receipt_operation_ids: set[str] = set()
        for batch_ordinal, batch in enumerate(self.rebuild_receipt_batches):
            if (
                batch.batch_ordinal != batch_ordinal
                or batch.first_entry_ordinal != expected_entry_ordinal
            ):
                raise ValueError("rebuild receipt batches must form one ordered prefix")
            expected_entry_ordinal = batch.past_last_entry_ordinal
            for receipt in batch.receipts:
                if receipt.operation_id in receipt_operation_ids:
                    raise ValueError("rebuild operation identifiers must be unique")
                receipt_operation_ids.add(receipt.operation_id)
        if (
            not isinstance(self.retired_rebuild_operation_ids, tuple)
            or len(self.retired_rebuild_operation_ids) > _MAX_CANDIDATE_BATCH_REFERENCES
            or len(set(self.retired_rebuild_operation_ids))
            != len(self.retired_rebuild_operation_ids)
        ):
            raise ValueError("retired rebuild operation identifiers are invalid")
        for operation_id in self.retired_rebuild_operation_ids:
            _bounded_text(operation_id, "retired rebuild operation identifier")
            if operation_id not in receipt_operation_ids:
                raise ValueError("retired rebuild operation lacks an exact receipt")
        if self.state is MaintenanceEvidenceState.ABORTED and receipt_operation_ids:
            if self.cleanup_debt:
                raise ValueError("ABORTED rebuild evidence cannot retain cleanup debt")
            if set(self.retired_rebuild_operation_ids) != receipt_operation_ids:
                raise ValueError(
                    "ABORTED rebuild evidence requires every exact receipt retirement"
                )
        if not isinstance(self.completed_output_digests, Mapping):
            raise ValueError("completed_output_digests must be a mapping")
        if set(self.completed_output_digests) - set(self.completed_steps):
            raise ValueError("completed output digest lacks a completed step")
        output_digests = dict(self.completed_output_digests)
        for step, digest in output_digests.items():
            _bounded_text(step, "completed output step")
            _sha256(digest, "completed output digest")
        object.__setattr__(self, "completed_output_digests", output_digests)
        _sha256(self.signing_key_fingerprint, "signing_key_fingerprint", allow_empty=True)

    def with_signing_key_fingerprint(self, fingerprint: str) -> "MaintenanceRunEvidence":
        """Bind a non-secret provider identity before persistence."""
        _sha256(fingerprint, "signing_key_fingerprint")
        return replace(self, signing_key_fingerprint=fingerprint)

    def to_record(self) -> dict[str, object]:
        """Return complete canonical fields while excluding secret key bytes."""
        record: dict[str, object] = {
            "acknowledgement": self.acknowledgement.to_record(),
            "activation_receipt": None,
            "authority_receipts": list(self.authority_receipts),
            "candidate_batch_references": list(self.candidate_batch_references),
            "candidate_byte_count": self.candidate_byte_count,
            "candidate_checkpoint_revision": self.candidate_checkpoint_revision,
            "candidate_entry_count": self.candidate_entry_count,
            "candidate_receipt": None,
            "cleanup_debt": list(self.cleanup_debt),
            "completed_output_digests": dict(sorted(self.completed_output_digests.items())),
            "completed_steps": list(self.completed_steps),
            "destination_identity": _identity_record(self.destination_identity),
            "destination_revision": self.destination_revision,
            "evidence_version": self.evidence_version,
            "plan_digest": self.plan_digest,
            "rebuild_receipt_batches": [
                batch.to_record() for batch in self.rebuild_receipt_batches
            ],
            "retired_rebuild_operation_ids": list(self.retired_rebuild_operation_ids),
            "run_id": self.run_id,
            "signing_key_fingerprint": self.signing_key_fingerprint,
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
        """Decode exact evidence fields after outer-envelope authentication."""
        required = {
            "acknowledgement",
            "activation_receipt",
            "authority_receipts",
            "candidate_batch_references",
            "candidate_byte_count",
            "candidate_checkpoint_revision",
            "candidate_entry_count",
            "candidate_receipt",
            "cleanup_debt",
            "completed_output_digests",
            "completed_steps",
            "destination_identity",
            "destination_revision",
            "evidence_version",
            "plan_digest",
            "rebuild_receipt_batches",
            "retired_rebuild_operation_ids",
            "run_id",
            "signing_key_fingerprint",
            "source_identity",
            "source_revision",
            "state",
        }
        if not isinstance(record, Mapping) or set(record) != required:
            raise ValueError("maintenance evidence fields are invalid")
        steps = record["completed_steps"]
        output_digests = record["completed_output_digests"]
        if (
            not isinstance(steps, list)
            or not isinstance(output_digests, Mapping)
            or not isinstance(record["authority_receipts"], list)
            or not isinstance(record["candidate_batch_references"], list)
            or not isinstance(record["cleanup_debt"], list)
            or not isinstance(record["rebuild_receipt_batches"], list)
            or not isinstance(record["retired_rebuild_operation_ids"], list)
        ):
            raise ValueError("maintenance evidence completion records are invalid")
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
            destination_revision=record["destination_revision"],
            acknowledgement=StoppedWorkerAcknowledgement.from_record(record["acknowledgement"]),
            state=state,
            completed_steps=tuple(steps),
            candidate_receipt=candidate_receipt,
            candidate_batch_references=tuple(record["candidate_batch_references"]),
            candidate_checkpoint_revision=record["candidate_checkpoint_revision"],
            candidate_entry_count=record["candidate_entry_count"],
            candidate_byte_count=record["candidate_byte_count"],
            activation_receipt=activation_receipt,
            authority_receipts=tuple(record["authority_receipts"]),
            cleanup_debt=tuple(record["cleanup_debt"]),
            rebuild_receipt_batches=tuple(
                RebuildReceiptBatch.from_record(value)
                for value in record["rebuild_receipt_batches"]
            ),
            retired_rebuild_operation_ids=tuple(record["retired_rebuild_operation_ids"]),
            completed_output_digests=output_digests,
            signing_key_fingerprint=record["signing_key_fingerprint"],
        )

    def render_report(self) -> str:
        """Render only validated public fields for an operator report."""
        return "\n".join(
            (
                f"Maintenance run: {self.run_id}",
                f"State: {self.state.value}",
                f"Plan digest: {self.plan_digest or 'not-yet-planned'}",
                f"Source: {self.source_identity.store_id}@{self.source_revision}",
                f"Destination: {self.destination_identity.store_id}@{self.destination_revision}",
                f"Signing key fingerprint: {self.signing_key_fingerprint or 'not-recorded'}",
            )
        )


def _canonical_bytes(record: Mapping[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encode_maintenance_evidence(evidence: MaintenanceRunEvidence, signing_key: bytes) -> bytes:
    """Authenticate a bounded evidence envelope using already-authorized bytes."""
    if not isinstance(evidence, MaintenanceRunEvidence):
        raise TypeError("evidence must be a MaintenanceRunEvidence")
    if type(signing_key) is not bytes or len(signing_key) != 32:
        raise ValueError("maintenance evidence requires a 32-byte signing key")
    unsigned = evidence.to_record()
    signature = hmac.new(
        signing_key, _EVIDENCE_DOMAIN + _canonical_bytes(unsigned), "sha256"
    ).hexdigest()
    encoded = _canonical_bytes({"evidence": unsigned, "signature": signature})
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
    expected = hmac.new(
        signing_key, _EVIDENCE_DOMAIN + _canonical_bytes(evidence), "sha256"
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("maintenance evidence authentication failed")
    if raw != _canonical_bytes({"evidence": evidence, "signature": signature}):
        raise ValueError("maintenance evidence is not canonical")
    return MaintenanceRunEvidence.from_record(evidence)


class MaintenanceEvidenceStore:
    """Contain and conditionally replace one run's authenticated evidence file.

    No discovery API exists: callers supply a work directory and run ID, and
    only the exact derived path for that pair can be read or replaced.
    """

    def __init__(
        self,
        work_directory: str | Path,
        run_id: str,
        key_provider: ManifestSigningKeyProvider,
    ) -> None:
        if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
            raise CacheBlobMigrationEvidenceError(
                "maintenance evidence run_id is invalid",
                context={"operation": "maintenance_evidence.configure"},
            )
        if not isinstance(key_provider, ManifestSigningKeyProvider):
            raise TypeError("key_provider must provide read-only get_key()")
        self._supplied_work_directory = Path(work_directory)
        self.run_id = run_id
        self._key_provider = key_provider
        self._work_directory = self._resolve_work_directory(create=False)

    @property
    def work_directory(self) -> Path:
        """Return the validated resolved work root."""
        return self._work_directory

    @property
    def evidence_path(self) -> Path:
        """Return the only allowed evidence path for this exact run."""
        return self._work_directory / f"{self.run_id}.maintenance.json"

    def _resolve_work_directory(self, *, create: bool) -> Path:
        supplied = self._supplied_work_directory
        absolute = supplied if supplied.is_absolute() else Path.cwd() / supplied
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current /= part
            if current.exists() and current.is_symlink():
                raise CacheBlobMigrationEvidenceError(
                    "maintenance evidence work directory cannot traverse a symlink",
                    context={"operation": "maintenance_evidence.path"},
                )
        if create:
            try:
                absolute.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise CacheBlobMigrationEvidenceError(
                    "maintenance evidence work directory cannot be created",
                    context={"operation": "maintenance_evidence.path"},
                ) from exc
            current = Path(absolute.anchor)
            for part in absolute.parts[1:]:
                current /= part
                if current.is_symlink():
                    raise CacheBlobMigrationEvidenceError(
                        "maintenance evidence work directory cannot traverse a symlink",
                        context={"operation": "maintenance_evidence.path"},
                    )
        elif not absolute.exists():
            return absolute.resolve(strict=False)
        return absolute.resolve(strict=True)

    def _assert_evidence_path(self) -> None:
        path = self.evidence_path
        if path.parent != self._work_directory or path.name != f"{self.run_id}.maintenance.json":
            raise CacheBlobMigrationEvidenceError(
                "maintenance evidence path is outside its exact work directory",
                context={"operation": "maintenance_evidence.path"},
            )
        if path.exists() and path.is_symlink():
            raise CacheBlobMigrationEvidenceError(
                "maintenance evidence path cannot be a symlink",
                context={"operation": "maintenance_evidence.path"},
            )

    def _signing_key(self) -> bytes:
        try:
            key = self._key_provider.get_key()
        except Exception as exc:
            raise CacheBlobMigrationEvidenceError(
                "maintenance evidence signing identity is unavailable",
                context={"operation": "maintenance_evidence.sign"},
            ) from exc
        if type(key) is not bytes or len(key) != 32:
            raise CacheBlobMigrationEvidenceError(
                "maintenance evidence signing identity is invalid",
                context={"operation": "maintenance_evidence.sign"},
            )
        return key

    def signing_key_fingerprint(self) -> str:
        """Expose a non-secret identity marker for signed evidence diagnostics."""
        return hashlib.sha256(self._signing_key()).hexdigest()

    def _bind_signing_identity(self, evidence: MaintenanceRunEvidence) -> MaintenanceRunEvidence:
        if not isinstance(evidence, MaintenanceRunEvidence) or evidence.run_id != self.run_id:
            raise CacheBlobMigrationEvidenceMismatchError(
                "maintenance evidence does not bind the configured run_id",
                context={"operation": "maintenance_evidence.bind", "run_id": self.run_id},
            )
        fingerprint = self.signing_key_fingerprint()
        if evidence.signing_key_fingerprint and not hmac.compare_digest(
            evidence.signing_key_fingerprint, fingerprint
        ):
            raise CacheBlobMigrationEvidenceMismatchError(
                "maintenance evidence signing identity does not match",
                context={"operation": "maintenance_evidence.bind", "run_id": self.run_id},
            )
        return evidence.with_signing_key_fingerprint(fingerprint)

    def _atomic_write(self, raw: bytes) -> None:
        self._work_directory = self._resolve_work_directory(create=True)
        self._assert_evidence_path()
        descriptor = None
        temporary_name = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".maintenance-evidence-", suffix=".tmp", dir=self._work_directory
            )
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            self._assert_evidence_path()
            os.replace(temporary_name, self.evidence_path)
            temporary_name = None
        except OSError as exc:
            raise CacheBlobMigrationEvidenceError(
                "maintenance evidence cannot be written",
                context={"operation": "maintenance_evidence.write", "run_id": self.run_id},
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass

    def read_bytes(self) -> bytes:
        """Read the exact contained bytes without interpreting a candidate."""
        self._work_directory = self._resolve_work_directory(create=False)
        self._assert_evidence_path()
        try:
            return self.evidence_path.read_bytes()
        except OSError as exc:
            raise CacheBlobMigrationEvidenceError(
                "maintenance evidence is required for this explicit run",
                context={"operation": "maintenance_evidence.read", "run_id": self.run_id},
            ) from exc

    def create(self, evidence: MaintenanceRunEvidence) -> MaintenanceRunEvidence:
        """Create first evidence only after binding its signing identity."""
        self._work_directory = self._resolve_work_directory(create=True)
        self._assert_evidence_path()
        if self.evidence_path.exists():
            raise CacheBlobMigrationEvidenceMismatchError(
                "maintenance evidence already exists for this run",
                context={"operation": "maintenance_evidence.create", "run_id": self.run_id},
            )
        if evidence.state is not MaintenanceEvidenceState.INSPECTED:
            raise CacheBlobMigrationEvidenceMismatchError(
                "maintenance evidence must begin at the inspection state",
                context={"operation": "maintenance_evidence.create", "run_id": self.run_id},
            )
        bound = self._bind_signing_identity(evidence)
        try:
            raw = encode_maintenance_evidence(bound, self._signing_key())
        except (TypeError, ValueError) as exc:
            raise CacheBlobMigrationEvidenceError(
                "maintenance evidence cannot be authenticated",
                context={"operation": "maintenance_evidence.create", "run_id": self.run_id},
            ) from exc
        self._atomic_write(raw)
        return self.load()

    def load(self) -> MaintenanceRunEvidence:
        """Authenticate and validate only this run's exact contained evidence."""
        raw = self.read_bytes()
        try:
            evidence = decode_maintenance_evidence(raw, self._signing_key())
        except (TypeError, ValueError) as exc:
            raise CacheBlobMigrationEvidenceError(
                "maintenance evidence authentication failed",
                context={"operation": "maintenance_evidence.load", "run_id": self.run_id},
            ) from exc
        bound = self._bind_signing_identity(evidence)
        if evidence != bound:
            raise CacheBlobMigrationEvidenceMismatchError(
                "maintenance evidence signing identity is incomplete",
                context={"operation": "maintenance_evidence.load", "run_id": self.run_id},
            )
        return evidence

    def checkpoint(
        self, expected_previous_bytes: bytes, evidence: MaintenanceRunEvidence
    ) -> MaintenanceRunEvidence:
        """Replace only from exact previous bytes and a legal next state."""
        if not isinstance(expected_previous_bytes, bytes):
            raise TypeError("expected_previous_bytes must be bytes")
        current = self.read_bytes()
        if not hmac.compare_digest(current, expected_previous_bytes):
            raise CacheBlobMigrationEvidenceMismatchError(
                "maintenance evidence no longer has the exact previous bytes",
                context={"operation": "maintenance_evidence.checkpoint", "run_id": self.run_id},
            )
        previous = self.load()
        next_evidence = self._bind_signing_identity(evidence)
        if next_evidence.state not in _LEGAL_TRANSITIONS[previous.state]:
            raise CacheBlobMigrationEvidenceMismatchError(
                "maintenance evidence state transition is invalid",
                context={"operation": "maintenance_evidence.checkpoint", "run_id": self.run_id},
            )
        immutable_fields = (
            "run_id",
            "source_identity",
            "source_revision",
            "destination_identity",
            "destination_revision",
            "signing_key_fingerprint",
        )
        if any(
            getattr(previous, name) != getattr(next_evidence, name)
            for name in immutable_fields
        ):
            raise CacheBlobMigrationEvidenceMismatchError(
                "maintenance evidence checkpoint changes immutable run binding",
                context={"operation": "maintenance_evidence.checkpoint", "run_id": self.run_id},
            )
        establishing_plan = (
            previous.state is MaintenanceEvidenceState.INSPECTED
            and next_evidence.state is MaintenanceEvidenceState.PLANNED
            and previous.plan_digest == ""
            and next_evidence.plan_digest
        )
        if not establishing_plan and (
            previous.plan_digest != next_evidence.plan_digest
            or previous.acknowledgement != next_evidence.acknowledgement
        ):
            raise CacheBlobMigrationEvidenceMismatchError(
                "maintenance evidence checkpoint changes immutable plan binding",
                context={"operation": "maintenance_evidence.checkpoint", "run_id": self.run_id},
            )
        try:
            raw = encode_maintenance_evidence(next_evidence, self._signing_key())
        except (TypeError, ValueError) as exc:
            raise CacheBlobMigrationEvidenceError(
                "maintenance evidence cannot be authenticated",
                context={"operation": "maintenance_evidence.checkpoint", "run_id": self.run_id},
            ) from exc
        self._atomic_write(raw)
        return self.load()


__all__ = [
    "MaintenanceEvidenceState",
    "MaintenanceEvidenceStore",
    "MaintenanceRunEvidence",
    "StoppedWorkerAcknowledgement",
    "decode_bounded_canonical_json",
    "decode_maintenance_evidence",
    "encode_maintenance_evidence",
]
