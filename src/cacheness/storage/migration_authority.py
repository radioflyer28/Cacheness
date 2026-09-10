"""Narrow authority contracts for explicit offline whole-store cutover.

The maintenance protocol deliberately does not extend ``LifecycleAuthority``.
Ordinary lifecycle operations retain their entry-level interface, while a
topology that elects to support offline cutover implements this small,
whole-store-only surface.  Candidate receipts are corroborating evidence for
the selected authority; they never become visibility authority themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Protocol, runtime_checkable

from .lifecycle_authority import EntrySnapshot

_MAX_TEXT_BYTES = 512
_MAX_CANDIDATE_ENTRIES = 256
_MAX_INVENTORY_PAGE_ENTRIES = 256
_MAX_INVENTORY_WORK_BYTES = 131_072
SQLITE_MIGRATION_AUTHORITY_SCHEMA_VERSION = 8
POSTGRESQL_MIGRATION_AUTHORITY_SCHEMA_VERSION = 4
POSTGRESQL_MIGRATION_AUTHORITY_CAPABILITY = "postgresql-lifecycle-authority-v4"


class AuthorityPublicationState(str, Enum):
    """The only authority-owned states for an offline whole-store cutover."""

    IDLE = "idle"
    CANDIDATE = "candidate"
    ACTIVATED_OFFLINE = "activated_offline"
    ACTIVE = "active"
    ROLLED_BACK = "rolled_back"


def _bounded_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ValueError(f"{field_name} must be a non-empty bounded string")
    return value


def _digest_bytes(value: bytes, field_name: str) -> str:
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{field_name} must be non-empty bytes")
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class AuthorityIdentitySnapshot:
    """Bounded identity and revision from the one canonical authority."""

    store_id: str
    revision: int
    authority_kind: str
    capability: str = "unknown"
    schema_version: int = 0

    def __post_init__(self) -> None:
        _bounded_text(self.store_id, "store_id")
        _bounded_text(self.authority_kind, "authority_kind")
        _bounded_text(self.capability, "capability")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision must be a non-negative integer")
        if type(self.schema_version) is not int or self.schema_version < 0:
            raise ValueError("schema_version must be a non-negative integer")


@dataclass(frozen=True)
class AuthorityInventoryCursor:
    """Opaque-to-callers continuation bound to one authority revision."""

    store_id: str
    revision: int
    last_key: str | None = None
    last_generation: str | None = None

    def __post_init__(self) -> None:
        _bounded_text(self.store_id, "store_id")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision must be a non-negative integer")
        if (self.last_key is None) != (self.last_generation is None):
            raise ValueError("inventory cursor key and generation must be supplied together")
        if self.last_key is not None:
            _bounded_text(self.last_key, "last_key")
            _bounded_text(self.last_generation, "last_generation")


def validate_inventory_page_request(
    *,
    limit: int | None,
    work_cap: int | None,
    default_limit: int,
    default_work_cap: int,
) -> tuple[int, int]:
    """Validate caller limits before an authority opens or scans a catalog."""
    effective_limit = default_limit if limit is None else limit
    effective_work_cap = default_work_cap if work_cap is None else work_cap
    if (
        type(effective_limit) is not int
        or not 1 <= effective_limit <= min(default_limit, _MAX_INVENTORY_PAGE_ENTRIES)
    ):
        raise ValueError("inventory page limit is out of bounds")
    if (
        type(effective_work_cap) is not int
        or not 1 <= effective_work_cap <= min(default_work_cap, _MAX_INVENTORY_WORK_BYTES)
    ):
        raise ValueError("inventory page work cap is out of bounds")
    return effective_limit, effective_work_cap


@dataclass(frozen=True)
class AuthorityInventoryEntry:
    """One verified-candidate descriptor, still invisible before activation."""

    key: str
    generation: str
    locator: str
    manifest: bytes
    payload_digest: str
    byte_size: int

    def __post_init__(self) -> None:
        for field_name in ("key", "generation", "locator"):
            _bounded_text(getattr(self, field_name), field_name)
        if len(self.manifest) > 1024 * 1024:
            raise ValueError("manifest exceeds the maintenance bound")
        _digest_bytes(self.manifest, "manifest")
        if (
            not isinstance(self.payload_digest, str)
            or len(self.payload_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.payload_digest)
        ):
            raise ValueError("payload_digest must be a SHA-256 hexadecimal value")
        if type(self.byte_size) is not int or self.byte_size < 0:
            raise ValueError("byte_size must be a non-negative integer")

    @property
    def manifest_digest(self) -> str:
        """Return the authenticated descriptor digest without exposing its bytes."""
        return hashlib.sha256(self.manifest).hexdigest()


@dataclass(frozen=True)
class AuthorityInventoryPage:
    """One raw bounded inventory page from a fixed authority revision.

    The entry descriptors deliberately remain ``EntrySnapshot`` values.  The
    maintenance layer authenticates and classifies manifest bytes after this
    authority read; inventory itself never parses payload or catalog schemas.
    """

    identity: AuthorityIdentitySnapshot
    entries: tuple["EntrySnapshot", ...]
    next_cursor: AuthorityInventoryCursor | None
    exhausted: bool

    def __post_init__(self) -> None:
        if len(self.entries) > _MAX_INVENTORY_PAGE_ENTRIES:
            raise ValueError("maintenance inventory page exceeds the entry bound")
        if not isinstance(self.exhausted, bool):
            raise ValueError("inventory exhausted must be a bool")
        if len({(entry.key, entry.generation) for entry in self.entries}) != len(self.entries):
            raise ValueError("maintenance inventory contains duplicate keys")
        if self.exhausted and self.next_cursor is not None:
            raise ValueError("exhausted inventory page cannot continue")
        if not self.exhausted and self.next_cursor is None:
            raise ValueError("non-exhausted inventory page requires a continuation")
        if self.next_cursor is not None:
            if (
                self.next_cursor.store_id != self.identity.store_id
                or self.next_cursor.revision != self.identity.revision
            ):
                raise ValueError("inventory cursor does not match page identity")
            if not self.entries or (
                self.next_cursor.last_key,
                self.next_cursor.last_generation,
            ) != (self.entries[-1].key, self.entries[-1].generation):
                raise ValueError("inventory cursor does not match the last emitted entry")


def candidate_digest(entries: tuple[AuthorityInventoryEntry, ...]) -> str:
    """Hash a canonical candidate description without serializing secret material."""
    if len(entries) > _MAX_CANDIDATE_ENTRIES:
        raise ValueError("maintenance candidate exceeds the entry bound")
    record = [
        {
            "byte_size": entry.byte_size,
            "generation": entry.generation,
            "key": entry.key,
            "locator": entry.locator,
            "manifest_digest": entry.manifest_digest,
            "payload_digest": entry.payload_digest,
        }
        for entry in sorted(entries, key=lambda item: item.key)
    ]
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class VerifiedCandidateReceipt:
    """Corroborating receipt for a completely staged and verified candidate."""

    run_id: str
    plan_digest: str
    source_identity: AuthorityIdentitySnapshot
    source_revision: int
    destination_identity: AuthorityIdentitySnapshot
    destination_revision: int
    candidate_digest: str
    entry_count: int
    byte_count: int

    def __post_init__(self) -> None:
        for field_name in ("run_id", "plan_digest", "candidate_digest"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or len(value) != 64 and field_name != "run_id":
                raise ValueError(f"{field_name} is invalid")
        _bounded_text(self.run_id, "run_id")
        for field_name in ("plan_digest", "candidate_digest"):
            value = getattr(self, field_name)
            if any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{field_name} must be a SHA-256 hexadecimal value")
        for field_name in ("source_revision", "destination_revision", "entry_count", "byte_count"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True)
class PriorStoreReceipt:
    """Authority evidence that the selected store was retained before activation."""

    run_id: str
    revision: int
    entry_count: int

    def __post_init__(self) -> None:
        _bounded_text(self.run_id, "run_id")
        for field_name in ("revision", "entry_count"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True)
class ActivationReceipt:
    """Authority-issued result of the sole whole-store visibility transition."""

    candidate_receipt: VerifiedCandidateReceipt
    activation_revision: int
    prior_store: PriorStoreReceipt | None = None
    state: AuthorityPublicationState = AuthorityPublicationState.ACTIVATED_OFFLINE

    def __post_init__(self) -> None:
        if type(self.activation_revision) is not int or self.activation_revision < 0:
            raise ValueError("activation_revision must be a non-negative integer")
        if self.state is not AuthorityPublicationState.ACTIVATED_OFFLINE:
            raise ValueError("activation receipt must record activated_offline state")
        if self.prior_store is not None and self.prior_store.run_id != self.candidate_receipt.run_id:
            raise ValueError("prior store receipt must belong to the activated run")


@dataclass(frozen=True)
class RollbackReceipt:
    """Authority result for the one offline return to the retained prior store."""

    run_id: str
    rollback_revision: int
    state: AuthorityPublicationState = AuthorityPublicationState.ROLLED_BACK

    def __post_init__(self) -> None:
        _bounded_text(self.run_id, "run_id")
        if type(self.rollback_revision) is not int or self.rollback_revision < 0:
            raise ValueError("rollback_revision must be a non-negative integer")
        if self.state is not AuthorityPublicationState.ROLLED_BACK:
            raise ValueError("rollback receipt must record rolled_back state")


@dataclass(frozen=True)
class FinalizeReceipt:
    """Authority result that permanently ends rollback eligibility for one run."""

    run_id: str
    finalized_revision: int
    state: AuthorityPublicationState = AuthorityPublicationState.ACTIVE

    def __post_init__(self) -> None:
        _bounded_text(self.run_id, "run_id")
        if type(self.finalized_revision) is not int or self.finalized_revision < 0:
            raise ValueError("finalized_revision must be a non-negative integer")
        if self.state is not AuthorityPublicationState.ACTIVE:
            raise ValueError("finalize receipt must record active state")


@runtime_checkable
class MigrationAuthority(Protocol):
    """Minimal protocol used only by explicit offline migration services."""

    def identity_snapshot(self) -> AuthorityIdentitySnapshot: ...

    def inventory_page(
        self,
        cursor: AuthorityInventoryCursor | None = None,
        *,
        limit: int | None = None,
        work_cap: int | None = None,
    ) -> AuthorityInventoryPage: ...

    def activate_verified_candidate(
        self,
        *,
        receipt: VerifiedCandidateReceipt,
        entries: tuple[AuthorityInventoryEntry, ...],
    ) -> ActivationReceipt: ...

__all__ = [
    "ActivationReceipt",
    "AuthorityPublicationState",
    "AuthorityInventoryCursor",
    "AuthorityIdentitySnapshot",
    "AuthorityInventoryEntry",
    "AuthorityInventoryPage",
    "FinalizeReceipt",
    "MigrationAuthority",
    "POSTGRESQL_MIGRATION_AUTHORITY_CAPABILITY",
    "POSTGRESQL_MIGRATION_AUTHORITY_SCHEMA_VERSION",
    "PriorStoreReceipt",
    "RollbackReceipt",
    "SQLITE_MIGRATION_AUTHORITY_SCHEMA_VERSION",
    "VerifiedCandidateReceipt",
    "candidate_digest",
    "validate_inventory_page_request",
]
