"""Narrow authority contracts for explicit offline whole-store cutover.

The maintenance protocol deliberately does not extend ``LifecycleAuthority``.
Ordinary lifecycle operations retain their entry-level interface, while a
topology that elects to support offline cutover implements this small,
whole-store-only surface.  Candidate receipts are corroborating evidence for
the selected authority; they never become visibility authority themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Protocol, runtime_checkable

_MAX_TEXT_BYTES = 512
_MAX_CANDIDATE_ENTRIES = 256


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

    def __post_init__(self) -> None:
        _bounded_text(self.store_id, "store_id")
        _bounded_text(self.authority_kind, "authority_kind")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision must be a non-negative integer")


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
    """One bounded immutable inventory page from a fixed authority revision."""

    identity: AuthorityIdentitySnapshot
    entries: tuple[AuthorityInventoryEntry, ...]
    total_entries: int
    total_bytes: int

    def __post_init__(self) -> None:
        if len(self.entries) > _MAX_CANDIDATE_ENTRIES:
            raise ValueError("maintenance inventory page exceeds the entry bound")
        if len({entry.key for entry in self.entries}) != len(self.entries):
            raise ValueError("maintenance inventory contains duplicate keys")
        if self.total_entries != len(self.entries):
            raise ValueError("maintenance inventory total_entries disagrees with entries")
        if self.total_bytes != sum(entry.byte_size for entry in self.entries):
            raise ValueError("maintenance inventory total_bytes disagrees with entries")


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
class ActivationReceipt:
    """Authority-issued result of the sole whole-store visibility transition."""

    candidate_receipt: VerifiedCandidateReceipt
    activation_revision: int

    def __post_init__(self) -> None:
        if type(self.activation_revision) is not int or self.activation_revision < 0:
            raise ValueError("activation_revision must be a non-negative integer")


@runtime_checkable
class MigrationAuthority(Protocol):
    """Minimal protocol used only by explicit offline migration services."""

    def migration_identity(self) -> AuthorityIdentitySnapshot: ...

    def migration_inventory(self, *, expected_revision: int | None = None) -> AuthorityInventoryPage: ...

    def activate_verified_candidate(
        self,
        *,
        receipt: VerifiedCandidateReceipt,
        entries: tuple[AuthorityInventoryEntry, ...],
    ) -> ActivationReceipt: ...


__all__ = [
    "ActivationReceipt",
    "AuthorityIdentitySnapshot",
    "AuthorityInventoryEntry",
    "AuthorityInventoryPage",
    "MigrationAuthority",
    "VerifiedCandidateReceipt",
    "candidate_digest",
]
