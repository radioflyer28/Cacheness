"""Backend-neutral transactional authority contracts for BlobStore lifecycle.

The authority owns committed manifest visibility and durable mutation evidence.
Payload serialization, publication, verification, and deletion deliberately remain
outside this seam so a database transaction never spans handler or filesystem I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


_MAX_TEXT_BYTES = 512
_MAX_MANIFEST_BYTES = 1024 * 1024


def _bounded_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ValueError(f"{field_name} must be a non-empty bounded string")
    return value


@dataclass(frozen=True)
class EntryExpectation:
    """Exact entry lineage observed before a conditional transition."""

    lineage: int | None
    revision: int | None

    @classmethod
    def absent(cls) -> "EntryExpectation":
        return cls(lineage=None, revision=None)


@dataclass(frozen=True)
class EntrySnapshot:
    """Immutable committed entry returned by the only visibility authority."""

    key: str
    generation: str
    locator: str
    manifest: bytes
    expectation: EntryExpectation


@dataclass(frozen=True)
class MutationSpec:
    """Bounded request to prepare an immutable candidate transition."""

    operation_id: str
    key: str
    generation: str
    candidate_locator: str
    expected: EntryExpectation
    manifest: bytes = b""

    def __post_init__(self) -> None:
        for field_name in ("operation_id", "key", "generation", "candidate_locator"):
            _bounded_text(getattr(self, field_name), field_name)
        if not isinstance(self.manifest, bytes) or len(self.manifest) > _MAX_MANIFEST_BYTES:
            raise ValueError("manifest must be bounded bytes")

    @classmethod
    def create(
        cls,
        *,
        operation_id: str,
        key: str,
        generation: str,
        candidate_locator: str,
        expected: EntryExpectation,
        manifest: bytes = b"",
    ) -> "MutationSpec":
        return cls(
            operation_id=operation_id,
            key=key,
            generation=generation,
            candidate_locator=candidate_locator,
            expected=expected,
            manifest=manifest,
        )


@dataclass(frozen=True)
class PreparedMutation:
    """Durably prepared intent; it is never a normal-read authority."""

    operation_id: str
    spec: MutationSpec


@dataclass(frozen=True)
class VerificationProof:
    """Digest/size evidence recorded after native candidate verification."""

    digest: str
    byte_size: int
    manifest: bytes = b""

    def __post_init__(self) -> None:
        if len(self.digest) != 64 or any(char not in "0123456789abcdef" for char in self.digest):
            raise ValueError("digest must be a lowercase SHA-256 hexadecimal value")
        if not isinstance(self.byte_size, int) or self.byte_size < 0:
            raise ValueError("byte_size must be a non-negative integer")
        if not isinstance(self.manifest, bytes) or len(self.manifest) > _MAX_MANIFEST_BYTES:
            raise ValueError("manifest must be bounded bytes")


@dataclass(frozen=True)
class CleanupDebt:
    """Exact post-promotion reclamation work, retained for recovery."""

    operation_id: str
    locator: str


@dataclass(frozen=True)
class PromotionResult:
    """Atomic committed-entry outcome of a verified mutation."""

    entry: EntrySnapshot
    cleanup_debt: tuple[CleanupDebt, ...] = ()


@dataclass(frozen=True)
class PageToken:
    """Opaque bounded cursor for future clear/reconciliation authority pages."""

    value: str = ""


@dataclass(frozen=True)
class ProjectionRevision:
    """Revision returned when a non-authoritative projection is marked current."""

    value: int


@dataclass(frozen=True)
class AuthorityCapabilities:
    """Truthful adapter capabilities, independent of SQL/path implementation."""

    durable: bool
    multiprocess: bool
    transactional: bool = True


@runtime_checkable
class LifecycleAuthority(Protocol):
    """Semantic complete-transition interface for lifecycle authority adapters."""

    capabilities: AuthorityCapabilities

    def read_entry(self, key: str) -> EntrySnapshot | None: ...

    def prepare_mutation(self, spec: MutationSpec) -> PreparedMutation: ...

    def record_verification(
        self, prepared: PreparedMutation, proof: VerificationProof
    ) -> None: ...

    def promote_mutation(self, prepared: PreparedMutation) -> PromotionResult: ...

    def abort_mutation(self, prepared: PreparedMutation) -> None: ...

    def delete_entry(self, key: str, *, expected: EntryExpectation) -> None: ...

    # The remaining complete transitions are intentionally semantic placeholders
    # for the next lifecycle waves; callers never access adapter tables directly.
    def retire_tombstone(self, key: str, *, expected: EntryExpectation) -> None: ...

    def begin_clear(self) -> PageToken: ...

    def page_clear(self, token: PageToken) -> tuple[EntrySnapshot, ...]: ...

    def checkpoint_clear(self, token: PageToken) -> None: ...

    def begin_reconciliation(self) -> PageToken: ...

    def page_reconciliation(self, token: PageToken) -> tuple[CleanupDebt, ...]: ...

    def checkpoint_reconciliation(self, token: PageToken) -> None: ...

    def compare_and_mark_projection(
        self, expected: ProjectionRevision | None
    ) -> ProjectionRevision: ...

    def close(self) -> None: ...


__all__ = [
    "AuthorityCapabilities",
    "CleanupDebt",
    "EntryExpectation",
    "EntrySnapshot",
    "LifecycleAuthority",
    "MutationSpec",
    "PageToken",
    "PreparedMutation",
    "ProjectionRevision",
    "PromotionResult",
    "VerificationProof",
]
