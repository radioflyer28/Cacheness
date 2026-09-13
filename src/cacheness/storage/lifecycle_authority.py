"""Backend-neutral transactional authority contracts for BlobStore lifecycle.

The authority owns committed manifest visibility and durable mutation evidence.
Payload serialization, publication, verification, and deletion deliberately remain
outside this seam so a database transaction never spans handler or filesystem I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from .catalog import CatalogPage, CatalogQuery, CatalogSchema
from .transport_evidence import MAX_TRANSPORT_EVIDENCE_BYTES


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
    generation: str | None = None
    manifest_digest: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("lineage", "revision"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{field_name} must be a non-negative integer or None")
        for field_name in ("generation", "manifest_digest"):
            value = getattr(self, field_name)
            if value is not None:
                _bounded_text(value, field_name)
        if self.manifest_digest is not None and (
            len(self.manifest_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.manifest_digest)
        ):
            raise ValueError("manifest_digest must be a lowercase SHA-256 hexadecimal value")

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
    transport_evidence: bytes | None = None

    def __post_init__(self) -> None:
        for field_name in ("key", "generation", "locator"):
            _bounded_text(getattr(self, field_name), field_name)
        if not isinstance(self.manifest, bytes) or len(self.manifest) > _MAX_MANIFEST_BYTES:
            raise ValueError("manifest must be bounded bytes")
        if self.expectation.generation != self.generation:
            raise ValueError("entry expectation generation must corroborate the entry")
        digest = hashlib.sha256(self.manifest).hexdigest()
        if self.expectation.manifest_digest != digest:
            raise ValueError("entry expectation digest must corroborate the manifest")
        if self.transport_evidence is not None and (
            not isinstance(self.transport_evidence, bytes)
            or not self.transport_evidence
            or len(self.transport_evidence) > MAX_TRANSPORT_EVIDENCE_BYTES
        ):
            raise ValueError("transport_evidence must be bounded authenticated bytes or None")


@dataclass(frozen=True)
class MutationSpec:
    """Bounded request to prepare an immutable candidate transition."""

    operation_id: str
    key: str
    generation: str
    candidate_locator: str
    expected: EntryExpectation
    # The lifecycle engine supplies one already-validated, signed canonical
    # descriptor. Authorities preserve these opaque bytes and never rebuild a
    # second catalog representation from individual fields.
    manifest: bytes = b""

    def __post_init__(self) -> None:
        for field_name in ("operation_id", "key", "generation", "candidate_locator"):
            _bounded_text(getattr(self, field_name), field_name)
        if not isinstance(self.manifest, bytes) or len(self.manifest) > _MAX_MANIFEST_BYTES:
            raise ValueError("manifest must be bounded bytes")

    @classmethod
    def validate_operation_id(cls, operation_id: str) -> str:
        """Validate one bounded operation identity before authority lookup."""
        return _bounded_text(operation_id, "operation_id")

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
    transport_evidence: bytes | None = None

    def __post_init__(self) -> None:
        if len(self.digest) != 64 or any(char not in "0123456789abcdef" for char in self.digest):
            raise ValueError("digest must be a lowercase SHA-256 hexadecimal value")
        if not isinstance(self.byte_size, int) or self.byte_size < 0:
            raise ValueError("byte_size must be a non-negative integer")
        if not isinstance(self.manifest, bytes) or len(self.manifest) > _MAX_MANIFEST_BYTES:
            raise ValueError("manifest must be bounded bytes")
        if self.transport_evidence is not None and (
            not isinstance(self.transport_evidence, bytes)
            or not self.transport_evidence
            or len(self.transport_evidence) > MAX_TRANSPORT_EVIDENCE_BYTES
        ):
            raise ValueError("transport_evidence must be bounded authenticated bytes or None")


@dataclass(frozen=True)
class CleanupDebt:
    """Exact post-promotion reclamation work, retained for recovery."""

    operation_id: str
    locator: str
    key: str = ""
    generation: str = ""
    role: str = "previous_generation"
    debt_id: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("operation_id", "locator", "role"):
            _bounded_text(getattr(self, field_name), field_name)
        for field_name in ("key", "generation"):
            value = getattr(self, field_name)
            if value:
                _bounded_text(value, field_name)
        if self.debt_id is not None and (
            type(self.debt_id) is not int or self.debt_id <= 0
        ):
            raise ValueError("debt_id must be a positive integer or None")


@dataclass(frozen=True)
class AuthorityStateSnapshot:
    """Bounded complete-state view for diagnostics and contract tests."""

    revision: int
    projection_dirty: bool
    mutation_states: tuple[tuple[str, str], ...]
    cleanup_debt: tuple[CleanupDebt, ...]

    def __post_init__(self) -> None:
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision must be a non-negative integer")
        if type(self.projection_dirty) is not bool:
            raise ValueError("projection_dirty must be a bool")


@dataclass(frozen=True)
class PromotionResult:
    """Atomic committed-entry outcome of a verified mutation."""

    entry: EntrySnapshot
    cleanup_debt: tuple[CleanupDebt, ...] = ()


@dataclass(frozen=True)
class MutationReplay:
    """One exact operation record reconstructed by the canonical authority.

    This is intentionally a bounded read model, not another lifecycle state
    machine.  The authority remains the only source that can attest to a
    prepared intent, optional verification, or a promoted canonical result.
    """

    prepared: PreparedMutation
    state: str
    verification: VerificationProof | None = None
    promotion: PromotionResult | None = None

    def __post_init__(self) -> None:
        if self.prepared.operation_id != self.prepared.spec.operation_id:
            raise ValueError("prepared operation identity must corroborate its spec")
        if self.state not in {"prepared", "promoted"}:
            raise ValueError("mutation replay state is unsupported")
        if self.verification is not None:
            if (
                self.prepared.spec.manifest
                and self.verification.manifest
                and self.prepared.spec.manifest != self.verification.manifest
            ):
                raise ValueError("verification descriptor differs from prepared descriptor")
        if self.state == "prepared":
            if self.promotion is not None:
                raise ValueError("prepared replay cannot carry a promotion result")
            return
        if self.promotion is None:
            raise ValueError("promoted replay requires a promotion result")
        entry = self.promotion.entry
        spec = self.prepared.spec
        if (
            entry.key != spec.key
            or entry.generation != spec.generation
            or entry.locator != spec.candidate_locator
        ):
            raise ValueError("promotion result does not corroborate the prepared spec")
        descriptor = (
            self.verification.manifest
            if self.verification is not None and self.verification.manifest
            else spec.manifest
        )
        if descriptor and entry.manifest != descriptor:
            raise ValueError("promotion result descriptor is incompatible with the replay")
        if any(debt.operation_id != self.prepared.operation_id for debt in self.promotion.cleanup_debt):
            raise ValueError("promotion cleanup debt does not belong to the replayed operation")


@dataclass(frozen=True)
class PageToken:
    """Opaque bounded cursor for future clear/reconciliation authority pages."""

    value: str = ""


@dataclass(frozen=True)
class ProjectionRevision:
    """Revision returned when a non-authoritative projection is marked current."""

    value: int


@dataclass(frozen=True)
class ProjectionBackup:
    """Private immutable SQLite snapshot used only to render a projection."""

    path: Path
    revision: ProjectionRevision


@dataclass(frozen=True)
class ReconciliationSnapshot:
    """Immutable authority work boundary captured without payload inspection."""

    authority_revision: int
    mutation_high_water: int
    debt_high_water: int
    run_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "authority_revision",
            "mutation_high_water",
            "debt_high_water",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.run_id is not None:
            _bounded_text(self.run_id, "run_id")


@dataclass(frozen=True)
class ReconciliationWork:
    """One authority-indexed residue row; never a payload inventory item."""

    source: str
    row_id: int
    state: str
    mutation: PreparedMutation | None = None
    debt: CleanupDebt | None = None

    def __post_init__(self) -> None:
        if self.source not in {"mutation", "debt"}:
            raise ValueError("Reconciliation work source is unsupported")
        if type(self.row_id) is not int or self.row_id <= 0:
            raise ValueError("Reconciliation work row_id must be positive")
        _bounded_text(self.state, "state")
        if (self.mutation is None) == (self.debt is None):
            raise ValueError("Reconciliation work must carry exactly one residue")


@dataclass(frozen=True)
class ReconciliationPage:
    """One bounded independent-keyset page over mutation and debt rows."""

    works: tuple[ReconciliationWork, ...]
    mutation_cursor: int
    debt_cursor: int

    def __post_init__(self) -> None:
        for field_name in ("mutation_cursor", "debt_cursor"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True)
class AuthorityCapabilities:
    """Truthful adapter capabilities, independent of SQL/path implementation."""

    durable: bool
    multiprocess: bool
    transactional: bool = True
    exact_cas: bool = True
    indexed_paging: bool = True
    projection: bool = True


@runtime_checkable
class LifecycleAuthority(Protocol):
    """Semantic complete-transition interface for lifecycle authority adapters."""

    capabilities: AuthorityCapabilities

    def preflight_mutation(self) -> None: ...

    def read_entry(self, key: str) -> EntrySnapshot | None: ...

    def read_expectation(self, key: str) -> EntryExpectation: ...

    def snapshot_state(self) -> AuthorityStateSnapshot: ...

    def prepare_mutation(self, spec: MutationSpec) -> PreparedMutation: ...

    def read_mutation(self, operation_id: str) -> MutationReplay | None: ...

    def record_verification(
        self, prepared: PreparedMutation, proof: VerificationProof
    ) -> None: ...

    def promote_mutation(self, prepared: PreparedMutation) -> PromotionResult: ...

    def replace_committed_metadata(
        self,
        entry: EntrySnapshot,
        *,
        expected: EntryExpectation,
        manifest: bytes,
    ) -> EntrySnapshot: ...

    def abort_mutation(
        self, prepared: PreparedMutation, *, candidate_persisted: bool = False
    ) -> None: ...

    def list_entries(self) -> tuple[EntrySnapshot, ...]: ...

    def catalog_page(
        self,
        query: CatalogQuery,
        cursor: str | None,
        *,
        schema: CatalogSchema,
        limit: int,
        work_cap: int,
        signing_key: bytes,
        manifest_loader: Callable[[bytes], Any],
    ) -> CatalogPage: ...

    def pending_cleanup_debts(
        self,
        *,
        key: str | None = None,
        operation_id: str | None = None,
    ) -> tuple[CleanupDebt, ...]: ...

    def pending_mutations(self) -> tuple[PreparedMutation, ...]: ...

    def retire_cleanup_debt(self, debt: CleanupDebt) -> None: ...

    def delete_entry(self, key: str, *, expected: EntryExpectation) -> None: ...

    # Additional complete transitions stay semantic; callers never access adapter
    # tables directly.
    def retire_tombstone(self, key: str, *, expected: EntryExpectation) -> None: ...

    def begin_clear(self) -> PageToken: ...

    def page_clear(self, token: PageToken) -> tuple[EntrySnapshot, ...]: ...

    def checkpoint_clear(
        self,
        token: PageToken,
        target: EntrySnapshot | None = None,
        *,
        state: str = "completed",
    ) -> None: ...

    def begin_reconciliation(self) -> PageToken: ...

    def reconciliation_snapshot(
        self, token: PageToken | None = None
    ) -> ReconciliationSnapshot: ...

    def page_reconciliation_work(
        self,
        snapshot: ReconciliationSnapshot,
        *,
        mutation_cursor: int,
        debt_cursor: int,
    ) -> ReconciliationPage: ...

    def page_reconciliation(self, token: PageToken) -> tuple[CleanupDebt, ...]: ...

    def checkpoint_reconciliation(
        self,
        token: PageToken,
        work: ReconciliationWork | None = None,
        *,
        state: str = "completed",
    ) -> None: ...

    def compare_and_mark_projection(
        self, expected: ProjectionRevision | None
    ) -> ProjectionRevision: ...

    def projection_backup(self) -> object: ...

    def close(self) -> None: ...


__all__ = [
    "AuthorityCapabilities",
    "AuthorityStateSnapshot",
    "CleanupDebt",
    "EntryExpectation",
    "EntrySnapshot",
    "LifecycleAuthority",
    "MutationSpec",
    "PageToken",
    "PreparedMutation",
    "ProjectionBackup",
    "ProjectionRevision",
    "PromotionResult",
    "ReconciliationPage",
    "ReconciliationSnapshot",
    "ReconciliationWork",
    "VerificationProof",
]
