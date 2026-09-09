"""Explicit offline migration tracer above the BlobStore lifecycle boundary.

This service is deliberately not reachable from normal construction, open,
read, cleanup, reconciliation, cache policy, or initialization operations.
It coordinates operator evidence and immutable candidate payloads, while the
selected migration authority performs the sole whole-store visibility change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Iterable

from .manifest import BlobManifest, StoreVersionDimensions, sign_current_manifest, verify_current_manifest
from .migration_authority import (
    AuthorityIdentitySnapshot,
    AuthorityInventoryEntry,
    MigrationAuthority,
    VerifiedCandidateReceipt,
    candidate_digest,
)
from .migration_evidence import (
    MaintenanceEvidenceState,
    MaintenanceRunEvidence,
    decode_maintenance_evidence,
    encode_maintenance_evidence,
)


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class MigrationDisposition(str, Enum):
    """The explicit action class for one canonical authority entry."""

    MIGRATABLE = "migratable"
    REBUILDABLE = "rebuildable"
    BLOCKED = "blocked"
    UNVERIFIABLE = "unverifiable"


class MigrationReason(str, Enum):
    """Stable bounded reasons used by the structured plan model."""

    COMPATIBLE_EDGE = "compatible_edge"
    UNSUPPORTED_DIMENSIONS = "unsupported_dimensions"
    MANIFEST_UNAUTHENTICATED = "manifest_unauthenticated"


@dataclass(frozen=True)
class MigrationCompatibilityEdge:
    """One declared payload/store compatibility edge for the offline tracer."""

    source: StoreVersionDimensions
    destination: StoreVersionDimensions
    name: str
    test_only: bool = False

    @classmethod
    def current_to_current_for_test(cls) -> "MigrationCompatibilityEdge":
        """Inject the tracer's only edge without changing any persisted version."""
        current = StoreVersionDimensions()
        return cls(
            source=current,
            destination=current,
            name="test-current-to-current",
            test_only=True,
        )

    def supports(self, versions: StoreVersionDimensions) -> bool:
        """Return whether the inspected explicit dimensions match this edge."""
        return versions == self.source


@dataclass(frozen=True)
class MigrationEntryAssessment:
    """One immutable classified canonical entry from a fixed source revision."""

    entry: AuthorityInventoryEntry
    disposition: MigrationDisposition
    reason: MigrationReason


@dataclass(frozen=True)
class MigrationInspection:
    """Non-mutating authority inventory from which an operator plan is made."""

    source_identity: AuthorityIdentitySnapshot
    destination_identity: AuthorityIdentitySnapshot
    assessments: tuple[MigrationEntryAssessment, ...]


@dataclass(frozen=True)
class MigrationPlan:
    """Exact bounded operator plan, bound to one source identity and revision."""

    run_id: str
    source_identity: AuthorityIdentitySnapshot
    destination_identity: AuthorityIdentitySnapshot
    assessments: tuple[MigrationEntryAssessment, ...]
    digest: str


@dataclass(frozen=True)
class MigrationStepResult:
    """Minimal result for one completed explicit maintenance action."""

    state: MaintenanceEvidenceState
    completed: bool
    evidence_path: Path


def _canonical_plan_digest(
    run_id: str,
    source_identity: AuthorityIdentitySnapshot,
    destination_identity: AuthorityIdentitySnapshot,
    assessments: tuple[MigrationEntryAssessment, ...],
) -> str:
    record = {
        "assessments": [
            {
                "disposition": assessment.disposition.value,
                "generation": assessment.entry.generation,
                "key": assessment.entry.key,
                "manifest_digest": assessment.entry.manifest_digest,
                "reason": assessment.reason.value,
            }
            for assessment in sorted(assessments, key=lambda item: item.entry.key)
        ],
        "destination": {
            "authority_kind": destination_identity.authority_kind,
            "revision": destination_identity.revision,
            "store_id": destination_identity.store_id,
        },
        "run_id": run_id,
        "source": {
            "authority_kind": source_identity.authority_kind,
            "revision": source_identity.revision,
            "store_id": source_identity.store_id,
        },
    }
    raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class OfflineMigrationService:
    """Coordinate an operator's explicit inspect-to-activate maintenance run.

    The service is purposefully limited to the current memory/memory tracer.
    A supplied work directory holds authenticated operator evidence only. It
    never becomes a candidate locator, lifecycle state source, or visibility
    authority; the destination ``MigrationAuthority`` owns activation.
    """

    def __init__(
        self,
        *,
        source,
        destination,
        work_directory: str | Path,
        run_id: str,
        stopped_workers_acknowledged: bool,
        compatibility_edges: Iterable[MigrationCompatibilityEdge],
    ) -> None:
        if not stopped_workers_acknowledged:
            raise ValueError("offline migration requires stopped-worker acknowledgement")
        if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("run_id must be a bounded opaque maintenance identifier")
        if source is destination:
            raise ValueError("source and destination must be separate stores")
        self.source = source
        self.destination = destination
        self.run_id = run_id
        self.work_directory = self._validated_work_directory(work_directory)
        self.compatibility_edges = tuple(compatibility_edges)
        if not self.compatibility_edges:
            raise ValueError("offline migration requires an explicit compatibility edge")
        if not all(isinstance(edge, MigrationCompatibilityEdge) for edge in self.compatibility_edges):
            raise TypeError("compatibility_edges must contain MigrationCompatibilityEdge values")
        self._source_authority = self._migration_authority(source, "source")
        self._destination_authority = self._migration_authority(destination, "destination")
        self._evidence_key = self._shared_signing_key()
        self._candidate_entries: tuple[AuthorityInventoryEntry, ...] = ()

    @property
    def evidence_path(self) -> Path:
        """Return the exact user-scoped evidence path; no latest-run lookup exists."""
        return self.work_directory / f"{self.run_id}.maintenance.json"

    @staticmethod
    def _migration_authority(store, role: str) -> MigrationAuthority:
        authority = getattr(store, "lifecycle_authority", None)
        if not isinstance(authority, MigrationAuthority):
            raise TypeError(f"{role} store does not provide explicit migration authority support")
        return authority

    def _validated_work_directory(self, work_directory: str | Path) -> Path:
        supplied = Path(work_directory)
        resolved = supplied.resolve()
        if supplied.exists() and supplied.is_symlink():
            raise ValueError("maintenance work directory may not be a symlink")
        source_root = Path(self.source.cache_dir).resolve()
        destination_root = Path(self.destination.cache_dir).resolve()
        if any(self._overlaps(resolved, root) for root in (source_root, destination_root)):
            raise ValueError("maintenance work directory must be separate from source and destination")
        return resolved

    @staticmethod
    def _overlaps(left: Path, right: Path) -> bool:
        try:
            left.relative_to(right)
            return True
        except ValueError:
            pass
        try:
            right.relative_to(left)
            return True
        except ValueError:
            return False

    def _shared_signing_key(self) -> bytes:
        source_key = self.source._authority_manifest_key()
        destination_key = self.destination._authority_manifest_key()
        if not hmac.compare_digest(source_key, destination_key):
            raise ValueError("source and destination must preserve one signing provider identity")
        return source_key

    def _revalidate_identities(self, plan: MigrationPlan | None = None) -> tuple[
        AuthorityIdentitySnapshot, AuthorityIdentitySnapshot
    ]:
        source_identity = self._source_authority.migration_identity()
        destination_identity = self._destination_authority.migration_identity()
        if plan is not None and (
            source_identity != plan.source_identity or destination_identity != plan.destination_identity
        ):
            raise ValueError("migration plan is stale; source or destination identity changed")
        return source_identity, destination_identity

    def _authenticated_manifest(self, raw: bytes, *, role: str) -> BlobManifest:
        try:
            manifest = BlobManifest.from_canonical_bytes(raw)
            verify_current_manifest(manifest, self._evidence_key)
        except Exception as exc:
            raise ValueError(f"{role} manifest cannot be authenticated for migration") from exc
        if manifest.canonical_bytes() != raw:
            raise ValueError(f"{role} manifest is not canonical")
        return manifest

    def _write_evidence(self, evidence: MaintenanceRunEvidence) -> MaintenanceRunEvidence:
        self.work_directory.mkdir(parents=True, exist_ok=True)
        if self.work_directory.is_symlink() or self.evidence_path.is_symlink():
            raise ValueError("maintenance evidence path may not be a symlink")
        encoded = encode_maintenance_evidence(evidence, self._evidence_key)
        temporary = self.evidence_path.with_suffix(".tmp")
        temporary.write_bytes(encoded)
        temporary.replace(self.evidence_path)
        # Reread each durable boundary before the next action trusts it.
        return self.read_evidence()

    def read_evidence(self) -> MaintenanceRunEvidence:
        """Read only this run's authenticated evidence; candidate presence is ignored."""
        try:
            raw = self.evidence_path.read_bytes()
        except OSError as exc:
            raise ValueError("maintenance evidence is required for this action") from exc
        evidence = decode_maintenance_evidence(raw, self._evidence_key)
        if evidence.run_id != self.run_id:
            raise ValueError("maintenance evidence run_id does not match request")
        return evidence

    def _expect_evidence(
        self, state: MaintenanceEvidenceState, *, plan_digest: str
    ) -> MaintenanceRunEvidence:
        evidence = self.read_evidence()
        if evidence.state is not state or evidence.plan_digest != plan_digest:
            raise ValueError(
                f"maintenance evidence is not verified for the required {state.value} step"
            )
        return evidence

    def inspect(self) -> MigrationInspection:
        """Inspect one fixed source revision without mutating either store."""
        page = self._source_authority.migration_inventory()
        destination_identity = self._destination_authority.migration_identity()
        assessments: list[MigrationEntryAssessment] = []
        for entry in page.entries:
            try:
                manifest = self._authenticated_manifest(entry.manifest, role="source")
            except ValueError:
                assessments.append(
                    MigrationEntryAssessment(
                        entry=entry,
                        disposition=MigrationDisposition.UNVERIFIABLE,
                        reason=MigrationReason.MANIFEST_UNAUTHENTICATED,
                    )
                )
                continue
            if any(edge.supports(manifest.versions) for edge in self.compatibility_edges):
                disposition = MigrationDisposition.MIGRATABLE
                reason = MigrationReason.COMPATIBLE_EDGE
            else:
                disposition = MigrationDisposition.REBUILDABLE
                reason = MigrationReason.UNSUPPORTED_DIMENSIONS
            assessments.append(
                MigrationEntryAssessment(
                    entry=entry,
                    disposition=disposition,
                    reason=reason,
                )
            )
        inspection = MigrationInspection(
            source_identity=page.identity,
            destination_identity=destination_identity,
            assessments=tuple(assessments),
        )
        self._write_evidence(
            MaintenanceRunEvidence(
                evidence_version=1,
                run_id=self.run_id,
                plan_digest="",
                source_identity=page.identity,
                source_revision=page.identity.revision,
                destination_identity=destination_identity,
                state=MaintenanceEvidenceState.INSPECTED,
                completed_steps=("inspect",),
            )
        )
        return inspection

    def plan(self, inspection: MigrationInspection) -> MigrationPlan:
        """Freeze an entry-complete plan after revalidating inspection evidence."""
        if not isinstance(inspection, MigrationInspection):
            raise TypeError("inspection must be a MigrationInspection")
        current_source, current_destination = self._revalidate_identities()
        if (
            current_source != inspection.source_identity
            or current_destination != inspection.destination_identity
        ):
            raise ValueError("inspection is stale; reinspection is required")
        evidence = self._expect_evidence(MaintenanceEvidenceState.INSPECTED, plan_digest="")
        if evidence.source_identity != inspection.source_identity:
            raise ValueError("inspection evidence does not corroborate the source")
        if any(
            assessment.disposition is not MigrationDisposition.MIGRATABLE
            for assessment in inspection.assessments
        ):
            raise ValueError("migration plan contains non-migratable entries")
        digest = _canonical_plan_digest(
            self.run_id,
            inspection.source_identity,
            inspection.destination_identity,
            inspection.assessments,
        )
        plan = MigrationPlan(
            run_id=self.run_id,
            source_identity=inspection.source_identity,
            destination_identity=inspection.destination_identity,
            assessments=inspection.assessments,
            digest=digest,
        )
        self._write_evidence(
            MaintenanceRunEvidence(
                evidence_version=1,
                run_id=self.run_id,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                source_revision=plan.source_identity.revision,
                destination_identity=plan.destination_identity,
                state=MaintenanceEvidenceState.PLANNED,
                completed_steps=("inspect", "plan"),
            )
        )
        return plan

    def _validate_plan(self, plan: MigrationPlan) -> None:
        if not isinstance(plan, MigrationPlan) or plan.run_id != self.run_id:
            raise ValueError("migration plan does not belong to this explicit run")
        expected_digest = _canonical_plan_digest(
            plan.run_id,
            plan.source_identity,
            plan.destination_identity,
            plan.assessments,
        )
        if not hmac.compare_digest(plan.digest, expected_digest):
            raise ValueError("migration plan digest is invalid")
        self._revalidate_identities(plan)

    def stage(self, plan: MigrationPlan) -> MigrationStepResult:
        """Copy authenticated immutable payloads into an invisible candidate."""
        self._validate_plan(plan)
        self._expect_evidence(MaintenanceEvidenceState.PLANNED, plan_digest=plan.digest)
        page = self._source_authority.migration_inventory(
            expected_revision=plan.source_identity.revision
        )
        expected = {assessment.entry.key: assessment for assessment in plan.assessments}
        if set(entry.key for entry in page.entries) != set(expected):
            raise ValueError("source inventory changed; reinspection is required")
        candidates: list[AuthorityInventoryEntry] = []
        for source_entry in page.entries:
            assessment = expected[source_entry.key]
            if assessment.entry.manifest_digest != source_entry.manifest_digest:
                raise ValueError("source entry changed; reinspection is required")
            manifest = self._authenticated_manifest(source_entry.manifest, role="source")
            source_io = self.source._materialize_authority_store()
            with source_io.open_snapshot(
                manifest.locator, dict(manifest.handler_metadata)
            ) as snapshot:
                payload = snapshot.path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != manifest.digest or len(payload) != manifest.byte_size:
                raise ValueError("source payload fails authenticated integrity verification")
            suffix = "".join(Path(manifest.locator).suffixes)
            candidate_locator = (
                f"generations/migration/{self.run_id}/{source_entry.generation}{suffix}"
            )
            self.destination.payload_backend.write_blob(candidate_locator, payload)
            candidate_manifest = sign_current_manifest(
                replace(manifest, locator=candidate_locator), self._evidence_key
            )
            candidate_raw = candidate_manifest.canonical_bytes()
            self._authenticated_manifest(candidate_raw, role="candidate")
            candidates.append(
                AuthorityInventoryEntry(
                    key=source_entry.key,
                    generation=source_entry.generation,
                    locator=candidate_locator,
                    manifest=candidate_raw,
                    payload_digest=manifest.digest,
                    byte_size=manifest.byte_size,
                )
            )
        self._revalidate_identities(plan)
        self._candidate_entries = tuple(candidates)
        receipt = VerifiedCandidateReceipt(
            run_id=self.run_id,
            plan_digest=plan.digest,
            source_identity=plan.source_identity,
            source_revision=plan.source_identity.revision,
            destination_identity=plan.destination_identity,
            destination_revision=plan.destination_identity.revision,
            candidate_digest=candidate_digest(self._candidate_entries),
            entry_count=len(self._candidate_entries),
            byte_count=sum(entry.byte_size for entry in self._candidate_entries),
        )
        self._write_evidence(
            MaintenanceRunEvidence(
                evidence_version=1,
                run_id=self.run_id,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                source_revision=plan.source_identity.revision,
                destination_identity=plan.destination_identity,
                state=MaintenanceEvidenceState.STAGED,
                completed_steps=("inspect", "plan", "stage"),
                candidate_receipt=receipt,
            )
        )
        return MigrationStepResult(MaintenanceEvidenceState.STAGED, True, self.evidence_path)

    def _candidate_from_evidence(
        self, evidence: MaintenanceRunEvidence, plan: MigrationPlan
    ) -> tuple[AuthorityInventoryEntry, ...]:
        receipt = evidence.candidate_receipt
        if receipt is None or not self._candidate_entries:
            raise ValueError(
                "candidate descriptors require explicit restaging; candidate presence is not evidence"
            )
        if receipt.plan_digest != plan.digest or candidate_digest(self._candidate_entries) != receipt.candidate_digest:
            raise ValueError("candidate descriptors do not match authenticated evidence")
        return self._candidate_entries

    def verify(self, plan: MigrationPlan) -> MigrationStepResult:
        """Verify every candidate payload and descriptor before activation is allowed."""
        self._validate_plan(plan)
        evidence = self._expect_evidence(MaintenanceEvidenceState.STAGED, plan_digest=plan.digest)
        candidates = self._candidate_from_evidence(evidence, plan)
        for candidate in candidates:
            manifest = self._authenticated_manifest(candidate.manifest, role="candidate")
            destination_io = self.destination._materialize_authority_store()
            with destination_io.open_snapshot(candidate.locator, {}) as snapshot:
                payload = snapshot.path.read_bytes()
            if (
                manifest.key != candidate.key
                or manifest.locator != candidate.locator
                or manifest.digest != candidate.payload_digest
                or manifest.byte_size != candidate.byte_size
                or hashlib.sha256(payload).hexdigest() != candidate.payload_digest
                or len(payload) != candidate.byte_size
            ):
                raise ValueError("candidate verification failed before activation")
        receipt = evidence.candidate_receipt
        assert receipt is not None
        self._write_evidence(
            MaintenanceRunEvidence(
                evidence_version=1,
                run_id=self.run_id,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                source_revision=plan.source_identity.revision,
                destination_identity=plan.destination_identity,
                state=MaintenanceEvidenceState.VERIFIED,
                completed_steps=("inspect", "plan", "stage", "verify"),
                candidate_receipt=receipt,
            )
        )
        return MigrationStepResult(MaintenanceEvidenceState.VERIFIED, True, self.evidence_path)

    def activate(self, plan: MigrationPlan) -> MigrationStepResult:
        """Ask the destination authority to publish the already-verified whole candidate."""
        self._validate_plan(plan)
        evidence = self._expect_evidence(MaintenanceEvidenceState.VERIFIED, plan_digest=plan.digest)
        candidates = self._candidate_from_evidence(evidence, plan)
        receipt = evidence.candidate_receipt
        assert receipt is not None
        activation = self._destination_authority.activate_verified_candidate(
            receipt=receipt,
            entries=candidates,
        )
        self._write_evidence(
            MaintenanceRunEvidence(
                evidence_version=1,
                run_id=self.run_id,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                source_revision=plan.source_identity.revision,
                destination_identity=plan.destination_identity,
                state=MaintenanceEvidenceState.ACTIVATED,
                completed_steps=("inspect", "plan", "stage", "verify", "activate"),
                candidate_receipt=receipt,
                activation_receipt=activation,
            )
        )
        return MigrationStepResult(MaintenanceEvidenceState.ACTIVATED, True, self.evidence_path)


__all__ = [
    "MigrationCompatibilityEdge",
    "MigrationDisposition",
    "MigrationEntryAssessment",
    "MigrationPlan",
    "MigrationReason",
    "MigrationStepResult",
    "OfflineMigrationService",
]
