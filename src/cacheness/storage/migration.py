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
from types import MappingProxyType
from typing import Iterable, Mapping

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
_MAX_COMPATIBILITY_TEXT_BYTES = 256


class CompatibilityDimension(str, Enum):
    """One independently persisted contract considered during migration planning."""

    STORE_LAYOUT = "store_layout"
    AUTHORITY = "authority"
    MANIFEST_SCHEMA = "manifest_schema"
    PAYLOAD = "payload"
    CATALOG = "catalog"


class CompatibilityOutcome(str, Enum):
    """The fail-closed result for a compatibility dimension or complete plan."""

    SUPPORTED = "supported"
    REBUILD_ONLY = "rebuild_only"
    BLOCKED = "blocked"
    UNVERIFIABLE = "unverifiable"


def _compatibility_text(value: object, field_name: str) -> str:
    """Validate an opaque, non-secret compatibility identifier."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if len(value.encode("utf-8")) > _MAX_COMPATIBILITY_TEXT_BYTES:
        raise ValueError(f"{field_name} exceeds the byte bound")
    if "*" in value:
        raise ValueError(f"{field_name} cannot contain a wildcard")
    return value


def _compatibility_value(value: object, field_name: str, *, size: int) -> tuple[str, ...]:
    """Freeze one exact dimension value without accepting broad matching patterns."""
    if not isinstance(value, tuple) or len(value) != size:
        raise ValueError(f"{field_name} must contain exactly {size} values")
    return tuple(
        _compatibility_text(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


@dataclass(frozen=True)
class ReleaseWindow:
    """The bounded direct-support promise for one released migration target."""

    current_release: str
    immediately_previous_release: str | None = None

    def __post_init__(self) -> None:
        _compatibility_text(self.current_release, "current_release")
        if self.immediately_previous_release is not None:
            _compatibility_text(self.immediately_previous_release, "immediately_previous_release")
            if self.immediately_previous_release == self.current_release:
                raise ValueError("immediately_previous_release must differ from current_release")

    @classmethod
    def first_release_baseline(cls) -> "ReleaseWindow":
        """Return the current baseline without inventing an earlier release edge."""
        return cls(current_release="current")

    def supports_direct_source(self, release: str) -> bool:
        """Return whether this release is in the only directly supported source window."""
        return release in {self.current_release, self.immediately_previous_release}

    def supports_direct_target(self, release: str) -> bool:
        """Return whether this plan targets the published current release."""
        return release == self.current_release


@dataclass(frozen=True)
class CompatibilityIdentity:
    """The independent, redacted persisted-contract identity of one store entry."""

    release: str
    store_layout: tuple[str, ...]
    authority: tuple[str, ...]
    manifest_schema: tuple[str, ...]
    payload: tuple[str, ...]
    catalog: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "release", _compatibility_text(self.release, "release"))
        object.__setattr__(
            self,
            "store_layout",
            _compatibility_value(self.store_layout, "store_layout", size=4),
        )
        object.__setattr__(
            self,
            "authority",
            _compatibility_value(self.authority, "authority", size=3),
        )
        object.__setattr__(
            self,
            "manifest_schema",
            _compatibility_value(self.manifest_schema, "manifest_schema", size=2),
        )
        object.__setattr__(self, "payload", _compatibility_value(self.payload, "payload", size=3))
        object.__setattr__(self, "catalog", _compatibility_value(self.catalog, "catalog", size=3))

    def value_for(self, dimension: CompatibilityDimension) -> tuple[str, ...]:
        """Return the exact value for one independently versioned contract."""
        if not isinstance(dimension, CompatibilityDimension):
            raise TypeError("dimension must be a CompatibilityDimension")
        return {
            CompatibilityDimension.STORE_LAYOUT: self.store_layout,
            CompatibilityDimension.AUTHORITY: self.authority,
            CompatibilityDimension.MANIFEST_SCHEMA: self.manifest_schema,
            CompatibilityDimension.PAYLOAD: self.payload,
            CompatibilityDimension.CATALOG: self.catalog,
        }[dimension]


@dataclass(frozen=True)
class VersionEdge:
    """One exact directed transition for one persisted compatibility dimension."""

    source_release: str
    destination_release: str
    dimension: CompatibilityDimension
    source_value: tuple[str, ...]
    destination_value: tuple[str, ...]

    def __post_init__(self) -> None:
        _compatibility_text(self.source_release, "source_release")
        _compatibility_text(self.destination_release, "destination_release")
        if self.source_release == self.destination_release:
            raise ValueError("version edges must connect distinct releases")
        if not isinstance(self.dimension, CompatibilityDimension):
            raise TypeError("dimension must be a CompatibilityDimension")
        sizes = {
            CompatibilityDimension.STORE_LAYOUT: 4,
            CompatibilityDimension.AUTHORITY: 3,
            CompatibilityDimension.MANIFEST_SCHEMA: 2,
            CompatibilityDimension.PAYLOAD: 3,
            CompatibilityDimension.CATALOG: 3,
        }
        size = sizes[self.dimension]
        object.__setattr__(
            self,
            "source_value",
            _compatibility_value(self.source_value, "source_value", size=size),
        )
        object.__setattr__(
            self,
            "destination_value",
            _compatibility_value(self.destination_value, "destination_value", size=size),
        )
        if self.source_value == self.destination_value:
            raise ValueError("version edges must change their declared dimension")


@dataclass(frozen=True)
class CompatibilityResult:
    """Immutable complete compatibility result derived from one matrix evaluation."""

    outcome: CompatibilityOutcome
    dimension_outcomes: Mapping[CompatibilityDimension, CompatibilityOutcome]
    reasons: Mapping[CompatibilityDimension, "MigrationReason"]
    release_window_reason: "MigrationReason" | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, CompatibilityOutcome):
            raise TypeError("outcome must be a CompatibilityOutcome")
        expected = set(CompatibilityDimension)
        if set(self.dimension_outcomes) != expected or set(self.reasons) != expected:
            raise ValueError("compatibility results must classify every dimension")
        if any(not isinstance(item, CompatibilityOutcome) for item in self.dimension_outcomes.values()):
            raise TypeError("dimension_outcomes contains an invalid outcome")
        if any(not isinstance(item, MigrationReason) for item in self.reasons.values()):
            raise TypeError("reasons contains an invalid migration reason")
        if self.release_window_reason is not None and not isinstance(
            self.release_window_reason, MigrationReason
        ):
            raise TypeError("release_window_reason is invalid")
        object.__setattr__(self, "dimension_outcomes", MappingProxyType(dict(self.dimension_outcomes)))
        object.__setattr__(self, "reasons", MappingProxyType(dict(self.reasons)))


class CompatibilityMatrix:
    """Classify independent version contracts through exact bounded directed edges."""

    def __init__(self, release_window: ReleaseWindow, *, edges: Iterable[VersionEdge] = ()) -> None:
        if not isinstance(release_window, ReleaseWindow):
            raise TypeError("release_window must be a ReleaseWindow")
        frozen_edges = tuple(edges)
        if not all(isinstance(edge, VersionEdge) for edge in frozen_edges):
            raise TypeError("edges must contain VersionEdge values")
        edge_index: dict[tuple[object, ...], VersionEdge] = {}
        for edge in frozen_edges:
            if not release_window.supports_direct_source(edge.source_release) or not release_window.supports_direct_target(
                edge.destination_release
            ):
                raise ValueError("version edge lies outside the direct release window")
            key = (
                edge.source_release,
                edge.destination_release,
                edge.dimension,
                edge.source_value,
                edge.destination_value,
            )
            if key in edge_index:
                raise ValueError("duplicate exact version edge")
            edge_index[key] = edge
        self.release_window = release_window
        self.edges = frozen_edges
        self._edge_index = MappingProxyType(edge_index)

    @classmethod
    def default(cls) -> "CompatibilityMatrix":
        """Return the first-release bounded policy with no historical migration edge."""
        return cls(ReleaseWindow.first_release_baseline())

    def _has_edge(
        self,
        source: CompatibilityIdentity,
        destination: CompatibilityIdentity,
        dimension: CompatibilityDimension,
    ) -> bool:
        return (
            source.release,
            destination.release,
            dimension,
            source.value_for(dimension),
            destination.value_for(dimension),
        ) in self._edge_index

    def classify(
        self,
        source: CompatibilityIdentity,
        destination: CompatibilityIdentity,
        *,
        blocked_dimensions: Iterable[CompatibilityDimension] = (),
        unverifiable_dimensions: Iterable[CompatibilityDimension] = (),
    ) -> CompatibilityResult:
        """Classify every dimension without inferring one contract from another."""
        if not isinstance(source, CompatibilityIdentity) or not isinstance(
            destination, CompatibilityIdentity
        ):
            raise TypeError("source and destination must be CompatibilityIdentity values")
        blocked = frozenset(blocked_dimensions)
        unverifiable = frozenset(unverifiable_dimensions)
        if not blocked <= set(CompatibilityDimension) or not unverifiable <= set(
            CompatibilityDimension
        ):
            raise TypeError("blocked and unverifiable dimensions must be CompatibilityDimension values")

        outcomes: dict[CompatibilityDimension, CompatibilityOutcome] = {}
        reasons: dict[CompatibilityDimension, MigrationReason] = {}
        for dimension in CompatibilityDimension:
            if dimension in unverifiable:
                outcomes[dimension] = CompatibilityOutcome.UNVERIFIABLE
                reasons[dimension] = MigrationReason.DIMENSION_UNVERIFIABLE
            elif dimension in blocked:
                outcomes[dimension] = CompatibilityOutcome.BLOCKED
                reasons[dimension] = MigrationReason.DIMENSION_BLOCKED
            elif source.value_for(dimension) == destination.value_for(dimension):
                outcomes[dimension] = CompatibilityOutcome.SUPPORTED
                reasons[dimension] = MigrationReason.COMPATIBLE_EDGE
            elif self._has_edge(source, destination, dimension):
                outcomes[dimension] = CompatibilityOutcome.SUPPORTED
                reasons[dimension] = MigrationReason.DIRECTED_EDGE
            else:
                outcomes[dimension] = CompatibilityOutcome.REBUILD_ONLY
                reasons[dimension] = MigrationReason.MISSING_DIRECTED_EDGE

        release_window_reason = None
        if not self.release_window.supports_direct_source(source.release) or not self.release_window.supports_direct_target(
            destination.release
        ):
            release_window_reason = MigrationReason.RELEASE_WINDOW_UNSUPPORTED

        present = set(outcomes.values())
        if CompatibilityOutcome.UNVERIFIABLE in present:
            outcome = CompatibilityOutcome.UNVERIFIABLE
        elif CompatibilityOutcome.BLOCKED in present:
            outcome = CompatibilityOutcome.BLOCKED
        elif release_window_reason is not None or CompatibilityOutcome.REBUILD_ONLY in present:
            outcome = CompatibilityOutcome.REBUILD_ONLY
        else:
            outcome = CompatibilityOutcome.SUPPORTED
        return CompatibilityResult(
            outcome=outcome,
            dimension_outcomes=outcomes,
            reasons=reasons,
            release_window_reason=release_window_reason,
        )


class MigrationDisposition(str, Enum):
    """The explicit action class for one canonical authority entry."""

    MIGRATABLE = "migratable"
    REBUILDABLE = "rebuildable"
    BLOCKED = "blocked"
    UNVERIFIABLE = "unverifiable"


class MigrationReason(str, Enum):
    """Stable bounded reasons used by the structured plan model."""

    COMPATIBLE_EDGE = "compatible_edge"
    DIRECTED_EDGE = "directed_edge"
    MISSING_DIRECTED_EDGE = "missing_directed_edge"
    RELEASE_WINDOW_UNSUPPORTED = "release_window_unsupported"
    DIMENSION_BLOCKED = "dimension_blocked"
    DIMENSION_UNVERIFIABLE = "dimension_unverifiable"
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
    "CompatibilityDimension",
    "CompatibilityIdentity",
    "CompatibilityMatrix",
    "CompatibilityOutcome",
    "CompatibilityResult",
    "MigrationCompatibilityEdge",
    "MigrationDisposition",
    "MigrationEntryAssessment",
    "MigrationPlan",
    "MigrationReason",
    "MigrationStepResult",
    "OfflineMigrationService",
    "ReleaseWindow",
    "VersionEdge",
]
