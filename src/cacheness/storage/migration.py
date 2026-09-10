"""Explicit offline migration tracer above the BlobStore lifecycle boundary.

This service is deliberately not reachable from normal construction, open,
read, cleanup, reconciliation, cache policy, or initialization operations.
It coordinates operator evidence and immutable candidate payloads, while the
selected migration authority performs the sole whole-store visibility change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import base64
import hashlib
import hmac
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Iterable, Mapping

from cacheness.error_handling import (
    CacheBlobMigrationEvidenceError,
    CacheBlobMigrationEvidenceMismatchError,
    CacheBlobMigrationOfflineDecisionRequiredError,
    CacheBlobMigrationPlanStaleError,
    CacheManifestIntegrityError,
    CacheManifestUnsupportedVersionError,
    CacheMigrationOrRebuildRequiredError,
)

from .catalog import STORE_FORMAT_VERSION
from .manifest import (
    CURRENT_MANIFEST_SCHEMA_VERSION,
    CURRENT_STORE_EPOCH,
    BlobManifest,
    StoreVersionDimensions,
    sign_current_manifest,
    verify_current_manifest,
)
from .migration_authority import (
    ActivationReceipt,
    AuthorityIdentitySnapshot,
    AuthorityInventoryEntry,
    MigrationAuthority,
    VerifiedCandidateReceipt,
    candidate_digest,
)
from .migration_evidence import (
    MaintenanceEvidenceState,
    MaintenanceEvidenceStore,
    MaintenanceRunEvidence,
    StoppedWorkerAcknowledgement,
    decode_bounded_canonical_json,
)


_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_MAX_COMPATIBILITY_TEXT_BYTES = 256
_MAX_PLAN_BYTES = 1_048_576
_MAX_PLAN_TEXT_BYTES = 262_144
_MAX_PLAN_DEPTH = 16
_MAX_PLAN_NODES = 16_384
_MAX_PLAN_COLLECTION_ITEMS = 4_096


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


def _freeze_plan_json(value: object, *, depth: int = 1, nodes: list[int] | None = None) -> object:
    """Freeze bounded JSON-compatible assessment metadata before it reaches a plan."""
    nodes = [0] if nodes is None else nodes
    nodes[0] += 1
    if nodes[0] > _MAX_PLAN_NODES or depth > _MAX_PLAN_DEPTH:
        raise ValueError("migration plan metadata exceeds structural bounds")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if len(value.encode("utf-8")) > _MAX_PLAN_TEXT_BYTES:
            raise ValueError("migration plan metadata string exceeds the byte bound")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_PLAN_COLLECTION_ITEMS:
            raise ValueError("migration plan metadata collection exceeds the item bound")
        return tuple(_freeze_plan_json(item, depth=depth + 1, nodes=nodes) for item in value)
    if isinstance(value, Mapping):
        if len(value) > _MAX_PLAN_COLLECTION_ITEMS:
            raise ValueError("migration plan metadata collection exceeds the item bound")
        frozen: dict[str, object] = {}
        for key, item in value.items():
            key = _compatibility_text(key, "migration plan metadata key")
            frozen[key] = _freeze_plan_json(item, depth=depth + 1, nodes=nodes)
        return MappingProxyType(frozen)
    raise ValueError("migration plan metadata must be JSON-compatible")


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
    catalog_values: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.entry, AuthorityInventoryEntry):
            raise TypeError("entry must be an AuthorityInventoryEntry")
        if not isinstance(self.disposition, MigrationDisposition):
            raise TypeError("disposition must be a MigrationDisposition")
        if not isinstance(self.reason, MigrationReason):
            raise TypeError("reason must be a MigrationReason")
        if not isinstance(self.catalog_values, Mapping):
            raise TypeError("catalog_values must be a mapping")
        frozen_values = _freeze_plan_json(self.catalog_values)
        if not isinstance(frozen_values, Mapping):
            raise TypeError("catalog_values must remain a mapping")
        object.__setattr__(self, "catalog_values", frozen_values)


@dataclass(frozen=True)
class MigrationInspection:
    """Non-mutating authority inventory from which an operator plan is made."""

    source_identity: AuthorityIdentitySnapshot
    destination_identity: AuthorityIdentitySnapshot
    assessments: tuple[MigrationEntryAssessment, ...]


class MigrationPlanKind(str, Enum):
    """The offline workflow selected by one complete, non-mutating inspection."""

    MIGRATION = "migration"
    REBUILD = "rebuild"
    REFUSED = "refused"


class MigrationPlanState(str, Enum):
    """The operator-visible state represented by a plan, never by report text."""

    PLANNED = "planned"
    REFUSED = "refused"


@dataclass(frozen=True)
class MigrationTotals:
    """Exact count and byte aggregates for every assessment disposition."""

    counts: Mapping[MigrationDisposition, int]
    bytes_by_disposition: Mapping[MigrationDisposition, int]
    total_entries: int
    total_bytes: int

    def __post_init__(self) -> None:
        expected = set(MigrationDisposition)
        if set(self.counts) != expected or set(self.bytes_by_disposition) != expected:
            raise ValueError("migration totals must include every disposition")
        if any(type(value) is not int or value < 0 for value in self.counts.values()):
            raise ValueError("migration counts must be non-negative integers")
        if any(
            type(value) is not int or value < 0 for value in self.bytes_by_disposition.values()
        ):
            raise ValueError("migration byte totals must be non-negative integers")
        if type(self.total_entries) is not int or type(self.total_bytes) is not int:
            raise ValueError("migration aggregate totals must be integers")
        if self.total_entries != sum(self.counts.values()):
            raise ValueError("migration total_entries disagrees with dispositions")
        if self.total_bytes != sum(self.bytes_by_disposition.values()):
            raise ValueError("migration total_bytes disagrees with dispositions")
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(
            self, "bytes_by_disposition", MappingProxyType(dict(self.bytes_by_disposition))
        )

    @classmethod
    def from_entries(cls, entries: tuple[MigrationEntryAssessment, ...]) -> "MigrationTotals":
        """Aggregate entry-complete classifications without opening payload bytes."""
        counts = {disposition: 0 for disposition in MigrationDisposition}
        byte_counts = {disposition: 0 for disposition in MigrationDisposition}
        for assessment in entries:
            counts[assessment.disposition] += 1
            byte_counts[assessment.disposition] += assessment.entry.byte_size
        return cls(
            counts=counts,
            bytes_by_disposition=byte_counts,
            total_entries=len(entries),
            total_bytes=sum(assessment.entry.byte_size for assessment in entries),
        )


def _all_supported_compatibility() -> CompatibilityResult:
    """Represent the current-to-current baseline without declaring a version edge."""
    outcomes = {dimension: CompatibilityOutcome.SUPPORTED for dimension in CompatibilityDimension}
    reasons = {dimension: MigrationReason.COMPATIBLE_EDGE for dimension in CompatibilityDimension}
    return CompatibilityResult(
        outcome=CompatibilityOutcome.SUPPORTED,
        dimension_outcomes=outcomes,
        reasons=reasons,
    )


def _identity_record(identity: AuthorityIdentitySnapshot) -> dict[str, object]:
    """Render only an authority fingerprint, never a path, DSN, or signing key."""
    return {
        "authority_kind": identity.authority_kind,
        "capability": identity.capability,
        "revision": identity.revision,
        "schema_version": identity.schema_version,
        "store_id": identity.store_id,
    }


def _identity_from_record(record: object, field_name: str) -> AuthorityIdentitySnapshot:
    if not isinstance(record, dict) or set(record) != {
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


def _compatibility_record(compatibility: CompatibilityResult) -> dict[str, object]:
    return {
        "dimension_outcomes": {
            dimension.value: compatibility.dimension_outcomes[dimension].value
            for dimension in CompatibilityDimension
        },
        "outcome": compatibility.outcome.value,
        "reasons": {
            dimension.value: compatibility.reasons[dimension].value
            for dimension in CompatibilityDimension
        },
        "release_window_reason": (
            None
            if compatibility.release_window_reason is None
            else compatibility.release_window_reason.value
        ),
    }


def _compatibility_from_record(record: object) -> CompatibilityResult:
    required = {"dimension_outcomes", "outcome", "reasons", "release_window_reason"}
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("compatibility is invalid")
    try:
        outcomes = {
            dimension: CompatibilityOutcome(record["dimension_outcomes"][dimension.value])
            for dimension in CompatibilityDimension
        }
        reasons = {
            dimension: MigrationReason(record["reasons"][dimension.value])
            for dimension in CompatibilityDimension
        }
        release_reason = record["release_window_reason"]
        return CompatibilityResult(
            outcome=CompatibilityOutcome(record["outcome"]),
            dimension_outcomes=outcomes,
            reasons=reasons,
            release_window_reason=(
                None if release_reason is None else MigrationReason(release_reason)
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("compatibility contains an invalid value") from exc


def _thaw_json(value: object) -> object:
    """Convert frozen assessment metadata back to JSON-compatible ordinary values."""
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class MigrationPlan:
    """The one immutable machine-authoritative plan for offline maintenance."""

    run_id: str
    source_identity: AuthorityIdentitySnapshot
    destination_identity: AuthorityIdentitySnapshot
    entries: tuple[MigrationEntryAssessment, ...]
    digest: str = ""
    plan_version: int = 1
    plan_id: str = ""
    plan_kind: MigrationPlanKind = MigrationPlanKind.MIGRATION
    source_revision: int = -1
    release_window: ReleaseWindow = field(default_factory=ReleaseWindow.first_release_baseline)
    compatibility: CompatibilityResult | None = None
    totals: MigrationTotals | None = None
    intended_actions: tuple[str, ...] = ()
    exclusions: tuple[Mapping[str, object], ...] = ()
    state: MigrationPlanState = MigrationPlanState.PLANNED

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be a bounded opaque maintenance identifier")
        if not isinstance(self.source_identity, AuthorityIdentitySnapshot) or not isinstance(
            self.destination_identity, AuthorityIdentitySnapshot
        ):
            raise TypeError("plan identities must be AuthorityIdentitySnapshot values")
        if not isinstance(self.entries, tuple) or not all(
            isinstance(entry, MigrationEntryAssessment) for entry in self.entries
        ):
            raise TypeError("entries must be immutable migration assessments")
        if len({entry.entry.key for entry in self.entries}) != len(self.entries):
            raise ValueError("migration plan entries cannot contain duplicate keys")
        if type(self.plan_version) is not int or self.plan_version != 1:
            raise ValueError("unsupported migration plan version")
        if not self.plan_id:
            object.__setattr__(self, "plan_id", self.run_id)
        if self.plan_id != self.run_id:
            raise ValueError("plan_id must bind exactly to the explicit run_id")
        if not isinstance(self.plan_kind, MigrationPlanKind) or not isinstance(
            self.state, MigrationPlanState
        ):
            raise TypeError("plan kind and state are invalid")
        if self.source_revision == -1:
            object.__setattr__(self, "source_revision", self.source_identity.revision)
        if self.source_revision != self.source_identity.revision:
            raise ValueError("source_revision must match source_identity")
        if not isinstance(self.release_window, ReleaseWindow):
            raise TypeError("release_window must be a ReleaseWindow")
        if self.compatibility is None:
            object.__setattr__(self, "compatibility", _all_supported_compatibility())
        if not isinstance(self.compatibility, CompatibilityResult):
            raise TypeError("compatibility must be a CompatibilityResult")
        if self.totals is None:
            object.__setattr__(self, "totals", MigrationTotals.from_entries(self.entries))
        if not isinstance(self.totals, MigrationTotals):
            raise TypeError("totals must be a MigrationTotals")
        expected_totals = MigrationTotals.from_entries(self.entries)
        if self.totals != expected_totals:
            raise ValueError("migration plan totals disagree with entries")
        if not isinstance(self.intended_actions, tuple) or any(
            not isinstance(action, str) or not action or "force" in action for action in self.intended_actions
        ):
            raise ValueError("intended actions are invalid")
        if not isinstance(self.exclusions, tuple) or any(
            not isinstance(exclusion, Mapping) for exclusion in self.exclusions
        ):
            raise TypeError("exclusions must be immutable mappings")
        frozen_exclusions = tuple(_freeze_plan_json(exclusion) for exclusion in self.exclusions)
        if not all(isinstance(exclusion, Mapping) for exclusion in frozen_exclusions):
            raise TypeError("exclusions must remain mappings")
        object.__setattr__(self, "exclusions", frozen_exclusions)
        if self.plan_kind is MigrationPlanKind.MIGRATION and self.state is not MigrationPlanState.PLANNED:
            raise ValueError("a migration plan must be planned before any mutation")
        if self.plan_kind is not MigrationPlanKind.MIGRATION and self.state is not MigrationPlanState.REFUSED:
            raise ValueError("rebuild-only and refused plans must remain non-mutating")
        required_actions = {
            MigrationPlanKind.MIGRATION: ("stage", "verify", "activate"),
            MigrationPlanKind.REBUILD: ("rebuild",),
            MigrationPlanKind.REFUSED: (),
        }
        if self.intended_actions != required_actions[self.plan_kind]:
            raise ValueError("plan kind has an invalid intended action sequence")
        expected_digest = self._expected_digest()
        if self.digest:
            if not isinstance(self.digest, str) or not re.fullmatch(r"[0-9a-f]{64}", self.digest):
                raise ValueError("migration plan digest is invalid")
            if not hmac.compare_digest(self.digest, expected_digest):
                raise ValueError("migration plan digest does not match its content")
        else:
            object.__setattr__(self, "digest", expected_digest)

    @property
    def assessments(self) -> tuple[MigrationEntryAssessment, ...]:
        """Retain the tracer's read-only name while plans expose ``entries`` publicly."""
        return self.entries

    @property
    def stopped_worker_acknowledgement_required(self) -> bool:
        """Require an explicit offline acknowledgement for every mutating intended action."""
        return bool(set(self.intended_actions) & {"stage", "verify", "activate", "rebuild"})

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        source_identity: AuthorityIdentitySnapshot,
        destination_identity: AuthorityIdentitySnapshot,
        entries: tuple[MigrationEntryAssessment, ...],
        release_window: ReleaseWindow,
        compatibility: CompatibilityResult,
    ) -> "MigrationPlan":
        """Build a complete plan whose kind follows its immutable classifications."""
        dispositions = {assessment.disposition for assessment in entries}
        if compatibility.outcome in {
            CompatibilityOutcome.BLOCKED,
            CompatibilityOutcome.UNVERIFIABLE,
        } or dispositions & {MigrationDisposition.BLOCKED, MigrationDisposition.UNVERIFIABLE}:
            plan_kind = MigrationPlanKind.REFUSED
            state = MigrationPlanState.REFUSED
            actions: tuple[str, ...] = ()
        elif compatibility.outcome is CompatibilityOutcome.REBUILD_ONLY or dispositions - {
            MigrationDisposition.MIGRATABLE
        }:
            plan_kind = MigrationPlanKind.REBUILD
            state = MigrationPlanState.REFUSED
            actions = ("rebuild",)
        else:
            plan_kind = MigrationPlanKind.MIGRATION
            state = MigrationPlanState.PLANNED
            actions = ("stage", "verify", "activate")
        return cls(
            run_id=run_id,
            source_identity=source_identity,
            destination_identity=destination_identity,
            entries=entries,
            plan_kind=plan_kind,
            release_window=release_window,
            compatibility=compatibility,
            intended_actions=actions,
            state=state,
        )

    def _entry_record(self, assessment: MigrationEntryAssessment) -> dict[str, object]:
        entry = assessment.entry
        return {
            "catalog_values": _thaw_json(assessment.catalog_values),
            "disposition": assessment.disposition.value,
            "entry": {
                "byte_size": entry.byte_size,
                "generation": entry.generation,
                "key": entry.key,
                "locator": entry.locator,
                "manifest": base64.b64encode(entry.manifest).decode("ascii"),
                "payload_digest": entry.payload_digest,
            },
            "reason": assessment.reason.value,
        }

    def _record(self, *, include_digest: bool) -> dict[str, object]:
        assert self.compatibility is not None
        assert self.totals is not None
        record: dict[str, object] = {
            "compatibility": _compatibility_record(self.compatibility),
            "destination_identity": _identity_record(self.destination_identity),
            "entries": [self._entry_record(entry) for entry in sorted(self.entries, key=lambda item: item.entry.key)],
            "exclusions": [_thaw_json(item) for item in self.exclusions],
            "intended_actions": list(self.intended_actions),
            "plan_id": self.plan_id,
            "plan_kind": self.plan_kind.value,
            "plan_version": self.plan_version,
            "release_window": {
                "current_release": self.release_window.current_release,
                "immediately_previous_release": self.release_window.immediately_previous_release,
            },
            "run_id": self.run_id,
            "source_identity": _identity_record(self.source_identity),
            "source_revision": self.source_revision,
            "state": self.state.value,
            "totals": {
                "bytes_by_disposition": {
                    item.value: self.totals.bytes_by_disposition[item]
                    for item in MigrationDisposition
                },
                "counts": {
                    item.value: self.totals.counts[item] for item in MigrationDisposition
                },
                "total_bytes": self.totals.total_bytes,
                "total_entries": self.totals.total_entries,
            },
        }
        if include_digest:
            record["digest"] = self.digest
        return record

    def _expected_digest(self) -> str:
        return hashlib.sha256(_canonical_plan_bytes(self._record(include_digest=False))).hexdigest()

    def to_canonical_bytes(self) -> bytes:
        """Encode the complete plan once in bounded canonical JSON."""
        encoded = _canonical_plan_bytes(self._record(include_digest=True))
        if len(encoded) > _MAX_PLAN_BYTES:
            raise ValueError("migration plan exceeds the byte bound")
        return encoded

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "MigrationPlan":
        """Decode one canonical plan without accepting duplicate keys or loose numbers."""
        record = decode_bounded_canonical_json(
            raw, max_bytes=_MAX_PLAN_BYTES, max_text_bytes=_MAX_PLAN_TEXT_BYTES
        )
        required = {
            "compatibility",
            "destination_identity",
            "digest",
            "entries",
            "exclusions",
            "intended_actions",
            "plan_id",
            "plan_kind",
            "plan_version",
            "release_window",
            "run_id",
            "source_identity",
            "source_revision",
            "state",
            "totals",
        }
        if set(record) != required:
            raise ValueError("migration plan fields are invalid")
        if _canonical_plan_bytes(record) != raw:
            raise ValueError("migration plan is not canonical")
        entries = _entries_from_record(record["entries"])
        totals = _totals_from_record(record["totals"])
        window_record = record["release_window"]
        if not isinstance(window_record, dict) or set(window_record) != {
            "current_release",
            "immediately_previous_release",
        }:
            raise ValueError("release_window is invalid")
        if not isinstance(record["intended_actions"], list) or not isinstance(
            record["exclusions"], list
        ):
            raise ValueError("migration plan collections are invalid")
        try:
            return cls(
                run_id=record["run_id"],
                source_identity=_identity_from_record(record["source_identity"], "source_identity"),
                destination_identity=_identity_from_record(
                    record["destination_identity"], "destination_identity"
                ),
                entries=entries,
                digest=record["digest"],
                plan_version=record["plan_version"],
                plan_id=record["plan_id"],
                plan_kind=MigrationPlanKind(record["plan_kind"]),
                source_revision=record["source_revision"],
                release_window=ReleaseWindow(
                    current_release=window_record["current_release"],
                    immediately_previous_release=window_record["immediately_previous_release"],
                ),
                compatibility=_compatibility_from_record(record["compatibility"]),
                totals=totals,
                intended_actions=tuple(record["intended_actions"]),
                exclusions=tuple(record["exclusions"]),
                state=MigrationPlanState(record["state"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("migration plan contains an invalid value") from exc


def _canonical_plan_bytes(record: Mapping[str, object]) -> bytes:
    """Encode bounded plan data deterministically without a second report model."""
    try:
        return json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("migration plan contains non-canonical JSON data") from exc


def _entries_from_record(record: object) -> tuple[MigrationEntryAssessment, ...]:
    if not isinstance(record, list):
        raise ValueError("entries are invalid")
    entries: list[MigrationEntryAssessment] = []
    for item in record:
        if not isinstance(item, dict) or set(item) != {
            "catalog_values", "disposition", "entry", "reason"
        }:
            raise ValueError("entry assessment is invalid")
        entry_record = item["entry"]
        if not isinstance(entry_record, dict) or set(entry_record) != {
            "byte_size", "generation", "key", "locator", "manifest", "payload_digest"
        }:
            raise ValueError("entry descriptor is invalid")
        if not isinstance(item["catalog_values"], dict):
            raise ValueError("catalog_values is invalid")
        try:
            manifest = base64.b64decode(entry_record["manifest"], validate=True)
            entries.append(
                MigrationEntryAssessment(
                    entry=AuthorityInventoryEntry(
                        key=entry_record["key"],
                        generation=entry_record["generation"],
                        locator=entry_record["locator"],
                        manifest=manifest,
                        payload_digest=entry_record["payload_digest"],
                        byte_size=entry_record["byte_size"],
                    ),
                    disposition=MigrationDisposition(item["disposition"]),
                    reason=MigrationReason(item["reason"]),
                    catalog_values=item["catalog_values"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("entry assessment contains an invalid value") from exc
    return tuple(entries)


def _totals_from_record(record: object) -> MigrationTotals:
    if not isinstance(record, dict) or set(record) != {
        "bytes_by_disposition", "counts", "total_bytes", "total_entries"
    }:
        raise ValueError("totals are invalid")
    try:
        counts = {
            disposition: record["counts"][disposition.value]
            for disposition in MigrationDisposition
        }
        byte_counts = {
            disposition: record["bytes_by_disposition"][disposition.value]
            for disposition in MigrationDisposition
        }
        return MigrationTotals(
            counts=counts,
            bytes_by_disposition=byte_counts,
            total_entries=record["total_entries"],
            total_bytes=record["total_bytes"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("totals contain an invalid value") from exc


def _current_contract(authority_kind: str) -> CompatibilityIdentity:
    """Describe the current baseline without publishing a historical migration edge."""
    if authority_kind == "sqlite":
        from .sqlite_lifecycle_authority import SQLITE_USER_VERSION

        authority = ("sqlite", "sqlite-lifecycle-authority", str(SQLITE_USER_VERSION))
    elif authority_kind == "memory":
        authority = ("memory", "memory-lifecycle-authority", "in-process")
    else:
        authority = (authority_kind, "unrecognized", "unrecognized")
    return CompatibilityIdentity(
        release="current",
        store_layout=("store-format", str(STORE_FORMAT_VERSION), "epoch", str(CURRENT_STORE_EPOCH)),
        authority=authority,
        manifest_schema=("manifest", str(CURRENT_MANIFEST_SCHEMA_VERSION)),
        payload=("entry-specific", "entry-specific", "current"),
        catalog=("application-catalog", "entry-specific", "authenticated"),
    )


def _historical_contract() -> CompatibilityIdentity:
    """Classify opaque legacy evidence without attempting a compatibility reader."""
    return CompatibilityIdentity(
        release="historical",
        store_layout=("unrecognized", "unrecognized", "unrecognized", "unrecognized"),
        authority=("unrecognized", "unrecognized", "unrecognized"),
        manifest_schema=("unrecognized", "unrecognized"),
        payload=("unrecognized", "unrecognized", "unrecognized"),
        catalog=("unrecognized", "unrecognized", "unrecognized"),
    )


def _read_only_authority_identity(store) -> AuthorityIdentitySnapshot:
    """Read an initialized authority fingerprint without opening a mutation boundary."""
    authority = getattr(store, "lifecycle_authority", None)
    identity_snapshot = getattr(authority, "identity_snapshot", None)
    if callable(identity_snapshot):
        identity = identity_snapshot()
        if isinstance(identity, AuthorityIdentitySnapshot):
            return identity
    raise ValueError("store authority does not expose a read-only inspection identity")


def inspect_migration_store(store) -> MigrationPlan:
    """Inspect one initialized current store without changing bytes, metadata, or authority state."""
    source_identity = _read_only_authority_identity(store)
    authority = getattr(store, "lifecycle_authority", None)
    inventory_page = getattr(authority, "inventory_page", None)
    if not callable(inventory_page):
        raise ValueError("store authority does not support read-only inventory inspection")
    signing_key = store._authority_manifest_key(initialize_new_store=False)
    assessments: list[MigrationEntryAssessment] = []
    limits = authority.lifecycle_limits
    cursor = None
    while True:
        page = inventory_page(
            cursor,
            limit=limits.manifest_page_size,
            work_cap=limits.max_operation_record_bytes,
        )
        if page.identity != source_identity:
            raise ValueError("store inventory changed; reinspection is required")
        for snapshot in page.entries:
            try:
                manifest = BlobManifest.from_canonical_bytes(bytes(snapshot.manifest))
                entry = AuthorityInventoryEntry(
                    key=snapshot.key,
                    generation=snapshot.generation,
                    locator=snapshot.locator,
                    manifest=bytes(snapshot.manifest),
                    payload_digest=manifest.digest,
                    byte_size=manifest.byte_size,
                )
                verify_current_manifest(manifest, signing_key)
            except (
                CacheManifestIntegrityError,
                CacheManifestUnsupportedVersionError,
                CacheMigrationOrRebuildRequiredError,
                ValueError,
            ):
                raw_manifest = bytes(snapshot.manifest)
                entry = AuthorityInventoryEntry(
                    key=snapshot.key,
                    generation=snapshot.generation,
                    locator=snapshot.locator,
                    manifest=raw_manifest,
                    payload_digest=hashlib.sha256(raw_manifest).hexdigest(),
                    byte_size=len(raw_manifest),
                )
                assessments.append(
                    MigrationEntryAssessment(
                        entry=entry,
                        disposition=MigrationDisposition.UNVERIFIABLE,
                        reason=MigrationReason.MANIFEST_UNAUTHENTICATED,
                    )
                )
                continue
            try:
                store.handlers.resolve_payload_contract(
                    manifest.handler_type, manifest.payload_format, manifest.payload_format_version
                )
            except (
                CacheManifestIntegrityError,
                CacheManifestUnsupportedVersionError,
                CacheMigrationOrRebuildRequiredError,
                ValueError,
            ):
                assessments.append(
                    MigrationEntryAssessment(
                        entry=entry,
                        disposition=MigrationDisposition.REBUILDABLE,
                        reason=MigrationReason.MISSING_DIRECTED_EDGE,
                        catalog_values=dict(manifest.catalog_values),
                    )
                )
                continue
            assessments.append(
                MigrationEntryAssessment(
                    entry=entry,
                    disposition=MigrationDisposition.MIGRATABLE,
                    reason=MigrationReason.COMPATIBLE_EDGE,
                    catalog_values=dict(manifest.catalog_values),
                )
            )
        if page.exhausted:
            break
        cursor = page.next_cursor
    contract = _current_contract(source_identity.authority_kind)
    return MigrationPlan.create(
        run_id=f"inspection-{hashlib.sha256(source_identity.store_id.encode('utf-8')).hexdigest()[:32]}",
        source_identity=source_identity,
        destination_identity=source_identity,
        entries=tuple(assessments),
        release_window=ReleaseWindow.first_release_baseline(),
        compatibility=CompatibilityMatrix.default().classify(contract, contract),
    )


def inspect_store_path(root: str | Path) -> MigrationPlan:
    """Produce a rebuild-only plan for historical evidence without opening or changing it."""
    root_path = Path(root).resolve(strict=False)
    fingerprint = hashlib.sha256(str(root_path).encode("utf-8")).hexdigest()
    source_identity = AuthorityIdentitySnapshot(
        store_id=fingerprint, revision=0, authority_kind="unrecognized"
    )
    destination_identity = AuthorityIdentitySnapshot(
        store_id=f"destination-{fingerprint[:32]}", revision=0, authority_kind="unconfigured"
    )
    matrix = CompatibilityMatrix.default()
    from .sqlite_lifecycle_authority import AUTHORITY_RELATIVE_PATH

    corrupted_authority = (root_path / AUTHORITY_RELATIVE_PATH).exists()
    return MigrationPlan.create(
        run_id=f"inspection-{fingerprint[:32]}",
        source_identity=source_identity,
        destination_identity=destination_identity,
        entries=(),
        release_window=matrix.release_window,
        compatibility=matrix.classify(
            _historical_contract(),
            _current_contract("unconfigured"),
            unverifiable_dimensions=(
                (CompatibilityDimension.AUTHORITY,) if corrupted_authority else ()
            ),
        ),
    )


def render_migration_report(plan: MigrationPlan) -> str:
    """Render a human report exclusively from the already-validated plan model."""
    if not isinstance(plan, MigrationPlan):
        raise TypeError("plan must be a MigrationPlan")
    lines = [
        f"Migration plan: {plan.plan_id}",
        f"Digest: {plan.digest}",
        f"Kind: {plan.plan_kind.value}",
        f"State: {plan.state.value}",
        "Release window: "
        f"current={plan.release_window.current_release}, "
        f"previous={plan.release_window.immediately_previous_release or 'none'}",
        f"Source: {plan.source_identity.authority_kind}/{plan.source_identity.store_id} rev {plan.source_revision}",
        f"Destination: {plan.destination_identity.authority_kind}/{plan.destination_identity.store_id} rev {plan.destination_identity.revision}",
        f"Compatibility: {plan.compatibility.outcome.value}",
        "Totals: "
        f"{plan.totals.total_entries} entries, {plan.totals.total_bytes} bytes",
        "Actions: " + (", ".join(plan.intended_actions) if plan.intended_actions else "none"),
        f"Exclusions: {len(plan.exclusions)}",
        "Stopped-worker acknowledgement required: "
        f"{str(plan.stopped_worker_acknowledgement_required).lower()}",
        "Entries:",
    ]
    for dimension in CompatibilityDimension:
        lines.append(
            f"- compatibility.{dimension.value}: "
            f"{plan.compatibility.dimension_outcomes[dimension].value} "
            f"({plan.compatibility.reasons[dimension].value})"
        )
    for assessment in sorted(plan.entries, key=lambda item: item.entry.key):
        lines.append(
            f"- {assessment.entry.key}: {assessment.disposition.value} "
            f"({assessment.reason.value}), {assessment.entry.byte_size} bytes"
        )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class MigrationStepResult:
    """Minimal result for one completed explicit maintenance action."""

    state: MaintenanceEvidenceState
    completed: bool
    evidence_path: Path


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
        self._stopped_workers_acknowledged = stopped_workers_acknowledged
        self.compatibility_edges = tuple(compatibility_edges)
        if not self.compatibility_edges:
            raise ValueError("offline migration requires an explicit compatibility edge")
        if not all(isinstance(edge, MigrationCompatibilityEdge) for edge in self.compatibility_edges):
            raise TypeError("compatibility_edges must contain MigrationCompatibilityEdge values")
        self._source_authority = self._migration_authority(source, "source")
        self._destination_authority = self._migration_authority(destination, "destination")
        source_provider = getattr(source, "_manifest_key_provider", None)
        destination_provider = getattr(destination, "_manifest_key_provider", None)
        if source_provider is None or destination_provider is None:
            raise CacheBlobMigrationEvidenceError(
                "offline migration requires existing source and destination signing providers",
                context={"operation": "migration.configure"},
            )
        self._evidence_store = MaintenanceEvidenceStore(
            self.work_directory, self.run_id, source_provider
        )
        self.work_directory = self._evidence_store.work_directory
        self._destination_key_provider = destination_provider
        self._evidence_key = self._shared_signing_key()
        self._candidate_entries: tuple[AuthorityInventoryEntry, ...] = ()

    @property
    def evidence_path(self) -> Path:
        """Return the exact user-scoped evidence path; no latest-run lookup exists."""
        return self._evidence_store.evidence_path

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
        source_key = self._evidence_store._signing_key()
        try:
            destination_key = self._destination_key_provider.get_key()
        except Exception as exc:
            raise CacheBlobMigrationEvidenceError(
                "destination maintenance signing identity is unavailable",
                context={"operation": "migration.configure"},
            ) from exc
        if not hmac.compare_digest(source_key, destination_key):
            raise CacheBlobMigrationEvidenceMismatchError(
                "source and destination must preserve one signing provider identity",
                context={"operation": "migration.configure", "run_id": self.run_id},
            )
        return source_key

    def _revalidate_identities(self, plan: MigrationPlan | None = None) -> tuple[
        AuthorityIdentitySnapshot, AuthorityIdentitySnapshot
    ]:
        source_identity = self._source_authority.identity_snapshot()
        destination_identity = self._destination_authority.identity_snapshot()
        if plan is not None and (
            source_identity != plan.source_identity or destination_identity != plan.destination_identity
        ):
            raise CacheBlobMigrationPlanStaleError(
                "migration plan is stale; source or destination identity changed",
                context={"operation": "migration.revalidate", "run_id": self.run_id},
            )
        return source_identity, destination_identity

    def _acknowledgement(
        self, plan_digest: str, source_identity: AuthorityIdentitySnapshot
    ) -> StoppedWorkerAcknowledgement:
        """Bind the operator assertion to the exact current source and plan."""
        if not self._stopped_workers_acknowledged:
            raise CacheBlobMigrationEvidenceMismatchError(
                "offline migration requires stopped-worker acknowledgement",
                context={"operation": "migration.acknowledgement", "run_id": self.run_id},
            )
        return StoppedWorkerAcknowledgement(
            run_id=self.run_id,
            plan_digest=plan_digest,
            source_identity=source_identity,
            source_revision=source_identity.revision,
        )

    def _new_evidence(
        self,
        *,
        state: MaintenanceEvidenceState,
        plan_digest: str,
        source_identity: AuthorityIdentitySnapshot,
        destination_identity: AuthorityIdentitySnapshot,
        completed_steps: tuple[str, ...],
        candidate_receipt: VerifiedCandidateReceipt | None = None,
        activation_receipt: ActivationReceipt | None = None,
        authority_receipts: tuple[str, ...] = (),
        completed_output_digests: Mapping[str, str] | None = None,
    ) -> MaintenanceRunEvidence:
        """Build a fully bound evidence record without serializing signing bytes."""
        return MaintenanceRunEvidence(
            evidence_version=1,
            run_id=self.run_id,
            plan_digest=plan_digest,
            source_identity=source_identity,
            source_revision=source_identity.revision,
            destination_identity=destination_identity,
            destination_revision=destination_identity.revision,
            acknowledgement=self._acknowledgement(plan_digest, source_identity),
            state=state,
            completed_steps=completed_steps,
            candidate_receipt=candidate_receipt,
            activation_receipt=activation_receipt,
            authority_receipts=authority_receipts,
            completed_output_digests=completed_output_digests or {},
        )

    @staticmethod
    def _inventory_page(authority, cursor=None):
        """Request one authority-declared bounded raw maintenance page."""
        limits = authority.lifecycle_limits
        return authority.inventory_page(
            cursor,
            limit=limits.manifest_page_size,
            work_cap=limits.max_operation_record_bytes,
        )

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
        if not self.evidence_path.exists():
            return self._evidence_store.create(evidence)
        return self._evidence_store.checkpoint(self._evidence_store.read_bytes(), evidence)

    def read_evidence(self) -> MaintenanceRunEvidence:
        """Read only this run's authenticated evidence; candidate presence is ignored."""
        return self._evidence_store.load()

    def _expect_evidence(
        self, state: MaintenanceEvidenceState, *, plan_digest: str
    ) -> MaintenanceRunEvidence:
        evidence = self.read_evidence()
        if evidence.state is not state or evidence.plan_digest != plan_digest:
            raise CacheBlobMigrationEvidenceMismatchError(
                f"maintenance evidence is not verified for the required {state.value} step",
                context={"operation": "migration.evidence", "run_id": self.run_id},
            )
        source_identity, destination_identity = self._revalidate_identities()
        if (
            evidence.source_identity != source_identity
            or evidence.source_revision != source_identity.revision
            or evidence.destination_identity != destination_identity
            or evidence.destination_revision != destination_identity.revision
            or evidence.acknowledgement
            != self._acknowledgement(plan_digest, source_identity)
        ):
            raise CacheBlobMigrationPlanStaleError(
                "maintenance evidence source or destination is stale; reinspection is required",
                context={"operation": "migration.evidence", "run_id": self.run_id},
            )
        return evidence

    def inspect(self) -> MigrationInspection:
        """Inspect one fixed source revision without mutating either store."""
        destination_identity = self._destination_authority.identity_snapshot()
        assessments: list[MigrationEntryAssessment] = []
        cursor = None
        source_identity = None
        while True:
            page = self._inventory_page(self._source_authority, cursor)
            if source_identity is None:
                source_identity = page.identity
            elif page.identity != source_identity:
                raise ValueError("source inventory changed; reinspection is required")
            for snapshot in page.entries:
                try:
                    manifest = self._authenticated_manifest(snapshot.manifest, role="source")
                    entry = AuthorityInventoryEntry(
                        key=snapshot.key,
                        generation=snapshot.generation,
                        locator=snapshot.locator,
                        manifest=bytes(snapshot.manifest),
                        payload_digest=manifest.digest,
                        byte_size=manifest.byte_size,
                    )
                except ValueError:
                    entry = AuthorityInventoryEntry(
                        key=snapshot.key,
                        generation=snapshot.generation,
                        locator=snapshot.locator,
                        manifest=bytes(snapshot.manifest),
                        payload_digest=hashlib.sha256(snapshot.manifest).hexdigest(),
                        byte_size=len(snapshot.manifest),
                    )
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
            if page.exhausted:
                break
            cursor = page.next_cursor
        assert source_identity is not None
        inspection = MigrationInspection(
            source_identity=source_identity,
            destination_identity=destination_identity,
            assessments=tuple(assessments),
        )
        self._write_evidence(
            self._new_evidence(
                state=MaintenanceEvidenceState.INSPECTED,
                plan_digest="",
                source_identity=source_identity,
                destination_identity=destination_identity,
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
            raise CacheBlobMigrationPlanStaleError(
                "inspection is stale; reinspection is required",
                context={"operation": "migration.plan", "run_id": self.run_id},
            )
        evidence = self._expect_evidence(MaintenanceEvidenceState.INSPECTED, plan_digest="")
        if evidence.source_identity != inspection.source_identity:
            raise CacheBlobMigrationEvidenceMismatchError(
                "inspection evidence does not corroborate the source",
                context={"operation": "migration.plan", "run_id": self.run_id},
            )
        if any(
            assessment.disposition is not MigrationDisposition.MIGRATABLE
            for assessment in inspection.assessments
        ):
            raise ValueError("migration plan contains non-migratable entries")
        plan = MigrationPlan.create(
            run_id=self.run_id,
            source_identity=inspection.source_identity,
            destination_identity=inspection.destination_identity,
            entries=inspection.assessments,
            release_window=ReleaseWindow.first_release_baseline(),
            compatibility=_all_supported_compatibility(),
        )
        self._write_evidence(
            self._new_evidence(
                state=MaintenanceEvidenceState.PLANNED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=("inspect", "plan"),
                completed_output_digests={"plan": plan.digest},
            )
        )
        return plan

    def _validate_plan(self, plan: MigrationPlan) -> None:
        if not isinstance(plan, MigrationPlan) or plan.run_id != self.run_id:
            raise CacheBlobMigrationEvidenceMismatchError(
                "migration plan does not belong to this explicit run",
                context={"operation": "migration.plan", "run_id": self.run_id},
            )
        if not hmac.compare_digest(plan.digest, plan._expected_digest()):
            raise CacheBlobMigrationEvidenceMismatchError(
                "migration plan digest is invalid",
                context={"operation": "migration.plan", "run_id": self.run_id},
            )
        self._revalidate_identities(plan)

    def stage(self, plan: MigrationPlan) -> MigrationStepResult:
        """Copy authenticated immutable payloads into an invisible candidate."""
        self._validate_plan(plan)
        current = self.read_evidence()
        if current.state is MaintenanceEvidenceState.PLANNED:
            self._expect_evidence(MaintenanceEvidenceState.PLANNED, plan_digest=plan.digest)
            self._write_evidence(
                self._new_evidence(
                    state=MaintenanceEvidenceState.STAGING,
                    plan_digest=plan.digest,
                    source_identity=plan.source_identity,
                    destination_identity=plan.destination_identity,
                    completed_steps=("inspect", "plan"),
                    completed_output_digests={"plan": plan.digest},
                )
            )
        elif current.state is MaintenanceEvidenceState.STAGING:
            self._expect_evidence(MaintenanceEvidenceState.STAGING, plan_digest=plan.digest)
        else:
            raise CacheBlobMigrationEvidenceMismatchError(
                "maintenance evidence is not ready to stage this plan",
                context={"operation": "migration.stage", "run_id": self.run_id},
            )
        expected = {assessment.entry.key: assessment for assessment in plan.assessments}
        candidates: list[AuthorityInventoryEntry] = []
        seen_keys: set[str] = set()
        cursor = None
        while True:
            page = self._inventory_page(self._source_authority, cursor)
            if page.identity != plan.source_identity:
                raise ValueError("source inventory changed; reinspection is required")
            for snapshot in page.entries:
                assessment = expected.get(snapshot.key)
                if assessment is None or assessment.entry.manifest_digest != hashlib.sha256(
                    snapshot.manifest
                ).hexdigest():
                    raise ValueError("source entry changed; reinspection is required")
                seen_keys.add(snapshot.key)
                manifest = self._authenticated_manifest(snapshot.manifest, role="source")
                source_io = self.source._materialize_authority_store()
                with source_io.open_snapshot(
                    manifest.locator, dict(manifest.handler_metadata)
                ) as source_snapshot:
                    payload = source_snapshot.path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != manifest.digest or len(payload) != manifest.byte_size:
                    raise ValueError("source payload fails authenticated integrity verification")
                suffix = "".join(Path(manifest.locator).suffixes)
                candidate_locator = (
                    f"generations/migration/{self.run_id}/{snapshot.generation}{suffix}"
                )
                self.destination.payload_backend.write_blob(candidate_locator, payload)
                candidate_manifest = sign_current_manifest(
                    replace(manifest, locator=candidate_locator), self._evidence_key
                )
                candidate_raw = candidate_manifest.canonical_bytes()
                self._authenticated_manifest(candidate_raw, role="candidate")
                candidates.append(
                    AuthorityInventoryEntry(
                        key=snapshot.key,
                        generation=snapshot.generation,
                        locator=candidate_locator,
                        manifest=candidate_raw,
                        payload_digest=manifest.digest,
                        byte_size=manifest.byte_size,
                    )
                )
            if page.exhausted:
                break
            cursor = page.next_cursor
        if seen_keys != set(expected):
            raise ValueError("source inventory changed; reinspection is required")
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
            self._new_evidence(
                state=MaintenanceEvidenceState.STAGED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=("inspect", "plan", "stage"),
                candidate_receipt=receipt,
                completed_output_digests={
                    "plan": plan.digest,
                    "stage": receipt.candidate_digest,
                },
            )
        )
        return MigrationStepResult(MaintenanceEvidenceState.STAGED, True, self.evidence_path)

    def _candidate_from_evidence(
        self, evidence: MaintenanceRunEvidence, plan: MigrationPlan
    ) -> tuple[AuthorityInventoryEntry, ...]:
        """Rebuild expected descriptors from the plan, then verify recorded outputs.

        Candidate presence is never progress evidence.  The plan plus signed
        receipt determines each expected immutable output, which is then read
        and hashed before a resume can skip the completed stage.
        """
        receipt = evidence.candidate_receipt
        if receipt is None:
            raise CacheBlobMigrationEvidenceMismatchError(
                "candidate receipt is required for resume",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )
        candidates = self._expected_candidate_entries(plan)
        if (
            receipt.plan_digest != plan.digest
            or receipt.candidate_digest != candidate_digest(candidates)
            or receipt.entry_count != len(candidates)
            or receipt.byte_count != sum(item.byte_size for item in candidates)
        ):
            raise CacheBlobMigrationEvidenceMismatchError(
                "candidate descriptors do not match authenticated evidence",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )
        self._verify_candidate_outputs(candidates)
        return candidates

    def _expected_candidate_entries(
        self, plan: MigrationPlan
    ) -> tuple[AuthorityInventoryEntry, ...]:
        """Derive the only valid candidate descriptors from current planned input."""
        expected = {assessment.entry.key: assessment for assessment in plan.assessments}
        candidates: list[AuthorityInventoryEntry] = []
        seen_keys: set[str] = set()
        cursor = None
        while True:
            page = self._inventory_page(self._source_authority, cursor)
            if page.identity != plan.source_identity:
                raise CacheBlobMigrationPlanStaleError(
                    "source inventory changed; reinspection is required",
                    context={"operation": "migration.resume", "run_id": self.run_id},
                )
            for snapshot in page.entries:
                assessment = expected.get(snapshot.key)
                if assessment is None or assessment.entry.manifest_digest != hashlib.sha256(
                    snapshot.manifest
                ).hexdigest():
                    raise CacheBlobMigrationPlanStaleError(
                        "source entry changed; reinspection is required",
                        context={"operation": "migration.resume", "run_id": self.run_id},
                    )
                seen_keys.add(snapshot.key)
                manifest = self._authenticated_manifest(snapshot.manifest, role="source")
                suffix = "".join(Path(manifest.locator).suffixes)
                candidate_locator = (
                    f"generations/migration/{self.run_id}/{snapshot.generation}{suffix}"
                )
                candidate_manifest = sign_current_manifest(
                    replace(manifest, locator=candidate_locator), self._evidence_key
                )
                candidate_raw = candidate_manifest.canonical_bytes()
                self._authenticated_manifest(candidate_raw, role="candidate")
                candidates.append(
                    AuthorityInventoryEntry(
                        key=snapshot.key,
                        generation=snapshot.generation,
                        locator=candidate_locator,
                        manifest=candidate_raw,
                        payload_digest=manifest.digest,
                        byte_size=manifest.byte_size,
                    )
                )
            if page.exhausted:
                break
            cursor = page.next_cursor
        if seen_keys != set(expected):
            raise CacheBlobMigrationPlanStaleError(
                "source inventory changed; reinspection is required",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )
        self._revalidate_identities(plan)
        return tuple(candidates)

    def _verify_candidate_outputs(
        self, candidates: tuple[AuthorityInventoryEntry, ...]
    ) -> None:
        """Validate every deterministic completed candidate output before reuse."""
        for candidate in candidates:
            manifest = self._authenticated_manifest(candidate.manifest, role="candidate")
            destination_io = self.destination._materialize_authority_store()
            try:
                with destination_io.open_snapshot(candidate.locator, {}) as snapshot:
                    payload = snapshot.path.read_bytes()
            except OSError as exc:
                raise CacheBlobMigrationEvidenceMismatchError(
                    "candidate output is missing during explicit resume",
                    context={"operation": "migration.resume", "run_id": self.run_id},
                ) from exc
            if (
                manifest.key != candidate.key
                or manifest.locator != candidate.locator
                or manifest.digest != candidate.payload_digest
                or manifest.byte_size != candidate.byte_size
                or hashlib.sha256(payload).hexdigest() != candidate.payload_digest
                or len(payload) != candidate.byte_size
            ):
                raise CacheBlobMigrationEvidenceMismatchError(
                    "candidate output does not match authenticated evidence",
                    context={"operation": "migration.resume", "run_id": self.run_id},
                )

    def verify(self, plan: MigrationPlan) -> MigrationStepResult:
        """Verify every candidate payload and descriptor before activation is allowed."""
        self._validate_plan(plan)
        evidence = self.read_evidence()
        if evidence.state is MaintenanceEvidenceState.STAGED:
            evidence = self._expect_evidence(
                MaintenanceEvidenceState.STAGED, plan_digest=plan.digest
            )
            self._candidate_from_evidence(evidence, plan)
            self._write_evidence(
                self._new_evidence(
                    state=MaintenanceEvidenceState.VERIFYING,
                    plan_digest=plan.digest,
                    source_identity=plan.source_identity,
                    destination_identity=plan.destination_identity,
                    completed_steps=("inspect", "plan", "stage"),
                    candidate_receipt=evidence.candidate_receipt,
                    completed_output_digests=dict(evidence.completed_output_digests),
                )
            )
        elif evidence.state is MaintenanceEvidenceState.VERIFYING:
            evidence = self._expect_evidence(
                MaintenanceEvidenceState.VERIFYING, plan_digest=plan.digest
            )
            self._candidate_from_evidence(evidence, plan)
        else:
            raise CacheBlobMigrationEvidenceMismatchError(
                "maintenance evidence is not ready to verify this plan",
                context={"operation": "migration.verify", "run_id": self.run_id},
            )
        receipt = evidence.candidate_receipt
        assert receipt is not None
        self._write_evidence(
            self._new_evidence(
                state=MaintenanceEvidenceState.VERIFIED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=("inspect", "plan", "stage", "verify"),
                candidate_receipt=receipt,
                completed_output_digests={
                    **evidence.completed_output_digests,
                    "verify": receipt.candidate_digest,
                },
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
        activation_digest = hashlib.sha256(
            f"{activation.activation_revision}:{receipt.candidate_digest}".encode("ascii")
        ).hexdigest()
        self._write_evidence(
            self._new_evidence(
                state=MaintenanceEvidenceState.ACTIVATED,
                plan_digest=plan.digest,
                source_identity=plan.source_identity,
                destination_identity=plan.destination_identity,
                completed_steps=("inspect", "plan", "stage", "verify", "activate"),
                candidate_receipt=receipt,
                activation_receipt=activation,
                authority_receipts=(activation_digest,),
                completed_output_digests={
                    **evidence.completed_output_digests,
                    "activate": activation_digest,
                },
            )
        )
        return MigrationStepResult(MaintenanceEvidenceState.ACTIVATED, True, self.evidence_path)

    def resume(
        self,
        plan: MigrationPlan,
        *,
        run_id: str | None,
        evidence_path: str | Path | None,
    ) -> MigrationStepResult:
        """Continue one explicit run only after authenticating and revalidating it.

        The caller must give the exact run ID and generated evidence path. This
        method never enumerates work roots, searches for a latest run, or scans
        payload/catalog state to infer an unrecorded completed action.
        """
        if run_id is None or evidence_path is None:
            raise CacheBlobMigrationOfflineDecisionRequiredError(
                "resume requires an explicit run_id and exact evidence path",
                context={"operation": "migration.resume"},
            )
        if run_id != self.run_id:
            raise CacheBlobMigrationEvidenceMismatchError(
                "resume run_id does not match this offline service",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )
        if Path(evidence_path) != self.evidence_path:
            raise CacheBlobMigrationEvidenceMismatchError(
                "resume evidence path does not match this explicit run",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )
        evidence = self.read_evidence()
        if evidence.run_id != run_id or evidence.plan_digest != plan.digest:
            raise CacheBlobMigrationEvidenceMismatchError(
                "resume evidence does not bind the supplied plan",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )
        if evidence.state is MaintenanceEvidenceState.ACTIVATED:
            self._revalidate_activated_receipt(evidence, plan)
            return MigrationStepResult(MaintenanceEvidenceState.ACTIVATED, True, self.evidence_path)
        self._validate_plan(plan)
        if evidence.state is MaintenanceEvidenceState.PLANNED:
            return self.stage(plan)
        if evidence.state is MaintenanceEvidenceState.STAGING:
            return self.stage(plan)
        if evidence.state is MaintenanceEvidenceState.STAGED:
            self._candidate_from_evidence(evidence, plan)
            return self.verify(plan)
        if evidence.state is MaintenanceEvidenceState.VERIFYING:
            self._candidate_from_evidence(evidence, plan)
            return self.verify(plan)
        if evidence.state is MaintenanceEvidenceState.VERIFIED:
            self._expect_evidence(MaintenanceEvidenceState.VERIFIED, plan_digest=plan.digest)
            self._candidate_from_evidence(evidence, plan)
            return MigrationStepResult(MaintenanceEvidenceState.VERIFIED, True, self.evidence_path)
        raise CacheBlobMigrationOfflineDecisionRequiredError(
            "resume state requires reinspection or a separately confirmed offline action",
            context={"operation": "migration.resume", "run_id": self.run_id},
        )

    def _revalidate_activated_receipt(
        self, evidence: MaintenanceRunEvidence, plan: MigrationPlan
    ) -> None:
        """Validate an already-authorized activation without treating evidence as authority."""
        receipt = evidence.activation_receipt
        if receipt is None or evidence.candidate_receipt is None:
            raise CacheBlobMigrationEvidenceMismatchError(
                "activated evidence lacks an authority receipt",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )
        source_identity = self._source_authority.identity_snapshot()
        destination_identity = self._destination_authority.identity_snapshot()
        if (
            source_identity != plan.source_identity
            or receipt.candidate_receipt != evidence.candidate_receipt
            or destination_identity.store_id != plan.destination_identity.store_id
            or destination_identity.revision != receipt.activation_revision
        ):
            raise CacheBlobMigrationPlanStaleError(
                "activated maintenance evidence no longer matches live authority state",
                context={"operation": "migration.resume", "run_id": self.run_id},
            )


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
    "MigrationPlanKind",
    "MigrationPlanState",
    "MigrationReason",
    "MigrationStepResult",
    "MigrationTotals",
    "OfflineMigrationService",
    "ReleaseWindow",
    "VersionEdge",
    "inspect_migration_store",
    "inspect_store_path",
    "render_migration_report",
]
